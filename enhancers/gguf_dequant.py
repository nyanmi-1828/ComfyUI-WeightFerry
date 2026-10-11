# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Re-implements the legacy-quant dequantisation of city96/ComfyUI-GGUF dequant.py ((c) City96, Apache-2.0) as one
# Triton kernel per call: the same operations, dtypes and roundings; and (2026-10-12) the LoRA application of ComfyUI
# comfy/lora.py calculate_weight / comfy/weight_adapter/lora.py LoRAAdapter.calculate_weight (GPL-3.0) after its
# torch.mm. See THIRD_PARTY_NOTICES.md.
"""
GGUF dequantisation in one Triton kernel (WEIGHTFERRY_GGUF_TRITON, on by default since 2026-10-12; 0 turns it off).

ComfyUI-GGUF dequantises a quantised weight on every use with a chain of PyTorch operations (byte splits, shifts,
masks, a multiply, an add, a cast): Anima resident at 896x1472, CFG 5, its dequantisation is about 30% of a step with
Q5_1 and 16% with Q4_1 (5% with Q8_0; testrun.py g1_profile, 2026-10-11). Here the whole chain is one kernel that
reads the blocks once and writes the weight once, with the same arithmetic as ComfyUI-GGUF's code:

* the scale d (and the minimum m) are the block's float16 values converted to the dequant dtype (float16 unless
  ComfyUI-GGUF's dequant_dtype says otherwise),
* q is the quant as ComfyUI-GGUF builds it (Q8_0: the int8; Q4_0 / Q5_0: the nibble (and high bit) minus 8 / 16;
  Q4_1 / Q5_1: the nibble (and high bit), 0 .. 15 / 31; value j of a block is the low nibble of byte j for j < 16, the
  high nibble of byte j - 16 otherwise, its high bit bit j of the 32-bit qh),
* d * q is computed in float32 and rounded to the dequant dtype (PyTorch's opmath for a float16 / bfloat16 multiply),
  then + m likewise (Q4_1 / Q5_1), then the cast to the layer's dtype -- each rounding to nearest even, done with
  explicit conversions so the compiler cannot drop one.

Only the five legacy formats; K-quants and anything else keep ComfyUI-GGUF's code, as do CPU tensors.

LoRA (2026-10-12): ComfyUI-GGUF's GGMLLayer.get_weight dequantises, then applies the patches with
comfy.lora.calculate_weight on every use. For plain LoRA patches (no DoRA / mid weight / reshape / offset /
function / strength_model) the difference stays comfy's own torch.mm(up, down) in float32 (cuBLAS chooses its order:
a sequential fma matched it on random factors but not on a real rank-32 LoRA, so it is not reproduced), and the
dequantisation, the scale by float32(strength * alpha), the cast to the weight dtype and the add run in one kernel
(each further LoRA in one more). The same bits on every layer of Anima Q8_0 with the workflow LoRAs; Anima resident
x1.011 (Q5_1) / x1.012 (Q8_0) / x1.005 (Q4_1): most of the LoRA cost is the per-layer CPU work, not the GPU passes.
"""

import sys
import weakref

import torch

try:  # module level: Triton resolves the tl.constexpr annotations of @triton.jit functions in the module globals
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = tl = None

# GGML quant type id (gguf.GGMLQuantizationType) -> name, bytes per 32-value block, has a minimum m, qh offset (-1:
# none), quants offset, value subtracted from the quant, int8 quants
_FORMATS = {2: ("Q4_0", 18, False, -1, 2, 8, False), 3: ("Q4_1", 20, True, -1, 4, 0, False),
            6: ("Q5_0", 22, False, 2, 6, 16, False), 7: ("Q5_1", 24, True, 4, 8, 0, False),
            8: ("Q8_0", 34, False, -1, 2, 0, True)}
_DT = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_kernel = {}
_state = {"orig": None, "ops": None, "orig_gw": None, "stats": {"used": 0, "fallback": 0}}
_LORA = True  # LoRA patches inside the kernel too (GGMLLayer.get_weight); module constant for benchmarks
_CACHE_MB = None  # the LoRA weight cache budget; None = WEIGHTFERRY_GGUF_LORA_CACHE_MB (module constant for benchmarks)
_cache = {"layers": weakref.WeakKeyDictionary(), "bytes": 0, "hooked": False}


def _get_kernel():
    if _kernel:
        return _kernel["k"]

    @triton.jit
    def _round(x, D: tl.constexpr):
        # float32 -> the nearest value of dtype D (0 float16, 1 bfloat16, 2 float32), still as float32; done in inline
        # PTX so that the compiler cannot fold a narrowing and widening pair away
        if D == 0:
            return tl.inline_asm_elementwise("{ .reg .b16 t; cvt.rn.f16.f32 t, $1; cvt.f32.f16 $0, t; }", "=r,r",
                                             [x], dtype=tl.float32, is_pure=True, pack=1)
        elif D == 1:
            u = x.to(tl.uint32, bitcast=True)
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            return u.to(tl.float32, bitcast=True)
        else:
            return x

    @triton.jit
    def _f16(src, off):
        # the little-endian float16 at byte offset off, as float32 (exact)
        lo = tl.load(src + off).to(tl.uint16)
        hi = tl.load(src + off + 1).to(tl.uint16)
        return ((hi << 8) | lo).to(tl.float16, bitcast=True).to(tl.float32)

    @triton.jit
    def _f16m(src, off, m):
        lo = tl.load(src + off, mask=m, other=0).to(tl.uint16)
        hi = tl.load(src + off + 1, mask=m, other=0).to(tl.uint16)
        return ((hi << 8) | lo).to(tl.float16, bitcast=True).to(tl.float32)

    @triton.jit
    def _deq_add(src, diff, out, n, sa, DEQ: tl.constexpr, TS: tl.constexpr, HAS_M: tl.constexpr, QH: tl.constexpr,
                 QS: tl.constexpr, SUB: tl.constexpr, IS_Q8: tl.constexpr, MID: tl.constexpr, OUT: tl.constexpr,
                 BLOCK: tl.constexpr):
        # w = w + (sa * diff).to(w.dtype) on a flat weight of n values (comfy's LoRA calculate_weight after its
        # torch.mm, the float32 diff given): w dequantised here first (DEQ, as _deq does) or read from out (a later
        # LoRA); every result rounded to the weight dtype
        lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = lin < n
        if DEQ:
            base = (lin // 32) * TS
            v = lin % 32
            d = _round(_f16m(src, base, m), MID)
            if IS_Q8:
                q = tl.load(src + base + QS + v, mask=m, other=0).to(tl.int8, bitcast=True).to(tl.float32)
            else:
                byte = tl.load(src + base + QS + (v % 16), mask=m, other=0).to(tl.int32)
                nib = (byte >> ((v // 16) * 4).to(tl.int32)) & 0xF
                if QH >= 0:
                    hb = tl.load(src + base + QH + (v // 8), mask=m, other=0).to(tl.int32)
                    nib = nib | (((hb >> (v % 8).to(tl.int32)) & 1) << 4)
                q = (nib - SUB).to(tl.float32)
            x = _round(d * q, MID)
            if HAS_M:
                x = _round(x + _round(_f16m(src, base + 2, m), MID), MID)
            w = _round(x, OUT)
        else:
            w = tl.load(out + lin, mask=m, other=0.0).to(tl.float32)
        dl = tl.load(diff + lin, mask=m, other=0.0)
        w = _round(w + _round(dl * sa, OUT), OUT)
        if OUT == 0:
            tl.store(out + lin, w.to(tl.float16), mask=m)
        elif OUT == 1:
            tl.store(out + lin, w.to(tl.bfloat16), mask=m)
        else:
            tl.store(out + lin, w, mask=m)

    @triton.jit
    def _deq(src, out, n_blocks, TS: tl.constexpr, HAS_M: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr,
             SUB: tl.constexpr, IS_Q8: tl.constexpr, MID: tl.constexpr, OUT: tl.constexpr, BB: tl.constexpr):
        # BB blocks of 32 values per program; byte offsets in a block: d at 0, m at 2 (HAS_M), qh at QH (Q5_x),
        # quants at QS; SUB is subtracted from the quant (Q4_0: 8, Q5_0: 16)
        blk = tl.program_id(0) * BB + tl.arange(0, BB)
        bm = blk < n_blocks
        base = blk.to(tl.int64) * TS
        v = tl.arange(0, 32)
        m2 = bm[:, None]
        d = _round(_f16(src, base), MID)
        if IS_Q8:
            q = tl.load(src + base[:, None] + QS + v[None, :], mask=m2, other=0).to(tl.int8, bitcast=True)
            q = q.to(tl.float32)
        else:
            byte = tl.load(src + base[:, None] + QS + (v[None, :] % 16), mask=m2, other=0).to(tl.int32)
            nib = (byte >> ((v[None, :] // 16) * 4)) & 0xF
            if QH >= 0:
                hb = tl.load(src + base[:, None] + QH + (v[None, :] // 8), mask=m2, other=0).to(tl.int32)
                nib = nib | (((hb >> (v[None, :] % 8)) & 1) << 4)
            q = (nib - SUB).to(tl.float32)
        x = _round(d[:, None] * q, MID)
        if HAS_M:
            mn = _round(_f16(src, base + 2), MID)
            x = _round(x + mn[:, None], MID)
        x = _round(x, OUT)
        o = out + blk[:, None].to(tl.int64) * 32 + v[None, :]
        if OUT == 0:
            tl.store(o, x.to(tl.float16), mask=m2)
        elif OUT == 1:
            tl.store(o, x.to(tl.bfloat16), mask=m2)
        else:
            tl.store(o, x, mask=m2)

    _kernel["k"] = _deq
    _kernel["add"] = _deq_add
    return _deq


def dequantize(data, qtype, oshape, mid_dtype, out_dtype):
    """ComfyUI-GGUF's dequantize(data, qtype, oshape, dtype=mid_dtype).to(out_dtype) for a legacy format, or None
    when this is not that case (other formats, CPU data, dtypes)."""
    fmt = _FORMATS.get(int(qtype)) if qtype is not None else None
    if fmt is None or triton is None or not data.is_cuda:
        return None
    mid = mid_dtype if mid_dtype is not None else torch.float16
    out = out_dtype if out_dtype is not None else mid
    if mid not in _DT or out not in _DT:
        return None
    _name, ts, has_m, qh, qs, sub, is_q8 = fmt
    raw = data.reshape(-1).view(torch.uint8) if data.is_contiguous() else None
    if raw is None or raw.numel() % ts:
        return None
    n_blocks = raw.numel() // ts
    if n_blocks * 32 != int(torch.Size(oshape).numel()):
        return None
    res = torch.empty(tuple(oshape), dtype=out, device=data.device)
    bb = 32
    _get_kernel()[(triton.cdiv(n_blocks, bb),)](raw, res, n_blocks, TS=ts, HAS_M=has_m, QH=qh, QS=qs, SUB=sub,
                                                IS_Q8=is_q8, MID=_DT[mid], OUT=_DT[out], BB=bb, num_warps=4)
    return res


def lora_factors(patch_list, n_out, n_in):
    """[(up, down, strength * alpha)] when every patch is a plain LoRA that comfy.lora.calculate_weight applies as
    weight += ((strength * alpha) * (up @ down)).type(weight.dtype) on a 2-D weight (no DoRA, mid weight, reshape,
    offset, function or strength_model), else None."""
    out = []
    for p in patch_list:
        if not isinstance(p, tuple) or len(p) != 5:
            return None
        strength, v, strength_model, offset, function = p
        if strength_model != 1.0 or offset is not None or function is not None or type(v).__name__ != "LoRAAdapter":
            return None
        w = getattr(v, "weights", None)
        if not isinstance(w, (tuple, list)) or len(w) < 6 or w[3] is not None or w[4] is not None or w[5] is not None:
            return None
        up, down, alpha_raw = w[0], w[1], w[2]
        if (not torch.is_tensor(up) or not torch.is_tensor(down) or up.dim() != 2 or down.dim() != 2
                or up.shape[0] != n_out or down.shape[1] != n_in or up.shape[1] != down.shape[0]
                or not isinstance(strength, (int, float)) or isinstance(strength, bool)):
            return None
        if alpha_raw is None:
            alpha = 1.0
        elif isinstance(alpha_raw, (int, float)) and not isinstance(alpha_raw, bool):
            alpha = alpha_raw / down.shape[0]
        else:
            return None
        out.append((up, down, strength * alpha))
    return out


def dequantize_lora(data, qtype, oshape, mid_dtype, out_dtype, loras):
    """ComfyUI-GGUF's get_weight with LoRA patches (dequantize, then comfy.lora.calculate_weight with the float32
    intermediate): each LoRA difference by the same torch.mm comfy calls (its bits are cuBLAS's to choose; not
    reproduced), the dequantisation and the first LoRA's scale / cast / add in one kernel, each further LoRA's in
    one more kernel (one float32 difference alive at a time, as in comfy's loop). The same bits; None when not that
    case."""
    fmt = _FORMATS.get(int(qtype)) if qtype is not None else None
    if fmt is None or triton is None or not data.is_cuda or len(oshape) != 2 or not loras:
        return None
    mid = mid_dtype if mid_dtype is not None else torch.float16
    if mid not in _DT or out_dtype not in _DT:
        return None
    n_out, n_in = int(oshape[0]), int(oshape[1])
    if n_in % 32:
        return None
    _name, ts, has_m, qh, qs, sub, is_q8 = fmt
    raw = data.reshape(-1).view(torch.uint8) if data.is_contiguous() else None
    if raw is None or raw.numel() != n_out * n_in // 32 * ts:
        return None
    dev = data.device
    res = torch.empty((n_out, n_in), dtype=out_dtype, device=dev)
    n = n_out * n_in
    BLOCK = 1024
    _get_kernel()
    for i, (up, down, sa) in enumerate(loras):
        mat1 = up.to(dev, torch.float32)  # comfy.model_management.cast_to_device(v[0], device, float32)
        mat2 = down.to(dev, torch.float32)
        diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))
        del mat1, mat2
        _kernel["add"][(triton.cdiv(n, BLOCK),)](
            raw, diff, res, n, float(sa), DEQ=(i == 0), TS=ts, HAS_M=has_m, QH=qh, QS=qs, SUB=sub, IS_Q8=is_q8,
            MID=_DT[mid], OUT=_DT[out_dtype], BLOCK=BLOCK, num_warps=4)
        del diff
    return res


def _patched_get_weight(orig):
    def get_weight(self, tensor, dtype):
        if tensor is None:
            return None
        pats = getattr(tensor, "patches", None)
        qtype = getattr(tensor, "tensor_type", None)
        if (pats and getattr(self, "patch_dtype", None) is None and qtype is not None and int(qtype) in _FORMATS
                and isinstance(dtype, torch.dtype) and getattr(tensor, "is_cuda", False)):
            mid = dtype if self.dequant_dtype == "target" else self.dequant_dtype
            if mid is None or isinstance(mid, torch.dtype):
                oshape = getattr(tensor, "tensor_shape", tensor.shape)
                patch_list = []
                for patches, _key in pats:
                    patch_list += list(patches)
                loras = lora_factors(patch_list, *oshape) if len(oshape) == 2 else None
                if loras:
                    try:
                        out = dequantize_lora(tensor.data, qtype, oshape, mid, dtype, loras)
                    except Exception as exc:  # noqa: BLE001 -- ComfyUI-GGUF's own code instead
                        from ..utils.logging import log_warn
                        log_warn(f"GGUF Triton dequant + LoRA: {exc!r}; ComfyUI-GGUF's code for this tensor.")
                        out = None
                    if out is not None:
                        _state["stats"]["lora_used"] = _state["stats"].get("lora_used", 0) + 1
                        return out
                _state["stats"]["lora_fallback"] = _state["stats"].get("lora_fallback", 0) + 1
        return orig(self, tensor, dtype)
    get_weight._weightferry = True
    return get_weight


def _cache_budget():
    if _CACHE_MB is not None:
        mb = _CACHE_MB
    else:
        from ..core import env as _env
        try:
            mb = float(_env.get("WEIGHTFERRY_GGUF_LORA_CACHE_MB") or 0)
        except ValueError:
            mb = 0
    return int(max(0.0, mb) * 2 ** 20)


def clear_cache():
    """Drop every cached LoRA weight (ComfyUI frees memory / unloads models: see _hook_comfy_memory)."""
    _cache["layers"] = weakref.WeakKeyDictionary()
    _cache["bytes"] = 0


def _hook_comfy_memory():
    """Clear the cache whenever ComfyUI frees memory or unloads models, so it never holds VRAM ComfyUI wants."""
    if _cache["hooked"]:
        return
    import comfy.model_management as mm
    for name in ("free_memory", "unload_all_models", "soft_empty_cache"):
        fn = getattr(mm, name, None)
        if fn is None or getattr(fn, "_weightferry_cache", False):
            continue

        def wrapped(*a, _fn=fn, **k):
            if _cache["bytes"]:
                clear_cache()
            return _fn(*a, **k)
        wrapped._weightferry_cache = True
        setattr(mm, name, wrapped)
    _cache["hooked"] = True


def _version(t):
    try:
        return t._version
    except RuntimeError:  # inference tensors (ComfyUI runs under torch.inference_mode) keep no version counter
        return None


def _patch_key(pats):
    """What the patched weight depends on besides the quantised weight: every patch's strength, adapter object,
    strength_model, offset, function and the identity / version of the adapter's tensors."""
    key = []
    for patches, k in pats:
        for p in patches:
            if not isinstance(p, tuple):
                return None
            v = p[1]
            ts = getattr(v, "weights", v if isinstance(v, (tuple, list)) else ())
            tens = tuple((id(t), t.data_ptr(), _version(t)) for t in ts if torch.is_tensor(t))
            key.append((k, p[0], id(v), p[2], p[3], id(p[4]) if p[4] is not None else None, tens))
    return tuple(key)


def _patched_cast_bias_weight(orig):
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        budget = _cache_budget()
        w = getattr(s, "weight", None)
        pats = getattr(w, "patches", None) if w is not None else None
        if not budget or not pats:
            return orig(s, input, dtype, device, bias_dtype)
        import comfy.model_management
        import comfy.ops
        if input is not None:  # ComfyUI-GGUF's own defaults
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        pk = _patch_key(pats)
        if pk is None:
            return orig(s, input, dtype, device, bias_dtype)
        key = (id(w), w.data_ptr(), str(w.device), dtype, str(device), s.dequant_dtype, s.patch_dtype, pk)
        hit = _cache["layers"].get(s)
        if hit is not None and hit[0] == key:
            bias = None
            if s.bias is not None:  # as ComfyUI-GGUF computes it (no cache: small)
                nb = comfy.model_management.device_supports_non_blocking(device)
                bias = s.get_weight(s.bias.to(device), dtype)
                bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=nb, copy=False)
            _state["stats"]["cache_hit"] = _state["stats"].get("cache_hit", 0) + 1
            return hit[1], bias
        weight, bias = orig(s, input, dtype, device, bias_dtype)
        if hit is not None:
            _cache["bytes"] -= hit[1].numel() * hit[1].element_size()
            del _cache["layers"][s]
        size = weight.numel() * weight.element_size()
        if weight.is_cuda and _cache["bytes"] + size <= budget:
            _hook_comfy_memory()
            _cache["layers"][s] = (key, weight)
            _cache["bytes"] += size
            _state["stats"]["cache_store"] = _state["stats"].get("cache_store", 0) + 1
        return weight, bias
    cast_bias_weight._weightferry = True
    return cast_bias_weight


def _patched(orig):
    def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
        qtype = getattr(tensor, "tensor_type", None)
        if qtype is not None and int(qtype) in _FORMATS and isinstance(tensor, torch.Tensor):
            mid = dtype if dequant_dtype == "target" else dequant_dtype
            if mid is None or isinstance(mid, torch.dtype):
                oshape = getattr(tensor, "tensor_shape", tensor.shape)
                try:
                    out = dequantize(tensor.data, qtype, oshape, mid, dtype)
                except Exception as exc:  # noqa: BLE001 -- ComfyUI-GGUF's own code instead
                    from ..utils.logging import log_warn
                    _state["stats"]["fallback"] += 1
                    log_warn(f"GGUF Triton dequant: {exc!r}; ComfyUI-GGUF's code for this tensor.")
                    out = None
                if out is not None:
                    _state["stats"]["used"] += 1
                    return out
        _state["stats"]["fallback"] += 1
        return orig(tensor, dtype, dequant_dtype)
    dequantize_tensor._weightferry = True
    return dequantize_tensor


def install(force=False):
    """Replace ComfyUI-GGUF's ops.dequantize_tensor (the name GGMLLayer.get_weight calls) with the one-kernel version,
    unless WEIGHTFERRY_GGUF_TRITON=0 (force: always), when ComfyUI-GGUF is loaded. Idempotent; True when installed."""
    from ..core import env as _env
    from ..utils.logging import log_info, log_warn
    if not force and _env.get("WEIGHTFERRY_GGUF_TRITON") != "1":
        return False
    if triton is None:
        log_warn("GGUF Triton dequant: Triton not available; ComfyUI-GGUF's code stays.")
        return False
    ops = next((m for n, m in sys.modules.items() if n.endswith("ComfyUI-GGUF.ops") and m is not None), None)
    if ops is None or not hasattr(ops, "dequantize_tensor"):
        return False
    if getattr(ops.dequantize_tensor, "_weightferry", False):
        return True
    _state["orig"], _state["ops"] = ops.dequantize_tensor, ops
    ops.dequantize_tensor = _patched(ops.dequantize_tensor)
    layer = getattr(ops, "GGMLLayer", None)
    if _LORA and layer is not None and not getattr(layer.get_weight, "_weightferry", False):
        _state["orig_gw"] = layer.get_weight
        layer.get_weight = _patched_get_weight(layer.get_weight)
    if layer is not None and not getattr(layer.cast_bias_weight, "_weightferry", False):
        # the LoRA weight cache (WEIGHTFERRY_GGUF_LORA_CACHE_MB, off at 0): costs one budget check per call when off
        _state["orig_cbw"] = layer.cast_bias_weight
        layer.cast_bias_weight = _patched_cast_bias_weight(layer.cast_bias_weight)
    log_info("GGUF Triton dequant: ComfyUI-GGUF's Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 dequantisation in one kernel "
             "(WEIGHTFERRY_GGUF_TRITON).")
    return True


def uninstall():
    clear_cache()
    if _state["ops"] is not None and _state.get("orig_cbw") is not None:
        _state["ops"].GGMLLayer.cast_bias_weight = _state["orig_cbw"]
        _state["orig_cbw"] = None
    if _state["ops"] is not None and _state.get("orig_gw") is not None:
        _state["ops"].GGMLLayer.get_weight = _state["orig_gw"]
        _state["orig_gw"] = None
    if _state["ops"] is not None and _state["orig"] is not None:
        _state["ops"].dequantize_tensor = _state["orig"]
    _state["orig"] = _state["ops"] = None


def stats(reset=False):
    out = dict(_state["stats"])
    if reset:
        _state["stats"] = {"used": 0, "fallback": 0}
    return out
