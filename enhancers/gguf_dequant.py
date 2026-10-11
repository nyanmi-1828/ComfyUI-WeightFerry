# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nyanmi-1828
# Re-implements the legacy-quant dequantisation of city96/ComfyUI-GGUF dequant.py ((c) City96, Apache-2.0) as one
# Triton kernel per call: the same operations, dtypes and roundings. See THIRD_PARTY_NOTICES.md.
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
"""

import sys

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
_state = {"orig": None, "ops": None, "stats": {"used": 0, "fallback": 0}}


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
    log_info("GGUF Triton dequant: ComfyUI-GGUF's Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 dequantisation in one kernel "
             "(WEIGHTFERRY_GGUF_TRITON).")
    return True


def uninstall():
    if _state["ops"] is not None and _state["orig"] is not None:
        _state["ops"].dequantize_tensor = _state["orig"]
    _state["orig"] = _state["ops"] = None


def stats(reset=False):
    out = dict(_state["stats"])
    if reset:
        _state["stats"] = {"used": 0, "fallback": 0}
    return out
