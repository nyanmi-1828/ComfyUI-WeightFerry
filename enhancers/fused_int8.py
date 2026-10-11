# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Parts re-implement ComfyUI comfy/ldm/cosmos/predict2.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""The int8 ConvRot linear path of the fused block (enhancers/fused_block.py): pending inputs (an operation left for
the next layer's quantiser: modq / gelu / ln), the input quantised once and shared by several layers (qkv), the gated
residual inside the output projection's GEMM (res), and the per-call _forward overrides of the linear layers."""

import atexit

import torch

from .fused_kernels import PRE_GELU, PRE_LNMOD, PRE_MOD, _get_kernels
from .fused_state import _count, _tls


# ------------------------------------------------------------------------------------------------------ pending inputs
def _key2d(t):
    return (t.data_ptr(), t.numel(), t.shape[-1])


def _set_pre(marker, kind, scale=None, shift=None, norm=None):
    """Register a pending input: the layer that receives ``marker`` quantises kind(marker) itself (modq / gelu; lnmod:
    marker is the block input, norm its LayerNorm)."""
    pre = {"key": _key2d(marker), "kind": kind, "src": marker, "used": False, "mat": None, "norm": norm}
    if kind in ("mod", "lnmod"):
        D = marker.shape[-1]
        groups = scale.numel() // D  # B * T (block) or B (row path)
        pre["s1"] = (1 + scale).contiguous().view(groups, D)
        pre["shift"] = shift.contiguous().view(groups, D)
        pre["scale5"], pre["shift5"], pre["rows_per_group"] = scale, shift, marker.numel() // D // groups
    _tls.pre = pre
    return pre


def _clear_pre():
    pre = getattr(_tls, "pre", None)
    _tls.pre = None
    return pre


def _materialise(pre):
    if pre["mat"] is None:
        from .fused_block import modulate  # fused_block imports this module: resolved at call time
        if pre["kind"] == "gelu":
            pre["mat"] = torch.nn.functional.gelu(pre["src"])
        elif pre["kind"] == "lnmod":
            pre["mat"] = modulate(pre["norm"](pre["src"]), pre["scale5"], pre["shift5"])
        else:
            pre["mat"] = modulate(pre["src"], pre["scale5"], pre["shift5"])
    return pre["mat"]


def _quantise_pre(pre, x_2d, x_q, x_s):
    """Fill x_q / x_s with kitchen's quantisation of the pending input's real value (the Triton reproduction)."""
    m, k = x_2d.shape
    quant_row = _get_kernels()["quant_row"]
    # 2 warps per row: the same bits as 8 (the amax and the rotation do not depend on the thread layout, _ln_stats
    # models its 128 threads in the tensor shapes), 1.5 - 2.2x faster on the RTX 4060 at Anima's widths 2048 / 8192
    # (2026-10-11, scratch/verify/quant_row_bench.py)
    if pre["kind"] == "gelu":
        quant_row[(m,)](x_2d, _gelu_table(x_2d.device), x_2d, x_q, x_s, 1, 0.0, K=k, PRE=PRE_GELU, num_warps=2)
    elif pre["kind"] == "lnmod":
        quant_row[(m,)](x_2d, pre["s1"], pre["shift"], x_q, x_s, pre["rows_per_group"], float(pre["norm"].eps), K=k,
                        PRE=PRE_LNMOD, num_warps=2)
    else:
        quant_row[(m,)](x_2d, pre["s1"], pre["shift"], x_q, x_s, pre["rows_per_group"], 0.0, K=k, PRE=PRE_MOD,
                        num_warps=2)


_gelu_tables = {}


def _release_gpu_caches():
    """At interpreter exit, before modules are torn down: drop the GPU tensors this module keeps (the GELU tables, a
    leftover pending input). Freed later, during module teardown, they reached ComfyUI's allocator (comfy-aimdo) after
    it had shut down and crashed the process on exit."""
    _gelu_tables.clear()
    for name in ("pre", "qkv", "res"):
        if getattr(_tls, name, None) is not None:
            setattr(_tls, name, None)


atexit.register(_release_gpu_caches)


def _gelu_table(device):
    """PyTorch's GELU (erf) of every bfloat16 value, indexed by its 16 bits. The GELU inside the quantiser looks its
    results up here: CUDA's erff differs between toolkits in the last bit (PyTorch's kernels came with CUDA 13.0,
    Triton brings its own libdevice: 0.06% of a real MLP input differed after rounding, all at x in -5.3 .. -3.1 where
    1 + erf cancels), and a bf16 input has only 65536 values."""
    key = str(device)
    tab = _gelu_tables.get(key)
    if tab is None:
        xs = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
        tab = torch.nn.functional.gelu(xs).contiguous()
        _gelu_tables[key] = tab
    return tab


def _int8_proj_ok(lin):
    """The layer hands its input unchanged to _forward (int8 weight-only path, no AWQ smoothing, no own forward)."""
    from comfy.quant_ops import QUANT_ALGOS
    return (hasattr(lin, "_forward") and "forward" not in lin.__dict__
            and getattr(lin, "pre_quant_scale", None) is None
            and QUANT_ALGOS.get(getattr(lin, "quant_format", None), {}).get("quantize_input", True) is False
            and getattr(getattr(lin, "weight", None), "_layout_cls", None) == "TensorWiseINT8Layout")


# ---------------------------------------------------------------------------------------------------------- int8 linear
def _kitchen():
    import comfy_kitchen.backends.cuda as kc
    return kc


def _int8_quantised(x, weight, bias):
    """The quantised input of ComfyUI's int8 ConvRot linear (comfy_kitchen's int8_linear: fused ConvRot quantiser +
    CUTLASS GEMM), reused while the same input tensor comes back (q / k / v); a pending input is quantised by the
    reproduction with its operation inside. None when ComfyUI would take any other path for this call."""
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    cache = getattr(_tls, "qkv", None)
    if (cache is None or bias is not None or not isinstance(weight, QuantizedTensor)
            or isinstance(x, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout"):
        return None
    params = weight._params
    if getattr(params, "transposed", False) or not getattr(params, "convrot", False):
        return None
    gs = getattr(params, "convrot_groupsize", 256)
    kc = _kitchen()
    x = x.contiguous()
    x_2d = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
    m, k = x_2d.shape
    w_q, w_scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    w_q = w_q.contiguous()
    n = w_q.shape[0]
    dev = x.device
    if (x.dtype != torch.bfloat16 or not x.is_cuda or m <= 2 or gs != 256 or k % 256 != 0
            or not 256 <= k <= kc._CONVROT_FUSED_MAX_K or not kc._convrot_fused_shared_memory_fits(x_2d, k, gs)
            or w_q.shape[1] != k):
        return None
    if (kc._prefer_turing_fused_int8(m, n, k) and kc._cuda_device_should_use_turing_kernels(x.get_device())
            and hasattr(kc._C, "cutlass_turing_int8_dequant")):
        return None
    stream_ptr = torch.cuda.current_stream(dev).cuda_stream
    pre = getattr(_tls, "pre", None)
    is_pre = pre is not None and pre["key"] == _key2d(x_2d)
    if is_pre and k & (k - 1):
        # the Triton reproduction handles power-of-2 widths only (Anima: 2048 / 8192; Cosmos-Predict2 14B: 5120 /
        # 20480): quantise the real input with kitchen's quantiser instead, as ComfyUI does (q / k / v still share it)
        _count({"gelu": "gelu", "lnmod": "ln"}.get(pre["kind"], "modq"), False, f"width {k} is not a power of 2")
        x_2d = _materialise(pre).reshape(-1, k).contiguous()
        is_pre = False
    key = (x_2d.data_ptr(), tuple(x_2d.shape), tuple(x_2d.stride()), gs, is_pre)
    if cache.get("key") == key:
        x_q, x_s = cache["q"]
    else:
        x_q = torch.empty((m, k), dtype=torch.int8, device=dev)
        x_s = torch.empty((m, 1), dtype=torch.float32, device=dev)
        if (kc._prefer_cublas_int8_fallback(m, n, k, x_q.get_device()) or kc._DISABLE_CUTLASS_INT8
                or not kc._cuda_device_supports_cutlass_int8_dequant(x_q)):
            return None
        if is_pre:  # the pending operation inside the quantiser: the same bits as kitchen's on the real input
            _quantise_pre(pre, x_2d, x_q, x_s)
            _count({"gelu": "gelu", "lnmod": "ln"}.get(pre["kind"], "modq"), True)
        else:
            kc._C.quantize_int8_rowwise_convrot64(
                kc._wrap_for_dlpack(x_2d), kc._wrap_for_dlpack(x_q), kc._wrap_for_dlpack(x_s), gs, False,
                kc._input_act_code(None), 0, kc._wrap_for_dlpack(kc._act_weight_arg(None, None, dev, x_2d.dtype)),
                0.0, stream_ptr)
        cache["key"], cache["q"] = key, (x_q, x_s)
    if kc._prefer_cublas_int8_fallback(m, n, k, x_q.get_device()):
        return None
    ws = kc._int8_weight_scale_arg(w_scale, dev)
    ws = ws if ws.numel() == n else ws.expand(n).contiguous()
    return {"x_q": x_q, "x_s": x_s, "w_q": w_q, "ws": ws, "m": m, "n": n, "dev": dev, "stream": stream_ptr,
            "dtype": x.dtype, "shape": x.shape}


def _int8_linear_shared(x, weight, bias):
    """ComfyUI's int8 ConvRot linear with _int8_quantised's input; None when ComfyUI would take another path."""
    q = _int8_quantised(x, weight, bias)
    if q is None:
        return None
    kc = _kitchen()
    m, n, dev, stream_ptr = q["m"], q["n"], q["dev"], q["stream"]
    x_q, x_s, w_q, ws = q["x_q"], q["x_s"], q["w_q"], q["ws"]
    out = torch.empty((m, n), dtype=x.dtype, device=dev)
    used = kc._C.cutlass_int8_dequant(
        kc._wrap_for_dlpack(x_q), kc._wrap_for_dlpack(w_q), kc._wrap_for_dlpack(x_s), kc._wrap_for_dlpack(ws),
        kc._wrap_for_dlpack(kc._empty_cuda_tensor(dev, x.dtype)), kc._wrap_for_dlpack(out),
        kc.DTYPE_TO_CODE[x.dtype], stream_ptr)
    if not used:
        return None
    return out if x.dim() == 2 else out.reshape(*x.shape[:-1], n)


def _int8_linear_residual(x, weight, bias, res):
    """x_res + gate * linear(x) for the block's residual x_res (B, T, H, W, D) and gate (B, T, 1, 1, D): one CUTLASS
    GEMM with the residual epilogue per (batch, frame) group of rows. None when not possible (the caller runs the
    plain linear and the block its addcmul)."""
    xr, gate = res["x"], res["gate"]
    q = _int8_quantised(x, weight, bias)
    if q is None:
        return None
    kc = _kitchen()
    m, n, dev, stream_ptr = q["m"], q["n"], q["dev"], q["stream"]
    D = xr.shape[-1]
    if xr.dim() == 5:  # the block: (B, T, H, W, D) contiguous, gate (B, T, 1, 1, D)
        B, T, H, W, _ = xr.shape
        if not xr.is_contiguous() or tuple(gate.shape) != (B, T, 1, 1, D):
            return None
        groups, rows = B * T, H * W
        out = torch.empty_like(xr)
        xr3, out3 = xr.view(groups, rows, D), out.view(groups, rows, D)
    elif xr.dim() == 3:  # the row path: (B, N, D) with each item's rows contiguous, gate (B, 1, D)
        groups, rows, _ = xr.shape
        if xr.stride(2) != 1 or xr.stride(1) != D or tuple(gate.shape) != (groups, 1, D):
            return None
        out = torch.empty((groups, rows, D), dtype=xr.dtype, device=xr.device)
        xr3, out3 = xr, out
    else:
        return None
    if n != D or m != groups * rows or xr.dtype != q["dtype"] or gate.dtype != q["dtype"]:
        return None
    g2 = gate.reshape(groups, D)
    for gi in range(groups):
        sl = slice(gi * rows, (gi + 1) * rows)
        resid, o = xr3[gi], out3[gi]
        if not (kc._aligned16(resid) and kc._aligned16(o)):
            return None
        used = kc._C.cutlass_int8_dequant_residual(
            kc._wrap_for_dlpack(q["x_q"][sl]), kc._wrap_for_dlpack(q["w_q"]), kc._wrap_for_dlpack(q["x_s"][sl]),
            kc._wrap_for_dlpack(q["ws"]), kc._wrap_for_dlpack(kc._empty_cuda_tensor(dev, q["dtype"])),
            kc._wrap_for_dlpack(kc._gemm_vector_arg(g2[gi].contiguous(), dev, q["dtype"])),
            kc._wrap_for_dlpack(resid), kc._wrap_for_dlpack(o), kc.DTYPE_TO_CODE[q["dtype"]], stream_ptr)
        if not used:
            return None
    return out.view(*q["shape"][:-1], n)


def _pending_for(input):
    """(the pending input, whether ``input`` is it)."""
    pre = getattr(_tls, "pre", None)
    is_pre = (pre is not None and torch.is_tensor(input) and input.dim() >= 2
              and pre["key"] == _key2d(input.reshape(-1, input.shape[-1])))
    return pre, is_pre


def _shared_forward(input, weight, bias):
    """The _forward of a linear layer under _Overrides (ComfyUI already prepared the weight)."""
    pre, is_pre = _pending_for(input)
    if is_pre:
        pre["used"] = True
    out = _int8_linear_shared(input, weight, bias)
    if out is None:
        _count("qkv", False, f"weight {type(weight).__name__} {getattr(weight, '_layout_cls', None)}, "
                             f"input {type(input).__name__} {getattr(input, 'dtype', None)}")
        if is_pre:  # this layer takes another path: give it the real input
            input = _materialise(pre).reshape(input.shape)
        return torch.nn.functional.linear(input, weight, bias)
    _count("qkv", True)
    return out


def _residual_forward(input, weight, bias):
    """The _forward of a sub-layer's output projection with the block's residual pending (res)."""
    res = getattr(_tls, "res", None)
    if res is not None and not res["done"]:
        pre, is_pre = _pending_for(input)
        out = _int8_linear_residual(input, weight, bias, res)
        if out is not None:
            if is_pre:
                pre["used"] = True
            res["done"] = True
            _count("res", True)
            return out
        _count("res", False, f"input {tuple(input.shape)} {input.dtype}")
    return _shared_forward(input, weight, bias)


class _Overrides:
    """_shared_forward (or the given function) on some linear layers for the duration of a call (instance attribute
    over the class method)."""

    def __init__(self, layers):
        pairs = [x if isinstance(x, tuple) else (x, _shared_forward) for x in layers]
        self.layers = [(lin, fn) for lin, fn in pairs if "_forward" not in lin.__dict__]

    def __enter__(self):
        self.outer = getattr(_tls, "qkv", None) is not None
        if not self.outer:
            _tls.qkv = {}
        for lin, fn in self.layers:
            lin._forward = fn
        return self

    def __exit__(self, *exc):
        for lin, _fn in self.layers:
            lin.__dict__.pop("_forward", None)
        if not self.outer:
            _tls.qkv = None
        return False
