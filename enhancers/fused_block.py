# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Parts re-implement ComfyUI comfy/ldm/cosmos/predict2.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
Fused element-wise work for the Cosmos-Predict2 / Anima DiT block, bit-exact with ComfyUI (WEIGHTFERRY_FUSED_BLOCK, on
by default).

Measured (RTX 4060, Anima 2B int8 ConvRot, 896x1472, CFG 5): of one block's 33.6 ms, 24% is element-wise work
(normalisation, scale and shift, gated residual, GELU, copies) and 14% the int8 quantisation of the linear layers'
inputs -- memory-bound passes over the whole activation, on a GPU with 272 GB/s. The parts here remove passes without
changing a single bit of the result:

* ``mod``: ``norm(x) * (1 + scale) + shift`` (three times per block) as one Triton kernel after ComfyUI's LayerNorm,
  rounding to bfloat16 after the product and after the sum exactly like PyTorch's two kernels.
* ``qkv``: the self-attention's q, k and v projections take the same input; with int8 ConvRot weights it is quantised
  once (ComfyUI's own comfy_kitchen kernel) and used by the three GEMMs (ComfyUI quantises it three times).
* ``nhd``: SageAttention called with the (batch, tokens, heads, dim) layout the projections produce, so its output is
  already in the layout of the output projection (ComfyUI's (batch, heads, tokens, dim) call ends with a full copy).
* ``modq`` / ``gelu`` (int8 ConvRot weights): the scale-and-shift before the q / k / v, the cross-attention's q and the
  MLP's first layer, and the MLP's GELU (looked up in a table of PyTorch's own GELU of all 65536 bfloat16 values),
  computed inside the activation quantiser instead of being written to memory and read back.
* ``res`` (int8 ConvRot weights): the gated residual ``x + gate * out`` of the three sub-layers inside the output
  projection's GEMM (comfy_kitchen's CUTLASS epilogue with a residual: the same bits as PyTorch's addcmul on the bf16
  output, compared on two layers), one GEMM per (batch, frame) since the gate differs between cond and uncond.
* ``ln`` (with modq): the LayerNorm before the scale-and-shift inside the quantiser as well, so the normalised input
  is never written: PyTorch's vectorized layer-norm kernel reproduced (128 threads per row, each reading 4 vectors of
  4 values, a Welford update with fused multiply-adds, a warp shuffle-down reduction 16 .. 1 and the 4 warps merged
  2 + 2 then 1 + 1, rsqrt.approx; verified bit-exact against PyTorch).
* ``tay``: TaylorSeer's prediction of a skipped step (Taylor predictor) and its addition to the block input in one
  pass, with PyTorch's order and bf16 rounding after every operation (zeros, + factor * 1/i! * dt^i per order, then
  hidden + prediction).
* ``rms``: the cross-attention's q normalisation (RMSNorm over rows of 128) as a Triton kernel with PyTorch's order
  of operations (4 consecutive squares per thread, a warp xor-shuffle sum 16, 8, 4, 2, 1, rsqrt.approx, (x * rstd) *
  weight; verified bit-exact against PyTorch).
* ``smk`` (2026-10-11): SageAttention's key smoothing (``k - mean(k)``, which sageattn() applies on
  RTX 40 GPUs) inside its key quantiser, V's per-channel fp8 in one kernel, and the queries' normalisation (the
  cross-attention's RMSNorm, the self-attention's RMSNorm + RoPE as comfy_kitchen computes it) inside the query
  quantiser, instead of written copies (enhancers/sage_fused.py; the same bits as kitchen + sageattn).

The int8 quantiser (modq / gelu / ln) is a Triton reproduction of comfy_kitchen's quantize_int8_rowwise_convrot64
(compiled, no source): the 256-wide "regular" Hadamard kron(H4, H4, H4, H4) / 16 in float32 (radix-4 stages of
stride 1, 4, 16, 64, each output summed left to right), the row scale amax / 127, then the rotated value and the scale
rounded to bfloat16, their quotient rounded to bfloat16 and to the nearest integer (ties to even). Verified against
kitchen's output on a real block's inputs (every value and row scale identical). A layer that receives such a pending
input but takes another path computes the real input first; a pending input that no layer consumed is recomputed the
plain way.

The row-chunked block of activation streaming (streaming/activation.py, _cosmos_block_rows) uses the same parts
through rows_project / rows_attention / rows_out_residual / rows_sub, on (batch, rows, channels) tensors.

Each part checks its preconditions per call (dtype, layout, the comfy_kitchen path ComfyUI itself would take) and
otherwise runs ComfyUI's code. Installed by patching the ComfyUI classes ``Block``, ``Attention`` and
``GPT2FeedForward`` of comfy/ldm/cosmos/predict2.py (the row-chunked block of activation streaming keeps working: it
calls the patched forward when it does not chunk).

Files: this module holds the public functions, the patched methods and the block; fused_kernels.py the Triton kernels,
fused_int8.py the int8 linear path (pending inputs, shared quantisation, residual GEMM), fused_state.py the shared
state. Every name of those modules used before the split stays importable from here.
"""

import torch

from ..core import env as _env
from ..utils.logging import log_info, log_warn
from .fused_state import _count, _state, _stats, _tls  # noqa: F401 -- shared state, one object each
from .fused_kernels import _get_kernels, _kernels, tl, triton  # noqa: F401 -- re-exported (fb._get_kernels)
from .fused_int8 import (  # noqa: F401 -- the int8 linear path; every name stays importable from here
    _clear_pre, _gelu_table, _gelu_tables, _int8_linear_residual, _int8_linear_shared, _int8_proj_ok,
    _int8_quantised, _key2d, _kitchen, _materialise, _Overrides, _pending_for, _quantise_pre, _residual_forward,
    _set_pre, _shared_forward,
)

PARTS = ("mod", "qkv", "nhd", "modq", "gelu", "rms", "res", "ln", "tay", "smk")
DEFAULT_PARTS = PARTS  # every part (smk on by default since 2026-10-11, the user's decision)
_DEFER_QNORM = True  # smk: the cross-attention's q norm inside the query quantiser (module constant for benchmarks)
_DEFER_QROPE = True  # smk: the self-attention's q norm + RoPE inside the query quantiser (module constant)
_TRITON_PARTS = ("mod", "modq", "gelu", "rms", "ln", "tay", "smk")
# activation streaming (rows): SageAttention's key / value preparation once per block, not per row chunk
# (rows_attention; WEIGHTFERRY_ACT_KV_ONCE, on by default since 2026-10-12, 0 turns it off; module constant)
_ROWS_KV = _env.get("WEIGHTFERRY_ACT_KV_ONCE") == "1"


def stats(reset=False):
    out = {k: dict(v) for k, v in _stats.items()}
    if reset:
        _stats.clear()
    return out


def wanted_parts():
    """WEIGHTFERRY_FUSED_BLOCK: 1 / unset / all = every part (default), 0 = off, or a comma separated list of
    PARTS."""
    v = (_env.get("WEIGHTFERRY_FUSED_BLOCK") or "1").strip().lower()
    if v in ("", "0", "off", "false"):
        return ()
    if v in ("1", "on", "true"):
        return DEFAULT_PARTS
    if v == "all":
        return PARTS
    return tuple(p for p in (s.strip() for s in v.split(",")) if p in PARTS)


def modulate(normed, scale, shift):
    """normed * (1 + scale) + shift for normed (B, T, H, W, D) and scale / shift (B, T, 1, 1, D), all bfloat16; or
    normed (B, N, D) with scale / shift (B, 1, D) (the row path)."""
    if normed.dim() == 3 and scale.dim() == 3:
        B3, N3, D3 = normed.shape
        if (normed.is_contiguous() and tuple(scale.shape) == (B3, 1, D3) and tuple(shift.shape) == (B3, 1, D3)):
            return modulate(normed.view(B3, 1, 1, N3, D3), scale.view(B3, 1, 1, 1, D3),
                            shift.view(B3, 1, 1, 1, D3)).view(B3, N3, D3)
        _count("mod", False, "3-D input not contiguous")
        return normed * (1 + scale) + shift
    B, T, H, W, D = normed.shape
    if (normed.dtype != torch.bfloat16 or scale.dtype != torch.bfloat16 or shift.dtype != torch.bfloat16
            or not normed.is_cuda or not normed.is_contiguous() or tuple(scale.shape) != (B, T, 1, 1, D)
            or tuple(shift.shape) != (B, T, 1, 1, D)):
        _count("mod", False, f"{normed.dtype}/{scale.dtype}/{shift.dtype} contiguous={normed.is_contiguous()}")
        return normed * (1 + scale) + shift
    _count("mod", True)
    one_plus = (1 + scale).contiguous()  # the same small bf16 tensor PyTorch makes
    out = torch.empty_like(normed)
    _get_kernels()["mod"][(B * T * H * W,)](normed, one_plus, shift.contiguous(), out, D, H * W,
                                            BLOCK=triton.next_power_of_2(D), num_warps=8 if D >= 2048 else 4)
    return out


def rms_norm_128(x, weight, eps, rs=0):
    """torch.nn.functional.rms_norm(x, (128,), weight, eps) for bf16 x (..., 128) and a bf16 weight; None when not
    eligible (the caller runs PyTorch's)."""
    if (x.dtype != torch.bfloat16 or not x.is_cuda or x.shape[-1] != 128 or not x.is_contiguous() or weight is None
            or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (128,) or weight.device != x.device
            or eps is None):
        return None
    n = x.numel() // 128
    out = torch.empty_like(x)
    rows = 32
    _get_kernels()["rms128"][(triton.cdiv(n, rows),)](x, weight.contiguous(), out, n, float(eps), ROWS=rows, RS=rs,
                                                      num_warps=4)
    return out


def _rms_forward(mod):
    """forward of ComfyUI's RMSNorm (comfy.ops) with the Triton kernel for rows of 128."""
    import comfy.ops

    def forward(x):
        comfy.ops.run_every_op()
        if mod.comfy_cast_weights or len(mod.weight_function) > 0 or len(mod.bias_function) > 0:
            with comfy.ops.CastBiasWeightContext(mod if mod.weight is not None else None, x,
                                                 offloadable=True) as (weight, bias):
                out = rms_norm_128(x, weight, mod.eps) if tuple(mod.normalized_shape) == (128,) else None
                if out is None:
                    _count("rms", False, f"{x.dtype} {tuple(x.shape)}")
                    return torch.nn.functional.rms_norm(x, mod.normalized_shape, weight, mod.eps)
        else:
            out = rms_norm_128(x, mod.weight, mod.eps) if tuple(mod.normalized_shape) == (128,) else None
            if out is None:
                _count("rms", False, f"{x.dtype} {tuple(x.shape)}")
                return torch.nn.functional.rms_norm(x, mod.normalized_shape, mod.weight, mod.eps)
        _count("rms", True)
        return out
    return forward


def ln_modulate(norm, x, scale, shift):
    """norm(x) * (1 + scale) + shift, bit-exact, in one pass when allowed (ln: the LayerNorm reproduction; TaylorSeer's
    per-step decision input), else with the fused scale-and-shift (mod), else PyTorch's."""
    if _ln_ok(norm, x) and x.dim() == 5:
        B, T, H, W, D = x.shape
        if (scale.dtype == torch.bfloat16 and shift.dtype == torch.bfloat16
                and tuple(scale.shape) == (B, T, 1, 1, D) and tuple(shift.shape) == (B, T, 1, 1, D)):
            _count("ln", True)
            out = torch.empty_like(x)
            _get_kernels()["lnmod"][(B * T * H * W,)](x, (1 + scale).contiguous(), shift.contiguous(), out, H * W,
                                                      float(norm.eps), K=D, num_warps=8)
            return out
    if "mod" in _state["parts"]:
        return modulate(norm(x), scale, shift)
    return norm(x) * (1 + scale) + shift


def taylor_predict_add(hidden, sections, factor_sets, dts):
    """hidden + cat([TaylorSeer._taylor_eval(factors, dt) for each section]) in one pass per section (tay), the same
    bits; None when not eligible (the caller runs TaylorSeer's code). factor_sets: per section the factors dict
    {0: f0, 1: f1, ...}."""
    import math
    if ("tay" not in _state["parts"] or triton is None or hidden.dtype != torch.bfloat16 or not hidden.is_cuda
            or not hidden.is_contiguous()):
        return None
    out = torch.empty_like(hidden)
    rows = 0
    plans = []
    for sec, factors, dt in zip(sections, factor_sets, dts):
        orders = sorted(factors)
        if orders != list(range(len(orders))) or not 1 <= len(orders) <= 3:
            return None
        part = hidden[sec]
        fs = [factors[i] for i in orders]
        if any(f.dtype != torch.bfloat16 or f.device != hidden.device or not f.is_contiguous()
               or f.shape != part.shape for f in fs):
            return None
        plans.append((sec, fs, dt, len(orders)))
        rows += part.shape[0]
    if rows != hidden.shape[0]:
        return None
    k = _get_kernels()["tay"]
    for sec, fs, dt, nf in plans:
        h, o = hidden[sec], out[sec]
        if not (h.is_contiguous() and o.is_contiguous()):
            return None
        n = h.numel()
        cs = [1.0 / math.factorial(i) for i in range(3)]
        ds = [float(dt ** i) for i in range(3)]
        f3 = (fs + [fs[0]] * 3)[:3]
        k[(triton.cdiv(n, 4096),)](h, f3[0], f3[1], f3[2], o, n, cs[0], cs[1], cs[2], ds[0], ds[1], ds[2],
                                   NF=nf, BLOCK=4096, num_warps=8)
    if not _stats.get("tay"):
        log_info("Anima fused block: TaylorSeer's prediction fused (tay).")
    _count("tay", True)
    return out


def _deferred_rms(mod):
    """forward of the cross-attention's q_norm when its output goes straight into SageAttention (smk): the input is
    returned unchanged and the normalisation left pending (_tls.qnorm) for the query quantiser, which applies it on
    the fly (enhancers/sage_fused.py); any other consumer gets it materialised first (_qnorm_real). A weight that
    ComfyUI casts or patches per call (Dynamic VRAM casts every call) is copied inside the cast (128 values): the
    cast buffer may be reused after it."""
    import comfy.ops
    plain = _rms_forward(mod)

    def forward(x):
        if (tuple(mod.normalized_shape) != (128,) or mod.eps is None or x.dtype != torch.bfloat16 or not x.is_cuda
                or not x.is_contiguous()):
            return plain(x)
        comfy.ops.run_every_op()
        if mod.comfy_cast_weights or len(mod.weight_function) > 0 or len(mod.bias_function) > 0:
            with comfy.ops.CastBiasWeightContext(mod if mod.weight is not None else None, x,
                                                 offloadable=True) as (weight, _bias):
                w = None if weight is None else weight.clone()
        else:
            w = mod.weight
        if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != (128,) or w.device != x.device:
            _count("qnorm", False, f"weight {None if w is None else (w.dtype, tuple(w.shape))}")
            return plain(x)
        _tls.qnorm = {"key": (x.data_ptr(), tuple(x.shape), tuple(x.stride())), "w": w.contiguous(),
                      "eps": float(mod.eps)}
        return x
    return forward


def _selfattn_qkv_rope(self, x, rope_emb, projs):
    """ComfyUI's Attention.compute_qkv for a self-attention without patches, with the q / k RMSNorm + split-half RoPE
    by the Suite's reproduction of comfy_kitchen's rms_rope_split_half (enhancers/sage_fused.py, the same bits): k is
    written, q is left pending (_tls.qnorm, kind "rope") for SageAttention's query quantiser (smk), or materialised by
    _qnorm_real for any other consumer. Shapes the reproduction does not cover take kitchen's call."""
    import comfy.ops
    import comfy.quant_ops
    from . import sage_fused
    with _Overrides(projs):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
    q, k, v = (t.view(*t.shape[:-1], self.n_heads, self.head_dim) for t in (q, k, v))
    v = self.v_norm(v)
    q_scale, _, q_os = comfy.ops.cast_bias_weight(self.q_norm, q, offloadable=True)
    k_scale, _, k_os = comfy.ops.cast_bias_weight(self.k_norm, k, offloadable=True)
    try:
        if sage_fused.rope_ok(q, k, rope_emb, q_scale, k_scale, self.q_norm.eps):
            k_out = sage_fused.rms_rope(k, rope_emb, k_scale, self.q_norm.eps)
            _tls.qnorm = {"kind": "rope", "key": (q.data_ptr(), tuple(q.shape), tuple(q.stride())), "f": rope_emb,
                          "w": q_scale.clone(), "eps": float(self.q_norm.eps)}
            q_out = q
        else:
            _count("qrope", False, f"q {tuple(q.shape)} {q.dtype} rope {tuple(rope_emb.shape)} {rope_emb.dtype}")
            q_out, k_out = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_emb, q_scale, k_scale, self.q_norm.eps)
    finally:
        comfy.ops.uncast_bias_weight(self.q_norm, q_scale, None, q_os)
        comfy.ops.uncast_bias_weight(self.k_norm, k_scale, None, k_os)
    return q_out, k_out, v


def _qnorm_match(q, pend):
    return (pend is not None and torch.is_tensor(q)
            and pend["key"] == (q.data_ptr(), tuple(q.shape), tuple(q.stride())))


def _qnorm_real(q, pend):
    """q with a pending RMSNorm (_deferred_rms) applied: the rms128 kernel's output, as without the deferral. q may
    come in ComfyUI's AttentionTensorContainer (Attention.forward wraps q / k / v): the result is wrapped again. A
    pending normalisation that does not belong to q is an error (q would go on unnormalised)."""
    if pend is None:
        return q
    from comfy.ldm.modules.attention import AttentionTensorContainer
    if isinstance(q, AttentionTensorContainer):
        return AttentionTensorContainer(_qnorm_real(q.take(), pend))
    if not _qnorm_match(q, pend):
        raise RuntimeError("WeightFerry fused block: a pending q normalisation does not match the attention's q")
    if pend.get("kind") == "rope":
        from . import sage_fused
        _count("qrope", False, "materialised for another consumer")
        return sage_fused.rms_rope(q, pend["f"], pend["w"], pend["eps"])
    _count("rms", True)
    _count("qnorm", False, "materialised for another consumer")
    return rms_norm_128(q, pend["w"], pend["eps"])


def _sage_next(attn):
    """Whether this attention's compute_attention will take SageAttention (the only consumer that applies a pending q
    normalisation on the fly; for any other the q normalisation is not deferred at all)."""
    import comfy.ldm.cosmos.predict2 as p2
    return _is_sage(attn, _attention_used(attn, p2.optimized_attention), p2)


def _rms_ok(self):
    return ("rms" in _state["parts"] and type(self.q_norm).__name__ == "RMSNorm"
            and "forward" not in self.q_norm.__dict__)


def _compute_qkv(self, x, context=None, rope_emb=None, transformer_options={}):
    orig = _state["orig"]["compute_qkv"]
    if (context is not None and not getattr(self, "is_selfattn", False) and "rms" in _state["parts"]
            and "attn2_patch" not in transformer_options.get("patches", {}) and _rms_ok(self)):
        # cross-attention: its q normalisation (rows of 128, no RoPE) by the Triton kernel, or with smk left to
        # SageAttention's query quantiser
        _tls.qnorm = None
        defer = (_DEFER_QNORM and "smk" in _state["parts"] and "nhd" in _state["parts"]  # read by _compute_attention
                 and _sage_next(self))
        self.q_norm.forward = (_deferred_rms if defer else _rms_forward)(self.q_norm)
        try:
            return orig(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
        finally:
            self.q_norm.__dict__.pop("forward", None)
    if (context is not None or not getattr(self, "is_selfattn", False) or "qkv" not in _state["parts"]
            or ("attn1_patch" in transformer_options.get("patches", {}))):
        return orig(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
    projs = [p for p in (self.q_proj, self.k_proj, self.v_proj) if hasattr(p, "_forward")]
    if len(projs) != 3:
        _count("qkv", False, "projections without _forward")
        return orig(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)
    if (_DEFER_QROPE and rope_emb is not None and "smk" in _state["parts"] and "nhd" in _state["parts"]
            and triton is not None and _sage_next(self)):
        _tls.qnorm = None
        return _selfattn_qkv_rope(self, x, rope_emb, projs)
    with _Overrides(projs):
        return orig(self, x, context, rope_emb=rope_emb, transformer_options=transformer_options)


# ---------------------------------------------------------------------------------------------------------- nhd
def _attention_used(attn, fn):
    """The attention function that will run: the module's preferred one (the streaming node's sage_attention option
    sets ComfyUI's registered "sage" there) or ComfyUI's global choice ``fn``."""
    return getattr(getattr(attn, "comfy_attention", None), "function", None) or fn


def _is_sage(attn, used, p2):
    """SageAttention through ComfyUI's torch attention op: the only function known to give the same bits in both
    layouts (measured); anything else keeps ComfyUI's call."""
    return (getattr(attn, "attn_op", None) is p2.torch_attention_op
            and getattr(used, "__name__", "") == "attention_sage")


def _compute_attention(self, q, k, v, transformer_options={}):
    import comfy.ldm.cosmos.predict2 as p2
    from comfy.ldm.modules.attention import AttentionTensorContainer
    orig = _state["orig"]["compute_attention"]
    pend = getattr(_tls, "qnorm", None)  # the cross-attention's pending q normalisation (_deferred_rms), if any
    _tls.qnorm = None
    fn = p2.optimized_attention
    used = _attention_used(self, fn)
    if not _is_sage(self, used, p2):
        _count("nhd", False, f"attention {getattr(used, '__name__', used)}")
        return orig(self, _qnorm_real(q, pend), k, v, transformer_options=transformer_options)
    qt, kt, vt = (t.peek() if isinstance(t, AttentionTensorContainer) else t for t in (q, k, v))
    if (qt.dim() != 4 or kt.dim() != 4 or not (qt.is_contiguous() and kt.is_contiguous() and vt.is_contiguous())
            or qt.dtype not in (torch.bfloat16, torch.float16)):
        _count("nhd", False, f"q {tuple(qt.shape)} contiguous={qt.is_contiguous()} {qt.dtype}")
        return orig(self, _qnorm_real(q, pend), k, v, transformer_options=transformer_options)
    _count("nhd", True)
    qt, kt, vt = (t.take() if isinstance(t, AttentionTensorContainer) else t for t in (q, k, v))
    b, s, h, d = qt.shape
    if pend is not None and not _qnorm_match(qt, pend):
        raise RuntimeError("WeightFerry fused block: a pending q normalisation does not match the attention's q")
    if "smk" in _state["parts"]:  # SageAttention's K / V (and a pending q norm) preparation (enhancers/sage_fused.py)
        out = _sage_smooth_k(qt, kt, vt, transformer_options, pend)
        if out is not None:
            return self.output_dropout(self.output_proj(out.reshape(b, s, h * d)))
    qt = _qnorm_real(qt, pend)
    out = fn(qt.view(b, s, h * d), kt.view(b, kt.shape[1], h * d), vt.view(b, vt.shape[1], h * d), h,
             skip_reshape=False, preferred_attention=self.comfy_attention, transformer_options=transformer_options)
    return self.output_dropout(self.output_proj(out))


def _sage_smooth_k(q, k, v, transformer_options, qnorm=None):
    """attention_sage's call on (batch, tokens, heads, dim) inputs with the key smoothing fused (smk); None when this
    call would not end in the sm89 kernel it reproduces (the caller then runs ComfyUI's)."""
    from . import sage_fused
    if "optimized_attention_override" in transformer_options:
        _count("smk", False, "optimized_attention_override")
        return None
    pv = sage_fused.available(q)
    if pv is None or not (k.is_contiguous() and v.is_contiguous()):
        _count("smk", False, "not SageAttention's sm89 FP8 P*V path")
        return None
    try:
        if qnorm is not None and qnorm.get("kind") == "rope":
            out = sage_fused.sage_nhd(q, k, v, pv, q_rope=(qnorm["f"], qnorm["w"], qnorm["eps"]))
        else:
            out = sage_fused.sage_nhd(q, k, v, pv, None if qnorm is None else (qnorm["w"], qnorm["eps"]))
    except Exception as exc:  # noqa: BLE001 -- ComfyUI's call instead
        _count("smk", False, repr(exc)[:120])
        return None
    _count("smk", out is not None, None if out is not None else f"q {tuple(q.shape)} k {tuple(k.shape)}")
    if out is not None and qnorm is not None:
        if qnorm.get("kind") == "rope":
            _count("qrope", True)
        else:
            _count("rms", True)
            _count("qnorm", True)
    return out


# ---------------------------------------------------------------------------------------------------------- mlp
def _mlp_forward(self, x):
    """ComfyUI's GPT2FeedForward.forward (layer1 -> GELU -> layer2), with GELU inside layer2's quantiser (gelu) and a
    pending scale-and-shift on x taken by layer1's quantiser (modq)."""
    orig = _state["orig"]["mlp_forward"]
    parts = _state["parts"]
    act = getattr(self, "activation", None)
    want_gelu = ("gelu" in parts and type(act) is torch.nn.GELU and getattr(act, "approximate", "none") == "none"
                 and _int8_proj_ok(self.layer2))
    pre = getattr(_tls, "pre", None)
    if not want_gelu and pre is None:
        return orig(self, x)
    with _Overrides([self.layer1, self.layer2]):
        h = self.layer1(x)
        if pre is not None:
            _clear_pre()
            if not pre["used"]:  # layer1 never saw the pending input: compute it the plain way
                _count("modq", False, "mlp.layer1 did not take the pending input")
                h = self.layer1(_materialise(pre).view(x.shape))
        if not (want_gelu and h.dtype == torch.bfloat16 and h.is_cuda and h.is_contiguous()):
            return self.layer2(self.activation(h))
        g = _set_pre(h, "gelu")
        try:
            out = self.layer2(h)
        finally:
            _clear_pre()
        if not g["used"]:
            _count("gelu", False, "mlp.layer2 did not take the pending input")
            out = self.layer2(self.activation(h))
        return out


# ---------------------------------------------------------------------------------------------------------- block
def _pending_ok(block, transformer_options, ln, compute_dtype):
    """modq is safe here: no patches see the normalised input, the attention / MLP modules run ComfyUI's code and their
    input projections hand the input unchanged to _forward."""
    # the self-attention's q / k / v take the pending input through compute_qkv's overrides (qkv)
    if "modq" not in _state["parts"] or "qkv" not in _state["parts"] or transformer_options.get("patches"):
        return False
    if ln.dtype != torch.bfloat16 or compute_dtype != torch.bfloat16 or not ln.is_cuda or not ln.is_contiguous():
        return False
    sa, ca, mlp = block.self_attn, block.cross_attn, block.mlp
    if any("forward" in m.__dict__ or "compute_qkv" in m.__dict__ for m in (sa, ca)) or "forward" in mlp.__dict__:
        return False
    return all(_int8_proj_ok(lin) for lin in (sa.q_proj, sa.k_proj, sa.v_proj, ca.q_proj, mlp.layer1))


def _ln_ok(norm, x):
    """The LayerNorm can go inside the quantiser (ln): ComfyUI's LayerNorm without affine on a contiguous bf16 input
    whose rows PyTorch's vectorized kernel splits like the reproduction (multiples of 512, and a power of 2 for the
    kernel's row range)."""
    return ("ln" in _state["parts"] and isinstance(norm, torch.nn.LayerNorm) and "forward" not in norm.__dict__
            and norm.weight is None and norm.bias is None and len(norm.normalized_shape) == 1
            and x.shape[-1] == norm.normalized_shape[0] and x.shape[-1] % 512 == 0
            and not x.shape[-1] & (x.shape[-1] - 1)  # the kernel's row is one power-of-2 range (not Cosmos 14B's 5120)
            and x.dtype == torch.bfloat16
            and x.is_cuda and x.is_contiguous()
            # ComfyUI's cast path (comfy_cast_weights) has nothing to cast without weight and bias
            and len(getattr(norm, "weight_function", ())) == 0 and len(getattr(norm, "bias_function", ())) == 0)


def _run_sub(block, transformer_options, compute_dtype, module, ln, scale, shift, call, x_res, gate, out_layer,
             overrides=()):
    """Run one sub-layer on modulate(ln): with the scale-and-shift pending when allowed, else materialised; with the
    gated residual x_res + gate * out inside out_layer's GEMM when allowed (res). Returns (out, residual done). For the
    block (5-D tensors) and the row path (3-D). ``ln`` may be a pair (norm, x): the LayerNorm of x, computed only when
    it is not taken into the quantiser (ln)."""
    use_res = ("res" in _state["parts"] and out_layer is not None and _int8_proj_ok(out_layer)
               and x_res is not None and x_res.dtype == compute_dtype and not transformer_options.get("patches"))

    def run(inp):
        r = {"x": x_res, "gate": gate, "done": False} if use_res else None
        _tls.res = r
        try:
            with _Overrides(list(overrides) + ([(out_layer, _residual_forward)] if use_res else [])):
                out = call(inp)
        finally:
            _tls.res = None
        return out, bool(r and r["done"])

    if isinstance(ln, tuple):
        norm, x_in = ln
        if _ln_ok(norm, x_in) and _pending_ok(block, transformer_options, x_in, compute_dtype):
            pre = _set_pre(x_in, "lnmod", scale, shift, norm=norm)
            try:
                out = run(x_in)
            finally:
                _clear_pre()
            if pre["used"]:
                return out
            _count("ln", False, f"{type(module).__name__} did not take the pending input")
        ln = norm(x_in)
    if _pending_ok(block, transformer_options, ln, compute_dtype):
        pre = _set_pre(ln, "mod", scale, shift)
        try:
            out = run(ln)
        finally:
            _clear_pre()
        if pre["used"]:
            return out
        _count("modq", False, f"{type(module).__name__} did not take the pending input")
    return run(modulate(ln, scale, shift))


def _add_residual(x_res, gate, result, done, residual_dtype):
    """x_res + gate * result (ComfyUI's addcmul), or ``result`` itself when the residual was already added inside
    the output projection's GEMM (done, res)."""
    if done:
        return result
    return torch.addcmul(x_res, gate.to(residual_dtype), result.to(residual_dtype))


# ---------------------------------------------------------------------------------------------------------- row path
def rows_active(block, transformer_options, compute_dtype):
    """The row-chunked block (activation streaming) uses the fused parts: installed, no patches, bf16."""
    return bool(_state["parts"]) and "mod" in _state["parts"] and not (transformer_options or {}).get("patches") \
        and compute_dtype == torch.bfloat16


def rows_project(block, layers, ln, scale, shift, compute_dtype, transformer_options):
    """[layer(modulate(ln, scale, shift)) for layer in layers] for one row chunk (B, N, D): the scale-and-shift pending
    for the quantiser (modq) and one quantisation shared by the layers (qkv) when allowed."""
    outs, _done = _run_sub(block, transformer_options, compute_dtype, layers[0], ln, scale, shift,
                           lambda n: [lin(n.to(compute_dtype)) for lin in layers], None, None, None, tuple(layers))
    return outs


def rows_attention(attn, q, K, V, transformer_options, kv_cache=None):
    """The self-attention of one query chunk q (B, n, H, Dh) against the full K / V (B, S, H, Dh), as (B, n, H * Dh):
    SageAttention in the projections' layout (nhd) when ComfyUI would run SageAttention, else ComfyUI's call. With
    smk and kv_cache (a dict the caller keeps for one block's K / V and drops after it): SageAttention's key / value
    preparation (key mean, smoothed int8 keys, fp8 values) is done once for all the chunks instead of in every call
    (enhancers/sage_fused.prepare_kv), the same bits."""
    import comfy.ldm.cosmos.predict2 as p2
    from comfy.ldm.modules.attention import AttentionTensorContainer
    fn = p2.optimized_attention
    used = _attention_used(attn, fn)
    if ("nhd" in _state["parts"] and _is_sage(attn, used, p2) and q.is_contiguous() and K.is_contiguous()
            and V.is_contiguous() and q.dim() == 4):
        _count("nhd", True)
        b, n, h, d = q.shape
        if (kv_cache is not None and "smk" in _state["parts"] and _ROWS_KV
                and "optimized_attention_override" not in transformer_options):
            from . import sage_fused
            pv = sage_fused.available(q)
            out = None
            if pv is not None:
                try:
                    if kv_cache.get("pv") != pv:
                        kv_cache["kv"], kv_cache["pv"] = sage_fused.prepare_kv(K, V, pv), pv
                    if kv_cache["kv"] is not None:
                        out = sage_fused.sage_nhd(q, K, V, pv, kv=kv_cache["kv"])
                except Exception as exc:  # noqa: BLE001 -- ComfyUI's call instead
                    _count("rows_kv", False, repr(exc)[:120])
                    out = None
            _count("rows_kv", out is not None, None if out is not None else "not SageAttention's sm89 case")
            if out is not None:
                return out.view(b, n, h * d)
        return fn(q.view(b, n, h * d), K.view(b, K.shape[1], h * d), V.view(b, V.shape[1], h * d), h,
                  skip_reshape=False, preferred_attention=attn.comfy_attention, transformer_options=transformer_options)
    return attn.attn_op(AttentionTensorContainer(q), AttentionTensorContainer(K), AttentionTensorContainer(V),
                        transformer_options=transformer_options, preferred_attention=attn.comfy_attention)


def rows_out_residual(lin, dropout, inp, x_res, gate, residual_dtype):
    """x_res + gate * dropout(lin(inp)) for one row chunk: inside lin's GEMM (res) when allowed, else addcmul."""
    if ("res" in _state["parts"] and isinstance(dropout, torch.nn.Identity) and _int8_proj_ok(lin)
            and x_res.dtype == residual_dtype):
        r = {"x": x_res, "gate": gate, "done": False}
        _tls.res = r
        try:
            with _Overrides([(lin, _residual_forward)]):
                out = lin(inp)
        finally:
            _tls.res = None
        return _add_residual(x_res, gate, out, r["done"], residual_dtype)
    return torch.addcmul(x_res, gate.to(residual_dtype), dropout(lin(inp)).to(residual_dtype))


def rows_sub(block, transformer_options, compute_dtype, residual_dtype, module, ln, scale, shift, call, x_res, gate,
             out_layer, overrides=()):
    """x_res + gate * sub-layer(modulate(ln)) for one row chunk, with the fused parts (_run_sub)."""
    out, done = _run_sub(block, transformer_options, compute_dtype, module, ln, scale, shift, call, x_res, gate,
                         out_layer, overrides)
    return _add_residual(x_res, gate, out, done, residual_dtype)


def _block_forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                   extra_per_block_pos_emb=None, transformer_options={}):
    """ComfyUI's Block.forward (comfy/ldm/cosmos/predict2.py) with ``modulate`` for the three scale-and-shift steps
    (or, with modq, the scale-and-shift left pending for the input projections' quantiser); the same operations in the
    same order otherwise."""
    if "mod" not in _state["parts"]:
        return _state["orig"]["forward"](self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D,
                                         adaln_lora_B_T_3D, extra_per_block_pos_emb, transformer_options)
    from einops import rearrange
    residual_dtype = x_B_T_H_W_D.dtype
    compute_dtype = emb_B_T_D.dtype
    if extra_per_block_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
    if self.use_adaln_lora:
        shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
    else:
        shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
        shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    def b5(t):
        return rearrange(t, "b t d -> b t 1 1 d")

    shift_sa, scale_sa, gate_sa = b5(shift_sa), b5(scale_sa), b5(gate_sa)
    shift_ca, scale_ca, gate_ca = b5(shift_ca), b5(scale_ca), b5(gate_ca)
    shift_mlp, scale_mlp, gate_mlp = b5(shift_mlp), b5(scale_mlp), b5(gate_mlp)
    B, T, H, W, D = x_B_T_H_W_D.shape

    def sub(module, ln, scale, shift, call, x_res, gate, out_layer, overrides=()):
        return _run_sub(self, transformer_options, compute_dtype, module, ln, scale, shift, call, x_res, gate,
                        out_layer, overrides)

    result, done = sub(self.self_attn, (self.layer_norm_self_attn, x_B_T_H_W_D), scale_sa, shift_sa,
                       lambda n: self.self_attn(rearrange(n.to(compute_dtype), "b t h w d -> b (t h w) d"), None,
                                                rope_emb=rope_emb_L_1_1_D, transformer_options=transformer_options),
                       x_B_T_H_W_D, gate_sa, self.self_attn.output_proj)
    result = rearrange(result, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = _add_residual(x_B_T_H_W_D, gate_sa, result, done, residual_dtype)

    result, done = sub(self.cross_attn, (self.layer_norm_cross_attn, x_B_T_H_W_D), scale_ca, shift_ca,
                       lambda n: self.cross_attn(rearrange(n.to(compute_dtype), "b t h w d -> b (t h w) d"),
                                                 crossattn_emb, rope_emb=rope_emb_L_1_1_D,
                                                 transformer_options=transformer_options),
                       x_B_T_H_W_D, gate_ca, self.cross_attn.output_proj, overrides=(self.cross_attn.q_proj,))
    result = rearrange(result, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = _add_residual(x_B_T_H_W_D, gate_ca, result, done, residual_dtype)

    patches = transformer_options.get("patches", {})
    if "mlp_patch" in patches:
        normalized = modulate(self.layer_norm_mlp(x_B_T_H_W_D), scale_mlp, shift_mlp).to(compute_dtype)
        args = {"x": normalized, "transformer_options": transformer_options}
        for patch in patches["mlp_patch"]:
            args = patch(args)
        result, done = self.mlp(args["x"]), False
    else:
        result, done = sub(self.mlp, (self.layer_norm_mlp, x_B_T_H_W_D), scale_mlp, shift_mlp,
                           lambda n: self.mlp(n.to(compute_dtype)), x_B_T_H_W_D, gate_mlp, self.mlp.layer2)
    return _add_residual(x_B_T_H_W_D, gate_mlp, result, done, residual_dtype)


def install(parts=None):
    """Patch ComfyUI's Cosmos Block / Attention / GPT2FeedForward for the given parts (default: WEIGHTFERRY_FUSED_BLOCK).
    Idempotent; install(()) restores ComfyUI's methods."""
    try:
        import comfy.ldm.cosmos.predict2 as p2
    except Exception as exc:  # noqa: BLE001
        log_warn(f"Anima fused block: ComfyUI's Cosmos model not found ({exc}); not installed.")
        return ()
    parts = tuple(wanted_parts() if parts is None else parts)
    if _state["orig"] is None:
        _state["orig"] = {"forward": p2.Block.forward, "compute_qkv": p2.Attention.compute_qkv,
                          "compute_attention": p2.Attention.compute_attention,
                          "mlp_forward": p2.GPT2FeedForward.forward}
    orig = _state["orig"]
    if any(p in parts for p in _TRITON_PARTS):
        try:
            if triton is None:
                raise ImportError("triton")
            _get_kernels()
        except Exception as exc:  # noqa: BLE001 -- no Triton: those parts stay off
            log_warn(f"Anima fused block: Triton not available ({exc}); {', '.join(_TRITON_PARTS)} stay off.")
            parts = tuple(p for p in parts if p not in _TRITON_PARTS)
    if "modq" in parts and "mod" not in parts:  # modq lives in the fused block forward
        parts = tuple(p for p in parts if p != "modq")
    _state["parts"] = parts
    p2.Block.forward = _block_forward if "mod" in parts else orig["forward"]
    p2.Attention.compute_qkv = _compute_qkv if ("qkv" in parts or "rms" in parts) else orig["compute_qkv"]
    p2.Attention.compute_attention = _compute_attention if "nhd" in parts else orig["compute_attention"]
    p2.GPT2FeedForward.forward = _mlp_forward if ("gelu" in parts or "modq" in parts) else orig["mlp_forward"]
    if parts:
        log_info(f"Anima fused block: {', '.join(parts)} (WEIGHTFERRY_FUSED_BLOCK).")
    return parts
