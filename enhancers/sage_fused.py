# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 by SageAttention team; modifications Copyright (c) 2026 nyanmi-1828
# Adapted from SageAttention 2.2.0 (thu-ml/SageAttention, Apache-2.0): sageattention/core.py
# (sageattn_qk_int8_pv_fp8_cuda) and sageattention/triton/quant_per_thread.py (quant_key_per_thread_int8_kernel).
# Modified 2026-10-11: the key quantiser subtracts the key mean itself (smooth_k) instead of reading a k - mean copy;
# V's per-channel fp8 (per_channel_fp8: transpose_pad_permute + scale_fuse_quant, compiled) reproduced as one Triton
# kernel from its outputs; the call is limited to the path sageattn() takes on Ada (sm89) for (batch, tokens, heads,
# dim) inputs.
# See THIRD_PARTY_NOTICES.md.
"""
``smk`` part of the fused block (WEIGHTFERRY_FUSED_BLOCK, on by default): SageAttention as ComfyUI calls it on an
RTX 40 GPU, with the key smoothing inside the key quantiser.

On Ada GPUs ComfyUI's attention_sage ends in sageattn_qk_int8_pv_fp8_cuda, which smooths the keys (smooth_k is on: the
"smooth_k": False that ComfyUI passes is not forwarded by sageattn() on sm89): km = k.mean over the tokens, then
``k - km`` is written as a new bf16 tensor and read back by the key quantiser. Here the quantiser reads k and km and
computes bf16(k - km) itself (PyTorch's bf16 subtraction: the float difference rounded to nearest even), so the copy
(one write and one read of k) disappears.

V: SageAttention writes a transposed, padded and permuted bf16 copy of V ((batch, dim, heads, tokens padded to 64),
the tokens of each 16 in the order 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15), then reads it to quantise each channel to
fp8. Here one kernel reads V twice (the channel maxima, then the values) and writes the fp8 values in that order
directly. The arithmetic was found from the compiled kernel's outputs (max|v| over all tokens):
fp8 = round-to-nearest-even(v * (scale_max * rcp.approx(max|v|))) (fast-math division), the stored scale
max|v| * float32(1 / scale_max); an all-zero channel gives scale 0 and NaN values as in SageAttention.

Everything else is SageAttention's own code, called with the arguments sageattn() would use: the query quantiser and
the sm89 attention kernel. Same bits as ComfyUI.
"""

import torch

try:  # module level: Triton resolves the tl.constexpr annotations of @triton.jit functions in the module globals
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = tl = None

_kernel = {}
_FUSE_V = True  # V's fp8 in one kernel (module constant: scratch/verify/ab_resident.py compares with False)


def _get_kernel():
    if _kernel:
        return _kernel["k"]

    @triton.jit
    def _quant_key_smooth(Input, Mean, Output, Scale, L,
                          stride_iz, stride_ih, stride_in, stride_mz, stride_mh,
                          stride_oz, stride_oh, stride_on, stride_sz, stride_sh,
                          C: tl.constexpr, BLK: tl.constexpr):
        # SageAttention's quant_key_per_thread_int8_kernel on bf16(k - km): rows of 8 apart, two interleaved sets
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)
        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)
        m0 = offs_n0[:, None] < L
        m1 = offs_n1[:, None] < L
        base = Input + off_b * stride_iz + off_h * stride_ih
        km = tl.load(Mean + off_b * stride_mz + off_h * stride_mh + offs_k).to(tl.float32)[None, :]
        x0 = tl.load(base + offs_n0[:, None] * stride_in + offs_k[None, :], mask=m0).to(tl.float32)
        x1 = tl.load(base + offs_n1[:, None] * stride_in + offs_k[None, :], mask=m1).to(tl.float32)
        # bf16(k - km) as PyTorch's subtraction writes it (round to nearest even on the bits); rows past L stay 0 as
        # the masked loads of the original kernel give them
        x0 = tl.where(m0, _r16(x0 - km), 0.0)
        x1 = tl.where(m1, _r16(x1 - km), 0.0)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        x0_int8 = x0_int8.to(tl.int8)
        x1_int8 = x1_int8.to(tl.int8)
        out = Output + off_b * stride_oz + off_h * stride_oh
        tl.store(out + offs_n0[:, None] * stride_on + offs_k[None, :], x0_int8, mask=m0)
        tl.store(out + offs_n1[:, None] * stride_on + offs_k[None, :], x1_int8, mask=m1)
        tl.store(Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld, scale)

    @triton.jit
    def _v_fp8(V, Out, Scale, L, P, H, smax, inv_smax, stride_vb, stride_vn, stride_vh,
               C: tl.constexpr, CB: tl.constexpr, TB: tl.constexpr):
        # one program: CB channels of one (batch, head). Pass 1 the channel maxima over the L tokens, pass 2 the fp8
        # values written at positions j of the permuted copy (token (j // 16) * 16 + perm16[j % 16], 0 past L)
        off_b = tl.program_id(2)
        off_h = tl.program_id(1)
        cs = tl.program_id(0) * CB + tl.arange(0, CB)
        base = V + off_b * stride_vb + off_h * stride_vh
        amax = tl.zeros((CB,), dtype=tl.float32)
        for n0 in range(0, L, TB):
            n = n0 + tl.arange(0, TB)
            x = tl.load(base + n[:, None] * stride_vn + cs[None, :], mask=n[:, None] < L, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
        tl.store(Scale + (off_b * H + off_h) * C + cs, amax * inv_smax)  # max * float32(1 / scale_max)
        rcp = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [amax], dtype=tl.float32,
                                        is_pure=True, pack=1)
        inv = smax * rcp
        # out[b, c, h, j]: channel stride H * P, position stride 1
        obase = Out + off_b * C * H * P + off_h * P
        for j0 in range(0, P, TB):
            j = j0 + tl.arange(0, TB)
            r = j % 16
            # perm16 = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]: pair r // 2 -> 0 8 2 10 4 12 6 14
            pr = (r // 2)
            src = (j - r) + (pr // 2) * 2 + (pr % 2) * 8 + (r % 2)
            x = tl.load(base + src[:, None] * stride_vn + cs[None, :], mask=src[:, None] < L,
                        other=0.0).to(tl.float32)
            q = (x * inv[None, :]).to(tl.float8e4nv)
            tl.store(obase + cs[None, :] * (H * P) + j[:, None], q, mask=j[:, None] < P)

    _kernel["k"] = _quant_key_smooth
    _kernel["v"] = _v_fp8
    return _quant_key_smooth


def v_fp8_nhd(v, scale_max):
    """per_channel_fp8(v, tensor_layout="NHD", scale_max=scale_max, smooth_v=False)[:2] in one kernel."""
    _get_kernel()
    b, L, h, d = v.shape
    P = (L + 63) // 64 * 64
    out = torch.empty((b, d, h, P), dtype=torch.float8_e4m3fn, device=v.device)
    scale = torch.empty((b, h, d), dtype=torch.float32, device=v.device)
    CB = 32
    inv_smax = float(torch.tensor(1.0 / scale_max, dtype=torch.float32))  # the float32 nearest to 1 / scale_max
    _kernel["v"][(d // CB, h, b)](v, out, scale, L, P, h, float(scale_max), inv_smax, v.stride(0), v.stride(1),
                                  v.stride(2), C=d, CB=CB, TB=64, num_warps=4)
    return out, scale


if triton is not None:
    @triton.jit
    def _r16(x):
        u = x.to(tl.uint32, bitcast=True)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.to(tl.float32, bitcast=True)


def available(q):
    """The path sageattn() takes for this query, as (pv_accum_dtype,) -- or None when it is not the sm89 FP8 P*V
    kernel this module reproduces (other GPUs, ComfyUI's sageattn replaced, e.g. by WEIGHTFERRY_SAGE_PV=fp16)."""
    if triton is None:
        return None
    try:
        import comfy.ldm.modules.attention as attn_mod
        import sageattention
        import sageattention.core as core
    except Exception:  # noqa: BLE001
        return None
    if getattr(attn_mod, "sageattn", None) is not sageattention.sageattn or not getattr(core, "SM89_ENABLED", False):
        return None
    archs = core._cuda_archs
    if q.device.index is None or q.device.index >= len(archs) or archs[q.device.index] != "sm89":
        return None
    return "fp32+fp32" if core.get_cuda_version() < (12, 8) else "fp32+fp16"


def sage_nhd(q, k, v, pv_accum_dtype):
    """sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None,
    pv_accum_dtype=pv_accum_dtype, smooth_k=True) for bf16 / fp16 (batch, tokens, heads, 64 or 128) inputs with the
    key smoothing inside the key quantiser. None when the shapes are not that case."""
    import sageattention.core as core
    from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel
    b, qo_len, h_qo, head_dim = q.shape
    _, kv_len, h_kv, _ = k.shape
    if head_dim not in (64, 128) or h_qo != h_kv or k.shape[-1] != head_dim or v.shape != k.shape:
        return None
    sm_scale = head_dim ** -0.5
    km = k.mean(dim=1, keepdim=True)  # as Sage: (b, 1, h, d) in the input dtype
    WARPQ, BLKQ, BLKK, WARPK = 32, 128, 64, 64
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device,
                          dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device,
                          dtype=torch.float32)
    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len, q.stride(0), q.stride(2), q.stride(1),
        q_int8.stride(0), q_int8.stride(2), q_int8.stride(1), q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ)
    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    _get_kernel()[grid](
        k, km, k_int8, k_scale, kv_len, k.stride(0), k.stride(2), k.stride(1), km.stride(0), km.stride(2),
        k_int8.stride(0), k_int8.stride(2), k_int8.stride(1), k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK)
    o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
    scale_max = 2.25 if pv_accum_dtype == "fp32+fp16" else 448.0
    if _FUSE_V and v.stride(-1) == 1 and v.stride(3) == 1:
        v_fp8, v_scale = v_fp8_nhd(v, scale_max)
    else:
        v_fp8, v_scale, _vm = core.per_channel_fp8(v, tensor_layout="NHD", scale_max=scale_max, smooth_v=False)
    if pv_accum_dtype == "fp32+fp16":
        core._qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 3, sm_scale, 0)
    else:
        core._qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 0, 0, 3, sm_scale, 0)
    return o
