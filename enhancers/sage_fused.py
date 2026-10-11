# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 by SageAttention team; modifications Copyright (c) 2026 nyanmi-1828
# Adapted from SageAttention 2.2.0 (thu-ml/SageAttention, Apache-2.0): sageattention/core.py
# (sageattn_qk_int8_pv_fp8_cuda) and sageattention/triton/quant_per_thread.py (quant_key_per_thread_int8_kernel),
# and from comfy-kitchen 0.2.36 (Comfy-Org/comfy-kitchen, commit 888b13e; Copyright (c) 2025 NVIDIA CORPORATION &
# AFFILIATES, Apache-2.0): comfy_kitchen/backends/cuda/ops/rms_rope.cu and rope_device.cuh (rope_kernel's RMSNorm +
# split-half RoPE, re-written in Triton).
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

    @triton.jit
    def _quant_query_rms(Input, W, Output, Scale, L, eps,
                         stride_iz, stride_ih, stride_in, stride_oz, stride_oh, stride_on, stride_sz, stride_sh,
                         C: tl.constexpr, BLK: tl.constexpr):
        # SageAttention's quant_query_per_thread_int8_kernel on RMSNorm(q) computed here (the cross-attention's q
        # normalisation, the fused block's rms128 order: 32 lanes x 4 consecutive squares ((a + b) + c) + d, an
        # xor-shuffle sum 16 .. 1, rsqrt.approx(sum / 128 + eps), bf16((x * rstd) * w)) -- never written as bf16
        R: tl.constexpr = BLK // 8
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)
        offs_n = off_blk * BLK + tl.arange(0, R) * 8 + off_tld
        offs_k = tl.arange(0, C)
        m = offs_n[:, None] < L
        x = tl.load(Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :],
                    mask=m, other=0.0).to(tl.float32)
        sq = tl.reshape(x * x, (R, 32, 2, 2))  # [row, lane, j1, j0], value 2 * j1 + j0 of the lane's 4
        e0, e1 = tl.split(sq)
        a, c = tl.split(e0)
        b, d = tl.split(e1)
        s = ((a + b) + c) + d
        s = _half(s, R, 32)
        s = _half(s, R, 16)
        s = _half(s, R, 8)
        s = _half(s, R, 4)
        s = _half(s, R, 2)
        tot = tl.reshape(s, (R,))
        rstd = tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [tot / 128.0 + eps], dtype=tl.float32,
                                         is_pure=True, pack=1)
        w = tl.load(W + offs_k).to(tl.float32)
        y = tl.where(m, _r16((x * rstd[:, None]) * w[None, :]), 0.0)
        scale = tl.max(tl.abs(y)) / 127. + 0.0000001
        q = y / scale
        q += 0.5 * tl.where(q >= 0, 1, -1)
        q = q.to(tl.int8)
        tl.store(Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :], q,
                 mask=m)
        tl.store(Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld, scale)

    @triton.jit
    def _rope_rows(X, F, W, Out, n_rows, H, L, eps, f_s1, f_s3, f_s4, f_s5, ROWS: tl.constexpr):
        # comfy_kitchen's rms_rope_split_half for rows of 128 of a contiguous (batch, tokens, heads, 128) tensor,
        # written as bf16 (the keys; the queries when another consumer needs them)
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        m = r < n_rows
        tok = (r // H) % L
        cols = tl.arange(0, 128)
        offs = r[:, None].to(tl.int64) * 128 + cols[None, :]
        x = tl.load(X + offs, mask=m[:, None], other=0.0).to(tl.float32)
        fb = F + tok[:, None].to(tl.int64) * f_s1 + tl.arange(0, 64)[None, :] * f_s3
        f00 = tl.load(fb, mask=m[:, None], other=0.0)
        f01 = tl.load(fb + f_s5, mask=m[:, None], other=0.0)
        f10 = tl.load(fb + f_s4, mask=m[:, None], other=0.0)
        f11 = tl.load(fb + f_s4 + f_s5, mask=m[:, None], other=0.0)
        w = tl.load(W + cols).to(tl.float32)
        y = _rms_rope128(x, w, f00, f01, f10, f11, eps, ROWS)
        tl.store(Out + offs, y.to(tl.bfloat16), mask=m[:, None])

    @triton.jit
    def _quant_query_rope(Input, F, W, Output, Scale, L, eps, f_s1, f_s3, f_s4, f_s5,
                          stride_iz, stride_ih, stride_in, stride_oz, stride_oh, stride_on, stride_sz, stride_sh,
                          C: tl.constexpr, BLK: tl.constexpr):
        # SageAttention's quant_query_per_thread_int8_kernel on the self-attention's q after comfy_kitchen's RMSNorm
        # + split-half RoPE computed here -- the rotated q is never written as bf16
        R: tl.constexpr = BLK // 8
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)
        offs_n = off_blk * BLK + tl.arange(0, R) * 8 + off_tld
        offs_k = tl.arange(0, C)
        m = offs_n[:, None] < L
        x = tl.load(Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :],
                    mask=m, other=0.0).to(tl.float32)
        fb = F + offs_n[:, None].to(tl.int64) * f_s1 + tl.arange(0, 64)[None, :] * f_s3
        f00 = tl.load(fb, mask=m, other=0.0)
        f01 = tl.load(fb + f_s5, mask=m, other=0.0)
        f10 = tl.load(fb + f_s4, mask=m, other=0.0)
        f11 = tl.load(fb + f_s4 + f_s5, mask=m, other=0.0)
        w = tl.load(W + offs_k).to(tl.float32)
        y = tl.where(m, _rms_rope128(x, w, f00, f01, f10, f11, eps, R), 0.0)
        scale = tl.max(tl.abs(y)) / 127. + 0.0000001
        q = y / scale
        q += 0.5 * tl.where(q >= 0, 1, -1)
        q = q.to(tl.int8)
        tl.store(Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :], q,
                 mask=m)
        tl.store(Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld, scale)

    _kernel["k"] = _quant_key_smooth
    _kernel["v"] = _v_fp8
    _kernel["q_rms"] = _quant_query_rms
    _kernel["rope_rows"] = _rope_rows
    _kernel["q_rope"] = _quant_query_rope
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
    def _half(s, ROWS: tl.constexpr, N: tl.constexpr):
        # one step of a warp xor-shuffle sum: lane t += lane t ^ (N / 2), kept for lanes t < N / 2
        y = tl.permute(tl.reshape(s, (ROWS, 2, N // 2)), (0, 2, 1))
        lo, hi = tl.split(y)
        return lo + hi

    @triton.jit
    def _rms_rope128(x, w, f00, f01, f10, f11, eps, R: tl.constexpr):
        # comfy_kitchen's rope_kernel<bf16, float, ..., HasRms, SplitHalf, ContigHead> for R rows of 128: lane t of
        # 32 sums the squares of values t, t + 32, t + 64, t + 96 with fmaf from 0, __shfl_down 16 .. 1,
        # rrms = rsqrtf(sum / 128 + eps), bf16((x * rrms) * w), then the split-half rotation of the pairs
        # (d, d + 64) as nvcc contracts it: y0 = fma(f00, x0, f01 * x1), y1 = fma(f10, x0, f11 * x1), rounded to bf16
        xs = tl.reshape(tl.permute(tl.reshape(x, (R, 4, 32)), (0, 2, 1)), (R, 32, 2, 2))  # [row, lane, j1, j0]
        e0, e1 = tl.split(xs)
        v0, v2 = tl.split(e0)
        v1, v3 = tl.split(e1)
        acc = v0 * v0
        acc = tl.math.fma(v1, v1, acc)
        acc = tl.math.fma(v2, v2, acc)
        acc = tl.math.fma(v3, v3, acc)
        acc = _half(acc, R, 32)
        acc = _half(acc, R, 16)
        acc = _half(acc, R, 8)
        acc = _half(acc, R, 4)
        acc = _half(acc, R, 2)
        tot = tl.reshape(acc, (R,))
        rr = tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [tot / 128.0 + eps], dtype=tl.float32,
                                       is_pure=True, pack=1)
        n = _r16((x * rr[:, None]) * w[None, :])
        lo, hi = tl.split(tl.permute(tl.reshape(n, (R, 2, 64)), (0, 2, 1)))  # dims d and d + 64
        y0 = tl.math.fma(f00, lo, f01 * hi)
        y1 = tl.math.fma(f10, lo, f11 * hi)
        return _r16(tl.reshape(tl.permute(tl.join(y0, y1), (0, 2, 1)), (R, 128)))

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


def rope_ok(q, k, f, q_scale, k_scale):
    """The case the Triton reproduction of comfy_kitchen's rms_rope_split_half covers: contiguous bf16
    (batch, tokens, heads, 128) q and k of the same shape, float32 rotations (1, tokens, 1, 64, 2, 2), scales of 128."""
    if triton is None or f is None:
        return False
    b, L, h, d = q.shape if q.dim() == 4 else (0, 0, 0, 0)
    return (d == 128 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and k.shape == q.shape
            and q.is_contiguous() and k.is_contiguous() and q.is_cuda and f.dtype == torch.float32 and f.dim() == 6
            and tuple(f.shape) == (1, L, 1, 64, 2, 2) and f.device == q.device
            and all(s is not None and s.dim() == 1 and s.shape[0] == 128 and s.is_contiguous() and s.device == q.device
                    and s.dtype in (torch.bfloat16, torch.float16, torch.float32) for s in (q_scale, k_scale)))


def rms_rope(x, f, w, eps):
    """comfy_kitchen's rms_rope_split_half for one of q / k (rope_ok's case): the same bits, bf16 out."""
    _get_kernel()
    b, L, h, d = x.shape
    out = torch.empty_like(x)
    n = b * L * h
    rows = 32
    _kernel["rope_rows"][(triton.cdiv(n, rows),)](x, f, w, out, n, h, L, float(eps), f.stride(1), f.stride(3),
                                                  f.stride(4), f.stride(5), ROWS=rows, num_warps=4)
    return out


def sage_nhd(q, k, v, pv_accum_dtype, q_norm=None, q_rope=None):
    """sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=None,
    pv_accum_dtype=pv_accum_dtype, smooth_k=True) for bf16 / fp16 (batch, tokens, heads, 64 or 128) inputs with the
    key smoothing inside the key quantiser. q_norm = (weight, eps): q is the input of an RMSNorm over its 128 values
    (bf16 weight), applied by the query quantiser; q_rope = (rotations, weight, eps): q is the input of comfy_kitchen's
    RMSNorm + split-half RoPE (rope_ok's case), applied by the query quantiser. None when the shapes are not that
    case."""
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
    if q_rope is not None:
        f, w, eps = q_rope
        _get_kernel()
        _kernel["q_rope"][grid](
            q, f, w, q_int8, q_scale, qo_len, float(eps), f.stride(1), f.stride(3), f.stride(4), f.stride(5),
            q.stride(0), q.stride(2), q.stride(1), q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
            q_scale.stride(0), q_scale.stride(1), C=head_dim, BLK=WARPQ)
    elif q_norm is not None:
        if head_dim != 128 or q.dtype != torch.bfloat16:
            return None
        _get_kernel()
        _kernel["q_rms"][grid](
            q, q_norm[0], q_int8, q_scale, qo_len, float(q_norm[1]), q.stride(0), q.stride(2), q.stride(1),
            q_int8.stride(0), q_int8.stride(2), q_int8.stride(1), q_scale.stride(0), q_scale.stride(1),
            C=head_dim, BLK=WARPQ)
    else:
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
