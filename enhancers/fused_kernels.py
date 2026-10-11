# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Parts re-implement ComfyUI comfy/ldm/cosmos/predict2.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""Triton kernels of the fused block (enhancers/fused_block.py, which describes each part): compiled on first use by
_get_kernels(). ``triton`` / ``tl`` stay module globals here: Triton resolves the tl.constexpr annotations of the
@triton.jit helpers in the globals of the module that defines them."""

try:  # module level: Triton resolves the tl.constexpr annotations of @triton.jit helpers in the module globals
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001 -- no Triton: the parts that need it stay off
    triton = tl = None

# the operation _quant_row applies before the rotation (its PRE constexpr; the kernel compares these literal values)
PRE_NONE = 0   # none: kitchen's quantiser on the input as it is
PRE_GELU = 1   # PyTorch's GELU of the bf16 input, looked up in a 65536-entry table
PRE_MOD = 2    # the scale-and-shift bf16(bf16(x * (1 + scale)) + shift)
PRE_LNMOD = 3  # LayerNorm (no affine), then the scale-and-shift

_kernels = {}


def _get_kernels():
    if _kernels:
        return _kernels

    @triton.jit
    def _r16(x):
        # float32 -> the nearest bfloat16 value (ties to even), still as float32; finite inputs. Rounded on the bits
        # because Triton folds a float -> bf16 -> float round trip away.
        u = x.to(tl.uint32, bitcast=True)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.to(tl.float32, bitcast=True)

    @triton.jit
    def _mod(a_ptr, s_ptr, h_ptr, o_ptr, n_cols, rows_per_group, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        grp = row // rows_per_group
        cols = tl.arange(0, BLOCK)
        m = cols < n_cols
        a = tl.load(a_ptr + row * n_cols + cols, mask=m).to(tl.float32)
        s = tl.load(s_ptr + grp * n_cols + cols, mask=m).to(tl.float32)
        h = tl.load(h_ptr + grp * n_cols + cols, mask=m).to(tl.float32)
        # PyTorch: bf16 * bf16 computed in float and rounded to bf16 (nearest even), then + bf16 likewise
        t = _r16(a * s)
        tl.store(o_ptr + row * n_cols + cols, _r16(t + h).to(tl.bfloat16), mask=m)

    @triton.jit
    def _h4(x, G: tl.constexpr, S: tl.constexpr):
        # each 256-group of the row tile [G, 256]: (a, b, c, d) at stride S -> (a, b, c, d) @ H4,
        # H4 = [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]
        y = tl.reshape(x, (G, 256 // (4 * S), 4, S))
        y = tl.permute(y, (0, 1, 3, 2))
        y = tl.reshape(y, (G, 256 // (4 * S), S, 2, 2))
        e0, e1 = tl.split(y)
        a, c = tl.split(e0)
        b, d = tl.split(e1)
        y0 = a + b + c - d
        y1 = a + b - c + d
        y2 = a - b + c + d
        y3 = -a + b + c + d
        y = tl.join(tl.join(y0, y2), tl.join(y1, y3))
        y = tl.reshape(y, (G, 256 // (4 * S), S, 4))
        y = tl.permute(y, (0, 1, 3, 2))
        return tl.reshape(y, (G, 256))

    @triton.jit
    def _wcombine(m_own, s_own, c_own, m_oth, s_oth, c_oth):
        # PyTorch cuWelfordCombine(dataB = own, dataA = other), with nvcc's contractions
        delta = m_own - m_oth
        count = c_oth + c_own
        coef = tl.math.div_rn(1.0, count)
        n_a = c_oth * coef
        n_b = c_own * coef
        mean = tl.math.fma(n_a, m_oth, n_b * m_own)
        sig = tl.math.fma(delta * delta * c_oth, n_b, s_oth + s_own)
        return mean, sig, count

    @triton.jit
    def _whalf(m, s, c, A: tl.constexpr, N: tl.constexpr):
        # one shuffle-down step over the last axis of [A, N]: lane t (t < N / 2) merges lane t + N / 2
        m = tl.permute(tl.reshape(m, (A, 2, N // 2)), (0, 2, 1))
        s = tl.permute(tl.reshape(s, (A, 2, N // 2)), (0, 2, 1))
        c = tl.permute(tl.reshape(c, (A, 2, N // 2)), (0, 2, 1))
        m0, m1 = tl.split(m)
        s0, s1 = tl.split(s)
        c0, c1 = tl.split(c)
        return _wcombine(m0, s0, c0, m1, s1, c1)

    @triton.jit
    def _ln_stats(x_ptr, row, K: tl.constexpr):
        # PyTorch's vectorized layer-norm statistics of one row: thread t of 128 reads the 4-value vectors t, t + 128,
        # ... and updates (mean, sigma2, count) per value; warps of 32 lanes reduce by shuffle-down, then the 4 warps
        # merge (warps 2, 3 into 0, 1, then 1 into 0). Returns mean and sigma2 / K
        t = tl.arange(0, 128)
        mean = tl.zeros((128,), dtype=tl.float32)
        sig = tl.zeros((128,), dtype=tl.float32)
        cnt = tl.zeros((128,), dtype=tl.float32)
        for j in tl.static_range(K // 512):
            for ii in tl.static_range(4):
                v = tl.load(x_ptr + row * K + (t + j * 128) * 4 + ii).to(tl.float32)
                delta = v - mean
                cnt = cnt + 1.0
                mean = tl.math.fma(delta, tl.math.div_rn(1.0, cnt), mean)
                sig = tl.math.fma(delta, v - mean, sig)
        m = tl.reshape(mean, (4, 32))
        s = tl.reshape(sig, (4, 32))
        c = tl.reshape(cnt, (4, 32))
        m, s, c = _whalf(m, s, c, 4, 32)
        m, s, c = _whalf(m, s, c, 4, 16)
        m, s, c = _whalf(m, s, c, 4, 8)
        m, s, c = _whalf(m, s, c, 4, 4)
        m, s, c = _whalf(m, s, c, 4, 2)   # [4, 1]: each warp's lane 0
        m = tl.reshape(m, (1, 4))
        s = tl.reshape(s, (1, 4))
        c = tl.reshape(c, (1, 4))
        m, s, c = _whalf(m, s, c, 1, 4)   # warps 2, 3 into 0, 1
        m, s, c = _whalf(m, s, c, 1, 2)   # warp 1 into 0
        return tl.sum(tl.reshape(m, (1,)), axis=0), tl.sum(tl.reshape(s, (1,)), axis=0) / K

    @triton.jit
    def _quant_row(x_ptr, s_ptr, h_ptr, q_ptr, sc_ptr, rows_per_group, eps, K: tl.constexpr, PRE: tl.constexpr):
        # one row per program, the whole row in registers: optional operation, rotation, amax, quantisation
        G: tl.constexpr = K // 256
        row = tl.program_id(0).to(tl.int64)
        grp = row // rows_per_group
        offs = tl.reshape(tl.arange(0, K), (G, 256))
        x = tl.load(x_ptr + row * K + offs).to(tl.float32)
        if PRE == 3:  # PRE_LNMOD: LayerNorm (no affine), rounded to bf16 as PyTorch writes it, then modulate
            mu, var = _ln_stats(x_ptr, row, K)
            rstd = tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [var + eps], dtype=tl.float32,
                                             is_pure=True, pack=1)
            x = _r16((x - mu) * rstd)
            s = tl.load(s_ptr + grp * K + offs).to(tl.float32)
            h = tl.load(h_ptr + grp * K + offs).to(tl.float32)
            x = _r16(_r16(x * s) + h)
        if PRE == 1:  # PRE_GELU: PyTorch's own GELU of this bf16 value, from the 65536-entry table (s_ptr)
            bits = tl.load((x_ptr + row * K + offs).to(tl.pointer_type(tl.int16))).to(tl.int32) & 0xFFFF
            x = tl.load(s_ptr + bits).to(tl.float32)
        elif PRE == 2:  # PRE_MOD: modulate: bf16(bf16(x * (1 + scale)) + shift)
            s = tl.load(s_ptr + grp * K + offs).to(tl.float32)
            h = tl.load(h_ptr + grp * K + offs).to(tl.float32)
            x = _r16(_r16(x * s) + h)
        x = _h4(x, G, 1)
        x = _h4(x, G, 4)
        x = _h4(x, G, 16)
        x = _h4(x, G, 64)
        x = x * 0.0625
        # the division as compiled in comfy_kitchen (not IEEE div_rn: Triton's default division matched all values)
        # kitchen clamps the amax to bf16's largest finite value and the scale to at least 1e-30 (a zero row: scale
        # 1e-30, not 0; the int8 values are 0 either way)
        amax = tl.minimum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 3.38953139e38)
        scale = tl.maximum(amax / 127.0, 1.0e-30)
        tl.store(sc_ptr + row, scale)
        sb = _r16(scale + tl.zeros((G, 256), dtype=tl.float32))
        v = tl.extra.cuda.libdevice.rint(_r16(_r16(x) / sb))
        v = tl.minimum(tl.maximum(v, -128.0), 127.0)
        tl.store(q_ptr + row * K + offs, v.to(tl.int8))

    @triton.jit
    def _half(s, ROWS: tl.constexpr, N: tl.constexpr):
        # one step of a warp xor-shuffle reduction: lane t += lane t ^ (N / 2), kept for lanes t < N / 2
        y = tl.reshape(s, (ROWS, 2, N // 2))
        y = tl.permute(y, (0, 2, 1))
        lo, hi = tl.split(y)
        return lo + hi

    @triton.jit
    def _rsqrt_approx(v):
        return tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True,
                                         pack=1)

    @triton.jit
    def _rms128(x_ptr, w_ptr, o_ptr, n_rows, eps, ROWS: tl.constexpr, RS: tl.constexpr):
        # PyTorch's RMSNorm (vectorized layer-norm kernel, rms_norm) on rows of 128: each of 32 threads sums the
        # squares of 4 consecutive values in order, the warp adds them by xor shuffles 16, 8, 4, 2, 1, then
        # rsqrt(sum / 128 + eps), y = (x * rstd) * w rounded to bf16
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        rmask = r < n_rows
        cols = tl.arange(0, 128)
        offs = r[:, None].to(tl.int64) * 128 + cols[None, :]
        x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0).to(tl.float32)
        q = tl.reshape(x * x, (ROWS, 32, 2, 2))   # [rows, thread, j1, j0], element j = 2 * j1 + j0
        e0, e1 = tl.split(q)
        a, c = tl.split(e0)
        b, d = tl.split(e1)
        s = ((a + b) + c) + d
        s = _half(s, ROWS, 32)
        s = _half(s, ROWS, 16)
        s = _half(s, ROWS, 8)
        s = _half(s, ROWS, 4)
        s = _half(s, ROWS, 2)
        tot = tl.reshape(s, (ROWS,))
        v = tot / 128.0 + eps
        if RS == 0:
            rstd = _rsqrt_approx(v)
        else:
            rstd = tl.math.rsqrt(v)
        w = tl.load(w_ptr + cols).to(tl.float32)
        y = _r16((x * rstd[:, None]) * w[None, :])
        tl.store(o_ptr + offs, y.to(tl.bfloat16), mask=rmask[:, None])

    @triton.jit
    def _lnmod(x_ptr, s_ptr, h_ptr, o_ptr, rows_per_group, eps, K: tl.constexpr):
        # modulate(LayerNorm(x)) of one row, written out (bf16): the LayerNorm of the quantiser's PRE 3
        row = tl.program_id(0).to(tl.int64)
        grp = row // rows_per_group
        offs = tl.arange(0, K)
        x = tl.load(x_ptr + row * K + offs).to(tl.float32)
        mu, var = _ln_stats(x_ptr, row, K)
        rstd = tl.inline_asm_elementwise("rsqrt.approx.f32 $0, $1;", "=r,r", [var + eps], dtype=tl.float32,
                                         is_pure=True, pack=1)
        x = _r16((x - mu) * rstd)
        s = tl.load(s_ptr + grp * K + offs).to(tl.float32)
        h = tl.load(h_ptr + grp * K + offs).to(tl.float32)
        tl.store(o_ptr + row * K + offs, _r16(_r16(x * s) + h).to(tl.bfloat16))

    @triton.jit
    def _tay(h_ptr, f0_ptr, f1_ptr, f2_ptr, o_ptr, n, c0, c1, c2, d0, d1, d2, NF: tl.constexpr, BLOCK: tl.constexpr):
        # TaylorSeer._taylor_eval then hidden + prediction: pred = zeros; pred += (c_i * f_i) * d_i for i < NF, every
        # product and sum rounded to bf16 as PyTorch's bf16 kernels do (float math, round to nearest even)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        pred = tl.zeros((BLOCK,), dtype=tl.float32)
        f = tl.load(f0_ptr + offs, mask=m, other=0.0).to(tl.float32)
        pred = _r16(pred + _r16(_r16(f * c0) * d0))
        if NF > 1:
            f = tl.load(f1_ptr + offs, mask=m, other=0.0).to(tl.float32)
            pred = _r16(pred + _r16(_r16(f * c1) * d1))
        if NF > 2:
            f = tl.load(f2_ptr + offs, mask=m, other=0.0).to(tl.float32)
            pred = _r16(pred + _r16(_r16(f * c2) * d2))
        h = tl.load(h_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(o_ptr + offs, _r16(h + pred).to(tl.bfloat16), mask=m)

    _kernels.update(mod=_mod, quant_row=_quant_row, rms128=_rms128, lnmod=_lnmod, tay=_tay)
    return _kernels
