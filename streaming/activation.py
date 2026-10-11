# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Parts re-implement ComfyUI comfy/ldm/cosmos/predict2.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
Activation streaming and row chunking: MLP / self-attention / whole Cosmos blocks computed by token rows,
activation host buffers, input shape tracking.

Provenance: the row-chunked Cosmos block / self-attention (_cosmos_block_rows, _cosmos_self_attention_rows,
_cosmos_block_mods) re-implement ComfyUI's Block.forward / Attention of comfy/ldm/cosmos/predict2.py (GPL-3.0) by
token rows, with the same operations in the same order (that is what makes them bit-exact).
"""

import torch

from ..utils.logging import log_info
from ..core import tiered_store
from ..core import vram_monitor
from .common import _ACTIVE_ATTR, _STATE_ATTR, _is_quant, _raw
from .sage import _sage_fp16_for_per_item
from ..core import env as _env


_MLP_NAMES = ("mlp", "ff", "feed_forward", "ffn", "img_mlp", "txt_mlp")
_TOKENWISE_LEAF_NAMES = ("GELU", "SiLU", "ReLU", "Tanh", "Sigmoid", "Dropout", "Identity")


def _tokenwise_mlp(module):
    """True if the module is only linear layers and element-wise activations (every token computed alone)."""
    for m in module.modules():
        if m is module or isinstance(m, (torch.nn.Sequential, torch.nn.ModuleList)):
            continue
        # ComfyUI's mixed-precision (scaled fp8) Linear is not a torch.nn.Linear subclass: match it by name
        if (isinstance(m, torch.nn.Linear) or "Linear" in type(m).__name__
                or any(n in type(m).__name__ for n in _TOKENWISE_LEAF_NAMES)):
            continue
        return False
    return True


def _mlp_weights_quantized(module):
    """True if every 2-D weight of the module is a QuantizedTensor (scaled fp8)."""
    ws = [p for p in module.parameters() if _raw(p.data).dim() == 2]
    return bool(ws) and all(_is_quant(p.data) for p in ws)


def _install_mlp_chunking(dit, blocks, owner, rows, any_dtype=False):
    """Compute the blocks' feed-forward (MLP) modules in row (token) chunks of ``rows`` while this owner streams.

    The MLP's hidden activation (Cosmos-Predict2 14B at 896x1472: 5152 tokens x 20480 x bf16, twice) is the largest
    intermediate of a block (402 MB of its 553 MB peak, measured). A token-wise MLP gives bit-identical
    results in row chunks: ComfyUI's scaled fp8 Linear quantizes its input with scale 1 element by element, and the
    GEMMs of one Cosmos block are bit-identical for chunks of 128..2576 rows.
    bf16 weights are NOT safe: the bf16 GEMMs are bit-identical only for some chunk sizes
    (cuBLAS picks other kernels / split-K for other row counts), so only MLPs whose weights are all scaled fp8 are
    chunked unless ``any_dtype``. The output is preallocated and filled chunk by chunk. During the chunk loop the
    chunk streaming state holds back slot hand-overs (the MLP's layers run once per chunk). Returns the modules."""
    installed = []
    for blk in blocks:
        for name in _MLP_NAMES:
            mlp = getattr(blk, name, None)
            if not isinstance(mlp, torch.nn.Module) or "_weightferry_mlp_orig" in mlp.__dict__ or not _tokenwise_mlp(mlp):
                continue
            if not any_dtype and not _mlp_weights_quantized(mlp):
                continue
            orig = mlp.forward

            def chunked(x, *args, _orig=orig, **kwargs):
                if (getattr(dit, _ACTIVE_ATTR, None) is not owner or args or kwargs
                        or not isinstance(x, torch.Tensor) or x.dim() < 2):
                    return _orig(x, *args, **kwargs)
                flat = x.reshape(-1, x.shape[-1])
                n = flat.shape[0]
                if n <= rows:
                    return _orig(x)
                state = getattr(dit, _STATE_ATTR, None)
                hold = getattr(state, "hold_completions", None)
                if hold is not None:
                    hold(1)
                try:
                    out = None
                    for i in range(0, n, rows):
                        part = _orig(flat[i:i + rows])
                        if out is None:
                            out = torch.empty((n, part.shape[-1]), dtype=part.dtype, device=part.device)
                        out[i:i + part.shape[0]].copy_(part)
                        del part
                finally:
                    if hold is not None:
                        hold(-1)
                return out.reshape(*x.shape[:-1], out.shape[-1])

            mlp._weightferry_mlp_orig = orig
            mlp.forward = chunked
            installed.append(mlp)
    return installed


def _uninstall_mlp_chunking(modules):
    for mlp in modules or ():
        mlp.__dict__.pop("forward", None)
        mlp.__dict__.pop("_weightferry_mlp_orig", None)


def _chunkable_self_attention(attn):
    """ComfyUI's Cosmos-Predict2 / Anima self-attention (comfy.ldm.cosmos.predict2.Attention) with scaled fp8
    projections: the only layout _cosmos_self_attention_rows reproduces."""
    need = ("q_proj", "k_proj", "v_proj", "output_proj", "q_norm", "k_norm", "v_norm", "output_dropout", "attn_op",
            "comfy_attention", "n_heads", "head_dim")
    if not getattr(attn, "is_selfattn", False) or not all(hasattr(attn, n) for n in need):
        return False
    if type(attn).__module__ != "comfy.ldm.cosmos.predict2":
        return False
    return all(_is_quant(getattr(attn, n).weight.data) for n in ("q_proj", "k_proj", "v_proj", "output_proj"))


def _cosmos_self_attention_rows(attn, x, rope_emb, transformer_options, rows):
    """ComfyUI's Cosmos-Predict2 self-attention (Attention.forward with context None and RoPE) computed by query rows,
    without the full Q (and without the unnormalized full Q / K next to the normalized ones).

    Same operations as ComfyUI, on row chunks: k_proj / v_proj (scaled fp8 GEMMs, bit-identical in row chunks), the
    fused RMSNorm + split-half RoPE kernel (per token and head; called with the chunk as both of its inputs and the
    matching output kept), then per query chunk q_proj -> RMSNorm + RoPE -> attention over the full K / V ->
    output_proj. The attention with the queries split at multiples of 128 rows is bit-identical.
    ``rows`` must be a multiple of 128. Peak: K, V and the output instead of Q, K, V, Q', K'."""
    import comfy.ops
    import comfy.quant_ops
    from comfy.ldm.cosmos.predict2 import AttentionTensorContainer
    ck = comfy.quant_ops.ck
    B, S, _ = x.shape
    H, D = attn.n_heads, attn.head_dim
    eps = attn.q_norm.eps
    K = V = out = None
    k_scale = k_os = q_scale = q_os = None
    try:
        for i in range(0, S, rows):
            xc = x[:, i:i + rows]
            n = xc.shape[1]
            rc = rope_emb[:, i:i + n]
            kc = attn.k_proj(xc).view(B, n, H, D)
            if k_scale is None:
                k_scale, _, k_os = comfy.ops.cast_bias_weight(attn.k_norm, kc, offloadable=True)
            kc = ck.rms_rope_split_half(kc, kc, rc, k_scale, k_scale, eps)[1]
            if K is None:
                K = torch.empty((B, S, H, D), dtype=kc.dtype, device=kc.device)
            K[:, i:i + n].copy_(kc)
            del kc
            vc = attn.v_norm(attn.v_proj(xc).view(B, n, H, D))
            if V is None:
                V = torch.empty((B, S, H, D), dtype=vc.dtype, device=vc.device)
            V[:, i:i + n].copy_(vc)
            del vc
        for i in range(0, S, rows):
            xc = x[:, i:i + rows]
            n = xc.shape[1]
            rc = rope_emb[:, i:i + n]
            qc = attn.q_proj(xc).view(B, n, H, D)
            if q_scale is None:
                q_scale, _, q_os = comfy.ops.cast_bias_weight(attn.q_norm, qc, offloadable=True)
            qc = ck.rms_rope_split_half(qc, qc, rc, q_scale, q_scale, eps)[0]
            res = attn.attn_op(AttentionTensorContainer(qc), AttentionTensorContainer(K), AttentionTensorContainer(V),
                               transformer_options=transformer_options, preferred_attention=attn.comfy_attention)
            del qc
            oc = attn.output_dropout(attn.output_proj(res))
            del res
            if out is None:
                out = torch.empty((B, S, oc.shape[-1]), dtype=oc.dtype, device=oc.device)
            out[:, i:i + n].copy_(oc)
            del oc
    finally:
        if q_scale is not None:
            comfy.ops.uncast_bias_weight(attn.q_norm, q_scale, None, q_os)
        if k_scale is not None:
            comfy.ops.uncast_bias_weight(attn.k_norm, k_scale, None, k_os)
    return out


def _chunkable_block(blk):
    """ComfyUI's Cosmos-Predict2 / Anima transformer block (comfy.ldm.cosmos.predict2.Block) whose self-attention
    _cosmos_self_attention_rows reproduces."""
    if type(blk).__name__ != "Block" or type(blk).__module__ != "comfy.ldm.cosmos.predict2":
        return False
    need = ("layer_norm_self_attn", "layer_norm_cross_attn", "layer_norm_mlp", "self_attn", "cross_attn", "mlp",
            "adaln_modulation_self_attn", "adaln_modulation_cross_attn", "adaln_modulation_mlp", "use_adaln_lora")
    return all(hasattr(blk, n) for n in need) and _chunkable_self_attention(blk.self_attn)


def _cosmos_block_mods(blk, emb_B_T_D, adaln_lora_B_T_3D):
    """The block's adaLN modulations (shift, scale, gate for self-attention, cross-attention, MLP) as ComfyUI
    computes them: for the whole batch at once (they are bf16 GEMMs, whose results can change with the row count)."""
    ms = (blk.adaln_modulation_self_attn, blk.adaln_modulation_cross_attn, blk.adaln_modulation_mlp)
    if blk.use_adaln_lora:
        return [(m(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1) for m in ms]
    return [m(emb_B_T_D).chunk(3, dim=-1) for m in ms]


def _cosmos_block_rows(blk, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D,
                       extra_per_block_pos_emb, transformer_options, rows, host_out=None, device=None, mods=None):
    """ComfyUI's Cosmos-Predict2 Block.forward computed by token rows (activation chunking).

    Only the self-attention's K and V are built for all tokens (pass 1, from the normalized input chunk by chunk);
    then each chunk of ``rows`` tokens goes through the whole block -- self-attention against the full K / V,
    gated residual, cross-attention, gated residual, MLP, gated residual -- into the preallocated output. No
    full-size normalized input, Q, attention result or MLP activation exists. Same operations as ComfyUI on row
    chunks: element-wise / per-token ops (layer norms, modulation, residual adds), scaled fp8 GEMMs (bit-identical in
    row chunks) and attention with the queries split at multiples of 128 rows (bit-identical).
    The adaLN modulations are computed once as in ComfyUI. Images only (T == 1).
    Host residual (``host_out``, a pinned CPU buffer of the block's output shape): the input may be on the CPU and the
    output is written to host_out, so the residual stream between blocks stays in pinned RAM and only K / V and one
    chunk are on the GPU (copies on the compute stream; values unchanged)."""
    import comfy.ops
    import comfy.quant_ops
    from comfy.ldm.cosmos.predict2 import AttentionTensorContainer
    ck = comfy.quant_ops.ck
    residual_dtype = x_B_T_H_W_D.dtype
    compute_dtype = emb_B_T_D.dtype
    if mods is None:
        mods = _cosmos_block_mods(blk, emb_B_T_D, adaln_lora_B_T_3D)
    # (B, T, D) -> (B, T, 1, 1, D) as in ComfyUI; with T == 1 the same values broadcast over the token rows (B, 1, D)
    (sh_sa, sc_sa, g_sa), (sh_ca, sc_ca, g_ca), (sh_mlp, sc_mlp, g_mlp) = [
        tuple(t.reshape(t.shape[0], 1, t.shape[-1]) for t in m) for m in mods]
    B, T, H, W, D = x_B_T_H_W_D.shape
    S = T * H * W
    x = x_B_T_H_W_D.reshape(B, S, D)
    extra = None if extra_per_block_pos_emb is None else extra_per_block_pos_emb.reshape(B, S, D)
    dev = device if device is not None else emb_B_T_D.device

    def mod_norm(xc, norm, scale, shift):
        return norm(xc) * (1 + scale) + shift

    # The fused block's parts (enhancers/fused_block.py, WEIGHTFERRY_FUSED_BLOCK, bit-exact) on the row chunks too: the
    # scale-and-shift inside the quantiser, k / v sharing one quantisation, SageAttention without the output copy,
    # the gated residuals inside the output projections' GEMMs
    from ..enhancers import fused_block as fb
    fused = fb.rows_active(blk, transformer_options, compute_dtype)

    attn = blk.self_attn
    Hh, Dh = attn.n_heads, attn.head_dim
    eps = attn.q_norm.eps
    K = V = out = None
    # WEIGHTFERRY_ACT_D2H_STREAM=1: the copies of the output rows to the pinned host buffer go on their
    # own stream (the GPU has several copy engines), so they overlap the next rows' compute; the compute stream waits
    # for them at the end of the block, before anything reads the host buffer. Values unchanged
    d2h = _act_d2h_stream(dev) if host_out is not None else None
    compute = torch.cuda.current_stream(dev) if d2h is not None else None
    # Two fixed GPU staging buffers used in turn (keeping each row chunk alive with record_stream instead would let
    # the CPU run ahead of the GPU while PyTorch cannot reuse that memory, so it allocates more and VRAM overflows).
    # Filled on the compute stream (device copy), sent to the host on the D2H stream; a buffer is refilled only after
    # its previous host copy (GPU-side wait)
    stage = stage_free = None
    if d2h is not None:
        stage, stage_free = _act_d2h_stage(dev, B, min(rows, S), D, x_B_T_H_W_D.dtype)
    k_scale = k_os = q_scale = q_os = None
    try:
        # pass 1: K and V of all tokens
        for i, xc in _row_chunks(x, extra, S, rows, dev):
            n = xc.shape[1]
            if fused:
                kl, vl = fb.rows_project(blk, (attn.k_proj, attn.v_proj), (blk.layer_norm_self_attn, xc), sc_sa,
                                         sh_sa, compute_dtype, transformer_options)
            else:
                nc = mod_norm(xc, blk.layer_norm_self_attn, sc_sa, sh_sa).to(compute_dtype)
                kl, vl = attn.k_proj(nc), attn.v_proj(nc)
                del nc
            rc = rope_emb_L_1_1_D[:, i:i + n]
            kc = kl.view(B, n, Hh, Dh)
            del kl
            if k_scale is None:
                k_scale, _, k_os = comfy.ops.cast_bias_weight(attn.k_norm, kc, offloadable=True)
            kc = ck.rms_rope_split_half(kc, kc, rc, k_scale, k_scale, eps)[1]
            if K is None:
                K = torch.empty((B, S, Hh, Dh), dtype=kc.dtype, device=kc.device)
            K[:, i:i + n].copy_(kc)
            del kc
            vc = attn.v_norm(vl.view(B, n, Hh, Dh))
            del vl
            if V is None:
                V = torch.empty((B, S, Hh, Dh), dtype=vc.dtype, device=vc.device)
            V[:, i:i + n].copy_(vc)
            del vc
        # pass 2: every chunk through the whole block
        kv_cache = {}  # SageAttention's key / value preparation for this block's K / V (fused_block.rows_attention)
        for i, xc in _row_chunks(x, extra, S, rows, dev):
            n = xc.shape[1]
            if fused:
                (ql,) = fb.rows_project(blk, (attn.q_proj,), (blk.layer_norm_self_attn, xc), sc_sa, sh_sa,
                                        compute_dtype, transformer_options)
            else:
                nc = mod_norm(xc, blk.layer_norm_self_attn, sc_sa, sh_sa).to(compute_dtype)
                ql = attn.q_proj(nc)
                del nc
            rc = rope_emb_L_1_1_D[:, i:i + n]
            qc = ql.view(B, n, Hh, Dh)
            del ql
            if q_scale is None:
                q_scale, _, q_os = comfy.ops.cast_bias_weight(attn.q_norm, qc, offloadable=True)
            qc = ck.rms_rope_split_half(qc, qc, rc, q_scale, q_scale, eps)[0]
            if fused:
                res = fb.rows_attention(attn, qc, K, V, transformer_options, kv_cache)
                del qc
                xc = fb.rows_out_residual(attn.output_proj, attn.output_dropout, res, xc, g_sa, residual_dtype)
                del res
                xc = fb.rows_sub(blk, transformer_options, compute_dtype, residual_dtype, blk.cross_attn,
                                 (blk.layer_norm_cross_attn, xc), sc_ca, sh_ca,
                                 lambda nn_: blk.cross_attn(nn_.to(compute_dtype), crossattn_emb,
                                                            rope_emb=rope_emb_L_1_1_D,
                                                            transformer_options=transformer_options),
                                 xc, g_ca, blk.cross_attn.output_proj, overrides=(blk.cross_attn.q_proj,))
                xc = fb.rows_sub(blk, transformer_options, compute_dtype, residual_dtype, blk.mlp,
                                 (blk.layer_norm_mlp, xc), sc_mlp, sh_mlp, lambda nn_: blk.mlp(nn_.to(compute_dtype)),
                                 xc, g_mlp, blk.mlp.layer2)
                if out is None:
                    out = (host_out.view(B, S, D) if host_out is not None
                           else torch.empty((B, S, D), dtype=xc.dtype, device=xc.device))
                _rows_store(out, xc, i, n, rows, d2h, compute, stage, stage_free, host_out)
                del xc
                continue
            res = attn.attn_op(AttentionTensorContainer(qc), AttentionTensorContainer(K), AttentionTensorContainer(V),
                               transformer_options=transformer_options, preferred_attention=attn.comfy_attention)
            del qc
            sa = attn.output_dropout(attn.output_proj(res))
            del res
            xc = torch.addcmul(xc, g_sa.to(residual_dtype), sa.to(residual_dtype))
            del sa
            nc = mod_norm(xc, blk.layer_norm_cross_attn, sc_ca, sh_ca)
            ca = blk.cross_attn(nc.to(compute_dtype), crossattn_emb, rope_emb=rope_emb_L_1_1_D,
                                transformer_options=transformer_options)
            del nc
            xc = torch.addcmul(xc, g_ca.to(residual_dtype), ca.to(residual_dtype))
            del ca
            nc = mod_norm(xc, blk.layer_norm_mlp, sc_mlp, sh_mlp).to(compute_dtype)
            mo = blk.mlp(nc)
            del nc
            xc = torch.addcmul(xc, g_mlp.to(residual_dtype), mo.to(residual_dtype))
            del mo
            if out is None:
                out = (host_out.view(B, S, D) if host_out is not None
                       else torch.empty((B, S, D), dtype=xc.dtype, device=xc.device))
            _rows_store(out, xc, i, n, rows, d2h, compute, stage, stage_free, host_out)
            del xc
        if d2h is not None:
            done = torch.cuda.Event()
            done.record(d2h)
            compute.wait_event(done)
    finally:
        if q_scale is not None:
            comfy.ops.uncast_bias_weight(attn.q_norm, q_scale, None, q_os)
        if k_scale is not None:
            comfy.ops.uncast_bias_weight(attn.k_norm, k_scale, None, k_os)
    return out.reshape(B, T, H, W, D)


def _rows_store(out, xc, i, n, rows, d2h, compute, stage, stage_free, host_out):
    """Write one finished row chunk to the block output (through a staging buffer on the D2H stream when the
    residual stream is in pinned RAM)."""
    if d2h is not None:
        k = (i // rows) % 2
        compute.wait_event(stage_free[k])      # the previous host copy out of this buffer has run
        st = stage[k][:, :n]
        st.copy_(xc)
        ready = torch.cuda.Event()
        ready.record(compute)
        d2h.wait_event(ready)
        with torch.cuda.stream(d2h):
            out[:, i:i + n].copy_(st, non_blocking=True)
        stage_free[k].record(d2h)
    else:
        out[:, i:i + n].copy_(xc, non_blocking=host_out is not None)


# Off by default (0): activation streaming only for images of at least this many tokens, e.g.
# 8192 = the 2x upscale without tiles (1792x2944 = 20608 tokens) but not the 896x1472 base (5152), which fits and
# would only get slower (forced on for everything at a 4 GB cap: base 9 -> 18 s)
_ACT_MIN_TOKENS = int(_env.get("WEIGHTFERRY_ACT_CHUNK_MIN_TOKENS"))
# Automatic mode (WEIGHTFERRY_ACT_CHUNK=auto, the default): images of at least this many tokens (never the 896x1472
# base) are computed by token rows when one block's activations would not fit in the free VRAM. Measured on an Anima
# block (2048 channels, bf16, SageAttention): the peak is 17.1 x (batch x tokens x channels x 2 bytes) at 5152 and at
# 20608 tokens alike (2.75 GB for the 2x upscale without tiles with cond + uncond); 18 x keeps a margin
_ACT_AUTO_MIN_TOKENS = 8192
_ACT_AUTO_FACTOR = 18.0


def _act_auto_needed(dit, x):
    """Automatic mode: should this (large) block input be computed by token rows? Sticky per token count once it
    was (the free VRAM grows while chunking, which must not switch it off again in the middle of a run)."""
    tokens = x.shape[2] * x.shape[3]
    done = dit.__dict__.setdefault("_weightferry_act_auto_tokens", set())
    if tokens in done:
        return True
    if tokens < max(_ACT_MIN_TOKENS, _ACT_AUTO_MIN_TOKENS):
        return False
    need = _ACT_AUTO_FACTOR * x.shape[0] * tokens * x.shape[-1] * x.element_size()
    free = vram_monitor.effective_free_vram(x.device)
    if need <= free:
        return False
    done.add(tokens)
    log_info(f"DiT Layer Streaming: activation streaming switched on for {tokens}-token images (one block needs "
             f"about {need / 1024 ** 2:.0f} MB of activations, {max(0, free) / 1024 ** 2:.0f} MB of VRAM free).")
    return True


def _act_host_wanted(dit, x):
    """Keep the residual stream between blocks in pinned RAM? WEIGHTFERRY_ACT_HOST: 1 yes, 0 no, auto (default): on
    the GPU when its two copies (block input and output) fit in the free VRAM with 64 MB to spare -- the round trips
    through pinned RAM are most of activation streaming's cost (Cosmos 14B, 896x1472: 3.39 s per forward against
    2.55 s with the residual on the GPU). Decided once per token count (sticky), values unchanged either way."""
    mode = _env.get("WEIGHTFERRY_ACT_HOST")
    if mode != "auto":
        return mode != "0"
    tokens = x.shape[2] * x.shape[3]
    decided = dit.__dict__.setdefault("_weightferry_act_host_auto", {})
    if tokens not in decided:
        need = 2 * x.numel() * x.element_size()
        free = vram_monitor.effective_free_vram(x.device)
        decided[tokens] = need + 64 * 1024 ** 2 > free
        log_info(f"DiT Layer Streaming: activation streaming keeps the residual stream "
                 f"{'in pinned RAM' if decided[tokens] else 'on the GPU'} for {tokens}-token images "
                 f"(two copies {need / 1024 ** 2:.0f} MB, {max(0, free) / 1024 ** 2:.0f} MB of VRAM free).")
    return decided[tokens]


def _install_block_chunking(dit, blocks, owner, rows, only_after_overflow=False):
    """Whole blocks by token rows (_cosmos_block_rows) while this owner streams.
    Falls back to ComfyUI's forward for video (T > 1), attention / MLP patches, short sequences, and when the chunk
    streaming state could not keep a whole block's chunks bound (more chunk types than slots + spare)."""
    installed = []
    n_blocks = len(blocks)
    _sage_fp16_for_per_item()
    for blk in blocks:
        if "_weightferry_block_orig" in blk.__dict__ or not _chunkable_block(blk):
            continue
        orig = blk.forward

        def chunked(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                    extra_per_block_pos_emb=None, transformer_options={}, _orig=orig, _blk=blk):
            patches = (transformer_options or {}).get("patches", {})
            state = getattr(dit, _STATE_ATTR, None)
            fits = (state is None or not getattr(state, "chunk_mode", False) or state.types is None
                    or state.n_types <= state.n_chunk_slots + 1)
            # Automatic mode: when one block's activations would not fit in the free VRAM (large images only), or from
            # the block after a VRAM overflow was noticed on (check_overflow, also inside a forward) -- not from the
            # next load: ComfyUI may keep the model loaded across generations
            if only_after_overflow and not int(getattr(dit, "_weightferry_vram_overflow_bytes", 0) or 0) and not (
                    x_B_T_H_W_D.dim() == 5 and _act_auto_needed(dit, x_B_T_H_W_D)):
                fits = False
            elif fits and only_after_overflow and not getattr(dit, "_weightferry_act_live", False):
                # Switching on: give PyTorch's cached GPU memory back first. The cached segments that overflowed
                # into shared GPU memory would otherwise serve the new, smaller allocations too and keep everything
                # slow
                dit._weightferry_act_live = True
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if (getattr(dit, _ACTIVE_ATTR, None) is not owner or not fits or rope_emb_L_1_1_D is None
                    or x_B_T_H_W_D.dim() != 5 or x_B_T_H_W_D.shape[1] != 1 or emb_B_T_D.shape[1] != 1
                    or x_B_T_H_W_D.shape[2] * x_B_T_H_W_D.shape[3] <= rows
                    or x_B_T_H_W_D.shape[2] * x_B_T_H_W_D.shape[3] < _ACT_MIN_TOKENS
                    or any(k in patches for k in ("attn1_patch", "attn2_patch", "mlp_patch"))):
                return _orig(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                             adaln_lora_B_T_3D=adaln_lora_B_T_3D, extra_per_block_pos_emb=extra_per_block_pos_emb,
                             transformer_options=transformer_options)
            hold = getattr(state, "hold_completions", None)
            host_out = None
            if _act_host_wanted(dit, x_B_T_H_W_D):
                # Residual stream in pinned RAM between blocks (two buffers used in turn); the last block returns
                # its output on the GPU for the final layer
                idx = (transformer_options or {}).get("block_index")
                if idx is not None and idx < n_blocks - 1:
                    host_out = _act_host_buffer(dit, x_B_T_H_W_D)
            if hold is not None:
                hold(1)
            try:
                B = x_B_T_H_W_D.shape[0]
                # A batch is computed as a batch by default: item by item halves K / V but is not bit-exact in the
                # whole model (Cosmos 14B, cfg 4; one block alone is) -- WEIGHTFERRY_ACT_PER_ITEM=1
                if B == 1 or _env.get("WEIGHTFERRY_ACT_PER_ITEM") != "1":
                    return _cosmos_block_rows(_blk, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D,
                                              adaln_lora_B_T_3D, extra_per_block_pos_emb, transformer_options, rows,
                                              host_out=host_out, device=emb_B_T_D.device)
                # A batch (e.g. cond + uncond with CFG): one item after the other, so K / V exist for one item only
                # (Cosmos 14B at 1792x2944, cfg 4, 2 GB: the batch of 2 doubles K / V and overflows).
                # Items are independent in every operation of the block.
                out = host_out
                if out is None:  # last block: the output on the GPU, written item by item
                    out = torch.empty(x_B_T_H_W_D.shape, dtype=x_B_T_H_W_D.dtype, device=emb_B_T_D.device)
                # the adaLN modulations for the whole batch at once, as ComfyUI (bf16 GEMMs: a batch of 1 can give
                # other results than a batch of 2)
                mods_all = _cosmos_block_mods(_blk, emb_B_T_D, adaln_lora_B_T_3D)
                for b in range(B):
                    sl = slice(b, b + 1)
                    pick = (lambda t: None if t is None else (t[sl] if t.shape[0] == B else t))
                    mods_b = [tuple(pick(t) for t in m) for m in mods_all]
                    _cosmos_block_rows(_blk, x_B_T_H_W_D[sl], pick(emb_B_T_D), pick(crossattn_emb),
                                       rope_emb_L_1_1_D, pick(adaln_lora_B_T_3D), pick(extra_per_block_pos_emb),
                                       transformer_options, rows, host_out=out[sl], device=emb_B_T_D.device,
                                       mods=mods_b)
                return out
            finally:
                if hold is not None:
                    hold(-1)

        blk._weightferry_block_orig = orig
        blk.forward = chunked
        installed.append(blk)
    return installed


def _prefetch_stream(device):
    """CUDA stream for weight transfers (+ dequantization / LoRA / decompression kernels). WEIGHTFERRY_PREFETCH_PRIORITY
    (experiment): a stream priority, lower = higher (device-dependent range). ComfyUI computes on the default
    stream (priority 0, the lowest), so only a higher priority for the transfer side can be tried."""
    prio = _env.get("WEIGHTFERRY_PREFETCH_PRIORITY")
    if prio:
        return torch.cuda.Stream(device=device, priority=int(prio))
    return torch.cuda.Stream(device=device)


_ACT_D2H_STREAMS = {}
_ACT_H2D_STREAMS = {}


def _act_stream_enabled(name, streams, device):
    """The copy stream of activation streaming for ``device`` (one per device, kept in ``streams``) when the switch
    ``name`` is 1, else None."""
    if _env.get(name) != "1":
        return None
    key = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    st = streams.get(key)
    if st is None:
        st = streams[key] = torch.cuda.Stream(device=key)
    return st


def _act_d2h_stream(device):
    """The device-to-host stream of activation streaming (WEIGHTFERRY_ACT_D2H_STREAM=1, the default), else None."""
    return _act_stream_enabled("WEIGHTFERRY_ACT_D2H_STREAM", _ACT_D2H_STREAMS, device)


def _act_h2d_stream(device):
    """The host-to-device stream of activation streaming (WEIGHTFERRY_ACT_H2D_STREAM=1, the default), else None."""
    return _act_stream_enabled("WEIGHTFERRY_ACT_H2D_STREAM", _ACT_H2D_STREAMS, device)


_ACT_D2H_STAGE = {}


def _act_d2h_stage(device, B, rows, D, dtype):
    """Two GPU staging buffers (B, rows, D) and their 'free' events for the D2H stream (allocated once per shape)."""
    key = (str(device), B, rows, D, dtype)
    v = _ACT_D2H_STAGE.get(key)
    if v is None:
        _ACT_D2H_STAGE.clear()
        bufs = [torch.empty((B, rows, D), dtype=dtype, device=device) for _ in range(2)]
        evs = [torch.cuda.Event() for _ in range(2)]
        for ev in evs:  # (allocated on the compute stream: recorded so the first wait is ordered after it)
            ev.record(torch.cuda.current_stream(device))
        v = _ACT_D2H_STAGE[key] = (bufs, evs)
    return v


def _row_chunks(x, extra, S, rows, dev):
    """(start, GPU rows) of x (B, S, D) in chunks of ``rows``, plus ``extra`` when given. With x in pinned RAM and
    WEIGHTFERRY_ACT_H2D_STREAM=1 the next chunk is copied to the GPU on its own stream while the current one is
    computed (two buffers in turn; a buffer is refilled only after the compute stream is past its chunk); otherwise
    each chunk is copied on the compute stream when it is needed. Values unchanged either way."""
    starts = list(range(0, S, rows))
    h2d = _act_h2d_stream(dev) if x.device != dev else None
    if h2d is None:
        for i in starts:
            xc = x[:, i:i + rows]
            if xc.device != dev:
                xc = xc.to(dev, non_blocking=True)
            yield i, (xc if extra is None else xc + extra[:, i:i + rows])
        return
    compute = torch.cuda.current_stream(dev)
    B, _, D = x.shape
    bufs = [torch.empty((B, min(rows, S), D), dtype=x.dtype, device=dev) for _ in range(2)]
    free = [torch.cuda.Event() for _ in range(2)]
    ready = [torch.cuda.Event() for _ in range(2)]
    for ev in free:  # (allocated on the compute stream: the first copy into a buffer waits for that)
        ev.record(compute)

    def issue(c):
        i, k = starts[c], c % 2
        n = min(rows, S - i)
        h2d.wait_event(free[k])
        with torch.cuda.stream(h2d):
            bufs[k][:, :n].copy_(x[:, i:i + n], non_blocking=True)
        ready[k].record(h2d)

    issue(0)
    for c, i in enumerate(starts):
        if c + 1 < len(starts):
            issue(c + 1)
        k = c % 2
        compute.wait_event(ready[k])
        xc = bufs[k][:, :min(rows, S - i)]
        yield i, (xc if extra is None else xc + extra[:, i:i + rows])
        free[k].record(compute)  # the compute stream is past every read of this chunk's buffer


def _track_input_shape(model_patcher, dit):
    """Remember the latent shape of the coming sampling run on the DiT (dit._weightferry_input_shape): ComfyUI asks the
    model's memory_required(input_shape) before it loads the model (before ON_LOAD), so the RAM tier plan can reserve
    the activation host buffers (_act_host_reserve). Wraps the BaseModel instance's method once."""
    base = getattr(model_patcher, "model", None)
    orig = getattr(base, "memory_required", None)
    if orig is None or getattr(orig, "_weightferry_tracks", False):
        return

    def memory_required(input_shape, *args, **kwargs):
        # ComfyUI asks twice before loading (batch x2 for cond + uncond, then the minimum x1) and again while
        # sampling: keep the largest batch while the latent size is the same
        try:
            shape = tuple(int(v) for v in input_shape)
            prev = getattr(dit, "_weightferry_input_shape", None)
            if prev and prev[1:] == shape[1:] and prev[0] > shape[0]:
                shape = prev
            dit._weightferry_input_shape = shape
        except Exception:  # noqa: BLE001 -- only a hint for the RAM plan; never break ComfyUI's estimate
            pass
        return orig(input_shape, *args, **kwargs)

    memory_required._weightferry_tracks = True
    base.memory_required = memory_required


def _act_host_reserve(dit):
    """Pinned RAM that activation streaming will take for the coming run (two residual-stream buffers of the batch,
    fp32 as measured: Cosmos 14B 1792x2944 cfg -> 1.7 GB), minus what this model already holds. 0 when unknown.
    The RAM tier plan keeps this free (without it a full RAM tier plus these buffers can exhaust the RAM)."""
    shape = getattr(dit, "_weightferry_input_shape", None)
    d = getattr(dit, "model_channels", None)
    if not shape or len(shape) < 4 or not d:
        return 0
    ps = int(getattr(dit, "patch_spatial", 2) or 2)
    pt = int(getattr(dit, "patch_temporal", 1) or 1)
    b, h, w = shape[0], shape[-2], shape[-1]
    t = shape[2] if len(shape) == 5 else 1
    tokens = -(-t // pt) * -(-h // ps) * -(-w // ps)
    need = 2 * b * tokens * int(d) * 4
    held = sum(hd.nbytes for bufs in (dit.__dict__.get("_weightferry_act_host") or {}).values() for hd, _t in bufs)
    return max(0, need - held)


def _act_host_buffer(dit, x):
    """A pinned CPU buffer for a block's output (residual stream on the host), not the one ``x`` lives in. Two
    buffers per shape / dtype, kept on the model (dit._weightferry_act_host) and used in turn."""
    key = (tuple(x.shape), x.dtype)
    cache = dit.__dict__.setdefault("_weightferry_act_host", {})
    bufs = cache.get(key)
    if bufs is None:
        cache.clear()  # a new size: drop the old buffers
        nbytes = x.numel() * x.element_size()
        holders = [tiered_store.PinnedHost(nbytes) for _ in range(2)]
        bufs = [(h, h.tensor.view(x.dtype).view(x.shape)) for h in holders]
        cache[key] = bufs
    src = x.untyped_storage().data_ptr() if x.device.type == "cpu" else None
    for _h, t in bufs:
        if t.untyped_storage().data_ptr() != src:
            return t
    return bufs[0][1]


def _uninstall_block_chunking(modules):
    for blk in modules or ():
        blk.__dict__.pop("forward", None)
        blk.__dict__.pop("_weightferry_block_orig", None)


def _install_attn_chunking(dit, blocks, owner, rows):
    """Self-attention by query rows (_cosmos_self_attention_rows) while this owner streams. Falls back to ComfyUI's
    forward for anything else (cross-attention, no RoPE, attention patches, short sequences). Returns the modules."""
    installed = []
    for blk in blocks:
        attn = getattr(blk, "self_attn", None)
        if (not isinstance(attn, torch.nn.Module) or "_weightferry_attn_orig" in attn.__dict__
                or not _chunkable_self_attention(attn)):
            continue
        orig = attn.forward

        def chunked(x, context=None, rope_emb=None, transformer_options={}, _orig=orig, _attn=attn):
            if (getattr(dit, _ACTIVE_ATTR, None) is not owner or context is not None or rope_emb is None
                    or not isinstance(x, torch.Tensor) or x.dim() != 3 or x.shape[1] <= rows
                    or rope_emb.dim() < 2 or rope_emb.shape[1] != x.shape[1]
                    or "attn1_patch" in (transformer_options or {}).get("patches", {})):
                return _orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            state = getattr(dit, _STATE_ATTR, None)
            hold = getattr(state, "hold_completions", None)
            if hold is not None:
                hold(1)
            try:
                return _cosmos_self_attention_rows(_attn, x, rope_emb, transformer_options, rows)
            finally:
                if hold is not None:
                    hold(-1)

        attn._weightferry_attn_orig = orig
        attn.forward = chunked
        installed.append(attn)
    return installed


def _uninstall_attn_chunking(modules):
    for attn in modules or ():
        attn.__dict__.pop("forward", None)
        attn.__dict__.pop("_weightferry_attn_orig", None)
