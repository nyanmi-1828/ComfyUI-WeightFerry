# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
SageAttention injection and its FP16 P*V kernel switch (WEIGHTFERRY_SAGE_PV)."""

from ..utils.logging import log_info, log_warn
from .common import _blocks
from ..core import env as _env


_SAGE_ORIG = None


def _sage_fp16_for_per_item():
    """Item-by-item batches (WEIGHTFERRY_ACT_PER_ITEM=1) with SageAttention: its FP8 P*V kernel (the one sageattn()
    picks on Ada) gives other results for an item computed alone than inside a batch of 2 in the real model
    (Cosmos 14B), the FP16 P*V kernel does not. Switch to it (process-wide, from the node's apply
    on, so batches and items use the same kernel) unless WEIGHTFERRY_SAGE_PV=fp8."""
    if (_env.get("WEIGHTFERRY_ACT_PER_ITEM") != "1"
            or _env.get("WEIGHTFERRY_SAGE_PV").lower() == "fp8" or _SAGE_ORIG is not None):
        return
    try:
        if _install_sage_pv_fp16():
            log_info("DiT Layer Streaming: item-by-item activation streaming -> SageAttention FP16 P*V "
                     "(batch-invariant; WEIGHTFERRY_SAGE_PV=fp8 keeps the FP8 kernel).")
    except Exception as exc:  # noqa: BLE001 -- optional kernel switch: keep the current one
        log_warn(f"DiT Layer Streaming: SageAttention FP16 P*V switch failed ({exc!r}).")


def _install_sage_pv_fp16():
    """Make ComfyUI's SageAttention use the FP16 P*V kernel (sageattn_qk_int8_pv_fp16_cuda) instead of the FP8 one
    that sageattn() picks on Ada GPUs. The FP8 kernel quantizes V with a per-channel scale; a V channel that is zero
    for every key token gets scale 0 and the kernel returns 0/0 = NaN (e.g. an fp8 copy of Cosmos-Predict2 14B
    with 183 v_proj rows rounded to zero: 3.6% of the cross-attention output was NaN; the FP16 kernel gives finite
    results within ~1e-5 of PyTorch SDPA). Global (patches comfy.ldm.modules.attention.sageattn),
    idempotent. Returns True when installed."""
    global _SAGE_ORIG
    import comfy.ldm.modules.attention as attn_mod
    if _SAGE_ORIG is not None:
        return True
    orig = getattr(attn_mod, "sageattn", None)
    try:
        import sageattention
        fp16_kernel = sageattention.sageattn_qk_int8_pv_fp16_cuda
    except (ImportError, AttributeError):  # SageAttention not installed / without that kernel
        fp16_kernel = None
    if orig is None or fp16_kernel is None:
        log_warn("DiT Layer Streaming: SageAttention FP16-PV kernel not available; nothing changed.")
        return False

    def sageattn_pv_fp16(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs):
        if "attn_mask" in kwargs:  # the FP16 CUDA kernel takes no mask
            return orig(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
                        return_lse=return_lse, **kwargs)
        return fp16_kernel(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
                           smooth_k=kwargs.get("smooth_k", True), return_lse=return_lse)

    _SAGE_ORIG = orig
    attn_mod.sageattn = sageattn_pv_fp16
    log_info("DiT Layer Streaming: SageAttention now uses the FP16 P*V kernel (no NaN on all-zero V channels).")
    return True


if _env.get("WEIGHTFERRY_SAGE_PV").lower() == "fp16":
    try:
        _install_sage_pv_fp16()
    except Exception as _exc:  # noqa: BLE001 -- at import: never stop ComfyUI from loading the nodes
        log_warn(f"DiT Layer Streaming: could not install the SageAttention FP16-PV kernel: {_exc}")


def _apply_sage_attention(dit):
    import comfy.ldm.modules.attention as comfy_attn

    sage_fn = comfy_attn.get_attention_function("sage", None)
    if sage_fn is None:
        log_warn("DiT Layer Streaming: SageAttention requested but 'sage' is not registered (not installed?).")
        return

    injected = 0
    for blk in _blocks(dit):
        for attn_name in ("self_attn", "cross_attn"):
            attn = getattr(blk, attn_name, None)
            if attn is not None and hasattr(attn, "comfy_attention"):
                attn.comfy_attention.function = sage_fn
                injected += 1
    if injected == 0:
        log_warn("DiT Layer Streaming: no attention module exposes 'comfy_attention'; nothing injected.")
    else:
        log_info(f"DiT Layer Streaming: SageAttention 2 injected into {injected} attention modules.")
