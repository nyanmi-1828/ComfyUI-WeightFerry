# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Releasing the streaming state and idle text encoder caches."""

import torch

from ..utils.logging import log_info, log_warn
from ..core import tiered_store
from .common import _ACTIVE_ATTR, _CACHE_ATTR, _STATE_ATTR, _TE_ROOTS
from .activation import _uninstall_attn_chunking, _uninstall_block_chunking, _uninstall_mlp_chunking
from ..core import env as _env

_STUCK_RINGS = []  # SSD rings whose reader threads did not stop: kept alive (their buffers may still be written)


def _release_idle_text_encoder_caches(reason):
    """Release the capture caches (pinned RAM tier, SSD cache file) of text encoders that are not encoding now.

    Called before a DiT capture plans its RAM tier with the automatic budget: a text encoder runs once per prompt
    change (it captures again when it encodes next, ~9 s for Qwen2.5-VL 7B), the DiT every step, so the RAM an idle
    encoder keeps pinned is better used by the DiT's RAM tier. WEIGHTFERRY_RELEASE_TE=0 keeps them. Returns MB
    released."""
    if _env.get("WEIGHTFERRY_RELEASE_TE") == "0":
        return 0
    freed = 0
    for root in list(_TE_ROOTS):
        if getattr(root, _ACTIVE_ATTR, None) is not None:
            continue
        cache = getattr(root, _CACHE_ATTR, None)
        if cache is None:
            continue
        _teardown(root, reason)
        plan = cache.get("plan") or {}
        freed += int(plan.get("host_ram_planned_mb", 0) or 0)
        for key in ("disk_store", "res_disk_store"):
            if cache.get(key) is not None:
                try:
                    cache[key].delete()
                except Exception:  # noqa: BLE001 -- teardown: free the rest anyway
                    pass
        tiered_store.release_all(cache.get("holders"))
        setattr(root, _CACHE_ATTR, None)
        cache = None  # its views keep the buffers in use: VirtualAlloc'd buffers are freed once none is left
    if freed:
        import gc
        gc.collect()
        tiered_store.empty_host_cache()  # also frees the buffers whose release had to wait for their views
    return freed


def _teardown(dit, reason):
    state = getattr(dit, _STATE_ATTR, None)
    if state is None:
        return
    for handle in state.handles:
        try:
            handle.remove()
        except Exception:  # noqa: BLE001 -- teardown: remove the other hooks anyway
            pass
    _uninstall_mlp_chunking(getattr(state, "mlp_chunked", None))
    state.mlp_chunked = None
    _uninstall_attn_chunking(getattr(state, "attn_chunked", None))
    state.attn_chunked = None
    _uninstall_block_chunking(getattr(state, "block_chunked", None))
    state.block_chunked = None
    # Make sure no in-flight DMA / kernel still touches the slots before releasing them
    torch.cuda.synchronize(state.device)
    state.harvest_block_timing(dit)
    state.restore_rest_binding()
    if state.uploader is not None:
        state.uploader.drain()
        torch.cuda.synchronize(state.device)
        state.uploader.close()
        log_info(f"DiT Layer Streaming [tiers]: main thread waited {state.uploader.main_wait_s * 1000:.0f} ms "
                 f"in total for SSD uploads (just in time, in pre-hooks); the background uploader waited "
                 f"{state.ring.wait_s * 1000:.0f} ms for SSD reads during this load.")
        state.uploader = None
    elif state.ring is not None:
        log_info(f"DiT Layer Streaming [tiers]: CPU waited {state.ring.wait_s * 1000:.0f} ms in total "
                 f"for SSD reads during this load.")
    if state.ring is not None:
        try:
            state.ring.close()
        except TimeoutError as exc:
            # a reader thread is still inside a read: keep the ring (its file handles and pinned buffers) alive
            # rather than free memory under it; the rest of the teardown goes on
            _STUCK_RINGS.append(state.ring)
            log_warn(f"DiT Layer Streaming [tiers]: {exc}; its buffers are kept until the process exits.")
        state.ring = None
    if state.profiler is not None:
        m = state.profiler.summary()["mean_excluding_first_forward"]
        if m:
            log_info(
                f"DiT Layer Streaming [profile]: per forward {m['forward_ms']} ms = compute resident "
                f"{m['resident_compute_ms']} + streamed {m['streamed_compute_ms']} + idle {m['compute_idle_ms']} ms; "
                f"DMA {m['dma_ms']} ms ({m['dma_gbps']} GB/s), dequant+LoRA {m['dequant_lora_ms']} ms, "
                f"transfer stall {m['stall_ms']} ms, CPU in hooks {m['cpu_hook_ms']} ms"
            )
    # Drop the buffers held by the state object: a lingering reference to it (a script, a closure) must not keep
    # host / GPU memory alive (e.g. a lingering reference would keep the previous capture's pinned RAM). Pinned
    # buffers owned by the
    # capture cache stay (they are reused); without a cache entry owning them they are released here.
    cache = getattr(dit, _CACHE_ATTR, None)
    holders = getattr(state, "holders", None)
    if holders and (cache is None or cache.get("holders") is not holders):
        tiered_store.release_all(holders)
    state.holders = []
    state.host_flats = state.host_views = state.rest = None
    state.slot_flats, state.slot_views = [], []
    state.staging_flat = state.staging_views = None
    state.resident_gpu_flats, state.resident_views, state.resident_rest = [], [], []
    state.res_disk_store = None
    state.pending_spans = []
    state.quant_bind = state.quant_rest = None
    state.res_quant_bind = state.res_quant_rest = None
    setattr(dit, _STATE_ATTR, None)
    log_info(f"DiT Layer Streaming: Released previous streaming state ({reason}).")
