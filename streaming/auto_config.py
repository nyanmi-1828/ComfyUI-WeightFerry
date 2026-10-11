# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Automatic configuration: resident block count and tiering from free VRAM / RAM."""

import torch

import comfy.model_management
from .common import _blocks, _param_nbytes
from .capture import _quant_lora_transfer_vram
from ..core import env as _env


def _auto_config(dit, device, prefetch_slots, is_double, patcher=None):
    """Pick resident blocks for the usable VRAM. Returns (resident_k or None = everything fits, info dict)."""
    blocks = list(_blocks(dit))
    num_blocks = len(blocks)
    block_ids = {id(prm) for blk in blocks for prm in blk.parameters()}
    block_bytes = max(sum(_param_nbytes(prm) for prm in blk.parameters()) for blk in blocks)
    nonblock_bytes = sum(_param_nbytes(prm) for prm in dit.parameters() if id(prm) not in block_ids)
    free, total = torch.cuda.mem_get_info(device)
    # What ComfyUI can make available before sampling: free VRAM, PyTorch's cached-but-unused memory, and the VRAM
    # held by loaded models (they are evicted when needed). Dynamic VRAM keeps weights in aimdo VBARs outside
    # PyTorch's allocator, so memory_reserved() does not see them (measured: right after an unstreamed
    # run of Anima 2.9B, free VRAM was 1.3 GB although ~5.8 GB were held by that model).
    stats = torch.cuda.memory_stats(device)
    torch_cached = stats.get("reserved_bytes.all.current", 0) - stats.get("active_bytes.all.current", 0)
    loaded = 0
    for lm in list(getattr(comfy.model_management, "current_loaded_models", [])):
        try:
            if str(lm.device) != str(device):
                continue
            if not lm.is_dead():
                loaded += int(lm.model_loaded_memory())
                continue
            # Dead entry (its ModelPatcher was garbage collected, e.g. a text encoder no longer referenced): the
            # model still holds its VRAM until ComfyUI frees it at the next load, so it is usable too
            # (e.g. a T5 encoder can leave ~3 GB on the GPU).
            rm = lm.real_model()
            if rm is not None:
                vbar = (getattr(rm, "dynamic_vbars", None) or {}).get(device)
                loaded += int(getattr(rm, "model_loaded_weight_memory", 0) or 0)
                if vbar is not None:
                    loaded += int(vbar.loaded_size())
        except Exception:  # noqa: BLE001 -- ComfyUI internals differ by version: estimate without them
            pass
    usable = free + torch_cached + loaded
    # VRAM that overflowed into system RAM on an earlier load of this model (check_overflow): keep it free now
    overflow_penalty = int(getattr(dit, "_weightferry_vram_overflow_bytes", 0) or 0)
    usable -= overflow_penalty
    headroom = int(_env.get("WEIGHTFERRY_AUTO_HEADROOM_MB")) * 1024 ** 2
    quant_lora_extra = _quant_lora_transfer_vram(dit, patcher)
    headroom += quant_lora_extra
    slots = (max(2, int(prefetch_slots)) if is_double else 1) * block_bytes
    fit_all = num_blocks * block_bytes + nonblock_bytes + headroom
    info = {"usable_vram_mb": round(usable / 1024 ** 2),
            "free_vram_mb": round(free / 1024 ** 2),
            "loaded_models_mb": round(loaded / 1024 ** 2),
            "overflow_penalty_mb": round(overflow_penalty / 1024 ** 2),
            "total_vram_mb": round(total / 1024 ** 2),
            "block_mb": round(block_bytes / 1024 ** 2, 1),
            "nonblock_mb": round(nonblock_bytes / 1024 ** 2),
            "headroom_mb": round(headroom / 1024 ** 2),
            "quant_lora_transfer_mb": round(quant_lora_extra / 1024 ** 2),
            "model_mb": round((num_blocks * block_bytes + nonblock_bytes) / 1024 ** 2)}
    if fit_all <= usable:
        info["decision"] = "fits: no streaming"
        return None, info
    resident = int((usable - headroom - nonblock_bytes - slots) // max(1, block_bytes))
    resident = max(1, min(num_blocks - 1, resident))
    info["decision"] = f"stream {num_blocks - resident} of {num_blocks} blocks, {resident} resident"
    info["resident_blocks"] = resident
    # Chunk streaming when even block streaming does not fit: block 0 resident + the whole-block slots + half the
    # activation headroom (otherwise VRAM overflows into shared GPU memory: Cosmos-Predict2 14B at 2 GB, 26 s/it
    # instead of 5)
    block_need = nonblock_bytes + slots + block_bytes + headroom // 2
    info["chunk"] = bool(usable < block_need)
    if info["chunk"]:
        info["decision"] += "; chunk streaming (whole-block slots do not fit)"
    return resident, info
