# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Adapted from sparklingcoffee777/ComfyUI-Anima-2.9B-loraPatch (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
ComfyUI-WeightFerry: Anima 2.9B LoRA Index Remapper
===================================================
Applies 28-block Anima LoRAs (anima-base-v1.0 / anima-preview and its finetunes)
to the 40-block Anima-2.9B model by remapping their block indices.

Provenance: adapted from sparklingcoffee777/ComfyUI-Anima-2.9B-loraPatch (GPL-3.0); the
inserted block indices (config.INSERTED_BLOCK_INDICES) are that project's measurements.
"""

import re
from typing import Dict, Set
import comfy.lora

from ..config import WeightFerryConfig
from ..utils.logging import log_info
from . import env as _env


def _build_block_map() -> Dict[int, int]:
    """Build mapping from 28-block base index to 40-block 2.9B index."""
    kept = [i for i in range(WeightFerryConfig.NUM_BLOCKS_29B) if i not in WeightFerryConfig.INSERTED_BLOCK_INDICES]
    return {j: i for j, i in enumerate(kept)}


BASE_TO_29B: Dict[int, int] = _build_block_map()
assert len(BASE_TO_29B) == WeightFerryConfig.NUM_BLOCKS_BASE

MODEL_BLOCK_RE = re.compile(r"^(diffusion_model\.blocks\.)(\d+)(\..*)$")
LORA_BLOCK_RES = (
    re.compile(r"^lora_unet_blocks_(\d+)_"),
    re.compile(r"^diffusion_model\.blocks\.(\d+)\."),
    re.compile(r"^blocks\.(\d+)\."),
)


def _is_remapping_enabled() -> bool:
    return _env.get("WEIGHTFERRY_LORA_REMAP").strip().lower() not in ("0", "off", "false", "no")


def _model_block_indices(to_load: dict) -> Set[int]:
    out = set()
    for value in to_load.values():
        if isinstance(value, str):
            m = MODEL_BLOCK_RE.match(value)
            if m is not None:
                out.add(int(m.group(2)))
    return out


def _lora_block_indices(lora: dict) -> Set[int]:
    out = set()
    for key in lora.keys():
        if isinstance(key, str):
            for rex in LORA_BLOCK_RES:
                m = rex.match(key)
                if m is not None:
                    out.add(int(m.group(1)))
                    break
    return out


def _is_anima_model(to_load: dict) -> bool:
    for value in to_load.values():
        if isinstance(value, str) and value.startswith("diffusion_model.llm_adapter."):
            return True
    return False


def _remap_key_map(to_load: dict) -> tuple[dict, int]:
    remapped = {}
    count = 0
    for lora_key, model_key in to_load.items():
        if isinstance(model_key, str):
            m = MODEL_BLOCK_RE.match(model_key)
            if m is not None:
                base_idx = int(m.group(2))
                if base_idx in BASE_TO_29B:
                    new_idx = BASE_TO_29B[base_idx]
                    remapped[lora_key] = f"{m.group(1)}{new_idx}{m.group(3)}"
                    count += 1
                    continue
        remapped[lora_key] = model_key
    return remapped, count


def install_lora_remap_patch() -> None:
    """Hooks comfy.lora.load_lora to dynamically remap 28-block LoRAs for Anima 2.9B."""
    orig_load_lora = comfy.lora.load_lora

    if getattr(orig_load_lora, "_weightferry_29b_patched", False):
        return

    def patched_load_lora(lora: dict, to_load: dict):
        if not _is_remapping_enabled():
            return orig_load_lora(lora, to_load)

        if not _is_anima_model(to_load):
            return orig_load_lora(lora, to_load)

        model_blocks = _model_block_indices(to_load)
        if model_blocks != set(range(WeightFerryConfig.NUM_BLOCKS_29B)):
            return orig_load_lora(lora, to_load)

        lora_blocks = _lora_block_indices(lora)
        if lora_blocks == set(range(WeightFerryConfig.NUM_BLOCKS_BASE)):
            remapped_to_load, count = _remap_key_map(to_load)
            log_info(f"Remapped {count} keys from 28-block base -> 40-block Anima-2.9B")
            return orig_load_lora(lora, remapped_to_load)

        return orig_load_lora(lora, to_load)

    patched_load_lora._weightferry_29b_patched = True
    comfy.lora.load_lora = patched_load_lora
    log_info("Anima 2.9B LoRA block-remapping hook installed")
