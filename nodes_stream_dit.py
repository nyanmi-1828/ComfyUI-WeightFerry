# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
DiT Layer Streaming Node for ComfyUI-WeightFerry
======================================================
Runs DiTs larger than VRAM on 8GB GPUs: Cosmos/Anima transformer blocks are streamed from host RAM (optionally
pinned) to a few GPU slots with asynchronous prefetch (``prefetch_slots``, default 3); a slot is only overwritten
after the compute that last read it has finished. Hooks act only while the owning ModelPatcher's DIFFUSION_MODEL
wrapper runs.

- Host buffers hold each block's weights as it computes with them (HOST_FORMAT_BAKED: LoRA baked, one DMA per block)
  or in the narrower storage dtype with LoRA re-applied on the GPU per forward (HOST_FORMAT_STORAGE, bit-exact with
  ComfyUI's cast path). Outside a forward every parameter is bound to a "rest" tensor that ComfyUI's own unload /
  unpatch paths can use. Scaled fp8 (QuantizedTensor) weights stream their fp8 bytes.
- Tiering (TIERING_RAM / TIERING_AUTO): AUTO splits blocks into pinned RAM and an SSD tier (a cache file, or the
  model's safetensors file itself when its bytes are already the compute bytes) read through a small pinned ring by
  background threads with unbuffered reads; see core/tiers/plan.py: plan_tiers.
- Resident blocks: RESIDENT_MODE_COMFY leaves them to ComfyUI / Dynamic VRAM; RESIDENT_MODE_SELF copies them once per
  load into dedicated GPU buffers so results do not depend on VRAM pressure.
- ``auto_config`` (WEIGHTFERRY_AUTO=1) picks resident_blocks / tiering from the usable VRAM; when everything fits,
  streaming is not used.
- Captured host buffers are cached per model and reused while the cache key is unchanged. Profiling
  (WEIGHTFERRY_PROFILE=1): _StreamState.profile_summary() in streaming/state.py.

Code: streaming/ (dit_node.py and te_node.py are the nodes; capture.py, load_steps.py, auto_config.py,
ssd_uploader.py, state.py, teardown.py). Environment variables: docs/ENVIRONMENT.md.
"""

# The implementation lives in the package streaming/ (split by concern). This module is the public entry
# point: the two nodes and their option values.
from .streaming.common import (  # noqa: F401
    HOST_FORMAT_BAKED, HOST_FORMAT_STORAGE, RESIDENT_MODE_COMFY, RESIDENT_MODE_SELF, TIERING_AUTO, TIERING_RAM,
)
from .streaming.dit_node import WeightFerryDiT  # noqa: F401
from .streaming.te_node import WeightFerryTextEncoder  # noqa: F401

__all__ = ["WeightFerryDiT", "WeightFerryTextEncoder", "HOST_FORMAT_BAKED", "HOST_FORMAT_STORAGE",
           "RESIDENT_MODE_COMFY", "RESIDENT_MODE_SELF", "TIERING_AUTO", "TIERING_RAM"]
