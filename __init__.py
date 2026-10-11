# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
ComfyUI-WeightFerry (https://github.com/nyanmi-1828/ComfyUI-WeightFerry)

Nodes for Anima / Cosmos-Predict2 and other DiT models in ComfyUI:
- layer streaming of diffusion models and text encoders larger than the VRAM (bit-exact with ComfyUI)
- the streamed LLM text encoder (an LLM text encoder run layer by layer from system RAM)
- TaylorSeer step cache, CFG early stop, the ReAL sampler and the TORS scheduler
- upscaling helpers (half-precision upscale model, Ultimate SD Upscale+ with tile skip and high-pass noise, USDU tile
  calculator, RoPE scale) and an exact-tile VAE decode
At import: fused element-wise kernels for the Cosmos / Anima block (bit-exact), the 28 -> 40 block LoRA remapping for
Anima 2.9B and a 1 ms Windows timer. See README.md.
"""

import comfy.samplers
from .utils.logging import log_info, log_warn
from .core import setup_low_level_optimizations, install_lora_remap_patch
from .samplers import (
    WeightFerryTORSScheduler,
    tors_scheduler_handler,
    WeightFerryReALSampler,
    sample_real,
)
from .enhancers import WeightFerryTaylorSeer, WeightFerryCFGEarlyStop
from .enhancers import WeightFerryRoPEScale  # positions for large images (no-tiles upscale), off by default
# prototypes: measured and not adopted, off by default (enhancers/prototypes.py)
from .enhancers import WeightFerryCFGCache, WeightFerryCoarseGlobalAttention, WeightFerryTGATE
from .utils import WeightFerryUSDUTileCalculator
from .utils import WeightFerryUpscaleWithModelHalf  # float32 by default
from .usdu import WeightFerryUltimateSDUpscalePlus  # bundled Ultimate SD Upscale; both additions off at 0
from .utils import WeightFerryVAEDecodeExactTiles  # Wan 2.1 VAE decode, tiles with an exact halo
from .nodes_text_encode import WeightFerryCLIPTextEncodeFile, WeightFerryCLIPTextEncode
from .nodes_stream_dit import WeightFerryDiT, WeightFerryTextEncoder
from .samplers.sampler_choice import WeightFerrySamplerChoice


# =========================================================================
# 1. Low-Level System & LoRA Remap Initialization
# =========================================================================
setup_low_level_optimizations()
install_lora_remap_patch()


# =========================================================================
# 2. Global ComfyUI Scheduler & Sampler Injections
# =========================================================================
def _register_name(owner, attr, name):
    """Add ``name`` to the list or tuple ``owner.<attr>`` if it is missing (ComfyUI keeps KSampler's choices in either)."""
    seq = getattr(owner, attr, None)
    if seq is None or name in seq:
        return
    if isinstance(seq, list):
        seq.append(name)
    elif isinstance(seq, tuple):
        setattr(owner, attr, seq + (name,))


# (A) "tors" scheduler: handler + the scheduler lists of comfy.samplers and KSampler
try:
    if hasattr(comfy.samplers, "SCHEDULER_HANDLERS"):
        comfy.samplers.SCHEDULER_HANDLERS["tors"] = comfy.samplers.SchedulerHandler(
            tors_scheduler_handler, use_ms=True
        )
        _register_name(comfy.samplers, "SCHEDULER_NAMES", "tors")
        _register_name(comfy.samplers.KSAMPLER, "SCHEDULERS", "tors")
    log_info("Injected 'tors' scheduler into ComfyUI KSampler")
except Exception as e:  # noqa: BLE001 -- at import: never stop ComfyUI from loading the nodes
    log_warn(f"Failed to inject 'tors' scheduler: {e}")

# (B) "real" sampler: k-diffusion function + the sampler lists of comfy.samplers and KSampler
try:
    import comfy.k_diffusion.sampling as k_diff_sampling
    k_diff_sampling.sample_real = sample_real

    if hasattr(comfy.samplers, "KSAMPLER_NAMES"):
        _register_name(comfy.samplers, "KSAMPLER_NAMES", "real")
        _register_name(comfy.samplers, "SAMPLER_NAMES", "real")
        _register_name(comfy.samplers.KSAMPLER, "SAMPLERS", "real")
    log_info("Injected 'real' sampler into ComfyUI KSampler")
except Exception as e:  # noqa: BLE001 -- at import: never stop ComfyUI from loading the nodes
    log_warn(f"Failed to inject 'real' sampler: {e}")

# (C) Fused element-wise work in the Cosmos / Anima DiT block, bit-exact (WEIGHTFERRY_FUSED_BLOCK=0 turns it off)
try:
    from .enhancers import fused_block as _fused_block
    _fused_block.install()
except Exception as e:  # noqa: BLE001 -- at import: never stop ComfyUI from loading the nodes
    log_warn(f"Anima fused block not installed: {e}")

# (C2) ComfyUI-GGUF's legacy-quant dequantisation in one Triton kernel, bit-exact (WEIGHTFERRY_GGUF_TRITON=0 turns it
# off). ComfyUI loads custom nodes in name order, so ComfyUI-GGUF is already imported here
try:
    from .enhancers import gguf_dequant as _gguf_dequant
    _gguf_dequant.install()
except Exception as e:  # noqa: BLE001 -- at import: never stop ComfyUI from loading the nodes
    log_warn(f"GGUF Triton dequant not installed: {e}")



# =========================================================================
# 3. Node class mappings
# =========================================================================
NODE_CLASS_MAPPINGS = {
    "WeightFerryCFGEarlyStop": WeightFerryCFGEarlyStop,
    "WeightFerryTaylorSeer": WeightFerryTaylorSeer,
    "WeightFerryUSDUTileCalculator": WeightFerryUSDUTileCalculator,
    "WeightFerryTORSScheduler": WeightFerryTORSScheduler,
    "WeightFerryReALSampler": WeightFerryReALSampler,
    "WeightFerryCLIPTextEncodeFile": WeightFerryCLIPTextEncodeFile,
    "WeightFerryCLIPTextEncode": WeightFerryCLIPTextEncode,
    "WeightFerryDiT": WeightFerryDiT,
    "WeightFerryTextEncoder": WeightFerryTextEncoder,
    "WeightFerrySamplerChoice": WeightFerrySamplerChoice,
    "WeightFerryRoPEScale": WeightFerryRoPEScale,
    "WeightFerryCFGCache": WeightFerryCFGCache,
    "WeightFerryCoarseGlobalAttention": WeightFerryCoarseGlobalAttention,
    "WeightFerryUpscaleWithModelHalf": WeightFerryUpscaleWithModelHalf,
    "WeightFerryUltimateSDUpscalePlus": WeightFerryUltimateSDUpscalePlus,
    "WeightFerryVAEDecodeExactTiles": WeightFerryVAEDecodeExactTiles,
    "WeightFerryTGATE": WeightFerryTGATE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeightFerryCFGEarlyStop": "CFG Early-Stop",
    "WeightFerryTaylorSeer": "TaylorSeer",
    "WeightFerryUSDUTileCalculator": "USDU Tile Calculator",
    "WeightFerryTORSScheduler": "TORS Scheduler",
    "WeightFerryReALSampler": "ReAL Sampler",
    "WeightFerryCLIPTextEncodeFile": "CLIP Text Encode (Streamed LLM, from File)",
    "WeightFerryCLIPTextEncode": WeightFerryCLIPTextEncode.TITLE,
    "WeightFerryDiT": "⚡ DiT Layer Streaming (low VRAM)",
    "WeightFerryTextEncoder": "⚡ Text Encoder Layer Streaming (low VRAM)",
    "WeightFerrySamplerChoice": WeightFerrySamplerChoice.TITLE,
    "WeightFerryRoPEScale": "RoPE Scale (large images)",
    "WeightFerryUpscaleWithModelHalf": "Upscale Image (using Model, half precision)",
    "WeightFerryUltimateSDUpscalePlus": "Ultimate SD Upscale+ (tile skip, high-pass noise)",
    "WeightFerryVAEDecodeExactTiles": "VAE Decode (Exact Tiles, low VRAM)",
    # prototypes: measured and not adopted, off by default (enhancers/prototypes.py)
    "WeightFerryCFGCache": "[prototype] CFG Negative Cache (not adopted)",
    "WeightFerryCoarseGlobalAttention": "[prototype] Coarse Global Attention (not adopted)",
    "WeightFerryTGATE": "[prototype] TGATE cross-attention reuse (not adopted)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
log_info("WeightFerry: nodes loaded")
