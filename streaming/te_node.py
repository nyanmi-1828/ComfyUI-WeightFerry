# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
The WeightFerryTextEncoder node."""

from ..utils.logging import log_info, log_warn
from .common import (HOST_FORMAT_BAKED, RESIDENT_MODE_COMFY, TIERING_AUTO, TIERING_RAM, _block_list_attr, _blocks,
                     _param_nbytes)
from .dit_node import WeightFerryDiT


class WeightFerryTextEncoder:
    """
    Layer streaming for a text encoder: the same engine as the DiT streaming node (pinned RAM / SSD tiers,
    model-file reads, scaled fp8 / QuantizedTensor weights, chunk streaming for very small VRAM, VRAM-overflow
    check), applied to the transformer layer list of ComfyUI's own text encoder (T5 encoder.block, Llama / Qwen
    model.layers, ...). Large text encoders (7B-9B) can then run with little VRAM, and the result is ComfyUI's own
    computation (the weights are only moved, not re-implemented as in the streamed LLM engine).
    Streaming is active only when the returned CLIP encodes (other clones of the same CLIP are unaffected).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "enabled": ("BOOLEAN", {"default": True}),
                "resident_layers": ("INT", {"default": 2, "min": 1, "max": 128,
                                            "tooltip": "Layers kept on the GPU (ComfyUI-managed); the rest stream"}),
                "tiering": ([TIERING_AUTO, TIERING_RAM], {"default": TIERING_AUTO}),
                "ram_budget_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5,
                                            "tooltip": "Pinned RAM for streamed layers (0 = automatic); the rest is "
                                                       "read from the model file / SSD"}),
                "auto_config": ("BOOLEAN", {"default": True, "label_on": "Auto (VRAM / RAM / SSD)",
                                            "label_off": "Manual"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "WeightFerry/streaming"

    @staticmethod
    def _find_layer_stack(model):
        """(path, module) of the module holding the largest transformer layer list, or (None, None)."""
        best, best_bytes = (None, None), 0
        for name, mod in model.named_modules():
            attr = _block_list_attr(mod)
            if attr is None or attr not in ("layers", "block", "blocks"):
                continue
            lst = getattr(mod, attr)
            if len(lst) < 4:
                continue
            nbytes = sum(_param_nbytes(p) for p in lst.parameters())
            if nbytes > best_bytes:
                best, best_bytes = (name, mod), nbytes
        return best

    def apply(self, clip, enabled=True, resident_layers=2, tiering=TIERING_AUTO, ram_budget_gb=0.0, auto_config=True):
        if not enabled:
            return (clip,)
        c = clip.clone()
        path, root = self._find_layer_stack(c.cond_stage_model)
        if root is None:
            log_warn("Text Encoder Layer Streaming: no transformer layer list found in this text encoder; "
                     "passing the CLIP unchanged.")
            return (clip,)
        prefix = (path + ".") if path else ""
        (patcher,) = WeightFerryDiT().apply_streaming(
            model=c.patcher, enabled=True, pin_memory=True, buffer_mode="Double Buffer (Async Prefetch, Recommended)",
            resident_blocks=resident_layers, sage_attention=False, host_format=HOST_FORMAT_BAKED, prefetch_slots=3,
            resident_mode=RESIDENT_MODE_COMFY, tiering=tiering, ram_budget_gb=ram_budget_gb, auto_config=auto_config,
            _root=root, _key_prefix=prefix, _text_encoder=True)
        owner = getattr(patcher, "_weightferry_stream_owner", None)
        log_info(f"Text Encoder Layer Streaming: streaming {path}.{_block_list_attr(root)} "
                 f"({len(_blocks(root))} layers) of {type(c.cond_stage_model).__name__}.")

        # Mark the calls made through this CLIP (the layer stack is shared with every clone)
        for name in ("encode_from_tokens", "encode_from_tokens_scheduled"):
            orig = getattr(c, name, None)
            if orig is None:
                continue

            def make(orig_fn):
                def marked(*args, **kwargs):
                    prev = getattr(root, "_weightferry_te_caller", None)
                    root._weightferry_te_caller = owner
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        root._weightferry_te_caller = prev
                return marked
            setattr(c, name, make(orig))
        return (c,)
