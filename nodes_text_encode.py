# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
CLIP Text Encode with the text encoder run layer by layer from system RAM, for text encoders larger than the VRAM.

* WeightFerryCLIPTextEncode -- "CLIP Text Encode (Streamed LLM)", any text encoder. Anima's (Qwen3 0.6B + the LLM
  adapter's t5xxl ids) runs on the streamed LLM engine (core/llm_stream.py, ~1.7-1.9x faster than ComfyUI's code on
  Qwen2.5-VL 7B); every other one runs ComfyUI's own encode with its layers streamed (the Text Encoder Layer Streaming
  engine, streaming/te_node.py). Both give ComfyUI's CLIP Text Encode conditioning bit for bit. Optional disk_cache
  (off by default): the result is kept on disk and reused without loading the text encoder (core/te_cache.py).
* WeightFerryCLIPTextEncodeFile -- "... from File", Anima only: the engine reads a text encoder file instead of the
  CLIP's weights (quantized text encoders, whose in-memory weights the engine cannot copy).
"""

from typing import Dict, Any, Tuple
import os
import torch
import folder_paths

from .core.llm_stream import StreamedLLMEngine
from .core import te_cache
from .utils.logging import log_info


# Module-level cache for engine instances
_ENGINE_CACHE: Dict[str, StreamedLLMEngine] = {}

FROM_CLIP = "(from clip)"
_EMBED = "model.embed_tokens.weight"


def _clip_llm_tensors(clip):
    """The LLM's weights in the CLIP object, keyed as in the text encoder file ("model.layers.<n>...")."""
    sd = clip.cond_stage_model.state_dict()
    found = [k for k in sd if k == _EMBED or k.endswith("." + _EMBED)]
    if not found:
        raise ValueError("WeightFerryCLIPTextEncodeFile: the CLIP input has no LLM text encoder (no model.embed_tokens); "
                         "choose the text encoder file in model_name")
    prefix = found[0][:-len(_EMBED)]
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def _compute_dtype() -> torch.dtype:
    import comfy.model_management
    dtype = comfy.model_management.text_encoder_dtype(comfy.model_management.get_torch_device())
    return dtype if dtype in (torch.float32, torch.bfloat16) else torch.bfloat16


class WeightFerryCLIPTextEncodeFile:
    """
    CLIP Text Encode (Streamed LLM, from File).
    Encodes the prompt with an LLM text encoder streamed layer by layer from system RAM (double-buffered prefetch, two
    layers on the GPU). For Anima use the Qwen3 0.6B text encoder.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Scan available text encoder / clip models
        models = []
        for folder_name in ["text_encoders", "clip"]:
            try:
                models.extend(folder_paths.get_filename_list(folder_name))
            except KeyError:  # this ComfyUI has no such model folder
                pass
        
        # Sort and ensure uniqueness; "(from clip)" first (the default)
        models = [FROM_CLIP] + sorted(list(set(models)))

        return {
            "required": {
                "model_name": (models, {
                    "default": FROM_CLIP,
                    "tooltip": "(from clip): the text encoder the CLIP input already holds (nothing read twice). A file: "
                               "stream that file instead (needed for quantized text encoders)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Prompt text to encode"
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Anima CLIP object (required): its tokenizers (the Qwen3 ids and the t5xxl ids for the LLM "
                               "adapter) and, with model_name (from clip), its weights"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "WeightFerry/streaming"
    TITLE = "CLIP Text Encode (Streamed LLM, from File)"

    def encode(self, model_name: str, prompt: str, clip: Any = None) -> Tuple[Any]:
        if clip is None or not hasattr(clip, "tokenize"):
            raise ValueError(
                "WeightFerryCLIPTextEncodeFile: connect the 'clip' input to an Anima text encoder (CLIPLoader). "
                "Its tokenizer provides both the Qwen3 ids and the t5xxl ids required by the LLMAdapter."
            )

        # 1. Get or create the streaming engine. Compute dtype follows ComfyUI's text encoder dtype
        #    (--fp32-text-enc -> float32), so the result matches CLIPTextEncode bit for bit.
        compute_dtype = _compute_dtype()
        if model_name == FROM_CLIP:
            engine_key = f"clip:{id(clip.cond_stage_model)}|{compute_dtype}"
        else:
            model_path = folder_paths.get_full_path("text_encoders", model_name)
            if model_path is None:
                model_path = folder_paths.get_full_path("clip", model_name)
            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f"Text encoder model '{model_name}' not found in text_encoders or clip folders.")
            engine_key = f"{model_name}|{compute_dtype}"
        if engine_key not in _ENGINE_CACHE:
            log_info(f"Streamed LLM: setting up the engine for {model_name} ({compute_dtype})...")
            _ENGINE_CACHE.clear()  # keep at most one engine (each holds pinned staging buffers)
            if model_name == FROM_CLIP:
                _ENGINE_CACHE[engine_key] = StreamedLLMEngine(
                    dtype=compute_dtype, tensors=_clip_llm_tensors(clip))
            else:
                _ENGINE_CACHE[engine_key] = StreamedLLMEngine(model_path, dtype=compute_dtype)

        engine = _ENGINE_CACHE[engine_key]

        # 2. Tokenize prompt
        tokens = clip.tokenize(prompt)
        if isinstance(tokens, dict) and "qwen3_06b" in tokens:
            raw_tokens = tokens["qwen3_06b"]
        elif isinstance(tokens, dict):
            raw_tokens = tokens[next(iter(tokens))]
        else:
            raw_tokens = tokens
        input_ids = torch.tensor([[t[0] for t in raw_tokens[0]]], dtype=torch.long)

        # Anima's LLMAdapter needs the T5 target token ids; without them the conditioning is invalid
        if not (isinstance(tokens, dict) and "t5xxl" in tokens and len(tokens["t5xxl"]) > 0):
            raise ValueError(
                "WeightFerryCLIPTextEncodeFile: t5xxl tokens are missing. Connect the 'clip' input to an Anima "
                "text encoder (CLIPLoader) so the LLMAdapter receives valid t5xxl_ids."
            )
        t5_tokens = tokens["t5xxl"][0]
        t5xxl_ids = torch.tensor([t[0] for t in t5_tokens], dtype=torch.int32)
        t5xxl_weights = torch.tensor([t[1] for t in t5_tokens], dtype=torch.float32)

        log_info(f"Streamed LLM: encoding the prompt ({input_ids.shape[1]} tokens) across {engine.num_layers} layers...")
        torch.cuda.synchronize(engine.device)
        base_alloc = torch.cuda.memory_allocated(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        out_embed = engine.encode_tokens(input_ids).float()
        peak_delta = torch.cuda.max_memory_allocated(engine.device) - base_alloc

        seq_len = out_embed.shape[1]

        conditioning = [[
            out_embed,
            {
                "pooled_output": None,
                "attention_mask": torch.ones((1, seq_len), dtype=torch.int64),
                "t5xxl_ids": t5xxl_ids,
                "t5xxl_weights": t5xxl_weights,
            }
        ]]
        log_info(
            f"Streamed LLM: encoding done. Shape={out_embed.shape}, "
            f"Peak VRAM during encode (above baseline)={peak_delta/(1024**2):.1f}MB"
        )
        return (conditioning,)


def _is_anima_clip(clip) -> bool:
    try:
        from comfy.text_encoders.anima import AnimaTEModel
    except ImportError:
        return False
    return isinstance(getattr(clip, "cond_stage_model", None), AnimaTEModel)


_STREAMED_CLIP: Dict[int, Any] = {}  # id(CLIP) -> (CLIP, its layer-streaming clone); one entry


def _streamed_clip(clip):
    """The CLIP with its text encoder's layers streamed (Text Encoder Layer Streaming, auto_config), made once per CLIP."""
    hit = _STREAMED_CLIP.get(id(clip))
    if hit is not None and hit[0] is clip:
        return hit[1]
    from .streaming.te_node import WeightFerryTextEncoder
    (streamed,) = WeightFerryTextEncoder().apply(clip)
    _STREAMED_CLIP.clear()
    _STREAMED_CLIP[id(clip)] = (clip, streamed)
    return streamed


class WeightFerryCLIPTextEncode:
    """CLIP Text Encode with the text encoder run layer by layer from system RAM (no model file to choose). Same inputs
    and output as ComfyUI's CLIPTextEncode, the same conditioning bit for bit, for any text encoder (module docstring)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": ("CLIP", {"lazy": True,
                                  "tooltip": "The CLIP model used for encoding the text; its layers are streamed from "
                                             "system RAM when it does not fit in the VRAM."}),
            },
            "optional": {
                "disk_cache": ("BOOLEAN", {"default": False, "tooltip":
                               "Keep the result on disk and reuse it for the same text and the same CLIP (also after a "
                               "restart). On a hit the CLIP is not requested: if nothing else uses it, the text "
                               "encoder is never loaded. Size cap: WEIGHTFERRY_TE_CACHE_GB."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "WeightFerry/streaming"
    TITLE = "CLIP Text Encode (Streamed LLM)"

    def _cached(self, text, disk_cache, prompt, unique_id):
        """(key, conditioning or None) for disk_cache; the hit found by check_lazy_status is reused."""
        if not disk_cache:
            return None, None
        key = te_cache.make_key(text, prompt, unique_id)
        if key is None:
            return None, None
        pending = getattr(self, "_pending", None)
        if pending is not None and pending[0] == key:
            return key, pending[1]
        cond = te_cache.load(key)
        self._pending = (key, cond) if cond is not None else None
        return key, cond

    def check_lazy_status(self, text, clip=None, disk_cache=False, prompt=None, unique_id=None):
        if clip is not None:
            return []
        _, cond = self._cached(text, disk_cache, prompt, unique_id)
        return [] if cond is not None else ["clip"]

    def encode(self, text: str, clip: Any = None, disk_cache: bool = False, prompt=None, unique_id=None) -> Tuple[Any]:
        key, cond = self._cached(text, disk_cache, prompt, unique_id)
        self._pending = None
        if cond is not None:
            log_info(f"Text encode cache: hit {key[:12]}" + ("" if clip is not None else "; the CLIP was not needed."))
            return (cond,)
        if clip is None:
            raise ValueError("CLIP Text Encode (Streamed LLM): the clip input is not connected")
        if _is_anima_clip(clip):
            out = WeightFerryCLIPTextEncodeFile().encode(FROM_CLIP, text, clip)
        else:
            import nodes
            out = nodes.CLIPTextEncode().encode(_streamed_clip(clip), text)
        if key is not None:
            te_cache.store(key, out[0])
            log_info(f"Text encode cache: stored {key[:12]}.")
        elif disk_cache:
            log_info("Text encode cache: no key for this node (workflow not available); not cached.")
        return out
