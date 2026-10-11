# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Portions adapted from CocyNoric/ComfyUI-Anima-TeaCache (Apache-2.0), Shenyi-Z/TaylorSeer (GPL-3.0) and
# ComfyUI comfy/ldm/cosmos/predict2.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
TaylorSeer cache for Anima (adaptive / step modes).

Provenance: the base structure, the polynomial coefficients (ANIMA_COEFFICIENTS) and the forward
were adapted from CocyNoric/ComfyUI-Anima-TeaCache (Apache-2.0); the Taylor factor update / evaluation follows the
official TaylorSeer implementation (Shenyi-Z/TaylorSeer, GPL-3.0; paper arXiv:2503.06923, ICCV 2025); the forward
mirrors ComfyUI's MiniTrainDIT._forward (comfy/ldm/cosmos/predict2.py, GPL-3.0).
"""

import logging
import math
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import comfy.model_management as model_management
import comfy.patcher_extension as patcher_extension
from ..core import env as _env
from ..core import vram_monitor

# Polynomial coefficients from CocyNoric/ComfyUI-Anima-TeaCache (Apache-2.0, THIRD_PARTY_NOTICES.md)
ANIMA_COEFFICIENTS = (
    16.440769544691957,
    -21.14123745515793,
    13.006701290921894,
    -1.6731977662609427,
    0.08632681059623949,
)
_STEP_INFO_KEY = "weightferry_taylorseer_step_info"
_WRAPPER_KEY = "weightferry_taylorseer"
_DEBUG = _env.get("WEIGHTFERRY_TAYLOR_DEBUG") == "1"  # one log line per decision
# Off by default (0): keep the cache (inputs, residuals, factors) in RAM instead of cache_device
# for images of at least this many tokens -- the 2x upscale without tiles (20608 tokens) holds ~0.5-0.7 GB there
# (cond + uncond), which overflowed a 2 GB GPU; the 896x1472 base (5152 tokens) stays on cache_device
_CACHE_CPU_MIN_TOKENS = int(_env.get("WEIGHTFERRY_TAYLOR_CACHE_CPU_MIN_TOKENS"))
# Unset (0): automatic for images of at least this many tokens -- the cache goes to RAM when the VRAM is short
_AUTO_CPU_MIN_TOKENS = 8192
def poly1d(coefficients, value):
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


@dataclass
class TaylorCacheEntry:
    previous_modulated_input: torch.Tensor | None = None
    accumulated_distance: float = 0.0
    signature: tuple[Any, ...] | None = None

    # TaylorSeer derivative factors: order (0, 1, 2) -> tensor factor
    # 0: R(t_0), 1: R'(t_0), 2: R''(t_0)
    factors: Dict[int, torch.Tensor] = field(default_factory=dict)
    current_activated_step: int = 0


@dataclass
class CacheDecision:
    should_calc: bool
    cacheable: bool
    keys: list[Any]
    slices: list[slice]


class TaylorSeerRuntime:
    def __init__(
        self,
        mode: str,
        rel_l1_thresh: float,
        step_interval: int,
        max_order: int,
        first_enhance: int,
        last_enhance: int,
        cache_device: torch.device,
        max_predict_steps: int = 0,
    ):
        self.mode = mode
        # 0 = no limit (as upstream TaylorSeer); n = compute when the prediction would reach more than n steps past the
        # last computed step (adaptive samplers such as ReAL can jump several steps in one call)
        self.max_predict_steps = max_predict_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.step_interval = step_interval
        self.max_order = max_order
        self.first_enhance = first_enhance
        self.last_enhance = last_enhance
        self.cache_device = cache_device
        self.entries: Dict[Any, TaylorCacheEntry] = {}
        self._auto_devices = {}  # image size (h, w in tokens) -> cache device chosen automatically for this sampling
        self.schedule_key = None
        self.last_step_index = None

    def _auto_cache_device(self, modulated_input):
        """Large images (never the 896x1472 base): the cache (the previous input and up to two factors for the whole
        batch) goes to RAM when it would take more than half of the free VRAM. Decided once per sampling and image size,
        not per batch size: the cached tensors must stay on one device when CFG early stop drops the negative half way
        (the free VRAM is larger then and would choose the GPU)."""
        key = tuple(modulated_input.shape[2:4])
        dev = self._auto_devices.get(key)
        if dev is None:
            need = 3 * modulated_input.numel() * modulated_input.element_size()
            free = vram_monitor.effective_free_vram(modulated_input.device)
            dev = torch.device("cpu") if need > 0.5 * free else self.cache_device
            self._auto_devices[key] = dev
            if dev.type == "cpu":
                logging.info("[WeightFerryTaylorSeer] cache in RAM for this %s-token sampling (needs about %d MB, %d MB of "
                             "VRAM free)", key[0] * key[1], need // 2 ** 20, max(0, free) // 2 ** 20)
        return dev

    def reset(self):
        self.entries.clear()
        self._auto_devices = {}
        self.schedule_key = None
        self.last_step_index = None

    def diffusion_model_wrapper(self, executor, *args, **kwargs):
        diffusion_model = executor.class_obj
        transformer_options = kwargs.get("transformer_options", {})
        step_info = self._build_step_info(
            transformer_options.get("sigmas"),
            transformer_options.get("sample_sigmas"),
        )
        if step_info is not None:
            transformer_options[_STEP_INFO_KEY] = step_info
        executor.original = lambda *inner_args, **inner_kwargs: self.forward(
            diffusion_model, *inner_args, **inner_kwargs
        )
        return executor(*args, **kwargs)
    def _build_step_info(self, current_sigmas, sample_sigmas):
        if not torch.is_tensor(current_sigmas) or not torch.is_tensor(sample_sigmas):
            return None

        # The schedule is copied to the CPU once per sampling run (the same tensor object comes with every call) instead of
        # on every call; the step is then found on the CPU with the same arithmetic (same dtype, argmin takes the first
        # minimum on both devices), so only the current sigma is read from the GPU
        cached = getattr(self, "_sigma_cache", None)
        if cached is None or cached[0]() is not sample_sigmas:
            sigmas = sample_sigmas.detach().flatten().cpu()
            cached = (weakref.ref(sample_sigmas), sigmas, tuple(sigmas.float().tolist()))
            self._sigma_cache = cached
        _ref, sigmas, schedule_key = cached
        total_steps = sigmas.numel() - 1
        if total_steps <= 0:
            return None

        current_sigma = current_sigmas.detach().flatten()[0].to(device="cpu", dtype=sigmas.dtype)
        step_index = int((sigmas[:-1] - current_sigma).abs().argmin())
        percent = step_index / total_steps
        return {
            "index": step_index,
            "total": total_steps,
            "percent": percent,
            "schedule_key": schedule_key,
        }

    def _split_layout(self, tensor, transformer_options):
        labels = transformer_options.get("cond_or_uncond")
        uuids = transformer_options.get("uuids")
        if not labels:
            return [0], [slice(0, tensor.shape[0])]
        if tensor.shape[0] % len(labels) != 0:
            return None

        chunk_size = tensor.shape[0] // len(labels)
        occurrences = {}
        keys = []
        slices = []
        for index, label in enumerate(labels):
            uuid = uuids[index] if uuids and len(uuids) == len(labels) else None
            identity = (label, uuid)
            occurrence = occurrences.get(identity, 0)
            occurrences[identity] = occurrence + 1
            keys.append((label, uuid, occurrence))
            slices.append(slice(index * chunk_size, (index + 1) * chunk_size))
        return keys, slices

    def _call_device(self, modulated_input):
        """The device of this call's cache: cache_device, or RAM for images of at least _CACHE_CPU_MIN_TOKENS tokens
        when that is set, else (unset) automatically from _AUTO_CPU_MIN_TOKENS tokens when the VRAM is short."""
        tokens = modulated_input.shape[2] * modulated_input.shape[3] if modulated_input.dim() == 5 else 0
        if _CACHE_CPU_MIN_TOKENS and tokens >= _CACHE_CPU_MIN_TOKENS:
            return torch.device("cpu")
        if not _CACHE_CPU_MIN_TOKENS and tokens >= _AUTO_CPU_MIN_TOKENS and self.cache_device.type == "cuda":
            return self._auto_cache_device(modulated_input)
        return self.cache_device

    @staticmethod
    def _relative_l1_distances(entries, current_inputs):
        """Per entry [denominator, relative L1 distance] of the current input to the previous one (None for an entry
        without a previous input or factors), computed as tensors first and read in one transfer."""
        def _distance(previous, current):
            denominator = previous.abs().mean()
            return torch.stack([denominator.float(), ((current - previous).abs().mean() / denominator).float()])

        pending = [_distance(e.previous_modulated_input, cur)
                   if e.previous_modulated_input is not None and len(e.factors) else None
                   for e, cur in zip(entries, current_inputs)]
        stacked = [p for p in pending if p is not None]
        values = iter(torch.stack(stacked).tolist() if stacked else [])
        return [next(values) if p is not None else None for p in pending]

    def decide(self, modulated_input, transformer_options):
        step_info = transformer_options.get(_STEP_INFO_KEY)
        layout = self._split_layout(modulated_input, transformer_options)
        if step_info is None or layout is None:
            self.reset()
            return CacheDecision(True, False, [], [])

        step_index = step_info["index"]
        total_steps = step_info["total"]
        schedule_key = step_info["schedule_key"]

        if (
            self.schedule_key != schedule_key
            or (step_index == 0 and self.last_step_index != 0)
            or (self.last_step_index is not None and step_index < self.last_step_index)
        ):
            self.reset()
            self.schedule_key = schedule_key
        self.last_step_index = step_index

        keys, slices = layout
        self._call_dev = self._call_device(modulated_input)
        is_first = step_index < self.first_enhance
        is_last = step_index >= (total_steps - self.last_enhance)
        force_full = is_first or is_last

        current_inputs = []
        entries = []
        for key, section in zip(keys, slices):
            current = modulated_input[section].detach().to(self._call_dev)
            current_inputs.append(current)
            entry = self.entries.setdefault(key, TaylorCacheEntry())
            entries.append(entry)
            signature = (tuple(current.shape), current.dtype, tuple(modulated_input[section].stride()))

            if entry.signature != signature:
                entry.previous_modulated_input = None
                entry.accumulated_distance = 0.0
                entry.factors.clear()
                entry.signature = signature
                force_full = True

        # Adaptive mode: the relative L1 distances of all keys are computed as tensors first and read in one transfer
        # (one GPU sync per call instead of up to three per key: the waits broke the overlap of the streamed weight
        # transfers). Same values: the denominator is read as float32 (exact for bf16 / fp16)
        distances = None
        if self.mode != "step" and not force_full:
            distances = self._relative_l1_distances(entries, current_inputs)

        for k, (entry, current) in enumerate(zip(entries, current_inputs)):
            previous = entry.previous_modulated_input
            if previous is None or len(entry.factors) == 0:
                force_full = True
            elif self.max_predict_steps > 0 and step_index - entry.current_activated_step > self.max_predict_steps:
                force_full = True
            elif not force_full:
                if self.mode == "step":
                    # Fixed step interval mode
                    if (step_index - self.first_enhance) % self.step_interval != 0:
                        # Skip and predict
                        pass
                    else:
                        force_full = True
                else:
                    # Adaptive mode using relative L1 polynomial distance
                    denominator, relative_l1 = distances[k]
                    if not math.isfinite(denominator) or denominator <= 0.0:
                        force_full = True
                    else:
                        scaled = poly1d(ANIMA_COEFFICIENTS, relative_l1)
                        if not math.isfinite(scaled):
                            force_full = True
                        else:
                            entry.accumulated_distance += scaled
                            if entry.accumulated_distance >= self.rel_l1_thresh:
                                force_full = True

        for key, current in zip(keys, current_inputs):
            self.entries[key].previous_modulated_input = current

        if force_full:
            for key in keys:
                self.entries[key].accumulated_distance = 0.0

        if _DEBUG:
            logging.info("[WeightFerryTaylorSeer] step %d/%d keys %s: %s, accumulated %s", step_index, total_steps,
                         [(k[0], step_index - self.entries[k].current_activated_step) for k in keys],
                         "compute" if force_full else "skip",
                         [round(self.entries[k].accumulated_distance, 4) for k in keys])
        return CacheDecision(force_full, True, keys, slices)

    def _taylor_update(self, factors, res, diff_dist):
        updated_factors = {0: res}
        for i in range(self.max_order):
            if i in factors:
                updated_factors[i + 1] = (updated_factors[i] - factors[i]) / diff_dist
            else:
                break
        return updated_factors

    @staticmethod
    def _taylor_eval(factors, dt):
        pred = torch.zeros_like(factors[0])
        for i, factor in factors.items():
            pred += (1.0 / math.factorial(i)) * factor * (dt ** i)
        return pred

    def commit_residual(self, decision: CacheDecision, residual: torch.Tensor, step_index: int):
        if not decision.cacheable:
            return
        for key, section in zip(decision.keys, decision.slices):
            entry = self.entries[key]
            res = residual[section].detach().to(getattr(self, "_call_dev", self.cache_device))

            diff_dist = step_index - entry.current_activated_step
            if diff_dist <= 0:
                diff_dist = 1

            entry.factors = self._taylor_update(entry.factors, res, diff_dist)
            entry.current_activated_step = step_index

    def predict_residual(self, decision: CacheDecision, hidden_states: torch.Tensor, step_index: int) -> torch.Tensor:
        if all(len(self.entries[k].factors) for k in decision.keys):
            # the Taylor prediction and the addition in one pass per batch item, the same bits (fused_block.taylor_predict_add)
            from .fused_block import taylor_predict_add
            out = taylor_predict_add(
                hidden_states, decision.slices, [self.entries[k].factors for k in decision.keys],
                [step_index - self.entries[k].current_activated_step for k in decision.keys])
            if out is not None:
                return out
        residuals = []
        for key in decision.keys:
            entry = self.entries[key]
            if len(entry.factors) == 0:
                raise RuntimeError("TaylorSeer factors are not initialized!")

            pred = self._taylor_eval(entry.factors, step_index - entry.current_activated_step)
            residuals.append(pred.to(device=hidden_states.device, dtype=hidden_states.dtype))
        return hidden_states + torch.cat(residuals, dim=0)

    def _first_block_modulated_input(
        self,
        diffusion_model,
        hidden_states,
        timestep_embedding,
        adaln_lora,
        extra_pos_emb,
    ):
        block = diffusion_model.blocks[0]
        block_input = hidden_states
        if extra_pos_emb is not None:
            block_input = block_input + extra_pos_emb

        modulation = block.adaln_modulation_self_attn(timestep_embedding)
        if block.use_adaln_lora:
            modulation = modulation + adaln_lora
        shift, scale, _ = modulation.chunk(3, dim=-1)
        shift = shift[:, :, None, None, :]
        scale = scale[:, :, None, None, :]
        # the same values in one pass (LayerNorm + scale-and-shift of the fused block, bit-exact: 1.2 -> about 0.5 ms
        # per step at 896x1472 with cond + uncond)
        from .fused_block import ln_modulate
        return ln_modulate(block.layer_norm_self_attn, block_input, scale, shift)

    @staticmethod
    def _embed(diffusion_model, x, timesteps, context, fps, padding_mask):
        """MiniTrainDIT._forward up to the blocks: the padded input, the token sequence with its RoPE and extra
        positional embeddings, and the timestep embedding (+ AdaLN-LoRA); sets the model's affline attributes."""
        import comfy.ldm.common_dit

        x = comfy.ldm.common_dit.pad_to_patch_size(
            x,
            (
                diffusion_model.patch_temporal,
                diffusion_model.patch_spatial,
                diffusion_model.patch_spatial,
            ),
        )
        hidden_states, rope_embedding, extra_pos_emb = diffusion_model.prepare_embedded_sequence(
            x, fps=fps, padding_mask=padding_mask
        )

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        sinusoidal_timestep = diffusion_model.t_embedder[0](timesteps).to(hidden_states.dtype)
        timestep_embedding, adaln_lora = diffusion_model.t_embedder[1](sinusoidal_timestep)
        timestep_embedding = diffusion_model.t_embedding_norm(timestep_embedding)

        diffusion_model.affline_scale_log_info = {
            "t_embedding_B_T_D": timestep_embedding.detach()
        }
        diffusion_model.affline_emb = timestep_embedding
        diffusion_model.crossattn_emb = context

        if extra_pos_emb is not None:
            assert hidden_states.shape == extra_pos_emb.shape
        return x, hidden_states, rope_embedding, extra_pos_emb, timestep_embedding, adaln_lora

    @staticmethod
    def _apply_post_input_patches(hidden_states, x, transformer_options):
        """ComfyUI's post_input patches on the token sequence (on a copy of transformer_options when there are any)."""
        patches = transformer_options.get("patches", {})
        if "post_input" in patches:
            transformer_options = transformer_options.copy()
            transformer_options["model_patch_data"] = {}
            for patch in patches["post_input"]:
                out = patch({
                    "img": hidden_states,
                    "x": x,
                    "transformer_options": transformer_options,
                })
                hidden_states = out["img"]
        return hidden_states, transformer_options

    @staticmethod
    def _run_blocks(diffusion_model, hidden_states, timestep_embedding, context, block_kwargs, transformer_options):
        for block_index, block in enumerate(diffusion_model.blocks):
            transformer_options["block_index"] = block_index
            hidden_states = block(
                hidden_states,
                timestep_embedding,
                context,
                **block_kwargs,
            )
        return hidden_states

    @staticmethod
    def _final(diffusion_model, hidden_states, context, timestep_embedding, adaln_lora, original_shape):
        """The final layer, unpatchify and the crop to the unpadded input shape."""
        output = diffusion_model.final_layer(
            hidden_states.to(context.dtype),
            timestep_embedding,
            adaln_lora_B_T_3D=adaln_lora,
        )
        output = diffusion_model.unpatchify(output)
        return output[
            :,
            :,
            : original_shape[-3],
            : original_shape[-2],
            : original_shape[-1],
        ]

    def forward(
        self,
        diffusion_model,
        x,
        timesteps,
        context,
        fps=None,
        padding_mask=None,
        **kwargs,
    ):
        original_shape = list(x.shape)
        x, hidden_states, rope_embedding, extra_pos_emb, timestep_embedding, adaln_lora = self._embed(
            diffusion_model, x, timesteps, context, fps, padding_mask)

        transformer_options = kwargs.get("transformer_options", {})
        hidden_states, transformer_options = self._apply_post_input_patches(hidden_states, x, transformer_options)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_embedding.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": transformer_options,
        }
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.float()

        # the cache decision on the first block's modulated input, then the blocks or the Taylor prediction
        modulated_input = self._first_block_modulated_input(
            diffusion_model,
            hidden_states,
            timestep_embedding,
            adaln_lora,
            extra_pos_emb,
        )
        decision = self.decide(modulated_input, transformer_options)
        step_info = transformer_options.get(_STEP_INFO_KEY)
        step_index = step_info["index"] if step_info is not None else 0

        if decision.should_calc:
            original_hidden_states = hidden_states
            hidden_states = self._run_blocks(diffusion_model, hidden_states, timestep_embedding, context, block_kwargs,
                                             transformer_options)
            self.commit_residual(decision, hidden_states - original_hidden_states, step_index)
        else:
            hidden_states = self.predict_residual(decision, hidden_states, step_index)

        return self._final(diffusion_model, hidden_states, context, timestep_embedding, adaln_lora, original_shape)

class WeightFerryTaylorSeer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The native ComfyUI Anima model."}),
                "mode": (["adaptive", "step"], {"default": "adaptive", "tooltip": "adaptive: dynamic L1 threshold, step: fixed skip interval"}),
                "rel_l1_thresh": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Relative L1 distance threshold for adaptive mode"},
                ),
                "step_interval": (
                    "INT",
                    {"default": 2, "min": 2, "max": 5, "step": 1, "tooltip": "Interval between full calculations in step mode"},
                ),
                "max_order": (
                    "INT",
                    {"default": 1, "min": 1, "max": 2, "step": 1, "tooltip": "Taylor expansion maximum order (1: 1st-order derivative, 2: 2nd-order)"},
                ),
                "first_enhance": (
                    "INT",
                    {"default": 2, "min": 1, "max": 10, "step": 1, "tooltip": "Number of initial steps forced to full calculation"},
                ),
                "last_enhance": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of final steps forced to full calculation"},
                ),
                "cache_device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "max_predict_steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "0: no limit (as upstream TaylorSeer). n: compute a step whose prediction would reach more than n steps past the last computed step (adaptive samplers such as ReAL can jump several steps)"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_taylorseer"
    CATEGORY = "WeightFerry/speed"
    TITLE = "TaylorSeer"

    def apply_taylorseer(
        self,
        model,
        mode,
        rel_l1_thresh,
        step_interval,
        max_order,
        first_enhance,
        last_enhance,
        cache_device,
        max_predict_steps=0,
    ):
        from comfy.ldm.anima.model import Anima

        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, Anima):
            raise TypeError("TaylorSeer only supports ComfyUI's native Anima model")

        device = (
            model_management.get_torch_device()
            if cache_device == "cuda"
            else torch.device("cpu")
        )
        runtime = TaylorSeerRuntime(
            mode=mode,
            rel_l1_thresh=rel_l1_thresh,
            step_interval=step_interval,
            max_order=max_order,
            first_enhance=first_enhance,
            last_enhance=last_enhance,
            cache_device=device,
            max_predict_steps=max_predict_steps,
        )
        patched_model = model.clone()
        transformer_options = patched_model.model_options.setdefault("transformer_options", {})
        wrappers = transformer_options.setdefault("wrappers", {})
        wrappers.setdefault(patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).pop(_WRAPPER_KEY, None)
        patcher_extension.add_wrapper_with_key(
            patcher_extension.WrappersMP.DIFFUSION_MODEL,
            _WRAPPER_KEY,
            runtime.diffusion_model_wrapper,
            patched_model.model_options,
            is_model_options=True,
        )
        return (patched_model,)

