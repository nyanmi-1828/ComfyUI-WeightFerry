# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nyanmi-1828
# _rope_embeddings is adapted (h / w frequencies changed) from NVIDIA Cosmos via ComfyUI
# comfy/ldm/cosmos/position_embedding.py: SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Licensed under the Apache License, Version 2.0 (LICENSES/Apache-2.0.txt).
"""
Position embedding (RoPE) for images larger than the base size; ntk with factor 2 is the recommended setting for a 2x
upscale without tiles. Off by default: with mode "off" it returns the model unchanged.

  WeightFerryRoPEScale  interpolate the positions, raise the RoPE base (NTK) or both by frequency band (YaRN-like)

Anima (Cosmos-Predict2 MiniTrainDIT): 3D RoPE (VideoRopePosition3DEmb, head dim 128 -> 42 h / 42 w / 44 t channels,
NTK factor of the h / w axes from extrapolation ratio 4.0), tokens in (t h w) order, patch 2x2.
"""


import functools
import math

import torch

import comfy.patcher_extension


_ROPE_KEY = "weightferry_rope_scale"


# -------------------------------------------------------------------------------------------------------------- RoPE
def _rope_embeddings(pe, mode, factor, beta, B_T_H_W_C, fps=None, h_ntk_factor=None, w_ntk_factor=None,
                     t_ntk_factor=None, device=None, dtype=None):
    """VideoRopePosition3DEmb.generate_embeddings with the h / w frequencies changed (same output layout)."""
    from einops import rearrange, repeat
    h_ntk = h_ntk_factor if h_ntk_factor is not None else pe.h_ntk_factor
    w_ntk = w_ntk_factor if w_ntk_factor is not None else pe.w_ntk_factor
    t_ntk = t_ntk_factor if t_ntk_factor is not None else pe.t_ntk_factor
    rng = pe.dim_spatial_range.to(device=device)
    dim_h = 2 * rng.numel()
    if mode == "ntk":  # raise the RoPE base like the model's own extrapolation ratio, by `factor` more
        h_ntk = h_ntk * factor ** (dim_h / (dim_h - 2))
        w_ntk = w_ntk * factor ** (dim_h / (dim_h - 2))
    h_freqs = 1.0 / ((10000.0 * h_ntk) ** rng)
    w_freqs = 1.0 / ((10000.0 * w_ntk) ** rng)
    t_freqs = 1.0 / ((10000.0 * t_ntk) ** pe.dim_temporal_range.to(device=device))
    B, T, H, W, _ = B_T_H_W_C
    if mode == "interpolate":  # positions / factor: the grid spans the base size's range at a finer step
        h_freqs = h_freqs / factor
        w_freqs = w_freqs / factor
    elif mode == "yarn":
        # by band (YaRN's "NTK-by-parts"): a frequency with few periods across the base-size axis is interpolated, one
        # with >= beta periods is kept (local detail keeps its trained step), a ramp in between
        def by_parts(freqs, n):
            ref_len = n / factor
            periods = ref_len * freqs / (2 * math.pi)
            keep = ((periods - 1.0) / max(beta - 1.0, 1e-6)).clamp(0.0, 1.0)
            return freqs * (keep + (1.0 - keep) / factor)
        h_freqs = by_parts(h_freqs, H)
        w_freqs = by_parts(w_freqs, W)
    seq = torch.arange(max(H, W, T), dtype=torch.float, device=device)
    half_h = torch.outer(seq[:H], h_freqs)
    half_w = torch.outer(seq[:W], w_freqs)
    if fps is None or pe.enable_fps_modulation is False:
        half_t = torch.outer(seq[:T], t_freqs)
    else:
        half_t = torch.outer(seq[:T] / fps * pe.base_fps, t_freqs)

    def rot(e):
        return torch.stack([torch.cos(e), -torch.sin(e), torch.sin(e), torch.cos(e)], dim=-1)
    half_h, half_w, half_t = rot(half_h), rot(half_w), rot(half_t)
    em = torch.cat([repeat(half_t, "t d x -> t h w d x", h=H, w=W),
                    repeat(half_h, "h d x -> t h w d x", t=T, w=W),
                    repeat(half_w, "w d x -> t h w d x", t=T, h=H)], dim=-2)
    return rearrange(em, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class WeightFerryRoPEScale:
    """Change the image position embedding while sampling (e.g. a whole-image redraw at 2x the base size)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["off", "interpolate", "ntk", "yarn"], {"default": "off", "tooltip":
                     "off: unchanged. interpolate: positions / factor. ntk: RoPE base raised by factor. yarn: low "
                     "frequencies interpolated, high kept (ramp by periods across the base-size axis)"}),
            "factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05,
                                 "tooltip": "size / base size per axis (2.0 for a 2x upscale)"}),
            "beta": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5,
                               "tooltip": "yarn: periods across the base-size axis from which a frequency is kept"}),
            "min_tokens": ("INT", {"default": 0, "min": 0, "max": 1000000,
                                   "tooltip": "apply only when the image has at least this many tokens (0 = always)"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "WeightFerry/upscale"

    def apply(self, model, mode="off", factor=2.0, beta=8.0, min_tokens=0):
        if mode == "off" or factor <= 1.0:
            return (model,)
        m = model.clone()

        def wrapper(executor, *args, **kwargs):
            dm = executor.class_obj
            pe = getattr(dm, "pos_embedder", None)
            x = args[0] if args else None
            n_tok = 0
            if x is not None and x.dim() >= 4:
                ps = getattr(dm, "patch_spatial", 2)
                n_tok = math.ceil(x.shape[-2] / ps) * math.ceil(x.shape[-1] / ps)
            if pe is None or not hasattr(pe, "dim_spatial_range") or n_tok < min_tokens:
                return executor(*args, **kwargs)
            had = "generate_embeddings" in pe.__dict__
            prev = pe.__dict__.get("generate_embeddings")
            pe.generate_embeddings = functools.partial(_rope_embeddings, pe, mode, float(factor), float(beta))
            try:
                return executor(*args, **kwargs)
            finally:
                if had:
                    pe.generate_embeddings = prev
                else:
                    del pe.generate_embeddings

        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _ROPE_KEY,
                                                     wrapper, m.model_options, is_model_options=True)
        return (m,)
