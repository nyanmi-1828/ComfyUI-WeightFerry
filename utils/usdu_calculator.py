# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
USDU Tile Calculator (Aspect-Ratio Aware Dynamic Tile Optimization)
========================================================================
Dynamically calculates optimal tile_width, tile_height, tile_padding, and mask_blur
matching the input image aspect ratio, close to an even split of the canvas.

Idea of the tile formula (no code): the ComfyUI workflow article by nobin, https://note.com/nobinlog/n/nba67cb72e47d
(2025-11-05).
"""

import math
from typing import Optional, Tuple
import torch

from ..config import WeightFerryConfig
from .logging import log_info


class WeightFerryUSDUTileCalculator:
    """
    USDU aspect-ratio aware dynamic tile dimension and parameter optimizer.
    Tiles close to an even split of the canvas, few of them.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_factor": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Target tile factor K (suggested 4-6): the base number for the tile split"
                }),
                "upscale_by": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 8.0, "step": 0.05,
                    "tooltip": "USDU upscale factor (usually 2.0)"
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to upscale (its size is read automatically)"}),
                "width": ("INT", {"default": 896, "min": 64, "max": 8192, "step": 8, "tooltip": "Source width (used when image is not connected)"}),
                "height": ("INT", {"default": 1472, "min": 64, "max": 8192, "step": 8, "tooltip": "Source height (used when image is not connected)"}),
                "tile_skip_threshold": ("FLOAT", {
                    "default": WeightFerryConfig.USDU_DEFAULT_SKIP_THRESHOLD, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "For the tile_skip_threshold input of Ultimate SD Upscale+: tiles with low edge variance (flat background) are not redrawn. 0.0 = off, 1.0-2.0 suggested"
                }),
                "highpass_noise_strength": ("FLOAT", {
                    "default": WeightFerryConfig.USDU_DEFAULT_HIGHPASS_NOISE, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "For the highpass_noise_strength input of Ultimate SD Upscale+: low-frequency noise removed against colour / brightness shifts between tiles. 0.0 = off, 0.6-0.8 suggested"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "tile_width",
        "tile_height",
        "tile_padding",
        "mask_blur",
        "tile_skip_threshold",
        "highpass_noise_strength"
    )
    FUNCTION = "calculate"
    CATEGORY = "WeightFerry/upscale"
    TITLE = "USDU Tile Calculator"

    def calculate(
        self,
        tile_factor: int,
        upscale_by: float,
        image: Optional[torch.Tensor] = None,
        width: int = 896,
        height: int = 1472,
        tile_skip_threshold: float = WeightFerryConfig.USDU_DEFAULT_SKIP_THRESHOLD,
        highpass_noise_strength: float = WeightFerryConfig.USDU_DEFAULT_HIGHPASS_NOISE
    ) -> Tuple[int, int, int, int, float, float]:
        if image is not None:
            # image shape: [B, H, W, C]
            h_orig = int(image.shape[1])
            w_orig = int(image.shape[2])
        else:
            w_orig = width
            h_orig = height

        # Upscaled target canvas dimensions
        W = round(w_orig * upscale_by)
        H = round(h_orig * upscale_by)

        aspect_ratio = W / max(1, H)

        # Aspect-ratio aware tile grid count
        nx = max(1, round(math.sqrt(tile_factor * aspect_ratio)))
        ny = max(1, round(tile_factor / nx))

        # Tile size in multiples of USDU_TILE_MULTIPLE (64)
        m = WeightFerryConfig.USDU_TILE_MULTIPLE
        tw = int(m * max(1, math.floor((W / nx) / m)))
        th = int(m * max(1, math.floor((H / ny) / m)))

        # Padding: approx 15% of tile width, multiple of 8, bounded by config
        pad_m = WeightFerryConfig.USDU_PADDING_MULTIPLE
        raw_pad_units = round((tw * WeightFerryConfig.USDU_PADDING_RATIO) / pad_m)
        bounded_pad_units = max(
            WeightFerryConfig.USDU_MIN_PADDING_BOUND,
            min(WeightFerryConfig.USDU_MAX_PADDING_BOUND, raw_pad_units)
        )
        pad = int(pad_m * bounded_pad_units)

        # Mask blur: half of padding multiple
        blur = int(WeightFerryConfig.USDU_BLUR_MULTIPLE * bounded_pad_units)

        cols = math.ceil(W / tw)
        rows = math.ceil(H / th)
        total_tiles = cols * rows

        log_info(f"TileCalc: {w_orig}x{h_orig} -> Canvas {W}x{H} (Scale {upscale_by}x) | Factor K={tile_factor}")
        log_info(f"  => Tile: {tw}x{th} (Pad: {pad}, Blur: {blur}) | Grid: {rows}x{cols} ({total_tiles} tiles)")

        return (tw, th, pad, blur, float(tile_skip_threshold), float(highpass_noise_strength))
