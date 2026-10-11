# SPDX-License-Identifier: GPL-3.0-only
# From ssitu/ComfyUI_UltimateSDUpscale 1.7.2 modules/images.py (GPL-3.0); modified 2026-10-10 for ComfyUI-WeightFerry: moved to usdu/a1111/

from PIL import Image


def flatten(img, bgcolor):
    # Replace transparency with bgcolor
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")
