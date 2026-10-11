# SPDX-License-Identifier: GPL-3.0-only
# From ssitu/ComfyUI_UltimateSDUpscale 1.7.2 modules/upscaler.py (GPL-3.0); modified 2026-10-10 for ComfyUI-WeightFerry: the upscale model in float16 by default (WEIGHTFERRY_USDU_UPSCALE_DTYPE)

from PIL import Image
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from ..image_utils import tensor_to_pil, pil_to_tensor
from . import shared
from ...core import env
from ...utils.upscale_half import WeightFerryUpscaleWithModelHalf

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def upscale(self, img: Image, scale, selected_model: str = None):
        if scale == 1.0:
            return img
        if (shared.actual_upscaler is None):
            return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
        dtype = env.get("WEIGHTFERRY_USDU_UPSCALE_DTYPE")
        if dtype in ("float16", "bfloat16"):  # this package's half-precision upscale (RTX 4060, 4x-UltraSharp: 13.0 -> 5.8 s)
            (upscaled,) = WeightFerryUpscaleWithModelHalf().upscale(shared.actual_upscaler, shared.batch_as_tensor, dtype)
        elif "execute" in dir(ImageUpscaleWithModel):  
            # V3 schema: https://github.com/comfyanonymous/ComfyUI/pull/10149
            (upscaled,) = ImageUpscaleWithModel.execute(shared.actual_upscaler, shared.batch_as_tensor)
        else:
            (upscaled,) = ImageUpscaleWithModel().upscale(shared.actual_upscaler, shared.batch_as_tensor)
        shared.batch = [tensor_to_pil(upscaled, i) for i in range(len(upscaled))]
        return shared.batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
