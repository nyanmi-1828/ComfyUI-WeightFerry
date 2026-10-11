# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Adapted from ComfyUI comfy_extras/nodes_upscale_model.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
Upscale with a model in half precision. ComfyUI's ImageUpscaleWithModel runs the model in float32
(`upscale_model(a.float())`); on an L4, 4x-UltraSharp on 896x1472 took 9.2 s in float32, 3.55 s in float16 (PSNR 67.6 dB
against float32) and 3.36 s in bfloat16 (49.1 dB). Same tiling (512, overlap 32, halved on OOM) and output as ComfyUI's
node; dtype "float32" (the default) gives ComfyUI's result.

Provenance: ComfyUI's ImageUpscaleWithModel.execute (comfy_extras/nodes_upscale_model.py, GPL-3.0) with a dtype
option added.
"""

import torch

import comfy.model_management as model_management
import comfy.utils


class WeightFerryUpscaleWithModelHalf:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale_model": ("UPSCALE_MODEL",),
            "image": ("IMAGE",),
            "dtype": (["float32", "float16", "bfloat16"], {"default": "float32", "tooltip":
                      "float32 = ComfyUI's ImageUpscaleWithModel. float16: ~2.6x faster on Ada, nearly identical "
                      "(4x-UltraSharp: PSNR 67.6 dB vs float32); bfloat16 slightly less close"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "WeightFerry/upscale"

    def upscale(self, upscale_model, image, dtype="float32"):
        dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        device = upscale_model.patcher.load_device
        alpha = None
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            image = image[..., :3]
        memory_required = (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)
        model = upscale_model.model
        orig_dtype = next(model.parameters()).dtype
        in_img = image.movedim(-1, -3).to(device)
        output_device = model_management.intermediate_device()
        tile, overlap = 512, 32
        try:
            if dt != torch.float32:
                model.to(dtype=dt)
            while True:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2],
                                                                                tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a.to(dt)).float(), tile_x=tile,
                                                tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale,
                                                pbar=pbar, output_device=output_device)
                    break
                except Exception as e:  # noqa: BLE001 -- as ComfyUI's node: re-raised unless it is an OOM
                    model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            if dt != torch.float32:
                model.to(dtype=orig_dtype)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0).to(model_management.intermediate_dtype())
        if alpha is not None:
            alpha = comfy.utils.common_upscale(alpha.movedim(-1, -3).to(s), s.shape[2], s.shape[1], "bilinear", "disabled")
            s = torch.cat((s, alpha.movedim(-3, -1)), dim=-1)
        return (s,)
