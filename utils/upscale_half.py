# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Adapted from ComfyUI comfy_extras/nodes_upscale_model.py (GPL-3.0); _rrdb_append_forward re-implements the
# ResidualDenseBlock_5C.forward of chaiNNer-org/spandrel (MIT) with an append buffer; see THIRD_PARTY_NOTICES.md
"""
Upscale with a model in half precision. ComfyUI's ImageUpscaleWithModel runs the model in float32
(`upscale_model(a.float())`); on an L4, 4x-UltraSharp on 896x1472 took 9.2 s in float32, 3.55 s in float16 (PSNR 67.6 dB
against float32) and 3.36 s in bfloat16 (49.1 dB). Same tiling (512, overlap 32, halved on OOM) and output as ComfyUI's
node; dtype "float32" (the default) gives ComfyUI's result.

Provenance: ComfyUI's ImageUpscaleWithModel.execute (comfy_extras/nodes_upscale_model.py, GPL-3.0) with a dtype
option added.
"""

import contextlib

import torch

import comfy.model_management as model_management
import comfy.utils


def _rrdb_append_forward(orig):
    """spandrel's ResidualDenseBlock_5C.forward (ESRGAN / RRDB models such as 4x-UltraSharp) with its growing
    concatenations torch.cat((x, x1, ...), 1) replaced by one buffer per call that x and each new conv output are
    copied into once; each conv reads a channel prefix of it, for batch 1 a contiguous NCHW view with the same shape
    and strides as the cat. The same convolutions, inputs and order: the same bits (2026-10-12, 4x-UltraSharp float16
    on 896x1472 as USDU runs it: identical output, x1.11; the concatenations were 15.8% of the GPU time). Other cases
    (batch > 1, the "plus" variant, non-contiguous input) take spandrel's code."""
    def forward(self, x):
        if self.conv1x1 is not None or x.shape[0] != 1 or not x.is_contiguous():
            return orig(self, x)
        nf = x.shape[1]
        x1 = self.conv1(x)
        gc = x1.shape[1]
        buf = torch.empty((1, nf + 4 * gc) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
        buf[:, :nf].copy_(x)
        buf[:, nf:nf + gc].copy_(x1)
        x2 = self.conv2(buf[:, :nf + gc])
        buf[:, nf + gc:nf + 2 * gc].copy_(x2)
        x3 = self.conv3(buf[:, :nf + 2 * gc])
        buf[:, nf + 2 * gc:nf + 3 * gc].copy_(x3)
        x4 = self.conv4(buf[:, :nf + 3 * gc])
        buf[:, nf + 3 * gc:].copy_(x4)
        x5 = self.conv5(buf)
        return x5 * 0.2 + x
    forward._weightferry = True
    return forward


@contextlib.contextmanager
def _rrdb_append():
    """While this node runs its model: spandrel's dense block with the append buffer (WEIGHTFERRY_UPSCALE_RRDB_APPEND),
    restored afterwards (ComfyUI's own upscale node is not touched)."""
    from ..core import env as _env
    cls = None
    if _env.get("WEIGHTFERRY_UPSCALE_RRDB_APPEND") == "1":
        try:
            from spandrel.architectures.__arch_helpers.block import ResidualDenseBlock_5C as cls
        except Exception:  # noqa: BLE001
            cls = None
    if cls is None or getattr(cls.forward, "_weightferry", False):
        yield
        return
    orig = cls.forward
    cls.forward = _rrdb_append_forward(orig)
    try:
        yield
    finally:
        cls.forward = orig


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
                    with _rrdb_append():
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
