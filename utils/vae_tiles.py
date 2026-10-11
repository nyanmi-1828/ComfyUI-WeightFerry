# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
"""
Exact tiled decode for ComfyUI's Wan 2.1 VAE (Anima / Qwen-Image's VAE, comfy/ldm/wan/vae.py), single images (one
latent frame), after nihui/qwenimage-ncnn-vulkan's VAE: the decoder's global part (conv1 and the middle
blocks with the attention over the whole latent) runs on the whole latent, optionally the first upsampling stages too;
the rest is purely local (3x3 convolutions, per-pixel RMS norm, SiLU, nearest-exact 2x upsampling) and runs in tiles
that carry a halo of exactly its receptive field, so every kept output pixel sees the same inputs as in the untiled
decode (ComfyUI's tiled decode tiles the attention too and blends overlaps: an approximation). Only the convolution
algorithms the backend picks for another tile shape can change the low bits.

Provenance: the layer walk follows Decoder3d.forward / run_up and WanVAE.decode of ComfyUI's comfy/ldm/wan/vae.py
(GPL-3.0; that file's original version is Wan-Video/Wan2.1 wan/modules/vae.py, Copyright 2024-2025 The Alibaba Wan
Team Authors, Apache-2.0) for one frame without the temporal cache: the modules are called, no code copied. The split
idea from nihui/qwenimage-ncnn-vulkan (Apache-2.0, src/vae.cpp), no code taken.
"""

import math

import torch


def _local_radius(layers):
    """Receptive-field radius (in pixels at the input resolution) and the upsampling factor of a run of decoder
    layers; None when one of them is not local (attention)."""
    from comfy.ldm.wan.vae import AttentionBlock, CausalConv3d, Resample, ResidualBlock
    r, scale = 0.0, 1

    def conv_r(conv):
        k = conv.kernel_size
        return (k[-1] // 2) / scale

    for layer in layers:
        if isinstance(layer, AttentionBlock):
            return None
        if isinstance(layer, ResidualBlock):
            for m in layer.residual:
                if isinstance(m, CausalConv3d):
                    r += conv_r(m)
        elif isinstance(layer, Resample):
            if not layer.mode.startswith("upsample"):
                return None
            scale *= 2
            r += conv_r(layer.resample[1])
        elif isinstance(layer, (CausalConv3d, torch.nn.Conv2d, torch.nn.Conv3d)):
            r += conv_r(layer)
    return math.ceil(r), scale


def split_points(wan):
    """Indices into decoder.upsamples after which the rest is local: 0 (the latent resolution) and after each
    upsampling Resample."""
    from comfy.ldm.wan.vae import Resample
    ups = list(wan.decoder.upsamples)
    return [0] + [i + 1 for i, layer in enumerate(ups) if isinstance(layer, Resample)]


def decode_exact_tiles(wan, z, tile=512, stage=0, out=None):
    """WanVAE.decode(z) for z [B, C, 1, h, w], the local part in tiles of about `tile` output pixels with an exact
    halo. stage = how many upsampling stages run untiled first (0 = tile right after the middle blocks). Returns
    [B, 3, 1, 8h, 8w] like WanVAE.decode, or writes into `out`."""
    if z.ndim != 5 or z.shape[2] != 1:
        raise ValueError("decode_exact_tiles: one latent frame only")
    d = wan.decoder
    ups = list(d.upsamples)
    split = split_points(wan)[stage]
    x = wan.conv2(z)
    x = d.conv1(x)
    for layer in d.middle:
        x = layer(x)
    for layer in ups[:split]:
        x = layer(x)
    rest = ups[split:] + list(d.head)
    rad = _local_radius(rest)
    if rad is None:
        raise ValueError("decode_exact_tiles: a non-local layer after the split")
    halo, f = rad
    B, _, T, H, W = x.shape
    t_in = max(1, tile // f)
    for y0 in range(0, H, t_in):
        y1 = min(H, y0 + t_in)
        ya, yb = max(0, y0 - halo), min(H, y1 + halo)
        for x0 in range(0, W, t_in):
            x1 = min(W, x0 + t_in)
            xa, xb = max(0, x0 - halo), min(W, x1 + halo)
            t = x[..., ya:yb, xa:xb]
            for layer in rest:
                t = layer(t)
            if out is None:
                out = torch.empty((B, t.shape[1], T, H * f, W * f), dtype=t.dtype, device=t.device)
            out[..., y0 * f:y1 * f, x0 * f:x1 * f] = t[..., (y0 - ya) * f:(y1 - ya) * f, (x0 - xa) * f:(x1 - xa) * f]
            del t
    return out


# ---------------------------------------------------------------------------------------------------------------- node
# Peak VRAM above the loaded VAE per mode, bf16, measured on an RTX 4060 (896x1472 and 1792x2944):
# (constant MB, bytes per output pixel). Untiled 1586 / 6342 MB; stage 2 tiles 786 / 3141 MB; stage 1 tiles 512:
# 443 / 786 MB; stage 0 tiles 512: 551 / 708 MB, tiles 256: 279 / 664 MB.
_PEAK = {"untiled": (0, 1200), "stage 2": (0, 595), "stage 1": (330, 87), "stage 0": (500, 40)}
_MARGIN = 1.15
MODES = ["auto", "untiled", "stage 2", "stage 1", "stage 0"]
_STAGE = {"stage 2": 2, "stage 1": 1, "stage 0": 0}


def peak_bytes(mode, out_px, dtype_size=2):
    c, per_px = _PEAK[mode]
    return int((c * 2 ** 20 + per_px * out_px) * dtype_size / 2 * _MARGIN)


def choose_mode(out_px, free_bytes, dtype_size=2):
    """auto: the first of untiled, stage 2, stage 1, stage 0 whose measured peak fits the free VRAM (stage 0 last)."""
    for mode in ("untiled", "stage 2", "stage 1"):
        if peak_bytes(mode, out_px, dtype_size) <= free_bytes:
            return mode
    return "stage 0"


class WeightFerryVAEDecodeExactTiles:
    """VAE Decode for the Anima / Qwen-Image VAE (Wan 2.1) that tiles only the decoder's local part, with an exact
    halo: the attention runs on the whole latent, so there are no seams (ComfyUI's tiled decode tiles the attention
    and blends overlaps). auto picks the least tiling that fits the free VRAM; any other VAE or a video latent goes
    to ComfyUI's VAEDecode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "vae": ("VAE",),
            "mode": (MODES, {"default": "auto", "tooltip":
                     "auto: untiled when it fits the free VRAM, else the least tiling that fits. "
                     "stage 2: tiles after two upsampling stages (half the peak of untiled; 1792x2944: 99.7% of pixels "
                     "identical to untiled, at most 1 level apart). stage 1 / stage 0: earlier tiles, far less VRAM "
                     "(786 / 708 MB at 1792x2944), more rounding difference (82% identical, at most 6 levels). The "
                     "differences come from other convolution algorithms for other shapes, not from the tiling."}),
            "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip":
                          "Output pixels per tile side (the halo is added on top)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "WeightFerry/vae"

    def decode(self, samples, vae, mode="auto", tile_size=512):
        import comfy.model_management as mm
        import nodes
        try:
            from comfy.ldm.wan.vae import WanVAE
        except Exception:  # noqa: BLE001
            WanVAE = None
        z = samples["samples"]
        if z.is_nested:
            z = z.unbind()[0]
        wan = vae.first_stage_model
        if z.ndim == 4:
            z = z.unsqueeze(2)
        if WanVAE is None or type(wan) is not WanVAE or z.ndim != 5 or z.shape[2] != 1:
            return nodes.VAEDecode().decode(vae, samples)
        from .logging import log_info
        out_px = z.shape[3] * 8 * z.shape[4] * 8
        dsize = mm.dtype_size(vae.vae_dtype)
        mm.load_models_gpu([vae.patcher], memory_required=1, force_full_load=vae.disable_offload)
        chosen = mode
        if mode == "auto":
            chosen = choose_mode(out_px, mm.get_free_memory(vae.device), dsize)
            mm.load_models_gpu([vae.patcher], memory_required=peak_bytes(chosen, out_px, dsize),
                               force_full_load=vae.disable_offload)
        log_info(f"VAE Decode (Exact Tiles): {tuple(z.shape)} -> {chosen}"
                 + (f", tiles {tile_size}" if chosen != "untiled" else ""))
        images = []
        with mm.cuda_device_context(vae.device):
            for i in range(z.shape[0]):
                zi = z[i:i + 1].to(device=vae.device, dtype=vae.vae_dtype)
                if chosen == "untiled":
                    dec = wan.decode(zi)
                else:
                    dec = decode_exact_tiles(wan, zi, tile=tile_size, stage=_STAGE[chosen])
                img = dec.to(device=vae.output_device, dtype=vae.vae_output_dtype(), copy=True)
                del dec
                vae.process_output(img)
                images.append(img.movedim(1, -1))
        out = torch.cat(images, 0)
        return (out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1]),)
