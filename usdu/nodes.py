# SPDX-License-Identifier: GPL-3.0-only
# From ssitu/ComfyUI_UltimateSDUpscale 1.7.2 usdu_nodes.py (GPL-3.0); modified 2026-10-10 for ComfyUI-WeightFerry: only
# the main node, as "Ultimate SD Upscale+" with tile skipping and high-pass noise; relative imports
"""
Ultimate SD Upscale+ (WeightFerryUltimateSDUpscalePlus): ComfyUI's Ultimate SD Upscale node (ssitu, around the
Ultimate SD Upscale script by Coyote-A, https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) bundled in this
package, with two optional additions, both off at 0:

* tile_skip_threshold -- a tile whose source pixels are flat (Laplacian variance x 1e4 below the threshold and peak edge
  below 0.08) is not redrawn: its upscaled pixels stay, and its sampling, VAE encode and decode are skipped.
* highpass_noise_strength -- each tile's initial noise loses that fraction of its low frequencies (Gaussian blur,
  sigma 3 latent pixels; the standard deviation is restored), against colour / brightness drift between tiles.

With both at 0 the output is the original node's (checked byte for byte, 2026-10-10). The separate
ComfyUI_UltimateSDUpscale package is not needed; when it is installed too, the two do not interfere.
"""

import logging
from contextlib import contextmanager
import torch
import comfy
from .patch import usdu
from .image_utils import tensor_to_pil, pil_to_tensor
from .a1111.processing import StableDiffusionProcessing
from .a1111 import shared
from .a1111.upscaler import UpscalerData
from ..utils.logging import log_info

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    """Context manager to temporarily suppress logging output."""
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = []

    return required, optional


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


class WeightFerryUltimateSDUpscalePlus:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        optional = optional + [
            ("tile_skip_threshold", ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1,
                "tooltip": "Skip flat tiles: a tile whose Laplacian variance x 1e4 is below this (and has no sharp "
                           "edge) is not redrawn. 1.0-2.0 suggested; 0 = off"})),
            ("highpass_noise_strength", ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Remove this fraction of the low frequencies from each tile's initial noise, against "
                           "colour / brightness drift between tiles. 0.6-0.8 suggested; 0 = off"})),
        ]
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "WeightFerry/upscale"
    TITLE = "Ultimate SD Upscale+ (tile skip, high-pass noise)"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = ("Upscales an image and runs image-to-image on tiles from the input image (Ultimate SD Upscale), "
                   "with two optional additions: skip flat tiles (tile_skip_threshold) and high-pass filtered noise "
                   "against colour drift between tiles (highpass_noise_strength). Both 0 = Ultimate SD Upscale.")

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                tile_skip_threshold=0.0, highpass_noise_strength=0.0):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        logger.debug("Ultimate SD Upscale+ using batch_size=%s", batch_size)
        assert batch_size == 1 or force_uniform_tiles, "batch_size greater than 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size."

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode,
            None, None, batch_size,
            tile_skip_threshold=tile_skip_threshold, highpass_noise_strength=highpass_noise_strength,
        )

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            #
            # Running the script
            #
            script = usdu.Script()
            script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                       mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                       seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                       upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                       save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                       seams_fix_type=seam_fix_mode, target_size_type=2,
                       custom_width=None, custom_height=None, custom_scale=upscale_by)
        if sdprocessing.skipped_tiles_count:
            log_info(f"Ultimate SD Upscale+: {sdprocessing.skipped_tiles_count} flat tile(s) not redrawn")

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)
