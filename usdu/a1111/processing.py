# SPDX-License-Identifier: GPL-3.0-only
# From ssitu/ComfyUI_UltimateSDUpscale 1.7.2 modules/processing.py (GPL-3.0); modified 2026-10-10 for ComfyUI-WeightFerry: relative imports, no config.json; tile skipping and high-pass noise (Ultimate SD Upscale+)
# 2026-10-12: the tile mask blur and composite on the crop region only (WEIGHTFERRY_USDU_LOCAL_COMPOSITE, the same bytes); the tile-skip check before the mask blur and the tile preparation

from PIL import Image, ImageFilter
import logging
import torch
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
import numpy as np
from tqdm import tqdm
import comfy.utils as comfy_utils
from enum import Enum
from typing import Optional
from ..image_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from .. import image_utils
from . import shared
from ..crop_model_patch import crop_model_cond

logger = logging.getLogger(__name__)

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

# Taken from the USDU script
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        batch_size=1,
        tile_skip_threshold=0.0,
        highpass_noise_strength=0.0,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = Image.new('L', init_img.size, 0)  # Placeholder mask
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            logger.warning("Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Ultimate SD Upscale+ additions (0 = off)
        self.tile_skip_threshold = tile_skip_threshold
        self.highpass_noise_strength = highpass_noise_strength
        self.skipped_tiles_count = 0

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.batch_size = batch_size
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            logger.info("Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if comfy_utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy_utils.PROGRESS_BAR_ENABLED = True  # USDU's config.json "per_tile_progress" (default True), not read
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar: Optional[tqdm] = None
            # self.pbar = tqdm(total=self.tiles, desc='USDU') # Creating the pbar here will cause an empty progress bar to be displayed

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy_utils.PROGRESS_BAR_ENABLED = True
    
class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def filter_highpass_noise(noise: torch.Tensor, cutoff_sigma: float = 3.0, strength: float = 0.0) -> torch.Tensor:
    """Removes ``strength`` of the low frequencies (Gaussian blur with ``cutoff_sigma``) from the noise, per channel and
    frame (any leading dimensions: 4-D image or 5-D video latents), and restores the standard deviation."""
    if strength <= 0.0:
        return noise
    import torchvision.transforms.functional as TF
    orig_shape = noise.shape
    orig_std = noise.std()
    h, w = orig_shape[-2], orig_shape[-1]
    flat_noise = noise.reshape(-1, 1, h, w)
    kernel_size = int(math.ceil(cutoff_sigma * 4)) | 1
    low_freq = TF.gaussian_blur(flat_noise, kernel_size=[kernel_size, kernel_size], sigma=[cutoff_sigma, cutoff_sigma])
    high_freq = flat_noise - strength * low_freq
    new_std = high_freq.std()
    if new_std > 1e-6:
        high_freq = high_freq * (orig_std / new_std)
    return high_freq.reshape(orig_shape)


_LAPLACIAN_KERNEL = None


def evaluate_tile_saliency(tile_pil, var_thresh: float = 1.0, max_thresh: float = 0.08):
    """(is_flat, Laplacian variance x 1e4, peak edge) of a PIL tile: flat = both below their thresholds (a flat sky is
    below 0.1, details above 50; the peak keeps tiles with a single thin line or hair)."""
    global _LAPLACIAN_KERNEL
    if var_thresh <= 0.0:
        return False, 0.0, 0.0
    if _LAPLACIAN_KERNEL is None:
        _LAPLACIAN_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
    arr = np.array(tile_pil.convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t_pad = torch.nn.functional.pad(t, (1, 1, 1, 1), mode="replicate")
    lap = torch.nn.functional.conv2d(t_pad, _LAPLACIAN_KERNEL, padding=0).abs()
    var_scaled = float(lap.var().item() * 10000.0)
    max_val = float(lap.max().item())
    return (var_scaled < var_thresh) and (max_val < max_thresh), var_scaled, max_val


def tiles_are_flat(images, crop_region, threshold):
    """Tile skipping: the tile is not redrawn when it is flat in every image of the batch."""
    return threshold > 0.0 and all(evaluate_tile_saliency(img.crop(crop_region), var_thresh=threshold)[0]
                                   for img in images)


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas,
           highpass_noise_strength: float = 0.0):
    """Choose the way to sample based on given inputs"""

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if "execute" in dir(SamplerCustom):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    # High-pass noise: ComfyUI's common_ksampler with the noise filtered between prepare_noise and sample
    if highpass_noise_strength > 0.0:
        import comfy.sample
        import latent_preview
        latent_image = comfy.sample.fix_empty_latent_channels(
            model, latent["samples"], latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None))
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        noise = filter_highpass_noise(noise, cutoff_sigma=3.0, strength=highpass_noise_strength)
        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False,
            noise_mask=latent.get("noise_mask", None), callback=latent_preview.prepare_callback(model, steps),
            disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED, seed=seed)
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples
        return out

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Tile skipping: a flat tile keeps its upscaled pixels (no encode, sampling or decode). WeightFerry 2026-10-12:
    # checked before the mask blur and the crops below, which it does not use (a skipped tile no longer blurs)
    if tiles_are_flat(shared.batch, crop_region, p.tile_skip_threshold):
        p.skipped_tiles_count += 1
        if p.progress_bar_enabled and p.pbar is not None:
            p.pbar.update(1)
        return Processed(p, [shared.batch[0]], p.seed, "")

    # Blur the mask (WeightFerry: on the crop region only with WEIGHTFERRY_USDU_LOCAL_COMPOSITE, the same values)
    if image_utils.LOCAL_COMPOSITE:
        image_mask = image_utils.local_blurred_mask(image_mask, crop_region, p.mask_blur)
    elif p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    with crop_model_cond(p.model, crop_region, p.init_size, init_image.size, tile_size) as model:
        # Generate samples
        samples = sample(model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                        negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas,
                        highpass_noise_strength=p.highpass_noise_strength)

    # Update the progress bar
    if p.progress_bar_enabled:
        assert p.pbar is not None
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        if image_utils.LOCAL_COMPOSITE:  # the composite below on the crop region only (the same bytes)
            shared.batch[i] = image_utils.composite_tile_local(init_image, tile_sampled, image_mask, crop_region)
            continue

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, "")
    return processed
