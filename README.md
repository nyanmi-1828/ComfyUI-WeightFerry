# ComfyUI-WeightFerry

<img src="assets/logo.png" width="128" alt="WeightFerry logo: a ferry carrying model layers over waves" align="right">

English | [日本語](README_ja.md) — by [nyanmi-1828](https://github.com/nyanmi-1828)

**Run diffusion models that don't fit in your VRAM inside ComfyUI, with ComfyUI's own computation.**

**Who it is for:** GPUs with 2–8 GB of VRAM, or any model larger than your VRAM. When the model already fits, the
streaming stays off by itself, and the other optimizations (fused block kernels for Anima / Cosmos-Predict2,
half-precision upscaling) still make generation a little faster.

* **Big models on a small GPU.** Cosmos-Predict2 **14B** (fp8) runs on an 8 GB RTX 4060 in about **3 GB of VRAM**
  instead of the 7 GB peak of ComfyUI's own run, at about the same speed; Qwen-Image **20B** in 3.3 GB. LLM text
  encoders up to 7B stream too.
* **Same output.** Streaming only changes where the weights live. Every streaming mode is checked
  **bit for bit** against ComfyUI's own run of the same model, in the same process with the same settings (activation
  streaming, used for very large images on a small GPU, in a small test only; a full low-VRAM run with it differs
  slightly, LPIPS ~0.008: see [How it was verified](#how-it-was-verified)). A few optional speed-ups, the float16
  upscale model inside Ultimate SD Upscale+ (its default) and one global default, TF32, do change pixels slightly: see
  [What this package changes globally](#what-this-package-changes-globally).
* **Faster when VRAM is tight: up to 1.7x on an RTX 4060 (2.9x on a Colab L4).** The next block is copied while the
  current one computes. With VRAM limited to 2 GB, Anima runs **1.65x** faster than with ComfyUI's Dynamic VRAM (its
  own on-demand offloading); a GGUF Wan 2.1 14B at 4 GB runs **1.7x** faster than with ComfyUI-GGUF alone (RTX 4060;
  2.9x on a Colab L4). On a Colab L4, Wan 2.1, Z-Image and Lumina 2 at 4 GB run within
  ±7% of Dynamic VRAM, some slightly slower ([Measured results](#measured-results)). It plans itself from the free VRAM / RAM.
* **For Anima** (an anime image model based on Cosmos-Predict2 2B): faster blocks with the same results (fused
  kernels), and two optional speed-ups that change the image slightly (TaylorSeer, CFG early stop).

**How it works:** the model's blocks live in system RAM (or on an SSD) and are copied to the GPU just before they run,
overlapped with the computation of the previous block. The arithmetic is ComfyUI's own.

> Status: a research project by one person.

## Tested models

| Model | Tested |
|---|---|
| **Anima** (Cosmos-Predict2 2B based) | Every node; VRAM 8 GB and limited to 4 / 2 GB |
| Cosmos-Predict2 **14B**, Qwen-Image **20B** | Streaming, bit-exact (RTX 4060 8 GB, 32 GB RAM) |
| Wan 2.1 T2V 14B, Z-Image Turbo, Lumina Image 2.0 | Streaming, bit-exact (Colab L4, VRAM limited to 4 / 2 GB). Wan: images (1 frame) |
| GGUF models (ComfyUI-GGUF): Wan 2.1 14B, Cosmos-Predict2 14B | Streaming, bit-exact, also with a LoRA |
| Text encoders: Qwen3 0.6B, Qwen2.5-VL 7B, T5-XXL | Streaming, bit-exact |

Other DiTs (Flux, SD3, video) are untested; UNet models (SDXL, SD 1.5) are not supported.

## Requirements

* **GPU**: NVIDIA; tested on Ada (RTX 40 series, L4).
* **ComfyUI**: 0.38.0 or later.
* **RAM**: about the model's file size plus a few GB (Anima: 16 GB is enough); with less, see the SSD cache in
  [Minimal setup](#minimal-setup-low-vram).
* **OS**: Windows 11 or Linux.
* **Optional**: [SageAttention](https://github.com/thu-ml/SageAttention); Triton (`triton-windows` on Windows) for
  the fused block kernels.

## Install

From inside ComfyUI: open the **Manager** (Custom Nodes Manager), search for **WeightFerry**, install, and restart
ComfyUI. The package is published on the [Comfy Registry](https://registry.comfy.org) as `comfyui-weightferry`.

Manual install:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nyanmi-1828/ComfyUI-WeightFerry
```

Restart ComfyUI. The nodes appear under **WeightFerry/** in the node menu. Loading the package changes a few things
globally (TF32 on, a fused Anima block, ...): see
[What this package changes globally](#what-this-package-changes-globally). To uninstall, delete the folder and
restart.

## Minimal setup (low VRAM)

1. Load the diffusion model as usual (Load Diffusion Model / Load Checkpoint / Unet Loader (GGUF)), then the LoRAs.
2. Add **⚡ DiT Layer Streaming** right after them and leave everything at the defaults (`auto_config` is on):

   ```
   Load Diffusion Model -> (LoRAs) -> ⚡ DiT Layer Streaming -> KSampler
   ```

3. If the text encoder does not fit either, add **⚡ Text Encoder Layer Streaming** after the CLIP loader.
4. Generate and look at the console. When the model does not fit, you see what was planned:

   ```
   (Cosmos-Predict2 14B, 36 blocks, on an 8 GB GPU)
   [WeightFerry] DiT Layer Streaming [auto]: {..., 'decision': 'stream 26 of 36 blocks, 10 resident', ...}
   [WeightFerry] DiT Layer Streaming: Registering Late Offload (10/36 resident, 26 streaming blocks, ...)
   ```

   When it fits: `[WeightFerry] DiT Layer Streaming [auto]: the whole model fits in VRAM; streaming not used (ComfyUI is faster then).`
   For large images that do not fit, activation streaming and the TaylorSeer RAM cache switch on by themselves; the
   log says so. If VRAM overflowed into shared GPU memory, a warning says so (see [Troubleshooting](#troubleshooting)).

The first generation after loading a model takes a few seconds longer (the weights are captured into RAM once). With
little RAM, the blocks that do not fit go to a cache file in ComfyUI's temp folder (`WEIGHTFERRY_DISK_DIR` puts it on
a faster SSD).

## Nodes

The streaming nodes work with every model in [Tested models](#tested-models). TaylorSeer is for Anima only, RoPE Scale
for Anima / Cosmos-Predict2.

| Menu | Node | What it does | Effect on the image |
|---|---|---|---|
| WeightFerry/streaming | ⚡ DiT Layer Streaming (low VRAM) | Streams the diffusion model's blocks from RAM / SSD; right after the model loader (after LoRAs) | None (bit-exact); its `sage_attention` option switches attention to SageAttention, like ComfyUI's `--use-sage-attention` |
| WeightFerry/streaming | ⚡ Text Encoder Layer Streaming (low VRAM) | The same engine for ComfyUI's own text encoders (T5, Llama / Qwen layer stacks) | None (bit-exact) |
| WeightFerry/streaming | CLIP Text Encode (Streamed LLM) | CLIP Text Encode with the text encoder run layer by layer from RAM; any text encoder | None (bit-exact) |
| WeightFerry/streaming | CLIP Text Encode (Streamed LLM, from File) | Anima only: the same, with a text encoder file to choose (`model_name`) | None (bit-exact) |
| WeightFerry/speed | TaylorSeer | Anima only: predicts the model output on steps it judges predictable (Taylor expansion of cached features) instead of computing them | **Slight**: the predicted steps differ a little |
| WeightFerry/speed | CFG Early-Stop | CFG for the first part of the sampling, CFG 1.0 (no negative branch) for the rest | **Slight**: the last steps without the negative prompt |
| WeightFerry/sampling | ReAL Sampler / TORS Scheduler | An adaptive-step sampler and a scheduler (also injected into KSampler as `real` / `tors`) | A different sampler / schedule: a different image, like any other choice |
| WeightFerry/sampling | Sampler Choice (shared sampler / scheduler) | One sampler / scheduler setting shared by several KSamplers | None (passes the setting on) |
| WeightFerry/upscale | Upscale Image (using Model, half precision) | ComfyUI's upscale-model node with a float16 / bfloat16 option; the default `float32` is ComfyUI's node | None with `float32`; tiny with `float16` (looks the same) |
| WeightFerry/upscale | RoPE Scale (large images) | Anima / Cosmos-Predict2 only: position-embedding scaling (NTK / interpolation / YaRN-like) for redrawing at 2x the trained size without tiles | Yes, by design (for sizes beyond the trained one) |
| WeightFerry/upscale | Ultimate SD Upscale+ (tile skip, high-pass noise) | [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale), bundled, with two optional additions: flat tiles are not redrawn, and each tile's initial noise loses its low frequencies | Tiny by default (float16 upscale model); the additions change it by design; the original node's output only with both at 0 and float32 ([Upscaling](#upscaling-weightferryupscale)) |
| WeightFerry/upscale | USDU Tile Calculator | Tile size for Ultimate SD Upscale from the image's aspect ratio (and the two values above) | None (computes values) |
| WeightFerry/vae | VAE Decode (Exact Tiles, low VRAM) | VAE Decode for the Wan 2.1 VAE (Anima, Qwen-Image) with seamless tiles when VRAM is short; untiled when it fits | None when untiled; small when tiled (see [its section](#vae-decode-exact-tiles-low-vram-weightferryvae)) |
| WeightFerry/prototype (not adopted) | CFG Negative Cache, Coarse Global Attention, TGATE | Measured and **not adopted**; kept for reference. Their `mode` is `off` by default: they do nothing until it is changed | Yes (results and reasons: [enhancers/prototypes.py](enhancers/prototypes.py)) |

## Usage

How to connect each node; the tables show the nodes' default values. The example workflow's own values are listed
in [examples/README.md](examples/README.md).

### The model chain (Anima)

```
UNET loader → LoRAs → ⚡ DiT Layer Streaming → CFG Early-Stop → TaylorSeer → KSampler
```

With more sampling passes (face detail, upscale, ...), give each its own CFG Early-Stop + TaylorSeer from the same
streaming node's output.

* **Model precision**: the official bf16 file works as is. An int8 ConvRot version (made with
  [the converters](#converters-tools)) is the fastest on an RTX 4060 and closer to bf16 than fp8 (finished images against bf16 with the same seed: LPIPS 0.31
  vs 0.42); the fused block
  kernels apply to it and to scaled fp8. The same holds for Cosmos-Predict2 14B: int8 ConvRot (14.3 GB, the size of
  the scaled fp8 file) against bf16 LPIPS 0.030 (scaled fp8: 0.33, a visibly different picture) and 5.30 against
  6.89 s/it (896x1472, 30 steps, CFG 4, RTX 4060).
* **⚡ DiT Layer Streaming**: `auto_config` on (the default) sets `resident_blocks` and `tiering` from the
  free VRAM / RAM; the other options apply in both modes. `vram_budget_gb` (0 = none, the default and the fastest)
  caps the VRAM ComfyUI may use, for running beside a game or another GPU app: the automatic choices plan inside it,
  the image stays the same and only the speed changes (a soft limit: a VAE or upscale peak can pass it, with a
  warning once).

### CFG Early-Stop (WeightFerry/speed)

| Setting | Default |
|---|---|
| `early_stop_percent` | 0.65 |
| `decay_mode` | `hard_stop` |

`linear_decay` lowers the CFG gradually from the full value at `early_stop_percent` to 1.0 at the last step
instead of dropping it at once. Switch the node off (Ctrl+B) for the most faithful CFG. Measured with a workflow of base image + face detail +
Ultimate SD Upscale+ (int8 ConvRot, RTX 4060): turning it off (the CFG Early-Stop
node of every pass) makes an image about 27% slower.

### TaylorSeer (WeightFerry/speed)

| Setting | Default |
|---|---|
| `mode` (`adaptive` or `step`) | `adaptive` |
| `rel_l1_thresh` (`adaptive`) | 0.08 |
| `step_interval` (`step`) | 2 |
| `max_order` | 1 |
| `first_enhance` (steps always computed at the start) | 2 |
| `last_enhance` | 1 |
| `max_predict_steps` (0 = no limit) | 0 |
| `cache_device` | `cuda` |

A lower `rel_l1_thresh` computes more steps (closer to no TaylorSeer, slower). Switching it off (Ctrl+B on the TaylorSeer node of every
pass) makes an image about 30% slower in the same measurement.

### Sampler and scheduler (WeightFerry/sampling)

Pick `real` / `tors` in KSampler like any other sampler / scheduler. **Sampler Choice** holds one sampler / scheduler pair and feeds it to several
KSamplers (base, face detail, Ultimate SD Upscale), so they change together.

### Upscaling (WeightFerry/upscale)

* **Ultimate SD Upscale**: use **Ultimate SD Upscale+**. It is Ultimate SD Upscale, bundled: the separate
  ComfyUI_UltimateSDUpscale is not needed, and if it is installed too, the two work independently.
  * Connect **USDU Tile Calculator**: the image and `upscale_by` (USDU's factor) in; its `tile_width`,
    `tile_height`, `tile_padding`, `mask_blur`, `tile_skip_threshold` and `highpass_noise_strength` outputs to the USDU
    inputs of the same names. It picks the tile size from the image's aspect ratio; `tile_factor` (default 4) is
    the approximate number of tiles.
  * The calculator also passes `tile_skip_threshold` 1.0 (flat tiles such as sky or plain walls are not redrawn) and
    `highpass_noise_strength` 0.6 (against colour shifts between tiles), so both are on once it is connected; 0 turns
    each off.
  * Its upscale model runs in float16 (RTX 4060, 4x-UltraSharp: 13.0 → 5.8 s; the final image differs slightly,
    LPIPS 0.012); `WEIGHTFERRY_USDU_UPSCALE_DTYPE=float32` gives Ultimate SD Upscale's float32. With both additions
    at 0 and float32 the output is the original node's, byte for byte.
  * With little VRAM it is the faster upscale: 52 / 65 s per image at a 4 / 2 GB limit, against 61 / 74 s without
    tiles (RTX 4060).
* **Without tiles** (one redraw of the whole image): **Upscale Image (using Model)** → resize → VAE Encode →
  KSampler, with the model through **RoPE Scale** (set its `mode` to `ntk`, it is `off` by default; `factor` 2.0
  for 2x) so the larger size keeps the trained position range.
* **Upscale Image (using Model)** on its own: `float32` = ComfyUI's node; `float16` about 2.2–2.6x faster and
  visually identical.
* With an ESRGAN-type model (e.g. 4x-UltraSharp), this package's upscale (this node and Ultimate SD Upscale+'s
  upscale step) runs the model's dense blocks with an append buffer and fused kernels, and in float16 its last stage
  in strips of 256 output rows, which lowers the VRAM peak of a 512 tile (process memory 2304 → 768 MB); the output is
  the same bit for bit ([docs/ENVIRONMENT.md](docs/ENVIRONMENT.md), `WEIGHTFERRY_UPSCALE_*`).

### VAE Decode (Exact Tiles, low VRAM) (WeightFerry/vae)

Use it in place of VAE Decode / VAE Decode (Tiled) for the Wan 2.1 VAE (Anima, Qwen-Image) when VRAM is short: large
images, small GPUs. The example workflows use ComfyUI's VAE Decode for the base image and VAE Decode (Tiled) on the
no-tiles upscale path; swap this node in where one of them runs out of VRAM.
`mode` = `auto`: untiled when it fits the free VRAM (then the same bytes as VAE Decode), otherwise `stage 2`
(half the peak, practically identical), then `stage 1` / `stage 0` (a fraction of the memory, more rounding
differences: 82% of pixels identical, at most 6 levels apart). Unlike ComfyUI's tiled decode it has no seams: only the decoder's local layers are tiled, with exactly
their receptive field as overlap. 1792×2944 on an RTX 4060: untiled 6.3 GB peak, `stage 2` 3.1 GB (99.7% of
pixels identical, at most 1 level apart), `stage 0` 0.7 GB. `tile_size` 512 is a good default. Other VAEs and video latents fall back to VAE Decode.

### Text encoders (WeightFerry/streaming)

* **⚡ Text Encoder Layer Streaming**: after a CLIP loader whose text encoder does not fit (T5, Llama / Qwen);
  `auto_config` on.
* **CLIP Text Encode (Streamed LLM)**: in place of CLIP Text Encode (same inputs: `clip`, `text`), for any text
  encoder larger than the VRAM; the conditioning is the same bit for bit. Anima's text encoder (Qwen3 0.6B) runs on
  this package's streamed LLM engine (the engine alone, measured on Qwen2.5-VL 7B: about 1.7–1.9x faster than
  ComfyUI's code); every other one runs
  ComfyUI's own encode with its layers streamed, as with ⚡ Text Encoder Layer Streaming. The example workflows use it
  for their prompts although Anima's 0.6B text encoder fits: same result, and a ready place for a larger one.
  * `disk_cache` (off by default): the result is kept on disk and reused for the same text and the same CLIP, also
    after a restart. On a hit the node does not request its CLIP input, so when nothing else uses the CLIP (no face
    detail with it, for example) the text encoder is never loaded: no load time and no RAM for it. The key is the
    text, the nodes that produce the CLIP (their inputs and model files) and the ComfyUI / PyTorch / GPU versions;
    the cached conditioning is the same bit for bit. Folder `<ComfyUI base>/.cache/weightferry/te_cache`, capped at
    1 GB (`WEIGHTFERRY_TE_CACHE_GB`).
* **CLIP Text Encode (Streamed LLM, from File)**: Anima only. The same with a `model_name` input: `(from clip)` (the
  default) uses the CLIP loader's weights, a file is streamed from disk instead (needed for quantized text encoders;
  the CLIP input is still needed for its tokenizers).

## Example workflows

[examples/](examples/) has two Anima workflows:

* `anima_txt2img.json` — base image → optional 2x upscale (Ultimate SD Upscale or no tiles), with the streaming
  node, TaylorSeer, CFG early stop and a model precision switch.
* `anima_draft_turbo.json` — a fast draft with a Turbo LoRA (6 steps, CFG 1).

Before you run them:

1. **Custom nodes**: besides this package they use only [comfyui-easy-use](https://github.com/yolain/ComfyUI-Easy-Use)
   (LoRA stack, switches); open the workflow, then Manager → **Install Missing Custom Nodes** finds it. Restart
   ComfyUI and open the workflow again.
2. **Model files**: the official Anima files from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima),
   an ESRGAN-type 4x upscale model and (draft only) a Turbo LoRA; file names, folders and sources
   in [examples/README.md](examples/README.md#model-files).
3. **Model switch**: it starts at `1` (bf16, the official file). Optional, faster: the fp8 / int8 ConvRot files (how to
   make them: [examples/README.md](examples/README.md#model-files)), then take their loader nodes out of
   bypass (Ctrl+B) and set the switch to `0` (fp8) or `2` (int8 ConvRot; `anima_txt2img.json` only).
4. **2x upscale**: off at first. In `anima_txt2img.json`, set "Upscale method" to `1` (no tiles), or keep it at `0`
   and take "USDU 2x" out of bypass (Ctrl+B) for Ultimate SD Upscale, the faster one with little VRAM. The draft has
   only the Ultimate SD Upscale node (see [examples/README.md](examples/README.md) for its Turbo setting).

## Measured results

RTX 4060 8 GB (PCIe 4.0 x8, 32 GB RAM) unless "L4" (a Colab L4 VM). Image size 896x1472, except Wan 832x480 (one
frame) and Z-Image / Lumina 2 1024x1024. s/it: seconds per sampling step, lower is better; measured warm (second run).
"VRAM limited to N GB": a helper process occupied the rest of the GPU memory, or PyTorch's per-process memory fraction
was set, so ComfyUI and WeightFerry had the same N GB. ComfyUI launch arguments: see
[Command-line reference](#comfyui-launch-arguments). "Bit-exact": see [How it was verified](#how-it-was-verified).

| Case | WeightFerry | Compared with | Output |
|---|---|---|---|
| Cosmos-Predict2 14B, fp8 | 6.56 s/it, peak VRAM **2.90 GB** | ComfyUI: 6.91 s/it, 7.01 GB | bit-exact |
| Cosmos-Predict2 14B, int8 ConvRot (made with the converters), 30 steps, CFG 4 | 5.30 s/it; against bf16 LPIPS 0.030 | scaled fp8: 6.89 s/it, LPIPS 0.33 | bit-exact (also with a LoRA) |
| Cosmos-Predict2 14B, bf16 (28.5 GB file), 32 GB RAM | 8.87 s/it (SSD tier) | ComfyUI: not run (28.5 GB of weights, 32 GB RAM) | bit-exact per block (each block's first forward; no whole run to compare) |
| Qwen-Image 20B | 6.21 s/it, peak VRAM 3.30 GB | ComfyUI: 6.54 s/it | bit-exact |
| Anima 2B, VRAM limited to 2 GB | 2.33 s/it | ComfyUI Dynamic VRAM: 3.86 s/it (**1.65x** faster) | bit-exact |
| Cosmos-Predict2 14B + LoRA, VRAM limited to 4 GB (L4) | 2.42 s/it | ComfyUI Dynamic VRAM: 3.78 s/it (1.56x) | bit-exact |
| Wan 2.1 14B (fp8) / Z-Image Turbo / Lumina 2, VRAM limited to 4 GB (L4) | 1.29 / 1.18 / 0.57 s/it | ComfyUI Dynamic VRAM: 1.21 / 1.26 / 0.55 s/it | bit-exact |
| Wan 2.1 14B GGUF Q8_0, VRAM limited to 4 GB | 2.35 s/it (L4: 1.46) | ComfyUI-GGUF alone: 4.04 s/it, 1.7x (L4: 4.23, 2.9x) | bit-exact |
| Anima 2B, model fits (no limit), streaming forced (`auto_config` does not stream here) | 0.94x the speed of not streaming, VRAM 3.70 -> 3.04 GB | ComfyUI without streaming | bit-exact |
| Qwen2.5-VL 7B text encoder on the streamed LLM engine (the engine alone) | ~1.7–1.9x faster, +2.0 GB VRAM | ComfyUI's own text-encoder code on the same tokens | bit-exact |
| Little RAM: SSD tier with a 2 GB RAM budget | 0.91x the speed of RAM only, 660 MB less pinned RAM | all blocks in RAM | bit-exact |
| Anima 2x upscale without tiles, 4 GB / 2 GB GPU | 61.0 / 74.3 s per image with `auto_config` (no manual settings) | the same workflow without a limit: 51.5 s | LPIPS ~0.008 against no limit (looks the same) |
| Anima fused block kernels (int8 ConvRot, CFG 5) | 22.1 ms per block | ComfyUI's block: 30.3 ms (1.37x; alternating, median) | bit-exact |
| Anima GGUF Q5_1 (ComfyUI-GGUF), model fits | 1.67 s/it with the one-kernel dequantisation | ComfyUI-GGUF's dequantisation: 2.37 s/it (1.42x; Q4_1 1.19x, Q8_0 1.03x) | bit-exact |
| Anima GGUF Q5_1 + LoRAs, model fits / VRAM limited to 4 GB, both with the one-kernel dequantisation | 1.68 / 0.97 s/it with the LoRA factors in pinned RAM | the factors copied from pageable RAM (as ComfyUI-GGUF keeps them): 1.89 / 1.18 s/it (1.12x / 1.22x) | bit-exact |
| Wan 2.1 14B GGUF Q4_K_M, VRAM limited to 4 GB | 1.91 s/it with the K-quant kernel | ComfyUI-GGUF's K-quant dequantisation: 2.81 s/it (1.47x, alternating in one process) | bit-exact |
| Anima 1792x2944 by token rows (the 2x upscale without tiles), VRAM limited to 4 GB | 2.39 s/it, SageAttention's key / value preparation once per block | once per row chunk: 3.2 s/it (1.34x) | bit-exact |

## What this package changes globally

Loading the package has side effects outside its own nodes:

* PyTorch TF32 matmul / cuDNN TF32 on (faster on Ampere / Ada; results differ slightly from strict float32). cuDNN
  autotuning is left to ComfyUI (its `--fast autotune` turns it on).
* Windows: 1 ms system timer resolution while ComfyUI runs.
* `TRITON_CACHE_DIR` = `<ComfyUI base directory>/.cache/triton` unless you set it.
* A hook that lets 28-block Anima LoRAs apply to Anima 2.9B (a 40-block variant) (`WEIGHTFERRY_LORA_REMAP=0` turns it off).
* The `real` sampler and the `tors` scheduler are added to KSampler's lists.
* ComfyUI's Cosmos-Predict2 / Anima `Block`, `Attention` and `GPT2FeedForward` classes get fused element-wise kernels (same results bit for
  bit; `WEIGHTFERRY_FUSED_BLOCK=0` turns it off).
* With ComfyUI-GGUF installed, its dequantisation (`ops.dequantize_tensor`, `GGMLLayer.get_weight`) is replaced by one
  Triton kernel per weight (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 and the K-quants Q2_K–Q6_K), with LoRA patches added in
  the same kernel after comfy's own `torch.mm` and the LoRA factors copied from pinned RAM; the same results bit for
  bit (`WEIGHTFERRY_GGUF_TRITON=0` turns it all off; the finer switches are in
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)). Once a LoRA factor is pinned, ComfyUI's `free_memory` /
  `unload_all_models` / `soft_empty_cache` are wrapped to give that pinned RAM back after a LoRA is removed.
* The streaming node's `sage_attention` / `sage_pv_fp16` options patch attention globally and are not undone.
* With a VRAM budget (`vram_budget_gb` or `WEIGHTFERRY_VRAM_BUDGET_GB`), ComfyUI's reserve (what `--reserve-vram`
  sets) and Dynamic VRAM's headroom are raised; they are restored when a WeightFerry node runs without a budget.

## Troubleshooting

* **"... MB of this process' GPU memory is in shared GPU memory (system RAM) ..."** (Windows): VRAM overflowed into
  system RAM, which makes generation much slower. `auto_config` keeps that much VRAM free from the next load on;
  closing other GPU applications helps too.
* **"Anima fused block: Triton not available ..."**: the fused block's Triton parts (`mod`, `modq`, `gelu`, `rms`,
  `ln`, `tay`) stay off; its other parts still apply (same images, a little slower). Install Triton
  (`triton-windows` on Windows) to use them.
* **Wan fails with "required rank 4 tensor to use channels_last format"**: start ComfyUI without
  `--force-channels-last` (a ComfyUI limitation with Wan's 5-D weights).
* **The example workflow cannot find `anima-base-v1.0_fp8_scaled.safetensors` / `..._int8_convrot.safetensors`**: make the file
  (see [Example workflows](#example-workflows), step 3), or bypass that loader again (Ctrl+B) and set the switch back
  to `1` (bf16).
* **Back to ComfyUI's own behaviour**: bypass the streaming node (Ctrl+B). Of the
  [global changes](#what-this-package-changes-globally), the fused block (`WEIGHTFERRY_FUSED_BLOCK=0`), the GGUF
  dequantisation (`WEIGHTFERRY_GGUF_TRITON=0`) and the LoRA remap (`WEIGHTFERRY_LORA_REMAP=0`) can be switched off; TF32, the 1 ms timer, the Triton cache folder (unless you set
  `TRITON_CACHE_DIR`) and the `real` / `tors` entries stay while the package is installed, and the SageAttention
  patch, once used, stays until ComfyUI restarts.

## Command-line reference

### ComfyUI launch arguments

None are required. The RTX 4060 measurements in this README were taken with these ComfyUI arguments (the Wan runs
without `--force-channels-last`, see [Troubleshooting](#troubleshooting)):
`--cuda-malloc --force-channels-last --bf16-unet --bf16-vae --fp32-text-enc --use-sage-attention --enable-triton-backend`

### Converters ([tools/](tools/))

Run with ComfyUI's Python from the ComfyUI folder, e.g.
`python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_fp8.py models/diffusion_models/MODEL.safetensors --scaled`.

| Script | Arguments | What it writes |
|---|---|---|
| `convert_to_fp8.py SRC` | `--scaled` (per-tensor scale, ComfyUI's comfy_quant format; recommended), `--keep-vanishing` (without `--scaled` only: keep tensors with a row that rounds entirely to zero in fp8), `--out FILE` | fp8 model; load with weight dtype `default`. Default name `<src>_fp8_scaled.safetensors` (`--scaled`) or `<src>_fp8_e4m3fn.safetensors` |
| `convert_quant.py SRC --like FP8_SCALED --format F` | `--format` `int8_convrot` (recommended), `int8`, `int8_channel`, `w4a4`; `--like` a scaled-fp8 file of the same model (its quantized layers are the ones converted); `--out FILE`; `--comfy-root DIR` (folder with ComfyUI's code, if it is not found: e.g. the desktop app) | ComfyUI's own quantized layouts; needs a CUDA GPU; written tensor by tensor, so the converted model does not have to fit in RAM. Default name `<src>_<format>.safetensors` |
| `convert_to_gguf.py SRC` | `--type` `Q8_0` (default), `Q5_1`, `Q5_0`, `Q4_1`, `Q4_0`, `BF16`; `--out FILE` | GGUF for ComfyUI-GGUF (Anima / Cosmos-Predict2 DiTs; needs `pip install gguf`). Default name `<src>-<type>.gguf`; load with "Unet Loader (GGUF)", the streaming node works after it |

The fp8 converter writes each block's weights as one contiguous range, so the streaming node can read blocks straight
from the file.

### Environment variables

75 optional `WEIGHTFERRY_*` variables (tuning, low-VRAM switches, debugging, off-switches for features on by default), each with its default and
the measurement behind it: [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). They can also be written to
`ComfyUI/user/weightferry.env` (one `NAME=value` per line; a real environment variable wins), handy with Comfy Desktop.

## Limitations

* Speed depends on PCIe and RAM bandwidth.
* It does not shrink models or change their weights (the converters in tools/ do, when you run them).

## How it was verified

Every streaming mode (block / chunk streaming, RAM / SSD tiers, scaled fp8, LoRA) is compared with ComfyUI's own
computation of the same model, in the same process with the same settings: weights used per block, the first forward of
every block, and whole generations, bit for bit. Activation streaming is checked bit for bit in a small
Cosmos-Predict2 test; a full Anima 2x upscale without tiles at a 4 / 2 GB limit (activation streaming and the
TaylorSeer RAM cache on) differs from the run without a limit by LPIPS ~0.008 (the images look the same; the cause has
not been pinned down). TaylorSeer and CFG early stop, which change the
image by design, were tuned by side-by-side comparisons (PSNR / SSIM / LPIPS and visual checks).

## Testing

A self-contained check, no model files needed (about a minute, NVIDIA GPU with ~2 GB free):

```bash
python custom_nodes/ComfyUI-WeightFerry/tests/test_streaming.py
```

Options: `--comfy-root DIR` (the folder with ComfyUI's code, if it is not found; default `$COMFYUI_ROOT` or the folder
above custom_nodes), `--legacy` (ComfyUI's classic ModelPatcher instead of Dynamic VRAM; the SSD-tier case is skipped there, because
that patcher loads the whole model to the GPU first). It builds a small random Cosmos-Predict2 DiT in memory and checks that block streaming, fewer slots, self-managed
resident blocks, chunk streaming and the SSD tier give results bit for bit identical to ComfyUI's own sampling.

## References

* TaylorSeer — Liu et al., ICCV 2025, [arXiv:2503.06923](https://arxiv.org/abs/2503.06923)
* ReAL sampler — Yin et al., 2026, [arXiv:2609.33202](https://arxiv.org/abs/2609.33202)
* TORS — Zhou et al., ECCV 2026, [arXiv:2603.00763](https://arxiv.org/abs/2603.00763). The `tors` scheduler here uses a sigma
  table that we computed in advance from the paper's formulation (not copied from the official schedules): 4–12, 14,
  15, 16, 20, 25 and 30 steps, other step counts interpolated from the 30-step one.
* TGATE (prototype) — [arXiv:2404.02747](https://arxiv.org/abs/2404.02747)
* Code adapted from ComfyUI, NVIDIA Cosmos, ComfyUI-Anima-TeaCache, TaylorSeer, ComfyUI-Anima-2.9B-loraPatch,
  ComfyUI_UltimateSDUpscale (with Coyote-A's Ultimate SD Upscale script) and ComfyUI-GGUF: see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

The package **as a whole** is distributed under the **GNU GPL v3** ([LICENSE](LICENSE)), because it contains files
adapted from GPL-3.0 projects. Inside it:

* Files adapted from other projects keep their licenses (GPL-3.0 from ComfyUI / TaylorSeer / loraPatch / Ultimate
  SD Upscale by ssitu and Coyote-A, Apache-2.0 from NVIDIA Cosmos / ComfyUI-GGUF; the ComfyUI-Anima-TeaCache parts,
  Apache-2.0, are inside the GPL-3.0 TaylorSeer file); each file's first line says which.
* This project's own code: **MIT-0** — taken on its own, use it for anything, no conditions. A mention of this project is
  appreciated if you find it useful, but not required.
* The example workflows ([examples/](examples/)) and the logo ([assets/](assets/)): **MIT-0** as well.

Details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
