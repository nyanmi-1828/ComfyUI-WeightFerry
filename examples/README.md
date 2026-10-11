# Example workflows

English | [日本語](README_ja.md)

Two Anima workflows with the official Anima base model and an empty LoRA stack. Open them in ComfyUI (drag the file
onto the canvas); the loaders name the files of [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
(or pick your own).

| File | What it does |
|---|---|
| `anima_txt2img.json` | Base image (saved) → optional 2x upscale → save. Switches: model precision (node "Model switch", index 1 = bf16, the official file, the default; 0 = fp8 / 2 = int8 ConvRot after you make those files), upscale method (index 0 = Ultimate SD Upscale, which is bypassed at first = no upscale; 1 = redraw without tiles; the "fast" in the switch's title holds when the VRAM is enough, with little VRAM Ultimate SD Upscale is faster). The TaylorSeer and CFG early-stop nodes (their titles start with "⚡") can be switched off with Ctrl+B for quality-first runs. |
| `anima_draft_turbo.json` | Fast draft: Turbo LoRA at 0.8, 6 steps, CFG 1, then a colour match; Ultimate SD Upscale bypassed (Ctrl+B to use it). Model switch: 1 = bf16 (default) / 0 = fp8. The Int "⚡ Turbo: 0 = off / 1 = on" (top left of the "⚡ Speed: Turbo" group, at the right end of the top row) switches the Turbo LoRA (for every pass), the base image's steps (6 / 16) and its CFG (1 / 5) together. Ultimate SD Upscale keeps 16 steps and CFG 5, so set Turbo to 0 before you use it. With Turbo on, the ⚡ TaylorSeer (`first_enhance` 8, more than the 6 steps) and the CFG early stop (CFG 1) have nothing to do. |

## The prompt

The prompt describes an original character with generic tags. Replace it with your own in the "Prompt" node (the
upscale uses the same prompt); the negative prompt has its own node ("Negative").

## Custom nodes they use

Besides this package and ComfyUI's own nodes, one pack (the Manager's "Install Missing Custom Nodes" finds it):

| Node | Package |
|---|---|
| `easy loraStack`, `easy loraStackApply`, `easy anythingIndexSwitch`, `easy imageColorMatch` (draft only) | [comfyui-easy-use](https://github.com/yolain/ComfyUI-Easy-Use) |

## Model files

The loaders name the official Anima release ([circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima),
`split_files/`). The fp8 / int8 versions of its base model are optional (made with this package's tools); their
loaders are saved bypassed, so a missing file does not stop the workflow.

| Loader | File in the workflow | What to use |
|---|---|---|
| UNET (index 1, bf16, default) | `anima-base-v1.0.safetensors` | The official base model (`split_files/diffusion_models/`) |
| UNET (index 0, fp8; optional, bypassed) | `anima-base-v1.0_fp8_scaled.safetensors` | The base in fp8 (scaled). From the ComfyUI folder: `python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_fp8.py models/diffusion_models/anima-base-v1.0.safetensors --scaled` (writes the file next to the original). Then take the loader out of bypass (Ctrl+B) and set the switch to 0 |
| UNET (index 2, int8 ConvRot; optional, bypassed; txt2img only) | `anima-base-v1.0_int8_convrot.safetensors` | The base in ComfyUI's int8 (ConvRot) layout: closer to bf16 than fp8, and the fastest on an RTX 4060 with this package's fused block. Make the fp8 file first, then from the ComfyUI folder `python custom_nodes/ComfyUI-WeightFerry/tools/convert_quant.py models/diffusion_models/anima-base-v1.0.safetensors --like models/diffusion_models/anima-base-v1.0_fp8_scaled.safetensors --format int8_convrot`; take the loader out of bypass and set the switch to 2 |
| Text encoder | `qwen_3_06b_base.safetensors` (type `qwen_image`) | Anima's Qwen3 0.6B text encoder (`split_files/text_encoders/`). The prompts are encoded by CLIP Text Encode (Streamed LLM) nodes, which use this loader's weights |
| VAE | `qwen_image_vae.safetensors` | Qwen-Image VAE (`split_files/vae/`) |
| Upscale model | `4x-UltraSharp.pth` | [Kim2091/UltraSharp](https://huggingface.co/Kim2091/UltraSharp) (CC BY-NC-SA 4.0), or any ESRGAN-type 4x model |
| Turbo LoRA (draft) | `anima_turbo_4step_v3.safetensors` | The Anima Turbo LoRA from [sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step) (license: see its model page), or another Anima few-step LoRA of your choice |

Put the files in `ComfyUI/models/`: `diffusion_models/`, `text_encoders/`, `vae/`, `upscale_models/`, `loras/`
(Turbo LoRA).

Models and LoRAs have their own licenses; check them before you share images.

## Settings in the workflow

The values in `anima_txt2img.json` (the draft workflow differs where its table row says so).

* **Sampling**: every KSampler uses `real` / `tors` (through "Shared sampler") and CFG 5; 16 steps.
* **CFG early stop**: `hard_stop`, 0.65 for the base image, 0.5 for the upscale.
* **TaylorSeer**: `adaptive`, `rel_l1_thresh` 0.06, `max_predict_steps` 3, `max_order` 1, `last_enhance` 1;
  `first_enhance` 8 for the base image, 1 for the upscale (the node's defaults are in the main README's
  [Usage](../README.md#taylorseer-weightferryspeed)). The draft workflow has TaylorSeer on the base image only.
* **DiT Layer Streaming**: `auto_config` on, `sage_attention` on (without the `sageattention` package installed, one
  warning in the log and ComfyUI's attention runs).
* **Ultimate SD Upscale** (this package's Ultimate SD Upscale+; bypassed at first, Ctrl+B to use it): 2x,
  denoise 0.38, 16 steps; "USDU tile calculator" (`tile_factor` 4) feeds its tiles and the two additions (see the main
  README's [Upscaling](../README.md#upscaling-weightferryupscale)).
* **Without tiles** (Upscale method = 1): the chain in the main README's
  [Upscaling](../README.md#upscaling-weightferryupscale), with lanczos 0.5 (4x model → 2x) and denoise 0.38. The
  "Upscale factor" node feeds only Ultimate SD Upscale and the tile calculator: for another factor on this path,
  change the lanczos scale and RoPE Scale's `factor`.

## License

The files in this folder (the workflows and these READMEs) are **MIT-0** ([../LICENSES/MIT-0.txt](../LICENSES/MIT-0.txt)):
use, copy, change and share them for anything, with no conditions and no attribution required.
