# Third-party notices

ComfyUI-WeightFerry's own code is under **MIT-0** (no conditions; attribution is appreciated but not required). A few
files adapt code from other projects and keep those projects' licenses. Each `.py` file states its license in its first
line (`SPDX-License-Identifier`). Because GPL-3.0 files are included, the package **as a whole** is distributed under
the **GNU GPL v3** (`LICENSE`); the MIT-0 and Apache-2.0 files can be reused on their own under their own terms.

License texts: [LICENSE](LICENSE) (GPL-3.0), [LICENSES/MIT-0.txt](LICENSES/MIT-0.txt),
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt).

## Adapted code

| File | License | Adapted from |
|---|---|---|
| `core/lora_remap.py` | GPL-3.0 | [sparklingcoffee777/ComfyUI-Anima-2.9B-loraPatch](https://github.com/sparklingcoffee777/ComfyUI-Anima-2.9B-loraPatch) (GPL-3.0): the 28 → 40 block LoRA remapping; the inserted block indices in `config.py` are that project's measurements |
| `enhancers/taylor_seer.py` | GPL-3.0 | [CocyNoric/ComfyUI-Anima-TeaCache](https://github.com/CocyNoric/ComfyUI-Anima-TeaCache) (Apache-2.0): base structure, the polynomial coefficients and the forward; [Shenyi-Z/TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) (GPL-3.0): the Taylor factor update / evaluation; [ComfyUI](https://github.com/comfyanonymous/ComfyUI) `comfy/ldm/cosmos/predict2.py` (GPL-3.0): the forward mirrors `MiniTrainDIT._forward` |
| `streaming/activation.py` | GPL-3.0 | ComfyUI `comfy/ldm/cosmos/predict2.py` (GPL-3.0): the Cosmos block / attention re-implemented by token rows (`_cosmos_block_rows`, `_cosmos_self_attention_rows`, `_cosmos_block_mods`) |
| `utils/upscale_half.py` | GPL-3.0 | ComfyUI `comfy_extras/nodes_upscale_model.py` (GPL-3.0): `ImageUpscaleWithModel.execute` with a dtype option |
| `enhancers/fused_block.py`, `fused_int8.py`, `fused_kernels.py`, `fused_state.py` | GPL-3.0 | ComfyUI `comfy/ldm/cosmos/predict2.py` (GPL-3.0): `Block.forward`, `Attention.compute_qkv` / `compute_attention` and `GPT2FeedForward.forward` re-implemented with fused kernels; [comfy-kitchen](https://pypi.org/project/comfy-kitchen/) (Apache-2.0, Comfy-Org): the int8 linear path's decisions follow its Python dispatch, its kernels are called (not bundled) and its int8 ConvRot quantiser is reproduced in Triton from its outputs (the compiled kernel's source was not used) |
| `utils/vae_tiles.py` | GPL-3.0 | ComfyUI `comfy/ldm/wan/vae.py` (GPL-3.0; original version [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) `wan/modules/vae.py`, Copyright 2024-2025 The Alibaba Wan Team Authors, Apache-2.0): the decoder's layers are called in the order of `Decoder3d.forward` / `run_up` and `WanVAE.decode` (one frame) |
| `usdu/` (Ultimate SD Upscale+) | GPL-3.0 | [ssitu/ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) 1.7.2 (GPL-3.0): `usdu_nodes.py` → `usdu/nodes.py`, `usdu_patch.py` → `usdu/patch.py`, `usdu_utils.py` → `usdu/image_utils.py`, `crop_model_patch.py`, `modules/*` → `usdu/a1111/*`; [Coyote-A/ultimate-upscale-for-automatic1111](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) (GPL-3.0) `scripts/ultimate-upscale.py` → `usdu/ultimate_upscale.py`. Modified 2026-10-10 (each file's header says how): only the main node, relative imports instead of the `sys.path` / `sys.modules` changes, no download at import, no gradio stand-in, tile skipping and high-pass noise added (the noise path follows ComfyUI's `nodes.common_ksampler`, GPL-3.0), the upscale model through `utils/upscale_half.py` |
| `tools/convert_to_gguf.py` | Apache-2.0 | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) `tools/convert.py` ((c) City96, Apache-2.0): the tensor layout rules (which tensors stay F32, ignored keys, arch "cosmos"). Modified: one tensor at a time, BF16 fallback. Uses the [gguf](https://pypi.org/project/gguf/) package (MIT, llama.cpp; not bundled) |
| `core/llm_stream.py` | GPL-3.0 | ComfyUI `comfy/text_encoders/llama.py` (GPL-3.0): a few lines (causal mask, RoPE call) |
| `enhancers/sage_fused.py` | Apache-2.0 | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) 2.2.0 (Copyright (c) 2024 by SageAttention team, Apache-2.0): `sageattention/triton/quant_per_thread.py` `quant_key_per_thread_int8_kernel` and the sm89 path of `sageattention/core.py` `sageattn_qk_int8_pv_fp8_cuda`. Modified 2026-10-11: the key quantiser subtracts the key mean itself (bf16, as PyTorch's subtraction); V's `per_channel_fp8` (compiled `transpose_pad_permute` + `scale_fuse_quant`) reproduced as one Triton kernel from its outputs (its source was not used); limited to the sm89 FP8 P·V path; SageAttention's own query quantiser and attention kernel are called (not bundled) |
| `enhancers/rope_scale.py` | Apache-2.0 | NVIDIA Cosmos `VideoRopePosition3DEmb.generate_embeddings`, via ComfyUI `comfy/ldm/cosmos/position_embedding.py` — Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, Apache-2.0. Modified: the h / w frequencies are scaled (interpolation / NTK / YaRN-like) |

Apache-2.0 notice for the TeaCache-derived parts: Licensed under the Apache License, Version 2.0; you may not use those
parts except in compliance with the License (LICENSES/Apache-2.0.txt). Distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND. Modified: integrated into a TaylorSeer-style cache with adaptive / step
modes, CPU cache option and tracing.

## Methods (ideas, no code taken)

| Where | Method |
|---|---|
| `enhancers/taylor_seer.py` | TaylorSeer: Liu et al., "From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers", ICCV 2025, [arXiv:2503.06923](https://arxiv.org/abs/2503.06923) |
| `samplers/real_sampler.py` | ReAL sampler: Yin et al., [arXiv:2609.33202](https://arxiv.org/abs/2609.33202) (2026), Algorithm 1 |
| `samplers/tors_scheduler.py`, `config.py` | Named after TORS: Zhou et al., "Analyzing and Improving Fast Sampling of Text-to-Image Diffusion Models", ECCV 2026, [arXiv:2603.00763](https://arxiv.org/abs/2603.00763) (official code [zju-pi/TORS](https://github.com/zju-pi/TORS), Apache-2.0). The sigma table (`TORS_PRECOMPUTED_TABLE`) was computed for this package from the paper's formulation; it is not copied from the official schedules (distributed as .pth files). The example workflows were tuned with it |
| `enhancers/prototypes.py` (`WeightFerryTGATE`, prototype) | TGATE: [arXiv:2404.02747](https://arxiv.org/abs/2404.02747) |
| `utils/usdu_calculator.py` | Tile formula idea: the ComfyUI workflow article by nobin, <https://note.com/nobinlog/n/nba67cb72e47d> (2025-11-05) |
| `utils/vae_tiles.py` | Tiling only the VAE decoder's local part with an exact halo, the attention on the whole latent: the VAE of [nihui/qwenimage-ncnn-vulkan](https://github.com/nihui/qwenimage-ncnn-vulkan) (Apache-2.0, `src/vae.cpp`) |
| `enhancers/fused_block.py`, `fused_int8.py`, `fused_kernels.py` | PyTorch's LayerNorm / RMSNorm / GELU numerics (BSD-3-Clause) reproduced by comparing outputs (order of operations and rounding); GELU values are taken from `torch.nn.functional.gelu` at run time |
| `streaming/sage.py` | Calls [SageAttention](https://github.com/thu-ml/SageAttention) when installed (not bundled) |

