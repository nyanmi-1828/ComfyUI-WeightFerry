# Environment variables (`WEIGHTFERRY_*`)

ComfyUI-WeightFerry reads 63 `WEIGHTFERRY_*` environment variables. **Normal use needs none of them**: every default
is the behaviour that was measured and adopted, and the main options are node inputs. The variables are for tuning a
default, switching on a low-VRAM / large-image mode, debugging / profiling / benchmarks, and turning off a fix or
improvement that is on by default (for comparisons).

All defaults are in one table in `core/env.py`, and the code reads the variables through it (`WEIGHTFERRY_QUANT_LORA`
is also checked directly, to tell whether it was set). Most variables only affect the **⚡ DiT Layer Streaming (low VRAM)** node, `WEIGHTFERRY_LLM_*` only the
streamed-LLM text encoder (the CLIP Text Encode (Streamed LLM) nodes), and `WEIGHTFERRY_TAYLOR_*` only the TaylorSeer
node. Many tuning values only matter with `tiering` = Auto (RAM tier + SSD tier, what `auto_config` selects) or with
chunk streaming.

## Settings file

Instead of environment variables (awkward with Comfy Desktop), the values can be written to
`<ComfyUI user directory>/weightferry.env` (usually `ComfyUI/user/weightferry.env`), one `NAME=value` per line, `#`
for comments. It is read once per process; a real environment variable still wins. `WEIGHTFERRY_SETTINGS_FILE` names
another file, or `0` for none.

```text
# ComfyUI/user/weightferry.env
WEIGHTFERRY_FUSED_BLOCK=0
```

## When a variable is read

Set variables **before starting ComfyUI**. Some are read only once, so changing them in a running process has no effect:

| Read | Variables |
|---|---|
| At import (once per process) | `WEIGHTFERRY_FUSED_BLOCK`, `WEIGHTFERRY_GGUF_TRITON`, `WEIGHTFERRY_ACT_CHUNK_MIN_TOKENS`, `WEIGHTFERRY_SAGE_PV=fp16` (installs the kernel at import), `WEIGHTFERRY_TAYLOR_DEBUG`, `WEIGHTFERRY_TAYLOR_CACHE_CPU_MIN_TOKENS`, `WEIGHTFERRY_OVERFLOW_SLACK_MB` |
| Once per process, at the first pinned allocation | `WEIGHTFERRY_VIRTUALALLOC` (result cached) |
| When the streaming node is applied (graph execution) | `WEIGHTFERRY_AUTO`, `WEIGHTFERRY_CHUNK`, `WEIGHTFERRY_CHUNK_BLOCK0`, `WEIGHTFERRY_TIERING`, `WEIGHTFERRY_RAM_BUDGET_GB`, `WEIGHTFERRY_AUTO_HEADROOM_MB`, `WEIGHTFERRY_QUANT_LORA_VRAM_MB`, `WEIGHTFERRY_GGUF_KEEP_MMAP` |
| When the streamed-LLM text encoder is set up (first encode with a model) | `WEIGHTFERRY_LLM_COPY_THREADS`, `WEIGHTFERRY_LLM_STAGING` |
| At each model load (ComfyUI `ON_LOAD`, i.e. the weight capture) | most other `WEIGHTFERRY_*` variables |
| Per call / per forward | `WEIGHTFERRY_OVERFLOW_CHECK`, `WEIGHTFERRY_TE_CACHE_GB` (per stored prompt), `WEIGHTFERRY_CHUNK_LOG`, `WEIGHTFERRY_CHUNK_PROFILE`, `WEIGHTFERRY_OUTSIDE_LEAF`, `WEIGHTFERRY_ACT_HOST`, `WEIGHTFERRY_ACT_PER_ITEM`, `WEIGHTFERRY_ACT_H2D_STREAM` (per block), `WEIGHTFERRY_ACT_D2H_STREAM`, `WEIGHTFERRY_TRIM_WS`, `WEIGHTFERRY_USDU_UPSCALE_DTYPE`, `WEIGHTFERRY_LORA_REMAP` (per LoRA load) |

For readers of the code: "on-load capture" is `streaming/dit_node.py` and the steps in `streaming/load_steps.py`;
the other functions named below are in `streaming/` (`capture.py`, `auto_config.py`, `chunk_state.py`,
`activation.py`, `sage.py`) and `core/tiers/` (re-exported as `tiered_store`).

Notes column: "Read in" gives file and function, then the measurement behind the default (RTX 4060 8 GB unless a
Colab L4 is named). Each row's numbers were measured together, with the code and workflow of that time: compare the
numbers within a row, not across rows. Current end-to-end times are in
[2× upscale without tiles](#2-upscale-without-tiles-17922944-at-4-gb-and-2-gb).

Abbreviations: "chunk streaming" = sub-block streaming (`_ChunkState`); "activation streaming" = blocks computed in
token-row chunks with the residual stream in pinned RAM; "overflow" = VRAM spilling into Windows shared GPU memory
(system RAM).

---

## 1. Tuning / defaults in use (26)

The defaults are the measured settings. Change them only if you know why.

| Variable | Default | Values | What it does | Notes |
|---|---|---|---|---|
| `WEIGHTFERRY_SETTINGS_FILE` | unset (= `<ComfyUI user directory>/weightferry.env`) | a path, or `0` = no settings file | Where the [settings file](#settings-file) is read from. | Read once, at the first variable read. |
| `WEIGHTFERRY_GGUF_LORA_FACTORS` | `pin` (on since 2026-10-12) | `pin`, `0` = ComfyUI-GGUF's copies as before | GGUF models with LoRA: ComfyUI-GGUF keeps the LoRA factors in RAM (its patch_on_device off) and copies them to the GPU on every layer use; from pageable RAM each copy waits until the GPU has finished everything queued, so the GPU idles between layers (25.7% of a step for Anima Q5_1 + the workflow LoRAs). `pin` keeps a pinned RAM copy of each factor (made on first use, kept while the LoRA is loaded) and copies from it without that wait. The same values, the same bits. Costs RAM: the size of the LoRA factors once more, no VRAM. Off when ComfyUI runs with `--disable-pinned-memory`. | Read per layer call (`enhancers/gguf_dequant.py`, needs `WEIGHTFERRY_GGUF_TRITON`). RTX 4060, Anima Q5_1 + the workflow LoRAs resident, 896×1472, CFG 5, alternating in one process, latents identical: ×1.12 (5/5), GPU idle 25.7% → 0%; streamed at 4 GB, 896×1472: 1.18 → 0.97 s/it (×1.22). Every layer bit-exact on Q8_0 / Q5_1 / Q5_0 / Q4_1 / Q4_0; the same latents across LoRA changes (strength, another LoRA set, a second generation); the pinned RAM stays at the factors' size (Anima + 2 LoRAs: 154 MB) over repeated LoRA swaps. Keeping the factors on the GPU instead was slower and, outside ComfyUI's VRAM estimate, ran VRAM short. |
| `WEIGHTFERRY_GGUF_KQUANTS` | `1` (on since 2026-10-12) | `0` = ComfyUI-GGUF's code | GGUF models in the K-quant formats (Q2_K / Q3_K / Q4_K / Q5_K / Q6_K, e.g. the Q4_K_M files): their dequantisation as one Triton kernel too, with ComfyUI-GGUF's operations and roundings (each product and sum rounded to the dequant dtype on its own, no fused multiply-add); LoRA patches are added after it as for the legacy formats. | Read per tensor (`enhancers/gguf_dequant.py`, needs `WEIGHTFERRY_GGUF_TRITON`). The same bits as ComfyUI-GGUF on synthetic blocks of every K-quant in every dequant / output dtype, on all 400 Q4_K / Q6_K tensors of Wan 2.1 14B Q4_K_M, with LoRAs, and streamed at 4 GB. RTX 4060: dequantisation ×8.0; Wan 2.1 14B Q4_K_M streamed at 4 GB, 832×480: alternating in one process 2.81 → 1.91 s/it (median, ×1.47, 4/4, the same latents); separate processes 2.89 / 2.90 → 2.03 / 2.10 s/it (×1.40). |
| `WEIGHTFERRY_ACT_KV_ONCE` | `1` (on since 2026-10-12) | `0` = SageAttention's preparation in every call as before | Activation streaming (images computed by token rows, e.g. the 2x upscale without tiles at a low VRAM limit) with SageAttention and the fused block: SageAttention's key / value preparation (the key mean, the smoothed int8 keys, the fp8 values) is done once per block instead of again for every row chunk against the same keys and values. The same bits. | Read at import (`enhancers/fused_block.py` `rows_attention`, `enhancers/sage_fused.py` `prepare_kv`). RTX 4060, Anima int8 1792×2944, CFG 5, 4 GB, rows forced (`WEIGHTFERRY_ACT_CHUNK=1`), alternating in one process with the order swapped each round: 3.18–3.21 → 2.39 s/it (×1.34, 3/3), the same latents every round, the process's VRAM peak 1426 → 1362 MB (the 4 GB limit: a ballast tensor holds the rest of the 8 GB card; no overflow into shared memory); the rows path bit-exact with ComfyUI's own DiT (residual on the GPU / in pinned RAM). |
| `WEIGHTFERRY_USDU_LOCAL_COMPOSITE` | `1` (on since 2026-10-12) | `0` = the whole-image mask and composite as USDU 1.7.2 | Ultimate SD Upscale+: each tile's mask blur and composite on the tile's crop region (plus the blur's reach) instead of on the whole image, in both passes (one tile at a time, the default, and the seams fix; batched tiles). The same bytes. | Read at import (`usdu/image_utils.py`). CPU, synthetic images (1792×2944 / 2048×2048 / 1001×1499, tiles 896×1472 / 1024×1024 / 512×768, padding 0 / 32 / 64, blur 0–64, uniform tiles on / off, every tile, rectangles, seam-fix bands and the seams fix's gradient masks): 672 cases, 9702 tiles, identical; about 127 → 67 ms per tile. |
| `WEIGHTFERRY_UPSCALE_RRDB_APPEND` | `1` (on since 2026-10-12) | `0` = spandrel's concatenations as before | The upscale models of this package's half-precision upscale ("Upscale with model (half)" and Ultimate SD Upscale+'s upscale step) that are ESRGAN / RRDB (e.g. 4x-UltraSharp): the dense blocks' growing channel concatenations replaced by one buffer per block that each new output is copied into once; the same convolutions on the same inputs. The same bits. Only while this package's node runs its model (ComfyUI's own upscale node is not touched). | Read per call (`utils/upscale_half.py`). RTX 4060, 4x-UltraSharp float16 on 896×1472 (6 tiles of 512): the concatenations were 15.8% of a tile's GPU time; alternating in one process 7.60 → 6.86 s (×1.11), the same output bits; the example txt2img with Ultimate SD Upscale+ on, seed 1001: the same PNG bytes off / on. |
| `WEIGHTFERRY_UPSCALE_RRDB_EPILOGUE` | `0` (off; added 2026-10-12) | `1` = on | With `WEIGHTFERRY_UPSCALE_RRDB_APPEND` on and Triton available: the dense block's conv1..conv4 (Conv2d -> LeakyReLU) run the same convolution without its bias, and one kernel adds the bias, rounds, applies the LeakyReLU and writes straight into the append buffer (PyTorch: a separate bias add_, leaky_relu_ and the copy). The same bits. Other layer structures, hooks, autocast or gradients take the plain append path. | Read per call (`utils/upscale_half.py`). RTX 4060: the kernel against PyTorch on every float16 / bfloat16 bit pattern and float32 specials, one conv block in three dtypes with cuDNN autotuning off / on, a random RRDBNet and 4x-UltraSharp float32 / float16 through the node: no differing bit. 4x-UltraSharp float16 on 896×1472, alternating in one process: 5.91 → 5.71 s (×1.03), the same output bits, the same CUDA peak. |
| `WEIGHTFERRY_RAM_MARGIN_GB` | `6` | float, GB | RAM left free besides the RAM tier when the RAM tier is planned automatically (budget = available RAM − margin). | Read in `core/tiers/plan.py: plan_tiers`, `streaming/capture.py: _ram_tier_replan` and the on-load capture. Tiering Auto with automatic budget only. 6 GB = 4 GB kept free for the system + room for ComfyUI. |
| `WEIGHTFERRY_TE_RESERVE_GB` | `3` | float, GB | Extra RAM kept free when an idle streamed text encoder's buffers were released for the DiT, so that encoder can capture again on its next encode. | Read in on-load capture (`streaming/load_steps.py: _capture_cache`). Measured without it: free RAM fell to 3.07 GB when the encoder captured again. |
| `WEIGHTFERRY_TE_CACHE_GB` | `1` | float, GB | Size cap of the text-encode disk cache (CLIP Text Encode (Streamed LLM) with `disk_cache` on, folder `te_cache/` under `<ComfyUI base>/.cache/weightferry` or `WEIGHTFERRY_DISK_DIR`); the least recently used entries are deleted above it. | Read in `core/te_cache.py: store`. One entry is the conditioning of one prompt: 130 KB and 260 KB for the two prompts of the Anima draft example. |
| `WEIGHTFERRY_RAM_REPLAN_HYST_GB` | `1.5` | float, GB | On a reload, no RAM-tier re-plan while the new budget is within this much of the budget of the cached capture. | Read in `_ram_tier_replan`. Without it, small changes in the available RAM re-captured Cosmos 14B on every sampling call (+14–24 s). |
| `WEIGHTFERRY_AUTO_HEADROOM_MB` | `1536` | int, MB | Activation headroom that `auto_config` keeps free when deciding resident blocks, "fits: no streaming", and chunk streaming. | Read in `_auto_config` (apply time). |
| `WEIGHTFERRY_VRAM_HEADROOM_MB` | `1024` | int, MB | Headroom in the check "self-managed resident blocks + slots fit in free VRAM"; if not, that load falls back to ComfyUI-managed resident blocks. | Read in on-load capture. 8 self-managed blocks under a 2 GB limit spilled into shared memory, RAM 98%. |
| `WEIGHTFERRY_DISK_DIR` | unset = `<ComfyUI temp>/weightferry_cache` for the run's SSD-tier cache files (system temp if `folder_paths` is unavailable), `<ComfyUI base>/.cache/weightferry` for the caches kept across restarts | directory path | Where SSD-tier cache files are written; when set, also the caches kept across restarts (`persistent/`, `te_cache/`). | Read in `tiered_store.default_cache_dir`, `core/tiers/cache_files.py: persistent_root`. Put it on a fast SSD. The default for the kept caches is not the temp folder: ComfyUI deletes its temp folder at every start. |
| `WEIGHTFERRY_DISK_THREADS` | unset = fastest of 1 / 2 / 4 threads measured by the disk probe | int ≥ 1 | Number of reader threads of the SSD ring. | Read in `plan_tiers`. |
| `WEIGHTFERRY_PERSIST_GB` | `0` (off) | float, GB (size cap) | > 0: the SSD-tier cache file is kept across ComfyUI restarts in `<ComfyUI base>/.cache/weightferry/persistent` (`WEIGHTFERRY_DISK_DIR/persistent` when that is set) and reused while model file, LoRA patches and layout match; least recently used entries are deleted above the cap. Not used when the SSD tier reads the model file directly. | Read in `streaming/load_steps.py: _persist_lookup`; the cap is applied by `tiered_store.persist_enforce_cap`. Re-capture 16.5 → 5.8 s, no SSD writes; off by default because it keeps several GB on disk. Recommended when the same LoRA is used repeatedly. |
| `WEIGHTFERRY_QUANT_LORA` | `auto` (see notes) | `bake`, `transfer`, `auto` | LoRA on scaled-fp8 weights when the SSD tier could read the model file: `bake` = re-quantize once at capture (cache file), `transfer` = read the model file and re-quantize after every transfer, `auto` = pick by estimated total time. | Read in on-load capture, `_direct_file_layout`, `_quant_lora_auto_mode`, `_quant_lora_transfer_vram`. `transfer`: ~2.3 s per forward and ~+2.6 GB VRAM on Cosmos 14B; not usable at 2 GB. With `auto_config`, `transfer` needs its VRAM reserved, which happens only when this variable is set to `transfer` / `auto` explicitly; left unset, an automatic choice of `transfer` falls back to `bake`. |
| `WEIGHTFERRY_SSD_WRITE_GBPS` | `0.3` | float, GB/s | SSD write speed assumed by `WEIGHTFERRY_QUANT_LORA=auto`. | Read in `_quant_lora_auto_mode`. Measured on a consumer NVMe SSD (10 GB in 35 s, including its slowdown after the write cache fills). |
| `WEIGHTFERRY_QUANT_LORA_MS_PER_MB` | `0.23` | float, ms per MB | Re-quantization cost per MB of block assumed by the auto estimate. | Read in `_quant_lora_auto_mode`. 2.3 s per forward for 27 × 375 MB (Cosmos 14B). |
| `WEIGHTFERRY_QUANT_LORA_FORWARDS` | `60` | int | Expected forwards per run in the auto estimate. | Read in `_quant_lora_auto_mode`. 30 steps with CFG. |
| `WEIGHTFERRY_QUANT_LORA_VRAM_MB` | unset = 24 bytes × elements of the largest LoRA'd scaled-fp8 weight | int, MB | Extra VRAM `auto_config` reserves for `transfer` mode. Only used when `WEIGHTFERRY_QUANT_LORA` is set to `transfer` or `auto`. | Read in `_quant_lora_transfer_vram`. +2.6 GB measured, ~24 B/element. |
| `WEIGHTFERRY_CHUNK_MB` | `96` | float, MB | Maximum chunk size in chunk streaming. | Read in on-load capture. |
| `WEIGHTFERRY_CHUNK_SLOTS` | `4` | int ≥ 2 | Number of chunk slots (+1 spare) in chunk streaming. | Read in on-load capture. |
| `WEIGHTFERRY_CHUNK_RECALIBRATE` | unset (off) | `1` | Chunk streaming learns the layers' call order in a calibration pass on every load. Default (unset): learned once per model and reused by later loads (ComfyUI loads the model again for every generation). | Read in `_ChunkState.use_learned_order`. Cosmos-Predict2 14B, Colab L4: calibration pass 6.1 s against 2.2 s for a chunked one. |
| `WEIGHTFERRY_AUTO` | unset (off) | `1` | Same as the node option `auto_config`: picks resident blocks / tiering from usable VRAM, skips streaming if the model fits. | Read in `apply_streaming`. |
| `WEIGHTFERRY_TIERING` | unset = node option `tiering` | `auto`; any other non-empty value = RAM only | Overrides the node's `tiering`. | Read in `apply_streaming`. For experiments / benchmarks. |
| `WEIGHTFERRY_RAM_BUDGET_GB` | unset = node option `ram_budget_gb` | float, GB; `0` = automatic | Overrides the node's RAM-tier cap (Tiering Auto). | Read in `apply_streaming`. |
| `WEIGHTFERRY_OVERFLOW_SLACK_MB` | `384` | int, MB | Pinned memory not tracked by the overflow detector that is tolerated before an overflow is reported. | Read at import in `core/vram_monitor.py`. Detector: 1 GB pinned → +1088 MB "Shared Usage". |
| `WEIGHTFERRY_SAGE_PV` | unset (= `auto`) | `fp16`, `fp8`, `auto` | `fp16`: at import, replace ComfyUI's SageAttention with the FP16 P·V kernel process-wide (same as node option `sage_pv_fp16`). `fp8`: keep the FP8 kernel even with `WEIGHTFERRY_ACT_PER_ITEM=1`, which otherwise switches to FP16 automatically. | Read in `streaming/sage.py` (import, `_sage_fp16_for_per_item`). FP8 P·V gives NaN on all-zero V channels (fp8 Cosmos 14B). The default stays FP8; FP16 is +8–9% time and not closer to the PyTorch reference. |
| `WEIGHTFERRY_LLM_COPY_THREADS` | `4` | int ≥ 1 (1 = no thread pool) | Threads for the CPU copy into pinned staging (32 MB pieces). | Read in `core/llm_stream.py: __init__`. |
| `WEIGHTFERRY_LLM_STAGING` | `3` | int ≥ 2 | Number of pinned staging buffers. | Read in `llm_stream.__init__`. ~0.7 s of a 1.69 s encode was waiting for a free staging buffer; 3 buffers + early release → 1.30 s (Qwen2.5-VL 7B). |

## 2. Low-VRAM / large-image switches (11)

Off (or automatic) by default; switch on for a small GPU or very large latents.

| Variable | Default | Values | What it does | Notes |
|---|---|---|---|---|
| `WEIGHTFERRY_CHUNK` | unset = only when `auto_config` finds that whole-block slots do not fit | `1` force, `0` never | Chunk (sub-block) streaming: the first forward learns the layer call order, later forwards prefetch ~96 MB chunks. Needs Double Buffer and no per-transfer conversion; lets block 0 stream too. | Read in `apply_streaming`. With 8 GB the example workflow's model fits, so it is not used there. At 4 GB, no-tiles 2× upscale: 87.1 s with chunks vs 80.0 s with block streaming. |
| `WEIGHTFERRY_ACT_CHUNK` | `auto` | `1`, `0`, `auto` | Activation streaming. `auto`: installed with block and chunk streaming; used from the block after a VRAM overflow is noticed, and from the start for images of at least 8192 tokens when one block's activations (about 18 × batch × tokens × channels × 2 bytes) do not fit in the free VRAM (logged once per image size). `1`: always. `0`: never. Falls back for video, attention/MLP patches and short sequences. | Read in on-load capture (`_install_row_chunking`). Cosmos 14B, 2 GB, 1792×2944: 80–111 → 25 s per forward; ~15% slower where nothing overflows. |
| `WEIGHTFERRY_ACT_CHUNK_MIN_TOKENS` | `0` (no minimum) | int, tokens = (image width / 16) × (image height / 16) | Activation streaming only for latents with at least this many tokens. | Read at import. `8192` = only the no-tiles 2× upscale (1792×2944 = 20608 tokens), not the 896×1472 base (5152). Forcing activation streaming for everything at 4 GB: base 9 → 18 s. |
| `WEIGHTFERRY_ACT_CHUNK_ROWS` | `2048` | int, rounded down to a multiple of 128 (min 128) | Tokens per chunk in activation streaming. | Read in on-load capture. `4096` is part of the 4 GB recipe below. |
| `WEIGHTFERRY_ACT_HOST` | `auto` | `1` = residual stream always in pinned RAM; `0` = always on the GPU; `auto` = on the GPU when its two copies fit in the free VRAM with 64 MB to spare, else pinned RAM (decided once per image size) | Where activation streaming keeps the residual stream between blocks. `0` also skips the activation RAM reserve. | Read in `_act_host_wanted` (`streaming/activation.py`, per block call) and on-load capture (activation RAM reserve). No-tiles 2× upscale: at 4 GB the GPU is faster (80.0 → 77.7 s), at 2 GB it overflows. The pinned-RAM round trips were 70% of activation streaming's cost; `auto` at 4 GB 69.3 → 65.4 s, byte-identical images; Colab L4 −11 to −25% per forward. |
| `WEIGHTFERRY_ACT_PER_ITEM` | `0` (batch computed as a batch) | `1` | Activation streaming computes a CFG batch one item at a time, so K/V exist for one item only. Also switches SageAttention to FP16 P·V (unless `WEIGHTFERRY_SAGE_PV=fp8`). | Read per block call. 896x1472, cfg 4, 2 GB: 25.5–26.8 s vs 35–467 s per forward batched; not bit-exact with ComfyUI under SageAttention FP8 P·V. |
| `WEIGHTFERRY_TAYLOR_CACHE_CPU_MIN_TOKENS` | `0` (automatic) | int, tokens | TaylorSeer keeps its cache (inputs, residuals, factors) in CPU RAM instead of `cache_device` for inputs with at least this many tokens. `0`: automatic -- for inputs of at least 8192 tokens on a CUDA `cache_device`, CPU RAM when the cache (about 3 × the input) needs more than half of the free VRAM (logged once per image size). | Read at import in `enhancers/taylor_seer.py`. The no-tiles 2× upscale holds ~0.5–0.7 GB there (cond + uncond), which overflowed a 2 GB GPU. |
| `WEIGHTFERRY_MLP_CHUNK` | `auto` | `1`, `0`, `auto` | Feed-forward layers computed in token-row chunks. `auto`: with chunk streaming, only token-wise MLPs whose weights are all scaled fp8 (bit-identical). `1`: any dtype (warning: bf16 not always bit-identical). | Read in on-load capture. MLP hidden activation = 402 of 553 MB block peak (Cosmos 14B); 7% slower, −375 MB VRAM. |
| `WEIGHTFERRY_MLP_CHUNK_ROWS` | `1024` | int, multiple of 128 (min 128) | Rows per MLP chunk. | Read in on-load capture. |
| `WEIGHTFERRY_ATTN_CHUNK` | `auto` | `1`, `0`, `auto` | Self-attention computed by query rows (bit-identical). `auto`: only with chunk streaming **and** after an earlier run of this model overflowed. | Read in on-load capture. Self-attention peak 337 → 277 MB but process VRAM only −23 MB, ~11% slower. Row splitting did not help the 4 GB no-tiles upscale (98.5 vs 86.3 s). The default `auto` turns it on only in the case above; forcing it on is not worth it. |
| `WEIGHTFERRY_ATTN_CHUNK_ROWS` | `1024` | int, multiple of 128 (min 128) | Query rows per self-attention chunk. | Read in on-load capture. |

### 2× upscale without tiles (1792×2944) at 4 GB and 2 GB

Measured on an RTX 4060 with the example workflow `examples/anima_txt2img.json` (upscale switch set to "no tiles",
workflow values unchanged), VRAM limited with a PyTorch allocation cap (an error instead of the Windows shared-memory
spill).

No variables are needed: activation streaming and the TaylorSeer RAM cache switch themselves on from the free VRAM
(see `WEIGHTFERRY_ACT_CHUNK` / `WEIGHTFERRY_TAYLOR_CACHE_CPU_MIN_TOKENS` above).

| VRAM | Time per image (warm) |
|---|---|
| no limit | 51.5 s |
| 4 GB | 61.0 s |
| 2 GB | 74.3 s |

- Quality: LPIPS ~0.008 against the run without a limit (the images look the same; the cause has not been pinned
  down).
- Setting the activation variables by hand did not beat the defaults (best at 4 GB: 65.9 s with
  `WEIGHTFERRY_ACT_CHUNK=1`, `WEIGHTFERRY_ACT_HOST=0`, `WEIGHTFERRY_ACT_CHUNK_ROWS=4096`).
- With the same cap, USDU (tiled) + TaylorSeer is faster (4 GB 52 s, 2 GB 65 s) and is the recommended low-VRAM
  upscale.

## 3. Debug, profiling and benchmark overrides (6)

Logging, tracing, profiling and overrides for benchmarks. Some add synchronization and therefore cost time.

| Variable | Default | Values | What it does | Notes |
|---|---|---|---|---|
| `WEIGHTFERRY_PROFILE` | unset (off) | `1` | CUDA-event timeline per streamed block (DMA, conversion, compute, stall, CPU time in hooks); summary logged when the state is released. Block streaming only. | Read in on-load capture. |
| `WEIGHTFERRY_CHUNK_PROFILE` | unset (off) | `1` | Per-forward GPU timeline of chunk streaming (copy start/end, compute waits, SSD waits). | Read in `_ChunkState.begin_forward`. |
| `WEIGHTFERRY_CHUNK_LOG` | unset (off) | `1` | Per-forward timing log of chunk streaming. Synchronizes the CPU with the GPU at the end of each forward. | Read in `_ChunkState._log_forward`. |
| `WEIGHTFERRY_TAYLOR_DEBUG` | unset (off) | `1` | One log line per TaylorSeer compute/skip decision. | Read at import in `enhancers/taylor_seer.py`. |
| `WEIGHTFERRY_PREFETCH_PRIORITY` | unset (default stream priority) | int, lower = higher (RTX 4060: 0 … −3) | Priority of the weight-prefetch CUDA stream. | Read in `_prefetch_stream`. No difference (within noise); transfers are already hidden. |
| `WEIGHTFERRY_SLOTS` | unset = node option `prefetch_slots` | int ≥ 2 | Overrides the number of GPU slots in block-streaming Double Buffer mode (and in the self-managed fit check). | Read in on-load capture. For benchmarks. |

## 4. On by default: off-switches and alternatives (20)

Improvements and fixes that are on by default; the listed value turns one off (for comparisons and as an escape
hatch; `WEIGHTFERRY_SSD_LAYOUT` and `WEIGHTFERRY_USDU_UPSCALE_DTYPE` choose between alternatives). Most are a speed or
memory improvement rather than a correctness fix; the notes say which.

| Variable | Default | Values | What it does | Notes |
|---|---|---|---|---|
| `WEIGHTFERRY_OUTSIDE_LEAF` | `1` | `0` | Chunk streaming: a layer called before any block (TaylorSeer's block-0 adaLN) uses a small buffer instead of starting the chunk pipeline. | Read in `_bind_outside`, `leaf_post`. Improvement: removes 12 unused chunk prefetches on 4 skipped steps; bit-exact. |
| `WEIGHTFERRY_ACT_RAM_RESERVE` | `1` | `0` | With chunk streaming, the pinned activation buffers of the coming run are reserved in the RAM-tier plan. | Read in on-load capture. +2.2 GB free RAM, +1.4% time. |
| `WEIGHTFERRY_RAM_REPLAN` | on | `0` | On a reload, re-capture when available RAM changed the RAM tier by ≥ max(2 blocks, 10%) and ≥ 1 GB. | Read in `_ram_tier_replan`. Cosmos 14B at 2 GB: 4.3 → 3.9 s per forward. |
| `WEIGHTFERRY_TRIM_WS` | on (Windows) | `0` | Moves this process' working set to standby (EmptyWorkingSet) before measuring available RAM. | Read in `tiered_store.available_ram`. Clean mmap pages hid RAM: 15.6 GB measured instead of ~18 GB. |
| `WEIGHTFERRY_VIRTUALALLOC` | `1` (Windows) | `0` | Pinned host buffers from VirtualAlloc / VirtualFree, so released memory returns to the OS. `0` = PyTorch's allocator. | Read once in `tiered_store._virtual_alloc_api`. PyTorch's allocator returned only 0.26 GB after unregister. |
| `WEIGHTFERRY_RELEASE_TE` | on | `0` | Before a DiT capture with automatic RAM budget, releases capture caches of streamed text encoders that are not encoding. | Read in `_release_idle_text_encoder_caches`. An encoder kept 6.2 GB pinned, so the DiT got 17 RAM-tier blocks instead of ~35. |
| `WEIGHTFERRY_OVERFLOW_CHECK` | `1` | `0` | Per-forward detection of VRAM overflow into shared GPU memory (Windows): warning, `auto_config` penalty on the next load, and the trigger for automatic activation streaming / attention chunking. | Read in `check_overflow`. |
| `WEIGHTFERRY_DIRECT_FILE` | `1` | `0` | Tiering Auto: the SSD tier reads the model's safetensors file directly (no cache file) when its bytes are exactly the compute weights. | Read in on-load capture. Improvement. |
| `WEIGHTFERRY_DIRECT_LORA` | `1` | `0` | Allows the direct file source with a cast-time LoRA applied on the GPU after each transfer. | Read in on-load capture. Improvement. |
| `WEIGHTFERRY_SSD_ASYNC` | `1` | `0` | SSD-tier uploads on a background thread (block streaming, Double Buffer). | Read in on-load capture. Improvement: without it, ~950 ms of CPU time per forward is blocked in hooks. |
| `WEIGHTFERRY_QUANT_SELF` | `1` | `0` | Self-managed resident blocks also for scaled-fp8 weights; `0` falls back to ComfyUI-managed resident blocks. | Read in on-load capture. Improvement: 118 vs 135 ms per block (124 vs 195 ms with a LoRA), Cosmos 14B. |
| `WEIGHTFERRY_SSD_LAYOUT` | `interleave` | `interleave`, `tail` | Which streaming blocks go to the SSD tier: spread evenly, or the last N (`tail`). | Read in `tiered_store.ssd_positions` and the capture cache key. With `tail` the SSD reads do not overlap compute. |
| `WEIGHTFERRY_CHUNK_BLOCK0` | `1` | `0` | With chunk streaming and `resident_blocks` ≤ 1, block 0 streams too; `0` keeps it resident. | Read in `apply_streaming`. Improvement: Cosmos 14B's 375 MB block 0 was most of what overflowed at 2 GB. |
| `WEIGHTFERRY_FUSED_BLOCK` | `1` (all parts) | `0` = ComfyUI's block, or a comma-separated list of parts: `mod`, `qkv`, `nhd`, `modq`, `gelu`, `rms`, `res`, `ln`, `tay`, `smk` | Cosmos-Predict2 / Anima DiT block with fused element-wise work, bit-exact with ComfyUI (`enhancers/fused_block.py`); the parts are listed [below](#fused-block-parts-weightferry_fused_block). | Read at import. One Anima block, int8 ConvRot: 32.6 → 26.8 ms; more measurements below. |
| `WEIGHTFERRY_GGUF_TRITON` | `1` | `0` = ComfyUI-GGUF's own code | ComfyUI-GGUF's dequantisation of the legacy formats Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 as one Triton kernel instead of its chain of PyTorch operations, with the same arithmetic and roundings (`enhancers/gguf_dequant.py`); K-quants and other formats keep ComfyUI-GGUF's code. Plain LoRA patches are applied in the same kernel after comfy's own `torch.mm` (the same bits). | Read at import (ComfyUI-GGUF must be loaded first, as ComfyUI's name order does). Bit-exact with ComfyUI-GGUF on every tensor of Anima Q8_0 / Q5_1 / Q4_1 and with the streaming node at 4 GB. RTX 4060, Anima resident 896×1472, CFG 5, alternating in one process, latents identical: Q5_1 2.37 → 1.67 s/it (×1.42), Q5_1 + LoRA ×1.31, Q4_1 ×1.19, Q8_0 ×1.03; the LoRA in the kernel too ×1.011 (Q5_1) / ×1.012 (Q8_0) / ×1.005 (Q4_1) more, every layer bit-exact. |
| `WEIGHTFERRY_GGUF_KEEP_MMAP` | `1` | `0` = ComfyUI-GGUF's RAM copy as before | GGUF models (ComfyUI-GGUF): the streamed weights stay memory-mapped from the .gguf file instead of ComfyUI-GGUF's private RAM copy, so the RAM goes to the pinned RAM tier. Layers that ComfyUI-GGUF's patcher put on the GPU before a capture get CPU copies for their SSD-tier blocks (that RAM is kept free) instead of disabling the SSD tier. | Read in `apply_streaming` and on-load capture. RTX 4060, 32 GB RAM, VRAM capped at 4 GB, Wan 2.1 14B Q8_0 (15.9 GB): without it RAM ran short (streamed without the switch: all blocks on the SSD, 4.99 s/it; ComfyUI-GGUF alone: 4.04 s/it); with it 2.35–2.56 s/it, bit-exact with ComfyUI-GGUF. Cosmos-Predict2 14B Q8_0 + LoRA 896×1472: 9.10 vs 10.06 s/it. |
| `WEIGHTFERRY_ACT_H2D_STREAM` | `1` | `1`, `0` | Activation streaming with the residual stream in pinned RAM: the next row chunk is copied to the GPU on its own stream while the current one is computed (two buffers), instead of on the compute stream when needed. Values unchanged. `0`: copied on the compute stream when needed. | Read per block in `_row_chunks` (`streaming/activation.py`). Colab L4, one Cosmos 14B forward: 3.39 → 3.18 s (896×1472), 15.2 → 14.3 s (1792×2944), bit-exact. |
| `WEIGHTFERRY_ACT_D2H_STREAM` | `1` | `1`, `0` | Activation streaming writes rows back to pinned RAM on a separate device-to-host stream. | Read in `_act_d2h_stream`. No speed-up at 896×1472 or 1792×2944 (20.67 vs 20.65 s); the write-back is already hidden. |
| `WEIGHTFERRY_USDU_UPSCALE_DTYPE` | `float16` | `float32` = ComfyUI's, `bfloat16` | The upscale model inside Ultimate SD Upscale+ (its `upscale_model` input) in this precision through this package's Upscale Image (using Model, half precision) node (same tiling 512 / overlap 32; `usdu/a1111/upscaler.py`). | Read per call. RTX 4060, 4x-UltraSharp on 896×1472: 13.0 → 5.8 s (2× image PSNR 56 dB against float32); a txt2img workflow with Ultimate SD Upscale 45.7 → 41.6 s per image, final image PSNR 32.8 dB / LPIPS 0.012 against float32 (the redraw amplifies the difference); float32 = byte-identical to ComfyUI's USDU. int8 / fp8 simulated on the convolutions: 41.8 / 30.0 dB, not pursued. |
| `WEIGHTFERRY_LORA_REMAP` | `1` (on) | `0`, `off`, `false`, `no` disable | Remaps 28-block Anima LoRAs (anima-base / preview) onto the 40-block Anima 2.9B when a LoRA is loaded. | Read in `core/lora_remap.py: patched_load_lora`, at every LoRA load. |


### Fused block parts (`WEIGHTFERRY_FUSED_BLOCK`)

Each part gives the same bits as ComfyUI's code. The parts marked (Triton) need Triton; without it they stay off and
the other parts still apply.

* `mod` (Triton): the scale-and-shift after each LayerNorm as one kernel.
* `qkv`: the self-attention's q / k / v inputs quantised once (int8 ConvRot weights).
* `nhd`: SageAttention in the projections' layout (no output copy).
* `modq` / `gelu` (Triton): the scale-and-shift and the MLP's GELU inside the int8 activation quantiser (a Triton
  reproduction of comfy_kitchen's; GELU from a table of PyTorch's own GELU of all bf16 values).
* `rms` (Triton): the cross-attention's q RMSNorm with PyTorch's order of operations.
* `res`: the gated residual (`x + gate * out`) inside the output projections' int8 GEMM (comfy_kitchen's CUTLASS
  residual epilogue, the same bits as PyTorch's addcmul).
* `ln` (Triton): the LayerNorm inside the quantiser too (PyTorch's vectorized layer-norm kernel reproduced).
* `tay` (Triton): TaylorSeer's Taylor prediction and its addition in one pass (same order and bf16 rounding);
  TaylorSeer's per-step decision input also uses the one-pass LayerNorm + scale-and-shift.
* `smk` (Triton): SageAttention's preparation of Q, K and V without intermediate copies
  (`enhancers/sage_fused.py`). On RTX 40 GPUs `sageattn()` smooths the keys (`k - mean(k)`, written as a copy and
  read back); here the key quantiser subtracts the mean itself. V's transposed / padded / permuted bf16 copy and its
  per-channel fp8 quantisation become one kernel. The queries' normalisation is done inside the query quantiser:
  the cross-attention's RMSNorm, and the self-attention's RMSNorm + RoPE (comfy_kitchen's `rms_rope_split_half`
  reproduced in Triton from its source; the keys are written by the same reproduction). Only when ComfyUI's call
  would end in SageAttention's sm89 FP8 P·V kernel; otherwise ComfyUI's call (a pending query is normalised first).
  Same bits as kitchen + `sageattn` (5 shapes, keys of 77 to 5152 tokens, an all-zero V channel included).

The int8 activation quantiser runs with 2 warps per row (1.5–2.2× faster than 8 at Anima's widths, the same bits) and
also quantises the output projections' plain inputs (the same bits as comfy_kitchen's quantiser at every width
256–8192).

The package patches ComfyUI's `Block` / `Attention` / `GPT2FeedForward` classes at import; activation streaming's
row-chunked block uses the same parts.

Measured on an RTX 4060 (every image byte-identical with and without the parts):

* One Anima block (896×1472, CFG 5), 9 alternating timings: int8 ConvRot 32.6 → 26.8 ms (−18%, all parts;
  `mod,qkv,nhd` alone −10%), scaled fp8 42.9 → 40.6 ms (−6%, `mod,nhd`).
* A full txt2img workflow (int8, streaming node, `modq,gelu` included): 12.4 → 11.4 / 13.3 → 12.3 s per image.
* 2× upscale without tiles at 4 GB (activation streaming): 50.4 → 45.8 s per image.
* Resident Anima int8 (896×1472, plain sampler), 9 rounds of 4 steps alternating in one process, the latents
  identical (2026-10-11): 1.0.0 → 2-warp quantiser ×1.060 (CFG 5) / ×1.057 (CFG 1); plain inputs too ×1.001 /
  ×1.011 on top; `smk` ×1.026 / ×1.020 on top (keys ×1.013 / ×1.013, V ×1.013 / ×1.007); the queries'
  normalisation inside the quantiser ×1.018 / ×1.015 (cross-attention) and ×1.014 / ×1.009 (self-attention RoPE)
  more, within the run-to-run spread for the last.

---

## Not an environment variable

`ANIMA_COEFFICIENTS` in `enhancers/taylor_seer.py` is a constant (TaylorSeer's
polynomial rescaling coefficients).
