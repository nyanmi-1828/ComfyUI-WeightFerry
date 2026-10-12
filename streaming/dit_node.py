# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
The WeightFerryDiT node: apply_streaming checks the model and the settings and registers the model-load
callback, whose steps (layouts, tiers, capture, state, hooks) are in streaming/load_steps.py.
"""

import time
import weakref

import torch

import comfy.model_management
import comfy.patcher_extension as patcher_extension
from comfy.patcher_extension import CallbacksMP, WrappersMP
from ..utils.logging import log_info, log_warn
from ..core import env as _env
from ..core import tiered_store
from ..core import vram_budget
from ..core import vram_monitor
from .common import (HOST_FORMAT_BAKED, HOST_FORMAT_STORAGE, RESIDENT_MODE_COMFY, RESIDENT_MODE_SELF, TIERING_AUTO,
                     TIERING_RAM, _ACTIVE_ATTR, _BLOCK_LIST_ATTRS, _STATE_ATTR, _TE_ROOTS, _WRAPPER_KEY, _blocks,
                     _param_nbytes, _raw)
from .activation import _track_input_shape
from .teardown import _teardown
from .sage import _apply_sage_attention, _install_sage_pv_fp16, _sage_fp16_for_per_item
from .auto_config import _auto_config
from .load_steps import (_Capture, _Layout, _LoadContext, _build_state, _capture_cache, _capture_host_buffers,
                         _check_block_layouts, _direct_file_source, _fit_self_managed, _host_layout, _install_hooks,
                         _lora_functions, _rest_binding, _scaled_fp8_params, _slot_layout)


class WeightFerryDiT:
    """
    Layer streaming for a ComfyUI DiT: the transformer blocks that do not stay resident are streamed from pinned
    RAM / the SSD into the GPU while the model runs (also used by WeightFerryTextEncoder for text encoders).
    - Late Offload: captures weights AFTER ModelPatcher prepared LoRA / quantization (CallbacksMP.ON_LOAD).
    - VRAM reduction: streamed blocks share 2-6 GPU slots (block streaming), or a few slots of layer groups
      (chunk streaming, for very small VRAM).
    - Double buffer async prefetch: one DMA per block, overlapped with compute (needs pinned host memory).
    Note: with the legacy ModelPatcher, ComfyUI still loads the full model to the GPU before ON_LOAD.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled (True VRAM Streaming)",
                    "label_off": "Disabled (Full GPU)",
                    "tooltip": "Enable layer streaming: DiT blocks are streamed between host RAM and the GPU while the model runs"
                }),
                "pin_memory": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Pinned RAM (Max Speed DMA)",
                    "label_off": "Pageable RAM (Save Host RAM)",
                    "tooltip": "True: asynchronous DMA from page-locked (pinned) RAM. False: pageable memory (transfers become effectively synchronous)"
                }),
                "buffer_mode": (
                    ["Double Buffer (Async Prefetch, Recommended)", "Single Buffer (Synchronous)"],
                    {
                        "default": "Double Buffer (Async Prefetch, Recommended)",
                        "tooltip": "Double: prefetch the next blocks asynchronously so transfers overlap the computation. Single: synchronous transfers"
                    }
                ),
                "resident_blocks": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Blocks kept resident in VRAM (minimum 1: block 0 always stays). At least the block count: no streaming"
                }),
                "sage_attention": ("BOOLEAN", {
                    "default": False,
                    "label_on": "SageAttention 2 (INT8 Max Speed)",
                    "label_off": "Keep current attention",
                    "tooltip": "Inject SageAttention 2 into every block. Note: it also applies to other clones sharing the diffusion "
                               "model (USDU / detailers) and is not undone when switched off. Not needed with --use-sage-attention"
                }),
            },
            "optional": {
                "host_format": ([HOST_FORMAT_BAKED, HOST_FORMAT_STORAGE], {
                    "default": HOST_FORMAT_BAKED,
                    "tooltip": "Format of the weights in host RAM. Compute dtype: LoRA baked in (more bytes to transfer, no per-forward work; "
                               "faster with LoRA). Storage dtype (fp8): half the transfer, LoRA re-applied on the GPU every forward "
                               "(for very large transfer-bound models without LoRA)"
                }),
                "tiering": ([TIERING_RAM, TIERING_AUTO], {
                    "default": TIERING_RAM,
                    "tooltip": "Where the streamed blocks live. RAM: all in pinned RAM. Auto: measures free RAM, SSD speed and compute "
                               "time and splits the blocks into a RAM tier and an SSD tier (a cache file read ahead by background "
                               "threads); for machines with little RAM. With resident_mode = Self-managed the resident blocks' "
                               "copy (read once per load) also goes to the SSD when RAM is short"
                }),
                "ram_budget_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 512.0,
                    "step": 0.5,
                    "tooltip": "RAM tier limit (GB) with tiering Auto. 0: automatic (free RAM minus a 6 GB margin)"
                }),
                "resident_mode": ([RESIDENT_MODE_COMFY, RESIDENT_MODE_SELF], {
                    "default": RESIDENT_MODE_COMFY,
                    "tooltip": "How the resident blocks are managed. ComfyUI: left to ComfyUI's Dynamic VRAM. Self-managed: kept in "
                               "dedicated GPU buffers with LoRA baked in; unaffected by Dynamic VRAM write-backs when VRAM is short, "
                               "so runs give identical results, but their VRAM stays occupied for the whole generation"
                }),
                "sage_pv_fp16": ("BOOLEAN", {
                    "default": False,
                    "label_on": "SageAttention: FP16 P*V (NaN-safe)",
                    "label_off": "SageAttention: default kernel",
                    "tooltip": "Use SageAttention's FP16 P*V kernel (global, not undone). The Ada default (FP8) returns NaN when a "
                               "channel of V is all zeros (seen with Cosmos 14B in fp8). Also enabled by WEIGHTFERRY_SAGE_PV=fp16"
                }),
                "auto_config": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto (VRAM / RAM / SSD)",
                    "label_off": "Manual",
                    "tooltip": "Auto: chooses the resident block count from the free VRAM and the block size, and sets tiering to Auto. "
                               "No streaming when every block fits in VRAM (faster). Overrides resident_blocks / tiering"
                }),
                "prefetch_slots": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 6,
                    "step": 1,
                    "tooltip": "GPU slots with Double Buffer (blocks prefetched + 1). 3: fastest (measured 0.94x of no streaming, "
                               "one more slot of VRAM). 2: less VRAM (0.90x). One slot = one block (about 132 MB for Anima 2B)"
                }),
                "vram_budget_gb": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 512.0,
                    "step": 0.5,
                    "tooltip": "Most VRAM ComfyUI may use (GB); the automatic choices (resident blocks, activation streaming, "
                               "caches, VAE tiles, ComfyUI's model loads) plan inside it, so only the speed changes, not the "
                               "image. 0: WEIGHTFERRY_VRAM_BUDGET_GB, else no budget (all free VRAM, fastest)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_streaming"
    CATEGORY = "WeightFerry/streaming"
    TITLE = "⚡ DiT Layer Streaming (low VRAM)"

    def apply_streaming(
        self,
        model,
        enabled: bool = True,
        pin_memory: bool = True,
        buffer_mode: str = "Double Buffer (Async Prefetch, Recommended)",
        resident_blocks: int = 8,
        sage_attention: bool = False,
        host_format: str = HOST_FORMAT_BAKED,
        prefetch_slots: int = 3,
        resident_mode: str = RESIDENT_MODE_COMFY,
        tiering: str = TIERING_RAM,
        ram_budget_gb: float = 0.0,
        auto_config: bool = False,  # the node's default is on (INPUT_TYPES); direct calls (tests) choose the mode
        sage_pv_fp16: bool = False,
        vram_budget_gb: float = 0.0,
        _root=None,
        _key_prefix=None,
        _text_encoder=False,
    ):
        """_root / _key_prefix / _text_encoder (internal, WeightFerryTextEncoder): stream the layer list of another
        module of this ModelPatcher (``model`` is then a ModelPatcher used as is, not cloned), with patch keys under
        ``_key_prefix``; the forward of ``_root`` is wrapped instead of the diffusion-model wrapper."""
        if sage_pv_fp16:
            _install_sage_pv_fp16()
        _sage_fp16_for_per_item()
        if not enabled:
            log_info("DiT Layer Streaming: Disabled, passing original model.")
            return (model,)

        # clone() copies the ModelPatcher only; diffusion_model itself is shared with other clones
        model_patcher = model if _text_encoder else model.clone()
        if _env.get("WEIGHTFERRY_GGUF_KEEP_MMAP") == "1" and getattr(model_patcher, "mmap_released", None) is False:
            # ComfyUI-GGUF's patcher copies every layer it does not put on the GPU out of the file's mmap into private
            # RAM on its first load; the streaming keeps those CPU originals as the rest binding, so a 15.9 GB GGUF
            # costs 15.9 GB of RAM on top of the pinned tier. Marked as released, the layers stay file-backed and
            # the capture reads the blocks from the file's pages.
            model_patcher.mmap_released = True
            log_info("DiT Layer Streaming: GGUF weights stay memory-mapped (WEIGHTFERRY_GGUF_KEEP_MMAP=1).")
        dit = _root if _root is not None else getattr(model_patcher.model, "diffusion_model", None)
        if dit is not None:
            dit._weightferry_key_prefix = _key_prefix or "diffusion_model."
            if _text_encoder:
                _TE_ROOTS.add(dit)
        if dit is None:
            log_warn("DiT Layer Streaming: Model has no 'diffusion_model', skipping.")
            return (model,)

        device = model_patcher.load_device
        if not comfy.model_management.is_device_cuda(device):
            log_warn("DiT Layer Streaming: Device is not CUDA, skipping.")
            return (model_patcher,)
        _track_input_shape(model_patcher, dit)

        if sage_attention:
            # Shared module: affects every clone of this model. Disabling leaves the current function untouched
            # (it may be a model-provided int8 kernel set from the checkpoint metadata).
            _apply_sage_attention(dit)

        blocks_list = _blocks(dit)
        if len(blocks_list) == 0:
            log_warn(f"DiT Layer Streaming: no transformer block list ({', '.join(_BLOCK_LIST_ATTRS)}) on "
                     f"{type(dit).__name__}, skipping.")
            return (model_patcher,)
        num_blocks = len(blocks_list)
        # Chunk (sub-block) streaming: WEIGHTFERRY_CHUNK=1 forces it, =0 disables it; auto_config enables it when
        # whole-block slots do not fit
        chunk_wanted = _env.get("WEIGHTFERRY_CHUNK") == "1"
        # VRAM that auto_config kept for re-quantizing LoRA'd scaled fp8 weights after each transfer (None: auto_config
        # not used). The load below may pick "transfer" only when it was kept
        quant_lora_reserved_mb = None
        if not _text_encoder:
            vram_budget.set_node_budget(vram_budget_gb)
        vram_budget.apply(device, "DiT")
        if auto_config or _env.get("WEIGHTFERRY_AUTO") == "1":
            auto_resident, auto_info = _auto_config(dit, device, prefetch_slots, "Double Buffer" in buffer_mode,
                                                    patcher=model_patcher)
            log_info(f"DiT Layer Streaming [auto]: {auto_info}")
            quant_lora_reserved_mb = auto_info.get("quant_lora_transfer_mb", 0)
            if auto_resident is None:
                log_info("DiT Layer Streaming [auto]: the whole model fits in VRAM; streaming not used "
                         "(ComfyUI is faster then).")
                return (model_patcher,)
            resident_blocks = auto_resident
            tiering = TIERING_AUTO
            if auto_info.get("chunk") and _env.get("WEIGHTFERRY_CHUNK") != "0":
                chunk_wanted = True
        # Block 0 stays resident in block streaming: caching wrappers (TaylorSeer / FBCache style) call its submodules
        # directly, bypassing the block-level hooks, and its pre-hook starts the per-step prefetch.
        # Chunk streaming hooks every layer, so a direct call is covered and block 0 can stream too (with
        # Cosmos-Predict2 14B at 2 GB, its 375 MB resident block 0 was most of what overflowed). Only when block 0 has
        # the same parameter layout as the others (not the T5 encoder, whose first block holds the position bias).
        # WEIGHTFERRY_CHUNK_BLOCK0=0 keeps block 0 resident.
        uniform0 = (num_blocks > 1 and [n for n, _ in blocks_list[0].named_parameters()]
                    == [n for n, _ in blocks_list[1].named_parameters()])
        stream_block0 = (chunk_wanted and uniform0 and int(resident_blocks) <= 1
                         and _env.get("WEIGHTFERRY_CHUNK_BLOCK0") != "0")
        resident_k = 0 if stream_block0 else max(1, min(num_blocks, int(resident_blocks)))
        if int(resident_blocks) < 1 and not stream_block0:
            log_info("DiT Layer Streaming: resident_blocks raised to 1 (block 0 must stay on the GPU).")
        if stream_block0:
            log_info("DiT Layer Streaming [chunks]: block 0 streams too (chunk streaming hooks every layer).")
        num_stream = num_blocks - resident_k
        # WEIGHTFERRY_RESIDENT_SPREAD (2026-10-13, web research 2.1; off by default): block 0 resident (as always) and
        # the other resident blocks spread evenly between the streamed ones instead of 1 .. k-1, so the transfer of a
        # streamed block can overlap a resident block's compute (vLLM / sglang PrefetchOffloader, musubi-tuner); the
        # same blocks run in the same order
        resident_idx = list(range(resident_k))
        if _env.get("WEIGHTFERRY_RESIDENT_SPREAD") == "1" and 1 < resident_k < num_blocks:
            rest = resident_k - 1
            picks = sorted({1 + int((j + 0.5) * (num_blocks - 1) / rest) for j in range(rest)})
            resident_idx = [0] + [min(num_blocks - 1, i) for i in picks]
            resident_idx = sorted(set(resident_idx))
        resident_set = set(resident_idx)
        stream_idx = [i for i in range(num_blocks) if i not in resident_set]
        num_stream = len(stream_idx)
        is_double = "Double Buffer" in buffer_mode

        if num_stream <= 0:
            log_info("DiT Layer Streaming: All blocks resident, no streaming offload registered.")
            return (model_patcher,)

        if is_double and not pin_memory:
            log_warn(
                "DiT Layer Streaming: Double Buffer with pageable memory cannot overlap DMA "
                "(non_blocking copies from pageable RAM are synchronous)."
            )

        # Overrides for experiments / benchmarks: WEIGHTFERRY_TIERING=auto|ram, WEIGHTFERRY_RAM_BUDGET_GB
        env_tiering = _env.get("WEIGHTFERRY_TIERING")
        tiering_auto = (env_tiering == "auto") if env_tiering else tiering == TIERING_AUTO
        log_info(
            f"DiT Layer Streaming: Registering Late Offload ({resident_k}/{num_blocks} resident, "
            f"{num_stream} streaming blocks, pin_memory={pin_memory}, double_buffer={is_double}, "
            f"host_format={'storage' if host_format == HOST_FORMAT_STORAGE else 'baked'}, "
            f"prefetch_slots={prefetch_slots if is_double else 1}, "
            f"resident_mode={'self' if resident_mode == RESIDENT_MODE_SELF else 'comfy'}, "
            f"tiering={'auto' if tiering_auto else 'ram'})..."
        )

        # Unique token identifying this patcher's streaming configuration
        owner = object()
        use_storage_dtype = host_format == HOST_FORMAT_STORAGE
        self_resident = resident_mode == RESIDENT_MODE_SELF
        env_budget = _env.get("WEIGHTFERRY_RAM_BUDGET_GB")
        ram_budget_bytes = int(float(env_budget if env_budget else ram_budget_gb) * 1024 ** 3) or None

        def on_load_streaming_late_offload(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            target_device = device_to if device_to is not None else patcher.load_device
            if not comfy.model_management.is_device_cuda(target_device):
                return

            # Any previous state (ours from an earlier load, or another config's) is discarded.
            # Thanks to the rest-binding invariant, parameters now hold real, unaliased weights.
            _teardown(dit, "re-load")
            vram_budget.apply(target_device, "DiT load")

            streaming_blocks = [blocks_list[i] for i in stream_idx]
            ctx = _LoadContext(
                dit=dit, patcher=patcher, owner=owner, target_device=target_device,
                compute_dtype=getattr(patcher.model, "manual_cast_dtype", None) or torch.bfloat16,
                blocks_list=blocks_list, streaming_blocks=streaming_blocks, resident_k=resident_k,
                num_blocks=num_blocks, num_stream=num_stream, pin_memory=pin_memory, is_double=is_double,
                prefetch_slots=prefetch_slots, chunk_wanted=chunk_wanted, tiering_auto=tiering_auto,
                ram_budget_bytes=ram_budget_bytes, quant_lora_reserved_mb=quant_lora_reserved_mb,
                text_encoder=_text_encoder, use_storage_dtype=use_storage_dtype,
                resident_idx=resident_idx, stream_idx=stream_idx)
            lay = _Layout(resident_list=[blocks_list[i] for i in resident_idx] if self_resident else [],
                          template_params=list(streaming_blocks[0].parameters()))

            # The block layout: checks, scaled fp8, GPU slots, host RAM, direct file reads, self-managed fit, LoRA
            _check_block_layouts(ctx, lay)
            _scaled_fp8_params(ctx, lay)
            cap = _Capture(t_capture=time.perf_counter())
            mem_start = tiered_store.mem_snapshot()
            _slot_layout(ctx, lay)
            _host_layout(ctx, lay)
            _direct_file_source(ctx, lay)
            _fit_self_managed(ctx, lay)
            _lora_functions(ctx, lay)

            # The host side: capture cache, RAM / SSD tiers and the capture itself, rest binding
            _capture_cache(ctx, lay, cap)
            _capture_host_buffers(ctx, lay, cap)
            _rest_binding(ctx, lay, cap)

            state, use_chunk, num_slots, staging_flat = _build_state(ctx, lay, cap)
            resident_list, rest, res_rest = lay.resident_list, cap.rest, cap.res_rest

            # Bind every streaming / self-managed resident param to its rest tensor (releases GPU originals)
            moved_off_gpu = 0
            for blk, blk_rest in zip(streaming_blocks + resident_list, rest + res_rest):
                for p, r in zip(blk.parameters(), blk_rest):
                    if _raw(p.data).is_cuda and not _raw(r).is_cuda:
                        moved_off_gpu += _param_nbytes(p)
                    p.data = r
            torch.cuda.empty_cache()
            is_dynamic = getattr(patcher, "is_dynamic", None)
            if moved_off_gpu and not (callable(is_dynamic) and is_dynamic()):
                # ComfyUI's classic ModelPatcher (--disable-dynamic-vram; ComfyUI-GGUF's patcher always) loaded these
                # layers before ON_LOAD and still counts them as on the GPU. When a later sampling re-uses the model,
                # load_models_gpu adds that count to the free VRAM and moves the streamed layers back to the GPU
                # (model_use_more_vram), which can run out of memory on the second sampling. Correct its count, and
                # mark the layers as not loaded.
                patcher.model.model_loaded_weight_memory = max(
                    0, getattr(patcher.model, "model_loaded_weight_memory", 0) - moved_off_gpu)
                for blk in streaming_blocks + resident_list:
                    for m in blk.modules():
                        if getattr(m, "comfy_patched_weights", False):
                            m.comfy_patched_weights = False
                log_info(f"DiT Layer Streaming: ComfyUI's classic loader told that {moved_off_gpu / 1024 ** 2:.0f} "
                         f"MB of streamed weights left the GPU.")

            capture_ms = (time.perf_counter() - cap.t_capture) * 1000
            state.capture_ms = capture_ms
            if cap.reused:
                detail = "reused cached host buffers"
            else:
                detail = f"captured, {cap.converted} tensors converted"
                if any(cap.cap_t.values()):
                    detail += " (" + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in cap.cap_t.items()) + ")"
            n_lora = sum(1 for fns in lay.lora_fns for f in fns if f is not None)
            mem_end = tiered_store.mem_snapshot()
            state.mem_snapshots = {"start": mem_start, "end": mem_end}
            log_info(f"DiT Layer Streaming [memory]: capture start {mem_start} -> end {mem_end}")
            disk_store = cap.disk_store
            if state.ring is not None:
                log_info(f"DiT Layer Streaming [tiers]: {state.ram_blocks} streaming blocks in pinned RAM, "
                         f"{disk_store.n_blocks} on SSD ("
                         f"{'model file' if disk_store.is_model_file else 'cache file'} {disk_store.path}), "
                         f"ring depth {state.ring.depth}, "
                         f"{state.ring.threads} reader thread(s), "
                         f"{'background' if state.uploader is not None else 'in-hook'} uploads "
                         f"({state.ring.depth * disk_store.buf_bytes / 1024 ** 2:.0f} MB pinned).")
            host_bytes, flat_bytes = lay.host_bytes, lay.flat_bytes
            log_info(
                f"DiT Layer Streaming [Late Offload]: {num_stream} blocks offloaded to host RAM "
                f"({detail}, {capture_ms:.0f} ms). Per block: host {host_bytes / 1024 ** 2:.1f} MB "
                f"({len(lay.convert_idx)} params in storage dtype, {n_lora} LoRA patches applied on GPU), "
                + (f"GPU slots {num_slots} x {flat_bytes / 1024 ** 2:.1f} MB" if not use_chunk else
                   f"chunk streaming: {num_slots} chunk slots of up to "
                   f"{float(_env.get('WEIGHTFERRY_CHUNK_MB')):g} MB from the second forward on (the first "
                   f"one learns the layer call order)")
                + (f" + staging {host_bytes / 1024 ** 2:.1f} MB" if staging_flat is not None else "")
                + (f"; {len(resident_list)} resident blocks self-managed on GPU "
                   f"({len(resident_list) * flat_bytes / 1024 ** 2:.0f} MB)." if resident_list else ".")
            )

            _install_hooks(ctx, state, use_chunk)

            setattr(dit, _STATE_ATTR, state)

        def on_detach_release(patcher, unpatch_all=True, *args, **kwargs):
            state = getattr(dit, _STATE_ATTR, None)
            if state is not None and state.owner is owner:
                _teardown(dit, "detach")

        # The wrapper lives in this patcher's model_options: a strong reference to the patcher would make a cycle, so
        # the patcher would die only in a full garbage collection together with its parent, and ComfyUI's loaded-model
        # entry could not move to the parent ("Potential memory leak detected", a full garbage collection at every
        # later model load: +1.7 s per image after a model switch)
        patcher_ref = weakref.ref(model_patcher)

        def streaming_diffusion_model_wrapper(executor, *args, **kwargs):
            state = getattr(dit, _STATE_ATTR, None)
            if state is None or state.owner is not owner:
                # ON_LOAD was not called for this run: the legacy ModelPatcher skips load() when switching back to
                # a clone whose weights / patches are already in place (streaming would silently stop after a
                # clone switch). The weights are ready now, so capture lazily (cached: ~35 ms).
                mp = patcher_ref()
                if mp is None:
                    return executor(*args, **kwargs)
                on_load_streaming_late_offload(mp, mp.load_device, 0, False, True)
                state = getattr(dit, _STATE_ATTR, None)
                if state is None or state.owner is not owner:
                    return executor(*args, **kwargs)
                log_info("DiT Layer Streaming: ON_LOAD was skipped by ComfyUI; captured lazily at forward.")
            setattr(dit, _ACTIVE_ATTR, owner)
            prof = state.profiler
            fwd_shape = ring_wait0 = None  # set in the try; still None if it stopped before (no timing then)
            try:
                state.begin_forward()
                state.bind_resident()
                # Light timing for the tier planner: fold in earlier forwards' block spans that have completed
                state.harvest_block_timing(dit)
                state.span_start = state.span_end = None
                x = args[0] if args else None
                fwd_shape = tuple(x.shape) if isinstance(x, torch.Tensor) else None
                ring_wait0 = state.ssd_main_wait_s()
                if prof is not None:
                    prof.begin_forward(torch.cuda.current_stream(state.device))
                return executor(*args, **kwargs)
            finally:
                if prof is not None:
                    prof.end_forward(torch.cuda.current_stream(state.device))
                setattr(dit, _ACTIVE_ATTR, None)
                state.restore_rest_binding()
                if ring_wait0 is not None:
                    ring_wait = state.ssd_main_wait_s() - ring_wait0
                    state.finish_forward_timing(fwd_shape, ring_wait)
                state.end_forward_tiers()
                try:
                    state.check_overflow(dit)
                except Exception:  # noqa: BLE001 -- monitoring never breaks a generation
                    pass

        vram_monitor.prewarm()  # the overflow check's first query, off the generation's critical path
        model_patcher.add_callback(CallbacksMP.ON_LOAD, on_load_streaming_late_offload)
        model_patcher.add_callback(CallbacksMP.ON_DETACH, on_detach_release)
        model_patcher._weightferry_stream_owner = owner

        if _text_encoder:
            # No wrapper mechanism for text encoders: the layer stack's forward is wrapped (the module is shared by
            # every clone of the CLIP, so only calls made through the streaming CLIP, which sets _weightferry_te_caller,
            # stream; any other clone runs ComfyUI's normal path)
            orig_forward = getattr(dit, "_weightferry_orig_forward", None) or dit.forward
            dit._weightferry_orig_forward = orig_forward

            def te_forward(*args, **kwargs):
                caller = getattr(dit, "_weightferry_te_caller", None)
                if caller is None:
                    return orig_forward(*args, **kwargs)
                if caller is not owner:
                    return orig_forward(*args, **kwargs)
                return streaming_diffusion_model_wrapper(orig_forward, *args, **kwargs)

            dit.forward = te_forward
            log_info("Text Encoder Layer Streaming: hooked; streaming starts when this CLIP encodes.")
            return (model_patcher,)

        transformer_options = model_patcher.model_options.setdefault("transformer_options", {})
        wrappers = transformer_options.setdefault("wrappers", {})
        wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {}).pop(_WRAPPER_KEY, None)
        patcher_extension.add_wrapper_with_key(
            WrappersMP.DIFFUSION_MODEL,
            _WRAPPER_KEY,
            streaming_diffusion_model_wrapper,
            model_patcher.model_options,
            is_model_options=True,
        )

        log_info(
            "DiT Layer Streaming: Hooked to ModelPatcher. Offload will activate upon model load."
        )
        return (model_patcher,)
