# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
The steps of the model-load callback of WeightFerryDiT (on_load_streaming_late_offload), called in order by
dit_node.py. They share three bundles: _LoadContext (fixed for the load), _Layout (block layout) and _Capture (host
side); each step reads its inputs at the top and writes its results back at the end.
"""

import dataclasses
import json
import os
import time

import torch

import comfy.model_management
from ..utils.logging import log_info, log_warn
from ..core import env as _env
from ..core import tiered_store
from ..core import vram_budget
from .common import (_ACTIVE_ATTR, _CACHE_ATTR, _flat_layout, _is_quant, _lora_apply_fn, _param_nbytes, _param_owner,
                     _quant_lora_requant, _quant_params_to, _quant_seed, _raw, _storage_dtype, _views)
from .profiler import _StreamProfiler
from .ssd_uploader import _SsdUploader
from .state import _StreamState
from .chunk_state import _ChunkState
from .activation import (_act_host_reserve, _install_attn_chunking, _install_block_chunking, _install_mlp_chunking,
                         _mlp_weights_quantized)
from .teardown import _release_idle_text_encoder_caches
from .capture import (_block_file_source, _direct_file_layout, _effective_weight, _persist_key, _quant_lora_auto_mode,
                      _ram_tier_replan)

try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:  # pragma: no cover
    QuantizedTensor = ()


@dataclasses.dataclass
class _LoadContext:
    """What one model load works with and does not change: the model, the device, the block split and the node's
    settings (set by streaming/dit_node.py)."""
    dit: object
    patcher: object
    owner: object
    target_device: object
    compute_dtype: object
    blocks_list: list
    streaming_blocks: list
    resident_k: int
    num_blocks: int
    num_stream: int
    pin_memory: bool
    is_double: bool
    prefetch_slots: int
    chunk_wanted: bool
    tiering_auto: bool
    ram_budget_bytes: object
    quant_lora_reserved_mb: object
    text_encoder: bool
    use_storage_dtype: bool
    resident_idx: list = None   # the resident blocks' indices (default: 0 .. resident_k - 1)
    stream_idx: list = None     # the streamed blocks' indices, in order (default: resident_k .. num_blocks - 1)

    def __post_init__(self):
        if self.resident_idx is None:
            self.resident_idx = list(range(self.resident_k))
        if self.stream_idx is None:
            self.stream_idx = list(range(self.resident_k, self.num_blocks))


@dataclasses.dataclass
class _Layout:
    """How a block is laid out: the parameters, their slots on the GPU and their bytes in host RAM / on the SSD.
    Filled step by step (scaled fp8, slot layout, host layout, direct file source, self-managed fit, LoRA)."""
    resident_list: list
    template_params: list
    quant_idx: list = None
    n_quant_lora: int = 0
    slot_specs: list = None
    slot_offsets: list = None
    flat_bytes: int = 0
    names: list = None
    host_dtypes: list = None
    convert_idx: list = None
    host_specs: list = None
    host_offsets: list = None
    host_bytes: int = 0
    direct: object = None
    lora_fns: list = None


@dataclasses.dataclass
class _Capture:
    """The host side of the load: the capture cache, the host buffers per block (RAM tier) and the SSD stores, the
    rest binding and the self-managed resident blocks' GPU buffers."""
    t_capture: float
    patches_uuid: object = None
    cache_key: object = None
    cache: object = None
    reused: bool = False
    ram_margin: object = None
    host_flats: list = None
    host_views: list = None
    quant_params: list = None
    res_quant_params: list = None
    res_host_flats: list = None
    res_host_views: list = None
    plan: object = None
    disk_store: object = None
    res_disk_store: object = None
    holders: list = None
    converted: int = 0
    cap_t: dict = None
    rest: list = None
    res_gpu_flats: list = None
    res_gpu_views: list = None
    res_rest: list = None
    n_res_disk: int = 0


def _check_block_layouts(ctx, lay):
    """Every streamed / self-managed block has the template's parameter layout, no meta parameter and no GPU storage
    shared with another block."""
    streaming_blocks = ctx.streaming_blocks
    resident_list, template_params = lay.resident_list, lay.template_params
    seen_ptrs = set()
    for blk in streaming_blocks + resident_list:
        params = list(blk.parameters())
        if len(params) != len(template_params) or any(
            p.shape != t.shape or p.dtype != t.dtype for p, t in zip(params, template_params)
        ):
            raise RuntimeError(
                "DiT Layer Streaming: streaming blocks have heterogeneous parameter layouts; "
                "shared slots cannot be used."
            )
        for p in params:
            if getattr(p, "is_meta", False):
                raise NotImplementedError("DiT Layer Streaming: meta parameters are not supported.")
            raw_p = _raw(p.data)
            if raw_p.device.type == "cuda":
                ptr = raw_p.data_ptr()
                if ptr in seen_ptrs:
                    raise RuntimeError(
                        "DiT Layer Streaming: detected aliased block weights (shared GPU storage). "
                        "Weights would be corrupted; please restart ComfyUI."
                    )
                seen_ptrs.add(ptr)


def _scaled_fp8_params(ctx, lay):
    """Indices of the scaled fp8 (QuantizedTensor) parameters of a block and how many carry a LoRA. Results:
    quant_idx, resident_list, n_quant_lora (set on the bundles); resident_list is emptied with
    WEIGHTFERRY_QUANT_SELF=0."""
    streaming_blocks = ctx.streaming_blocks
    resident_list, template_params = lay.resident_list, lay.template_params
    n_quant_lora = 0
    # Scaled fp8 (QuantizedTensor) parameters: streamed as their stored bytes, scales kept on the GPU. A LoRA on
    # them is applied like ComfyUI does at cast time (dequantize, LoRA, re-quantize: _quant_lora_requant),
    # baked into the host buffers at capture, or on the GPU after each transfer when the SSD tier reads the
    # model file. Self-managed resident blocks hold the fp8 bytes (LoRA baked) in their GPU buffer and get the
    # same parameter-object swap for the whole forward call (ComfyUI-managed resident blocks compute slower than
    # streamed ones, Cosmos-Predict2 14B with a LoRA: 195 vs 124 ms per block).
    # WEIGHTFERRY_QUANT_SELF=0 falls back to ComfyUI-managed resident blocks.
    quant_idx = [j for j, prm in enumerate(template_params) if _is_quant(prm.data)]
    if quant_idx:
        if resident_list and _env.get("WEIGHTFERRY_QUANT_SELF") == "0":
            log_info("DiT Layer Streaming: WEIGHTFERRY_QUANT_SELF=0; using ComfyUI-managed resident "
                     "blocks with scaled fp8 weights.")
            resident_list = []
        q_names = [n for n, _ in streaming_blocks[0].named_parameters()]
        n_quant_lora = 0
        for blk in list(streaming_blocks) + list(resident_list):
            for j in quant_idx:
                module, param_key = _param_owner(blk, q_names[j])
                if getattr(module, param_key + "_lowvram_function", None) is None:
                    continue
                if not hasattr(module, "_v") or _quant_seed(module) is None:
                    raise NotImplementedError(
                        "DiT Layer Streaming: a LoRA on a scaled fp8 weight outside ComfyUI's Dynamic "
                        "VRAM cast path is not supported; disable streaming for this model.")
                n_quant_lora += 1
        log_info(f"DiT Layer Streaming: {len(quant_idx)} scaled fp8 (QuantizedTensor) parameters per "
                 f"block, streamed as their fp8 bytes; scales kept on the GPU"
                 + (f"; LoRA on {n_quant_lora} of them (re-quantized like ComfyUI)." if n_quant_lora else "."))
    lay.quant_idx, lay.resident_list, lay.n_quant_lora = quant_idx, resident_list, n_quant_lora


def _slot_layout(ctx, lay):
    """(shape, dtype) of every parameter in the GPU slots (fp8 parameters with a LoRA promoted to the compute dtype)
    and the flat layout. Results: slot_specs, slot_offsets, flat_bytes (set on the bundles)."""
    streaming_blocks, compute_dtype = ctx.streaming_blocks, ctx.compute_dtype
    resident_list, template_params, quant_idx = lay.resident_list, lay.template_params, lay.quant_idx
    slot_specs = [(tuple(_raw(p.data).shape), _raw(p.data).dtype) for p in template_params]
    # LoRA on fp8 parameters (e.g. an fp8 model file): ComfyUI's Dynamic VRAM path casts the weight to the
    # compute dtype, applies the LoRA there and computes with that result directly. Storing the result in
    # the fp8 parameter dtype would round it (the weights and the forward would no longer match ComfyUI). Such
    # parameters get the
    # compute dtype in the slots and host buffers; the module then receives a compute-dtype weight.
    param_names = [n for n, _ in streaming_blocks[0].named_parameters()]
    promoted = set()
    for j, (shape, dt) in enumerate(slot_specs):
        if not dt.is_floating_point or dt.itemsize >= compute_dtype.itemsize or j in quant_idx:
            continue  # (a scaled fp8 weight with LoRA stays fp8: ComfyUI re-quantizes the LoRA result)
        for blk in list(streaming_blocks) + list(resident_list):
            module, param_key = _param_owner(blk, param_names[j])
            if getattr(module, param_key + "_lowvram_function", None) is not None:
                promoted.add(j)
                break
    if promoted:
        slot_specs = [(shape, compute_dtype if j in promoted else dt) for j, (shape, dt) in enumerate(slot_specs)]
        log_info(f"DiT Layer Streaming: {len(promoted)} fp8 parameters with LoRA kept in "
                 f"{compute_dtype} in the slots (no rounding of the LoRA result back to fp8).")
    slot_offsets, flat_bytes = _flat_layout(slot_specs)
    lay.slot_specs, lay.slot_offsets, lay.flat_bytes = slot_specs, slot_offsets, flat_bytes


def _host_layout(ctx, lay):
    """Layout of a block in host RAM (the storage dtype when every streamed block agrees on it and host_format is
    the storage dtype). Results: names, host_dtypes, convert_idx, host_specs, host_offsets, host_bytes (set on the
    bundles)."""
    streaming_blocks, use_storage_dtype = ctx.streaming_blocks, ctx.use_storage_dtype
    slot_specs = lay.slot_specs
    # Host dtype per parameter: the storage dtype when every streamed block agrees on it
    names = [n for n, _ in streaming_blocks[0].named_parameters()]
    host_dtypes = []
    for j, (shape, dtype) in enumerate(slot_specs):
        chosen = None
        for blk in streaming_blocks:
            module, param_key = _param_owner(blk, names[j])
            sd = _storage_dtype(module, param_key, list(blk.parameters())[j].data)
            if blk is streaming_blocks[0]:
                chosen = sd
            elif sd != chosen:
                chosen = None
                break
        host_dtypes.append((chosen if use_storage_dtype else None) or dtype)
    convert_idx = [j for j, (spec, hd) in enumerate(zip(slot_specs, host_dtypes)) if hd != spec[1]]
    host_specs = [(shape, hd) for (shape, _), hd in zip(slot_specs, host_dtypes)]
    host_offsets, host_bytes = _flat_layout(host_specs)
    lay.names, lay.host_dtypes, lay.convert_idx, lay.host_specs = names, host_dtypes, convert_idx, host_specs
    lay.host_offsets, lay.host_bytes = host_offsets, host_bytes


def _fit_self_managed(ctx, lay):
    """resident_list, or [] (ComfyUI-managed resident blocks for this load) when the self-managed resident blocks
    plus the slots and a headroom do not fit in the free VRAM."""
    dit, is_double, prefetch_slots, target_device = ctx.dit, ctx.is_double, ctx.prefetch_slots, ctx.target_device
    resident_list, flat_bytes, host_bytes = lay.resident_list, lay.flat_bytes, lay.host_bytes
    convert_idx = lay.convert_idx
    # Self-managed resident blocks cannot be evicted by Dynamic VRAM. If they (plus the streaming slots and
    # a headroom for activations) do not fit in the free VRAM, fall back to ComfyUI-managed resident blocks
    # for this load: on Windows an overflow does not raise OOM but silently spills into shared GPU memory
    # (system RAM).
    # Headroom override: WEIGHTFERRY_VRAM_HEADROOM_MB (default 1024).
    if resident_list:
        planned_slots = max(2, int(prefetch_slots)) if is_double else 1
        if is_double and _env.get("WEIGHTFERRY_SLOTS"):
            planned_slots = max(2, int(_env.get("WEIGHTFERRY_SLOTS")))
        headroom = int(_env.get("WEIGHTFERRY_VRAM_HEADROOM_MB")) * 1024 ** 2
        need = (len(resident_list) + planned_slots) * flat_bytes \
            + (host_bytes if convert_idx else 0) + headroom
        free = comfy.model_management.get_free_memory(target_device)
        free -= int(getattr(dit, "_weightferry_vram_overflow_bytes", 0) or 0)  # overflow seen earlier (check_overflow)
        free -= vram_budget.kept_free()  # VRAM kept free by the VRAM budget (0 without one)
        if need > free:
            log_warn(
                f"DiT Layer Streaming: self-managed resident blocks do not fit in free VRAM "
                f"(need {need / 1024 ** 2:.0f} MB = {len(resident_list)} resident + {planned_slots} slots "
                f"x {flat_bytes / 1024 ** 2:.0f} MB + headroom {headroom / 1024 ** 2:.0f} MB, free "
                f"{free / 1024 ** 2:.0f} MB). Falling back to ComfyUI-managed resident blocks for this load; "
                f"lower resident_blocks to use Self-managed under this VRAM budget."
            )
            resident_list = []
    lay.resident_list = resident_list


def _lora_functions(ctx, lay):
    """Per streaming block, the LoRA function of every converted parameter (None elsewhere)."""
    streaming_blocks, target_device = ctx.streaming_blocks, ctx.target_device
    names, convert_idx = lay.names, lay.convert_idx
    # LoRA functions for converted parameters (re-read every load: ComfyUI recreates them)
    lora_fns = []
    for blk in streaming_blocks:
        fns = [None] * len(names)
        for j in convert_idx:
            module, param_key = _param_owner(blk, names[j])
            fns[j] = _lora_apply_fn(module, param_key, target_device)
        lora_fns.append(fns)
    lay.lora_fns = lora_fns


def _span_event(state):
    """A timing event recorded now on the compute stream (block span for the tier planner)."""
    ev = torch.cuda.Event(enable_timing=True)
    ev.record(torch.cuda.current_stream(state.device))
    return ev


def _install_hooks(ctx, state, use_chunk):
    """The forward hooks of this load (no-op unless this patcher's wrapper is active) and the optional row
    chunking of MLPs / self-attention / whole blocks. Registration order (it matters when block 0 streams: its
    first-block pre-hook runs before its chunk pre-hook): first-block hook, resident profiler hooks, chunk streaming
    hooks, row chunking, block streaming hooks."""
    _install_first_block_hook(ctx, state)
    if state.profiler is not None:
        _install_resident_profiler_hooks(ctx, state)
    if use_chunk:
        _install_chunk_hooks(ctx, state)
    _install_row_chunking(ctx, state, use_chunk)
    if not use_chunk:
        _install_block_stream_hooks(ctx, state)


def _install_first_block_hook(ctx, state):
    """Pre-hook of block 0: starts the block span and the prefetch of the first slots (only when block 0 really
    runs, so cached steps cost no transfer)."""
    dit, owner = ctx.dit, ctx.owner

    def first_block_pre_hook(module, args):
        if getattr(dit, _ACTIVE_ATTR, None) is owner:
            t0 = time.perf_counter()
            if state.span_start is None:
                state.span_start = _span_event(state)
            state.prefetch_first_slots()
            if state.profiler is not None:
                state.profiler.add_cpu(time.perf_counter() - t0)
    state.handles.append(ctx.blocks_list[0].register_forward_pre_hook(first_block_pre_hook))


def _install_resident_profiler_hooks(ctx, state):
    """Profiler (WEIGHTFERRY_PROFILE=1): compute start / end of every resident block."""
    dit, owner, blocks_list = ctx.dit, ctx.owner, ctx.blocks_list

    def make_resident_hooks(b):
        def pre(module, args):
            if getattr(dit, _ACTIVE_ATTR, None) is owner:
                state.profiler.record_resident(b, "r_start", torch.cuda.current_stream(state.device))

        def post(module, args, output):
            if getattr(dit, _ACTIVE_ATTR, None) is owner:
                state.profiler.record_resident(b, "r_end", torch.cuda.current_stream(state.device))
        return pre, post

    for b_idx in ctx.resident_idx:
        pre_h, post_h = make_resident_hooks(b_idx)
        state.handles.append(blocks_list[b_idx].register_forward_pre_hook(pre_h))
        state.handles.append(blocks_list[b_idx].register_forward_hook(post_h))


def _install_chunk_hooks(ctx, state):
    """Chunk streaming: pre / post hooks on every streamed block and on each of its leaf modules."""
    dit, owner = ctx.dit, ctx.owner

    def make_chunk_block_hooks(cur_s):
        def chunk_block_pre(module, args):
            if getattr(dit, _ACTIVE_ATTR, None) is owner:
                state.block_pre(cur_s)

        def chunk_block_post(module, args, output):
            if getattr(dit, _ACTIVE_ATTR, None) is owner:
                if cur_s == state.num_stream - 1 and state.span_start is not None:
                    state.span_end = _span_event(state)
                state.block_post(cur_s)
                # VRAM overflow is also checked inside the forward, every 12th block (~20 ms per check, the
                # Windows performance counter): at its end the overflow may be gone already, so a forward
                # that overflowed throughout would go unnoticed
                if cur_s % 12 == 11:
                    try:
                        state.check_overflow(dit)
                    except Exception:  # noqa: BLE001 -- monitoring never breaks a generation
                        pass
        return chunk_block_pre, chunk_block_post

    def chunk_leaf_pre(module, args):
        if getattr(dit, _ACTIVE_ATTR, None) is owner:
            state.leaf_pre(module)

    def chunk_leaf_post(module, args, output):
        if getattr(dit, _ACTIVE_ATTR, None) is owner:
            state.leaf_post(module)

    for s_idx, blk in enumerate(ctx.streaming_blocks):
        pre_h, post_h = make_chunk_block_hooks(s_idx)
        state.handles.append(blk.register_forward_pre_hook(pre_h))
        state.handles.append(blk.register_forward_hook(post_h))
        for leaf_mod in state.leaf_modules[s_idx].values():
            state.handles.append(leaf_mod.register_forward_pre_hook(chunk_leaf_pre))
            state.handles.append(leaf_mod.register_forward_hook(chunk_leaf_post))


def _install_row_chunking(ctx, state, use_chunk):
    """Row chunking of feed-forward layers, whole blocks (activation streaming) and self-attention, each by its
    environment switch."""
    dit, owner, blocks_list = ctx.dit, ctx.owner, ctx.blocks_list
    # MLP row chunking (smaller activations; bit-identical, see _install_mlp_chunking):
    # WEIGHTFERRY_MLP_CHUNK=1 / 0 forces / disables it; by default on with chunk streaming (very small VRAM).
    # WEIGHTFERRY_MLP_CHUNK_ROWS (default 1024, rounded down to a multiple of 128).
    mlp_env = _env.get("WEIGHTFERRY_MLP_CHUNK")
    if mlp_env == "1" or (mlp_env == "auto" and use_chunk):
        rows = max(128, int(_env.get("WEIGHTFERRY_MLP_CHUNK_ROWS")) // 128 * 128)
        state.mlp_chunked = _install_mlp_chunking(dit, list(blocks_list), owner, rows,
                                                  any_dtype=(mlp_env == "1"))
        if state.mlp_chunked:
            log_info(f"DiT Layer Streaming: feed-forward layers computed in chunks of {rows} tokens "
                     f"({len(state.mlp_chunked)} modules; smaller activations).")
            if mlp_env == "1" and not all(_mlp_weights_quantized(m) for m in state.mlp_chunked):
                log_warn("DiT Layer Streaming: MLP row chunking forced on non-scaled-fp8 weights: bf16 "
                         "matmuls in row chunks are not always bit-identical to ComfyUI (they were for some chunk "
                         "sizes only).")
        else:
            log_info("DiT Layer Streaming: no feed-forward module eligible for row chunking "
                     "(only token-wise MLPs with scaled fp8 weights are chunked automatically).")

    # Whole blocks by token rows, the residual stream optionally in pinned RAM (activation streaming):
    # WEIGHTFERRY_ACT_CHUNK=1 / 0 forces / disables it. Automatic (default): installed for block and chunk
    # streaming; it only computes by token rows when a large image's block activations would not fit in the free
    # VRAM, or after a VRAM overflow (streaming/activation.py). WEIGHTFERRY_ACT_CHUNK_ROWS (default 2048, rounded
    # down to a multiple of 128); the residual stream stays on the GPU with WEIGHTFERRY_ACT_HOST=0.
    act_env = _env.get("WEIGHTFERRY_ACT_CHUNK")
    if act_env in ("1", "auto"):
        a_rows = max(128, int(_env.get("WEIGHTFERRY_ACT_CHUNK_ROWS")) // 128 * 128)
        state.block_chunked = _install_block_chunking(dit, list(blocks_list), owner, a_rows,
                                                      only_after_overflow=(act_env != "1"))
        if state.block_chunked and act_env == "1":
            log_info(f"DiT Layer Streaming: activation streaming ({a_rows}-token chunks, residual "
                     f"stream in pinned RAM) for {len(state.block_chunked)} blocks.")
    # Self-attention by query rows (bit-identical, see _cosmos_self_attention_rows): WEIGHTFERRY_ATTN_CHUNK=1 / 0
    # forces / disables it. By default only with chunk streaming AND after an earlier run of this model
    # overflowed into shared GPU memory: it saves little (Cosmos 14B at 2 GB: self-attention peak 337 -> 277 MB)
    # and costs time (+11 ms per block; ComfyUI's attention quantizes the full K again for every query chunk).
    # WEIGHTFERRY_ATTN_CHUNK_ROWS (default 1024, rounded down to a multiple of 128: 128-row attention tiles).
    attn_env = _env.get("WEIGHTFERRY_ATTN_CHUNK")
    overflowed = int(getattr(dit, "_weightferry_vram_overflow_bytes", 0) or 0) > 0
    if attn_env == "1" or (attn_env == "auto" and use_chunk and overflowed):
        rows = max(128, int(_env.get("WEIGHTFERRY_ATTN_CHUNK_ROWS")) // 128 * 128)
        state.attn_chunked = _install_attn_chunking(dit, list(blocks_list), owner, rows)
        if state.attn_chunked:
            log_info(f"DiT Layer Streaming: self-attention computed in query chunks of {rows} tokens "
                     f"({len(state.attn_chunked)} modules; smaller activations).")


def _install_block_stream_hooks(ctx, state):
    """Block streaming: per streamed block a pre-hook (wait for its transfer, bind the slot) and a post-hook (free
    the slot, prefetch the block that takes it next)."""
    dit, owner = ctx.dit, ctx.owner

    def make_pre_hook(cur_s):
        def stream_pre_hook(module, args):
            if getattr(dit, _ACTIVE_ATTR, None) is not owner:
                return
            t0 = time.perf_counter()
            prof = state.profiler
            if prof is not None:
                prof.mark(cur_s, "cpu_pre_hook")
            compute_stream = torch.cuda.current_stream(state.device)
            slot_idx = state.slot_of(cur_s)
            if state.is_double:
                if not state.issued[cur_s]:
                    # Prefetch chain was broken (e.g. a block was skipped): issue now
                    state.issue_prefetch(cur_s)
                if state.uploader is not None and state.disk_index[cur_s] >= 0:
                    # Just in time: the copy must be issued before the compute stream can wait on it
                    state.uploader.wait(cur_s)
                if prof is not None:
                    prof.record(cur_s, "wait0", compute_stream)
                compute_stream.wait_event(state.transferred_events[cur_s])
            else:
                # Same stream as compute: ordered after the previous reader of slot 0
                if prof is not None:
                    prof.record(cur_s, "wait0", compute_stream)
                state.materialize(cur_s, 0)
            if prof is not None:
                prof.record(cur_s, "start", compute_stream)
            if state.quant_bind is None:
                for p, sv in zip(module.parameters(), state.slot_views[slot_idx]):
                    p.data = sv
            else:
                prms = list(module.parameters())
                qset = state.quant_idx_set
                for j, (p, sv) in enumerate(zip(prms, state.slot_views[slot_idx])):
                    if j not in qset:
                        p.data = sv
                for q_module, q_key, q_param in state.quant_bind[cur_s]:
                    q_module._parameters[q_key] = q_param
            if prof is not None:
                prof.add_cpu(time.perf_counter() - t0)
        return stream_pre_hook

    def make_post_hook(cur_s):
        def stream_post_hook(module, args, output):
            if getattr(dit, _ACTIVE_ATTR, None) is not owner:
                return
            prof = state.profiler
            if prof is not None:
                prof.record(cur_s, "end", torch.cuda.current_stream(state.device))
            if cur_s == state.num_stream - 1 and state.span_start is not None:
                # Block span for the tier planner (the last block is always a streaming block)
                state.span_end = _span_event(state)
            if not state.is_double:
                return
            t0 = time.perf_counter()
            slot_idx = state.slot_of(cur_s)
            state.computed_events[slot_idx].record(torch.cuda.current_stream(state.device))
            next_s = cur_s + state.num_slots
            if next_s < state.num_stream and not state.issued[next_s]:
                state.issue_prefetch(next_s)
            if prof is not None:
                prof.add_cpu(time.perf_counter() - t0)
        return stream_post_hook

    for s_idx, blk in enumerate(ctx.streaming_blocks):
        state.handles.append(blk.register_forward_pre_hook(make_pre_hook(s_idx)))
        state.handles.append(blk.register_forward_hook(make_post_hook(s_idx)))


def _direct_file_source(ctx, lay):
    """Direct file source (tiering Auto): whether the SSD tier can read blocks straight from the model file (the
    slots / host layouts then follow the file), and how a LoRA on scaled fp8 weights is applied (bake / transfer).
    Results: direct, slot_offsets, flat_bytes, host_specs, host_dtypes, host_offsets, host_bytes, convert_idx (set
    on the bundles)."""
    dit, patcher, streaming_blocks, resident_k = ctx.dit, ctx.patcher, ctx.streaming_blocks, ctx.resident_k
    num_blocks, num_stream, quant_lora_reserved_mb = ctx.num_blocks, ctx.num_stream, ctx.quant_lora_reserved_mb
    tiering_auto, ram_budget_bytes, target_device = ctx.tiering_auto, ctx.ram_budget_bytes, ctx.target_device
    compute_dtype = ctx.compute_dtype
    resident_list, template_params, quant_idx = lay.resident_list, lay.template_params, lay.quant_idx
    n_quant_lora, slot_offsets, flat_bytes = lay.n_quant_lora, lay.slot_offsets, lay.flat_bytes
    host_specs, host_dtypes, host_offsets = lay.host_specs, lay.host_dtypes, lay.host_offsets
    host_bytes, convert_idx = lay.host_bytes, lay.convert_idx
    # Direct file source: slots follow the file layout and SSD-tier blocks are read from the model file
    direct = None
    if tiering_auto and not convert_idx and _env.get("WEIGHTFERRY_DIRECT_FILE") != "0":
        all_blocks = list(resident_list) + list(streaming_blocks)
        all_idx = list(ctx.resident_idx) if resident_list else []
        all_idx += list(ctx.stream_idx)
        # LoRA on scaled fp8 weights: bake, transfer, or auto (estimate)
        quant_lora_mode = _env.get("WEIGHTFERRY_QUANT_LORA")
        if quant_lora_mode == "auto":
            quant_lora_mode = "bake"
            if quant_idx and n_quant_lora:
                try:
                    pre_plan = tiered_store.plan_tiers(
                        num_stream, host_bytes, ram_budget_bytes=ram_budget_bytes,
                        compute_ms_per_block=getattr(dit, "_weightferry_stream_block_ms", None),
                        total_blocks=num_blocks, resident_blocks=len(resident_list),
                        resident_block_bytes=flat_bytes)
                    quant_lora_mode, q_info = _quant_lora_auto_mode(num_stream, pre_plan["ram_blocks"],
                                                                    host_bytes)
                    log_info(f"DiT Layer Streaming: LoRA on scaled fp8 weights, auto mode: {q_info}")
                except Exception as exc:  # noqa: BLE001 -- keep the default
                    log_warn(f"DiT Layer Streaming: auto bake/transfer estimate failed ({exc}); baking.")
                # auto_config reserves the transfer mode's extra VRAM only when WEIGHTFERRY_QUANT_LORA is set
                # to transfer / auto explicitly (streaming/capture.py: _quant_lora_transfer_vram); without that
                # reservation transfer could overflow the VRAM auto_config planned, so bake
                if quant_lora_mode == "transfer" and quant_lora_reserved_mb == 0:
                    quant_lora_mode = "bake"
                    log_info("DiT Layer Streaming: transfer mode needs VRAM that auto_config did not keep "
                             "(set WEIGHTFERRY_QUANT_LORA=auto or transfer to reserve it); baking instead.")
        try:
            direct, why = _direct_file_layout(patcher, dit, all_blocks, all_idx, target_device, compute_dtype,
                                              allow_lora=_env.get("WEIGHTFERRY_DIRECT_LORA") != "0",
                                              quant_lora_mode=quant_lora_mode)
        except Exception as exc:  # noqa: BLE001 -- fall back to the cache file
            direct, why = None, f"error: {exc}"
        if direct is None:
            log_info(f"DiT Layer Streaming [tiers]: SSD tier uses a cache file ({why}).")
        elif direct["lora_idx"]:
            # LoRA on the GPU: the host buffers / SSD hold the file's bytes (staging in the file layout);
            # after each transfer the LoRA parameters are cast to the compute dtype, patched and written into
            # the slot (compute layout, promoted dtypes), exactly ComfyUI's cast-time path
            host_specs = [(tuple(_raw(prm.data).shape), _raw(prm.data).dtype) for prm in template_params]
            host_dtypes = [dt for _, dt in host_specs]
            host_offsets = direct["rel_offsets"]
            host_bytes = direct["block_bytes"]
            convert_idx = list(direct["lora_idx"])
            n_res_direct = len(resident_list)
            direct["resident_ranges"] = None  # resident buffers are in the compute layout (cache file)
            direct["stream_ranges"] = direct["ranges"][n_res_direct:]
            log_info(f"DiT Layer Streaming [tiers]: SSD tier reads the model file directly with "
                     f"{len(convert_idx)} LoRA parameters per block applied on the GPU after each transfer "
                     f"({direct['path']}, {host_bytes / 1024 ** 2:.1f} MB per block, no cache file).")
        else:
            slot_offsets = host_offsets = direct["rel_offsets"]
            flat_bytes = host_bytes = direct["block_bytes"]
            n_res_direct = len(resident_list)
            direct["resident_ranges"] = direct["ranges"][:n_res_direct]
            direct["stream_ranges"] = direct["ranges"][n_res_direct:]
            log_info(f"DiT Layer Streaming [tiers]: SSD tier reads the model file directly "
                     f"({direct['path']}, {direct['block_bytes'] / 1024 ** 2:.1f} MB per block, no cache file).")
    lay.direct, lay.slot_offsets, lay.flat_bytes, lay.host_specs = direct, slot_offsets, flat_bytes, host_specs
    lay.host_dtypes, lay.host_offsets, lay.host_bytes = host_dtypes, host_offsets, host_bytes
    lay.convert_idx = convert_idx


def _capture_cache(ctx, lay, cap):
    """Whether the host buffers of the previous capture can be reused (same patches and layout, RAM not changed
    much), and the RAM to keep free besides the RAM tier (released text encoder, activation host buffers). Results:
    patches_uuid, cache_key, cache, reused, ram_margin (set on the bundles)."""
    dit, patcher, resident_k, num_blocks = ctx.dit, ctx.patcher, ctx.resident_k, ctx.num_blocks
    num_stream, pin_memory, compute_dtype = ctx.num_stream, ctx.pin_memory, ctx.compute_dtype
    tiering_auto, ram_budget_bytes, chunk_wanted = ctx.tiering_auto, ctx.ram_budget_bytes, ctx.chunk_wanted
    _text_encoder = ctx.text_encoder
    resident_list, direct, slot_specs, host_dtypes = lay.resident_list, lay.direct, lay.slot_specs, lay.host_dtypes
    host_bytes = lay.host_bytes
    # Host buffers (one flat buffer per block). Reused from the capture cache while the patches
    # and the layout are unchanged.
    patches_uuid = getattr(patcher, "patches_uuid", None)
    cache_key = (
        "v3-resident-mode", patches_uuid, tuple(ctx.resident_idx), num_blocks, bool(pin_memory), str(compute_dtype),
        bool(resident_list), bool(tiering_auto), ram_budget_bytes if tiering_auto else None,
        bool(direct), _env.get("WEIGHTFERRY_SSD_LAYOUT") if tiering_auto else None,
        tuple((shape, str(dt), str(hd)) for (shape, dt), hd in zip(slot_specs, host_dtypes)),
    )
    cache = getattr(dit, _CACHE_ATTR, None)
    reused = patches_uuid is not None and cache is not None and cache["key"] == cache_key
    # Activation host buffers of the coming run (chunk streaming may switch activation streaming on mid-run):
    # kept free besides the RAM tier. WEIGHTFERRY_ACT_RAM_RESERVE=0 disables it
    act_reserve = 0
    if (chunk_wanted and not _text_encoder and _env.get("WEIGHTFERRY_ACT_RAM_RESERVE") != "0"
            and _env.get("WEIGHTFERRY_ACT_CHUNK") != "0"
            and _env.get("WEIGHTFERRY_ACT_HOST") != "0"):
        act_reserve = _act_host_reserve(dit)
    if reused and tiering_auto and ram_budget_bytes is None:
        change = _ram_tier_replan(cache.get("plan"), num_stream, host_bytes, extra_margin=act_reserve)
        if change is not None:
            log_info(f"DiT Layer Streaming [tiers]: available RAM changed since the last capture; RAM "
                     f"tier {change[0]} -> about {change[1]} of {num_stream} blocks: capturing again.")
            reused = False
    # RAM left free besides the RAM tier: WEIGHTFERRY_RAM_MARGIN_GB (6), plus, when an idle text encoder's
    # buffers were released for this DiT, room for that encoder's next encode (it captures again with its SSD
    # tier; Qwen2.5-VL 7B + VAE take ~3 GB more after a DiT plan that used all of the released RAM).
    # WEIGHTFERRY_TE_RESERVE_GB (3).
    ram_margin = None
    if not reused and not _text_encoder and tiering_auto and ram_budget_bytes is None:
        released = _release_idle_text_encoder_caches("RAM for the DiT")
        if released:
            te_reserve = int(float(_env.get("WEIGHTFERRY_TE_RESERVE_GB")) * 1024 ** 3)
            ram_margin = int(float(_env.get("WEIGHTFERRY_RAM_MARGIN_GB")) * 1024 ** 3) + te_reserve
            log_info(f"DiT Layer Streaming [tiers]: released the idle text encoder's host buffers "
                     f"(~{released} MB; it captures again on its next encode) for the DiT's RAM tier; "
                     f"{te_reserve / 1024 ** 3:.1f} GB kept free for that encode.")
    if act_reserve and not reused and tiering_auto and ram_budget_bytes is None:
        if ram_margin is None:
            ram_margin = int(float(_env.get("WEIGHTFERRY_RAM_MARGIN_GB")) * 1024 ** 3)
        ram_margin += act_reserve
        log_info(f"DiT Layer Streaming [tiers]: {act_reserve / 1024 ** 2:.0f} MB kept free for the "
                 f"activation host buffers (latent {getattr(dit, '_weightferry_input_shape', None)}).")
    cap.patches_uuid, cap.cache_key, cap.cache, cap.reused = patches_uuid, cache_key, cache, reused
    cap.ram_margin = ram_margin


def _rest_value(view):
    """Rest tensor of a GPU original from its host view. A small GPU original (force-loaded RMSNorm etc.) gets a
    private copy, not a view into the host block: a view keeps the whole block buffer alive, so releasing the capture
    cache would free nothing."""
    return view.clone() if view.numel() * view.element_size() <= 16 * 1024 ** 2 else view


def _rest_tensors(blk, views):
    """Per parameter of ``blk``, the tensor it points to between forwards: CPU originals are kept (unpatched,
    patched at cast time by ComfyUI); a GPU original is replaced by its host view (``views``), or, without host
    views (block on the SSD; small force-loaded module, e.g. RMSNorm), by a private CPU copy."""
    blk_rest = []
    for j, p in enumerate(blk.parameters()):
        src = p.data
        if src.device.type == "cpu":
            blk_rest.append(src)
        elif views is not None:
            blk_rest.append(_rest_value(views[j]))
        else:
            blk_rest.append(src.detach().to("cpu"))
    return blk_rest


def _rest_binding(ctx, lay, cap):
    """The tensors the parameters point to between forwards (rest binding), and the GPU buffers of the self-managed
    resident blocks (filled from the host cache / SSD once per load). Results: rest, res_gpu_flats, res_gpu_views,
    res_rest, n_res_disk (set on the bundles)."""
    streaming_blocks, pin_memory, target_device = ctx.streaming_blocks, ctx.pin_memory, ctx.target_device
    resident_list, flat_bytes, slot_specs = lay.resident_list, lay.flat_bytes, lay.slot_specs
    slot_offsets = lay.slot_offsets
    host_views, res_host_flats, res_host_views = cap.host_views, cap.res_host_flats, cap.res_host_views
    res_disk_store = cap.res_disk_store
    rest = []
    for blk, views in zip(streaming_blocks, host_views):
        rest.append(_rest_tensors(blk, views))

    # Self-managed resident blocks: one GPU buffer per block, filled once per load from the host cache
    res_gpu_flats, res_gpu_views, res_rest = [], [], []
    for blk, h_flat, h_views in zip(resident_list, res_host_flats, res_host_views):
        g_flat = torch.empty(flat_bytes, dtype=torch.uint8, device=target_device)
        if h_flat is not None:
            g_flat.copy_(h_flat, non_blocking=pin_memory)
        res_gpu_flats.append(g_flat)
        res_gpu_views.append(_views(g_flat, slot_specs, slot_offsets))
        res_rest.append(_rest_tensors(blk, h_views))
    n_res_disk = 0 if res_disk_store is None else res_disk_store.n_blocks
    if n_res_disk:
        t_res = time.perf_counter()
        first_disk = len(resident_list) - n_res_disk
        for d, view in res_disk_store.read_each(range(n_res_disk), pin=pin_memory):
            res_gpu_flats[first_disk + d].copy_(view)  # blocking: the read buffer is reused
        log_info(f"DiT Layer Streaming [tiers]: {n_res_disk} self-managed resident blocks read back "
                 f"from the SSD ({(time.perf_counter() - t_res) * 1000:.0f} ms).")
    if resident_list:
        torch.cuda.synchronize(target_device)
    cap.rest, cap.res_gpu_flats, cap.res_gpu_views, cap.res_rest = rest, res_gpu_flats, res_gpu_views, res_rest
    cap.n_res_disk = n_res_disk


def _build_state(ctx, lay, cap):
    """The streaming state of this load: GPU slots (or chunk slots), the SSD ring and uploader, the self-managed
    resident blocks, the scaled fp8 parameter swaps, the compression prototype and the profiler. Returns (state,
    use_chunk, num_slots, staging_flat)."""
    owner, target_device, streaming_blocks = ctx.owner, ctx.target_device, ctx.streaming_blocks
    num_stream, is_double, chunk_wanted = ctx.num_stream, ctx.is_double, ctx.chunk_wanted
    prefetch_slots, compute_dtype = ctx.prefetch_slots, ctx.compute_dtype
    resident_list, quant_idx, lora_fns = lay.resident_list, lay.quant_idx, lay.lora_fns
    convert_idx, host_specs, host_offsets = lay.convert_idx, lay.host_specs, lay.host_offsets
    host_bytes, slot_specs, slot_offsets = lay.host_bytes, lay.slot_specs, lay.slot_offsets
    flat_bytes = lay.flat_bytes
    quant_params, host_flats, host_views, rest = cap.quant_params, cap.host_flats, cap.host_views, cap.rest
    # Scaled fp8 parameters whose LoRA was baked at capture are copied as they are after the transfer
    for s_i, baked in enumerate(quant_params):
        for j in baked:
            lora_fns[s_i][j] = None

    use_chunk = bool(chunk_wanted) and is_double and not convert_idx
    if chunk_wanted and not use_chunk:
        log_info("DiT Layer Streaming [chunks]: chunk streaming needs the double buffer and host buffers "
                 "in the compute layout (no per-transfer dequantization / LoRA); using block streaming.")

    # Allocate ping-pong slots on GPU (compute layout) and, for the storage-dtype path, one staging
    # buffer in the host layout
    num_slots = max(2, int(prefetch_slots)) if is_double else 1
    if is_double and _env.get("WEIGHTFERRY_SLOTS"):
        # Override for experiments / benchmarks
        num_slots = max(2, int(_env.get("WEIGHTFERRY_SLOTS")))
    staging_flat = staging_views = None
    if use_chunk:
        num_slots = max(2, int(_env.get("WEIGHTFERRY_CHUNK_SLOTS")))
        chunk_bytes = int(float(_env.get("WEIGHTFERRY_CHUNK_MB")) * 1024 ** 2)
        q_info = _chunk_quant_info(ctx, lay, cap)
        state = _ChunkState(owner, target_device, streaming_blocks, host_flats, host_views, rest,
                            host_specs, host_offsets, chunk_bytes, num_slots, q_info)
        state.use_learned_order(ctx.dit.__dict__.setdefault("_weightferry_chunk_order", {}))
    else:
        slot_flats = [torch.empty(flat_bytes, dtype=torch.uint8, device=target_device)
                      for _ in range(num_slots)]
        slot_views = [_views(f, slot_specs, slot_offsets) for f in slot_flats]
        if convert_idx:
            staging_flat = torch.empty(host_bytes, dtype=torch.uint8, device=target_device)
            staging_views = _views(staging_flat, host_specs, host_offsets)

        state = _StreamState(owner, target_device, streaming_blocks, host_flats, host_views, rest,
                             slot_flats, slot_views, is_double, convert_idx=convert_idx, lora_fns=lora_fns,
                             compute_dtype=compute_dtype, staging_flat=staging_flat,
                             staging_views=staging_views)
    state.reused_capture = cap.reused
    if not use_chunk:  # the slot layout, for re-allocating released slots (_StreamState.release_slots)
        state.slot_specs, state.slot_offsets = slot_specs, slot_offsets
    state.host_specs, state.host_offsets = host_specs, host_offsets
    state.tier_plan = cap.plan
    state.holders = cap.holders  # keeps exact-size pinned buffers alive even without a capture cache
    state.res_disk_store = cap.res_disk_store
    state.resident_disk_blocks = cap.n_res_disk
    _setup_ssd_tier(ctx, cap, state, use_chunk)
    state.set_resident(resident_list, cap.res_gpu_flats, cap.res_gpu_views, cap.res_rest)
    if quant_idx and resident_list:
        _bind_resident_quant(ctx, lay, cap, state)
    state.quant_bind = state.quant_rest = None
    if quant_idx and not use_chunk:
        _bind_streamed_quant(ctx, lay, cap, state)
    state.quant_idx_set = set(quant_idx)
    if _env.get("WEIGHTFERRY_PROFILE") == "1" and not use_chunk:
        state.profiler = _StreamProfiler(num_stream, host_bytes, is_double, staging_flat is None)
    return state, use_chunk, num_slots, staging_flat


def _chunk_quant_info(ctx, lay, cap):
    """Chunk streaming, scaled fp8: per streaming block {param index: (module, key, layout class, GPU layout params,
    original parameter)}; the layout params are the capture's re-quantized ones when a LoRA was baked."""
    streaming_blocks, target_device = ctx.streaming_blocks, ctx.target_device
    quant_idx, quant_params = lay.quant_idx, cap.quant_params
    q_info = [dict() for _ in streaming_blocks]
    if quant_idx:
        q_names = [n for n, _ in streaming_blocks[0].named_parameters()]
        for s_i, blk in enumerate(streaming_blocks):
            for j in quant_idx:
                module, param_key = _param_owner(blk, q_names[j])
                orig = module._parameters[param_key]
                bound = (quant_params[s_i][j] if j in quant_params[s_i]
                         else _quant_params_to(orig._params, target_device))
                q_info[s_i][j] = (module, param_key, orig._layout_cls, bound, orig)
    return q_info


def _setup_ssd_tier(ctx, cap, state, use_chunk):
    """The state's SSD tier: whether it reads the model file, which streamed blocks it serves, the read-ahead ring
    and the background uploader."""
    num_stream, is_double, pin_memory = ctx.num_stream, ctx.is_double, ctx.pin_memory
    disk_store, plan = cap.disk_store, cap.plan
    state.direct_file = disk_store is not None and disk_store.is_model_file
    if disk_store is not None:
        state.ram_blocks = num_stream - disk_store.n_blocks
        # Older cache entries (no recorded positions) were written in the "tail" layout
        positions = (plan or {}).get("ssd_positions") or list(range(state.ram_blocks, num_stream))
        for d, s_i in enumerate(positions):
            state.disk_index[s_i] = d
        ring_depth = (plan or {}).get("ring_depth") or 4
        state.ring = tiered_store.DiskRing(disk_store, ring_depth, pin=pin_memory,
                                           threads=(plan or {}).get("disk_threads") or 1)
        # Background uploads need the double buffer (per-block slots + events); with a staging buffer the
        # uploader uses its own copy of it
        if is_double and _env.get("WEIGHTFERRY_SSD_ASYNC") != "0" and not use_chunk:
            state.uploader = _SsdUploader(state)


def _bind_resident_quant(ctx, lay, cap, state):
    """Scaled fp8 in self-managed resident blocks: per block the GPU QuantizedTensor parameters around the block's GPU
    buffer (res_quant_bind) and the original parameters to put back (res_quant_rest)."""
    target_device, quant_idx = ctx.target_device, lay.quant_idx
    state.res_quant_bind, state.res_quant_rest = [], []
    for r_i, (blk, views) in enumerate(zip(lay.resident_list, cap.res_gpu_views)):
        binds, rests = [], []
        for j, (name, _p) in enumerate(blk.named_parameters()):
            if j not in quant_idx:
                continue
            module, param_key = _param_owner(blk, name)
            orig = module._parameters[param_key]
            bound = cap.res_quant_params[r_i].get(name) or _quant_params_to(orig._params, target_device)
            binds.append((module, param_key, torch.nn.Parameter(
                QuantizedTensor(views[j], orig._layout_cls, bound), requires_grad=False)))
            rests.append((module, param_key, orig))
        state.res_quant_bind.append(binds)
        state.res_quant_rest.append(rests)


def _bind_streamed_quant(ctx, lay, cap, state):
    """Scaled fp8 in block streaming: the hooks swap the whole parameter object of the module. Measured:
     * assigning p.data swaps only the wrapper; the matmul read the old CPU _qdata ("mat2 is on cpu");
     * swapping _qdata on the parameter leaves the wrapper's device on the CPU, so ComfyUI's
       cast_bias_weight (s.weight.device != device) took its own Dynamic VRAM copy and ignored the slot.
    A GPU QuantizedTensor parameter around the slot's bytes (scales on the GPU) is built once per block.
    With a LoRA, the bound parameters carry the re-quantized scale: from the capture (baked), or a private
    copy that convert_param overwrites after each transfer (direct file source). Sets quant_bind, quant_rest and
    quant_conv on the state."""
    streaming_blocks, target_device = ctx.streaming_blocks, ctx.target_device
    quant_idx, convert_idx, lora_fns = lay.quant_idx, lay.convert_idx, lay.lora_fns
    quant_params = cap.quant_params
    state.quant_bind, state.quant_rest = [], []
    state.quant_conv = [dict() for _ in streaming_blocks]
    q_names = [n for n, _ in streaming_blocks[0].named_parameters()]
    for s_i, blk in enumerate(streaming_blocks):
        views = state.slot_views[state.slot_of(s_i)]
        binds, rests = [], []
        for j in quant_idx:
            module, param_key = _param_owner(blk, q_names[j])
            orig = module._parameters[param_key]
            per_transfer = j in convert_idx and lora_fns[s_i][j] is not None and j not in quant_params[s_i]
            if j in quant_params[s_i]:
                bound = quant_params[s_i][j]
            else:
                bound = _quant_params_to(orig._params, target_device, copy=per_transfer)
            gpu_param = torch.nn.Parameter(QuantizedTensor(views[j], orig._layout_cls, bound),
                                           requires_grad=False)
            if per_transfer:
                state.quant_conv[s_i][j] = (orig._layout_cls, _quant_params_to(orig._params, target_device),
                                            gpu_param._params, _quant_seed(module))
            binds.append((module, param_key, gpu_param))
            rests.append((module, param_key, orig))
        state.quant_bind.append(binds)
        state.quant_rest.append(rests)


_GC_PINNED_BYTES = 512 * 1024 ** 2


def _pinned_ahead(nbytes, count):
    """Queue that a thread fills with ``count`` PinnedHost buffers of ``nbytes``, so registering the pinned memory
    overlaps the capture's GPU work and file reads. An error is put in the queue in place of a buffer."""
    import queue
    import threading
    out = queue.Queue()

    def work():
        try:
            for _ in range(count):
                out.put(tiered_store.PinnedHost(nbytes))
        except BaseException as exc:  # noqa: BLE001 -- handed to the capture loop
            out.put(exc)

    threading.Thread(target=work, daemon=True, name="weightferry-pin-ahead").start()
    return out


@dataclasses.dataclass
class _SsdTier:
    """The SSD side of a new capture (_ssd_stores): the SSD positions of the streamed blocks and their store, the
    persistent cache entry, the self-managed resident blocks' store, and the staging buffers of the cache files
    being written (dropped as soon as their file is finalized)."""
    disk_pos: list
    disk_d: dict  # streaming block index -> position in the SSD store
    n_disk_plan: int
    disk_store: object = None
    disk_tmp: object = None
    persist_cap: float = 0.0
    persist_key: object = None
    persist_hit: object = None  # metadata of a complete persistent entry (its blocks are not captured)
    p_bin: str = None
    p_json: str = None
    res_ram_plan: int = 0
    res_disk_store: object = None
    res_disk_tmp: object = None


def _capture_host_buffers(ctx, lay, cap):
    """The host side of this load: reuse the previous capture's buffers, or plan the RAM / SSD tiers (with the
    persistent SSD cache and the model-file source) and capture every streamed block's final bytes (LoRA baked,
    scaled fp8 re-quantized) into pinned RAM or the SSD cache file, and the self-managed resident blocks likewise.
    Results: host_flats, host_views, quant_params, res_quant_params, res_host_flats, res_host_views, plan,
    disk_store, res_disk_store, holders, converted, cap_t (set on the bundles)."""
    dit, streaming_blocks, resident_list = ctx.dit, ctx.streaming_blocks, lay.resident_list
    reused, cache, cache_key = cap.reused, cap.cache, cap.cache_key
    patches_uuid, t_capture = cap.patches_uuid, cap.t_capture
    converted = 0
    cap_t = {"before_buffers": time.perf_counter() - t_capture, "file_read": 0.0, "quant_lora": 0.0,
             "ssd_write": 0.0, "gc": 0.0, "pin": 0.0}  # capture time split (s)
    # Scaled fp8 + LoRA baked at capture: per streaming block {param index: re-quantized layout params}
    quant_params = [dict() for _ in streaming_blocks]
    if reused:
        host_flats, host_views = cache["host_flats"], cache["host_views"]
        quant_params = cache.get("quant_params") or quant_params
        res_quant_params = cache.get("res_quant_params") or [dict() for _ in resident_list]
        res_host_flats, res_host_views = cache["resident_flats"], cache["resident_views"]
        plan, disk_store = cache.get("plan"), cache.get("disk_store")
        res_disk_store = cache.get("res_disk_store")
        holders = cache.get("holders", [])
    else:
        _drop_previous_capture(dit, cache, cap_t)
        holders = []
        plan = _plan_tiers(ctx, lay, cap.ram_margin)
        tier = _ssd_stores(ctx, lay, plan)
        disk_store, res_disk_store = tier.disk_store, tier.res_disk_store
        cap_t["plan"] = time.perf_counter() - t_capture - cap_t["before_buffers"]
        host_flats, host_views, n_conv = _capture_streamed_blocks(ctx, lay, tier, holders, quant_params, cap_t)
        converted += n_conv
        _finish_ssd_cache(lay, tier, quant_params)
        res_host_flats, res_host_views, res_quant_params, n_conv = _capture_resident_blocks(ctx, lay, tier, holders)
        converted += n_conv
        if patches_uuid is not None:
            setattr(dit, _CACHE_ATTR, {"key": cache_key, "host_flats": host_flats, "host_views": host_views,
                                       "quant_params": quant_params, "res_quant_params": res_quant_params,
                                       "resident_flats": res_host_flats, "resident_views": res_host_views,
                                       "plan": plan, "disk_store": disk_store, "holders": holders,
                                       "res_disk_store": res_disk_store})
    cap.host_flats, cap.host_views, cap.quant_params = host_flats, host_views, quant_params
    cap.res_quant_params, cap.res_host_flats, cap.res_host_views = res_quant_params, res_host_flats, res_host_views
    cap.plan, cap.disk_store, cap.res_disk_store, cap.holders = plan, disk_store, res_disk_store, holders
    cap.converted, cap.cap_t = converted, cap_t


def _drop_previous_capture(dit, cache, cap_t):
    """Before a new capture: drop the previous entry first so two captures never coexist in pinned memory / on the
    SSD. Pinned buffers are released explicitly (exact-size registered memory goes back to the OS at once; PyTorch's
    pinned allocator would keep freed blocks cached in the process)."""
    had_cache = cache is not None
    if cache is not None:
        for key in ("disk_store", "res_disk_store"):
            if cache.get(key) is not None:
                cache[key].delete()
        tiered_store.release_all(cache.get("holders"))
    setattr(dit, _CACHE_ATTR, None)
    # A full garbage collection (~0.2 s in ComfyUI) frees what reference cycles still hold of a previous capture.
    # The first capture of a model skips it unless much pinned memory is still registered (e.g. another model's
    # capture that is now garbage)
    if had_cache or tiered_store._REGISTERED["bytes"] > _GC_PINNED_BYTES:
        import gc
        t_gc = time.perf_counter()
        gc.collect()
        cap_t["gc"] = time.perf_counter() - t_gc
    tiered_store.empty_host_cache()


def _plan_tiers(ctx, lay, ram_margin):
    """Tier plan (TIERING_AUTO): how many streaming blocks stay in pinned RAM, the rest on the SSD. None with tiering
    RAM, or when the weights' originals live on the GPU (legacy patcher)."""
    dit, patcher, streaming_blocks = ctx.dit, ctx.patcher, ctx.streaming_blocks
    num_blocks, num_stream, ram_budget_bytes = ctx.num_blocks, ctx.num_stream, ctx.ram_budget_bytes
    resident_list, flat_bytes, host_bytes = lay.resident_list, lay.flat_bytes, lay.host_bytes
    if not ctx.tiering_auto:
        return None
    plan = None
    gpu_origin = sum(_param_nbytes(p) for blk in streaming_blocks
                     for p in blk.parameters() if _raw(p.data).device.type == "cuda")
    gguf_keep = (gpu_origin > 0 and _env.get("WEIGHTFERRY_GGUF_KEEP_MMAP") == "1"
                 and getattr(patcher, "mmap_released", None) is not None)
    if gguf_keep:
        # ComfyUI-GGUF's (classic) patcher put part of the streamed layers on the GPU before this capture
        # (more of them when more VRAM was free, e.g. on a later sampling). The SSD-tier blocks among them
        # get a CPU copy at the rest binding: at most these bytes, so they are kept free besides the RAM tier
        # instead of disabling the SSD tier (all blocks in RAM can exhaust the RAM, e.g. Cosmos-Predict2 14B
        # Q8_0 + LoRA on a 32 GB PC)
        if ram_margin is None and ram_budget_bytes is None:
            ram_margin = int(float(_env.get("WEIGHTFERRY_RAM_MARGIN_GB")) * 1024 ** 3)
        if ram_margin is not None:
            ram_margin += gpu_origin
        log_info(f"DiT Layer Streaming [tiers]: {gpu_origin / 1024 ** 2:.0f} MB of the streamed "
                 f"layers' originals are on the GPU (ComfyUI-GGUF); kept free for their CPU copies.")
    if gpu_origin > 0.05 * num_stream * host_bytes and not gguf_keep:
        log_warn(
            "DiT Layer Streaming [tiers]: the weights' originals live on the GPU (legacy "
            "patcher?); a CPU copy per SSD-tier block would be needed between forwards, so the SSD "
            "tier is disabled and all streaming blocks stay in RAM."
        )
    else:
        plan = tiered_store.plan_tiers(
            num_stream, host_bytes, ram_budget_bytes=ram_budget_bytes, ram_margin_bytes=ram_margin,
            compute_ms_per_block=getattr(dit, "_weightferry_stream_block_ms", None),
            total_blocks=num_blocks, resident_blocks=len(resident_list),
            resident_block_bytes=flat_bytes)
        samples = getattr(dit, "_weightferry_stream_block_samples", None)
        plan["compute_samples"] = len(samples["ms"]) if samples else 0
        log_info(f"DiT Layer Streaming [tiers]: plan {plan}")
        if plan.get("disk_bound"):
            log_warn("DiT Layer Streaming [tiers]: SSD reads are predicted to take longer than "
                     "a forward pass of compute -- expect the SSD to be the bottleneck.")
        if plan.get("ring_depth", 0) < plan.get("ring_depth_wanted", 0):
            log_warn(f"DiT Layer Streaming [tiers]: the RAM budget only holds a read-ahead ring "
                     f"of {plan['ring_depth']} blocks (wanted {plan['ring_depth_wanted']}); SSD reads "
                     f"may stall compute.")
        if plan.get("over_budget"):
            log_warn(f"DiT Layer Streaming [tiers]: the minimum pinned RAM "
                     f"({plan['host_ram_planned_mb']:.0f} MB) exceeds the RAM budget "
                     f"({plan['ram_budget_mb']:.0f} MB)"
                     + (f"; the SSD tier is unavailable ({plan['disk_error']}), so every block is kept in RAM."
                        if plan.get("disk_error") else "."))
    return plan


def _ssd_stores(ctx, lay, plan):
    """The SSD stores of a new capture: for the streamed SSD-tier blocks the model file, a persistent cache entry
    (found or new) or a cache file of this process; for self-managed resident blocks beyond the RAM plan the model
    file or a cache file. Cache files being written get a staging buffer. Returns an _SsdTier."""
    num_stream, resident_list, direct = ctx.num_stream, lay.resident_list, lay.direct
    flat_bytes, host_bytes = lay.flat_bytes, lay.host_bytes
    ram_blocks_plan = num_stream if plan is None else plan["ram_blocks"]
    n_disk_plan = num_stream - ram_blocks_plan
    disk_pos = tiered_store.ssd_positions(num_stream, n_disk_plan)
    tier = _SsdTier(disk_pos=disk_pos, disk_d={s_i: d for d, s_i in enumerate(disk_pos)}, n_disk_plan=n_disk_plan)
    if plan is not None:
        plan["ssd_positions"] = disk_pos
    if n_disk_plan > 0 and direct is not None:
        tier.disk_store = tiered_store.file_source(direct["path"], [direct["stream_ranges"][i] for i in disk_pos])
    elif n_disk_plan > 0:
        _persist_lookup(ctx, lay, plan, tier)
        if tier.persist_hit is not None:
            tier.disk_store = tiered_store.PersistentBlockStore(tier.p_bin, host_bytes, n_disk_plan, existing=True)
            log_info(f"DiT Layer Streaming [persist]: reusing the SSD cache of an earlier run "
                     f"({tier.p_bin}, {n_disk_plan} blocks; no capture / write for them).")
        elif tier.persist_key is not None:
            tier.disk_store = tiered_store.PersistentBlockStore(tier.p_bin, host_bytes, n_disk_plan)
        else:
            tier.disk_store = tiered_store.DiskBlockStore(
                tiered_store.new_cache_path(plan["disk_dir"], "blocks"), host_bytes, n_disk_plan)
        rep_ = tier.disk_store.cleanup_report
        if rep_ and rep_["freed_bytes"]:
            log_info(f"DiT Layer Streaming [tiers]: removed {rep_['freed_bytes'] / 1024 ** 3:.1f} GB of "
                     f"cache files left by ended processes in {plan['disk_dir']}.")
        tier.disk_tmp = tier.disk_store.staging() if tier.persist_hit is None else None

    res_ram_plan = len(resident_list)
    if plan is not None:
        res_ram_plan = plan.get("resident_ram_blocks", res_ram_plan)
    tier.res_ram_plan = res_ram_plan
    if (len(resident_list) > res_ram_plan and direct is not None
            and direct.get("resident_ranges") is not None):
        tier.res_disk_store = tiered_store.file_source(direct["path"], direct["resident_ranges"][res_ram_plan:])
    elif len(resident_list) > res_ram_plan:
        tier.res_disk_store = tiered_store.DiskBlockStore(
            tiered_store.new_cache_path(plan["disk_dir"], "resident"), flat_bytes,
            len(resident_list) - res_ram_plan)
        tier.res_disk_tmp = tier.res_disk_store.staging()
    return tier


def _persist_lookup(ctx, lay, plan, tier):
    """Persistent SSD cache (opt-in, WEIGHTFERRY_PERSIST_GB = size cap): reused across ComfyUI restarts while the
    model file, the LoRA patches and the layout are the same. Sets persist_cap, persist_key, p_bin, p_json and
    persist_hit (the metadata of a complete earlier entry) on ``tier``."""
    tier.persist_cap = float(_env.get("WEIGHTFERRY_PERSIST_GB") or 0) * 1024 ** 3
    if tier.persist_cap > 0:
        try:
            tier.persist_key = _persist_key(ctx.patcher, ctx.dit, ctx.streaming_blocks,
                                            list(ctx.stream_idx), lay.host_specs,
                                            lay.host_offsets, lay.host_bytes, tier.disk_pos, ctx.compute_dtype)
        except Exception as exc:  # noqa: BLE001 -- no persistence for this load
            tier.persist_key = None
            log_warn(f"DiT Layer Streaming [persist]: key failed ({exc}); not persisting.")
    if tier.persist_key is not None:
        p_dir = tiered_store.persist_dir(plan["disk_dir"])
        os.makedirs(p_dir, exist_ok=True)
        tier.p_bin = os.path.join(p_dir, f"dit_persist_{tier.persist_key}.bin")
        tier.p_json = tier.p_bin[:-4] + ".json"
        if os.path.isfile(tier.p_bin) and os.path.isfile(tier.p_json):
            try:
                with open(tier.p_json, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):  # damaged entry: capture and write it again
                meta = None
            if meta is not None and _persist_meta_ok(meta, tier.persist_key, tier.n_disk_plan, lay.host_bytes,
                                                     tier.p_bin):
                tier.persist_hit = meta
                try:
                    os.utime(tier.p_json)  # LRU; a failure here does not make the entry unusable
                except OSError:
                    pass
            elif meta is not None:
                log_warn(f"DiT Layer Streaming [persist]: the SSD cache entry {tier.p_bin} does not match its "
                         f"metadata (incomplete or stale); writing it again.")
        if tier.persist_hit is None:
            # the entry is about to be written again (the bin opened for writing): its old metadata must not make a
            # half-written file look complete if this run stops before the new metadata (Codex CPU check 2026-10-13)
            try:
                os.remove(tier.p_json)
            except OSError:
                pass


def _persist_meta_ok(meta, key, blocks, block_bytes, p_bin):
    """A persistent entry's metadata belongs to this key and layout and its data file has the length the blocks need
    (the last block's padding may be absent). Not a content check: same-length damage is not detected."""
    if not (isinstance(meta, dict) and meta.get("key") == key and meta.get("blocks") == blocks
            and meta.get("block_bytes") == block_bytes and isinstance(meta.get("quant_scales", {}), dict)):
        return False
    try:
        size = os.path.getsize(p_bin)
    except OSError:
        return False
    stride = tiered_store.align_up(block_bytes)
    return blocks > 0 and (blocks - 1) * stride + block_bytes <= size <= blocks * stride


def _capture_streamed_blocks(ctx, lay, tier, holders, quant_params, cap_t):
    """Capture of the streamed blocks' final bytes in the host layout (LoRA baked, scaled fp8 re-quantized): into
    pinned RAM (exact-size buffers, appended to ``holders``) or, for SSD-tier blocks, through the staging buffer into
    the cache file. Blocks read from the model file at run time or found in the persistent cache are not captured
    (a persistent hit's re-quantized scales go into quant_params). Returns (host_flats, host_views, converted
    tensors)."""
    t_loop = time.perf_counter()
    dit, streaming_blocks, resident_k, num_blocks = ctx.dit, ctx.streaming_blocks, ctx.resident_k, ctx.num_blocks
    target_device, pin_memory, compute_dtype = ctx.target_device, ctx.pin_memory, ctx.compute_dtype
    names, quant_idx, convert_idx, direct = lay.names, lay.quant_idx, lay.convert_idx, lay.direct
    host_specs, host_dtypes, host_offsets, host_bytes = lay.host_specs, lay.host_dtypes, lay.host_offsets, lay.host_bytes
    disk_d, disk_store, persist_hit = tier.disk_d, tier.disk_store, tier.persist_hit
    converted = 0
    host_flats, host_views = [], []
    # Scaled fp8 + baked LoRA reads every streamed weight once: take the bytes from the file
    # (unbuffered) rather than the mmap, so the capture does not fill the working set
    file_blocks = None

    def _quant_lora_on(blk, j):
        module, param_key = _param_owner(blk, names[j])
        return getattr(module, param_key + "_lowvram_function", None) is not None
    # (blocks read from the model file at run time are not captured)
    captured_idx = [s_i for s_i in range(len(streaming_blocks))
                    if not (s_i in disk_d and (direct is not None or persist_hit is not None))]
    if quant_idx and any(_quant_lora_on(streaming_blocks[s_i], j) for s_i in captured_idx for j in quant_idx):
        try:
            f_src, f_rel = _block_file_source(dit, streaming_blocks, list(ctx.stream_idx))
        except Exception as exc:  # noqa: BLE001 -- fall back to the mmap
            f_src, f_rel = None, None
            log_warn(f"DiT Layer Streaming: reading the capture from the file failed ({exc}); "
                     f"using the mmap.")
        if f_src is not None:
            file_blocks = f_src.read_each(captured_idx, ahead=True)
    pinned_ahead = None
    pending_d2h = False
    if pin_memory:
        pinned_ahead = _pinned_ahead(host_bytes, sum(1 for s_i in captured_idx if s_i not in disk_d))
    for s_i, blk in enumerate(streaming_blocks):
        on_disk = s_i in disk_d
        if on_disk and direct is not None:
            host_flats.append(None)  # read from the model file
            host_views.append(None)
            continue
        if on_disk and persist_hit is not None:
            host_flats.append(None)  # bytes already in the persistent cache file
            host_views.append(None)
            for j_str, scale in (persist_hit.get("quant_scales", {}).get(str(s_i)) or {}).items():
                j = int(j_str)
                orig = list(blk.parameters())[j].data._params
                quant_params[s_i][j] = dataclasses.replace(
                    orig, scale=torch.tensor(scale["v"], dtype=torch.float32,
                                             device=target_device).reshape(scale["shape"]))
            continue
        t_rd = time.perf_counter()
        blk_bytes = next(file_blocks)[1] if file_blocks is not None else None
        cap_t["file_read"] += time.perf_counter() - t_rd
        if on_disk:
            flat = tier.disk_tmp
        elif pin_memory:
            t_pin = time.perf_counter()
            holder = pinned_ahead.get()
            if isinstance(holder, BaseException):
                raise holder
            cap_t["pin"] += time.perf_counter() - t_pin
            holders.append(holder)
            flat = holder.tensor
        else:
            flat = torch.empty(host_bytes, dtype=torch.uint8, device="cpu")
        views = _views(flat, host_specs, host_offsets)
        for j, ((name, p), v) in enumerate(zip(blk.named_parameters(), views)):
            module, param_key = _param_owner(blk, name)
            src = p.data  # note: every p.data access returns a new tensor object
            if _is_quant(src):
                # Stored fp8 bytes (the scale stays with the bound wrapper). A LoRA is baked here; with
                # WEIGHTFERRY_QUANT_LORA=transfer only blocks read from the model file at run time
                # (not captured) re-quantize after each transfer
                q_data = src._qdata
                if blk_bytes is not None and q_data.device.type == "cpu":
                    nb = q_data.numel() * q_data.element_size()
                    q_data = blk_bytes[f_rel[j]:f_rel[j] + nb].view(q_data.dtype).view(q_data.shape)
                q_fn = _lora_apply_fn(module, param_key, target_device)
                if q_fn is None:
                    v.copy_(q_data)
                    continue
                q_src = QuantizedTensor(q_data.to(device=target_device), src._layout_cls,
                                        _quant_params_to(src._params, target_device))
                t_q = time.perf_counter()
                y = _quant_lora_requant(q_src, q_fn, compute_dtype, _quant_seed(module))
                # Into pinned RAM the copy runs without waiting (one wait after the loop): the GPU goes on
                # with the next weight meanwhile
                async_d2h = pin_memory and not on_disk and y._qdata.is_cuda
                v.copy_(y._qdata, non_blocking=async_d2h)
                pending_d2h = pending_d2h or async_d2h
                cap_t["quant_lora"] += time.perf_counter() - t_q
                quant_params[s_i][j] = y._params
                converted += 1
                continue
            if j in convert_idx:
                # Storage dtype, converted on the GPU like ComfyUI's VBAR staging copy;
                # dequantization and LoRA happen after the DMA (StreamState.materialize)
                v.copy_(src.to(device=target_device).to(dtype=host_dtypes[j]))
                converted += 1
                continue
            value = _effective_weight(module, param_key, src, target_device, compute_dtype,
                                      out_dtype=v.dtype)
            if value is not src:
                converted += 1
            v.copy_(value)
        if on_disk:
            t_wr = time.perf_counter()
            disk_store.write_block(disk_d[s_i], flat)
            cap_t["ssd_write"] += time.perf_counter() - t_wr
            host_flats.append(None)
            host_views.append(None)
        else:
            host_flats.append(flat)
            host_views.append(views)
    if pending_d2h:
        torch.cuda.current_stream(target_device).synchronize()
    cap_t["block_loop"] = time.perf_counter() - t_loop
    if file_blocks is not None:
        file_blocks.close()  # releases the read buffer and the file handle
    return host_flats, host_views, converted


def _finish_ssd_cache(lay, tier, quant_params):
    """After the streamed blocks' capture: the SSD cache file is completed and its staging buffer dropped; a new
    persistent entry gets its metadata and the persistent directory's size cap is applied."""
    disk_store = tier.disk_store
    if disk_store is not None and not disk_store.is_model_file and tier.persist_hit is None:
        disk_store.finalize()
        tier.disk_tmp = None  # the staging buffer
        if tier.persist_key is not None:
            _write_persist_metadata(tier, quant_params, lay.host_bytes)


def _write_persist_metadata(tier, quant_params, host_bytes):
    """Completed persistent entry: write the metadata last (an entry without it is never reused), with the
    re-quantized scales of its blocks; then remove older entries beyond the size cap."""
    q_scales = {}
    for s_i in tier.disk_d:
        for j, prm in quant_params[s_i].items():
            sc = prm.scale.detach().float().cpu()
            q_scales.setdefault(str(s_i), {})[str(j)] = {"v": sc.reshape(-1).tolist(),
                                                          "shape": list(sc.shape)}
    with open(tier.p_json, "w", encoding="utf-8") as f:
        json.dump({"key": tier.persist_key, "blocks": tier.n_disk_plan, "block_bytes": host_bytes,
                   "created": time.time(), "quant_scales": q_scales}, f)
    freed = tiered_store.persist_enforce_cap(os.path.dirname(tier.p_bin), tier.persist_cap,
                                             keep_key=tier.persist_key)
    log_info(f"DiT Layer Streaming [persist]: SSD cache kept for later runs ({tier.p_bin})"
             + (f"; removed {freed / 1024 ** 3:.1f} GB of older entries." if freed else "."))


def _capture_resident_blocks(ctx, lay, tier, holders):
    """Self-managed resident blocks: final compute values (LoRA baked), compute layout. Blocks >= res_ram_plan go
    to the SSD (read back once per load); its cache file is completed and its staging buffer dropped at the end.
    Returns (res_host_flats, res_host_views, res_quant_params, converted tensors)."""
    target_device, pin_memory, compute_dtype = ctx.target_device, ctx.pin_memory, ctx.compute_dtype
    resident_list, direct, slot_specs, slot_offsets = lay.resident_list, lay.direct, lay.slot_specs, lay.slot_offsets
    flat_bytes = lay.flat_bytes
    res_ram_plan, res_disk_store = tier.res_ram_plan, tier.res_disk_store
    converted = 0
    res_host_flats, res_host_views = [], []
    res_quant_params = [dict() for _ in resident_list]
    for r_i, blk in enumerate(resident_list):
        res_on_disk = r_i >= res_ram_plan
        if res_on_disk and direct is not None and direct.get("resident_ranges") is not None:
            res_host_flats.append(None)  # read from the model file
            res_host_views.append(None)
            continue
        if res_on_disk:
            flat = tier.res_disk_tmp
        elif pin_memory:
            holder = tiered_store.PinnedHost(flat_bytes)
            holders.append(holder)
            flat = holder.tensor
        else:
            flat = torch.empty(flat_bytes, dtype=torch.uint8, device="cpu")
        views = _views(flat, slot_specs, slot_offsets)
        for (name, p), v in zip(blk.named_parameters(), views):
            module, param_key = _param_owner(blk, name)
            src = p.data
            if _is_quant(src):
                q_fn = _lora_apply_fn(module, param_key, target_device)
                if q_fn is None:
                    v.copy_(src._qdata)
                    continue
                q_src = QuantizedTensor(src._qdata.to(device=target_device), src._layout_cls,
                                        _quant_params_to(src._params, target_device))
                y = _quant_lora_requant(q_src, q_fn, compute_dtype, _quant_seed(module))
                v.copy_(y._qdata)
                res_quant_params[r_i][name] = y._params
                converted += 1
                continue
            value = _effective_weight(module, param_key, src, target_device, compute_dtype,
                                      out_dtype=v.dtype)
            if value is not src:
                converted += 1
            v.copy_(value)
        if res_on_disk:
            res_disk_store.write_block(r_i - res_ram_plan, flat)
            res_host_flats.append(None)
            res_host_views.append(None)
        else:
            res_host_flats.append(flat)
            res_host_views.append(views)
    if res_disk_store is not None and not res_disk_store.is_model_file:
        res_disk_store.finalize()
        tier.res_disk_tmp = None  # the staging buffer
    return res_host_flats, res_host_views, res_quant_params, converted
