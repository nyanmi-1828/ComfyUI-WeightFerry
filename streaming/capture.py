# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Capture-time helpers: effective weights, persistent cache keys, direct reads from the model file, scaled fp8
+ LoRA modes, RAM tier re-planning."""

import os

import torch

from ..core import tiered_store
from .common import _block_list_attr, _blocks, _is_quant, _key_prefix, _param_owner, _quant_seed, _raw
from ..core import env as _env


def _model_file_path(blocks, cpu_only):
    """Path of the model file the parameters of ``blocks`` were loaded from (the first parameter whose storage
    records one), or None. cpu_only: only CPU parameters are looked at (GPU ones are copies, not mmap views)."""
    for blk in blocks:
        for prm in blk.parameters():
            raw = _raw(prm.data)
            if cpu_only and raw.device.type != "cpu":
                continue
            path = getattr(raw.untyped_storage(), "_comfy_source_path", None)
            if path:
                return path
    return None


def _effective_weight(module, param_key, tensor, device, compute_dtype, out_dtype=None):
    """Value the block must compute with, reproducing ComfyUI's Dynamic VRAM cast path
    (ops.cast_modules_with_vbar + resolve_cast_module_with_vbar, first run after load):
      1. the weight is staged in its storage dtype ``*_comfy_model_dtype`` (e.g. fp8 for fp8_e4m3fn_fast),
         converted with a plain (round-to-nearest) copy;
      2. it is dequantized to the compute dtype;
      3. a ``*_lowvram_function`` LoRA patch, if any, is applied there and the result is used directly.
    The result is returned in the parameter's dtype. fp8 values are exactly representable in bf16,
    so storing them in the (bf16) host buffer is lossless.

    Only VBAR-managed modules (``_v`` attribute) take that path. Small modules (<= 16KB, e.g. RMSNorm)
    are force-loaded by ModelPatcherDynamic.load: their parameter already holds the final value in its
    original dtype (patches baked by patch_weight_to_device), as do all modules under the legacy patcher."""
    if not hasattr(module, "_v"):
        return tensor
    model_dtype = getattr(module, param_key + "_comfy_model_dtype", None) or tensor.dtype
    lowvram_fn = getattr(module, param_key + "_lowvram_function", None)
    if model_dtype == tensor.dtype and lowvram_fn is None:
        return tensor
    value = tensor.to(device=device, dtype=model_dtype)
    if lowvram_fn is not None:
        value = lowvram_fn(value.to(dtype=compute_dtype))
    # out_dtype: the buffer's dtype. A LoRA result must not be rounded back to an fp8 parameter dtype -- ComfyUI
    # computes with it in the compute dtype (see "LoRA on fp8 parameters" in load_steps._slot_layout)
    return value.to(dtype=out_dtype or tensor.dtype)


def _persist_put(h, tag, data):
    """One framed item: a type tag and the length before the bytes, so different structures never feed the same
    byte string (key v2; v1 concatenated, e.g. [1, 23] and [12, 3] both gave "(123)" -- Codex CPU check 2026-10-13)."""
    h.update(tag + len(data).to_bytes(8, "little") + data)


def _persist_hash_tree(h, obj):
    """Feed a LoRA patch structure (tensors, adapters, tuples, numbers) into hash object h. Another type raises
    TypeError: the entry is then not persisted (the caller logs it) rather than keyed by a guess."""
    from comfy import weight_adapter
    if isinstance(obj, torch.Tensor):
        t = obj.detach().contiguous().cpu()
        _persist_put(h, b"T", f"{tuple(t.shape)}{t.dtype}".encode())
        _persist_put(h, b"D", t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, weight_adapter.WeightAdapterBase):
        _persist_put(h, b"A", type(obj).__name__.encode())
        _persist_hash_tree(h, obj.weights)
    elif isinstance(obj, (tuple, list)):
        _persist_put(h, b"S", str(len(obj)).encode())
        for o in obj:
            _persist_hash_tree(h, o)
    elif isinstance(obj, dict):
        _persist_put(h, b"M", str(len(obj)).encode())
        for k in sorted(obj, key=str):
            _persist_put(h, b"K", repr(k).encode())
            _persist_hash_tree(h, obj[k])
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        _persist_put(h, b"V", f"{type(obj).__name__}:{obj!r}".encode())
    else:
        raise TypeError(f"no stable cache key for a LoRA patch part of type {type(obj).__qualname__}")


def _persist_key(patcher, dit, streaming_blocks, block_indices, host_specs, host_offsets, host_bytes, disk_pos,
                 compute_dtype):
    """Content key of an SSD cache for later runs: model file identity, the LoRA patches of the
    streamed blocks (values, strengths), the host layout, the SSD block positions, compute dtype and versions.
    None if the model is not a single safetensors file."""
    import hashlib
    # Parameters on any device are looked at here (the file reads in _block_file_source / _direct_file_layout look at
    # CPU parameters only)
    path = _model_file_path([streaming_blocks[0]], cpu_only=False)
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    h = hashlib.blake2b(digest_size=16)
    try:
        import comfyui_version
        comfy_ver = comfyui_version.__version__
    except (ImportError, AttributeError):  # older ComfyUI
        comfy_ver = "?"
    h.update(repr(("v2", os.path.abspath(path), st.st_size, st.st_mtime_ns, comfy_ver, torch.__version__,
                   str(compute_dtype), list(block_indices), [(tuple(sh), str(dt)) for sh, dt in host_specs],
                   list(host_offsets), host_bytes, list(disk_pos))).encode())
    attr = _block_list_attr(dit) or "blocks"
    names = [n for n, _ in streaming_blocks[0].named_parameters()]
    patches = getattr(patcher, "patches", {}) or {}
    for i in block_indices:
        for n in names:
            key = f"{_key_prefix(dit)}{attr}.{i}.{n}"
            if key in patches:
                _persist_put(h, b"P", key.encode())
                _persist_hash_tree(h, patches[key])
    return h.hexdigest()


def _block_file_source(dit, blocks, block_indices):
    """(FileBlockSource, rel_offsets) when every CPU parameter of ``blocks`` (the block list's entries at
    block_indices) is an mmap view of one safetensors file at its header offset and each block is one contiguous
    range of the file, else (None, None). Lets a capture read the bytes with unbuffered I/O instead of touching the
    mmap: every touched mmap page stays in the process working set (baking a LoRA into 27 Cosmos-Predict2 14B
    blocks through the mmap grows the working set by ~10 GB)."""
    names = [n for n, _ in blocks[0].named_parameters()]
    attr = _block_list_attr(dit) or "blocks"
    path = _model_file_path([blocks[0]], cpu_only=True)
    if not path or not str(path).lower().endswith(".safetensors"):
        return None, None
    layout = tiered_store.file_block_layout(path, [[f"{attr}.{i}.{n}" for n in names] for i in block_indices])
    if layout is None:
        return None, None
    for blk, (start, _n) in zip(blocks, layout["ranges"]):
        for j, p in enumerate(blk.parameters()):
            raw = _raw(p.data)
            if raw.device.type != "cpu":
                continue
            info = getattr(raw.untyped_storage(), "_comfy_tensor_file_slice", None)
            if (info is None or info.offset != start + layout["rel_offsets"][j] or raw.storage_offset() != 0
                    or raw.dtype != layout["dtypes"][j] or tuple(raw.shape) != layout["shapes"][j]):
                return None, None
    return tiered_store.FileBlockSource(path, layout["ranges"]), layout["rel_offsets"]


def _quant_lora_auto_mode(num_stream, ram_blocks, block_bytes):
    """bake or transfer for LoRA on scaled fp8 weights (WEIGHTFERRY_QUANT_LORA=auto), by estimated total time:
      bake     = SSD-tier bytes / SSD write speed (once per LoRA set; the capture cache keeps it while the patches
                 are unchanged)
      transfer = SSD-tier blocks x re-quantization time per block x expected forwards (every forward)
    Defaults from Cosmos-Predict2 14B measurements: writes ~0.3 GB/s effective (consumer NVMe including its
    slowdown after the write cache fills), re-quantization ~0.23 ms per MB of block (2.3 s per
    forward for 27 x 375 MB), 60 forwards (30 steps with CFG). Overrides: WEIGHTFERRY_SSD_WRITE_GBPS,
    WEIGHTFERRY_QUANT_LORA_MS_PER_MB, WEIGHTFERRY_QUANT_LORA_FORWARDS. Returns (mode, info dict)."""
    write_gbps = float(_env.get("WEIGHTFERRY_SSD_WRITE_GBPS"))
    ms_per_mb = float(_env.get("WEIGHTFERRY_QUANT_LORA_MS_PER_MB"))
    forwards = int(_env.get("WEIGHTFERRY_QUANT_LORA_FORWARDS"))
    n_ssd = max(0, num_stream - ram_blocks)
    bake_s = n_ssd * block_bytes / (write_gbps * 1024 ** 3)
    transfer_s = n_ssd * (block_bytes / 1024 ** 2) * ms_per_mb / 1000 * forwards
    mode = "bake" if bake_s <= transfer_s else "transfer"
    return mode, {"ssd_blocks": n_ssd, "bake_write_s": round(bake_s, 1), "transfer_total_s": round(transfer_s, 1),
                  "forwards": forwards, "mode": mode}


def _direct_file_layout(patcher, dit, blocks, block_indices, target_device, compute_dtype, allow_lora=False,
                        quant_lora_mode="bake"):
    """File byte layout for ``blocks`` (the block list's entries at block_indices) if they can be streamed straight from
    the model's safetensors file, else (None, reason). Conditions:
      * every parameter's value is the file's bytes: no patch registered on it, _effective_weight returns the
        parameter itself (no dtype conversion, no LoRA function), CPU parameters are mmap views of the file at the
        expected offset, GPU parameters (small force-loaded modules) equal the file's bytes;
      * each block is one contiguous range of the file with the same relative layout (file_block_layout).
    allow_lora: a LoRA that ComfyUI applies at cast time (``*_lowvram_function`` on a VBAR-managed module whose
    parameter is the file's mmap view, no dtype conversion) is allowed; the indices of such parameters are returned
    in "lora_idx" so the caller can apply the LoRA on the GPU after each transfer. A patch baked into a force-loaded
    GPU parameter is still rejected (its bytes are no longer the file's)."""
    names = [n for n, _ in blocks[0].named_parameters()]
    attr = _block_list_attr(dit) or "blocks"
    path = _model_file_path(blocks, cpu_only=True)
    if not path or not str(path).lower().endswith(".safetensors"):
        return None, "parameters are not mmap views of a safetensors file"
    patched = set(getattr(patcher, "patches", {}) or {})
    block_names = [[f"{attr}.{i}.{n}" for n in names] for i in block_indices]
    lora_idx = set()
    layout = tiered_store.file_block_layout(path, block_names)
    if layout is None:
        # Several ranges per block (e.g. a file grouped by dtype): read piece by piece
        layout = tiered_store.file_param_layout(path, block_names)
        if layout is None:
            return None, "blocks are not uniform in the file"
    else:
        layout["param_abs"] = [[start + r for r in layout["rel_offsets"]] for start, _n in layout["ranges"]]
        layout["block_bytes"] = max(n for _, n in layout["ranges"])
    header_dtypes, header_shapes = layout["dtypes"], layout["shapes"]
    with open(path, "rb") as f:
        for b_i, (i, blk) in enumerate(zip(block_indices, blocks)):
            for j, (name, p) in enumerate(blk.named_parameters()):
                src = p.data
                raw_src = _raw(src)
                if raw_src.dtype != header_dtypes[j] or tuple(raw_src.shape) != header_shapes[j]:
                    return None, f"dtype/shape differs from the file ({name}: {raw_src.dtype} vs {header_dtypes[j]})"
                module, param_key = _param_owner(blk, name)
                lora_fn = getattr(module, param_key + "_lowvram_function", None)
                cast_time_lora = (allow_lora and lora_fn is not None and hasattr(module, "_v")
                                  and raw_src.device.type == "cpu")
                if f"{_key_prefix(dit)}{attr}.{i}.{name}" in patched and not cast_time_lora:
                    return None, f"patch on {attr}.{i}.{name}"
                if cast_time_lora:
                    if _is_quant(src):
                        if _quant_seed(module) is None:
                            return None, f"no seed key for re-quantizing the LoRA result ({name})"
                        # Re-quantizing after every transfer is memory-bound GPU work (dequantize, LoRA, amax,
                        # stochastic rounding): on Cosmos-Predict2 14B, 2.3 s per forward and +2.6 GB VRAM.
                        # ComfyUI itself computes it once and caches it, so the
                        # default bakes it into the host buffers / a cache file at capture.
                        # WEIGHTFERRY_QUANT_LORA=transfer keeps the model file as the source instead.
                        if quant_lora_mode != "transfer":
                            return None, ("LoRA on scaled fp8 weights is baked (re-quantized once at capture); "
                                          "WEIGHTFERRY_QUANT_LORA=transfer applies it after each transfer instead")
                    else:
                        model_dtype = getattr(module, param_key + "_comfy_model_dtype", None) or src.dtype
                        if model_dtype != src.dtype:
                            return None, f"dtype conversion at load ({name}: {src.dtype} -> {model_dtype})"
                    lora_idx.add(j)
                elif (lora_fn is None and not _is_quant(src)
                      and _exact_widening(raw_src.dtype, getattr(module, param_key + "_comfy_model_dtype", None))):
                    # Converted at load to a wider dtype (e.g. Qwen-Image's fp8 biases to bf16 under --bf16-unet):
                    # every value is exact in the wider dtype, so the value the block computes with (_effective_weight:
                    # widened, then stored back in the parameter's dtype) is the file's bytes
                    pass
                elif _effective_weight(module, param_key, src, target_device, compute_dtype) is not src:
                    return None, f"value differs from the stored weight ({name}: dtype conversion or LoRA)"
                abs_off = layout["param_abs"][b_i][j]
                if raw_src.device.type == "cpu":
                    info = getattr(raw_src.untyped_storage(), "_comfy_tensor_file_slice", None)
                    if info is None or info.offset != abs_off or raw_src.storage_offset() != 0:
                        return None, f"{name} is not the file's mmap view"
                else:
                    nbytes = raw_src.numel() * raw_src.element_size()
                    f.seek(abs_off)
                    raw = torch.frombuffer(bytearray(f.read(nbytes)), dtype=torch.uint8)
                    if not torch.equal(raw, raw_src.detach().contiguous().view(-1).view(torch.uint8).cpu()):
                        return None, f"{name} on the GPU differs from the file"
    return {"path": path, "ranges": layout["ranges"], "rel_offsets": layout["rel_offsets"],
            "block_bytes": layout["block_bytes"], "lora_idx": sorted(lora_idx)}, None


# dtype -> dtypes that hold each of its values exactly (fp8 e4m3 / e5m2 fit in fp16, bf16 and fp32; fp16 / bf16 in fp32)
_EXACT_WIDER = {
    torch.float8_e4m3fn: (torch.float16, torch.bfloat16, torch.float32),
    torch.float8_e5m2: (torch.float16, torch.bfloat16, torch.float32),
    torch.float16: (torch.float32,),
    torch.bfloat16: (torch.float32,),
}


def _exact_widening(src_dtype, model_dtype):
    """True if a load-time conversion src_dtype -> model_dtype keeps every value exactly (and back again)."""
    return model_dtype is not None and model_dtype != src_dtype and model_dtype in _EXACT_WIDER.get(src_dtype, ())


def _quant_lora_transfer_vram(dit, patcher):
    """Extra VRAM for re-quantizing LoRA'd scaled fp8 weights after each transfer (WEIGHTFERRY_QUANT_LORA=transfer
    or auto): dequantized copy, LoRA result, scaled copy, random bytes and the new fp8 data of the largest weight,
    plus the caching allocator's pool on the upload stream: ~24 bytes per element of the largest weight (measured
    on Cosmos-Predict2 14B).
    Runs before ComfyUI loads the model, so the LoRA is found in the patcher's patch list.
    Override: WEIGHTFERRY_QUANT_LORA_VRAM_MB."""
    # Only when WEIGHTFERRY_QUANT_LORA is set explicitly (raw read on purpose: unset means auto at load time, but the
    # load then keeps to bake when nothing was reserved here -- load_steps._direct_file_source)
    if os.environ.get("WEIGHTFERRY_QUANT_LORA", "bake") not in ("transfer", "auto") or patcher is None:
        return 0
    patched = set(getattr(patcher, "patches", {}) or {})
    if not patched:
        return 0
    attr = _block_list_attr(dit) or "blocks"
    largest = 0
    for i, blk in enumerate(_blocks(dit)):
        for name, prm in blk.named_parameters():
            if _is_quant(prm.data) and f"{_key_prefix(dit)}{attr}.{i}.{name}" in patched:
                largest = max(largest, _raw(prm.data).numel())
    if not largest:
        return 0
    env = _env.get("WEIGHTFERRY_QUANT_LORA_VRAM_MB")
    return int(env) * 1024 ** 2 if env else 24 * largest


def _ram_tier_replan(plan, num_stream, block_bytes, extra_margin=0):
    """RAM tier re-plan on a reload (TIERING_AUTO, automatic RAM budget): the tier plan of the cached host buffers was
    made with the RAM available at that time. If the RAM now available (+ the cached buffers, released before a new
    capture) gives a RAM tier at least max(2, 10 % of the streaming blocks) blocks and 1 GB larger or smaller, returns
    (old RAM blocks, new RAM blocks) so the caller captures again; else None. Same rule as tiered_store.plan_tiers:
    the budget is available RAM - WEIGHTFERRY_RAM_MARGIN_GB (default 6 GB); other processes' memory is already
    excluded from "available".
    WEIGHTFERRY_RAM_REPLAN=0 disables it."""
    if not plan or _env.get("WEIGHTFERRY_RAM_REPLAN") == "0" or plan.get("disk_read_gbps") is None:
        return None
    avail = tiered_store.available_ram()
    # + extra_margin: the activation host buffers the coming run will take (_act_host_reserve)
    margin = int(float(_env.get("WEIGHTFERRY_RAM_MARGIN_GB")) * 1024 ** 3) + int(extra_margin)
    # Still held now: the cached RAM-tier blocks (+ resident copies); the SSD ring was released with the old state
    held = (int(plan.get("ram_blocks", 0)) + int(plan.get("resident_ram_blocks", 0))) * block_bytes
    budget = max(0, avail + held - margin)
    # Hysteresis on the budget itself: near "every block fits" the estimate below (which reserves a ring)
    # and plan_tiers (no ring when everything fits) disagree, so a few hundred MB of noise in the available RAM would
    # flip the plan back and forth and capture the model again on every sampling call (+14-24 s each for Cosmos 14B).
    # No re-plan while the budget is within WEIGHTFERRY_RAM_REPLAN_HYST_GB (1.5) of the budget the cached capture
    # was planned with
    prev = plan.get("ram_budget_mb")
    hyst = float(_env.get("WEIGHTFERRY_RAM_REPLAN_HYST_GB")) * 1024 ** 3
    if prev is not None and abs(budget - float(prev) * 1024 ** 2) < hyst:
        return None
    depth = int(plan.get("ring_depth_wanted") or plan.get("ring_depth") or 4)
    if budget >= num_stream * block_bytes:
        new = num_stream
    else:
        new = int(max(0, min(num_stream, (budget - depth * block_bytes) // max(1, block_bytes))))
    old = int(plan.get("ram_blocks", num_stream))
    # A re-capture costs a file read + LoRA bake (Cosmos 14B: ~14 s) and SSD writes: only for a clear change
    if abs(new - old) >= max(2, num_stream // 10) and abs(new - old) * block_bytes >= 1024 ** 3:
        return old, new
    return None
