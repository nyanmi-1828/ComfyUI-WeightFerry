# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Shared helpers of the DiT / text encoder layer streaming: scaled fp8 (QuantizedTensor) helpers, constants,
flat block layouts, block discovery, parameter owners, LoRA application."""

import dataclasses
import weakref

import torch

try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:  # pragma: no cover
    QuantizedTensor = ()


# Scaled fp8 weights (ComfyUI's comfy_quant formats) are QuantizedTensor parameters: a wrapper around the stored
# fp8 data (_qdata, mmap-backed under Dynamic VRAM) plus layout parameters (_params: per-tensor scale, orig dtype /
# shape). Streaming moves only the _qdata bytes; the parameters are kept on the GPU per block and a QuantizedTensor
# around the slot's bytes is bound for compute.
def _is_quant(t):
    return isinstance(t, QuantizedTensor) if QuantizedTensor != () else False


def _raw(t):
    """The stored bytes' tensor: _qdata of a QuantizedTensor, the tensor itself otherwise."""
    return t._qdata if _is_quant(t) else t


def _param_nbytes(prm):
    r = _raw(prm.data)
    return r.numel() * r.element_size()


def _quant_params_to(params, device, copy=False):
    """Copy of QuantizedTensor layout parameters with every tensor field on ``device`` (copy=True: always new
    tensors, so they can be overwritten in place)."""
    changes = {}
    for fld in dataclasses.fields(params):
        val = getattr(params, fld.name)
        if isinstance(val, torch.Tensor):
            changes[fld.name] = val.to(device=device, copy=copy)
    return dataclasses.replace(params, **changes) if changes else params


def _copy_quant_params_(dst, src):
    """In-place copy of the tensor fields of QuantizedTensor layout parameters (current stream)."""
    for fld in dataclasses.fields(dst):
        val = getattr(dst, fld.name)
        if isinstance(val, torch.Tensor):
            val.copy_(getattr(src, fld.name), non_blocking=True)


def _quant_seed(module):
    """Seed of ComfyUI's stochastic rounding when it re-quantizes this module's LoRA result (ops.py:
    string_to_seed(s.seed_key)), or None when the module has no seed key."""
    key = getattr(module, "seed_key", None)
    if key is None:
        return None
    import comfy.utils
    return comfy.utils.string_to_seed(key)


def _quant_lora_requant(src_qt, lora_fn, compute_dtype, seed):
    """ComfyUI's cast-time LoRA on a scaled fp8 weight (ops.resolve_cast_module_with_vbar, post_cast with
    want_requant=True, which the mixed-precision Linear passes when it quantizes its input): dequantize to the
    compute dtype, apply the LoRA there, quantize again with a recalculated scale and seeded stochastic rounding.
    The module computes with that re-quantized weight (fp8 matmul), and ComfyUI caches it for later forwards.
    Deterministic, so it can be computed once at capture or after each transfer.
    Known gap: a module that also has weight wrapper patches (``weight_function``) does not quantize its input, so
    ComfyUI's first forward uses the un-requantized LoRA result there (later forwards use the cached re-quantized
    one); streaming always uses the re-quantized weight, so only that first forward would differ."""
    x = src_qt.to(dtype=compute_dtype)
    if _is_quant(x):
        x = x.dequantize()
    x = lora_fn(x)
    return src_qt.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)


_STATE_ATTR = "_weightferry_stream_state"
_ACTIVE_ATTR = "_weightferry_stream_active_owner"
_CACHE_ATTR = "_weightferry_stream_capture_cache"
# Layer stacks streamed by WeightFerryTextEncoder (weak: a CLIP that is gone drops out)
_TE_ROOTS = weakref.WeakSet()
_WRAPPER_KEY = "weightferry_dit"
_ALIGN = 256  # bytes, alignment of each parameter view inside a flat buffer

TIERING_RAM = "RAM (all streamed blocks pinned in RAM)"
TIERING_AUTO = "Auto (GPU / RAM / SSD)"

RESIDENT_MODE_COMFY = "ComfyUI (Dynamic VRAM)"
RESIDENT_MODE_SELF = "Self-managed (fixed on GPU)"

HOST_FORMAT_BAKED = "Compute dtype, LoRA baked (fast with LoRA)"
HOST_FORMAT_STORAGE = "Storage dtype (fp8), LoRA on GPU (half transfer)"


def _nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * dtype.itemsize


def _flat_layout(specs):
    """Byte offsets of each (shape, dtype) spec inside one flat buffer, and the total size (aligned)."""
    offsets, offset = [], 0
    for shape, dtype in specs:
        offset = (offset + _ALIGN - 1) // _ALIGN * _ALIGN
        offsets.append(offset)
        offset += _nbytes(shape, dtype)
    return offsets, (offset + _ALIGN - 1) // _ALIGN * _ALIGN


def _views(flat, specs, offsets):
    return [
        flat[o:o + _nbytes(shape, dtype)].view(dtype).view(shape)
        for (shape, dtype), o in zip(specs, offsets)
    ]


# Attribute holding the transformer block list: Cosmos / Anima ("blocks"), Qwen-Image ("transformer_blocks").
# The blocks of a list must share one parameter layout (checked at load time).
_BLOCK_LIST_ATTRS = ("blocks", "transformer_blocks", "layers", "block")  # DiT lists first; then text encoders


def _block_list_attr(dit):
    for name in _BLOCK_LIST_ATTRS:
        lst = getattr(dit, name, None)
        if isinstance(lst, torch.nn.ModuleList) and len(lst) > 0:
            return name
    return None


def _key_prefix(dit):
    """Prefix of this streamed module's parameters in the ModelPatcher's patch keys ("diffusion_model." for a DiT;
    the text encoder's path inside cond_stage_model for text encoder streaming)."""
    return getattr(dit, "_weightferry_key_prefix", "diffusion_model.")


def _blocks(dit):
    name = _block_list_attr(dit)
    return getattr(dit, name) if name else []


def _param_owner(blk, name):
    """(module, parameter key) of the parameter ``name`` (as named_parameters() of ``blk`` gives it)."""
    mod_name, _, param_key = name.rpartition(".")
    return (blk.get_submodule(mod_name) if mod_name else blk), param_key


def _storage_dtype(module, param_key, src):
    """Narrower storage dtype ComfyUI computes this parameter from (VBAR-managed modules only), else None."""
    if not hasattr(module, "_v") or src.device.type != "cpu":
        return None
    model_dtype = getattr(module, param_key + "_comfy_model_dtype", None)
    if model_dtype is None or model_dtype == src.dtype or model_dtype.itemsize >= src.dtype.itemsize:
        return None
    return model_dtype


def _to_device_tree(obj, device):
    """Copy of a LoRA patch structure with its tensors moved to ``device`` (dtype unchanged).
    0-dim tensors (e.g. alpha) stay where they are so the arithmetic is identical to ComfyUI's."""
    import copy
    from comfy import weight_adapter
    if isinstance(obj, torch.Tensor):
        return obj.to(device) if obj.dim() > 0 else obj
    if isinstance(obj, weight_adapter.WeightAdapterBase):
        clone = copy.copy(obj)
        clone.weights = _to_device_tree(obj.weights, device)
        return clone
    if isinstance(obj, tuple):
        return tuple(_to_device_tree(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device_tree(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device_tree(v, device) for k, v in obj.items()}
    return obj


def _lora_apply_fn(module, param_key, device):
    """ComfyUI's Dynamic VRAM LoRA patch for this parameter, as a function of the dequantized weight.
    Equivalent to LowVramPatch.__call__ without its (possibly stale) prepared patches. The LoRA tensors are
    staged on the GPU once per load: comfy.lora would otherwise copy them synchronously from pageable host
    memory on every forward, stalling the CPU that feeds the compute stream."""
    fn = getattr(module, param_key + "_lowvram_function", None)
    if fn is None:
        return None
    import comfy.lora
    key = fn.key
    gpu_patches = [
        (p[0], _to_device_tree(p[1], device)) + tuple(p[2:]) for p in fn.patches[key]
    ]
    return lambda x: comfy.lora.calculate_weight(gpu_patches, x, key, intermediate_dtype=x.dtype)
