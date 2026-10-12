# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Tier planning: how many streaming blocks stay in RAM and which go to the SSD (plan_tiers), and the safetensors
layouts used to read blocks straight from a model file.
"""

import json
import math
import sys

import torch

from .common import MB
from .pinned import _free_deferred
from .disk_io import probe_disk
from .cache_files import default_cache_dir
from .. import env as _env


# ---------------------------------------------------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------------------------------------------------
def ssd_positions(num_stream, n_disk, layout=None):
    """Streaming-block indices served from the SSD (ascending), for ``n_disk`` of ``num_stream`` blocks.

    "interleave" (default, env WEIGHTFERRY_SSD_LAYOUT): spread evenly, the last block always on the SSD. Between
    two SSD-tier blocks the GPU computes RAM-tier blocks, so the reader keeps reading with a small ring.
    "tail": the last n_disk blocks. With a ring of a few buffers the reader then sits idle while the RAM-tier
    blocks compute (measured: the SSD-tier reads did not overlap compute)."""
    layout = layout or _env.get("WEIGHTFERRY_SSD_LAYOUT")
    if n_disk <= 0:
        return []
    if layout == "tail" or n_disk >= num_stream:
        return list(range(num_stream - n_disk, num_stream))
    return [int((k + 1) * num_stream / n_disk) - 1 for k in range(n_disk)]


def available_ram(trim=True):
    """Available system RAM for the RAM tier plan. On Windows this process' working set is first moved to the
    standby list (EmptyWorkingSet): pages of model files read through mmap (ComfyUI's weights, or the LoRA bake of a
    capture) stay in the working set and are not counted as available although they are only a clean file cache
    (otherwise available RAM was under-counted by ~2.4 GB).
    Nothing is lost, the pages stay cached; pinned (locked) pages are not affected. WEIGHTFERRY_TRIM_WS=0 skips it."""
    import psutil
    _free_deferred()
    if trim and sys.platform == "win32" and _env.get("WEIGHTFERRY_TRIM_WS") != "0":
        try:
            import ctypes
            psapi = ctypes.WinDLL("psapi")
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            psapi.EmptyWorkingSet(k32.GetCurrentProcess())
        except Exception:  # noqa: BLE001 -- trimming is only for a more accurate measurement
            pass
    return psutil.virtual_memory().available


def plan_tiers(num_stream, block_bytes, ram_budget_bytes=None, ram_margin_bytes=None, disk_dir=None,
               compute_ms_per_block=None, total_blocks=None, resident_blocks=0, resident_block_bytes=None):
    """Decide how many streaming blocks stay in pinned RAM and how many are served from the SSD.

    - RAM budget: explicit, or (available system RAM - margin). The margin keeps headroom for ComfyUI,
      activations and the OS (default 6 GB, env WEIGHTFERRY_RAM_MARGIN_GB).
    - Ring depth: enough read-ahead to cover one block's SSD read time with block compute time (+2), 2..8.
      The ring is pinned RAM too, so it counts against the budget: when the budget cannot hold the wanted depth,
      the depth is lowered to what fits (at least 2; the plan reports ``over_budget`` if even 2 do not fit).
    - The SSD tier is predicted to be the bottleneck when reading all SSD blocks takes longer than one forward
      of compute; the plan says so (it still runs, just slower than RAM).
    - Self-managed resident blocks (``resident_blocks``): their host copies are only read once per model load
      (to fill their GPU buffers), so they get the RAM left over after the streaming blocks and go to the SSD
      first (resident_ram_blocks / resident_disk_blocks).
    """
    avail = available_ram(trim=ram_budget_bytes is None)
    if ram_margin_bytes is None:
        ram_margin_bytes = int(float(_env.get("WEIGHTFERRY_RAM_MARGIN_GB")) * 1024 ** 3)
    budget = ram_budget_bytes if ram_budget_bytes is not None else max(0, avail - ram_margin_bytes)

    disk_dir = disk_dir or default_cache_dir()
    disk_threads, disk_by_threads = 1, None
    try:
        probe = probe_disk(disk_dir)
    except Exception as exc:  # noqa: BLE001 -- no SSD tier speed: plan without it (error kept in the plan)
        disk_gbps = None
        disk_error = str(exc)
    else:
        disk_error = None
        disk_by_threads = probe["by_threads"]
        disk_threads, disk_gbps = probe["threads"], probe["gbps"]
        env_threads = _env.get("WEIGHTFERRY_DISK_THREADS")
        if env_threads:
            disk_threads = max(1, int(env_threads))
            disk_gbps = disk_by_threads.get(disk_threads, disk_gbps)

    t_disk_ms = None if not disk_gbps else block_bytes / (disk_gbps * 1e9) * 1000
    if t_disk_ms is not None and compute_ms_per_block:
        depth = max(2, min(8, math.ceil(t_disk_ms / compute_ms_per_block) + 2))
    else:
        depth = 4
    depth_wanted = depth

    n_res = int(resident_blocks or 0)
    res_bytes = int(resident_block_bytes or block_bytes)
    if disk_gbps is None or budget >= num_stream * block_bytes + n_res * res_bytes:
        ram_blocks, res_ram = num_stream, n_res
    else:
        # Streaming blocks first (read every forward), then the resident blocks' copies (read once per load)
        if budget >= num_stream * block_bytes:
            ram_blocks = num_stream
        else:
            ram_blocks = int(max(0, min(num_stream, (budget - depth * block_bytes) // block_bytes)))
            if ram_blocks == 0 and depth * block_bytes > budget:
                # Not even the wanted ring fits: shrink it to the budget (2 at least)
                depth = max(2, min(depth, int(budget // block_bytes)))
        left = budget - (ram_blocks + (depth if ram_blocks < num_stream else 0)) * block_bytes
        res_ram = int(max(0, min(n_res, left // res_bytes)))
    n_disk = num_stream - ram_blocks
    if n_disk == 0:
        depth = 0

    host_planned = (ram_blocks + depth) * block_bytes + res_ram * res_bytes
    plan = {
        "ram_blocks": ram_blocks,
        "disk_blocks": n_disk,
        "ring_depth": depth,
        "ring_depth_wanted": depth_wanted if n_disk else 0,
        "block_mb": round(block_bytes / MB, 1),
        "ram_budget_mb": round(budget / MB, 1),
        "available_ram_mb": round(avail / MB, 1),
        "resident_ram_blocks": res_ram,
        "resident_disk_blocks": n_res - res_ram,
        "host_ram_planned_mb": round(host_planned / MB, 1),
        # also when the SSD tier could not be used (no read speed: everything in RAM) and RAM is short (Codex CPU
        # check 2026-10-13: the all-RAM fallback reported over_budget False with 3.7 GB planned against 0)
        "over_budget": host_planned > budget,
        "disk_dir": disk_dir,
        "disk_read_gbps": disk_gbps,
        "disk_threads": disk_threads,
        "disk_read_gbps_by_threads": disk_by_threads,
        "disk_error": disk_error,
        "disk_ms_per_block": None if t_disk_ms is None else round(t_disk_ms, 2),
        "compute_ms_per_block": None if compute_ms_per_block is None else round(compute_ms_per_block, 2),
    }
    if t_disk_ms is not None and compute_ms_per_block and total_blocks:
        disk_ms = n_disk * t_disk_ms
        forward_ms = total_blocks * compute_ms_per_block
        plan["disk_ms_per_forward"] = round(disk_ms, 1)
        plan["compute_ms_per_forward_est"] = round(forward_ms, 1)
        plan["disk_bound"] = disk_ms > forward_ms
    return plan


def safetensors_header(path):
    """(header dict without __metadata__, absolute offset of the data section)."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


_PREFIXES = ("", "net.", "model.diffusion_model.", "diffusion_model.", "model.")


def _find(header, name):
    """Header entry of ``name`` under any file prefix ("", "net.", "model.diffusion_model.", ...), or None."""
    for pre in _PREFIXES:
        info = header.get(pre + name)
        if info is not None:
            return info
    return None


_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "I64": torch.int64, "I32": torch.int32,
              "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def file_block_layout(path, block_names):
    """Byte layout of whole blocks inside a safetensors file.

    ``block_names``: per block, the parameter names relative to the diffusion model, e.g. "blocks.3.mlp.layer1.weight"
    (any file prefix such as "net." or "model.diffusion_model." is matched). Returns None if a tensor is missing or a
    block is not one contiguous range with the same relative layout as the first block; otherwise
    {"ranges": [(abs offset, bytes)], "rel_offsets": [offset of each param in its block], "dtypes": [...],
    "shapes": [...]}."""
    header, base = safetensors_header(path)
    ranges, rel0, dtypes, shapes = [], None, None, None
    for names in block_names:
        infos = []
        for name in names:
            info = _find(header, name)
            if info is None or info["dtype"] not in _ST_DTYPES:
                return None
            infos.append(info)
        starts = [i["data_offsets"][0] for i in infos]
        ends = [i["data_offsets"][1] for i in infos]
        lo, hi = min(starts), max(ends)
        if sum(e - s for s, e in zip(starts, ends)) != hi - lo:
            return None  # gaps or foreign tensors inside the block's range
        rel = [st - lo for st in starts]
        if rel0 is None:
            rel0 = rel
            dtypes = [_ST_DTYPES[i["dtype"]] for i in infos]
            shapes = [tuple(i["shape"]) for i in infos]
        elif rel != rel0 or [tuple(i["shape"]) for i in infos] != shapes:
            return None
        ranges.append((base + lo, hi - lo))
    return {"ranges": ranges, "rel_offsets": rel0, "dtypes": dtypes, "shapes": shapes}


def file_param_layout(path, block_names):
    """Byte layout of blocks whose tensors are scattered in a safetensors file.

    Some files group tensors by dtype (e.g. qwen_2.5_vl_7b_fp8_scaled.safetensors stores all F32 scales first,
    then the fp8 weights, then the bf16 values, so a layer is 2 ranges). The host block packs the parameters densely
    in file order, which must be the same in every block. Returns None if a tensor is missing, shapes differ from the
    first block or the file order differs; else {"ranges": per block a list of pieces (abs offset, bytes, offset in
    the packed block), adjacent tensors merged, "rel_offsets": per parameter (block_names order) its offset in the
    packed block, "param_abs": per block the absolute file offset of each parameter, "block_bytes", "dtypes",
    "shapes", "pieces_per_block"}."""
    header, base = safetensors_header(path)
    ranges, param_abs, rel, dtypes, shapes, order0, block_bytes = [], [], None, None, None, None, 0
    for names in block_names:
        infos = []
        for name in names:
            info = _find(header, name)
            if info is None or info["dtype"] not in _ST_DTYPES:
                return None
            infos.append(info)
        order = sorted(range(len(infos)), key=lambda k: infos[k]["data_offsets"][0])
        sh = [tuple(i["shape"]) for i in infos]
        if order0 is None:
            order0, shapes, dtypes = order, sh, [_ST_DTYPES[i["dtype"]] for i in infos]
            rel, off = [0] * len(infos), 0
            for k in order:
                rel[k] = off
                off += infos[k]["data_offsets"][1] - infos[k]["data_offsets"][0]
            block_bytes = off
        elif order != order0 or sh != shapes:
            return None
        abs_offs = [base + i["data_offsets"][0] for i in infos]
        pieces = []
        for k in order:
            o, n = abs_offs[k], infos[k]["data_offsets"][1] - infos[k]["data_offsets"][0]
            if pieces and pieces[-1][0] + pieces[-1][1] == o and pieces[-1][2] + pieces[-1][1] == rel[k]:
                pieces[-1] = (pieces[-1][0], pieces[-1][1] + n, pieces[-1][2])
            else:
                pieces.append((o, n, rel[k]))
        ranges.append(pieces)
        param_abs.append(abs_offs)
    return {"ranges": ranges, "rel_offsets": rel, "param_abs": param_abs, "block_bytes": block_bytes,
            "dtypes": dtypes, "shapes": shapes, "pieces_per_block": [len(r) for r in ranges]}
