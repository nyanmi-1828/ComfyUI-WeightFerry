# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
VRAM overflow into shared GPU memory (system RAM) on Windows.

When a CUDA process needs more VRAM than the GPU has, the NVIDIA driver on Windows (WDDM, "CUDA - Sysmem Fallback
Policy" at its default) does not raise out-of-memory: it places the allocations in shared GPU memory, i.e. system RAM
over PCIe, and everything that touches them becomes many times slower. Users with little VRAM are the ones who hit it,
and nothing in PyTorch reports it.

The Windows performance counters "GPU Process Memory(pid_<pid>_*)\\Dedicated Usage / Shared Usage" do show it per
process. Pinned (page-locked) host memory is counted in Shared Usage too (measured: 1 GB pinned -> +1088 MB
Shared Usage; a 1 GB VRAM overflow -> +1472 MB), so the overflow estimate subtracts the pinned memory this process
is known to hold: ComfyUI's pinned weights (comfy.model_management.TOTAL_PINNED_MEMORY), this package's pinned
buffers (tiered_store) and PyTorch's pinned-memory allocator.

  sample()          -> {"dedicated", "shared", "pinned_known", "overflow"} in bytes, or None (not Windows / no counter)
  overflow_bytes()  -> estimated bytes of this process' GPU memory living in system RAM (0 if none / unknown)
"""

import os
import sys
import threading
from . import env as _env

_IS_WINDOWS = sys.platform == "win32"
# Pinned memory not tracked by the three sources (driver / runtime internals): tolerated before reporting an overflow
SLACK_BYTES = int(_env.get("WEIGHTFERRY_OVERFLOW_SLACK_MB")) * 1024 ** 2

_pdh = None
# The first query of a process takes ~220 ms (the PDH library and the counter set are loaded; later queries take
# ~0.04 ms, measured on Windows 11): prewarm() pays it on a background thread when the streaming node is applied, so
# the first generation does not wait for it. The lock keeps the warm-up and a real query from overlapping.
_PDH_LOCK = threading.Lock()
_prewarmed = False


def _pdh_lib():
    global _pdh
    if _pdh is None:
        import ctypes
        from ctypes import wintypes

        class FMT_VALUE(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

        class FMT_ITEM(ctypes.Structure):
            _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", FMT_VALUE)]

        lib = ctypes.WinDLL("pdh")
        _pdh = (ctypes, wintypes, lib, FMT_ITEM)
    return _pdh


def _gpu_process_memory(pid):
    """(dedicated, shared) bytes of process ``pid`` summed over its GPU adapters, or None."""
    if not _IS_WINDOWS:
        return None
    with _PDH_LOCK:
        return _gpu_process_memory_locked(pid)


def prewarm():
    """Load the counters once in the background (Windows only, once per process)."""
    global _prewarmed
    if not _IS_WINDOWS or _prewarmed:
        return
    _prewarmed = True
    threading.Thread(target=_gpu_process_memory, args=(os.getpid(),), name="weightferry-vram-monitor-prewarm",
                     daemon=True).start()


def _gpu_process_memory_locked(pid):
    ctypes, wintypes, pdh, FMT_ITEM = _pdh_lib()
    PDH_FMT_LARGE = 0x00000400
    PDH_MORE_DATA = 0x800007D2
    q = wintypes.HANDLE()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
        return None
    try:
        handles = []
        for name in ("Dedicated Usage", "Shared Usage"):
            c = wintypes.HANDLE()
            if pdh.PdhAddEnglishCounterW(q, f"\\GPU Process Memory(pid_{pid}_*)\\{name}", None, ctypes.byref(c)) != 0:
                return None
            handles.append(c)
        if pdh.PdhCollectQueryData(q) != 0:
            return None
        values = []
        for c in handles:
            size = wintypes.DWORD(0)
            count = wintypes.DWORD(0)
            r = pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
            if (r & 0xFFFFFFFF) != PDH_MORE_DATA:
                values.append(0)  # no instance yet (the process has not touched the GPU)
                continue
            buf = (ctypes.c_byte * size.value)()
            if pdh.PdhGetFormattedCounterArrayW(c, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf) != 0:
                return None
            items = ctypes.cast(buf, ctypes.POINTER(FMT_ITEM))
            values.append(sum(items[i].FmtValue.largeValue for i in range(count.value)))
        return values[0], values[1]
    except Exception:  # noqa: BLE001 -- monitoring must never break generation
        return None
    finally:
        pdh.PdhCloseQuery(q)


def known_pinned_bytes():
    """Pinned host memory this process is known to hold (counted in Shared Usage by Windows)."""
    total = 0
    try:
        import comfy.model_management as mm
        total += int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0)
    except Exception:  # noqa: BLE001 -- monitoring only
        pass
    try:
        from . import tiered_store
        total += int(tiered_store._REGISTERED["bytes"])
    except Exception:  # noqa: BLE001 -- monitoring only
        pass
    try:
        import torch
        stats = torch.cuda.host_memory_stats()
        total += int(stats.get("reserved_bytes.all.current", 0) or 0)
    except Exception:  # noqa: BLE001 -- monitoring only
        pass
    return total


def sample(pid=None):
    """Current GPU memory of this process and the overflow estimate, or None when the counters are unavailable."""
    mem = _gpu_process_memory(pid or os.getpid())
    if mem is None:
        return None
    dedicated, shared = mem
    pinned = known_pinned_bytes()
    return {"dedicated": dedicated, "shared": shared, "pinned_known": pinned,
            "overflow": max(0, shared - pinned - SLACK_BYTES)}


def overflow_bytes(pid=None):
    s = sample(pid)
    return 0 if s is None else s["overflow"]


def effective_free_vram(device):
    """VRAM still free for this process as ComfyUI plans it: ComfyUI's free memory (device free + PyTorch's unused
    cache) minus ComfyUI's reserve (--reserve-vram), and at most what PyTorch's per-process cap still allows. Used for
    the automatic low-VRAM switches (activation streaming, TaylorSeer's cache in RAM) on large images."""
    import torch
    import comfy.model_management as mm
    free = mm.get_free_memory(device) - mm.extra_reserved_memory()
    try:
        fraction = torch.cuda.get_per_process_memory_fraction(device)
        if fraction < 1.0:
            total = torch.cuda.get_device_properties(device).total_memory
            free = min(free, fraction * total - torch.cuda.memory_allocated(device))
    except (AttributeError, RuntimeError):  # older PyTorch: no cap to read
        pass
    return free
