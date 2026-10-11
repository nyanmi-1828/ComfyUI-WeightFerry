# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Page-locked (pinned) host memory: exact-size cudaHostRegister buffers (PinnedHost), their release, and
process memory snapshots.
"""

import threading

import torch

from .common import MB, _IS_WINDOWS, align_up
from .. import env as _env
try:
    from .common import ctypes
except ImportError:
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Exact-size pinned host memory
# ---------------------------------------------------------------------------------------------------------------------
_REGISTERED = {"bytes": 0, "count": 0}
_LOCK = threading.RLock()  # the counters and the deferred frees (buffers are also made in helper threads)


# Windows: host buffers straight from VirtualAlloc, given back with VirtualFree. PyTorch's CPU allocator keeps freed
# memory in the process (measured: of 28 x 222 MB freed, 1.8 GB went back to the OS; after
# cudaHostRegister / unregister only 0.26 GB), so releasing a streamed text encoder's 6.2 GB RAM tier freed almost
# nothing for the DiT. WEIGHTFERRY_VIRTUALALLOC=0 uses PyTorch's allocator again.
_VA = None
_DEFERRED_FREE = []  # (base tensor, ptr): views were still alive at release; freed once they are gone


def _virtual_alloc_api():
    global _VA
    if _VA is None:
        _VA = False
        if _IS_WINDOWS and _env.get("WEIGHTFERRY_VIRTUALALLOC") != "0":
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.VirtualAlloc.restype = ctypes.c_void_p
                k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
                k32.VirtualFree.restype = ctypes.c_int
                k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
                _VA = k32
            except Exception:  # noqa: BLE001 -- no VirtualAlloc: PinnedHost uses torch.empty
                _VA = False
    return _VA


def _storage_users(t):
    """References to t's storage besides t itself (views, other tensors); None if unknown."""
    fn = getattr(torch._C, "_storage_Use_Count", None)
    if fn is None:
        return None
    return int(fn(t.untyped_storage()._cdata)) - 2  # the base tensor + the temporary storage object


def _free_deferred():
    k32 = _virtual_alloc_api()
    with _LOCK:
        keep = []
        for base, ptr in _DEFERRED_FREE:
            users = _storage_users(base)
            if users == 0 and k32:
                del base
                k32.VirtualFree(ptr, 0, 0x8000)  # MEM_RELEASE
            else:
                keep.append((base, ptr))
        _DEFERRED_FREE[:] = keep


class PinnedHost:
    """Page-locked host buffer of exactly ``nbytes`` (plus optional alignment), via cudaHostRegister.

    PyTorch's pinned allocator (pin_memory=True) rounds every allocation up to a power of two and keeps freed
    blocks cached in the process: a 132 MB block took 256 MB, so 20 blocks used 5.1 GB instead of 2.6 GB
    (measured). Registering an ordinary allocation pins exactly what is needed, and release() returns it
    to the OS immediately. Falls back to PyTorch's allocator if registration fails. Call release() when done
    (also done on garbage collection).
    """

    def __init__(self, nbytes, align=0):
        self.nbytes = int(nbytes)
        self._va_ptr = None
        k32 = _virtual_alloc_api()
        if k32:
            _free_deferred()
            total = align_up(self.nbytes + align)
            ptr = k32.VirtualAlloc(None, total, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
            if ptr:
                self._va_ptr = ptr
                self._base = torch.frombuffer((ctypes.c_uint8 * total).from_address(ptr), dtype=torch.uint8)
        if self._va_ptr is None:
            self._base = torch.empty(self.nbytes + align, dtype=torch.uint8, device="cpu")
        self._ptr = self._base.data_ptr()
        self.registered = False
        try:
            err = torch.cuda.cudart().cudaHostRegister(self._ptr, self._base.numel(), 0)
            self.registered = int(err) == 0
        except Exception:  # noqa: BLE001 -- not registered = pageable buffer (still correct, slower DMA)
            self.registered = False
        if self.registered:
            with _LOCK:
                _REGISTERED["bytes"] += self._base.numel()
                _REGISTERED["count"] += 1
        else:
            self._free_va()
            self._base = torch.empty(self.nbytes + align, dtype=torch.uint8, device="cpu", pin_memory=True)
            self._ptr = self._base.data_ptr()
        off = (-self._ptr) % align if align else 0
        self.tensor = self._base[off:off + self.nbytes]

    def release(self):
        if self._base is None:
            return
        if self.registered:
            try:
                torch.cuda.cudart().cudaHostUnregister(self._ptr)
            except Exception:  # noqa: BLE001 -- releasing: nothing left to do if CUDA is gone
                pass
            with _LOCK:
                _REGISTERED["bytes"] -= self._base.numel()
                _REGISTERED["count"] -= 1
            self.registered = False
        self.tensor = None
        self._free_va()
        self._base = None

    def _free_va(self):
        """VirtualFree the buffer if nothing else uses it, else free it later (_free_deferred)."""
        if self._va_ptr is None:
            return
        ptr, base = self._va_ptr, self._base
        self._va_ptr = None
        self._base = None
        if _storage_users(base) == 0:
            del base
            _virtual_alloc_api().VirtualFree(ptr, 0, 0x8000)  # MEM_RELEASE
        else:
            with _LOCK:
                _DEFERRED_FREE.append((base, ptr))

    def __del__(self):
        try:
            self.release()
        except Exception:  # noqa: BLE001 -- __del__ must not raise (interpreter shutdown)
            pass


def release_all(holders):
    for h in holders or ():
        if h is not None:
            h.release()


def empty_host_cache():
    """Return memory cached by PyTorch's pinned allocator to the OS, if this PyTorch exposes it."""
    _free_deferred()
    fn = getattr(torch._C, "_host_emptyCache", None)
    if fn is not None:
        try:
            fn()
        except Exception:  # noqa: BLE001 -- deferred free: best effort, the memory goes with the process
            pass


# ---------------------------------------------------------------------------------------------------------------------
# Memory snapshot (for logging where host RAM goes)
# ---------------------------------------------------------------------------------------------------------------------
def mem_snapshot():
    """Process working set / private bytes, system available RAM and PyTorch pinned host memory, in MB."""
    snap = {}
    try:
        import psutil
        mi = psutil.Process().memory_info()
        snap["rss_mb"] = round(mi.rss / MB, 1)
        if hasattr(mi, "private"):
            snap["private_mb"] = round(mi.private / MB, 1)
        snap["sys_available_mb"] = round(psutil.virtual_memory().available / MB, 1)
    except Exception:  # noqa: BLE001 -- logging only
        pass
    snap["weightferry_registered_pinned_mb"] = round(_REGISTERED["bytes"] / MB, 1)
    try:
        stats = torch.cuda.host_memory_stats()
        for k in ("allocated_bytes.all.current", "allocated_bytes.current", "reserved_bytes.all.current"):
            if k in stats:
                snap["torch_pinned_" + k.split(".")[0] + "_mb"] = round(stats[k] / MB, 1)
    except Exception:  # noqa: BLE001 -- logging only
        pass
    return snap
