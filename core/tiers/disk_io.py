# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Low-level file I/O: unbuffered reads / writes on Windows (FILE_FLAG_NO_BUFFERING), aligned host buffers and the
SSD read-speed probe.
"""

import json
import os
import threading
import time

import torch

from .common import MB, _ALIGN_IO, _IS_WINDOWS
try:
    from .common import _FILE_FLAG_NO_BUFFERING, _FILE_FLAG_SEQUENTIAL_SCAN, _FILE_SHARE_READ, _GENERIC_READ, _GENERIC_WRITE, _INVALID_HANDLE, _OPEN_EXISTING, _k32, ctypes, wintypes
except ImportError:
    pass
from .pinned import PinnedHost
import weakref as _weakref  # noqa: E402
_OPEN_READERS = _weakref.WeakSet()


class _UnbufferedReader:
    """Reads aligned ranges of a file into aligned host memory. Windows: FILE_FLAG_NO_BUFFERING (no page cache).
    Elsewhere: plain reads (the OS may cache them)."""

    _CHUNK = 64 * MB

    def __init__(self, path):
        self.path = path
        _OPEN_READERS.add(self)
        if _IS_WINDOWS:
            h = _k32.CreateFileW(path, _GENERIC_READ, _FILE_SHARE_READ, None, _OPEN_EXISTING,
                                 _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN, None)
            if h == _INVALID_HANDLE or h is None:
                raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
            self.handle = h
            self.f = None
        else:
            self.handle = None
            self.f = open(path, "rb", buffering=0)

    def read_into(self, offset, nbytes, dst_ptr, min_bytes=None):
        """Read ``nbytes`` (multiple of 4 KiB) at ``offset`` (multiple of 4 KiB) into address ``dst_ptr``.
        Reaching the end of the file is fine once ``min_bytes`` (default: all) have been read."""
        need = nbytes if min_bytes is None else min_bytes
        if self.handle is not None:
            if not _k32.SetFilePointerEx(self.handle, offset, None, 0):
                raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
            done = 0
            got = wintypes.DWORD(0)
            while done < nbytes:
                n = min(self._CHUNK, nbytes - done)
                if not _k32.ReadFile(self.handle, ctypes.c_void_p(dst_ptr + done), n, ctypes.byref(got), None):
                    raise OSError(ctypes.get_last_error(), "ReadFile failed")
                if got.value == 0:
                    if done >= need:
                        return
                    raise EOFError(f"unexpected EOF in {self.path}")
                done += got.value
        else:
            import ctypes as _ct
            self.f.seek(offset)
            buf = (_ct.c_char * nbytes).from_address(dst_ptr)
            view = memoryview(buf).cast("B")
            done = 0
            while done < nbytes:
                n = self.f.readinto(view[done:])
                if not n:
                    if done >= need:
                        return
                    raise EOFError(f"unexpected EOF in {self.path}")
                done += n

    def close(self):
        if self.handle is not None:
            _k32.CloseHandle(self.handle)
            self.handle = None
        if self.f is not None:
            self.f.close()
            self.f = None


class _UnbufferedWriter:
    """Windows: writes aligned ranges from aligned host memory with FILE_FLAG_NO_BUFFERING (no page cache, so no
    dirty pages to flush). Measured (9.4 GB cache file): buffered writes extending the file with an
    fsync per block ran at ~0.3 GB/s."""

    _CHUNK = 64 * MB

    def __init__(self, path):
        self.path = path
        h = _k32.CreateFileW(path, _GENERIC_WRITE, 0, None, _OPEN_EXISTING, _FILE_FLAG_NO_BUFFERING, None)
        if h == _INVALID_HANDLE or h is None:
            raise OSError(ctypes.get_last_error(), f"CreateFileW (write) failed for {path}")
        self.handle = h

    def write_from(self, offset, nbytes, src_ptr):
        """Write ``nbytes`` (multiple of 4 KiB) at ``offset`` (multiple of 4 KiB) from 4 KiB aligned ``src_ptr``."""
        if not _k32.SetFilePointerEx(self.handle, offset, None, 0):
            raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
        done = 0
        got = wintypes.DWORD(0)
        while done < nbytes:
            n = min(self._CHUNK, nbytes - done)
            if not _k32.WriteFile(self.handle, ctypes.c_void_p(src_ptr + done), n, ctypes.byref(got), None):
                raise OSError(ctypes.get_last_error(), "WriteFile failed")
            done += got.value

    def close(self):
        if self.handle is not None:
            _k32.CloseHandle(self.handle)
            self.handle = None


def aligned_host(nbytes, pin=True):
    """4 KiB aligned uint8 CPU buffer of ``nbytes``. Returns (tensor, PinnedHost holder or None)."""
    if pin:
        h = PinnedHost(nbytes, align=_ALIGN_IO)
        return h.tensor, h
    raw = torch.empty(nbytes + _ALIGN_IO, dtype=torch.uint8, device="cpu")
    off = (-raw.data_ptr()) % _ALIGN_IO
    return raw[off:off + nbytes], None


# ---------------------------------------------------------------------------------------------------------------------
# SSD probe
# ---------------------------------------------------------------------------------------------------------------------
_PROBE_FILE = "weightferry_disk_probe.json"
_PROBE_VERSION = 2


def _timed_parallel_read(path, nbytes, chunk_bytes, threads):
    """Read ``nbytes`` of ``path`` in ``chunk_bytes`` pieces with ``threads`` readers (chunk i -> thread i % n)."""
    errors = []

    def work(t):
        reader = None
        try:
            buf, _ = aligned_host(chunk_bytes, pin=False)
            reader = _UnbufferedReader(path)
            for i, off in enumerate(range(0, nbytes, chunk_bytes)):
                if i % threads == t:
                    reader.read_into(off, chunk_bytes, buf.data_ptr())
        except Exception as exc:  # noqa: BLE001 -- re-raised by the caller after the threads end
            errors.append(exc)
        finally:
            if reader is not None:
                reader.close()

    workers = [threading.Thread(target=work, args=(t,), daemon=True) for t in range(threads)]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    dt = time.perf_counter() - t0
    if errors:
        raise errors[0]
    return nbytes / dt / 1e9


def probe_disk(directory, size_mb=512, chunk_mb=32, thread_counts=(1, 2, 4), max_age_s=7 * 24 * 3600):
    """Unbuffered read bandwidth of the drive holding ``directory`` with 1 / 2 / 4 reader threads, cached for a week.
    Returns {"gbps": best GB/s, "threads": thread count of the best, "by_threads": {n: GB/s}}."""
    os.makedirs(directory, exist_ok=True)
    cache_path = os.path.join(directory, _PROBE_FILE)
    try:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if (time.time() - cached.get("time", 0) < max_age_s and cached.get("version") == _PROBE_VERSION
                and cached.get("size_mb") == size_mb and cached.get("chunk_mb") == chunk_mb
                and sorted(int(k) for k in cached.get("by_threads", {})) == sorted(thread_counts)):
            by = {int(k): v for k, v in cached["by_threads"].items()}
            best = max(by, key=by.get)
            return {"gbps": by[best], "threads": best, "by_threads": by}
    except (OSError, ValueError, KeyError, TypeError):  # no / old / damaged probe cache: measure again
        pass
    path = os.path.join(directory, "weightferry_disk_probe.bin")
    nbytes = size_mb * MB
    chunk = os.urandom(MB)
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    try:
        by = {n: round(_timed_parallel_read(path, nbytes, chunk_mb * MB, n), 2) for n in thread_counts}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    best = max(by, key=by.get)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"version": _PROBE_VERSION, "by_threads": by, "size_mb": size_mb, "chunk_mb": chunk_mb,
                       "time": time.time(), "unbuffered": _IS_WINDOWS}, f)
    except OSError:  # the result is only cached to skip the probe next time
        pass
    return {"gbps": by[best], "threads": best, "by_threads": by}
