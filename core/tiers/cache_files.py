# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
SSD cache files: their directory, names, clean-up of stale files, the block stores that write them
(DiskBlockStore, PersistentBlockStore) and the persistent cache's size cap.
"""

import os

import torch

from .common import _IS_WINDOWS, align_up
from .disk_io import _OPEN_READERS, _UnbufferedWriter, aligned_host
from .sources import _read_each
from .. import env as _env


def default_cache_dir():
    d = _env.get("WEIGHTFERRY_DISK_DIR")
    if d:
        return d
    try:
        import folder_paths
        return os.path.join(folder_paths.get_temp_directory(), "weightferry_cache")
    except (ImportError, AttributeError):  # outside ComfyUI: the system temp folder
        import tempfile
        return os.path.join(tempfile.gettempdir(), "weightferry_cache")


# ---------------------------------------------------------------------------------------------------------------------
# SSD block store + background ring reader
# ---------------------------------------------------------------------------------------------------------------------
# Cache files of this process (deleted at exit) and stale-file cleanup. ComfyUI v0.38 no longer empties its temp
# directory, so a cache file left by a process that ended stays forever.
# File names carry the creating process id; files of processes that no longer run are removed when a new store is
# created.
_LIVE_CACHE_FILES = set()
_CLEANED_DIRS = set()
_CACHE_FILE_RE = None


def _delete_live_cache_files():
    # Windows cannot delete a file that is still open: close the SSD readers first (observed: a text encoder's
    # 6.5 GB cache file stayed after its process ended, its ring readers still had it open)
    for reader in list(_OPEN_READERS):
        try:
            reader.close()
        except Exception:  # noqa: BLE001 -- at exit: close what can be closed, then delete
            pass
    for path in list(_LIVE_CACHE_FILES):
        try:
            os.remove(path)
        except OSError:
            pass
        _LIVE_CACHE_FILES.discard(path)


import atexit as _atexit  # noqa: E402
_atexit.register(_delete_live_cache_files)


def new_cache_path(directory, kind):
    """Path for a new cache file ``dit_<kind>_<pid>_<uuid>.bin`` in ``directory``."""
    import uuid
    return os.path.join(directory, f"dit_{kind}_{os.getpid()}_{uuid.uuid4().hex}.bin")


def cleanup_stale_cache_files(directory):
    """Delete cache files of processes that no longer run. Returns the deleted bytes."""
    global _CACHE_FILE_RE
    import re
    if _CACHE_FILE_RE is None:
        _CACHE_FILE_RE = re.compile(r"^dit_(?:blocks|resident)_(\d+)_[0-9a-f]{32}\.bin$")
    try:
        import psutil
        pid_exists = psutil.pid_exists
    except ImportError:  # pragma: no cover -- without psutil no clean-up
        return 0
    deleted = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    for name in names:
        m = _CACHE_FILE_RE.match(name)
        if m is None:
            continue
        path = os.path.join(directory, name)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        pid = int(m.group(1))
        if pid == os.getpid() or pid_exists(pid):
            continue
        try:
            os.remove(path)
            deleted += size
        except OSError:
            pass
    return deleted


class DiskBlockStore:
    """Fixed-stride block file: block i lives at offset i * stride (stride = block bytes rounded up to 4 KiB).

    Read interface shared with FileBlockSource: ``buf_bytes`` (aligned read buffer size) and ``read_spec(i)``
    -> (aligned file offset, aligned length, offset of the block inside the buffer, block bytes)."""

    is_model_file = False

    def __init__(self, path, block_bytes, n_blocks):
        self.path = path
        self.block_bytes = block_bytes
        self.stride = align_up(block_bytes)
        self.buf_bytes = self.stride
        self.n_blocks = n_blocks
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if directory not in _CLEANED_DIRS:
            _CLEANED_DIRS.add(directory)
            self.cleanup_report = {"freed_bytes": cleanup_stale_cache_files(directory)}
        else:
            self.cleanup_report = None
        _LIVE_CACHE_FILES.add(path)
        # Created empty, not pre-sized: on Windows f.truncate() to a larger size writes zeros for the whole length
        # (MSVC _chsize), measured: 20-50 s for a 9.4 GB file. Blocks are written in increasing order
        # (and a gap would be zero-filled by the file system anyway); reads tolerate the short last stride.
        with open(path, "wb"):
            pass
        self._stage = None    # aligned stride-sized buffer for unbuffered writes (Windows)
        self._writer = None

    def staging(self):
        """A block_bytes CPU buffer to fill and pass to write_block. On Windows it is the first part of an aligned
        stride-sized buffer, so the block is written unbuffered without a copy."""
        if not _IS_WINDOWS:
            return torch.empty(self.block_bytes, dtype=torch.uint8, device="cpu")
        if self._stage is None:
            self._stage, _ = aligned_host(self.stride, pin=False)
        return self._stage[:self.block_bytes]

    def write_block(self, i, cpu_uint8):
        """Write one block (uint8 CPU tensor of block_bytes) at its slot."""
        assert cpu_uint8.numel() == self.block_bytes
        if _IS_WINDOWS:
            if self._stage is None:
                self._stage, _ = aligned_host(self.stride, pin=False)
            if cpu_uint8.data_ptr() != self._stage.data_ptr():
                self._stage[:self.block_bytes].copy_(cpu_uint8)
            if self._writer is None:
                self._writer = _UnbufferedWriter(self.path)
            self._writer.write_from(i * self.stride, self.stride, self._stage.data_ptr())
            return
        with open(self.path, "r+b") as f:
            f.seek(i * self.stride)
            f.write(memoryview(cpu_uint8.numpy()))
            # Flush every block: otherwise up to the whole file stays in RAM as dirty pages until the end,
            # which is exactly the host memory the SSD tier is meant to save
            f.flush()
            os.fsync(f.fileno())

    def finalize(self):
        if self._writer is not None:
            self._writer.close()  # readers open the file with read sharing only
            self._writer = None
        self._stage = None
        with open(self.path, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())

    def read_spec(self, i):
        return i * self.stride, self.stride, 0, self.block_bytes

    def read_each(self, indices, pin=True):
        return _read_each(self, indices, pin)

    def delete(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        try:
            os.remove(self.path)
        except OSError:
            pass
        _LIVE_CACHE_FILES.discard(self.path)


class PersistentBlockStore(DiskBlockStore):
    """DiskBlockStore whose file outlives the process: not deleted at exit or by delete(); the size cap
    of the persistent directory removes old entries. existing=True opens a complete file for reading only."""

    def __init__(self, path, block_bytes, n_blocks, existing=False):
        if existing:
            self.path = path
            self.block_bytes = block_bytes
            self.stride = align_up(block_bytes)
            self.buf_bytes = self.stride
            self.n_blocks = n_blocks
            self.cleanup_report = None
            self._stage = None
            self._writer = None
        else:
            super().__init__(path, block_bytes, n_blocks)
            _LIVE_CACHE_FILES.discard(path)

    def delete(self):
        try:
            if self._writer is not None:
                self.finalize()
        except Exception:  # noqa: BLE001 -- the entry is incomplete; it has no .json and is never reused
            pass


def comfy_base_dir():
    """ComfyUI's base directory (--base-directory); the directory above custom_nodes outside ComfyUI."""
    try:
        import folder_paths
        return folder_paths.base_path
    except (ImportError, AttributeError):  # outside ComfyUI (scripts, tests)
        here = os.path.abspath(__file__)
        for _ in range(5):  # core/tiers/cache_files.py -> <package> -> custom_nodes -> base
            here = os.path.dirname(here)
        return here


def persistent_root():
    """Where caches that must survive a restart live: WEIGHTFERRY_DISK_DIR when set, else
    <ComfyUI base directory>/.cache/weightferry. Not ComfyUI's temp folder: ComfyUI deletes it at every start."""
    d = _env.get("WEIGHTFERRY_DISK_DIR")
    return d if d else os.path.join(comfy_base_dir(), ".cache", "weightferry")


def persist_dir(disk_dir):
    """The persistent SSD cache's folder. With WEIGHTFERRY_DISK_DIR set it is under that folder (as the run's cache
    files); by default under persistent_root(), because the default disk_dir is in ComfyUI's temp folder."""
    if _env.get("WEIGHTFERRY_DISK_DIR"):
        return os.path.join(disk_dir, "persistent")
    return os.path.join(persistent_root(), "persistent")


def persist_enforce_cap(directory, cap_bytes, keep_key=None):
    """Delete the least recently used persistent entries (dit_persist_<key>.bin + .json) until the directory holds at
    most cap_bytes. Entries without a .json (interrupted writes) count as oldest. Returns the deleted bytes."""
    try:
        names = os.listdir(directory)
    except OSError:
        return 0
    entries = {}
    for name in names:
        if not name.startswith("dit_persist_"):
            continue
        key = name[len("dit_persist_"):].rsplit(".", 1)[0]
        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        e = entries.setdefault(key, {"bytes": 0, "used": 0.0, "paths": []})
        e["bytes"] += st.st_size
        e["paths"].append((path, st.st_size))
        if name.endswith(".json"):
            e["used"] = st.st_mtime
    total = sum(e["bytes"] for e in entries.values())
    deleted = 0
    for key, e in sorted(entries.items(), key=lambda kv: kv[1]["used"]):
        if total <= cap_bytes:
            break
        if key == keep_key:
            continue
        for path, size in e["paths"]:
            try:
                os.remove(path)
            except OSError:  # in use / no permission: still on disk, so not counted (Codex CPU check 2026-10-13)
                continue
            total -= size
            deleted += size
    return deleted
