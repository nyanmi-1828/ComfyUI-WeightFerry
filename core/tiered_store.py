# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Tiered weight store for DiT layer streaming: SSD file -> pinned RAM ring -> GPU slots.

Streaming blocks are split into
  * a RAM tier: one pinned host buffer per block (fast, costs host RAM), and
  * an SSD tier: the block's final compute bytes are written once to a cache file on the SSD and streamed back
    through a small ring of pinned buffers by a background reader thread (costs only ring_depth x block bytes of RAM).
The split and the ring depth are planned from the measured resources (free RAM, SSD read bandwidth, per-block
compute time) by ``plan_tiers``.

Disk reads bypass the OS file cache on Windows (FILE_FLAG_NO_BUFFERING): a cached read would fill the standby list
with the same bytes we are trying to keep out of RAM. Buffers and offsets are 4 KiB aligned for that reason.

Direct file source (FileBlockSource): when a block's bytes in the model's own safetensors file are exactly what
the GPU needs (no LoRA / patches, same dtype, the block's tensors contiguous in the file), the SSD tier reads the
blocks straight from the model file -- no cache file is written and nothing is captured for those blocks.

NVMe drives usually read faster with several requests in flight, so the ring uses several reader threads (each
reads whole blocks through its own file handle). The probe measures 1 / 2 / 4 threads and the planner picks the
fastest (override: WEIGHTFERRY_DISK_THREADS).
"""

# The implementation lives in the package tiers/ (split by concern). This module re-exports the names used
# outside it.
from .tiers.common import _ALIGN_IO, align_up  # noqa: F401
from .tiers.pinned import (  # noqa: F401
    PinnedHost, _REGISTERED, empty_host_cache, mem_snapshot, release_all,
)
from .tiers.disk_io import _UnbufferedReader, aligned_host  # noqa: F401
from .tiers.plan import (  # noqa: F401
    _ST_DTYPES, available_ram, file_block_layout, file_param_layout, plan_tiers, ssd_positions,
)
from .tiers.cache_files import (  # noqa: F401
    DiskBlockStore, PersistentBlockStore, _LIVE_CACHE_FILES, default_cache_dir, new_cache_path, persist_dir,
    persist_enforce_cap,
)
from .tiers.sources import DiskRing, FileBlockSource, file_source  # noqa: F401
