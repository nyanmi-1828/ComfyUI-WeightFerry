# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Block sources read by the SSD tier (a cache file, or the model file itself) and the reader ring (DiskRing).
"""

import sys
import queue
import threading
import time

import torch

from .common import _ALIGN_IO, align_up
from .pinned import release_all
from .disk_io import _UnbufferedReader, aligned_host


def _read_each(store, indices, pin=True, ahead=False):
    """Synchronously read blocks one by one into a single reused 4 KiB aligned (pinned) buffer.
    Yields (i, uint8 view of the block); the view is only valid until the next iteration, so consume it
    synchronously (e.g. a blocking copy to the GPU). ahead=True: a thread reads the next block into a second
    buffer while the caller works on the current one (same views and contract, one more buffer)."""
    if ahead:
        yield from _read_each_ahead(store, list(indices), pin)
        return
    buf, holder = aligned_host(store.buf_bytes, pin=pin)
    reader = _UnbufferedReader(store.path)
    try:
        for i in indices:
            off, length, voff, n = store.read_spec(i)
            reader.read_into(off, length, buf.data_ptr(), min_bytes=voff + n)
            yield i, buf[voff:voff + n]
    finally:
        reader.close()
        if holder is not None:
            holder.release()


def _read_each_ahead(store, indices, pin):
    """_read_each with read-ahead: two buffers; the reader thread fills one while the caller uses the other. A buffer
    goes back to the reader when the caller asks for the next block."""
    bufs = [aligned_host(store.buf_bytes, pin=pin) for _ in range(2)]
    free, full = queue.Queue(), queue.Queue()
    for b in range(len(bufs)):
        free.put(b)

    def work():
        reader = None
        try:
            reader = _UnbufferedReader(store.path)
            for i in indices:
                b = free.get()
                if b is None:
                    return
                off, length, voff, n = store.read_spec(i)
                reader.read_into(off, length, bufs[b][0].data_ptr(), min_bytes=voff + n)
                full.put((b, i, voff, n))
        except BaseException as exc:  # noqa: BLE001 -- handed to the caller
            full.put(exc)
        finally:
            if reader is not None:
                reader.close()

    thread = threading.Thread(target=work, daemon=True, name="weightferry-read-ahead")
    thread.start()
    prev = None
    try:
        for _ in indices:
            item = full.get()
            if isinstance(item, BaseException):
                raise item
            if prev is not None:
                free.put(prev)
            b, i, voff, n = item
            prev = b
            yield i, bufs[b][0][voff:voff + n]
    finally:
        free.put(None)
        free.put(None)
        thread.join()
        for _buf, holder in bufs:
            if holder is not None:
                holder.release()


class FileBlockSource:
    """Blocks read straight from the model's own file: block i is the byte range ranges[i] = (absolute offset,
    size). Reads cover the 4 KiB aligned superset of the range (unbuffered I/O); the block starts ``voff`` bytes
    into the read buffer. Never deletes the file."""

    is_model_file = True

    def __init__(self, path, ranges):
        self.path = path
        self.ranges = [(int(o), int(n)) for o, n in ranges]
        self.n_blocks = len(self.ranges)
        self.block_bytes = max(n for _, n in self.ranges)
        self.buf_bytes = max(align_up(o + n) - (o // _ALIGN_IO * _ALIGN_IO) for o, n in self.ranges)

    def read_spec(self, i):
        o, n = self.ranges[i]
        start = o // _ALIGN_IO * _ALIGN_IO
        return start, align_up(o + n) - start, o - start, n

    def read_each(self, indices, pin=True, ahead=False):
        return _read_each(self, indices, pin, ahead)

    def delete(self):
        pass  # the model file is not ours


class PieceFileSource:
    """Blocks read straight from the model's own file when each block is several ranges (file_param_layout).
    ``pieces[i]``: list of (absolute offset, bytes, offset in the packed block). Each piece is read as its 4 KiB
    aligned superset into a scratch buffer and copied to its place in the destination. Never deletes the file."""

    is_model_file = True

    def __init__(self, path, pieces):
        self.path = path
        self.pieces = [[(int(o), int(n), int(d)) for o, n, d in blk] for blk in pieces]
        self.n_blocks = len(self.pieces)
        self.block_bytes = max(max(d + n for _, n, d in blk) for blk in self.pieces)
        self.buf_bytes = align_up(self.block_bytes)
        self.scratch_bytes = max(align_up(o + n) - (o // _ALIGN_IO * _ALIGN_IO) for blk in self.pieces
                                 for o, n, _d in blk)

    def read_block(self, reader, i, dst_ptr, scratch_ptr):
        import ctypes
        for o, n, d in self.pieces[i]:
            start = o // _ALIGN_IO * _ALIGN_IO
            reader.read_into(start, align_up(o + n) - start, scratch_ptr, min_bytes=o - start + n)
            ctypes.memmove(dst_ptr + d, scratch_ptr + (o - start), n)
        return max(d + n for _, n, d in self.pieces[i])

    def read_each(self, indices, pin=True):
        buf, holder = aligned_host(self.buf_bytes, pin=pin)
        scratch, s_holder = aligned_host(self.scratch_bytes, pin=False)
        reader = _UnbufferedReader(self.path)
        try:
            for i in indices:
                n = self.read_block(reader, i, buf.data_ptr(), scratch.data_ptr())
                yield i, buf[:n]
        finally:
            reader.close()
            for h in (holder, s_holder):
                if h is not None:
                    h.release()

    def delete(self):
        pass  # the model file is not ours


def file_source(path, ranges):
    """FileBlockSource for one range per block, PieceFileSource for lists of pieces (file_param_layout)."""
    if ranges and isinstance(ranges[0], list):
        return PieceFileSource(path, ranges)
    return FileBlockSource(path, ranges)


class DiskRing:
    """Background reader feeding SSD-tier blocks, in cyclic block order, into ``depth`` pinned ring buffers.

    Forward passes consume the SSD-tier blocks in order 0..n-1 (all of them, or none for a step skipped by a
    caching wrapper), so reads simply continue cyclically into the next forward pass. ``acquire(d)`` waits for
    block d's bytes; ``release(slot, stream)`` records when the DMA out of that buffer finished and queues the next
    read into it. ``resync()`` restarts the cycle if a forward pass stopped half way (e.g. an exception).
    """

    def __init__(self, store, depth, pin=True, threads=1):
        self.store = store
        self.n = store.n_blocks
        self.depth = depth
        allocated = [aligned_host(store.buf_bytes, pin=pin) for _ in range(depth)]
        self.bufs = [t for t, _ in allocated]
        self._holders = [h for _, h in allocated]
        self.slot_view = [(0, store.block_bytes)] * depth  # (offset in buffer, bytes) of the block read last
        self.ready = [threading.Event() for _ in range(depth)]
        self.slot_pos = [-1] * depth
        self.consumed = [None] * depth
        self.req = 0
        self.cons = 0
        self.wait_s = 0.0
        # Several reads in flight (one whole block per reader thread, each with its own file handle)
        self.threads = max(1, min(int(threads), depth))
        self._start_readers()
        self.prime()

    def _start_readers(self):
        self._q = queue.Queue()
        self._readers = [_UnbufferedReader(self.store.path) for _ in range(self.threads)]
        self._workers = [threading.Thread(target=self._run, args=(r,), daemon=True, name=f"weightferry-ssd-ring-{i}")
                         for i, r in enumerate(self._readers)]
        for w in self._workers:
            w.start()

    def _stop_readers(self):
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            # at interpreter exit daemon threads cannot run any more: join() would wait forever
            if not sys.is_finalizing():
                w.join(timeout=60)
        for r in self._readers:
            r.close()
        self._workers, self._readers = [], []

    def _run(self, reader):
        scratch = s_holder = None
        if hasattr(self.store, "read_block"):
            scratch, s_holder = aligned_host(self.store.scratch_bytes, pin=False)
        try:
            self._loop(reader, scratch)
        except OSError:
            # the process is exiting: _delete_live_cache_files closed this reader's handle under it
            if getattr(reader, "handle", None) is None and getattr(reader, "f", None) is None:
                return
            raise
        finally:
            if s_holder is not None:
                s_holder.release()

    def _loop(self, reader, scratch):
        while True:
            item = self._q.get()
            if item is None:
                return
            pos, slot = item
            ev = self.consumed[slot]
            if ev is not None:
                ev.synchronize()  # the previous DMA out of this buffer must be done
            d = pos % self.n
            t_read = time.perf_counter()
            if scratch is not None:  # PieceFileSource: several ranges per block
                n = self.store.read_block(reader, d, self.bufs[slot].data_ptr(), scratch.data_ptr())
                self.slot_view[slot] = (0, n)
            else:
                off, length, voff, n = self.store.read_spec(d)
                reader.read_into(off, length, self.bufs[slot].data_ptr(), min_bytes=voff + n)
                self.slot_view[slot] = (voff, n)
            dt = time.perf_counter() - t_read
            # Reads far slower than the drive's rate (e.g. a QLC SSD that stalls for 10-30 s now and then):
            # log them with the block and size so stalls can be told apart from a slow region of the file
            if dt > 1.0:
                print(f"[tiers] slow SSD read: block {d} ({self.slot_view[slot][1] / 2 ** 20:.0f} MB) took {dt:.1f} s",
                      flush=True)
            self.slot_pos[slot] = pos
            self.ready[slot].set()

    def _request_next(self):
        pos = self.req
        slot = pos % self.depth
        self.ready[slot].clear()
        self.req += 1
        self._q.put((pos, slot))

    def prime(self):
        while self.req - self.cons < self.depth:
            self._request_next()

    def acquire(self, d):
        """Pinned view holding SSD-tier block d (waits for the reader). Returns (view, slot)."""
        if self.cons % self.n != d:
            raise RuntimeError(f"SSD ring out of sync: expected block {self.cons % self.n}, got {d}")
        slot = self.cons % self.depth
        if not self.ready[slot].is_set():
            t0 = time.perf_counter()
            # A wait of minutes is either a very slow drive (e.g. right after large writes) or a stuck pipeline:
            # say which block is awaited and what the readers hold, every 60 s (without this a stall could not be
            # told apart from a slow SSD afterwards)
            while not self.ready[slot].wait(60):
                pending = [(i, self.slot_pos[i], self.ready[i].is_set(),
                            None if self.consumed[i] is None else self.consumed[i].query())
                           for i in range(self.depth)]
                print(f"[tiers] still waiting for SSD block {d} after {time.perf_counter() - t0:.0f} s "
                      f"(requested {self.req}, consumed {self.cons}, queue {self._q.qsize()}, "
                      f"slots (slot, block pos, read, DMA done) {pending})", flush=True)
            self.wait_s += time.perf_counter() - t0
        assert self.slot_pos[slot] == self.cons
        voff, n = self.slot_view[slot]
        return self.bufs[slot][voff:voff + n], slot

    def release(self, slot, stream):
        ev = torch.cuda.Event()
        ev.record(stream)
        self.consumed[slot] = ev
        self.cons += 1
        self._request_next()

    def at_cycle_start(self):
        return self.cons % self.n == 0

    def close(self, release_buffers=True):
        self._stop_readers()
        if release_buffers:
            for ev in self.consumed:
                if ev is not None:
                    ev.synchronize()
            release_all(self._holders)
            self._holders = []

    def resync(self):
        """Drain pending reads and restart at block 0 (after a forward pass consumed only part of the cycle)."""
        self.close(release_buffers=False)
        for ev in self.consumed:
            if ev is not None:
                ev.synchronize()
        self.consumed = [None] * self.depth
        self.ready = [threading.Event() for _ in range(self.depth)]
        self.slot_pos = [-1] * self.depth
        self.req = self.cons = 0
        self._start_readers()
        self.prime()
