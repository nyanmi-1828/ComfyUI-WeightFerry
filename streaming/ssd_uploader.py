# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Background SSD-tier uploader thread of block streaming."""

import queue
import sys
import threading
import time

import torch

from .common import _views
from .activation import _prefetch_stream


class _SsdUploader:
    """Background thread feeding SSD-tier blocks into their GPU slots (double buffer).

    Jobs (block, slot, slot-free event) are submitted in block order when the main thread would have issued the
    prefetch. The thread takes the block's bytes from the SSD ring (waiting for the reader threads if needed),
    waits on its own CUDA stream for the slot to be free, copies, gives the ring buffer back and records the
    block's transferred event. ``wait(s)`` (block s's pre-hook) returns once that event has been recorded.
    """

    def __init__(self, state):
        self.state = state
        self.device = state.device
        self.stream = _prefetch_stream(state.device)
        self.ready = [threading.Event() for _ in range(state.num_stream)]
        self.main_wait_s = 0.0
        self.error = None
        # Staging path (storage format or LoRA on the GPU): an own staging buffer, so the prefetch stream's shared
        # one (ordered on that stream) is never touched from here
        self.staging = self.staging_views = None
        if state.staging_flat is not None:
            self.staging = torch.empty_like(state.staging_flat)
            self.staging_views = _views(self.staging, state.host_specs, state.host_offsets)
        self._q = queue.Queue()
        # The thread copies into slots created under ComfyUI's torch.inference_mode (inference tensors); threads do
        # not inherit the mode, so it is taken over explicitly (same bug as the streamed LLM's copy threads)
        self.inference = torch.is_inference_mode_enabled()
        self._thread = threading.Thread(target=self._run, daemon=True, name="weightferry-ssd-upload")
        self._thread.start()

    def submit(self, s_idx, slot_idx):
        # Fresh event for "the slot's last reader finished": computed_events[slot] is re-recorded by later blocks,
        # and waiting on a later recording would make this copy wait for compute that needs the copy (deadlock)
        free_ev = torch.cuda.Event()
        free_ev.record(torch.cuda.current_stream(self.device))
        self.ready[s_idx].clear()
        self._q.put((s_idx, slot_idx, free_ev))

    def _run(self):
        with torch.inference_mode(self.inference):
            self._loop()

    def _loop(self):
        torch.cuda.set_device(self.device)
        st = self.state
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            s_idx, slot_idx, free_ev = item
            try:
                if self.error is None:
                    src, ring_slot = st.ring.acquire(st.disk_index[s_idx])
                    prof = st.profiler
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(free_ev)
                        if prof is not None:
                            prof.record(s_idx, "dma0", self.stream)
                        if self.staging is None:
                            st.slot_flats[slot_idx].copy_(src, non_blocking=True)
                            st.ring.release(ring_slot, self.stream)
                            if prof is not None:
                                prof.record(s_idx, "dma1", self.stream)
                                prof.record(s_idx, "ready", self.stream)
                        else:
                            self.staging.copy_(src, non_blocking=True)
                            st.ring.release(ring_slot, self.stream)
                            if prof is not None:
                                prof.record(s_idx, "dma1", self.stream)
                            for j, (dst, stg) in enumerate(zip(st.slot_views[slot_idx], self.staging_views)):
                                if j in st.convert_idx:
                                    st.convert_param(s_idx, j, dst, stg)
                                else:
                                    dst.copy_(stg)
                            if prof is not None:
                                prof.record(s_idx, "ready", self.stream)
                        st.transferred_events[s_idx].record(self.stream)
                    st.transferred_events[s_idx].query()  # WDDM: submit now (see issue_prefetch)
            except Exception as exc:  # noqa: BLE001 -- surfaced on the main thread in wait()
                self.error = exc
            finally:
                self.ready[s_idx].set()
                self._q.task_done()

    def wait(self, s_idx):
        if not self.ready[s_idx].is_set():
            t0 = time.perf_counter()
            self.ready[s_idx].wait()
            self.main_wait_s += time.perf_counter() - t0
        if self.error is not None:
            raise RuntimeError(f"DiT Layer Streaming: SSD upload failed: {self.error}") from self.error

    def drain(self):
        if sys.is_finalizing():
            return  # see close()
        self._q.join()

    def close(self):
        self._q.put(None)
        # At interpreter exit (e.g. ModelPatcher.__del__ of a script that never unloaded the model) daemon threads
        # cannot run any more, so join() would hang at interpreter exit
        if sys.is_finalizing():
            return
        self._thread.join(timeout=60)
