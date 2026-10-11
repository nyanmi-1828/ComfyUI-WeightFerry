# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Chunk streaming state: weights streamed by layer groups inside a block (_ChunkState)."""

import time

import torch

from ..utils.logging import log_info, log_warn

try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:  # pragma: no cover
    QuantizedTensor = ()

from .common import _flat_layout, _nbytes, _raw
from .state import _StreamState
from ..core import env as _env


class _ChunkState(_StreamState):
    """Sub-block ("chunk") streaming for very small VRAM.

    Block streaming needs (slots + 1) whole blocks on the GPU besides the activations; for Cosmos-Predict2 14B
    (375 MB blocks) that does not fit a 2 GB budget: the process overflows into shared GPU memory (26 s/it instead
    of 5). Here the GPU only holds a few chunks of ``chunk_bytes`` (default 96 MB) plus one spare.

    The chunks follow the order in which the block's layers are really called (registration order differs: the
    Cosmos block calls its adaLN modulation, self-attention, cross-attention and MLP layers in another order than it
    registers them). That order is learned in the first forward pass after the model's first load ("calibration"):
    every layer's weights are copied, just before the layer runs, into a scratch buffer on the compute stream
    (synchronous, little VRAM, slow) and the call order of the first streamed block is recorded. The order is kept
    with the model (``order_cache``), so later loads -- ComfyUI loads the model again for every generation -- start
    chunked at once (Cosmos-Predict2 14B: a calibration forward takes 6.1 s, a chunked one 2.2 s).
    WEIGHTFERRY_CHUNK_RECALIBRATE=1 learns it on every load. From then on each block is cut into
    chunk types (consecutive layers of that order, up to chunk_bytes) and the chunks are prefetched on the prefetch
    stream into ``num_slots`` rotating slots, ``num_slots`` chunks ahead.

    A slot is handed to the next chunk only when every layer of its chunk has run (layer post-hooks), when the
    block's post-hook runs, or, for a block that a caching wrapper skipped, when a later block starts. A layer whose
    chunk could not be prefetched in order (its slot still in use) is copied synchronously into the spare slot.

    The values streamed are the host buffers of the capture (baked effective weights / stored fp8 bytes), the same
    as block streaming, so the results match ComfyUI. Not supported (block streaming is used instead): the storage-
    dtype / per-transfer LoRA path (convert_idx), single-buffer mode, the profiler, background SSD uploads.
    """

    def __init__(self, owner, device, streaming_blocks, host_flats, host_views, rest, host_specs, host_offsets,
                 chunk_bytes, num_slots, quant_bind_info):
        super().__init__(owner, device, streaming_blocks, host_flats, host_views, rest, [], [], True)
        self.chunk_mode = True
        self.host_specs, self.host_offsets = host_specs, host_offsets
        self.chunk_bytes = int(chunk_bytes)
        self.n_chunk_slots = max(2, int(num_slots))
        names = [n for n, _ in streaming_blocks[0].named_parameters()]
        self.leaf_params = {}
        for j, n in enumerate(names):
            self.leaf_params.setdefault(n.rpartition(".")[0], []).append(j)
        self.block_params = [list(blk.parameters()) for blk in streaming_blocks]
        self.leaf_modules = [{leaf: (blk.get_submodule(leaf) if leaf else blk) for leaf in self.leaf_params}
                             for blk in streaming_blocks]
        self.mod_info = {}
        for s, mods in enumerate(self.leaf_modules):
            for leaf, m in mods.items():
                self.mod_info[id(m)] = (s, leaf)
        # Scaled fp8: per block {j: (module, key, layout_cls, GPU layout params, original parameter)}
        self.qinfo = quant_bind_info or [dict() for _ in streaming_blocks]
        self.quant_idx_set = set(j for d in self.qinfo for j in d)
        self.leaf_layout = {}
        max_leaf = 0
        for leaf, js in self.leaf_params.items():
            offs, total = _flat_layout([host_specs[j] for j in js])
            self.leaf_layout[leaf] = (js, offs)
            max_leaf = max(max_leaf, total)
        self.scratch = torch.empty(max_leaf, dtype=torch.uint8, device=device)
        self.calibrating = True
        self.order = []                    # leaf names in first-call order (first streamed block)
        self.types = None                  # chunk types: [[leaf names]]
        self.chunk_slots = []
        self.slot_type_views = None        # [slot][type] -> [(j, view)]
        self.type_of_leaf = {}
        self._buffers = [self.scratch]
        self._calib_ring = {}              # calibration: block -> ring slot (SSD tier)
        self.spare_warned = False
        self.order_cache = None            # dict kept with the model: {"order": learned order} (set by the loader)
        self._hold = 0                     # > 0 while an MLP runs in row chunks (its layers run once per chunk)
        self._held = set()
        self.begin_forward()

    # ---- helpers
    def _src_param(self, s, j, block_bytes):
        """Host bytes of parameter j of block s: its pinned RAM view, or a view into the SSD ring buffer."""
        if block_bytes is None:
            return self.host_views[s][j]
        shape, dtype = self.host_specs[j]
        off = self.host_offsets[j]
        return block_bytes[off:off + _nbytes(shape, dtype)].view(dtype).view(shape)

    def _block_source(self, s):
        """(SSD ring buffer of block s or None, ring slot or None). Waits for the SSD read if needed."""
        if self.ring is not None and self.disk_index[s] >= 0:
            return self.ring.acquire(self.disk_index[s])
        return None, None

    def _bind(self, s, pairs):
        """Bind parameters j of block s to GPU views: [(j, view)]."""
        for j, view in pairs:
            q = self.qinfo[s].get(j)
            if q is None:
                self.block_params[s][j].data = view
            else:
                module, key, layout_cls, params, _orig = q
                module._parameters[key] = torch.nn.Parameter(QuantizedTensor(view, layout_cls, params),
                                                             requires_grad=False)

    def _in_my_buffers(self, ptr):
        for b in self._buffers:
            start = b.data_ptr()
            if start <= ptr < start + b.numel():
                return True
        return False

    # ---- calibration (first forward after the load)
    def calib_leaf_pre(self, module):
        s, leaf = self.mod_info[id(module)]
        if s == 0 and leaf not in self.order:
            self.order.append(leaf)
        buf = None
        if self.ring is not None and self.disk_index[s] >= 0:
            if s not in self._calib_ring:
                self._calib_ring[s] = self._block_source(s)
            buf = self._calib_ring[s][0]
        js, offs = self.leaf_layout[leaf]
        pairs = []
        for j, off in zip(js, offs):
            shape, dtype = self.host_specs[j]
            dst = self.scratch[off:off + _nbytes(shape, dtype)].view(dtype).view(shape)
            dst.copy_(self._src_param(s, j, buf), non_blocking=True)  # compute stream: ordered after the last use
            pairs.append((j, dst))
        self._bind(s, pairs)

    def _release_ring(self, ring_slot, stream):
        """Give an SSD ring buffer back after the copies queued on ``stream``; submit at once (WDDM batching: the
        ring's reader thread waits on this event while the CPU may wait on the reader)."""
        self.ring.release(ring_slot, stream)
        ev = self.ring.consumed[ring_slot]
        if ev is not None:
            ev.query()

    def calib_block_post(self, s):
        ring = self._calib_ring.pop(s, None)
        if ring is not None:
            self._release_ring(ring[1], torch.cuda.current_stream(self.device))

    def use_learned_order(self, order_cache):
        """Keep the learned order in ``order_cache`` (a dict kept with the model); when it already holds one for these
        layers, skip the calibration forward: chunk types and slots now, the first forward is chunked."""
        self.order_cache = order_cache
        order = order_cache.get("order")
        if (not order or _env.get("WEIGHTFERRY_CHUNK_RECALIBRATE") == "1"
                or not set(order) <= set(self.leaf_params)):
            return False
        self.order = list(order)
        self.finish_calibration()
        self.begin_forward()
        log_info("DiT Layer Streaming [chunks]: call order of an earlier load reused (no calibration pass).")
        return True

    def finish_calibration(self):
        """Chunk types from the learned call order, then the chunk slots."""
        if self.order_cache is not None and self.order:
            self.order_cache["order"] = list(self.order)
        order = list(self.order) + [leaf for leaf in self.leaf_params if leaf not in self.order]
        types, cur, cur_bytes = [], [], 0
        for leaf in order:
            js, _ = self.leaf_layout[leaf]
            nb = _flat_layout([self.host_specs[j] for j in js])[1]
            if cur and cur_bytes + nb > self.chunk_bytes:
                types.append(cur)
                cur, cur_bytes = [], 0
            cur.append(leaf)
            cur_bytes += nb
        if cur:
            types.append(cur)
        self.types = types
        self.type_of_leaf = {leaf: t for t, leaves in enumerate(types) for leaf in leaves}
        layouts = []
        max_bytes = 0
        for leaves in types:
            js = [j for leaf in leaves for j in self.leaf_params[leaf]]
            offs, total = _flat_layout([self.host_specs[j] for j in js])
            layouts.append((js, offs))
            max_bytes = max(max_bytes, total)
        self.type_layouts = layouts
        # num_slots regular slots + 1 spare (out-of-order chunks)
        self.chunk_slots = [torch.empty(max_bytes, dtype=torch.uint8, device=self.device)
                            for _ in range(self.n_chunk_slots + 1)]
        self.slot_type_views = []
        for buf in self.chunk_slots:
            per_type = []
            for js, offs in layouts:
                pairs = []
                for j, off in zip(js, offs):
                    shape, dtype = self.host_specs[j]
                    pairs.append((j, buf[off:off + _nbytes(shape, dtype)].view(dtype).view(shape)))
                per_type.append(pairs)
            self.slot_type_views.append(per_type)
        # The calibration scratch buffer (largest layer, ~100 MB for Cosmos-Predict2 14B) is not needed any more; the
        # parameters bound to it were put back by restore_rest_binding before this runs
        self.scratch = None
        self._buffers = list(self.chunk_slots)
        self.slot_free_events = [torch.cuda.Event() for _ in range(self.n_chunk_slots + 1)]
        # The slots were allocated on the compute stream and may reuse memory it freed during this (calibration)
        # forward while its kernels are still queued (the CPU runs ahead). The prefetch stream writes into them, so
        # its first copy into each slot waits for the compute stream's work up to here. Without this wait the
        # prefetch could overwrite memory still read by queued kernels (corrupted images in chunk mode).
        compute = torch.cuda.current_stream(self.device)
        for ev in self.slot_free_events:
            ev.record(compute)
        self.n_types = len(types)
        self.n_units = self.num_stream * self.n_types
        self.unit_events = [torch.cuda.Event() for _ in range(self.n_units)]
        self.calibrating = False
        log_info(f"DiT Layer Streaming [chunks]: call order learned; {self.n_types} chunks per block "
                 f"(max {max_bytes / 1024 ** 2:.0f} MB), {self.n_chunk_slots} slots + 1 spare = "
                 f"{(self.n_chunk_slots + 1) * max_bytes / 1024 ** 2:.0f} MB of VRAM for the streamed weights.")

    # ---- streaming forwards
    def begin_forward(self):
        self.issued = [False] * self.num_stream  # (base class bookkeeping, unused here)
        # Per-forward timing log (WEIGHTFERRY_CHUNK_LOG=1): wall time incl. GPU, SSD wait, spare-slot uses
        self._fwd_t0 = time.perf_counter()
        self._fwd_ring0 = self.ring.wait_s if self.ring is not None else 0.0
        self._fwd_spares = 0
        self._fwd_calib = self.calibrating
        if self.types is None:
            return
        # Per-forward GPU timeline (WEIGHTFERRY_CHUNK_PROFILE=1): per chunk the copy start / end on the prefetch
        # stream and, on the compute stream, the moment it reached the chunk and the moment its copy let it go on
        self._prof = None
        if _env.get("WEIGHTFERRY_CHUNK_PROFILE") == "1":
            ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
            self._prof = {"t0": ev(), "end": ev(), "cs": [None] * self.n_units, "ce": [None] * self.n_units,
                          "pre": [None] * self.n_units, "post": [None] * self.n_units, "bytes": [0] * self.n_units,
                          "ring_wait": [0.0] * self.n_units, "ev": ev}
            self._prof["t0"].record(torch.cuda.current_stream(self.device))
        self.u_issued = [False] * self.n_units
        self.u_bound = [False] * self.n_units
        self.u_complete = [False] * self.n_units
        self.u_spare = [False] * self.n_units
        self.u_done = [set() for _ in range(self.n_units)]
        self.next_issue = 0
        self.ring_held = {}
        self.blk_issued = [0] * self.num_stream
        self.slot_owner = [None] * self.n_chunk_slots
        self.blocks_started = 0
        self._fwd_units = 0      # chunks issued in this forward (WEIGHTFERRY_CHUNK_LOG=1)
        self._fwd_outside = 0    # layer calls outside a block (served from the outside buffer)

    def _issue(self, u, k=None):
        """Copy chunk u into slot k (default: its rotating slot; the spare slot for an out-of-order chunk) on the
        prefetch stream, after the slot's previous chunk finished. The SSD ring buffer of the block is given back
        once all of the block's chunks have been issued (all copies out of it are then queued on this stream)."""
        s, t = divmod(u, self.n_types)
        spare = k is not None
        if k is None:
            k = u % self.n_chunk_slots
        buf = None
        prof = getattr(self, "_prof", None)
        if self.ring is not None and self.disk_index[s] >= 0:
            if s not in self.ring_held:
                w0 = self.ring.wait_s
                self.ring_held[s] = self._block_source(s)
                if prof is not None:
                    prof["ring_wait"][u] = self.ring.wait_s - w0
            buf = self.ring_held[s][0]
        with torch.cuda.stream(self.prefetch_stream):
            self.prefetch_stream.wait_event(self.slot_free_events[k])
            if prof is not None:
                prof["cs"][u] = prof["ev"]()
                prof["cs"][u].record(self.prefetch_stream)
            for j, dst in self.slot_type_views[k][t]:
                dst.copy_(self._src_param(s, j, buf), non_blocking=True)
                if prof is not None:
                    prof["bytes"][u] += dst.numel() * dst.element_size()
            if prof is not None:
                prof["ce"][u] = prof["ev"]()
                prof["ce"][u].record(self.prefetch_stream)
            self.unit_events[u].record(self.prefetch_stream)
            self.blk_issued[s] += 1
            if buf is not None and self.blk_issued[s] == self.n_types:
                self._release_ring(self.ring_held.pop(s)[1], self.prefetch_stream)
        # WDDM batches CUDA commands: submit now, or the CPU may block on the SSD ring while this copy (and the
        # buffer release the ring's reader waits for) is still held back (see issue_prefetch)
        self.unit_events[u].query()
        if spare:
            self.u_spare[u] = True
        else:
            self.slot_owner[k] = u
        self.u_issued[u] = True
        self._fwd_units = getattr(self, "_fwd_units", 0) + 1

    def pump(self):
        """Issue chunks in order while their slot's previous chunk has completed."""
        while self.next_issue < self.n_units:
            u = self.next_issue
            if self.u_issued[u]:  # loaded into the spare slot already
                self.next_issue += 1
                continue
            k = u % self.n_chunk_slots
            owner = self.slot_owner[k]
            if owner is not None and not self.u_complete[owner]:
                return
            self._issue(u)
            self.next_issue += 1

    def _complete(self, u):
        if self.u_complete[u]:
            return
        self.u_complete[u] = True
        stream = torch.cuda.current_stream(self.device)
        ev = None
        if self.u_spare[u]:
            ev = self.slot_free_events[self.n_chunk_slots]
        elif self.u_issued[u]:
            ev = self.slot_free_events[u % self.n_chunk_slots]
        if ev is not None:
            ev.record(stream)
            # Submit the compute stream's work up to here: the prefetch stream waits on this event, and the SSD
            # reader waits on the prefetch stream, while the CPU may block on that reader (without this the
            # forward can hang)
            ev.query()
        self.pump()

    def leaf_pre(self, module):
        if self.calibrating:
            self.calib_leaf_pre(module)
            return
        s, leaf = self.mod_info[id(module)]
        if self.blocks_started == 0 and self._bind_outside(s, leaf):
            return
        u = s * self.n_types + self.type_of_leaf[leaf]
        compute = torch.cuda.current_stream(self.device)
        if not self.u_bound[u]:
            if not self.u_issued[u]:
                self.pump()
            if not self.u_issued[u]:
                # Out of order (its rotating slot still holds an unfinished chunk): copy it into the spare slot now
                if not self.spare_warned:
                    self.spare_warned = True
                    log_info(f"DiT Layer Streaming [chunks]: a layer ran out of the learned order (block {s}, "
                             f"{leaf}); its chunk goes through the spare slot.")
                self._issue(u, self.n_chunk_slots)
                self._fwd_spares += 1
            k = self.n_chunk_slots if self.u_spare[u] else u % self.n_chunk_slots
            prof = getattr(self, "_prof", None)
            if prof is not None:
                prof["pre"][u] = prof["ev"]()
                prof["pre"][u].record(compute)
            compute.wait_event(self.unit_events[u])
            if prof is not None:
                prof["post"][u] = prof["ev"]()
                prof["post"][u].record(compute)
            self._bind(s, self.slot_type_views[k][self.type_of_leaf[leaf]])
            self.u_bound[u] = True

    def hold_completions(self, delta):
        """MLP row chunking: while held, a chunk whose layers have all run is not completed (its layers will run
        again for the next rows); the held chunks complete when the MLP is done."""
        self._hold += delta
        if self._hold == 0 and self._held:
            held, self._held = sorted(self._held), set()
            if self.calibrating:
                return
            for u in held:
                if len(self.u_done[u]) == len(self.types[u % self.n_types]):
                    self._complete(u)

    def _bind_outside(self, s, leaf):
        """A streamed layer called before any block of this forward ran -- TaylorSeer calls the first block's adaLN
        modulation directly on every step, also on the steps it skips. Starting the chunk pipeline there would stream
        chunks that a skipped step never uses. Instead only that layer's weights are copied into a small
        buffer on the compute stream and bound; the pipeline starts with the first block as usual. RAM-tier blocks
        only (an SSD-tier block would take the ring out of order). WEIGHTFERRY_OUTSIDE_LEAF=0 starts the chunk
        pipeline there instead."""
        if _env.get("WEIGHTFERRY_OUTSIDE_LEAF") == "0":
            return False
        if self.ring is not None and self.disk_index[s] >= 0:
            return False
        js, offs = self.leaf_layout[leaf]
        total = _flat_layout([self.host_specs[j] for j in js])[1]
        buf = getattr(self, "_outside_buf", None)
        if buf is None or buf.numel() < total:
            buf = torch.empty(total, dtype=torch.uint8, device=self.device)  # used on the compute stream only
            self._outside_buf = buf
        pairs = []
        for j, off in zip(js, offs):
            shape, dtype = self.host_specs[j]
            dst = buf[off:off + _nbytes(shape, dtype)].view(dtype).view(shape)
            dst.copy_(self.host_views[s][j], non_blocking=True)
            pairs.append((j, dst))
        self._bind(s, pairs)
        self._fwd_outside = getattr(self, "_fwd_outside", 0) + 1
        return True

    def leaf_post(self, module):
        if self.calibrating:
            return
        if self.blocks_started == 0 and _env.get("WEIGHTFERRY_OUTSIDE_LEAF") != "0":
            s0, _leaf0 = self.mod_info[id(module)]
            if not (self.ring is not None and self.disk_index[s0] >= 0):
                return  # served by _bind_outside: no chunk to complete
        s, leaf = self.mod_info[id(module)]
        t = self.type_of_leaf[leaf]
        u = s * self.n_types + t
        self.u_done[u].add(leaf)
        if self._hold > 0:
            self._held.add(u)
            return
        if len(self.u_done[u]) == len(self.types[t]):
            self._complete(u)

    def block_pre(self, s):
        if self.calibrating:
            return
        # Blocks before s that a caching wrapper skipped: their chunks never ran, their slots are free
        for u in range(0, s * self.n_types):
            if not self.u_complete[u] and self.u_issued[u]:
                self._complete(u)
        if self.blocks_started == 0:
            self.pump()
        self.blocks_started += 1

    def block_post(self, s):
        if self.calibrating:
            self.calib_block_post(s)
            return
        for t in range(self.n_types):
            u = s * self.n_types + t
            if self.u_issued[u]:
                self._complete(u)

    def prefetch_first_slots(self):
        if not self.calibrating:
            self.pump()

    def _profile_forward(self):
        """Summary of the GPU timeline of this forward (WEIGHTFERRY_CHUNK_PROFILE=1), logged: total time, compute-stream stalls waiting for chunk copies, copy busy time / rate, how early
        the copies started before they were needed, the CPU's SSD waits."""
        prof = getattr(self, "_prof", None)
        self._prof = None
        if prof is None or self._fwd_calib:
            return
        prof["end"].record(torch.cuda.current_stream(self.device))
        torch.cuda.synchronize(self.device)
        us = [u for u in range(self.n_units) if prof["pre"][u] is not None and prof["ce"][u] is not None]
        total = prof["t0"].elapsed_time(prof["end"])
        stalls = [prof["pre"][u].elapsed_time(prof["post"][u]) for u in us]
        copies = [prof["cs"][u].elapsed_time(prof["ce"][u]) for u in us]
        lead = [prof["cs"][u].elapsed_time(prof["pre"][u]) for u in us]  # > 0: copy started before it was needed
        nbytes = sum(prof["bytes"][u] for u in us)
        # compute-stream time between the end of one chunk's wait and the start of the next one's (work + CPU gaps)
        work = [prof["post"][a].elapsed_time(prof["pre"][b]) for a, b in zip(us, us[1:])]
        copy_s = sum(copies) / 1000
        summary = {"forward_ms": round(total, 1), "units": len(us),
                   "compute_stall_ms": round(sum(stalls), 1), "stalled_units": sum(1 for x in stalls if x > 0.5),
                   "between_waits_ms": round(sum(work), 1),
                   "copy_busy_ms": round(sum(copies), 1), "copy_gb": round(nbytes / 1e9, 2),
                   "copy_gbps": round(nbytes / 1e9 / copy_s, 2) if copy_s else None,
                   "copy_lead_ms_median": round(sorted(lead)[len(lead) // 2], 1) if lead else None,
                   "cpu_ssd_wait_ms": round(sum(prof["ring_wait"]) * 1000, 1)}
        top = sorted(range(len(us)), key=lambda i: -stalls[i])[:3]
        summary["worst_stalls"] = [(divmod(us[i], self.n_types), round(stalls[i], 1)) for i in top]
        log_info(f"DiT Layer Streaming [chunks profile]: {summary}")

    def _log_forward(self):
        self._profile_forward()
        if _env.get("WEIGHTFERRY_CHUNK_LOG") != "1" or not hasattr(self, "_fwd_t0"):
            return
        torch.cuda.synchronize(self.device)
        self._n_fwd = getattr(self, "_n_fwd", 0) + 1
        ring = (self.ring.wait_s if self.ring is not None else 0.0) - self._fwd_ring0
        log_info(f"DiT Layer Streaming [chunks]: forward {self._n_fwd} "
                 f"({'calibration' if self._fwd_calib else 'chunked'}): {(time.perf_counter() - self._fwd_t0) * 1000:.0f} ms, "
                 f"CPU waited {ring * 1000:.0f} ms for SSD reads, {self._fwd_spares} chunks via the spare slot, "
                 f"{getattr(self, '_fwd_units', 0)} chunks issued, {getattr(self, 'blocks_started', 0)} blocks run, "
                 f"{getattr(self, '_fwd_outside', 0)} layer calls outside a block")

    def end_forward_tiers(self):
        self._log_forward()
        if self.calibrating:
            for s in list(self._calib_ring):
                self.calib_block_post(s)
            self.finish_calibration()
        elif self.ring is not None:
            # every chunk of every block is issued in a full forward; an interrupted one restarts the read cycle
            for s, held in list(self.ring_held.items()):
                self._release_ring(held[1], torch.cuda.current_stream(self.device))
            self.ring_held = {}
        if self.ring is not None and not self.ring.at_cycle_start():
            log_warn("DiT Layer Streaming [chunks]: forward pass stopped part way through the SSD tier; "
                     "restarting the SSD read-ahead.")
            self.ring.resync()
        self.begin_forward()

    def restore_rest_binding(self):
        for s in range(self.num_stream):
            for j, q in self.qinfo[s].items():
                module, key, _l, _p, orig = q
                if module._parameters[key] is not orig:
                    module._parameters[key] = orig
            for j, p in enumerate(self.block_params[s]):
                if j in self.qinfo[s]:
                    continue
                if self._in_my_buffers(p.data.data_ptr()):
                    p.data = self.rest[s][j]
        for blk, rest in zip(self.resident_blocks, self.resident_rest):
            for p, r in zip(blk.parameters(), rest):
                if _raw(p.data).data_ptr() in self.resident_ptrs:
                    p.data = r
