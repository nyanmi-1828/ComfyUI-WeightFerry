# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Block streaming state: GPU slots, prefetch, tiers, overflow checks (_StreamState)."""

import torch

from ..utils.logging import log_warn
from ..core import vram_monitor

try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:  # pragma: no cover
    QuantizedTensor = ()

from .common import _blocks, _copy_quant_params_, _quant_lora_requant, _raw, _views
from .activation import _prefetch_stream
from ..core import env as _env


class _StreamState:
    """Per-diffusion_model streaming state, owned by a single ModelPatcher (owner token)."""

    def __init__(self, owner, device, streaming_blocks, host_flats, host_views, rest, slot_flats, slot_views,
                 is_double, convert_idx=(), lora_fns=None, compute_dtype=None, staging_flat=None,
                 staging_views=None):
        self.owner = owner
        self.device = device
        self.streaming_blocks = streaming_blocks
        self.host_flats = host_flats      # per block: one flat (pinned) uint8 buffer = DMA source
        self.host_views = host_views      # per block: parameter views into host_flats
        self.rest = rest                  # per block: tensors params are bound to between forwards
        self.slot_flats = slot_flats      # per slot: one flat GPU uint8 buffer
        self.slot_views = slot_views      # per slot: parameter views into slot_flats
        self.is_double = is_double
        # Storage-dtype path: DMA into one staging buffer (host layout), then dequantize (+LoRA) into the slot
        self.convert_idx = set(convert_idx)
        self.lora_fns = lora_fns          # per block: per parameter LoRA function or None
        self.compute_dtype = compute_dtype
        self.staging_flat = staging_flat
        self.staging_views = staging_views
        self.no_staging = staging_flat is None  # DMA straight into the slot (no staging copy / conversion)
        self.num_stream = len(streaming_blocks)
        self.num_slots = len(slot_flats)
        self.handles = []

        self.prefetch_stream = _prefetch_stream(device)
        self.transferred_events = [torch.cuda.Event() for _ in range(self.num_stream)]
        self.computed_events = [torch.cuda.Event() for _ in range(self.num_slots)]
        # The slots were allocated on the compute stream; memory it freed may still be in use by queued kernels. The
        # first prefetch into each slot waits for the compute stream's work up to here (same race as the chunk
        # slots; for block slots it was not seen, the capture ends with blocking copies).
        compute = torch.cuda.current_stream(device)
        for ev in self.computed_events:
            ev.record(compute)
        # Whether block s has had its prefetch issued in the current forward pass
        self.issued = [False] * self.num_stream
        self.slot_ptrs = {v.data_ptr() for views in slot_views for v in views}
        self.profiler = None
        # Tiering: blocks with disk_index >= 0 are served by the SSD ring (TIERING_AUTO)
        self.ring = None
        self.uploader = None              # _SsdUploader when SSD-tier uploads run in the background
        self.quant_bind = None            # scaled fp8: per streaming block [(module, key, GPU QuantizedTensor param)]
        self.quant_rest = None            # scaled fp8: per streaming block [(module, key, original param)]
        self.quant_idx_set = set()
        self.quant_conv = None            # scaled fp8 + LoRA on the GPU: per block {param index: (layout, file params,
        #                                   bound params, seed)}
        self.direct_file = False          # SSD tier reads the model's own safetensors file
        self.ram_blocks = len(streaming_blocks)
        self.disk_index = [-1] * len(streaming_blocks)  # per streaming block: position in the SSD store, -1 = RAM
        self.tier_plan = None
        # Light per-forward timing for the tier planner: GPU time from block 0 start to the last block's end
        self.span_start = None
        self.span_end = None
        self.pending_spans = []       # finished forwards whose GPU work may still be running
        self.forwards_done = 0
        self.overflow_peak = 0            # bytes of this process' GPU memory found in shared GPU memory (system RAM)
        self.overflow_warned = False
        self.resident_disk_blocks = 0
        # Self-managed resident blocks (RESIDENT_MODE_SELF): GPU buffers bound for the whole forward call
        self.resident_blocks = []
        self.resident_gpu_flats = []
        self.resident_views = []
        self.resident_rest = []
        self.resident_ptrs = set()
        # Scaled fp8 in self-managed resident blocks: per block [(module, key, GPU QuantizedTensor param)] and the
        # originals to put back
        self.res_quant_bind = None
        self.res_quant_rest = None

    def set_resident(self, blocks, gpu_flats, views, rest):
        self.resident_blocks = blocks
        self.resident_gpu_flats = gpu_flats
        self.resident_views = views
        self.resident_rest = rest
        self.resident_ptrs = {v.data_ptr() for vs in views for v in vs}

    def bind_resident(self):
        for r_i, (blk, views) in enumerate(zip(self.resident_blocks, self.resident_views)):
            for j, (p, v) in enumerate(zip(blk.parameters(), views)):
                if j not in self.quant_idx_set:
                    p.data = v
            if self.res_quant_bind is not None:
                for q_module, q_key, q_param in self.res_quant_bind[r_i]:
                    q_module._parameters[q_key] = q_param

    def profile_summary(self):
        return None if self.profiler is None else self.profiler.summary()

    def slot_of(self, s_idx):
        return s_idx % self.num_slots

    def convert_param(self, s_idx, j, dst, stg):
        """Storage-dtype path: write converted parameter j of block s_idx (staged bytes ``stg``) into slot view
        ``dst`` on the current stream -- dequantize + LoRA, or for a scaled fp8 weight dequantize + LoRA +
        re-quantize (its new scale goes into the block's bound QuantizedTensor parameters)."""
        fn = self.lora_fns[s_idx][j]
        qc = self.quant_conv[s_idx].get(j) if self.quant_conv is not None else None
        if j in self.quant_idx_set and (qc is None or fn is None):
            dst.copy_(stg)  # stored fp8 bytes, or a LoRA already baked into the host buffer
            return
        if qc is not None:
            layout_cls, file_params, bound_params, seed = qc
            y = _quant_lora_requant(QuantizedTensor(stg, layout_cls, file_params), fn, self.compute_dtype, seed)
            dst.copy_(y._qdata)
            _copy_quant_params_(bound_params, y._params)
            return
        x = stg.to(self.compute_dtype)
        if fn is not None:
            x = fn(x)
        dst.copy_(x)

    def materialize(self, s_idx, slot_idx):
        """Fill slot slot_idx with block s_idx's compute weights, on the current stream.
        The staging buffer is shared by all slots: work on one stream is ordered, so the next DMA into it
        only starts after this block's dequantization has consumed it."""
        prof = self.profiler
        stream = torch.cuda.current_stream(self.device)
        src_flat, ring_slot = self._host_source(s_idx)
        if prof is not None:
            prof.record(s_idx, "dma0", stream)
        if self.no_staging:
            self.slot_flats[slot_idx].copy_(src_flat, non_blocking=True)
            if ring_slot is not None:
                self.ring.release(ring_slot, stream)
            if prof is not None:
                prof.record(s_idx, "dma1", stream)
                prof.record(s_idx, "ready", stream)
            return
        self.staging_flat.copy_(src_flat, non_blocking=True)
        if ring_slot is not None:
            self.ring.release(ring_slot, stream)
        if prof is not None:
            prof.record(s_idx, "dma1", stream)
        for j, (dst, src) in enumerate(zip(self.slot_views[slot_idx], self.staging_views)):
            if j in self.convert_idx:
                self.convert_param(s_idx, j, dst, src)
            else:
                dst.copy_(src)
        if prof is not None:
            prof.record(s_idx, "ready", stream)

    def _host_source(self, s_idx):
        """Host bytes of block s_idx: its pinned RAM buffer, or the SSD ring buffer holding it (waits for the
        background reader). Returns (uint8 tensor, ring slot or None)."""
        if self.ring is not None and self.disk_index[s_idx] >= 0:
            return self.ring.acquire(self.disk_index[s_idx])
        return self.host_flats[s_idx], None

    def ssd_main_wait_s(self):
        """CPU time the main thread has spent waiting for SSD data (in hooks), for this load."""
        if self.uploader is not None:
            return self.uploader.main_wait_s
        return self.ring.wait_s if self.ring is not None else 0.0

    def finish_forward_timing(self, shape, ring_wait_s):
        """Queue this forward's block span (block 0 start -> last block end, GPU events) for harvesting.
        The first forward of a load (kernel autotune etc.) and forwards in which the CPU waited for SSD reads
        (they measure the SSD, not the compute) are not used."""
        start, end = self.span_start, self.span_end
        self.span_start = self.span_end = None
        self.forwards_done += 1
        if start is None or end is None or self.forwards_done <= 1:
            return
        self.pending_spans.append((start, end, shape, ring_wait_s))
        del self.pending_spans[:-8]

    def harvest_block_timing(self, dit):
        """Fold completed spans into the per-block compute estimate: median of the last 16 at the latest input
        shape. Called at forward start (the CPU usually runs ahead of the GPU, so recent spans may still be
        pending) and at teardown after a synchronize."""
        keep = []
        for start, end, shape, ring_wait_s in self.pending_spans:
            if not end.query():
                keep.append((start, end, shape, ring_wait_s))
                continue
            span_ms = start.elapsed_time(end)
            if span_ms <= 0 or ring_wait_s * 1000 > 0.02 * span_ms:
                continue
            samples = getattr(dit, "_weightferry_stream_block_samples", None)
            if samples is None or samples.get("shape") != shape:
                samples = {"shape": shape, "ms": []}
                setattr(dit, "_weightferry_stream_block_samples", samples)
            samples["ms"] = (samples["ms"] + [span_ms / max(1, len(_blocks(dit)))])[-16:]
            ordered = sorted(samples["ms"])
            setattr(dit, "_weightferry_stream_block_ms", ordered[len(ordered) // 2])
        self.pending_spans = keep

    def check_overflow(self, dit):
        """After a forward pass: has VRAM overflowed into shared GPU memory (system RAM, Windows sysmem fallback)?
        Silent and many times slower for the user, so it is reported once per load and remembered on the
        diffusion model: _auto_config keeps that much VRAM free on the next load (fewer resident blocks)."""
        if _env.get("WEIGHTFERRY_OVERFLOW_CHECK") == "0":
            return
        over = vram_monitor.overflow_bytes()
        if over <= 0:
            return
        self.overflow_peak = max(self.overflow_peak, over)
        prev = getattr(dit, "_weightferry_vram_overflow_bytes", 0) or 0
        setattr(dit, "_weightferry_vram_overflow_bytes", max(prev, over))
        if not self.overflow_warned:
            self.overflow_warned = True
            log_warn(f"DiT Layer Streaming: about {over / 1024 ** 2:.0f} MB of this process' GPU memory is in "
                     f"shared GPU memory (system RAM), so the GPU reads it over PCIe and generation is much slower. "
                     f"auto_config keeps that much VRAM free from the next load on; with manual settings lower "
                     f"resident_blocks (or close other GPU applications).")

    def end_forward_tiers(self):
        """A forward pass must consume all SSD-tier blocks or none; otherwise restart the reader's cycle."""
        if self.uploader is not None:
            self.uploader.drain()  # normally empty: every submitted block was waited for in its pre-hook
        if self.ring is not None:
            failed_upload = self.uploader is not None and self.uploader.error is not None
            if failed_upload and getattr(self.ring, "_error", None) is None:
                self.ring._error = self.uploader.error  # restart the cycle the failed upload left half consumed
            if self.ring.restart_if_needed(lambda m: log_warn("DiT Layer Streaming: " + m)) and failed_upload:
                self.uploader.error = None  # a fresh read cycle: the next forward may upload again (review 30)

    def issue_prefetch(self, s_idx):
        """Async DMA (+ dequantization / LoRA) of block s_idx into its slot, after that slot's last reader finished."""
        if self.profiler is not None:
            self.profiler.mark(s_idx, "cpu_issue")
        slot_idx = self.slot_of(s_idx)
        if self.uploader is not None and self.disk_index[s_idx] >= 0:
            self.uploader.submit(s_idx, slot_idx)
            self.issued[s_idx] = True
            return
        with torch.cuda.stream(self.prefetch_stream):
            self.prefetch_stream.wait_event(self.computed_events[slot_idx])
            self.materialize(s_idx, slot_idx)
            self.transferred_events[s_idx].record(self.prefetch_stream)
        # Windows (WDDM) batches CUDA commands and may hold the copy back until a later flush (measured:
        # ~1 block of compute late, so copies never overlapped compute). event.query() forces submission
        # without blocking the CPU (stream.query() blocked until the copy finished).
        self.transferred_events[s_idx].query()
        self.issued[s_idx] = True

    def begin_forward(self):
        self.ensure_slots()
        self.issued = [False] * self.num_stream

    def release_slots(self):
        """Give the GPU slots back between samplings (WEIGHTFERRY_VAE_FREE_SLOTS, 2026-10-13): every block is
        transferred into a slot again before it runs, so a slot holds nothing a later forward needs; ensure_slots()
        allocates them again (same layout) when the next forward begins. Only for plain block streaming (no SSD ring,
        no scaled fp8 parameter swaps bound to the slots). Returns the bytes released."""
        if (getattr(self, "slots_released", False) or not self.slot_flats or self.quant_bind is not None
                or self.ring is not None or self.uploader is not None or getattr(self, "slot_specs", None) is None):
            return 0
        torch.cuda.synchronize(self.device)  # no copy / kernel may still use a slot
        self.restore_rest_binding()
        n = sum(int(f.numel()) for f in self.slot_flats)
        self._slot_flat_bytes = int(self.slot_flats[0].numel())
        self.slot_flats, self.slot_views = [], []
        self.slot_ptrs = set()  # the freed addresses may be reused by other tensors (restore_rest_binding checks them)
        self.slots_released = True
        return n

    def ensure_slots(self):
        if not getattr(self, "slots_released", False):
            return
        # built aside and published only when complete (Codex CPU check 2026-10-13: a failure halfway left slots in the
        # state while it still read as released, so release_slots would skip them); the error is not swallowed
        flats = [torch.empty(self._slot_flat_bytes, dtype=torch.uint8, device=self.device)
                 for _ in range(self.num_slots)]
        views = [_views(f, self.slot_specs, self.slot_offsets) for f in flats]
        compute = torch.cuda.current_stream(self.device)
        for ev in self.computed_events:  # the first prefetch into a new slot waits for the work queued so far
            ev.record(compute)
        self.slot_flats, self.slot_views = flats, views
        self.slot_ptrs = {v.data_ptr() for vs in views for v in vs}
        self.slots_released = False

    def prefetch_first_slots(self):
        """Called when block 0 actually runs, so cached steps (e.g. TaylorSeer skips) cost no DMA."""
        if self.is_double:
            for s_idx in range(min(self.num_slots, self.num_stream)):
                if not self.issued[s_idx]:
                    self.issue_prefetch(s_idx)

    def restore_rest_binding(self):
        """Rebind any parameter that still points at a GPU slot / self-managed resident buffer to its rest tensor."""
        for s_i, (blk, rest) in enumerate(zip(self.streaming_blocks, self.rest)):
            prms = list(blk.parameters())
            if self.quant_rest is not None:
                for q_module, q_key, q_orig in self.quant_rest[s_i]:
                    if q_module._parameters[q_key] is not q_orig:
                        q_module._parameters[q_key] = q_orig
            for j, (p, r) in enumerate(zip(prms, rest)):
                if j in self.quant_idx_set:
                    continue
                if p.data.data_ptr() in self.slot_ptrs:
                    p.data = r
        for r_i, (blk, rest) in enumerate(zip(self.resident_blocks, self.resident_rest)):
            if self.res_quant_rest is not None:
                for q_module, q_key, q_orig in self.res_quant_rest[r_i]:
                    if q_module._parameters[q_key] is not q_orig:
                        q_module._parameters[q_key] = q_orig
            for j, (p, r) in enumerate(zip(blk.parameters(), rest)):
                if j in self.quant_idx_set:
                    continue
                if _raw(p.data).data_ptr() in self.resident_ptrs:
                    p.data = r
