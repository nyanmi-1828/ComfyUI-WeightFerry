# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Pipeline timeline profiler of block streaming (WEIGHTFERRY_PROFILE=1)."""

import torch


class _StreamProfiler:
    """CUDA-event timeline of the streaming pipeline, per forward call.

    Per streamed block (all events on the GPU timeline, so streams can be compared):
      dma0 -> dma1   : host -> GPU copy (staging or slot)
      dma1 -> ready  : dequantization + LoRA (storage-dtype path only)
      wait0 -> start : compute stream stalled waiting for this block's transfer (exposed transfer time)
      start -> end   : block compute
    Resident blocks record r_start / r_end, so the summary can report how long the compute stream was NOT
    computing blocks during a forward (idle: CPU not feeding it fast enough, or stalled on transfers).
    """
    MAX_FORWARDS = 64

    def __init__(self, num_stream, host_bytes, is_double, no_staging):
        self.num_stream = num_stream
        self.host_bytes = host_bytes
        self.is_double = is_double
        self.no_staging = no_staging  # DMA straight into the slot (no staging buffer / storage-dtype conversion)
        self.forwards = []
        self.cur = None
        # Idle stream: an event recorded on it completes as soon as the CPU issues it, so its GPU timestamp
        # marks *when the CPU reached that point* on the GPU timeline
        self.marker_stream = torch.cuda.Stream()

    def mark(self, s_idx, name):
        if self.cur is not None:
            self.cur["blocks"].setdefault(s_idx, {})[name] = self._event(self.marker_stream)

    @staticmethod
    def _event(stream):
        e = torch.cuda.Event(enable_timing=True)
        e.record(stream)
        return e

    def begin_forward(self, stream):
        if len(self.forwards) >= self.MAX_FORWARDS:
            self.cur = None
            return
        self.cur = {"start": self._event(stream), "blocks": {}, "cpu_hook_s": 0.0}
        self.forwards.append(self.cur)

    def end_forward(self, stream):
        if self.cur is not None:
            self.cur["end"] = self._event(stream)
            self.cur = None

    def record(self, s_idx, name, stream):
        if self.cur is not None:
            self.cur["blocks"].setdefault(s_idx, {})[name] = self._event(stream)

    def record_resident(self, b_idx, name, stream):
        if self.cur is not None:
            self.cur.setdefault("resident", {}).setdefault(b_idx, {})[name] = self._event(stream)

    def add_cpu(self, seconds):
        if self.cur is not None:
            self.cur["cpu_hook_s"] += seconds

    def summary(self):
        torch.cuda.synchronize()
        per_forward = []
        for f in self.forwards:
            if "end" not in f:
                continue
            dma = conv = stall = comp = 0.0
            for b in f["blocks"].values():
                if "dma0" in b and "dma1" in b:
                    dma += b["dma0"].elapsed_time(b["dma1"])
                if "dma1" in b and "ready" in b:
                    conv += b["dma1"].elapsed_time(b["ready"])
                if "wait0" in b and "start" in b:
                    stall += b["wait0"].elapsed_time(b["start"])
                if "start" in b and "end" in b:
                    comp += b["start"].elapsed_time(b["end"])
            resident = 0.0
            for b in f.get("resident", {}).values():
                if "r_start" in b and "r_end" in b:
                    resident += b["r_start"].elapsed_time(b["r_end"])
            transfer = dma + conv
            n_blocks = len(f["blocks"])
            forward_ms = f["start"].elapsed_time(f["end"])
            per_forward.append({
                "forward_ms": round(forward_ms, 2),
                "streamed_blocks": n_blocks,
                "resident_compute_ms": round(resident, 2),
                "streamed_compute_ms": round(comp, 2),
                # Compute stream not running any block (embedding/final layer included, a few ms)
                "compute_idle_ms": round(forward_ms - resident - comp, 2),
                "dma_ms": round(dma, 2),
                "dequant_lora_ms": round(conv, 2),
                "stall_ms": round(stall, 2),
                # GPU-side only: share of transfer+dequant not waited on by the compute stream. Does not see
                # CPU-bound gaps (see compute_idle_ms).
                "overlap_ratio": round(max(0.0, min(1.0, 1.0 - stall / transfer)), 3) if transfer > 0 else None,
                "dma_gbps": round(self.host_bytes * n_blocks / (dma / 1000) / 1e9, 2) if dma > 0 else None,
                "cpu_hook_ms": round(f["cpu_hook_s"] * 1000, 2),
            })
        later = per_forward[1:] or per_forward
        mean = {}
        if later:
            for k in ("forward_ms", "resident_compute_ms", "streamed_compute_ms", "compute_idle_ms", "dma_ms",
                      "dequant_lora_ms", "stall_ms", "overlap_ratio", "dma_gbps", "cpu_hook_ms"):
                vals = [x[k] for x in later if x[k] is not None]
                mean[k] = round(sum(vals) / len(vals), 3) if vals else None
        # GPU timeline (ms since forward start) of one representative forward (the second one, if any)
        timeline = None
        complete = [f for f in self.forwards if "end" in f]
        if complete:
            f = complete[1] if len(complete) > 1 else complete[0]
            t0 = f["start"]
            timeline = {
                "forward_ms": round(t0.elapsed_time(f["end"]), 2),
                "resident": {str(b): {k: round(t0.elapsed_time(e), 2) for k, e in ev.items()}
                             for b, ev in sorted(f.get("resident", {}).items())},
                "streamed": {str(s): {k: round(t0.elapsed_time(e), 2) for k, e in ev.items()}
                             for s, ev in sorted(f["blocks"].items())},
            }
        return {
            "timeline_one_forward": timeline,
            "forwards_recorded": len(per_forward),
            "host_mb_per_block": round(self.host_bytes / 1024 ** 2, 1),
            "double_buffer": self.is_double,
            "storage_dtype_path": not self.no_staging,
            "mean_excluding_first_forward": mean,
            "per_forward": per_forward,
        }
