# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
VRAM budget (2026-10-12): the most VRAM ComfyUI (this process) may use, in GB, from WEIGHTFERRY_VRAM_BUDGET_GB (or the
settings file) or the DiT node's vram_budget_gb (which wins when > 0). 0 / unset: no budget -- every free byte is
used for speed, exactly as before.

A budget is enforced by the means ComfyUI and Dynamic VRAM already have for keeping VRAM free for other
applications: ComfyUI's reserve (model_management.EXTRA_RESERVED_VRAM, what --reserve-vram sets) and comfy-aimdo's
simple VRAM headroom are raised so that the free VRAM they keep leaves this process at most the budget. Every
automatic choice that reads free VRAM (resident blocks, activation streaming, TaylorSeer's cache, the VAE tiles,
ComfyUI's model loads) then plans inside the budget. The headroom is computed when a stage starts (apply(): the DiT
node, its load, the VAE and upscale nodes), from the device's free VRAM and this process's own use at that moment;
other applications' use is taken as it is then.

All the choices it moves are ones that do not change the output (which blocks stream, where the activations and
caches live, how the VAE is tiled): only the speed depends on the budget.
"""

import os

from . import env as _env

_node_budget_gb = [0.0]   # the DiT node's vram_budget_gb of the last run (0 = the setting)
_orig = {}                # ComfyUI's reserve / aimdo's headroom before the first raise
_extra = [0]              # bytes added to ComfyUI's reserve by the budget now
_peak_warned = []
_peak_seen = [0]          # PyTorch's reserved peak at the previous stage start (a peak from before the budget is not
                          # the budget's)


def set_node_budget(gb):
    _node_budget_gb[0] = float(gb or 0.0)


def budget_bytes():
    """The budget in bytes, or None (no budget: use all free VRAM)."""
    gb = _node_budget_gb[0]
    if gb <= 0:
        try:
            gb = float(_env.get("WEIGHTFERRY_VRAM_BUDGET_GB") or 0)
        except ValueError:
            gb = 0.0
    return int(gb * 1024 ** 3) if gb > 0 else None


def process_used(device):
    """VRAM this process uses now: Windows' per-process dedicated counter when readable (includes the CUDA context and
    Dynamic VRAM's pages), else PyTorch's reserved memory plus Dynamic VRAM's loaded pages."""
    from . import vram_monitor
    try:
        mem = vram_monitor._gpu_process_memory(os.getpid())
        if mem is not None and mem[0] > 0:
            return int(mem[0])
    except Exception:  # noqa: BLE001 -- counters unavailable: estimate below
        pass
    import torch
    used = int(torch.cuda.memory_reserved(device))
    try:
        import comfy_aimdo.control
        used += int(comfy_aimdo.control.get_total_vram_usage())
    except Exception:  # noqa: BLE001 -- no Dynamic VRAM
        pass
    return used


def extra_reserved():
    """Bytes the budget currently adds to ComfyUI's reserve (0 without a budget)."""
    return _extra[0]


def kept_free():
    """For WeightFerry's own planners, which do not subtract ComfyUI's reserve (their own headroom covers it): the VRAM
    to keep free while the budget binds -- the whole reserve it set, so that their plan stays inside the budget -- and
    0 when it does not bind (no budget, or more budget than the device can give: the same plan as without one)."""
    return _extra[0] + _orig.get("reserve", 0) if _extra[0] > 0 else 0


def _warn_peak(device, budget):
    """Once per process: the budget is a soft limit (an allocation beyond it is not refused, so no out-of-memory
    fallback changes the image); say so when this process's peak so far went past it (review 23)."""
    import torch
    peak = int(torch.cuda.max_memory_reserved(device))
    grew, _peak_seen[0] = peak > _peak_seen[0], max(peak, _peak_seen[0])
    if _peak_warned:
        return
    if grew and peak > budget and _extra[0] > 0:
        _peak_warned.append(True)
        from ..utils.logging import log_warn
        log_warn(f"VRAM budget {budget / 1024 ** 3:.2f} GB: PyTorch's peak so far {peak / 1024 ** 2:.0f} MB went past it "
                 "(the budget steers the automatic choices; allocations beyond it are not refused).")


def free_dit_slots(stage=""):
    """WEIGHTFERRY_VAE_FREE_SLOTS (2026-10-13, Codex round 6 B2; on by default): before a VAE / upscale stage, give back
    the GPU transfer slots of every streamed DiT ComfyUI still has loaded (3 x 132 MiB for Anima at prefetch 3; they
    were measured still held at the first USDU encode). They are allocated again when that DiT's next forward begins;
    every block is transferred into its slot before it runs anyway, so the output is the same. "1": always;
    "budget": only while the VRAM budget binds."""
    mode = _env.get("WEIGHTFERRY_VAE_FREE_SLOTS")
    if mode not in ("1", "budget") or (mode == "budget" and _extra[0] <= 0):
        return 0
    import comfy.model_management as mm
    from ..streaming.common import _STATE_ATTR
    freed = 0
    for lm in list(getattr(mm, "current_loaded_models", [])):
        try:
            rm = lm.real_model() if lm.is_dead() else lm.model.model
            dit = getattr(rm, "diffusion_model", None)
            state = getattr(dit, _STATE_ATTR, None) if dit is not None else None
            if state is not None and hasattr(state, "release_slots"):
                freed += state.release_slots()
        except Exception:  # noqa: BLE001 -- ComfyUI internals differ by version: nothing released
            pass
    if freed:
        from ..utils.logging import log_info
        log_info(f"VRAM ({stage}): released the streamed DiT's GPU slots ({freed / 1024 ** 2:.0f} MB) until its "
                 "next forward.")
    return freed


def apply(device, stage=""):
    """Re-derive the reserve for the budget at the start of a stage; restore ComfyUI's / aimdo's own values when there
    is no budget. Returns the bytes added to ComfyUI's reserve."""
    import comfy.model_management as mm
    budget = budget_bytes()
    if not mm.is_device_cuda(device):
        return _extra[0]
    if "reserve" not in _orig:
        if budget is None:
            return 0
        _orig["reserve"] = int(mm.EXTRA_RESERVED_VRAM)
        try:
            import comfy_aimdo.control
            _orig["aimdo"] = comfy_aimdo.control.get_simple_vram_headroom()
        except Exception:  # noqa: BLE001 -- no Dynamic VRAM
            _orig["aimdo"] = None
    if budget is None:
        new = _orig["reserve"]
    else:
        import torch
        _warn_peak(device, budget)
        free, _total = torch.cuda.mem_get_info(device)
        # free = total - others - ours: keeping R = free + ours - budget free leaves ours <= budget
        new = max(_orig["reserve"], int(free + process_used(device) - budget))
    if new != int(mm.EXTRA_RESERVED_VRAM):
        mm.EXTRA_RESERVED_VRAM = new
        if _orig.get("aimdo") is not None:
            try:
                import comfy_aimdo.control
                comfy_aimdo.control.set_simple_vram_headroom(_orig["aimdo"] if budget is None
                                                             else max(_orig["aimdo"], new))
            except Exception:  # noqa: BLE001 -- the reserve still holds for ComfyUI's own loads
                pass
        from ..utils.logging import log_info
        if budget is None:
            log_info(f"VRAM budget: none; ComfyUI's reserve back to {_orig['reserve'] / 1024 ** 2:.0f} MB.")
        else:
            log_info(f"VRAM budget {budget / 1024 ** 3:.2f} GB ({stage}): VRAM kept free "
                     f"{new / 1024 ** 2:.0f} MB (ComfyUI's own reserve {_orig['reserve'] / 1024 ** 2:.0f} MB).")
    _extra[0] = new - _orig["reserve"]
    return _extra[0]
