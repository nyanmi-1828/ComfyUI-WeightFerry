# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
CFG Early-Stop
====================
A PREDICT_NOISE wrapper on the model: during the final steps of sampling it sets the guider's CFG to 1.0 for the call,
so ComfyUI's native cfg1_optimization skips the negative (unconditional) pass, halving the model calls of those steps.
"""

from typing import Tuple, Any
import weakref
import torch
import comfy.patcher_extension

from ..config import WeightFerryConfig


_HOST_SIGMAS_CACHE = {"ref": None, "values": None}


def _host_sigmas(sample_sigmas):
    """Schedule as a Python list, transferred once per sampling run.
    ComfyUI may move sample_sigmas to the GPU, where per-element float() would sync once per sigma."""
    if sample_sigmas is None or len(sample_sigmas) < 2:
        return None
    if not isinstance(sample_sigmas, torch.Tensor):
        return [float(s) for s in sample_sigmas]
    # Identity of the live tensor object (the same object is passed on every call of one sampling run).
    # A data_ptr key is unsafe: another run's schedule of the same length may reuse the allocation.
    ref = _HOST_SIGMAS_CACHE["ref"]
    if ref is None or ref() is not sample_sigmas:
        _HOST_SIGMAS_CACHE["ref"] = weakref.ref(sample_sigmas)
        _HOST_SIGMAS_CACHE["values"] = sample_sigmas.detach().float().cpu().tolist()
    return _HOST_SIGMAS_CACHE["values"]


def _step_index(sig_list, sigma, rel_tol=1e-5):
    """Index i of the step whose interval (sig[i+1], sig[i]] contains sigma.
    Robust to samplers that evaluate the model at a slightly lowered sigma within a step
    (e.g. Detail Daemon), where the nearest schedule entry could already be the next step."""
    total_steps = len(sig_list) - 1
    count = sum(1 for s in sig_list[:-1] if s >= sigma * (1.0 - rel_tol))
    return min(max(count - 1, 0), total_steps)


def _progress(guider, timestep, model_options):
    """Sampling progress of this call (0.0 -> 1.0): the step of the schedule (transformer_options' sample_sigmas)
    over the number of steps, else from the model's sigma range, else 0.0."""
    transformer_options = model_options.get("transformer_options", {})
    sample_sigmas = transformer_options.get("sample_sigmas", None)

    # One host transfer per call (batch-safe: timestep may hold one sigma per batch item)
    if isinstance(timestep, torch.Tensor):
        current_sigma = float(timestep.detach().max())
    else:
        current_sigma = float(timestep)

    sig_list = _host_sigmas(sample_sigmas)
    if sig_list is not None:
        return _step_index(sig_list, current_sigma) / (len(sig_list) - 1)

    # Fallback based on model dynamic range
    model_sampling = getattr(guider.inner_model, "model_sampling", None)
    if model_sampling is not None:
        s_max = float(model_sampling.sigma_max)
        s_min = float(model_sampling.sigma_min)
        if s_max > s_min:
            return max(0.0, min(1.0, (s_max - current_sigma) / (s_max - s_min)))
    return 0.0


class WeightFerryCFGEarlyStop:
    """
    CFG Early-Stop Node: returns the model with a PREDICT_NOISE wrapper.
    Applies CFG scale normally up to early_stop_percent, then drops CFG to 1.0 (hard_stop)
    or decays it smoothly to 1.0 (linear_decay).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "early_stop_percent": (
                    "FLOAT",
                    {
                        "default": WeightFerryConfig.CFG_EARLY_STOP_DEFAULT_PERCENT,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Fraction of the sampling with normal CFG (0.65 = CFG for the first 65%, CFG 1.0 for the rest, which halves those steps' model calls)"
                    },
                ),
                "decay_mode": (
                    ["hard_stop", "linear_decay"],
                    {
                        "default": WeightFerryConfig.CFG_EARLY_STOP_DEFAULT_MODE,
                        "tooltip": "hard_stop: CFG 1.0 as soon as the threshold is reached (fastest). linear_decay: decays smoothly to 1.0"
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_early_stop"
    CATEGORY = "WeightFerry/speed"
    TITLE = "CFG Early-Stop"

    def apply_early_stop(
        self,
        model: Any,
        early_stop_percent: float = WeightFerryConfig.CFG_EARLY_STOP_DEFAULT_PERCENT,
        decay_mode: str = WeightFerryConfig.CFG_EARLY_STOP_DEFAULT_MODE
    ) -> Tuple[Any]:
        m = model.clone()

        def early_stop_predict_noise_wrapper(executor, x, timestep, model_options={}, seed=None):
            guider = getattr(executor, "class_obj", None)
            if guider is None or not hasattr(guider, "cfg"):
                return executor(x, timestep, model_options, seed)

            orig_cfg = guider.cfg
            if orig_cfg <= 1.0:
                return executor(x, timestep, model_options, seed)

            progress = _progress(guider, timestep, model_options)

            # Early stop determination
            if progress >= early_stop_percent:
                if decay_mode == "hard_stop":
                    new_cfg = 1.0
                elif decay_mode == "linear_decay":
                    denom = max(1e-5, 1.0 - early_stop_percent)
                    ratio = min(1.0, max(0.0, (progress - early_stop_percent) / denom))
                    new_cfg = orig_cfg + ratio * (1.0 - orig_cfg)
                else:
                    new_cfg = 1.0

                try:
                    guider.cfg = new_cfg
                    return executor(x, timestep, model_options, seed)
                finally:
                    guider.cfg = orig_cfg
            else:
                return executor(x, timestep, model_options, seed)

        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.PREDICT_NOISE,
            early_stop_predict_noise_wrapper,
            m.model_options,
            is_model_options=True,
        )
        return (m,)
