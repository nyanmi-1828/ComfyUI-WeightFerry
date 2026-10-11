# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Anima TORS (Total Rotation Schedule) Scheduler Module
=====================================================
Named after TORS (arXiv:2603.00763, "Analyzing and Improving Fast Sampling of Text-to-Image Diffusion Models",
ECCV 2026; official code zju-pi/TORS). The sigmas come from config.TORS_PRECOMPUTED_TABLE, computed for this package from
the paper's formulation (not copied from the official .pth schedules).
"""

from typing import Optional, Tuple
import torch
import numpy as np

from ..config import WeightFerryConfig


def calculate_tors_sigmas(steps: int) -> torch.Tensor:
    """
    Calculate TORS sigmas for Flow Matching models [1.0 -> 0.0].
    Returns a 1D float32 tensor of length steps + 1, terminating strictly in 0.0.
    """
    table = WeightFerryConfig.TORS_PRECOMPUTED_TABLE
    if steps in table:
        base_schedule = list(table[steps])
    else:
        # Interpolate from 30-step reference curve
        ref_30 = table[30]
        x_ref = np.linspace(0, 1, len(ref_30))
        x_target = np.linspace(0, 1, steps)
        base_schedule = np.interp(x_target, x_ref, ref_30).tolist()

    sigmas = list(base_schedule) + [0.0]
    return torch.tensor(sigmas, dtype=torch.float32)


def tors_scheduler_handler(model_sampling: object, steps: int) -> torch.Tensor:
    """
    ComfyUI SCHEDULER_HANDLER entrypoint.
    Called by comfy.samplers.calculate_sigmas(model_sampling, "tors", steps).
    """
    sigmas = calculate_tors_sigmas(steps)

    # Model-aware scaling if sigma_max differs from standard Flow Matching [1.0, 0.0]
    if hasattr(model_sampling, "sigma_max"):
        s_max = float(model_sampling.sigma_max)
        if abs(s_max - 1.0) > 1e-4:
            sigmas = sigmas * s_max

    sigmas[-1] = 0.0
    return sigmas


class WeightFerryTORSScheduler:
    """
    TORS Scheduler Node: sigmas from the precomputed table (see the module docstring for its provenance).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": WeightFerryConfig.TORS_DEFAULT_STEPS,
                    "min": WeightFerryConfig.TORS_MIN_STEPS,
                    "max": WeightFerryConfig.TORS_MAX_STEPS,
                    "step": 1,
                    "tooltip": "Sampling steps (suggested 10-12)"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise (1.0 = full generation, below 1.0 = start part way, e.g. for upscaling)"
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional: connect to scale the schedule to the model's sigma_max"}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "WeightFerry/sampling"
    TITLE = "TORS Scheduler"

    def get_sigmas(
        self,
        steps: int = WeightFerryConfig.TORS_DEFAULT_STEPS,
        denoise: float = 1.0,
        model: Optional[object] = None
    ) -> Tuple[torch.Tensor]:
        if model is not None and hasattr(model, "get_model_object"):
            ms = model.get_model_object("model_sampling")
            sigmas = tors_scheduler_handler(ms, steps)
        else:
            sigmas = calculate_tors_sigmas(steps)

        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = len(sigmas) - 1
            start_step = int(round(total_steps * (1.0 - denoise)))
            sigmas = sigmas[start_step:]

        return (sigmas,)
