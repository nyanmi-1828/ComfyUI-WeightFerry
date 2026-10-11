# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
ReAL: Accelerating Flow Matching through Segment Advancement with Shared Lookahead
===================================================================================
Paper: arXiv:2609.33202 (2026)
Algorithm 1: ReAL with Carry Policy and Dense Tail Protection.
"""

from typing import Optional, Callable, Dict, Any, Tuple
import torch
import comfy.samplers
import comfy.utils

from ..config import WeightFerryConfig
from ..utils.math_ops import to_d_vector, relative_mismatch


@torch.no_grad()
def sample_real(
    model: Any,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[Dict[str, Any]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    disable: Optional[bool] = None,
    K: int = WeightFerryConfig.REAL_DEFAULT_K,
    q_loc: float = WeightFerryConfig.REAL_DEFAULT_Q_LOC,
    q_seg: float = WeightFerryConfig.REAL_DEFAULT_Q_SEG,
    min_k: int = WeightFerryConfig.REAL_DEFAULT_MIN_K,
    carry_policy: str = WeightFerryConfig.REAL_DEFAULT_CARRY_POLICY,
    tail_protect_sigma: float = WeightFerryConfig.REAL_TAIL_PROTECT_SIGMA,
    **kwargs
) -> torch.Tensor:
    """
    ReAL Solver for Flow Matching models.
    Performs curvature-aware adaptive segment advancement with shared lookahead reuse.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    T = len(sigmas) - 1
    i = 0
    b: Optional[torch.Tensor] = None  # Carry buffer for lookahead velocity

    pbar = comfy.utils.model_trange(T, disable=disable)

    def eval_velocity(curr_x: torch.Tensor, sigma_val: float) -> Tuple[torch.Tensor, torch.Tensor]:
        denoised = model(curr_x, sigma_val * s_in, **extra_args)
        d = to_d_vector(curr_x, sigma_val, denoised)
        return d, denoised

    while i < T:
        sigma_i = sigmas[i]
        sigma_i1 = sigmas[i + 1]
        delta_sigma_i = sigma_i1 - sigma_i

        # 1. Starting velocity u_i (Algorithm 1, Lines 6-12)
        if b is None:
            u_i, denoised_i = eval_velocity(x, sigma_i)
        else:
            u_i = b
            b = None
            denoised_i = None

        if callback is not None and denoised_i is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_i,
                'sigma_hat': sigma_i,
                'denoised': denoised_i
            })

        K_a = min(int(K), T - i)
        # Lookahead candidate state z_i (1-step Euler)
        z_i = x + delta_sigma_i * u_i

        # Dense tail protection: if remaining steps < min_k OR noise level is in fine-detail tail
        is_tail = (K_a < min_k) or (float(sigma_i1) < tail_protect_sigma)

        if is_tail:
            # Dense 1-step advancement
            x = z_i
            i += 1
            b = None
            pbar.update(1)
        else:
            # Algorithm 1, Lines 18-20: Shared lookahead evaluation
            w_i, denoised_w = eval_velocity(z_i, sigma_i1)

            if callback is not None:
                callback({
                    'x': z_i,
                    'i': i,
                    'sigma': sigma_i1,
                    'sigma_hat': sigma_i1,
                    'denoised': denoised_w
                })

            # Algorithm 1, Line 21 & Eq. (4): Local mismatch residual
            r_loc = relative_mismatch(u_i, w_i, eps=WeightFerryConfig.EPSILON_FP32)
            h_i = abs((delta_sigma_i).item())

            # Algorithm 1, Line 24 & Eq. (5): Clipped candidate set in descending order
            cands = sorted(list(set([
                K_a,
                max(min_k, int(3 * K_a // 4)),
                max(min_k, int(K_a // 2)),
                max(min_k, int(K_a // 4))
            ])), reverse=True)

            k_star = 1
            if r_loc <= q_loc:
                for k_cand in cands:
                    s_ik = abs((sigmas[i + k_cand] - sigma_i1).item())
                    r_seg = (s_ik / (h_i + WeightFerryConfig.EPSILON_SIGMA)) * r_loc
                    if r_seg <= q_seg:
                        k_star = k_cand
                        break

            # Algorithm 1, Lines 31-35: State update
            if k_star == 1:
                # Fallback to 1-step lookahead
                x = z_i
                b = w_i
            else:
                # Lookahead-corrected segment advancement
                advance_dt = sigmas[i + k_star] - sigma_i1
                x = z_i + advance_dt * w_i

                # Carry policy (Table 18 from paper)
                if carry_policy == "always":
                    b = w_i
                else:
                    b = None

            i += k_star
            pbar.update(k_star)

    pbar.close()
    return x


class WeightFerryReALSampler:
    """
    ReAL Sampler Node (arXiv:2609.33202, 2026)
    Accelerating Flow Matching through Segment Advancement with Shared Lookahead.
    Outputs a SAMPLER object usable in SamplerCustomAdvanced / KSampler.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_span_K": ("INT", {
                    "default": WeightFerryConfig.REAL_DEFAULT_K,
                    "min": 2,
                    "max": 24,
                    "step": 1,
                    "tooltip": "Maximum jump K (for 16 steps 3-4 is safe; about 12 for 50 steps)"
                }),
                "q_loc": ("FLOAT", {
                    "default": WeightFerryConfig.REAL_DEFAULT_Q_LOC,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Local residual tolerance q_loc (default 0.70)"
                }),
                "q_seg": ("FLOAT", {
                    "default": WeightFerryConfig.REAL_DEFAULT_Q_SEG,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Segment residual tolerance q_seg (quality: 1.25, faster: 1.65)"
                }),
                "carry_policy": (["fallback_only", "always"], {
                    "default": WeightFerryConfig.REAL_DEFAULT_CARRY_POLICY,
                    "tooltip": "Carry policy: fallback_only (recompute the velocity after a long jump; recommended against artifacts with few-step models), always (always reuse)"
                }),
                "tail_protect_sigma": ("FLOAT", {
                    "default": WeightFerryConfig.REAL_TAIL_PROTECT_SIGMA,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Tail protection sigma (below it no step is skipped, which protects colours and outlines)"
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "WeightFerry/sampling"
    FUNCTION = "get_sampler"
    TITLE = "ReAL Sampler"

    def get_sampler(
        self,
        max_span_K: int = WeightFerryConfig.REAL_DEFAULT_K,
        q_loc: float = WeightFerryConfig.REAL_DEFAULT_Q_LOC,
        q_seg: float = WeightFerryConfig.REAL_DEFAULT_Q_SEG,
        carry_policy: str = WeightFerryConfig.REAL_DEFAULT_CARRY_POLICY,
        tail_protect_sigma: float = WeightFerryConfig.REAL_TAIL_PROTECT_SIGMA
    ):
        extra_options = {
            "K": max_span_K,
            "q_loc": q_loc,
            "q_seg": q_seg,
            "min_k": WeightFerryConfig.REAL_DEFAULT_MIN_K,
            "carry_policy": carry_policy,
            "tail_protect_sigma": tail_protect_sigma,
        }
        sampler = comfy.samplers.KSAMPLER(sample_real, extra_options=extra_options)
        return (sampler,)
