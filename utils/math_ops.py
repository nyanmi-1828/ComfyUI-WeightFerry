# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
ComfyUI-WeightFerry: Common Mathematical and Tensor Operations
=============================================================
Provides clean, reusable vector field calculations, RMS normalization,
and stability guards across samplers and extrapolation engines.
"""

from typing import Union
import torch
from ..config import WeightFerryConfig


def to_d_vector(
    x: torch.Tensor,
    sigma: Union[torch.Tensor, float],
    denoised: torch.Tensor
) -> torch.Tensor:
    """
    Compute velocity vector field d = (x - denoised) / sigma.
    Safe with broadcast alignment for [B, C, H, W] latents and scalar/tensor sigmas.
    """
    if isinstance(sigma, torch.Tensor):
        s = sigma.view([-1] + [1] * (x.ndim - 1))
    else:
        s = float(sigma)
    return (x - denoised) / s


def rms_norm(tensor: torch.Tensor, eps: float = WeightFerryConfig.EPSILON_FP32) -> float:
    """
    Compute root-mean-square (RMS) scalar of a tensor in float32.
    Formula: sqrt( mean( tensor^2 ) )
    """
    t_f = tensor.float()
    return torch.sqrt(torch.mean(t_f ** 2) + eps).item()


def relative_mismatch(
    u: torch.Tensor,
    w: torch.Tensor,
    eps: float = WeightFerryConfig.EPSILON_FP32
) -> float:
    """
    Compute relative local mismatch between two velocity vectors u and w (Eq. 4 in ReAL paper).
    r_loc = ||u - w||_rms / (||u||_rms + eps)
    """
    diff_rms = rms_norm(u - w, eps=0.0)
    u_rms = rms_norm(u, eps=0.0)
    return diff_rms / (u_rms + eps)
