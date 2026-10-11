# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
ComfyUI-WeightFerry: Shared Constants
=====================================
Developed by nyanmi-1828 (https://github.com/nyanmi-1828).
Constants shared by the nodes: runtime settings, epsilons, the Anima 2.9B block layout, the TORS table,
ReAL sampler / CFG early-stop defaults and the USDU tile calculator's parameters.
"""

import os
from typing import Final, Dict, List, Tuple


class WeightFerryConfig:
    # ==========================================
    # 1. System & Low-Level Runtime Constants
    # ==========================================
    # Windows high-resolution multimedia timer period (ms)
    WIN32_TIMER_RESOLUTION_MS: Final[int] = 1

    # PyTorch TF32 & precision settings
    FLOAT32_MATMUL_PRECISION: Final[str] = "high"
    # Triton JIT cache: <ComfyUI base directory>/.cache/triton unless TRITON_CACHE_DIR is set (core/tensor_optimizer.py)
    TRITON_CACHE_SUBDIR: Final[str] = os.path.join(".cache", "triton")

    # ==========================================
    # 2. Numerical Stability & Floating-Point Epsilon
    # ==========================================
    EPSILON_FP32: Final[float] = 1e-8
    EPSILON_SIGMA: Final[float] = 1e-8

    # ==========================================
    # 3. Anima 2.9B Architecture Constants
    # ==========================================
    # Anima 2.9B has 40 blocks total (28 original base blocks + 12 inserted blocks)
    NUM_BLOCKS_29B: Final[int] = 40
    # Inserted block indices measured by sparklingcoffee777/ComfyUI-Anima-2.9B-loraPatch (THIRD_PARTY_NOTICES.md)
    INSERTED_BLOCK_INDICES: Final[Tuple[int, ...]] = (
        2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36
    )
    NUM_BLOCKS_BASE: Final[int] = NUM_BLOCKS_29B - len(INSERTED_BLOCK_INDICES)  # 28

    # ==========================================
    # 4. TORS Scheduler (ECCV 2026 / arXiv:2603.00763)
    # ==========================================
    TORS_DEFAULT_STEPS: Final[int] = 10
    TORS_MIN_STEPS: Final[int] = 4
    TORS_MAX_STEPS: Final[int] = 50

    # TORS sigma schedules (4-30 steps), precomputed for this package from the TORS paper's formulation
    # (arXiv:2603.00763) and stored here so that no computation is needed at sampling time. They are not copied from the
    # official repository zju-pi/TORS (which distributes its schedules as .pth files).
    TORS_PRECOMPUTED_TABLE: Final[Dict[int, List[float]]] = {
        4: [1.0, 0.8822, 0.7737, 0.4354],
        5: [1.0, 0.9098, 0.8164, 0.6879, 0.3877],
        6: [1.0, 0.9317, 0.8390, 0.7737, 0.5625, 0.3501],
        7: [1.0, 0.9438, 0.8578, 0.7925, 0.7008, 0.4354, 0.2786],
        8: [1.0, 0.9532, 0.8822, 0.8254, 0.7737, 0.6591, 0.4354, 0.2786],
        9: [1.0, 0.9585, 0.8970, 0.8408, 0.7925, 0.7145, 0.5625, 0.3877, 0.2528],
        10: [1.0, 0.9628, 0.9098, 0.8578, 0.8164, 0.7737, 0.6879, 0.5214, 0.3877, 0.2130],
        11: [1.0, 0.9645, 0.9204, 0.8715, 0.8296, 0.7925, 0.7145, 0.6102, 0.4786, 0.3501, 0.1702],
        12: [1.0, 0.9661, 0.9317, 0.8822, 0.8390, 0.8098, 0.7737, 0.7008, 0.5625, 0.4354, 0.3501, 0.1275],
        14: [1.0, 0.9678, 0.9438, 0.9020, 0.8578, 0.8254, 0.7990, 0.7737, 0.7145, 0.6373, 0.5214, 0.4111, 0.3118, 0.0825],
        15: [1.0, 0.9685, 0.9485, 0.9098, 0.8660, 0.8345, 0.8098, 0.7850, 0.7410, 0.6879, 0.5869, 0.4786, 0.3877, 0.2786, 0.0601],
        16: [1.0, 0.9693, 0.9532, 0.9204, 0.8822, 0.8484, 0.8254, 0.7990, 0.7737, 0.7256, 0.6591, 0.5380, 0.4354, 0.3775, 0.2786, 0.0376],
        20: [1.0, 0.9725, 0.9628, 0.9438, 0.9098, 0.8822, 0.8578, 0.8345, 0.8164, 0.7925, 0.7737, 0.7513, 0.6879, 0.6102, 0.5214, 0.4354, 0.3877, 0.3288, 0.2130, 0.0275],
        25: [1.0, 0.9755, 0.9678, 0.9585, 0.9438, 0.9258, 0.9020, 0.8822, 0.8660, 0.8484, 0.8345, 0.8164, 0.7990, 0.7850, 0.7685, 0.7410, 0.7008, 0.6524, 0.5869, 0.5214, 0.4432, 0.3877, 0.3284, 0.2528, 0.0195],
        30: [1.0, 0.9780, 0.9725, 0.9661, 0.9532, 0.9408, 0.9278, 0.9147, 0.9020, 0.8875, 0.8715, 0.8578, 0.8440, 0.8296, 0.8164, 0.8040, 0.7909, 0.7737, 0.7513, 0.7256, 0.6879, 0.6450, 0.5985, 0.5470, 0.4888, 0.4354, 0.3775, 0.3118, 0.2130, 0.0150]
    }

    # ==========================================
    # 5. ReAL Sampler (arXiv:2609.33202, 2026)
    # ==========================================
    # Conservative safe jump span K for Anima (16-20 steps)
    REAL_DEFAULT_K: Final[int] = 4
    REAL_DEFAULT_Q_LOC: Final[float] = 0.70
    REAL_DEFAULT_Q_SEG: Final[float] = 1.25
    REAL_DEFAULT_MIN_K: Final[int] = 3
    # Fallback-only prevents Carry Error on short-step models
    REAL_DEFAULT_CARRY_POLICY: Final[str] = "fallback_only"
    # Sigmas below this are solved densely to protect color & fine texture
    REAL_TAIL_PROTECT_SIGMA: Final[float] = 0.15

    # (TaylorSeer's defaults and polynomial coefficients live in enhancers/taylor_seer.py)

    # ==========================================
    # 6. CFG Early-Stop Guider
    # ==========================================
    CFG_EARLY_STOP_DEFAULT_PERCENT: Final[float] = 0.65
    CFG_EARLY_STOP_DEFAULT_MODE: Final[str] = "hard_stop"

    # ==========================================
    # 7. USDU Aspect-Ratio Tile Calculator
    # ==========================================
    USDU_TILE_MULTIPLE: Final[int] = 64
    USDU_PADDING_RATIO: Final[float] = 0.15
    USDU_PADDING_MULTIPLE: Final[int] = 8
    USDU_MIN_PADDING_BOUND: Final[int] = 4    # 4 * 8 = 32
    USDU_MAX_PADDING_BOUND: Final[int] = 20   # 20 * 8 = 160
    USDU_BLUR_MULTIPLE: Final[int] = 4
    USDU_DEFAULT_SKIP_THRESHOLD: Final[float] = 1.0
    USDU_DEFAULT_HIGHPASS_NOISE: Final[float] = 0.6
