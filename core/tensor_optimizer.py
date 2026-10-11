# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Process-wide settings applied at import: the Windows 1 ms timer resolution, TF32 for float32 matmul / convolution,
and a persistent Triton kernel cache directory under ComfyUI's base directory.
"""

import os
import atexit
import torch

from ..config import WeightFerryConfig
from . import env as _env
from ..utils.logging import log_info, log_warn


def _default_triton_cache() -> str:
    """<ComfyUI base directory (--base-directory)>/.cache/triton; the directory above custom_nodes if ComfyUI's
    folder_paths is not importable (e.g. a test outside ComfyUI)."""
    try:
        import folder_paths
        base = folder_paths.base_path
    except (ImportError, AttributeError):  # outside ComfyUI (scripts, tests)
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(base, WeightFerryConfig.TRITON_CACHE_SUBDIR)


def setup_low_level_optimizations() -> None:
    """
    1. Win32 timeBeginPeriod(1): 1 ms sleep resolution.
    2. TRITON_CACHE_DIR set to a directory that survives restarts (unless already set).
    3. TF32 for float32 matmul / convolution on Ampere and newer; cuDNN autotuning left to ComfyUI.
    """
    # 1. Windows Multimedia Timer (1.0ms / 1000Hz resolution)
    if os.name == "nt":
        try:
            import ctypes
            winmm = ctypes.windll.winmm
            res_ms = WeightFerryConfig.WIN32_TIMER_RESOLUTION_MS
            if winmm.timeBeginPeriod(res_ms) == 0:
                atexit.register(lambda: winmm.timeEndPeriod(res_ms))
                log_info(f"Windows high-resolution multimedia timer locked to {res_ms}.0ms")
        except Exception as e:  # noqa: BLE001 -- optional: the default timer still works
            log_warn(f"Failed to set Windows multimedia timer: {e}")

    # 2. Persistent Triton kernel cache
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or _default_triton_cache()
    try:
        os.makedirs(triton_cache, exist_ok=True)
        if os.environ.get("TRITON_CACHE_DIR") != triton_cache:
            os.environ["TRITON_CACHE_DIR"] = triton_cache
        log_info(f"Triton kernel cache: {triton_cache}")
    except Exception as e:  # noqa: BLE001 -- optional: Triton then uses its own default cache
        log_warn(f"Failed to create Triton cache directory: {e}")

    # 3. TF32; cuDNN autotuning
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision(WeightFerryConfig.FLOAT32_MATMUL_PRECISION)
        except Exception:  # noqa: BLE001 -- older PyTorch: TF32 flags above are enough
            pass

        # cuDNN autotuning is left to ComfyUI (off unless --fast autotune). ComfyUI re-applies its policy after loading
        # the custom nodes, so a setting made here at import would not last. Measured on an RTX 4060: the face detail's
        # crop size changes with every image, and with autotuning each new size cost ~1.2 s of tuning per face, while
        # tuned and untuned ran equally fast
        log_info("TensorCore TF32 activated (matmul.allow_tf32=True)")
