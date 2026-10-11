# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
ComfyUI-WeightFerry: Unified Logging Module
===========================================
Provides standardized, structured console and logger outputs across all submodules.

Each message goes through the "WeightFerry" logger only, so ComfyUI's root handlers show it once: on the console
(coloured by ComfyUI's formatter), in ComfyUI's log buffer and in its log file. Without any handler (a plain script
outside ComfyUI) the message is printed instead.
"""

import logging

LOGGER_NAME = "WeightFerry"
logger = logging.getLogger(LOGGER_NAME)

PREFIX = "[WeightFerry]"


def _emit(level: int, message: str, color: str, ansi: str) -> None:
    if logger.hasHandlers():
        logger.log(level, f"{PREFIX} {message}", extra={"color": color})
    else:
        tag = " [WARN]" if level >= logging.WARNING else ""
        print(f"{ansi}{PREFIX}{tag} {message}\033[0m")


def log_info(message: str) -> None:
    """Green info message (ComfyUI's console, log buffer and log file)."""
    _emit(logging.INFO, message, "green", "\033[92m")


def log_warn(message: str) -> None:
    """Yellow warning message (ComfyUI's console, log buffer and log file)."""
    _emit(logging.WARNING, message, "yellow", "\033[93m")
