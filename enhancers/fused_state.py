# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
"""Shared state of the fused block (enhancers/fused_block.py): the installed parts, the per-thread pending input /
shared quantisation / pending residual, and the per-part call counters. One object each, imported by fused_block
and fused_int8."""

import threading

_state = {"parts": (), "orig": None}
# calls per part: used / fell back to ComfyUI's code, with the first reason seen (stats())
_stats = {}
_tls = threading.local()


def _count(part, used, why=None):
    st = _stats.setdefault(part, {"used": 0, "fallback": 0, "why": None})
    if used:
        st["used"] += 1
    else:
        st["fallback"] += 1
        if st["why"] is None:
            st["why"] = why
