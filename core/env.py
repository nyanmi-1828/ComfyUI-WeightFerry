# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
All WEIGHTFERRY_* environment variables the package reads, with their defaults in one table.
`get(name)`: the environment variable, else the settings file's value, else the default. Each call site reads its
variable at its own time (import / load / per call); an unknown name raises KeyError (catches typos). Meanings:
docs/ENVIRONMENT.md.

Settings file (2026-10-12): `<ComfyUI user directory>/weightferry.env` (or the path in WEIGHTFERRY_SETTINGS_FILE; 0 =
none), lines `NAME=value`, `#` comments; read once per process. For installs where environment variables are awkward
to set (Comfy Desktop); a real environment variable still wins.
"""

import os

DEFAULTS = {
    "WEIGHTFERRY_FUSED_BLOCK": "1",
    "WEIGHTFERRY_LORA_REMAP": "1",
    "WEIGHTFERRY_OVERFLOW_SLACK_MB": "384",
    "WEIGHTFERRY_SAGE_PV": "auto",
    "WEIGHTFERRY_LLM_COPY_THREADS": "4",
    "WEIGHTFERRY_LLM_STAGING": "3",
    "WEIGHTFERRY_ACT_CHUNK": "auto",
    "WEIGHTFERRY_ACT_CHUNK_MIN_TOKENS": "0",
    "WEIGHTFERRY_ACT_CHUNK_ROWS": "2048",
    "WEIGHTFERRY_ACT_H2D_STREAM": "1",
    "WEIGHTFERRY_ACT_D2H_STREAM": "1",
    "WEIGHTFERRY_ACT_HOST": "auto",
    "WEIGHTFERRY_ACT_KV_ONCE": "1",
    "WEIGHTFERRY_ACT_PER_ITEM": "0",
    "WEIGHTFERRY_ACT_RAM_RESERVE": "1",
    "WEIGHTFERRY_ATTN_CHUNK": "auto",
    "WEIGHTFERRY_ATTN_CHUNK_ROWS": "1024",
    "WEIGHTFERRY_AUTO": None,
    "WEIGHTFERRY_AUTO_HEADROOM_MB": "1536",
    "WEIGHTFERRY_CHUNK": None,
    "WEIGHTFERRY_CHUNK_BLOCK0": "1",
    "WEIGHTFERRY_CHUNK_LOG": None,
    "WEIGHTFERRY_CHUNK_MB": "96",
    "WEIGHTFERRY_CHUNK_PROFILE": None,
    "WEIGHTFERRY_CHUNK_RECALIBRATE": None,
    "WEIGHTFERRY_CHUNK_SLOTS": "4",
    "WEIGHTFERRY_DIRECT_FILE": "1",
    "WEIGHTFERRY_DIRECT_LORA": "1",
    "WEIGHTFERRY_DISK_DIR": None,
    "WEIGHTFERRY_DISK_THREADS": None,
    "WEIGHTFERRY_GGUF_KEEP_MMAP": "1",
    "WEIGHTFERRY_GGUF_TRITON": "1",
    "WEIGHTFERRY_GGUF_LORA_FACTORS": "pin",
    "WEIGHTFERRY_GGUF_KQUANTS": "1",
    "WEIGHTFERRY_MLP_CHUNK": "auto",
    "WEIGHTFERRY_MLP_CHUNK_ROWS": "1024",
    "WEIGHTFERRY_OUTSIDE_LEAF": "1",
    "WEIGHTFERRY_OVERFLOW_CHECK": "1",
    "WEIGHTFERRY_PERSIST_GB": "0",
    "WEIGHTFERRY_PREFETCH_PRIORITY": None,
    "WEIGHTFERRY_PROFILE": None,
    "WEIGHTFERRY_QUANT_LORA": "auto",
    "WEIGHTFERRY_QUANT_LORA_FORWARDS": "60",
    "WEIGHTFERRY_QUANT_LORA_MS_PER_MB": "0.23",
    "WEIGHTFERRY_QUANT_LORA_VRAM_MB": None,
    "WEIGHTFERRY_QUANT_SELF": "1",
    "WEIGHTFERRY_RAM_BUDGET_GB": None,
    "WEIGHTFERRY_RAM_MARGIN_GB": "6",
    "WEIGHTFERRY_RAM_REPLAN": None,
    "WEIGHTFERRY_RAM_REPLAN_HYST_GB": "1.5",
    "WEIGHTFERRY_RELEASE_TE": None,
    "WEIGHTFERRY_SLOTS": None,
    "WEIGHTFERRY_SSD_ASYNC": "1",
    "WEIGHTFERRY_SSD_LAYOUT": "interleave",
    "WEIGHTFERRY_SSD_WRITE_GBPS": "0.3",
    "WEIGHTFERRY_TE_RESERVE_GB": "3",
    "WEIGHTFERRY_USDU_LOCAL_COMPOSITE": "1",
    "WEIGHTFERRY_UPSCALE_RRDB_APPEND": "1",
    "WEIGHTFERRY_UPSCALE_RRDB_EPILOGUE": "1",
    "WEIGHTFERRY_UPSCALE_RRDB_RESIDUAL": "0",
    "WEIGHTFERRY_TE_CACHE_GB": "1",
    "WEIGHTFERRY_TIERING": None,
    "WEIGHTFERRY_TRIM_WS": None,
    "WEIGHTFERRY_VIRTUALALLOC": "1",
    "WEIGHTFERRY_VRAM_HEADROOM_MB": "1024",
    "WEIGHTFERRY_TAYLOR_CACHE_CPU_MIN_TOKENS": "0",
    "WEIGHTFERRY_TAYLOR_DEBUG": None,
    "WEIGHTFERRY_USDU_UPSCALE_DTYPE": "float16",
    "WEIGHTFERRY_SETTINGS_FILE": None,
}

_file = {"values": None}


def _settings_path():
    spec = os.environ.get("WEIGHTFERRY_SETTINGS_FILE")
    if spec is not None:
        return None if spec.strip().lower() in ("", "0", "off", "false", "none") else spec
    try:
        import folder_paths
        return os.path.join(folder_paths.get_user_directory(), "weightferry.env")
    except Exception:  # noqa: BLE001 -- outside ComfyUI: no settings file
        return None


def file_values():
    """The settings file's WEIGHTFERRY_* values (read once; {} without a file)."""
    if _file["values"] is None:
        vals = {}
        path = _settings_path()
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except OSError:
                lines = []
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = (x.strip() for x in s.split("=", 1))
                if k in DEFAULTS and k != "WEIGHTFERRY_SETTINGS_FILE":
                    vals[k] = v
        _file["values"] = vals
    return _file["values"]


def get(name):
    """The environment variable, else the settings file's value, else DEFAULTS[name]."""
    default = DEFAULTS[name]
    if name in os.environ:
        return os.environ[name]
    return file_values().get(name, default)
