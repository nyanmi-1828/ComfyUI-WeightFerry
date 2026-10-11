# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Disk cache of text-encoder outputs (CLIP Text Encode (Streamed LLM), input disk_cache, off by default).

A cached prompt is not encoded again, also after a restart. The key needs no text encoder: the prompt text, the part
of the workflow that produces the CLIP (every upstream node's class and inputs, recursively, and size / modification
time of the model files they name) and the environment that decides the numbers (ComfyUI, PyTorch, GPU, text-encoder
dtype). So on a hit the node does not even request its CLIP input (a lazy input): when nothing else in the workflow
uses the CLIP, its loader never runs and the text encoder is never read from disk.

Entries are torch.save files of the conditioning (tensors, dicts, lists, numbers, strings only; anything else is not
cached), read back with weights_only=True. WEIGHTFERRY_TE_CACHE_GB caps the folder: the least recently used entries
are deleted above it. Folder: <ComfyUI base>/.cache/weightferry/te_cache (or WEIGHTFERRY_DISK_DIR/te_cache).
"""

import hashlib
import json
import os

import torch

from . import env as _env
from .tiers.cache_files import persistent_root
from ..utils.logging import log_info, log_warn

FORMAT = 1  # bump when the key or the stored layout changes


def cache_dir():
    return os.path.join(persistent_root(), "te_cache")


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def _file_stat(value, memo):
    """(size, mtime_ns) of the model file a string input names, or None. Every model folder is tried."""
    if value in memo:
        return memo[value]
    found = None
    try:
        import folder_paths
        for folder in list(folder_paths.folder_names_and_paths):
            try:
                path = folder_paths.get_full_path(folder, value)
            except Exception:  # noqa: BLE001 -- a folder kind that does not resolve names
                path = None
            if path and os.path.isfile(path):
                st = os.stat(path)
                found = [st.st_size, st.st_mtime_ns]
                break
    except ImportError:
        pass
    memo[value] = found
    return found


def upstream_signature(prompt, node_id, input_name):
    """A JSON-able description of what feeds ``input_name`` of ``node_id`` in the API prompt: the source nodes'
    class types and inputs, recursively, with the files they name. None if the prompt does not describe it."""
    files, done = {}, {}

    def sig(link, depth=0):
        if depth > 256:
            raise ValueError("graph too deep")
        src, out = link
        node = prompt.get(str(src))
        if node is None:
            raise KeyError(src)
        if str(src) not in done:
            items = []
            for k in sorted(node.get("inputs", {})):
                v = node["inputs"][k]
                if _is_link(v):
                    items.append([k, sig(v, depth + 1)])
                elif isinstance(v, str):
                    items.append([k, v, _file_stat(v, files)])
                else:
                    items.append([k, v])
            done[str(src)] = [node.get("class_type"), items]
        return [out, done[str(src)]]

    try:
        link = prompt[str(node_id)]["inputs"][input_name]
        return sig(link) if _is_link(link) else None
    except (KeyError, TypeError, ValueError):
        return None


def _environment():
    try:
        import comfyui_version
        comfy_version = comfyui_version.__version__
    except Exception:  # noqa: BLE001
        comfy_version = "?"
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
        te_dtype = str(mm.text_encoder_dtype(dev))
        gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    except Exception:  # noqa: BLE001
        te_dtype, gpu = "?", "?"
    return {"format": FORMAT, "comfyui": comfy_version, "torch": torch.__version__, "gpu": gpu, "te_dtype": te_dtype}


def make_key(text, prompt, node_id, input_name="clip"):
    """The cache key of encoding ``text`` with the CLIP that feeds ``input_name``, or None (not cacheable)."""
    if not isinstance(prompt, dict) or node_id is None:
        return None
    up = upstream_signature(prompt, node_id, input_name)
    if up is None:
        return None
    payload = {"env": _environment(), "text": text, "clip": up}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _plain(obj):
    if obj is None or isinstance(obj, (bool, int, float, str, torch.Tensor)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_plain(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _plain(v) for k, v in obj.items())
    return False


def load(key):
    """The cached conditioning for ``key`` (CPU tensors), or None."""
    if key is None:
        return None
    path = os.path.join(cache_dir(), key + ".pt")
    if not os.path.isfile(path):
        return None
    try:
        cond = torch.load(path, map_location="cpu", weights_only=True)
        os.utime(path)  # least recently used = oldest modification time
        return cond
    except Exception as exc:  # noqa: BLE001 -- unreadable entry: drop it, encode again
        log_warn(f"Text encode cache: entry {key[:12]} unreadable ({exc}); encoding again.")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def store(key, cond):
    """Write ``cond`` under ``key`` (if it holds plain data only) and keep the folder under the size cap."""
    if key is None:
        return
    if not _plain(cond):
        log_info("Text encode cache: this conditioning holds objects other than tensors / numbers / strings; "
                 "not cached.")
        return
    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, key + ".pt")
    tmp = path + ".tmp"
    torch.save(cond, tmp)
    os.replace(tmp, path)
    enforce_cap(d, float(_env.get("WEIGHTFERRY_TE_CACHE_GB")) * 1024 ** 3, keep=path)


def enforce_cap(directory, cap_bytes, keep=None):
    """Delete the least recently used entries until the folder holds at most cap_bytes (never ``keep``)."""
    try:
        entries = [os.path.join(directory, n) for n in os.listdir(directory) if n.endswith(".pt")]
    except OSError:
        return
    entries = [(os.stat(p).st_mtime, os.path.getsize(p), p) for p in entries if os.path.isfile(p)]
    total = sum(e[1] for e in entries)
    for _, size, p in sorted(entries):
        if total <= cap_bytes:
            break
        if keep is not None and os.path.abspath(p) == os.path.abspath(keep):
            continue
        try:
            os.remove(p)
            total -= size
        except OSError:
            pass
