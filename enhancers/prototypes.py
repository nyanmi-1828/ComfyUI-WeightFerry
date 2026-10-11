# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
PROTOTYPES -- measured and NOT adopted (kept for reference and re-testing; every node is off by default: with mode
"off" it returns the model unchanged). Results and reasons:

  WeightFerryCFGCache               cheaper negative (uncond) branch: reuse its last difference to the
                              positive, or compute it at half resolution. Rejected: the base image breaks, the
                              upscale gains only ~5%.
  WeightFerryCoarseGlobalAttention  long self-attention as windows: each window's tokens attend to the
                              window at full resolution plus the whole image pooled 2x2. Rejected: slower, seams.
  WeightFerryTGATE                  the cross-attention output reused from a step on. Rejected: with
                              TaylorSeer + CFG early stop the base is not faster and changes; face detail -7%
                              (under 1% of a whole image).
"""


import functools
import logging
import math

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.samplers


_CGA_KEY = "weightferry_coarse_global_attention"


# --------------------------------------------------------------------------------------------------------- CFG cache
def _pool2(t):
    """2x2 average pool over the last two dims (any leading dims; odd sizes keep the last row / column)."""
    lead = t.shape[:-2]
    x = t.reshape(-1, 1, *t.shape[-2:]).float()
    return F.avg_pool2d(x, 2, ceil_mode=True).reshape(*lead, *[math.ceil(s / 2) for s in t.shape[-2:]]).to(t.dtype)


def _up_to(t, size):
    lead = t.shape[:-2]
    x = t.reshape(-1, 1, *t.shape[-2:]).float()
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).reshape(*lead, *size).to(t.dtype)


class WeightFerryCFGCache:
    """A cheaper negative branch on CFG steps (only where both branches would run; CFG 1 steps are untouched).
    reuse: the negative is computed on the first `first_full` calls and then every `interval`-th call (or, with
      thresh > 0, when the input changed by more than thresh since -- relative L1, accumulated); in between it is the
      positive plus the last computed (negative - positive).
    lowres: after `first_full` calls the negative runs on the input pooled 2x2 and is upsampled; the high frequencies
      come from the positive: neg = up(neg_low) + (pos - up(pool(pos)))."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["off", "reuse", "lowres"], {"default": "off"}),
            "first_full": ("INT", {"default": 2, "min": 0, "max": 100,
                                   "tooltip": "model calls with both branches computed at the start of each sampling"}),
            "interval": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "reuse: compute the negative every n-th call"}),
            "thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.005,
                                 "tooltip": "reuse: > 0 = adaptive (input change since the last computed negative)"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "WeightFerry/prototype (not adopted)"

    def apply(self, model, mode="off", first_full=2, interval=2, thresh=0.0):
        if mode == "off":
            return (model,)
        m = model.clone()
        prev_fn = m.model_options.get("sampler_calc_cond_batch_function")
        st = {"sigma": None, "calls": 0, "delta": None, "x_full": None, "acc": 0.0, "x_prev": None,
              "stats": {"full": 0, "cheap": 0}}

        def full(args):
            if prev_fn is not None:
                return prev_fn(args)
            return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"],
                                                  args["model_options"])

        def fn(args):
            conds, x, sigma = args["conds"], args["input"], args["sigma"]
            if len(conds) != 2 or conds[1] is None or conds[0] is None:
                return full(args)
            s = float(sigma.detach().max())
            if st["sigma"] is None or s > st["sigma"] + 1e-6:  # a new sampling (sigmas only fall within one)
                if st["calls"]:
                    logging.info("[WeightFerryCFGCache] %s: negative computed %d, cheap %d", mode, st["stats"]["full"],
                                 st["stats"]["cheap"])
                st.update(calls=0, delta=None, x_full=None, acc=0.0, x_prev=None, stats={"full": 0, "cheap": 0})
            st["sigma"] = s
            st["calls"] += 1
            if mode == "reuse":
                need = st["calls"] <= first_full or st["delta"] is None or st["delta"].shape != x.shape
                if not need:
                    if thresh > 0:
                        prev = st["x_prev"]
                        st["acc"] += float(((x - prev).abs().mean() / prev.abs().mean().clamp_min(1e-12)).item())
                        need = st["acc"] >= thresh
                    else:
                        need = (st["calls"] - first_full) % interval == 0
                st["x_prev"] = x
                if need:
                    out = full(args)
                    st["delta"] = out[1] - out[0]
                    st["acc"] = 0.0
                    st["stats"]["full"] += 1
                    return out
                pos = full(dict(args, conds=[conds[0]]))[0]
                st["stats"]["cheap"] += 1
                return [pos, pos + st["delta"]]
            # lowres
            if st["calls"] <= first_full:
                st["stats"]["full"] += 1
                return full(args)
            pos = full(dict(args, conds=[conds[0]]))[0]
            neg_low = full(dict(args, conds=[conds[1]], input=_pool2(x)))[0]
            size = tuple(x.shape[-2:])
            neg = _up_to(neg_low, size) + (pos - _up_to(_pool2(pos), size))
            st["stats"]["cheap"] += 1
            return [pos, neg]

        m.model_options["sampler_calc_cond_batch_function"] = fn
        return (m,)


# ------------------------------------------------------------------------------------------- coarse global attention
def _cga_attention(cfg, func, q, k, v, heads, *args, **kwargs):
    grid = cfg.get("grid")
    if (grid is None or q.dim() != 4 or not kwargs.get("skip_reshape") or kwargs.get("skip_output_reshape")
            or kwargs.get("mask") is not None or q.shape[2] != k.shape[2] or q.shape[2] < cfg["min_tokens"]):
        return func(q, k, v, heads, *args, **kwargs)
    T, Hp, Wp = grid
    B, Hh, S, D = q.shape
    if T != 1 or Hp * Wp != S:
        return func(q, k, v, heads, *args, **kwargs)
    qg, kg, vg = (t.reshape(B, Hh, Hp, Wp, D) for t in (q, k, v))

    def pool(t):  # [B, Hh, Hp, Wp, D] -> [B, Hh, P, D]
        x = t.permute(0, 1, 4, 2, 3).reshape(B * Hh, D, Hp, Wp).float()
        x = F.avg_pool2d(x, cfg["pool"], ceil_mode=True)
        return x.reshape(B, Hh, D, -1).permute(0, 1, 3, 2).to(t.dtype)
    kp, vp = pool(kg), pool(vg)
    n = cfg["windows"]
    hb = [round(i * Hp / n) for i in range(n + 1)]
    wb = [round(i * Wp / n) for i in range(n + 1)]
    out = q.new_empty(B, Hp, Wp, Hh * D)
    bias = math.log(cfg["pool"] ** 2) if cfg["kernel"] == "sdpa_bias" else None
    for i in range(n):
        for j in range(n):
            h0, h1, w0, w1 = hb[i], hb[i + 1], wb[j], wb[j + 1]
            if h1 <= h0 or w1 <= w0:
                continue
            qw = qg[:, :, h0:h1, w0:w1].reshape(B, Hh, -1, D)
            kw = torch.cat([kg[:, :, h0:h1, w0:w1].reshape(B, Hh, -1, D), kp], dim=2)
            vw = torch.cat([vg[:, :, h0:h1, w0:w1].reshape(B, Hh, -1, D), vp], dim=2)
            if bias is None:
                o = func(qw.contiguous(), kw.contiguous(), vw.contiguous(), heads, *args, **kwargs)
            else:  # a pooled key stands for pool^2 tokens: + log(pool^2) on its logits
                nl = (h1 - h0) * (w1 - w0)  # the block's own keys
                mask = torch.zeros(1, 1, 1, kw.shape[2], device=q.device, dtype=q.dtype)
                mask[..., nl:] = bias
                o = F.scaled_dot_product_attention(qw, kw, vw, attn_mask=mask)
                o = o.transpose(1, 2).reshape(B, -1, Hh * D)
            out[:, h0:h1, w0:w1] = o.reshape(B, h1 - h0, w1 - w0, Hh * D)
    return out.reshape(B, S, Hh * D)


class WeightFerryCoarseGlobalAttention:
    """Self-attention with at least `min_tokens` keys as windows x windows blocks: a block's queries attend to the
    block's own keys at full resolution and to all keys pooled pool x pool (cost ~ S * (S / windows^2 + S / pool^2)
    instead of S * S). Cross-attention and short sequences are unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["off", "window_global"], {"default": "off"}),
            "windows": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "blocks per axis"}),
            "pool": ("INT", {"default": 2, "min": 2, "max": 8}),
            "min_tokens": ("INT", {"default": 12000, "min": 0, "max": 1000000,
                                   "tooltip": "only self-attention with at least this many tokens (896x1472 = 5152)"}),
            "kernel": (["attention", "sdpa_bias"], {"default": "attention", "tooltip":
                       "attention: the current attention function (e.g. SageAttention). sdpa_bias: PyTorch "
                       "SDPA with + log(pool^2) on the pooled keys (slower, for the quality comparison)"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "WeightFerry/prototype (not adopted)"

    def apply(self, model, mode="off", windows=4, pool=2, min_tokens=12000, kernel="attention"):
        if mode == "off":
            return (model,)
        m = model.clone()
        cfg = {"windows": int(windows), "pool": int(pool), "min_tokens": int(min_tokens), "kernel": kernel}

        def wrapper(executor, *args, **kwargs):
            dm = executor.class_obj
            x = args[0] if args else None
            to = kwargs.get("transformer_options")
            if x is None or to is None or x.dim() not in (4, 5):
                return executor(*args, **kwargs)
            ps, pt = getattr(dm, "patch_spatial", 2), getattr(dm, "patch_temporal", 1)
            T = math.ceil(x.shape[2] / pt) if x.dim() == 5 else 1
            cfg["grid"] = (T, math.ceil(x.shape[-2] / ps), math.ceil(x.shape[-1] / ps))
            kwargs["transformer_options"] = dict(to, optimized_attention_override=functools.partial(_cga_attention, cfg))
            return executor(*args, **kwargs)

        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _CGA_KEY,
                                                     wrapper, m.model_options, is_model_options=True)
        return (m,)


# ---------------------------------------------------------------------------------------------------------------- TGATE
_TGATE_KEY = "weightferry_tgate"


def _tgate_step(to):
    """(step index, schedule key) of this model call from transformer_options' sigmas (as TaylorSeer), or None."""
    sig, sample = to.get("sigmas"), to.get("sample_sigmas")
    if not torch.is_tensor(sig) or not torch.is_tensor(sample) or sample.numel() < 2:
        return None
    s = sample.detach().flatten().float()
    cur = sig.detach().flatten()[0].float().to(s.device)
    return int((s[:-1] - cur).abs().argmin().item()), tuple(s.cpu().tolist())


def _tgate_install(dm):
    """Wraps every block's cross_attn.forward once per diffusion model; the wrapper does nothing unless a TGATE state
    is active on the model (set by WeightFerryTGATE's DIFFUSION_MODEL wrapper for the duration of one model call)."""
    if getattr(dm, "_weightferry_tgate_installed", False):
        return
    for blk in getattr(dm, "blocks", []):
        ca = getattr(blk, "cross_attn", None)
        if ca is None:
            continue
        orig = ca.forward

        def fwd(x, context=None, *args, _orig=orig, **kwargs):
            st = getattr(dm, "_weightferry_tgate_active", None)
            to = kwargs.get("transformer_options") or {}
            idx = to.get("block_index")
            # whole images only: a row-chunked block (activation streaming) calls cross_attn per row chunk
            if st is None or idx is None or x.dim() != 3 or x.shape[1] != st.get("tokens"):
                return _orig(x, context, *args, **kwargs)
            labels = list(to.get("cond_or_uncond") or [0])
            if x.shape[0] % len(labels):
                return _orig(x, context, *args, **kwargs)
            per = x.shape[0] // len(labels)
            cache = st["cache"]
            keys = [(idx, lab) for lab in labels]
            if st["step"] >= st["gate"] and all(k in cache and cache[k].shape[1:] == x.shape[1:]
                                                and cache[k].shape[0] == per for k in keys):
                st["reused"] += 1
                outs = [cache[k].to(device=x.device, non_blocking=True) for k in keys]
                return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
            out = _orig(x, context, *args, **kwargs)
            st["computed"] += 1
            for i, k in enumerate(keys):  # the latest computed result per block and branch
                cache[k] = out[i * per:(i + 1) * per].detach().to(st["device"], non_blocking=True)
            return out
        ca.forward = fwd
    dm._weightferry_tgate_installed = True


class WeightFerryTGATE:
    """TGATE (experiment): from sampling step `gate_step` on, each block's cross-attention output
    is the one of the last computed step (per block and cond / uncond branch) instead of being computed again; the
    adaLN gate of the current step still scales it. Steps are counted per sampling like TaylorSeer (from the sigmas), so
    the face detail / upscale samplings start over. The cache holds one output per block and branch (896x1472: about
    42 MB per block for cond + uncond, 1.2 GB for 28 blocks; cache_device cpu keeps it in RAM)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["off", "on"], {"default": "off"}),
            "gate_step": ("INT", {"default": 6, "min": 1, "max": 1000,
                                  "tooltip": "from this step index on the cross-attention output is reused"}),
            "cache_device": (["cuda", "cpu"], {"default": "cuda"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "WeightFerry/prototype (not adopted)"

    def apply(self, model, mode="off", gate_step=6, cache_device="cuda"):
        if mode == "off":
            return (model,)
        m = model.clone()
        st = {"gate": int(gate_step), "device": torch.device(cache_device), "cache": {}, "schedule": None, "last": None,
              "step": 0, "reused": 0, "computed": 0}

        def wrapper(executor, *args, **kwargs):
            dm = executor.class_obj
            to = kwargs.get("transformer_options") or {}
            info = _tgate_step(to)
            if info is None or not hasattr(dm, "blocks"):
                return executor(*args, **kwargs)
            step, sched = info
            if sched != st["schedule"] or (st["last"] is not None and step < st["last"]):
                if st["schedule"] is not None and (st["reused"] or st["computed"]):
                    logging.info("[WeightFerryTGATE] gate %d: cross-attention reused %d, computed %d", st["gate"],
                                 st["reused"], st["computed"])
                st.update(cache={}, schedule=sched, reused=0, computed=0)
            st["last"] = step
            st["step"] = step
            x = args[0] if args else None
            if x is not None and x.dim() == 5:
                ps, pt = getattr(dm, "patch_spatial", 2), getattr(dm, "patch_temporal", 1)
                st["tokens"] = math.ceil(x.shape[2] / pt) * math.ceil(x.shape[3] / ps) * math.ceil(x.shape[4] / ps)
            _tgate_install(dm)
            dm._weightferry_tgate_active = st
            try:
                return executor(*args, **kwargs)
            finally:
                dm._weightferry_tgate_active = None

        comfy.patcher_extension.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _TGATE_KEY,
                                                     wrapper, m.model_options, is_model_options=True)
        return (m,)
