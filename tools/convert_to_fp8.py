# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Convert a bf16/fp16 DiT safetensors file to plain fp8 e4m3fn weights (no scales), one tensor at a time.

2-D tensors (Linear weights) become float8_e4m3fn with round-to-nearest (torch .to(), the same conversion as
UNETLoader's "fp8_e4m3fn" weight dtype); 1-D tensors (norm weights, biases) stay in their dtype. A tensor whose
values exceed the fp8 e4m3fn range (|x| > 448, which would turn into NaN) is kept as is and reported. Parameters are written in
sorted key order (non-parameter tensors last), so each transformer block remains one contiguous byte range (needed by the streaming node's direct
file source). Writing needs one tensor of RAM at a time.

  python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_fp8.py models/diffusion_models/MODEL.safetensors [--scaled]
  (writes ..._fp8_e4m3fn.safetensors, or ..._fp8_scaled.safetensors with --scaled, next to it; use ComfyUI's Python)
"""

import argparse
import json
import math
import os
import struct
import time

FP8_MAX = 448.0
ST = {"BF16": "bf16", "F16": "fp16", "F32": "fp32", "F8_E4M3": "fp8"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="bf16 / fp16 / fp32 .safetensors model")
    p.add_argument("--out", default=None, help="output file (default: <src>_fp8_scaled.safetensors with --scaled, else <src>_fp8_e4m3fn.safetensors)")
    p.add_argument("--scaled", action="store_true",
                   help="Scaled fp8 in ComfyUI's comfy_quant format: per-tensor scale = max|w| / 448 stored as "
                        "<layer>.weight_scale (fp32) plus <layer>.comfy_quant = {\"format\": \"float8_e4m3fn\"}. "
                        "Load with UNETLoader weight dtype 'default' (ComfyUI builds QuantizedTensor weights). Scales and "
                        "configs go to the end of the file so each block's weights stay one contiguous range")
    p.add_argument("--keep-vanishing", action="store_true",
                   help="Keep (in every block) tensors that have rows rounding entirely to zero in fp8. On Cosmos-"
                        "Predict2 14B nearly every tensor qualifies (23.8 GB), so this is off by default")
    cli = p.parse_args()
    out = cli.out or os.path.splitext(cli.src)[0] + ("_fp8_scaled.safetensors" if cli.scaled
                                                      else "_fp8_e4m3fn.safetensors")
    import torch
    from safetensors import safe_open

    with open(cli.src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    meta = hdr.pop("__metadata__", {}) or {}
    # Sorted keys keep each block's parameters together. Non-parameter tensors (integer dtypes such as uint8
    # "_extra_state" or training counters) go to the end, like in the original Cosmos files: a few odd-sized bytes
    # between a block's weights would break the block's contiguity and the alignment of the tensors after them.
    def is_param(k):
        return hdr[k]["dtype"] in ("BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2") and not k.endswith("_extra_state")
    keys = sorted(k for k in hdr if is_param(k)) + sorted(k for k in hdr if not is_param(k))
    itemsize = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "BOOL": 1,
                "I16": 2, "I32": 4, "I64": 8}

    with safe_open(cli.src, "pt") as src:
        # Pass 1: decide each tensor's output dtype (the checks need the values)
        #  * out of range: |x| > 448 would become NaN
        #  * vanishing rows: an output row whose weights are all below fp8's resolution (~0.001) becomes exactly zero.
        #    The layer's output channel is then zero for every input -- information lost, and SageAttention's fp8 PV
        #    kernel divides 0 by a zero per-channel scale (NaN) when that layer produces V (found:
        #    Cosmos-Predict2 14B, blocks.0.cross_attn.v_proj had 183 of 5120 rows vanish, max |w| 0.00098).
        #  A tensor kept in its dtype is kept for every block (same name with any block index), so all blocks keep one
        #  layout (the streaming node's direct file source needs that).
        import re as _re

        def pattern(k):
            return _re.sub(r"\.(blocks|transformer_blocks|layers)\.\d+\.", lambda m: "." + m.group(1) + ".*.", k)

        plan, kept, reasons, errors, scales = {}, set(), {}, [], {}
        for k in keys:
            info = hdr[k]
            if cli.scaled and len(info["shape"]) == 2 and info["dtype"] in ("BF16", "F16", "F32"):
                if not k.endswith(".weight"):
                    kept.add(pattern(k))
                    reasons.setdefault(pattern(k), "2-D but not a .weight (no comfy_quant layer)")
                    continue
                if info["shape"][0] % 16 or info["shape"][1] % 16:
                    # fp8 matmul (ComfyUI's comfy_kitchen) needs both dims divisible by 16, e.g. x_embedder 5120x68
                    kept.add(pattern(k))
                    reasons.setdefault(pattern(k), f"shape {info['shape']} not divisible by 16 (fp8 matmul)")
                    continue
                tf = src.get_tensor(k).float()
                amax = float(tf.abs().max())
                scale = torch.tensor(amax / FP8_MAX if amax > 0 else 1.0, dtype=torch.float32)
                q = (tf / scale).to(torch.float8_e4m3fn).float() * scale
                errors.append((float((q - tf).norm() / tf.norm().clamp_min(1e-30)), k))
                scales[k] = scale
                continue
            if len(info["shape"]) == 2 and info["dtype"] in ("BF16", "F16", "F32"):
                t = src.get_tensor(k)
                if float(t.abs().max()) > FP8_MAX:
                    kept.add(pattern(k))
                    reasons.setdefault(pattern(k), "out of fp8 range")
                    continue
                q = t.to(torch.float8_e4m3fn).float()
                tf = t.float()
                rel_err = float((q - tf).norm() / tf.norm().clamp_min(1e-30))
                errors.append((rel_err, k))
                vanished = (q.abs().amax(dim=1) == 0) & (tf.abs().amax(dim=1) > 0)
                if bool(vanished.any()) and cli.keep_vanishing:
                    kept.add(pattern(k))
                    reasons.setdefault(pattern(k), f"{int(vanished.sum())} rows vanish in fp8 (e.g. {k})")
        for k in keys:
            info = hdr[k]
            if k in scales:
                plan[k] = "F8_E4M3"
            elif cli.scaled:
                plan[k] = info["dtype"]
            elif len(info["shape"]) == 2 and info["dtype"] in ("BF16", "F16", "F32") and pattern(k) not in kept:
                plan[k] = "F8_E4M3"
            else:
                plan[k] = info["dtype"]
        kept = sorted(kept)
        # Scaled mode: per-layer scale + config tensors, written after everything else
        extras = {}
        quant_conf = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
        for k in sorted(scales):
            base = k[:-len("weight")]
            extras[base + "weight_scale"] = ("F32", [], scales[k].numpy().tobytes())
            extras[base + "comfy_quant"] = ("U8", [len(quant_conf)], quant_conf)
        header, offset = {}, 0
        for k in keys:
            nbytes = itemsize[plan[k]] * math.prod(hdr[k]["shape"])
            header[k] = {"dtype": plan[k], "shape": hdr[k]["shape"], "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        for k, (dt, shape, data) in extras.items():
            header[k] = {"dtype": dt, "shape": shape, "data_offsets": [offset, offset + len(data)]}
            offset += len(data)
        meta = dict(meta)
        if cli.scaled:
            meta["converted_to_fp8_scaled"] = (f"from {os.path.basename(cli.src)}; per-tensor scale max|w|/448, "
                                               f"comfy_quant float8_e4m3fn; {len(scales)} layers")
        else:
            meta["converted_to_fp8_e4m3fn"] = (f"from {os.path.basename(cli.src)}; 2-D weights round-to-nearest, no "
                                               f"scales; kept (out of range / vanishing rows): {len(kept)} name patterns")
        header["__metadata__"] = {k: str(v) for k, v in meta.items()}
        hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hbytes += b" " * ((8 - len(hbytes) % 8) % 8)
        n_fp8 = sum(1 for k in keys if plan[k] == "F8_E4M3")
        print(f"[fp8] {len(keys)} tensors: {n_fp8} -> fp8, {len(keys) - n_fp8} kept; output "
              f"{(8 + len(hbytes) + offset) / 1e9:.2f} GB -> {out}", flush=True)
        for pat in kept:
            print(f"[fp8] kept in original dtype: {pat} ({reasons[pat]})", flush=True)
        # Plain (unscaled) fp8 resolves small weights badly: report the relative error ||q - w|| / ||w||
        converted_err = sorted(e for e in errors if plan[e[1]] == "F8_E4M3")
        if converted_err:
            mean_err = sum(e for e, _ in converted_err) / len(converted_err)
            print(f"[fp8] relative error of converted tensors: mean {mean_err:.3f}, worst "
                  f"{', '.join(f'{k} {e:.3f}' for e, k in converted_err[-3:][::-1])}", flush=True)
            if converted_err[-1][0] > 0.1 and not cli.scaled:
                print("[fp8] WARNING: plain fp8 loses a lot on this model (small weights below fp8's resolution); "
                      "a scaled fp8 format would be needed for fidelity", flush=True)
        # Pass 2: write
        t0 = time.perf_counter()
        tmp = out + ".partial"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(hbytes)))
            f.write(hbytes)
            for i, k in enumerate(keys):
                t = src.get_tensor(k)
                if k in scales:
                    t = (t.float() / scales[k]).to(torch.float8_e4m3fn)
                elif plan[k] == "F8_E4M3" and t.dtype != torch.float8_e4m3fn:
                    t = t.to(torch.float8_e4m3fn)
                data = t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
                assert len(data) == header[k]["data_offsets"][1] - header[k]["data_offsets"][0], k
                f.write(data)
                if i % 100 == 0:
                    print(f"[fp8] {i}/{len(keys)} tensors, {time.perf_counter() - t0:.0f} s", flush=True)
            for k, (_dt, _shape, data) in extras.items():
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"[fp8] done in {time.perf_counter() - t0:.0f} s: {out}", flush=True)


if __name__ == "__main__":
    main()
