# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
r"""
Convert a bf16 DiT to ComfyUI's native integer formats with ComfyUI's own quantizers
(comfy_kitchen layouts via QuantizedTensor.from_float), the same layers as an existing scaled-fp8 file (--like: every
layer that has a comfy_quant entry there; the rest is copied as is):
  int8          int8_tensorwise (W8A8, one scale per tensor)
  int8_channel  int8_tensorwise, one scale per output channel
  int8_convrot  int8_tensorwise per channel with ConvRot (rotated W8A8)
  w4a4          convrot_w4a4 (rotated 4-bit weights and activations)
Each quantized layer: weight + its scale tensors (QuantizedTensor.state_dict) + "comfy_quant" (the JSON ComfyUI reads).
  python tools/convert_quant.py SRC_bf16.safetensors --like FP8_SCALED.safetensors --format int8_convrot [--out PATH]
      [--comfy-root DIR]
The --like file says which layers to quantize: make it first with tools/convert_to_fp8.py --scaled.
Needs a CUDA GPU (the comfy_kitchen quantizers). Prints the relative weight error (median / max over layers).
The output is written tensor by tensor (two passes over the source: the layout, then the data), so the converted model
never has to fit in RAM; each layer's scale tensors and comfy_quant follow its weight.
"""

import argparse
import json
import os
import struct
import sys
import time

import torch
from safetensors import safe_open


def _add_comfy_to_path(root=None):
    """Make ComfyUI's code importable (comfy.quant_ops): --comfy-root, else COMFYUI_ROOT, else already importable, else
    the directory above custom_nodes/ (a plain ComfyUI checkout; the desktop app keeps its code elsewhere -- pass
    --comfy-root then)."""
    for cand in (root, os.environ.get("COMFYUI_ROOT")):
        if cand:
            sys.path.insert(0, cand)
            return
    try:
        import comfy.quant_ops  # noqa: F401
        return
    except ImportError:
        pass
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


FORMATS = {  # name -> (layout, quantize kwargs, comfy_quant json)
    "int8": ("TensorWiseINT8Layout", {}, {"format": "int8_tensorwise"}),
    "int8_channel": ("TensorWiseINT8Layout", {"per_channel": True}, {"format": "int8_tensorwise"}),
    "int8_convrot": ("TensorWiseINT8Layout", {"per_channel": True, "convrot": True, "convrot_groupsize": 256},
                     {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}),
    "w4a4": ("TensorCoreConvRotW4A4Layout", {"convrot_groupsize": 256},
             {"format": "convrot_w4a4", "convrot_groupsize": 256}),
}


_ST_DTYPE = {torch.int8: "I8", torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16",
             torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2", torch.int32: "I32", torch.int64: "I64",
             torch.bool: "BOOL", torch.float64: "F64", torch.int16: "I16"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="bf16 / fp16 / fp32 .safetensors model")
    ap.add_argument("--like", required=True, help="a scaled-fp8 file of the same model (convert_to_fp8.py --scaled): "
                    "the layers that have a comfy_quant entry there are quantized")
    ap.add_argument("--format", required=True, choices=sorted(FORMATS), help="quantized layout (see above)")
    ap.add_argument("--out", default=None, help="output file (default: <src>_<format>.safetensors)")
    ap.add_argument("--comfy-root", default=None, help="directory with ComfyUI's code (the one containing comfy/)")
    cli = ap.parse_args()
    _add_comfy_to_path(cli.comfy_root)
    from comfy.quant_ops import QuantizedTensor
    layout, kw, conf = FORMATS[cli.format]
    out = cli.out or os.path.splitext(cli.src)[0] + f"_{cli.format}.safetensors"
    with safe_open(cli.like, "pt") as f:
        targets = {k[:-len("comfy_quant")] for k in f.keys() if k.endswith("comfy_quant")}
    if not targets:
        raise SystemExit(f"--like {cli.like}: no quantized layers (no '.comfy_quant' keys); use a scaled-fp8 file made "
                         "with convert_to_fp8.py --scaled")
    conf_t = torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)
    t0 = time.time()

    def converted(f, k):
        """The output tensors for source key k, in file order (a quantized weight: its state_dict + comfy_quant), and
        the relative weight error of a quantized one."""
        t = f.get_tensor(k)
        prefix = k[:-len("weight")] if k.endswith("weight") else None
        # Linear weights only: an embedding table (llm_adapter.embed) is not a Linear -- its W4A4 packing halves a
        # dimension and ComfyUI's Embedding cannot load it (int8 keeps the shape and loads)
        if prefix is not None and prefix in targets and t.dim() == 2 and not (cli.format == "w4a4" and "embed" in k):
            w = t.to("cuda", torch.bfloat16)
            qt = QuantizedTensor.from_float(w, layout, **kw)
            items = [(kk, v.detach().to("cpu").contiguous()) for kk, v in qt.state_dict(k).items()]
            items.append((prefix + "comfy_quant", conf_t.clone()))
            err = ((qt.dequantize().float() - w.float()).norm() / w.float().norm().clamp_min(1e-12)).item()
            return items, err
        return [(k, t.contiguous())], None

    with safe_open(cli.src, "pt") as f:
        meta = dict(f.metadata() or {})
        keys = list(f.keys())
        # Pass 1: the output layout (names, dtypes, shapes) and the errors; the data is dropped
        header, offset, errs = {}, 0, []
        for i, k in enumerate(keys):
            items, err = converted(f, k)
            if err is not None:
                errs.append(err)
            for kk, v in items:
                nbytes = v.element_size() * v.numel()
                header[kk] = {"dtype": _ST_DTYPE[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + nbytes]}
                offset += nbytes
            if i % 200 == 0:
                print(f"[quant] layout {i}/{len(keys)} tensors, {time.time() - t0:.0f} s", flush=True)
        meta["converted_to"] = f"{cli.format} from {os.path.basename(cli.src)}; layers as {os.path.basename(cli.like)}"
        header["__metadata__"] = {k: str(v) for k, v in meta.items()}
        hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        hbytes += b" " * ((8 - len(hbytes) % 8) % 8)
        # Pass 2: the data, in the same order (the quantizers are deterministic; every size is checked)
        tmp = out + ".partial"
        with open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hbytes)))
            fo.write(hbytes)
            for i, k in enumerate(keys):
                for kk, v in converted(f, k)[0]:
                    data = v.reshape(-1).view(torch.uint8).numpy().tobytes() if v.numel() else b""
                    if len(data) != header[kk]["data_offsets"][1] - header[kk]["data_offsets"][0]:
                        raise SystemExit(f"{kk}: size changed between the passes")
                    fo.write(data)
                if i % 200 == 0:
                    print(f"[quant] write {i}/{len(keys)} tensors, {time.time() - t0:.0f} s", flush=True)
            fo.flush()
            os.fsync(fo.fileno())
    os.replace(tmp, out)
    errs.sort()
    print(json.dumps({"out": out, "format": cli.format, "layers": len(errs), "rel_err_median": round(errs[len(errs) // 2], 4),
                      "rel_err_max": round(errs[-1], 4), "mb": round(os.path.getsize(out) / 1024 ** 2), "sec": round(time.time() - t0)}),
          flush=True)


if __name__ == "__main__":
    main()
