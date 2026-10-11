# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 nyanmi-1828
# Portions (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0) -- from city96/ComfyUI-GGUF tools/convert.py
# The tensor layout rules (QUANTIZATION_THRESHOLD, KEYS_HIPREC, KEYS_IGNORE, F32 for 1-D / small tensors, arch
# "cosmos") are adapted from city96/ComfyUI-GGUF tools/convert.py, licensed under the Apache License, Version 2.0
# (LICENSES/Apache-2.0.txt). Modified: one tensor at a time, BF16 for tensors whose rows do not split into blocks.
"""
Convert a bf16/fp16 Cosmos-Predict2-family DiT (Anima, Cosmos-Predict2) safetensors file to GGUF for ComfyUI-GGUF
(city96), one tensor at a time.

The layout follows ComfyUI-GGUF's own converter (tools/convert.py): architecture "cosmos"; 1-D tensors, tensors of at
most 1024 elements and the position embedder stay F32; "_extra_state" / "accum_" keys are dropped; the other tensors
are quantized to the chosen type. A tensor whose rows do not split into the type's blocks (e.g. the 68-wide patch
embedder) is kept in BF16. Quantization uses the gguf package's reference code, which offers Q8_0, Q5_0, Q5_1, Q4_0
and Q4_1 (the K-quants such as Q4_K_M need llama.cpp's quantizer).

  python custom_nodes/ComfyUI-WeightFerry/tools/convert_to_gguf.py models/diffusion_models/MODEL.safetensors --type Q8_0
  (writes MODEL-Q8_0.gguf next to it; load it with ComfyUI-GGUF's "Unet Loader (GGUF)"; use ComfyUI's Python with the
  gguf package installed: pip install gguf)
"""

import argparse
import os
import time

QUANTIZATION_THRESHOLD = 1024
TYPES = ("Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "BF16")
FILE_TYPES = {"Q8_0": "MOSTLY_Q8_0", "Q5_1": "MOSTLY_Q5_1", "Q5_0": "MOSTLY_Q5_0", "Q4_1": "MOSTLY_Q4_1",
              "Q4_0": "MOSTLY_Q4_0", "BF16": "MOSTLY_BF16"}
KEYS_HIPREC = ("pos_embedder",)
KEYS_IGNORE = ("_extra_state", "accum_")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="bf16 / fp16 .safetensors DiT (Anima / Cosmos-Predict2)")
    p.add_argument("--type", default="Q8_0", choices=TYPES, help="quantization type (default Q8_0)")
    p.add_argument("--out", default=None, help="output file (default: <src>-<type>.gguf)")
    cli = p.parse_args()
    out = cli.out or os.path.splitext(cli.src)[0] + f"-{cli.type}.gguf"
    import gguf
    import torch
    from safetensors import safe_open

    qtype = getattr(gguf.GGMLQuantizationType, cli.type)
    block, _ = gguf.GGML_QUANT_SIZES[qtype]
    writer = gguf.GGUFWriter(path=None, arch="cosmos")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(getattr(gguf.LlamaFileType, FILE_TYPES[cli.type]))
    counts = {}
    t0 = time.perf_counter()
    with safe_open(cli.src, "pt") as f:
        keys = sorted(f.keys())
        for i, key in enumerate(keys):
            if any(x in key for x in KEYS_IGNORE):
                continue
            t = f.get_tensor(key)
            data = t.to(torch.float32).numpy() if t.dtype == torch.bfloat16 else t.numpy()
            n = data.size
            if data.ndim == 1 or n <= QUANTIZATION_THRESHOLD or any(x in key for x in KEYS_HIPREC):
                kind = gguf.GGMLQuantizationType.F32
            elif data.shape[-1] % block == 0:
                kind = qtype
            else:
                kind = gguf.GGMLQuantizationType.BF16
            q = gguf.quants.quantize(data.astype("float32"), kind)
            writer.add_tensor(key, q, raw_dtype=kind)
            counts[kind.name] = counts.get(kind.name, 0) + 1
            if i % 100 == 0:
                print(f"[gguf] {i}/{len(keys)} tensors, {time.perf_counter() - t0:.0f} s", flush=True)
    writer.write_header_to_file(path=out)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"[gguf] {counts} -> {out} ({os.path.getsize(out) / 1e9:.2f} GB, {time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
