# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# A few lines follow ComfyUI comfy/text_encoders/llama.py (GPL-3.0); see THIRD_PARTY_NOTICES.md
"""
Streamed LLM text encoder engine
===========================================
Streams Transformer language models layer-by-layer with double-buffered prefetching.
Purpose: run LLM text encoders that do not fit in VRAM (7B-14B class) on 8GB GPUs.


Pipeline per layer:  mmap (safetensors) --CPU memcpy--> pinned staging --async DMA--> GPU block buffer
The source is a safetensors file, or the weights a ComfyUI CLIP object already holds (``tensors``: its state dict with
the "model." keys, so the file is not read a second time).
- Staging buffers are pinned, so the H2D copy is truly asynchronous and overlaps the
  current layer's compute.
- A GPU block buffer is only overwritten after the compute that last read it finished.
- CPU side: tensors are zero-copy views of the file's mmap; the staging copy runs in 32 MB pieces over
  WEIGHTFERRY_LLM_COPY_THREADS threads (default 4). Three staging buffers (WEIGHTFERRY_LLM_STAGING), each released as
  soon as its DMAs are issued; dequantization and the copy into the block run afterwards from the GPU-side copies
  (with 2, ~0.7 s of a 1.69 s Qwen2.5-VL 7B encode waited for a free buffer).

Provenance: a few lines (the causal mask, the RoPE call) follow ComfyUI's comfy/text_encoders/llama.py (GPL-3.0).
"""

from typing import Tuple, Any, Optional
import concurrent.futures
import json
import mmap
import os
import struct
import warnings
import torch
import safetensors.torch

from .tiered_store import _ST_DTYPES, PinnedHost

from comfy.text_encoders import llama
from comfy.ldm.modules.attention import optimized_attention_for_device
from . import env as _env


class StreamedLLMEngine:
    """
    Core Streaming Engine that reads safetensors on CPU and transfers weights
    to GPU on-demand with double-buffered CUDA Stream prefetching.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.bfloat16, tensors: Optional[dict] = None, label: Optional[str] = None):
        self.model_path = model_path
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.sf = None
        self._tensors = None
        self._mm = self._mm_bytes = None
        self._tensor_info = {}
        if tensors is not None:
            for k, t in tensors.items():
                if type(t) not in (torch.Tensor, torch.nn.Parameter):  # e.g. ComfyUI's QuantizedTensor
                    raise ValueError(f"Streamed LLM: the CLIP's weight {k} is a {type(t).__name__}; choose the text encoder "
                                     f"file in model_name for quantized text encoders")
            self._tensors = tensors
            self.keys = list(tensors.keys())
            self.label = label or "the CLIP input's text encoder"
        else:
            if model_path is None or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            self.sf = safetensors.safe_open(model_path, framework="pt", device="cpu")
            self.keys = list(self.sf.keys())
            self.label = label or os.path.basename(model_path)
            self._open_mmap(model_path)
        self._copy_threads = max(1, int(_env.get("WEIGHTFERRY_LLM_COPY_THREADS")))
        self._pool = (concurrent.futures.ThreadPoolExecutor(self._copy_threads, thread_name_prefix="llm-copy")
                      if self._copy_threads > 1 else None)

        # Architecture detection
        self.config, self.block_cls, self.model_type = self._detect_architecture()
        self.num_layers = self.config.num_hidden_layers
        self.hidden_size = self.config.hidden_size

        # Per-layer key lists (avoid scanning every key for every layer)
        self.layer_keys = [[] for _ in range(self.num_layers)]
        for k in self.keys:
            if k.startswith("model.layers."):
                part = k[len("model.layers."):].split(".")[0]
                if part.isdigit() and int(part) < self.num_layers:
                    self.layer_keys[int(part)].append(k)
        missing = [i for i, keys in enumerate(self.layer_keys) if not keys]
        if missing:
            raise ValueError(
                f"Streamed LLM: unsupported text encoder {self.label}: no weights under "
                f"'model.layers.<n>.' for {len(missing)} of {self.num_layers} layers (first: {missing[0]}). Supported: "
                f"Qwen3 0.6B, Gemma-2 2B, Qwen2.5-VL 7B in ComfyUI's text encoder format.")

        # Streams & Attention
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        # H2D copies run on their own stream, so a layer's transfer does not queue behind the previous layer's
        # dequantization / block writes on the prefetch stream
        self.dma_stream = torch.cuda.Stream(device=self.device)
        # A causal mask is always passed, so the attention kernel must support masks
        self.optimized_attention = optimized_attention_for_device(self.device, mask=True, small_input=True)
        self._staging = None  # lazily allocated pinned staging buffers
        self._n_staging = max(2, int(_env.get("WEIGHTFERRY_LLM_STAGING")))

    def _detect_architecture(self) -> Tuple[Any, Any, str]:
        layer_indices = set()
        for k in self.keys:
            if "layers." in k:
                part = k.split("layers.")[1].split(".")[0]
                if part.isdigit():
                    layer_indices.add(int(part))
        num_layers = len(layer_indices) if layer_indices else 28

        # Check embedding dimensions directly from safetensors header
        embed_k = "model.embed_tokens.weight" if "model.embed_tokens.weight" in self.keys else "embed_tokens.weight"
        if embed_k not in self.keys:
            raise ValueError(f"Streamed LLM: unsupported text encoder {self.label}: no "
                             f"embed_tokens.weight. Supported: Qwen3 0.6B, Gemma-2 2B, Qwen2.5-VL 7B.")
        hidden_dim = self._shape(embed_k)[1]

        # Anima Native Qwen3_06B (1024 dim)
        if hidden_dim == 1024:
            config = llama.Qwen3_06BConfig()
            config.num_hidden_layers = num_layers
            return config, llama.TransformerBlock, f"Anima-Qwen3 ({num_layers}L, 1024D)"

        # Gemma 2 detection
        elif any("pre_feedforward_layernorm" in k for k in self.keys) or "gemma" in self.label.lower():
            config = llama.Gemma2_2B_Config()
            if hidden_dim != config.hidden_size:
                raise ValueError(f"Streamed LLM: unsupported Gemma text encoder {self.label} (hidden "
                                 f"size {hidden_dim}). Supported: Gemma-2 2B ({config.hidden_size}).")
            config.num_hidden_layers = num_layers
            return config, llama.TransformerBlockGemma2, f"Gemma-2 ({num_layers}L, {config.hidden_size}D)"

        # Qwen 2.5 7B detection (3584 dim)
        elif hidden_dim == 3584:
            config = llama.Qwen25_7BVLI_Config()
            config.num_hidden_layers = num_layers
            return config, llama.TransformerBlock, f"Qwen-2.5-7B ({num_layers}L, 3584D)"

        raise ValueError(f"Streamed LLM: unsupported text encoder {self.label} (hidden size "
                         f"{hidden_dim}). Supported: Qwen3 0.6B (1024), Gemma-2 2B (2304), Qwen2.5-VL 7B (3584).")

    def _open_mmap(self, model_path):
        """Zero-copy byte view of the whole file plus each tensor's (offset, bytes, dtype, shape)."""
        self._mm = self._mm_bytes = None
        self._tensor_info = {}
        try:
            self._mm_file = open(model_path, "rb")
            n = struct.unpack("<Q", self._mm_file.read(8))[0]
            header = json.loads(self._mm_file.read(n))
            header.pop("__metadata__", None)
            base = 8 + n
            info = {}
            for k, v in header.items():
                if v["dtype"] not in _ST_DTYPES:
                    return  # unknown dtype: keep the safe_open path for everything
                start, end = v["data_offsets"]
                info[k] = (base + start, end - start, _ST_DTYPES[v["dtype"]], tuple(v["shape"]))
            self._mm = mmap.mmap(self._mm_file.fileno(), 0, access=mmap.ACCESS_READ)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                self._mm_bytes = torch.frombuffer(self._mm, dtype=torch.uint8)
            self._tensor_info = info
        except Exception:  # noqa: BLE001 -- fall back to safe_open.get_tensor
            self._mm = self._mm_bytes = None
            self._tensor_info = {}

    def _shape(self, k):
        return tuple(self._tensors[k].shape) if self._tensors is not None else tuple(self.sf.get_slice(k).get_shape())

    def _nbytes(self, k):
        if self._tensors is not None:
            t = self._tensors[k]
            return t.numel() * t.element_size()
        sl = self.sf.get_slice(k)
        numel = 1
        for d in sl.get_shape():
            numel *= d
        return numel * _dtype_size(sl.get_dtype())

    def _tensor(self, k):
        return self._tensors[k] if self._tensors is not None else self.sf.get_tensor(k)

    def _embed_rows(self, k, ids):
        """Rows ``ids`` of the embedding table (from the file only those rows are read: the table is ~1GB for 7B)."""
        if self._tensors is not None:
            t = self._tensors[k]
            return t[torch.tensor(ids, dtype=torch.long, device=t.device)]
        sl = self.sf.get_slice(k)
        return torch.cat([sl[i:i + 1] for i in ids], dim=0)

    def _source_bytes(self, k):
        """(uint8 CPU view of tensor k's bytes, dtype, shape): a view of the mmap or of the CLIP's CPU tensor when
        possible, else a copy."""
        if self._tensors is not None:
            t = self._tensors[k].detach()
            if t.device.type != "cpu":
                t = t.cpu()
            t = t.contiguous()
            return t.view(-1).view(torch.uint8), t.dtype, tuple(t.shape)
        info = self._tensor_info.get(k)
        if info is not None and self._mm_bytes is not None:
            off, nbytes, dtype, shape = info
            return self._mm_bytes[off:off + nbytes], dtype, shape
        t = self.sf.get_tensor(k)
        return t.contiguous().view(-1).view(torch.uint8), t.dtype, tuple(t.shape)

    def _copy_pieces(self, pairs, piece=32 * 1024 * 1024):
        """Copy (dst, src) uint8 views, split into ``piece``-byte chunks, on the copy threads."""
        jobs = []
        for dst, src in pairs:
            n = src.numel()
            for o in range(0, n, piece):
                jobs.append((dst[o:o + piece], src[o:o + piece]))
        if self._pool is None or len(jobs) < 2:
            for d, s_ in jobs:
                d.copy_(s_)
            return
        # Worker threads do not inherit torch.inference_mode (ComfyUI runs nodes under it); copying into inference
        # tensors from a plain thread raises. Run the copies in the caller's mode.
        inference = torch.is_inference_mode_enabled()

        def run(job):
            with torch.inference_mode(inference):
                job[0].copy_(job[1])

        list(self._pool.map(run, jobs))

    def _ensure_staging(self):
        """``_n_staging`` pinned byte buffers, each large enough for the biggest layer."""
        if self._staging is not None:
            return
        max_bytes = 0
        for keys in self.layer_keys:
            total = 0
            for k in keys:
                total += self._nbytes(k)
                total += 64  # alignment padding per tensor
            max_bytes = max(max_bytes, total)
        # Exact-size registered pinned buffers (tiered_store.PinnedHost) instead of PyTorch's pinned allocator: that one
        # rounds up to a power of two, keeps freed blocks cached, and is invisible to the VRAM-overflow estimate
        # (vram_monitor subtracts known pinned memory from Windows' "Shared Usage")
        self._staging_holders = [PinnedHost(max_bytes) for _ in range(self._n_staging)]
        self._staging = [h.tensor for h in self._staging_holders]
        self._staging_events = [torch.cuda.Event() for _ in range(self._n_staging)]

    def _fetch_layer_to_gpu(self, layer_idx: int, target_block: torch.nn.Module, staging_idx: int,
                            stream: torch.cuda.Stream, block_free_event: Optional[torch.cuda.Event] = None):
        """Stage layer ``layer_idx`` and upload it into ``target_block`` on ``stream``. The DMAs only need the
        staging buffer; ``block_free_event`` (the last compute that read target_block) is waited for only before
        the writes into the block, so the transfer overlaps that compute."""
        prefix = f"model.layers.{layer_idx}."
        params = dict(target_block.named_parameters())
        params.update(dict(target_block.named_buffers()))

        # The previous DMA out of this staging buffer must be complete before CPU overwrites it
        self._staging_events[staging_idx].synchronize()
        staging = self._staging[staging_idx]

        staged = {}
        scales = {}
        offset = 0
        pairs = []
        views = []
        for k in self.layer_keys[layer_idx]:
            src, dtype, shape = self._source_bytes(k)
            nbytes = src.numel()
            offset = (offset + 63) // 64 * 64
            dst = staging[offset:offset + nbytes]
            pairs.append((dst, src))
            views.append((k, dst.view(dtype).view(shape)))
            offset += nbytes
        self._copy_pieces(pairs)
        for k, view in views:
            short_k = k[len(prefix):]
            if short_k.endswith(".scale_weight"):
                # legacy comfy scaled_fp8 format
                scales[short_k[:-len(".scale_weight")] + ".weight"] = view
            elif short_k.endswith(".weight_scale"):
                # comfy_quant format (per-tensor scale)
                scales[short_k[:-len(".weight_scale")] + ".weight"] = view
            else:
                staged[short_k] = view

        dma = self.dma_stream
        with torch.cuda.stream(dma):
            # 1. All DMAs out of the staging buffer, then release it (the CPU can refill it while the GPU-side work
            #    below runs)
            uploads = []
            for k, v in staged.items():
                dst = params.get(k)
                if dst is None:
                    continue
                s_gpu = None
                if k in scales:
                    if scales[k].numel() != 1:
                        raise NotImplementedError(
                            f"Streamed LLM: non per-tensor weight scale for '{k}' (shape {tuple(scales[k].shape)}) is not supported."
                        )
                    s_gpu = scales[k].to(self.device, non_blocking=True)
                uploads.append((dst, v.to(self.device, non_blocking=True), s_gpu))
            self._staging_events[staging_idx].record(dma)
        # Windows (WDDM) batches commands; querying the event submits the copies now (same fix as the DiT streaming)
        self._staging_events[staging_idx].query()
        with torch.cuda.stream(stream):
            stream.wait_stream(dma)
            for _dst, v_gpu, s_gpu in uploads:
                # Allocated on the DMA stream, consumed here: keep the caching allocator from reusing them early
                v_gpu.record_stream(stream)
                if s_gpu is not None:
                    s_gpu.record_stream(stream)
            if block_free_event is not None:
                stream.wait_event(block_free_event)
            # 2. Into the block's parameters. Scaled weights: same elementwise math as dtype(v) * dtype(s), in two
            #    passes instead of three (the cast happens in the copy, the scale is applied in place)
            for dst, v_gpu, s_gpu in uploads:
                dst.data.copy_(v_gpu)
                if s_gpu is not None:
                    dst.data.mul_(s_gpu.to(self.dtype))

    @torch.no_grad()
    def encode_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Runs the prompt token IDs through the full LLM via double-buffered layer streaming.
        Returns tensor of shape [batch, seq_len, hidden_size] in target dtype.
        """
        self._ensure_staging()
        compute_stream = torch.cuda.current_stream(self.device)
        seq_len = input_ids.shape[1]

        # 1. Embedding Layer: read only the rows that are needed (the full table is ~1GB for 7B models)
        embed_k = "model.embed_tokens.weight" if "model.embed_tokens.weight" in self.keys else "embed_tokens.weight"
        unique_ids, inverse = torch.unique(input_ids.cpu(), return_inverse=True)
        rows = self._embed_rows(embed_k, unique_ids.tolist())
        x = rows[inverse.to(rows.device)].to(self.device)
        embed_scale_k = embed_k[:-len(".weight")] + ".weight_scale"
        if embed_scale_k in self.keys:
            x = x.to(self.dtype) * self._tensor(embed_scale_k).to(self.device, dtype=self.dtype)
        x = x.to(self.dtype)
        if getattr(self.config, "transformer_type", None) in ("gemma2", "gemma3"):
            # Gemma scales embeddings by sqrt(hidden_size) (llama._make_scaled_embedding)
            x = x * (self.hidden_size ** 0.5)

        # 2. Position IDs & RoPE & Causal Mask
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        # Same as Llama2_.compute_freqs_cis, without instantiating the whole model on CPU
        freqs_cis = llama.precompute_freqs_cis(
            self.config.head_dim, position_ids, self.config.rope_theta, self.config.rope_scale,
            self.config.rope_dims, interleaved_mrope=getattr(self.config, "interleaved_mrope", False),
            device=self.device,
        )
        causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=self.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)

        # 3. Double-Buffer Blocks on GPU (torch.nn ops to bypass AIMDO VBAR casting interference)
        blocks = [
            self.block_cls(self.config, index=i, device=self.device, dtype=self.dtype, ops=torch.nn)
            for i in range(2)
        ]
        computed_events = [torch.cuda.Event() for _ in range(2)]
        # Order the weight writes (prefetch / DMA streams) after the blocks' allocation / init on the compute
        # stream; otherwise an init kernel can overwrite the weights under load
        self.prefetch_stream.wait_stream(compute_stream)
        self.dma_stream.wait_stream(compute_stream)

        # Prefetch Layer 0
        self._fetch_layer_to_gpu(0, blocks[0], 0 % self._n_staging, self.prefetch_stream)
        compute_stream.wait_stream(self.prefetch_stream)

        # Per-layer attributes that TransformerBlock derives from its index (e.g. Gemma sliding window)
        sliding = getattr(self.config, "sliding_attention", None)

        # 4. Asynchronous Streaming Loop
        for layer_idx in range(self.num_layers):
            curr_buf_idx = layer_idx % 2
            next_buf_idx = (layer_idx + 1) % 2
            if sliding is not None and hasattr(blocks[curr_buf_idx], "sliding_attention"):
                blocks[curr_buf_idx].sliding_attention = sliding[layer_idx % len(sliding)]

            # Enqueue current layer first so the CPU-side staging memcpy below overlaps its GPU compute
            x, _ = blocks[curr_buf_idx](x, attention_mask=causal_mask, freqs_cis=freqs_cis,
                                        optimized_attention=self.optimized_attention)
            computed_events[curr_buf_idx].record(compute_stream)

            if layer_idx + 1 < self.num_layers:
                # Overwrite the other block buffer only after its previous reader (layer_idx - 1) finished;
                # the DMAs themselves start right away (see _fetch_layer_to_gpu)
                self._fetch_layer_to_gpu(layer_idx + 1, blocks[next_buf_idx], (layer_idx + 1) % self._n_staging,
                                         self.prefetch_stream, block_free_event=computed_events[next_buf_idx])
                # Next layer's weights must have arrived before it is computed
                compute_stream.wait_stream(self.prefetch_stream)

        # 5. Final Normalization
        norm_k = "model.norm.weight" if "model.norm.weight" in self.keys else "norm.weight"
        if norm_k in self.keys:
            norm_weight = self._tensor(norm_k).to(self.device, dtype=self.dtype)
            norm_layer = llama.RMSNorm(
                self.hidden_size,
                eps=self.config.rms_norm_eps,
                add=getattr(self.config, "rms_norm_add", False),
                device=self.device,
                dtype=self.dtype
            )
            norm_layer.weight.data.copy_(norm_weight)
            x = norm_layer(x)

        torch.cuda.synchronize(self.device)
        # Clean up double buffers from VRAM
        del blocks
        torch.cuda.empty_cache()
        return x


_SAFETENSORS_DTYPE_SIZES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def _dtype_size(dtype_name: str) -> int:
    return _SAFETENSORS_DTYPE_SIZES.get(dtype_name.upper(), 4)
