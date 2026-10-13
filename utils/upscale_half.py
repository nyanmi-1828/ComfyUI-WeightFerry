# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 nyanmi-1828
# Adapted from ComfyUI comfy_extras/nodes_upscale_model.py (GPL-3.0); _rrdb_append_forward re-implements the
# ResidualDenseBlock_5C.forward of chaiNNer-org/spandrel (MIT) with an append buffer; see THIRD_PARTY_NOTICES.md
"""
Upscale with a model in half precision. ComfyUI's ImageUpscaleWithModel runs the model in float32
(`upscale_model(a.float())`); on an L4, 4x-UltraSharp on 896x1472 took 9.2 s in float32, 3.55 s in float16 (PSNR 67.6 dB
against float32) and 3.36 s in bfloat16 (49.1 dB). Same tiling (512, overlap 32, halved on OOM) and output as ComfyUI's
node; dtype "float32" (the default) gives ComfyUI's result.

Provenance: ComfyUI's ImageUpscaleWithModel.execute (comfy_extras/nodes_upscale_model.py, GPL-3.0) with a dtype
option added.
"""

import contextlib

import torch

import comfy.model_management as model_management
import comfy.utils


def _rrdb_append_forward(orig, epilogue=False, residual=False):
    """spandrel's ResidualDenseBlock_5C.forward (ESRGAN / RRDB models such as 4x-UltraSharp) with its growing
    concatenations torch.cat((x, x1, ...), 1) replaced by one buffer per call that x and each new conv output are
    copied into once; each conv reads a channel prefix of it, for batch 1 a contiguous NCHW view with the same shape
    and strides as the cat. The same convolutions, inputs and order: the same bits (2026-10-12, 4x-UltraSharp float16
    on 896x1472 as USDU runs it: identical output, x1.11; the concatenations were 15.8% of the GPU time). Other cases
    (batch > 1, the "plus" variant, non-contiguous input) take spandrel's code.
    epilogue (WEIGHTFERRY_UPSCALE_RRDB_EPILOGUE, on by default since 2026-10-12): conv1..conv4 (Conv2d -> LeakyReLU)
    run without their bias and one kernel adds the bias, applies the LeakyReLU and writes straight into the buffer
    (_conv_act_into); the same bits, x1.03 on the same upscale.
    residual (WEIGHTFERRY_UPSCALE_RRDB_RESIDUAL, on by default since 2026-10-12): conv5 runs without its bias and one
    kernel does the bias add, x5 * 0.2 and + x (_conv5_residual); the same bits, x1.04 on the same upscale."""
    def tail(self, buf, x):
        if residual:
            return _conv5_residual(self.conv5, buf, x)
        x5 = self.conv5(buf)
        return x5 * 0.2 + x

    def forward(self, x):
        if self.conv1x1 is not None or x.shape[0] != 1 or not x.is_contiguous():
            return orig(self, x)
        nf = x.shape[1]
        if epilogue:
            gc = _out_channels(self.conv1)
            if gc is not None:
                buf = torch.empty((1, nf + 4 * gc) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
                buf[:, :nf].copy_(x)
                _conv_act_into(self.conv1, x, buf[:, nf:nf + gc])
                _conv_act_into(self.conv2, buf[:, :nf + gc], buf[:, nf + gc:nf + 2 * gc])
                _conv_act_into(self.conv3, buf[:, :nf + 2 * gc], buf[:, nf + 2 * gc:nf + 3 * gc])
                _conv_act_into(self.conv4, buf[:, :nf + 3 * gc], buf[:, nf + 3 * gc:])
                return tail(self, buf, x)
        x1 = self.conv1(x)
        gc = x1.shape[1]
        buf = torch.empty((1, nf + 4 * gc) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
        buf[:, :nf].copy_(x)
        buf[:, nf:nf + gc].copy_(x1)
        x2 = self.conv2(buf[:, :nf + gc])
        buf[:, nf + gc:nf + 2 * gc].copy_(x2)
        x3 = self.conv3(buf[:, :nf + 2 * gc])
        buf[:, nf + 2 * gc:nf + 3 * gc].copy_(x3)
        x4 = self.conv4(buf[:, :nf + 3 * gc])
        buf[:, nf + 3 * gc:].copy_(x4)
        return tail(self, buf, x)
    forward._weightferry = True
    return forward


def _rrdb_residual_forward(orig):
    """spandrel's RRDB.forward (three dense blocks, then out * 0.2 + x) with the last step in one kernel
    (_residual_into); WEIGHTFERRY_UPSCALE_RRDB_RESIDUAL."""
    def forward(self, x):
        out = self.RDB3(self.RDB2(self.RDB1(x)))
        return _residual_into(out, x)
    forward._weightferry = True
    return forward


def _out_channels(block):
    try:
        return int(block[0].out_channels)
    except Exception:  # noqa: BLE001 -- not a Sequential(Conv2d, ...): the plain append path
        return None


_epi = {}


def _epilogue_kernel(name="k"):
    """The fused element-wise steps after the dense block's convolutions, in Triton, launched with fused multiply-add
    off; every intermediate that PyTorch rounds to the tensor's dtype is rounded the same way. None without Triton.
    "k": y (the conv output without bias, contiguous) + bias[c], rounded, then LeakyReLU, written to dst: PyTorch's own
    two steps -- the cuDNN convolution's separate bias add_ (Convolution.cpp: computed in float, rounded to the dtype)
    and LeakyReLU (ActivationLeakyReluKernel.cu: in float, x > 0 ? x : x * slope, slope as a float) -- in one pass
    instead of three (add_, leaky_relu_, copy_).
    "r": (y + bias[c], rounded, when a bias is given) * scale, rounded, + x, rounded: spandrel's x5 * 0.2 + x (a
    multiplication by a Python scalar: in float, the scalar as a float) after the conv's bias add, in one pass instead
    of three (add_, mul, add)."""
    if "k" in _epi:
        return _epi[name]
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001 -- no Triton: the plain append path
        _epi["k"] = _epi["r"] = None
        from .logging import log_warn
        log_warn(f"Upscale: Triton is not available ({exc!r}); WEIGHTFERRY_UPSCALE_RRDB_EPILOGUE / _RESIDUAL have no "
                 "effect.")
        return None

    @triton.jit
    def _bias_leaky(y_ptr, b_ptr, o_ptr, n, hw, slope, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(y_ptr + offs, mask=m).to(tl.float32)
        b = tl.load(b_ptr + offs // hw, mask=m).to(tl.float32)
        t = (v + b).to(o_ptr.dtype.element_ty).to(tl.float32)
        r = tl.where(t > 0, t, t * slope)
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _bias_scale_add(y_ptr, b_ptr, x_ptr, o_ptr, n, hw, scale, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(y_ptr + offs, mask=m).to(tl.float32)
        if HAS_BIAS:
            v = (v + tl.load(b_ptr + offs // hw, mask=m).to(tl.float32)).to(o_ptr.dtype.element_ty).to(tl.float32)
        t = (v * scale).to(o_ptr.dtype.element_ty).to(tl.float32)
        r = t + tl.load(x_ptr + offs, mask=m).to(tl.float32)
        tl.store(o_ptr + offs, r.to(o_ptr.dtype.element_ty), mask=m)

    def run(y, bias, dst, slope):
        n = y.numel()
        _bias_leaky[(triton.cdiv(n, 1024),)](y, bias, dst, n, y.shape[2] * y.shape[3], float(slope), BLOCK=1024,
                                             enable_fp_fusion=False)

    def run_res(y, bias, x, dst, scale):
        n = y.numel()
        _bias_scale_add[(triton.cdiv(n, 1024),)](y, y if bias is None else bias, x, dst, n, y.shape[2] * y.shape[3],
                                                 float(scale), HAS_BIAS=bias is not None, BLOCK=1024,
                                                 enable_fp_fusion=False)
    _epi["k"], _epi["r"] = run, run_res
    return _epi[name]


def _backend_ok():
    # the cuDNN path adds the bias separately after the convolution (Convolution.cpp); other backends (ROCm, cuDNN off)
    # may add it inside the convolution before rounding, which these kernels would not match
    return (torch.version.hip is None and torch.backends.cudnn.enabled and torch.backends.cudnn.is_available()
            and not torch.is_autocast_enabled())


def _plain_conv(c, inp):
    nn = torch.nn
    return (type(c) is nn.Conv2d and c.bias is not None and c.padding_mode == "zeros"
            and isinstance(c.padding, tuple) and inp.is_cuda and inp.is_contiguous()
            and inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
            and c.weight.dtype == inp.dtype and c.bias.dtype == inp.dtype
            and not (torch.is_grad_enabled() and c.weight.requires_grad)
            and not (c._forward_hooks or c._forward_pre_hooks))


def _failed(name, exc):
    _epi[name] = None
    what = "epilogue" if name == "k" else "residual"
    from .logging import log_warn
    log_warn(f"Upscale: the RRDB {what} kernel failed ({exc!r}); "
             f"WEIGHTFERRY_UPSCALE_RRDB_{what.upper()} has no effect.")


def _conv_act_into(block, inp, dst):
    """dst <- block(inp) for spandrel's conv_block Sequential(Conv2d, LeakyReLU): the same F.conv2d without its bias,
    then the epilogue kernel; anything else (other layers, hooks, padding modes, dtypes, autocast, gradients, CPU)
    takes block(inp) and a copy; so do non-NVIDIA builds, cuDNN off, and every call after the kernel failed once."""
    k = _epilogue_kernel()
    nn = torch.nn
    ok = (k is not None and _backend_ok() and type(block) is nn.Sequential and len(block) == 2
          and type(block[1]) is nn.LeakyReLU and _plain_conv(block[0], inp)
          and not (block._forward_hooks or block._forward_pre_hooks or block[1]._forward_hooks
                   or block[1]._forward_pre_hooks))
    if not ok:
        dst.copy_(block(inp))
        return
    c = block[0]
    y = torch.nn.functional.conv2d(inp, c.weight, None, c.stride, c.padding, c.dilation, c.groups)
    if y.shape != dst.shape or not y.is_contiguous():
        dst.copy_(block[1](y + c.bias.view(1, -1, 1, 1)))  # not expected; the reference steps
        return
    try:
        k(y, c.bias, dst, block[1].negative_slope)
    except Exception as exc:  # noqa: BLE001 -- Triton could not compile / launch it: the reference steps from now on
        _failed("k", exc)
        dst.copy_(block[1](y.add_(c.bias.view(1, -1, 1, 1))))


def _conv5_residual(block, inp, x):
    """block(inp) * 0.2 + x for the dense block's last conv (a Conv2d, or a Sequential holding only one): the same
    F.conv2d without its bias, then one kernel for the bias add, * 0.2 and + x, written over the conv output. Anything
    else takes spandrel's steps (the same conditions as _conv_act_into)."""
    r = _epilogue_kernel("r")
    nn = torch.nn
    c = block[0] if type(block) is nn.Sequential and len(block) == 1 else block
    ok = (r is not None and _backend_ok() and _plain_conv(c, inp) and x.is_contiguous() and x.dtype == inp.dtype
          and not (block is not c and (block._forward_hooks or block._forward_pre_hooks)))
    if not ok:
        return block(inp) * 0.2 + x
    y = torch.nn.functional.conv2d(inp, c.weight, None, c.stride, c.padding, c.dilation, c.groups)
    if y.shape != x.shape or not y.is_contiguous():
        return (y + c.bias.view(1, -1, 1, 1)) * 0.2 + x  # not expected; the reference steps
    try:
        r(y, c.bias, x, y, 0.2)
    except Exception as exc:  # noqa: BLE001 -- the reference steps from now on
        _failed("r", exc)
        return y.add_(c.bias.view(1, -1, 1, 1)) * 0.2 + x
    return y


def _residual_into(out, x):
    """out * 0.2 + x (spandrel's RRDB), written over out by the residual kernel when out is a separate contiguous CUDA
    tensor of x's shape and dtype; otherwise PyTorch's two steps."""
    r = _epilogue_kernel("r")
    if not (r is not None and out.is_cuda and out.shape == x.shape and out.dtype == x.dtype
            and out.dtype in (torch.float16, torch.float32, torch.bfloat16) and out.is_contiguous()
            and x.is_contiguous() and out.data_ptr() != x.data_ptr()
            and not (torch.is_grad_enabled() and out.requires_grad)):
        return out * 0.2 + x
    try:
        r(out, None, x, out, 0.2)
    except Exception as exc:  # noqa: BLE001 -- the reference steps from now on
        _failed("r", exc)
        return out * 0.2 + x
    return out


_warned = []


_STRIP_STATS = {"strips": 0, "whole": 0}  # tiles computed in strips / as a whole (tests read it)


def _strip_forward(orig, rows):
    """WEIGHTFERRY_UPSCALE_STRIPS (2026-10-13; 256 rows by default): spandrel's RRDBNet.forward with the model's last stage
    -- from its last nn.Upsample (nearest x2) to the end, the 3x3 convolutions and LeakyReLUs at the output resolution
    (spandrel flattens them into the model's top-level Sequential) -- computed in strips of ``rows`` output rows, each
    with a halo of one row per convolution, so every output pixel sees the same inputs (at the image's top / bottom the
    strip starts at the edge, where the zero padding is the model's own). Those convolutions on 64 x 2048 x 2048 maps
    are the tile's CUDA peak (Codex B, measured per stage); in strips the process's dedicated memory of one 512 tile
    in float16 peaked 2304 -> 768 MB. float16 only: the same bits there (cuDNN computes each pixel the same way for
    the strip's shape), float32 did not give the same bits."""
    nn = torch.nn

    def forward(self, x):
        seq = getattr(self, "model", None)
        # cuDNN on and its heuristics (not autotuning, which may pick another algorithm per strip shape), NVIDIA
        # (review 47); the measured float16 case only
        if (self.shuffle_factor or type(seq) is not nn.Sequential or x.shape[0] != 1 or x.dtype != torch.float16
                or not x.is_cuda or not _backend_ok() or torch.backends.cudnn.benchmark):
            _STRIP_STATS["whole"] += 1
            return orig(self, x)
        ups = [i for i, m in enumerate(seq) if type(m) is nn.Upsample]
        if not ups:
            _STRIP_STATS["whole"] += 1
            return orig(self, x)
        u = ups[-1]
        up, tail = seq[u], list(seq)[u + 1:]
        if (up.mode != "nearest" or up.scale_factor is None or float(up.scale_factor) != 2.0
                or not tail or not all(type(m) in (nn.Conv2d, nn.LeakyReLU) for m in tail)
                or not all(m.kernel_size == (3, 3) and m.stride == (1, 1) and m.padding == (1, 1)
                           and m.dilation == (1, 1) and m.groups == 1 and m.padding_mode == "zeros"
                           for m in tail if type(m) is nn.Conv2d)
                or any(m._forward_hooks or m._forward_pre_hooks for m in [up] + tail)):
            _STRIP_STATS["whole"] += 1
            return orig(self, x)
        halo = sum(1 for m in tail if type(m) is nn.Conv2d)
        f = seq[:u](x)
        H, W = f.shape[2] * 2, f.shape[3] * 2
        _STRIP_STATS["strips" if H > rows else "whole"] += 1
        if H <= rows:
            y = up(f)
            for m in tail:
                y = m(y)
            return y
        out = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            lo, hi = max(0, r0 - halo), min(H, r1 + halo)
            a, b = lo // 2, (hi + 1) // 2
            y = up(f[:, :, a:b])[:, :, lo - 2 * a:hi - 2 * a]
            for m in tail:
                y = m(y)
            y = y[:, :, r0 - lo:r1 - lo]
            if out is None:
                out = torch.empty((1, y.shape[1], H, W), dtype=y.dtype, device=y.device)
            out[:, :, r0:r1] = y
        return out
    forward._weightferry = True
    return forward


@contextlib.contextmanager
def _strips():
    """While this node runs its model: spandrel's RRDBNet with its last stage in strips (WEIGHTFERRY_UPSCALE_STRIPS =
    output rows per strip, 0 = off), restored afterwards."""
    from ..core import env as _env
    try:
        rows = int(_env.get("WEIGHTFERRY_UPSCALE_STRIPS") or 0)
    except ValueError:
        rows = 0
    cls = None
    if rows > 0:
        try:
            from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet as cls
        except Exception:  # noqa: BLE001 -- spandrel moved it: the whole tile at once
            cls = None
    if cls is None or getattr(cls.forward, "_weightferry", False):
        yield
        return
    orig = cls.forward
    cls.forward = _strip_forward(orig, rows)
    try:
        yield
    finally:
        cls.forward = orig


@contextlib.contextmanager
def _rrdb_append():
    """While this node runs its model: spandrel's dense block with the append buffer (WEIGHTFERRY_UPSCALE_RRDB_APPEND),
    restored afterwards (ComfyUI's own upscale node is not touched). On by default since 2026-10-12."""
    from ..core import env as _env
    cls = None
    if _env.get("WEIGHTFERRY_UPSCALE_RRDB_APPEND") == "1":
        try:
            from spandrel.architectures.__arch_helpers.block import ResidualDenseBlock_5C as cls
        except Exception as exc:  # noqa: BLE001 -- spandrel moved / renamed it: spandrel's own code runs
            cls = None
            if not _warned:
                _warned.append(True)
                from .logging import log_warn
                log_warn(f"Upscale: spandrel's ResidualDenseBlock_5C not found ({exc!r}); "
                         "WEIGHTFERRY_UPSCALE_RRDB_APPEND has no effect.")
    if cls is None or getattr(cls.forward, "_weightferry", False):
        yield
        return
    residual = _env.get("WEIGHTFERRY_UPSCALE_RRDB_RESIDUAL") == "1"
    outer = None
    if residual:
        try:
            from spandrel.architectures.__arch_helpers.block import RRDB as outer
        except Exception:  # noqa: BLE001 -- only the dense block's own residual is fused
            outer = None
    orig = cls.forward
    orig_outer = outer.forward if outer is not None else None
    cls.forward = _rrdb_append_forward(orig, _env.get("WEIGHTFERRY_UPSCALE_RRDB_EPILOGUE") == "1", residual)
    if outer is not None:
        outer.forward = _rrdb_residual_forward(orig_outer)
    try:
        yield
    finally:
        cls.forward = orig
        if outer is not None:
            outer.forward = orig_outer


class WeightFerryUpscaleWithModelHalf:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "upscale_model": ("UPSCALE_MODEL",),
            "image": ("IMAGE",),
            "dtype": (["float32", "float16", "bfloat16"], {"default": "float32", "tooltip":
                      "float32 = ComfyUI's ImageUpscaleWithModel. float16: ~2.6x faster on Ada, nearly identical "
                      "(4x-UltraSharp: PSNR 67.6 dB vs float32); bfloat16 slightly less close"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "WeightFerry/upscale"

    def upscale(self, upscale_model, image, dtype="float32"):
        dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        device = upscale_model.patcher.load_device
        alpha = None
        if image.shape[-1] == 4:
            alpha = image[..., 3:4]
            image = image[..., :3]
        memory_required = (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        from ..core import vram_budget
        vram_budget.apply(device, "upscale")
        vram_budget.free_dit_slots("upscale")
        model_management.load_models_gpu([upscale_model.patcher], memory_required=memory_required, force_full_load=True)
        model = upscale_model.model
        orig_dtype = next(model.parameters()).dtype
        in_img = image.movedim(-1, -3).to(device)
        output_device = model_management.intermediate_device()
        tile, overlap = 512, 32
        from ..core import env as _env
        # WEIGHTFERRY_UPSCALE_HALF_OUT (2026-10-13, off by default): each tile goes to the CPU in the model's dtype and
        # tiled_scale's float32 accumulation (ps * mask into a float32 output) widens it there -- the same values
        # (float16 / bfloat16 -> float32 is exact) with half the bytes copied from the GPU
        half_out = (_env.get("WEIGHTFERRY_UPSCALE_HALF_OUT") == "1" and dt != torch.float32
                    and torch.device(output_device).type == "cpu")

        def run_tile(a):
            y = upscale_model(a.to(dt))
            return y if half_out else y.float()
        try:
            if dt != torch.float32:
                model.to(dtype=dt)
            while True:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2],
                                                                                tile_x=tile, tile_y=tile, overlap=overlap)
                    pbar = comfy.utils.ProgressBar(steps)
                    with _rrdb_append(), _strips():
                        s = comfy.utils.tiled_scale(in_img, run_tile, tile_x=tile,
                                                    tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale,
                                                    pbar=pbar, output_device=output_device)
                    break
                except Exception as e:  # noqa: BLE001 -- as ComfyUI's node: re-raised unless it is an OOM
                    model_management.raise_non_oom(e)
                    tile //= 2
                    if tile < 128:
                        raise
        finally:
            if dt != torch.float32:
                model.to(dtype=orig_dtype)
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0).to(model_management.intermediate_dtype())
        if alpha is not None:
            alpha = comfy.utils.common_upscale(alpha.movedim(-1, -3).to(s), s.shape[2], s.shape[1], "bilinear", "disabled")
            s = torch.cat((s, alpha.movedim(-3, -1)), dim=-1)
        return (s,)
