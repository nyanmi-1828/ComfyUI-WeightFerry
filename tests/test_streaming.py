# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
Self-contained check of the DiT layer streaming: no model files needed.

Builds a small random Cosmos-Predict2 DiT in memory (2048 channels, 3 blocks, bfloat16 -- ComfyUI's own model class,
loaded through ComfyUI's own loader), samples a few steps with ComfyUI as is, then with the streaming node in several
modes, and checks that every streamed result is bit for bit the same as ComfyUI's.

    cd ComfyUI
    python custom_nodes/ComfyUI-WeightFerry/tests/test_streaming.py [--comfy-root DIR]

--comfy-root: the folder that contains ComfyUI's code (comfy/, nodes.py); default: the folder above custom_nodes
(a git / portable install). The desktop app keeps its code elsewhere -- pass that folder then.
Needs an NVIDIA GPU (about 2 GB of VRAM) and about 3 GB of RAM; takes about a minute. Uses Dynamic VRAM when this
ComfyUI has it (--legacy: ComfyUI's classic ModelPatcher). Exit code 0 = all cases passed.
With ComfyUI-GGUF (custom_nodes/ComfyUI-GGUF) and the gguf package installed, the same model quantized to Q8_0 is also
streamed and compared with ComfyUI-GGUF's own result.
"""

import argparse
import importlib
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(HERE)
CUSTOM_NODES = os.path.dirname(SUITE_DIR)
BASE_DIR = os.path.dirname(CUSTOM_NODES)

DIT_CONFIG = dict(  # what ComfyUI's model detection reads from a Cosmos-Predict2 text-to-image checkpoint
    max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
    concat_padding_mask=True, model_channels=2048, num_blocks=3, num_heads=16, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256, extra_per_block_abs_pos_emb=False, rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0, image_model="cosmos_predict2",
)


def start_comfy(comfy_root, legacy):
    """ComfyUI's startup as far as the test needs it: the arguments now, and a function that switches Dynamic VRAM
    (comfy-aimdo) on like main.py when it is available (else, or with --legacy, the classic ModelPatcher stays) and
    returns the patcher kind. The switch comes after the random model is built: with Dynamic VRAM ComfyUI's layers
    only get their weights when a model is loaded."""
    for p in (comfy_root, CUSTOM_NODES):
        if p not in sys.path:
            sys.path.insert(0, p)
    import comfy.options
    comfy.options.enable_args_parsing()
    saved = sys.argv
    sys.argv = [saved[0], "--base-directory", BASE_DIR] + (["--disable-dynamic-vram"] if legacy else [])
    try:
        from comfy.cli_args import args
        try:
            from comfy.cli_args import enables_dynamic_vram
        except ImportError:  # older ComfyUI: no Dynamic VRAM
            enables_dynamic_vram = None
    finally:
        sys.argv = saved
    aimdo = None
    if not legacy and enables_dynamic_vram is not None and enables_dynamic_vram():
        try:  # like main.py: comfy-aimdo is initialised before torch touches the GPU
            import comfy_aimdo.control as aimdo
            headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
            try:
                aimdo.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
            except (TypeError, AttributeError):  # older comfy-aimdo
                try:
                    aimdo.init(simple_vram_headroom=headroom)
                except TypeError:
                    aimdo.init()
        except ImportError:
            aimdo = None
    return lambda: _enable_dynamic_vram(args, aimdo)


def _enable_dynamic_vram(args, aimdo):
    if aimdo is None:
        return "classic ModelPatcher"
    import comfy.model_management
    import comfy.model_patcher
    devices = comfy.model_management.get_all_torch_devices()
    try:
        ok = aimdo.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in devices)
    except (TypeError, AttributeError):
        ok = aimdo.init_devices(d.index for d in devices)
    if not ok:
        return "classic ModelPatcher (Dynamic VRAM not available on this GPU)"
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    import comfy.memory_management
    comfy.memory_management.aimdo_enabled = True
    return "Dynamic VRAM (ModelPatcherDynamic)"


def random_dit_state_dict(torch):
    """State dict of a small random Cosmos-Predict2 DiT (weights ~ N(0, 1/fan_in), norms 1, biases 0)."""
    import comfy.ops
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT
    dit = MiniTrainDIT(**DIT_CONFIG, dtype=torch.bfloat16, device="cpu", operations=comfy.ops.disable_weight_init)
    gen = torch.Generator().manual_seed(1234)
    sd = {}
    for name, prm in dit.named_parameters():
        if prm.dim() >= 2:
            fan_in = prm[0].numel()
            sd[name] = (torch.randn(prm.shape, generator=gen) / fan_in ** 0.5).to(torch.bfloat16)
        elif name.endswith("bias"):
            sd[name] = torch.zeros(prm.shape, dtype=torch.bfloat16)
        else:
            sd[name] = torch.ones(prm.shape, dtype=torch.bfloat16)
    for name, buf in dit.named_buffers():
        sd[name] = buf.clone()
    return sd


def gguf_patcher(state_dict, torch):
    """The same random DiT as a GGUF model the way ComfyUI-GGUF loads one: 2-D weights quantized to Q8_0 (GGMLTensor
    raw bytes + quant type + shape), the rest as F32 GGMLTensors, GGMLOps layers, GGUFModelPatcher. A string (the
    reason) when ComfyUI-GGUF (custom_nodes/ComfyUI-GGUF) or the gguf package cannot be imported."""
    try:
        import gguf
        ops = importlib.import_module("ComfyUI-GGUF.ops")
        gnodes = importlib.import_module("ComfyUI-GGUF.nodes")
    except ImportError as exc:
        return f"{type(exc).__name__}: {exc}"
    import comfy.sd
    sd = {}
    for k, v in state_dict.items():
        if v.dim() == 2 and v.shape[1] % 32 == 0:
            q = gguf.quants.quantize(v.float().numpy(), gguf.GGMLQuantizationType.Q8_0)
            sd[k] = ops.GGMLTensor(torch.from_numpy(q), tensor_type=gguf.GGMLQuantizationType.Q8_0,
                                   tensor_shape=v.shape)
        else:
            sd[k] = ops.GGMLTensor(v.float().contiguous(), tensor_type=gguf.GGMLQuantizationType.F32,
                                   tensor_shape=v.shape)
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops.GGMLOps()})
    return None if model is None else gnodes.GGUFModelPatcher.clone(model)


def _set_env(env):
    """Set the case's environment variables; returns what to restore."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-root", default=os.environ.get("COMFYUI_ROOT") or BASE_DIR,
                    help="folder with ComfyUI's code (comfy/, nodes.py); default: $COMFYUI_ROOT or the folder above custom_nodes")
    ap.add_argument("--legacy", action="store_true", help="classic ModelPatcher even when Dynamic VRAM is available")
    cli = ap.parse_args()
    enable_dynamic_vram = start_comfy(cli.comfy_root, cli.legacy)

    import torch
    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU")
        return 0
    import comfy.model_management as mm
    import comfy.sd
    import nodes
    suite = importlib.import_module(os.path.basename(SUITE_DIR))
    streaming_node = suite.NODE_CLASS_MAPPINGS["WeightFerryDiT"]()

    state_dict = random_dit_state_dict(torch)
    # for the GGUF cases: ComfyUI's loader consumes the state dict it is given, and with Dynamic VRAM switched on a new
    # random model would get no weights -- copy it now
    gguf_source = {k: v.clone() for k, v in state_dict.items()}
    patcher_kind = enable_dynamic_vram()
    print(f"ComfyUI: {patcher_kind}")
    model = comfy.sd.load_diffusion_model_state_dict(state_dict)
    if model is None:
        print("FAIL: ComfyUI did not recognise the test model")
        return 1
    gen = torch.Generator().manual_seed(99)
    positive = [[torch.randn(1, 8, 1024, generator=gen), {}]]
    negative = [[torch.randn(1, 8, 1024, generator=gen), {}]]

    seen = {}

    def sample(m):
        latent = {"samples": torch.zeros(1, 16, 1, 16, 24)}
        (out,) = nodes.common_ksampler(m, 42, 3, 4.0, "euler", "simple", positive, negative, latent, denoise=1.0)
        state = getattr(m.model.diffusion_model, "_weightferry_stream_state", None)
        seen["ssd_blocks"] = state.ring.store.n_blocks if state is not None and state.ring is not None else 0
        mm.unload_all_models()
        mm.soft_empty_cache()
        return out["samples"].clone()

    reference = sample(model)
    if not torch.isfinite(reference).all():
        print("FAIL: ComfyUI's own result is not finite")
        return 1
    common = importlib.import_module(f"{os.path.basename(SUITE_DIR)}.streaming.common")
    disk_dir = tempfile.mkdtemp(prefix="weightferry_test_")
    cases = [
        ("block streaming, 1 resident block", {}, dict(resident_blocks=1)),
        ("block streaming, 2 slots", {}, dict(resident_blocks=1, prefetch_slots=2)),
        ("self-managed resident blocks", {}, dict(resident_blocks=2, resident_mode=common.RESIDENT_MODE_SELF)),
        ("chunk streaming", {"WEIGHTFERRY_CHUNK": "1"}, dict(resident_blocks=1)),
        # a new load reuses the call order the first chunk-streaming load learned (no calibration pass)
        ("chunk streaming, learned order reused", {"WEIGHTFERRY_CHUNK": "1"}, dict(resident_blocks=1)),
        ("SSD tier", {"WEIGHTFERRY_DISK_DIR": disk_dir},
         dict(resident_blocks=1, tiering=common.TIERING_AUTO, ram_budget_gb=0.1)),
    ]
    failed = 0
    for label, env, options in cases:
        saved = _set_env(env)
        try:
            t = time.perf_counter()
            (streamed,) = streaming_node.apply_streaming(model, **options)
            out = sample(streamed)
            same = torch.equal(out, reference)
            diff = (out.float() - reference.float()).abs().max().item()
            note = ""
            if label == "SSD tier":
                note = f", {seen['ssd_blocks']} blocks on the SSD"
                if not seen["ssd_blocks"]:
                    # the classic ModelPatcher loads the whole model to the GPU first; the SSD tier then stays off
                    print(f"SKIP  {label}  (not used with the {patcher_kind}; the result was "
                          f"{'bit-exact' if same else 'NOT bit-exact'})")
                    failed += not same
                    continue
            print(f"{'PASS' if same else 'FAIL'}  {label}  ({time.perf_counter() - t:.1f} s{note}"
                  + ("" if same else f", max abs diff {diff:.3e}") + ")")
            failed += not same
        except Exception as exc:  # noqa: BLE001 -- report the case and go on with the others
            print(f"FAIL  {label}  ({type(exc).__name__}: {exc})")
            failed += 1
        finally:
            _restore_env(saved)
    # GGUF (ComfyUI-GGUF): the streaming node on a Q8_0 model, against ComfyUI-GGUF alone
    n_cases = len(cases)
    g_model = gguf_patcher(gguf_source, torch)
    if isinstance(g_model, str) or g_model is None:
        print(f"SKIP  GGUF  (ComfyUI-GGUF or the gguf package not available: {g_model})")
    else:
        g_reference = sample(g_model)
        for label, env, options in [("GGUF Q8_0, block streaming", {}, dict(resident_blocks=1)),
                                    ("GGUF Q8_0, chunk streaming", {"WEIGHTFERRY_CHUNK": "1"}, dict(resident_blocks=1))]:
            n_cases += 1
            saved = _set_env(env)
            try:
                t = time.perf_counter()
                (streamed,) = streaming_node.apply_streaming(g_model, **options)
                out = sample(streamed)
                same = torch.equal(out, g_reference)
                diff = (out.float() - g_reference.float()).abs().max().item()
                print(f"{'PASS' if same else 'FAIL'}  {label}  ({time.perf_counter() - t:.1f} s"
                      + ("" if same else f", max abs diff {diff:.3e}") + ")")
                failed += not same
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL  {label}  ({type(exc).__name__}: {exc})")
                failed += 1
            finally:
                _restore_env(saved)
    print(f"{n_cases - failed} / {n_cases} cases bit-exact with ComfyUI", "=> FAIL" if failed else "=> PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
