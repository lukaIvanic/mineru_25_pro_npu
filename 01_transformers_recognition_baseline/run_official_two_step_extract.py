#!/usr/bin/env python3
"""Run the highest-level official MinerUClient two-step extraction example.

This intentionally mirrors the mineru-vl-utils Transformers tutorial before we
start separating layout detection, crop preparation, and recognition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image


DEFAULT_MODEL = "opendatalab/MinerU2.5-Pro-2605-1.2B"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "crops" / "crop_01_text_block_en.png"
NPU_JIT_COMPILE_CHOICES = ("off", "on", "default")
NPU_CONV3D_MODE_CHOICES = ("auto", "inference_patch", "never")


def _normalize_3d_param(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value, value, value]
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return [int(value[0]), int(value[0]), int(value[0])]
        if len(value) == 3:
            return [int(value[0]), int(value[1]), int(value[2])]
    raise ValueError(f"Expected int or len-1/len-3 sequence, got {value!r}")


def maybe_patch_npu_conv3d(verbose: bool = True) -> None:
    try:
        import torch
        import torch.nn.functional as F
        import torch_npu
    except Exception as exc:
        if verbose:
            print(f"[npu] skipped Conv3D patch: {exc.__class__.__name__}: {exc}", flush=True)
        return

    if getattr(F.conv3d, "_mineru_npu_patch", False):
        return

    original_conv3d = F.conv3d

    def _patched_conv3d(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(input_tensor, torch.Tensor) and input_tensor.device.type == "npu":
            return torch_npu.npu_conv3d(
                input_tensor,
                weight,
                bias,
                _normalize_3d_param(stride),
                _normalize_3d_param(padding),
                _normalize_3d_param(dilation),
                int(groups),
            )
        return original_conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)

    _patched_conv3d._mineru_npu_patch = True
    F.conv3d = _patched_conv3d
    if verbose:
        print("[npu] patched torch.nn.functional.conv3d -> torch_npu.npu_conv3d", flush=True)


def configure_npu_conv3d_mode(mode: str, *, device: Optional[str], verbose: bool = True) -> None:
    if mode not in NPU_CONV3D_MODE_CHOICES:
        raise ValueError(f"Unsupported npu_conv3d_mode={mode!r}")
    if device is None or not str(device).startswith("npu"):
        return
    effective_mode = "inference_patch" if mode == "auto" else mode
    if effective_mode == "never":
        if verbose:
            print("[npu] leaving torch.nn.functional.conv3d unpatched", flush=True)
        return
    maybe_patch_npu_conv3d(verbose=verbose)


def configure_npu_jit_compile(mode: str, *, device: Optional[str], verbose: bool = True) -> None:
    if mode not in NPU_JIT_COMPILE_CHOICES:
        raise ValueError(f"Unsupported npu_jit_compile={mode!r}")
    if mode == "default" or device is None or not str(device).startswith("npu"):
        return
    try:
        import torch
        import torch_npu  # noqa: F401

        requested = mode == "on"
        torch.npu.set_compile_mode(jit_compile=requested)
        if verbose:
            print(f"[npu] set torch.npu compile mode: jit_compile={requested}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to set NPU jit_compile={mode}: {exc.__class__.__name__}: {exc}") from exc


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--device-map", default="auto", help='Transformers device_map value. Use "none" to omit it.')
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit device after loading, e.g. cuda:0 or npu:0. Use with --device-map none.",
    )
    parser.add_argument("--dtype", default="auto", help="Passed to Transformers as dtype=... for transformers>=4.56.")
    parser.add_argument("--torch-dtype", default=None, help="Fallback/override for older Transformers torch_dtype=...")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help='Optional Transformers attention implementation, e.g. "eager" or "sdpa".',
    )
    parser.add_argument("--npu-jit-compile", choices=NPU_JIT_COMPILE_CHOICES, default="off")
    parser.add_argument("--npu-conv3d-mode", choices=NPU_CONV3D_MODE_CHOICES, default="auto")
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    from mineru_vl_utils import MinerUClient
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from transformers import __version__ as transformers_version

    if args.device is not None and str(args.device_map).lower() not in {"none", "null", ""}:
        raise ValueError("Use --device-map none when passing --device; mixing both can create ambiguous placement.")

    if args.device is not None and str(args.device).startswith("npu"):
        import torch
        import torch_npu  # noqa: F401  # Required so torch recognizes the npu device type.

        torch.npu.set_device(args.device)
        configure_npu_jit_compile(args.npu_jit_compile, device=args.device, verbose=True)
        configure_npu_conv3d_mode(args.npu_conv3d_mode, device=args.device, verbose=True)

    model_kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
    }
    if str(args.device_map).lower() not in {"none", "null", ""}:
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation
    version_parts = transformers_version.split(".")
    use_dtype_key = len(version_parts) >= 2 and int(version_parts[0]) >= 4 and int(version_parts[1]) >= 56
    if args.torch_dtype is not None:
        model_kwargs["torch_dtype"] = args.torch_dtype
    elif use_dtype_key:
        model_kwargs["dtype"] = args.dtype
    else:
        model_kwargs["torch_dtype"] = args.dtype

    setup_start = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    if args.device is not None:
        model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        use_fast=bool(args.use_fast),
        local_files_only=bool(args.local_files_only),
    )
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=bool(args.image_analysis),
        use_tqdm=not bool(args.no_tqdm),
    )
    setup_s = time.perf_counter() - setup_start

    image = Image.open(image_path).convert("RGB")
    run_start = time.perf_counter()
    blocks = client.two_step_extract(image)
    run_s = time.perf_counter() - run_start

    payload = {
        "experiment": "01_official_mineru_client_two_step_extract",
        "scope": "Official mineru-vl-utils high-level tutorial path: MinerUClient.two_step_extract(image).",
        "model": str(args.model),
        "image": str(image_path),
        "image_size": [int(image.width), int(image.height)],
        "transformers_version": str(transformers_version),
        "device_map": str(args.device_map),
        "device": None if args.device is None else str(args.device),
        "dtype_arg": str(args.torch_dtype if args.torch_dtype is not None else args.dtype),
        "attn_implementation": None if args.attn_implementation is None else str(args.attn_implementation),
        "npu_jit_compile": str(args.npu_jit_compile),
        "npu_conv3d_mode": str(args.npu_conv3d_mode),
        "use_fast": bool(args.use_fast),
        "image_analysis": bool(args.image_analysis),
        "timing_s": {
            "setup_model_processor_client_s": float(setup_s),
            "two_step_extract_s": float(run_s),
        },
        "block_count": int(len(blocks)),
        "blocks": clean_json(blocks),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
