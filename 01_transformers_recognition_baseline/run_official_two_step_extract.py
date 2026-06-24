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
from typing import Any

from PIL import Image


DEFAULT_MODEL = "opendatalab/MinerU2.5-Pro-2605-1.2B"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "crops" / "crop_01_text_block_en.png"


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
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=True)
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
        import torch_npu  # noqa: F401  # Required so torch recognizes the npu device type.

    model_kwargs: dict[str, Any] = {
        "local_files_only": bool(args.local_files_only),
    }
    if str(args.device_map).lower() not in {"none", "null", ""}:
        model_kwargs["device_map"] = args.device_map
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
