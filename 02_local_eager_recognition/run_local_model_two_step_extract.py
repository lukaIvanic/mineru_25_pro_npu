#!/usr/bin/env python3
"""Run the local MinerU2.5-Pro model implementation through the two-step protocol.

This script still uses the Hugging Face AutoProcessor for tokenization and image
preprocessing, but the model class itself is implemented locally in
local_modeling_mineru.py and does not import Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration


DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "crops" / "crop_01_text_block_en.png"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Detection:",
}
LAYOUT_RE = (
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)"
)
ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}
BLOCK_TYPES = {
    "algorithm",
    "aside_text",
    "chart",
    "code",
    "code_caption",
    "equation",
    "equation_block",
    "footer",
    "formula_number",
    "header",
    "image",
    "image_block",
    "image_caption",
    "image_footnote",
    "index",
    "list",
    "list_item",
    "page_footnote",
    "page_number",
    "phonetic",
    "ref_text",
    "table",
    "table_caption",
    "table_footnote",
    "text",
    "title",
    "unknown",
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file_record(path: Path, *, hash_contents: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": path.name,
        "size_bytes": int(path.stat().st_size),
    }
    if hash_contents:
        record["sha256"] = sha256_file(path)
    return record


def collect_model_identity(model_dir: Path, *, hash_model_files: bool = False) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    safetensors_paths = sorted(model_dir.glob("*.safetensors"))
    return {
        "model_dir": str(model_dir),
        "snapshot_revision": model_dir.name,
        "config_json": None if not config_path.exists() else model_file_record(config_path, hash_contents=True),
        "safetensors": [model_file_record(path, hash_contents=hash_model_files) for path in safetensors_paths],
        "safetensors_sha256_computed": bool(hash_model_files),
    }


def get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def prepare_for_layout(image: Image.Image, layout_image_size: tuple[int, int]) -> Image.Image:
    return get_rgb_image(image).resize(layout_image_size, Image.Resampling.BICUBIC)


def convert_bbox(coords: tuple[str, str, str, str]) -> Optional[list[float]]:
    bbox = tuple(map(int, coords))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [num / 1000.0 for num in (x1, y1, x2, y2)]


def parse_layout_output(output: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(LAYOUT_RE, output, re.DOTALL):
        x1, y1, x2, y2, ref_type, rotate_token, tail = match.groups()
        bbox = convert_bbox((x1, y1, x2, y2))
        if bbox is None:
            continue
        block_type = ref_type.lower()
        if block_type == "unknown":
            block_type = "image"
        if block_type == "inline_formula":
            continue
        if block_type not in BLOCK_TYPES:
            continue
        angle = ANGLE_MAPPING.get(rotate_token)
        block = {
            "type": block_type,
            "bbox": bbox,
            "angle": angle,
        }
        if block_type == "text":
            block["merge_prev"] = "txt_contd_tgt" in tail
        blocks.append(block)
    return blocks


def resize_by_need(
    image: Image.Image,
    *,
    min_image_edge: int = 28,
    max_image_edge_ratio: float = 50.0,
) -> Image.Image:
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > max_image_edge_ratio:
        width, height = image.size
        if width > height:
            new_w, new_h = width, math.ceil(width / max_image_edge_ratio)
        else:
            new_w, new_h = math.ceil(height / max_image_edge_ratio), height
        new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
        image = new_image
    if min(image.size) < min_image_edge:
        scale = min_image_edge / min(image.size)
        new_w, new_h = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image


def crop_block(image: Image.Image, block: dict[str, Any]) -> Image.Image:
    image = get_rgb_image(image)
    width, height = image.size
    x1, y1, x2, y2 = block["bbox"]
    crop = image.crop((x1 * width, y1 * height, x2 * width, y2 * height))
    if crop.width < 1 or crop.height < 1:
        raise ValueError(f"Cropped block image has invalid size {crop.size}")
    angle = block.get("angle")
    if angle in (90, 180, 270):
        crop = crop.rotate(angle, expand=True)
    return resize_by_need(crop)


def select_prompt(block_type: str) -> str:
    return DEFAULT_PROMPTS.get(block_type) or DEFAULT_PROMPTS["[default]"]


class LocalMinerUModelPredictor:
    def __init__(
        self,
        model: LocalMinerU2_5ForConditionalGeneration,
        processor: Any,
        *,
        max_new_tokens: int,
        benchmark_decode: bool = False,
        decode_warmup_steps: int = 8,
        decode_measure_steps: int = 64,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.benchmark_decode = bool(benchmark_decode)
        self.decode_warmup_steps = int(decode_warmup_steps)
        self.decode_measure_steps = int(decode_measure_steps)
        skip_token_ids: set[int] = set()
        for owner in (model.config, model.config.text_config, processor.tokenizer):
            for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
                token_id = getattr(owner, field, None)
                if isinstance(token_id, int):
                    skip_token_ids.add(token_id)
        self.skip_token_ids = skip_token_ids

    def build_messages(self, prompt: str, *, has_image: bool = True) -> list[dict[str, Any]]:
        user_content = [{"type": "text", "text": prompt}]
        if has_image:
            user_content = [{"type": "image"}, *user_content]
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def tensor_shapes(batch_encoding: Any) -> dict[str, list[int]]:
        shapes: dict[str, list[int]] = {}
        for key, value in batch_encoding.items():
            if hasattr(value, "shape"):
                shapes[str(key)] = [int(dim) for dim in value.shape]
        return shapes

    def predict(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        import torch

        messages = self.build_messages(prompt, has_image=image is not None)
        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        input_shapes = self.tensor_shapes(inputs)
        prompt_input_ids = inputs.input_ids[0].tolist()
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)

        start = time.perf_counter()
        with torch.inference_mode():
            generated_tensor = self.model.generate_ids(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=getattr(inputs, "pixel_values", None),
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                max_new_tokens=self.max_new_tokens,
                eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", self.model.config.eos_token_id),
                pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", self.model.config.pad_token_id),
            )
        generate_s = time.perf_counter() - start
        decode_benchmark: dict[str, Any] = {"enabled": False}
        if self.benchmark_decode:
            decode_benchmark = self.model.benchmark_decode(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=getattr(inputs, "pixel_values", None),
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                warmup_steps=self.decode_warmup_steps,
                measure_steps=self.decode_measure_steps,
                eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", self.model.config.eos_token_id),
                pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", self.model.config.pad_token_id),
            )

        generated_ids = generated_tensor.cpu().tolist()[0]
        filtered_ids = [token_id for token_id in generated_ids if token_id not in self.skip_token_ids]
        text = self.processor.batch_decode(
            [filtered_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "prompt": prompt,
            "chat_prompt": chat_prompt,
            "input_shapes": input_shapes,
            "input_token_count": len(prompt_input_ids),
            "generated_token_ids": generated_ids,
            "filtered_token_ids": filtered_ids,
            "generated_token_count": len(generated_ids),
            "filtered_token_count": len(filtered_ids),
            "text": text,
            "generate_s": generate_s,
            "generate_kwargs": {
                "do_sample": False,
                "use_cache": True,
                "max_new_tokens": self.max_new_tokens,
            },
            "decode_benchmark": decode_benchmark,
        }


class LocalMinerUTwoStepClient:
    """Minimal local replacement for the MinerU two-step client protocol."""

    def __init__(
        self,
        predictor: Any,
        *,
        layout_image_size: tuple[int, int] = (1036, 1036),
    ) -> None:
        self.predictor = predictor
        self.layout_image_size = layout_image_size

    def layout_detect(self, image: Image.Image) -> dict[str, Any]:
        prompt = DEFAULT_PROMPTS["[layout]"]
        layout_image = prepare_for_layout(image, self.layout_image_size)
        prediction = self.predictor.predict(layout_image, prompt)
        parsed_blocks = parse_layout_output(prediction["text"])
        return {
            "prompt": prompt,
            "raw_text": prediction["text"],
            "parsed_blocks": parsed_blocks,
            "input_shapes": prediction["input_shapes"],
            "input_token_count": prediction["input_token_count"],
            "generated_token_count": prediction["generated_token_count"],
            "filtered_token_count": prediction["filtered_token_count"],
            "generated_token_ids": prediction["generated_token_ids"],
            "filtered_token_ids": prediction["filtered_token_ids"],
            "generate_s": prediction["generate_s"],
            "decode_benchmark": prediction["decode_benchmark"],
        }

    def prepare_selected_block(
        self,
        image: Image.Image,
        blocks: list[dict[str, Any]],
        *,
        block_index: int,
    ) -> dict[str, Any]:
        if not blocks:
            raise RuntimeError("No layout blocks parsed from model output.")
        if block_index < 0 or block_index >= len(blocks):
            raise IndexError(f"--block-index {block_index} out of range for {len(blocks)} blocks")

        selected_block = blocks[block_index]
        selected_crop = crop_block(image, selected_block)
        return {
            "block_index": block_index,
            "block": selected_block,
            "crop": selected_crop,
            "crop_size": [int(selected_crop.width), int(selected_crop.height)],
            "prompt": select_prompt(str(selected_block["type"])),
        }

    def recognize_crop(self, crop: Image.Image, prompt: str, block: dict[str, Any]) -> dict[str, Any]:
        prediction = self.predictor.predict(crop, prompt)
        return {
            "selected_block": block,
            "prompt": prompt,
            "chat_prompt": prediction["chat_prompt"],
            "input_shapes": prediction["input_shapes"],
            "input_token_count": prediction["input_token_count"],
            "generated_token_count": prediction["generated_token_count"],
            "filtered_token_count": prediction["filtered_token_count"],
            "generated_token_ids": prediction["generated_token_ids"],
            "filtered_token_ids": prediction["filtered_token_ids"],
            "text": prediction["text"],
            "generate_s": prediction["generate_s"],
            "decode_benchmark": prediction["decode_benchmark"],
        }

    def two_step_extract(self, image: Image.Image, *, block_index: int = 0) -> dict[str, Any]:
        rgb_image = get_rgb_image(image)
        layout = self.layout_detect(rgb_image)
        if not layout["parsed_blocks"]:
            raise RuntimeError(f"No layout blocks parsed from output: {layout['raw_text']!r}")

        selected = self.prepare_selected_block(
            rgb_image,
            layout["parsed_blocks"],
            block_index=block_index,
        )
        recognition = self.recognize_crop(
            selected["crop"],
            selected["prompt"],
            selected["block"],
        )
        return {
            "image": rgb_image,
            "layout": layout,
            "selected_block_index": selected["block_index"],
            "selected_crop": selected["crop"],
            "selected_crop_size": selected["crop_size"],
            "recognition": recognition,
        }


def parse_torch_dtype(value: str):
    import torch

    normalized = str(value).lower()
    aliases = {
        "auto": None,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype {value!r}; expected one of {sorted(aliases)}")
    return aliases[normalized]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MinerU2.5-Pro model directory with safetensors files.")
    parser.add_argument("--processor", default=None, help="Optional processor directory; defaults to --model.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--device", default=None, help="Optional explicit device, e.g. cuda:0 or npu:0.")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--npu-jit-compile", choices=NPU_JIT_COMPILE_CHOICES, default="off")
    parser.add_argument("--npu-conv3d-mode", choices=NPU_CONV3D_MODE_CHOICES, default="auto")
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layout-image-size", type=int, nargs=2, default=(1036, 1036), metavar=("W", "H"))
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--benchmark-decode", action="store_true", help="Run a separate warmed decode-only tok/s benchmark for each generation call.")
    parser.add_argument("--decode-warmup-steps", type=int, default=8)
    parser.add_argument("--decode-measure-steps", type=int, default=64)
    parser.add_argument("--hash-model-files", action="store_true", help="Compute sha256 for safetensors files for model-version audit. This may add startup wall time.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-selected-crop", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model).expanduser().resolve()
    processor_dir = Path(args.processor).expanduser().resolve() if args.processor is not None else model_dir
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    model_identity = collect_model_identity(model_dir, hash_model_files=bool(args.hash_model_files))

    if args.device is not None and str(args.device).startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        configure_npu_jit_compile(args.npu_jit_compile, device=args.device, verbose=True)
        configure_npu_conv3d_mode(args.npu_conv3d_mode, device=args.device, verbose=True)

    from transformers import AutoProcessor

    setup_start = time.perf_counter()
    dtype = parse_torch_dtype(args.dtype)
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=args.device,
    )
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        use_fast=bool(args.use_fast),
        local_files_only=True,
    )
    predictor = LocalMinerUModelPredictor(
        model,
        processor,
        max_new_tokens=args.max_new_tokens,
        benchmark_decode=bool(args.benchmark_decode),
        decode_warmup_steps=int(args.decode_warmup_steps),
        decode_measure_steps=int(args.decode_measure_steps),
    )
    client = LocalMinerUTwoStepClient(
        predictor,
        layout_image_size=(int(args.layout_image_size[0]), int(args.layout_image_size[1])),
    )
    setup_s = time.perf_counter() - setup_start

    image = Image.open(image_path)
    result = client.two_step_extract(image, block_index=args.block_index)
    rgb_image = result["image"]
    selected_crop = result["selected_crop"]
    if args.save_selected_crop is not None:
        crop_path = args.save_selected_crop.expanduser().resolve()
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        selected_crop.save(crop_path)
    else:
        crop_path = None

    layout = result["layout"]
    recognition = result["recognition"]
    payload = {
        "experiment": "02_local_mineru_model_two_step_extract",
        "scope": "Manual local MinerU two-step protocol with local torch model implementation; AutoProcessor remains external.",
        "model": str(model_dir),
        "model_identity": model_identity,
        "processor": str(processor_dir),
        "image": str(image_path),
        "image_size": [int(rgb_image.width), int(rgb_image.height)],
        "device": None if args.device is None else str(args.device),
        "dtype": str(args.dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "npu_conv3d_mode": str(args.npu_conv3d_mode),
        "use_fast": bool(args.use_fast),
        "layout_image_size": [int(args.layout_image_size[0]), int(args.layout_image_size[1])],
        "selected_block_index": int(result["selected_block_index"]),
        "selected_crop_path": None if crop_path is None else str(crop_path),
        "selected_crop_size": result["selected_crop_size"],
        "timing_s": {
            "setup_model_processor_s": float(setup_s),
            "layout_generate_s": float(layout["generate_s"]),
            "recognition_generate_s": float(recognition["generate_s"]),
        },
        "decode_benchmark_config": {
            "enabled": bool(args.benchmark_decode),
            "decode_warmup_steps": int(args.decode_warmup_steps),
            "decode_measure_steps": int(args.decode_measure_steps),
            "scope": "decode-only forward calls after prefill; prefill_s is reported separately and excluded from decode_tok_s",
        },
        "layout": {
            "prompt": layout["prompt"],
            "raw_text": layout["raw_text"],
            "parsed_blocks": layout["parsed_blocks"],
            "input_shapes": layout["input_shapes"],
            "input_token_count": layout["input_token_count"],
            "generated_token_count": layout["generated_token_count"],
            "filtered_token_count": layout["filtered_token_count"],
            "generated_token_ids": layout["generated_token_ids"],
            "filtered_token_ids": layout["filtered_token_ids"],
            "decode_benchmark": layout["decode_benchmark"],
        },
        "recognition": {
            "selected_block": recognition["selected_block"],
            "prompt": recognition["prompt"],
            "chat_prompt": recognition["chat_prompt"],
            "input_shapes": recognition["input_shapes"],
            "input_token_count": recognition["input_token_count"],
            "generated_token_count": recognition["generated_token_count"],
            "filtered_token_count": recognition["filtered_token_count"],
            "generated_token_ids": recognition["generated_token_ids"],
            "filtered_token_ids": recognition["filtered_token_ids"],
            "text": recognition["text"],
            "decode_benchmark": recognition["decode_benchmark"],
        },
    }

    text = json.dumps(clean_json(payload), ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
