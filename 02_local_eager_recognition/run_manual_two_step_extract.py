#!/usr/bin/env python3
"""Minimal local MinerU two-step protocol using the official HF model.

This script intentionally does not import MinerUClient or mineru_vl_utils.
It keeps AutoProcessor and Qwen2VLForConditionalGeneration for now, while
locally implementing the small protocol surface we need to replace first:

1. layout resize + layout prompt
2. layout output parsing
3. selected-block crop preparation
4. recognition prompt + generate + decode

The next experiment-2 scripts can replace model.generate and then the model
implementation itself while comparing against this protocol boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image


DEFAULT_MODEL = "opendatalab/MinerU2.5-Pro-2605-1.2B"
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


class LocalTransformersPredictor:
    def __init__(self, model: Any, processor: Any, *, max_new_tokens: Optional[int] = None):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.model_max_length = int(model.config.max_position_embeddings)
        skip_token_ids: set[int] = set()
        for owner in (model.config, processor.tokenizer):
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

    def build_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 100,
            "do_sample": False,
        }
        if self.max_new_tokens is None:
            kwargs["max_length"] = self.model_max_length
        else:
            kwargs["max_new_tokens"] = int(self.max_new_tokens)
        return kwargs

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
        generate_kwargs = self.build_generate_kwargs()

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids_tensor = self.model.generate(
                **inputs,
                use_cache=True,
                **generate_kwargs,
            )
        generate_s = time.perf_counter() - start

        output_ids = output_ids_tensor.cpu().tolist()[0]
        generated_ids = output_ids[len(prompt_input_ids) :]
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
            "generate_kwargs": generate_kwargs,
        }


class LocalMinerUTwoStepClient:
    """Minimal local replacement for the MinerU two-step client protocol.

    This class owns the page/crop protocol. It does not own the model internals;
    those live behind LocalTransformersPredictor for this step of experiment 02.
    """

    def __init__(
        self,
        predictor: LocalTransformersPredictor,
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
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--npu-jit-compile", choices=NPU_JIT_COMPILE_CHOICES, default="off")
    parser.add_argument("--npu-conv3d-mode", choices=NPU_CONV3D_MODE_CHOICES, default="auto")
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layout-image-size", type=int, nargs=2, default=(1036, 1036), metavar=("W", "H"))
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-selected-crop", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace, transformers_version: str) -> dict[str, Any]:
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
    return model_kwargs


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if args.device is not None and str(args.device_map).lower() not in {"none", "null", ""}:
        raise ValueError("Use --device-map none when passing --device; mixing both can create ambiguous placement.")

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from transformers import __version__ as transformers_version

    if args.device is not None and str(args.device).startswith("npu"):
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
        configure_npu_jit_compile(args.npu_jit_compile, device=args.device, verbose=True)
        configure_npu_conv3d_mode(args.npu_conv3d_mode, device=args.device, verbose=True)

    setup_start = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        **build_model_kwargs(args, transformers_version),
    )
    if args.device is not None:
        model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model,
        use_fast=bool(args.use_fast),
        local_files_only=bool(args.local_files_only),
    )
    predictor = LocalTransformersPredictor(model, processor, max_new_tokens=args.max_new_tokens)
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
        "experiment": "02_manual_mineru_protocol_official_hf_model",
        "scope": "Manual local layout/crop/prompt/generate protocol; no MinerUClient/mineru_vl_utils import.",
        "model": str(args.model),
        "image": str(image_path),
        "image_size": [int(rgb_image.width), int(rgb_image.height)],
        "transformers_version": str(transformers_version),
        "device_map": str(args.device_map),
        "device": None if args.device is None else str(args.device),
        "dtype_arg": str(args.torch_dtype if args.torch_dtype is not None else args.dtype),
        "attn_implementation": None if args.attn_implementation is None else str(args.attn_implementation),
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
