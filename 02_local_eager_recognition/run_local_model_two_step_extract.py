#!/usr/bin/env python3
"""Run the local MinerU2.5-Pro model implementation through the two-step protocol.

This script still uses the Hugging Face AutoProcessor for tokenization and image
preprocessing, but the model class itself is implemented locally in
local_modeling_mineru.py and does not import Transformers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from run_manual_two_step_extract import (
    DEFAULT_IMAGE,
    DEFAULT_SYSTEM_PROMPT,
    LocalMinerUTwoStepClient,
    clean_json,
    configure_npu_conv3d_mode,
    configure_npu_jit_compile,
    NPU_CONV3D_MODE_CHOICES,
    NPU_JIT_COMPILE_CHOICES,
)


class LocalMinerUModelPredictor:
    def __init__(self, model: LocalMinerU2_5ForConditionalGeneration, processor: Any, *, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
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
    predictor = LocalMinerUModelPredictor(model, processor, max_new_tokens=args.max_new_tokens)
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
