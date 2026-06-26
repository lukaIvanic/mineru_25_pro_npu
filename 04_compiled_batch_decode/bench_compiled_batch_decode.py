#!/usr/bin/env python3
"""Benchmark fixed-step compiled batched recognition decode for MinerU2.5-Pro.

This is intentionally a decode-only experiment. It uses different real crop
images for each batch row, performs sequential eager prefill outside the timed
decode window, then measures a static compiled one-token decode graph for an
exact number of forward calls. EOS is ignored by design.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from local_modeling_mineru import (
    DECODE_ROTARY_IMPL_CHOICES,
    DECODE_ROTARY_IMPL_MANUAL,
    DECODE_WEIGHT_FORMAT_CHOICES,
    LocalMinerU2_5ForConditionalGeneration,
    configure_decode_rotary_impl,
    configure_decode_weight_format,
)
from run_local_model_two_step_extract import (
    DEFAULT_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TORCHAIR_CACHE_DIR,
    NPU_CONV3D_MODE_CHOICES,
    NPU_JIT_COMPILE_CHOICES,
    clean_json,
    collect_model_identity,
    compile_static_decode,
    configure_npu_conv3d_mode,
    configure_npu_jit_compile,
    get_rgb_image,
    maybe_sync_device,
    parse_torch_dtype,
    select_prompt,
)


DEFAULT_CROPS_DIR = Path(__file__).resolve().parents[1] / "crops"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class PreparedCrop:
    path: Path
    block_type: str
    prompt: str
    chat_prompt: str
    inputs: Any
    input_shapes: dict[str, list[int]]
    image_size: list[int]


def tensor_shapes(batch_encoding: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in batch_encoding.items():
        if hasattr(value, "shape"):
            shapes[str(key)] = [int(dim) for dim in value.shape]
    return shapes


def infer_block_type(path: Path) -> str:
    name = path.stem.lower()
    if "table" in name:
        return "table"
    if "equation" in name or "formula" in name:
        return "equation"
    if "chart" in name:
        return "chart"
    if "image" in name and "caption" not in name:
        return "image"
    return "text"


def discover_crop_paths(crops_dir: Path, *, start_index: int, batch_size: int) -> list[Path]:
    paths = [
        path
        for path in sorted(crops_dir.expanduser().resolve().iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name.lower() != "contact_sheet.jpg"
    ]
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    selected = paths[int(start_index) : int(start_index) + int(batch_size)]
    if len(selected) < int(batch_size):
        raise ValueError(
            f"Need {int(batch_size)} distinct crops from {crops_dir}, "
            f"but only found {len(selected)} after start_index={int(start_index)}"
        )
    return selected


def resolve_crop_paths(args: argparse.Namespace) -> list[Path]:
    if args.crop:
        paths = [Path(path).expanduser().resolve() for path in args.crop]
        if len(paths) < int(args.batch_size):
            raise ValueError(f"--crop supplied {len(paths)} paths, but --batch-size is {int(args.batch_size)}")
        selected = paths[: int(args.batch_size)]
    else:
        selected = discover_crop_paths(args.crops_dir, start_index=int(args.start_index), batch_size=int(args.batch_size))
    missing = [str(path) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"crop image(s) not found: {missing}")
    if len({str(path) for path in selected}) != len(selected):
        raise ValueError("Batch rows must use distinct crop paths.")
    return selected


def build_chat_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_crop(
    *,
    path: Path,
    processor: Any,
    model: LocalMinerU2_5ForConditionalGeneration,
) -> PreparedCrop:
    with Image.open(path) as raw_image:
        image = get_rgb_image(raw_image).copy()
    block_type = infer_block_type(path)
    prompt = select_prompt(block_type)
    chat_prompt = build_chat_prompt(processor, prompt)
    inputs = processor(
        text=[chat_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    input_shapes = tensor_shapes(inputs)
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    return PreparedCrop(
        path=path,
        block_type=block_type,
        prompt=prompt,
        chat_prompt=chat_prompt,
        inputs=inputs,
        input_shapes=input_shapes,
        image_size=[int(image.width), int(image.height)],
    )


def prefill_one(
    model: LocalMinerU2_5ForConditionalGeneration,
    prepared: PreparedCrop,
    *,
    cache_length: int,
) -> dict[str, Any]:
    input_len = int(prepared.inputs.input_ids.shape[1])
    if input_len > int(cache_length):
        raise ValueError(
            f"{prepared.path.name} has input length {input_len}, which exceeds cache_length={int(cache_length)}"
        )
    maybe_sync_device(model.device)
    start = time.perf_counter()
    output = model.forward_static_prefill(
        input_ids=prepared.inputs.input_ids,
        attention_mask=prepared.inputs.attention_mask,
        pixel_values=getattr(prepared.inputs, "pixel_values", None),
        image_grid_thw=getattr(prepared.inputs, "image_grid_thw", None),
        cache_length=int(cache_length),
        logits_to_keep=1,
    )
    maybe_sync_device(model.device)
    prefill_s = time.perf_counter() - start
    next_token = torch.argmax(output.logits[:, -1, :].float(), dim=-1, keepdim=True)
    return {
        "prepared": prepared,
        "output": output,
        "next_token": next_token,
        "cache_position": output.next_cache_position.clone(),
        "prefill_s": float(prefill_s),
    }


def build_batch_state(
    model: LocalMinerU2_5ForConditionalGeneration,
    prepared_crops: list[PreparedCrop],
    *,
    cache_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item_states = [prefill_one(model, prepared, cache_length=cache_length) for prepared in prepared_crops]

    maybe_sync_device(model.device)
    assembly_start = time.perf_counter()
    key_caches = tuple(
        torch.cat([state["output"].cache.key_caches[layer_idx] for state in item_states], dim=0).contiguous()
        for layer_idx in range(model.config.text_config.num_hidden_layers)
    )
    value_caches = tuple(
        torch.cat([state["output"].cache.value_caches[layer_idx] for state in item_states], dim=0).contiguous()
        for layer_idx in range(model.config.text_config.num_hidden_layers)
    )
    next_token = torch.cat([state["next_token"] for state in item_states], dim=0).contiguous()
    cache_position = torch.cat([state["cache_position"] for state in item_states], dim=0).contiguous()
    rope_deltas = torch.cat([state["output"].rope_deltas for state in item_states], dim=0).contiguous()
    maybe_sync_device(model.device)
    assembly_s = time.perf_counter() - assembly_start

    state = {
        "next_token": next_token,
        "cache_position": cache_position,
        "rope_deltas": rope_deltas,
        "flat_cache": (*key_caches, *value_caches),
    }
    metadata = {
        "batch_size": len(prepared_crops),
        "cache_length": int(cache_length),
        "prefill_total_s": float(sum(state["prefill_s"] for state in item_states)),
        "batch_assembly_s": float(assembly_s),
        "items": [
            {
                "batch_row": idx,
                "path": str(item_state["prepared"].path),
                "name": item_state["prepared"].path.name,
                "block_type": item_state["prepared"].block_type,
                "prompt": item_state["prepared"].prompt,
                "image_size": item_state["prepared"].image_size,
                "input_shapes": item_state["prepared"].input_shapes,
                "input_token_count": int(item_state["prepared"].inputs.input_ids.shape[1]),
                "image_grid_thw": None
                if getattr(item_state["prepared"].inputs, "image_grid_thw", None) is None
                else [int(value) for value in item_state["prepared"].inputs.image_grid_thw[0].detach().cpu().tolist()],
                "prefill_s": float(item_state["prefill_s"]),
                "prefill_next_token_id": int(item_state["next_token"][0, 0].detach().cpu().item()),
                "prefill_next_cache_position": int(item_state["cache_position"][0].detach().cpu().item()),
            }
            for idx, item_state in enumerate(item_states)
        ],
    }
    return state, metadata


def run_compiled_steps(
    compiled_decode: Any,
    state: dict[str, Any],
    *,
    steps: int,
    collect_tokens: bool = False,
) -> tuple[torch.Tensor, list[list[int]] | None]:
    next_token = state["next_token"]
    cache_position = state["cache_position"]
    rope_deltas = state["rope_deltas"]
    flat_cache = state["flat_cache"]
    token_history = [next_token.detach().cpu()] if collect_tokens else None
    logits = None
    for _step in range(int(steps)):
        logits = compiled_decode(next_token, cache_position, rope_deltas, *flat_cache)
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        cache_position.add_(1)
        if token_history is not None:
            token_history.append(next_token.detach().cpu())
    state["next_token"] = next_token
    if logits is None:
        logits = torch.empty(0)
    if token_history is None:
        return logits, None
    tokens = torch.cat(token_history, dim=1).tolist()
    return logits, [[int(token_id) for token_id in row] for row in tokens]


def run_single_static_reference(
    model: LocalMinerU2_5ForConditionalGeneration,
    prepared: PreparedCrop,
    *,
    cache_length: int,
    steps: int,
) -> list[int]:
    output = model.forward_static_prefill(
        input_ids=prepared.inputs.input_ids,
        attention_mask=prepared.inputs.attention_mask,
        pixel_values=getattr(prepared.inputs, "pixel_values", None),
        image_grid_thw=getattr(prepared.inputs, "image_grid_thw", None),
        cache_length=int(cache_length),
        logits_to_keep=1,
    )
    cache = output.cache
    cache_position = output.next_cache_position.clone()
    next_token = torch.argmax(output.logits[:, -1, :].float(), dim=-1, keepdim=True)
    generated = [int(next_token[0, 0].detach().cpu().item())]
    for _step in range(int(steps)):
        decoded = model.forward_static_decode(
            input_ids=next_token,
            cache=cache,
            cache_position=cache_position,
            rope_deltas=output.rope_deltas,
            logits_to_keep=1,
        )
        next_token = torch.argmax(decoded.logits[:, -1, :].float(), dim=-1, keepdim=True)
        cache_position.add_(1)
        generated.append(int(next_token[0, 0].detach().cpu().item()))
    return generated


def set_decode_rotary_impl_for_validation(
    model: LocalMinerU2_5ForConditionalGeneration,
    mode: str,
) -> list[tuple[Any, str]]:
    previous: list[tuple[Any, str]] = []
    for module in model.modules():
        if hasattr(module, "decode_rotary_impl"):
            previous.append((module, str(module.decode_rotary_impl)))
            module.decode_rotary_impl = str(mode)
    return previous


def restore_decode_rotary_impl(previous: list[tuple[Any, str]]) -> None:
    for module, mode in previous:
        module.decode_rotary_impl = mode


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return int(idx)
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def validate_batched_decode(
    *,
    model: LocalMinerU2_5ForConditionalGeneration,
    prepared_crops: list[PreparedCrop],
    compiled_decode: Any,
    cache_length: int,
    steps: int,
) -> dict[str, Any]:
    if int(steps) <= 0:
        return {"enabled": False, "reason": "validation_steps <= 0"}
    batch_state, _metadata = build_batch_state(model, prepared_crops, cache_length=cache_length)
    maybe_sync_device(model.device)
    _logits, batched_tokens = run_compiled_steps(
        compiled_decode,
        batch_state,
        steps=int(steps),
        collect_tokens=True,
    )
    maybe_sync_device(model.device)
    previous_rotary = set_decode_rotary_impl_for_validation(model, DECODE_ROTARY_IMPL_MANUAL)
    try:
        references = [
            run_single_static_reference(model, prepared, cache_length=cache_length, steps=int(steps))
            for prepared in prepared_crops
        ]
    finally:
        restore_decode_rotary_impl(previous_rotary)
    per_item = []
    for idx, (batched, reference) in enumerate(zip(batched_tokens or [], references)):
        mismatch = first_mismatch(batched, reference)
        per_item.append(
            {
                "batch_row": idx,
                "name": prepared_crops[idx].path.name,
                "token_match": mismatch is None,
                "first_mismatch_index": mismatch,
                "batched_token_ids": batched,
                "single_item_reference_token_ids": reference,
            }
        )
    return {
        "enabled": True,
        "reference": "single_item_static_eager_manual_rotary_decode_same_prefill_contract",
        "validation_steps": int(steps),
        "token_match_all": all(item["token_match"] for item in per_item),
        "mismatch_count": sum(0 if item["token_match"] else 1 for item in per_item),
        "per_item": per_item,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MinerU2.5-Pro model directory.")
    parser.add_argument("--processor", default=None, help="Optional processor directory; defaults to --model.")
    parser.add_argument("--device", default=None, help="cuda:0, npu:0, or cpu.")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--npu-jit-compile", choices=NPU_JIT_COMPILE_CHOICES, default="off")
    parser.add_argument("--npu-conv3d-mode", choices=NPU_CONV3D_MODE_CHOICES, default="auto")
    parser.add_argument("--use-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    parser.add_argument("--crop", type=Path, action="append", default=None, help="Explicit crop path. Repeat to provide the batch rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index into sorted crops when --crop is not supplied.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-length", type=int, default=512)
    parser.add_argument("--measure-steps", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=8, help="Compiled decode warmup forwards after the first compile call.")
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--decode-weight-format", choices=DECODE_WEIGHT_FORMAT_CHOICES, default="none")
    parser.add_argument("--decode-rotary-impl", choices=DECODE_ROTARY_IMPL_CHOICES, default=DECODE_ROTARY_IMPL_MANUAL)
    parser.add_argument("--hash-model-files", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.measure_steps) <= 0:
        raise ValueError("--measure-steps must be positive")
    if int(args.cache_length) <= 0:
        raise ValueError("--cache-length must be positive")

    model_dir = Path(args.model).expanduser().resolve()
    processor_dir = Path(args.processor).expanduser().resolve() if args.processor is not None else model_dir
    crop_paths = resolve_crop_paths(args)

    if args.device is not None and str(args.device).startswith("npu"):
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
    maybe_sync_device(model.device)
    format_start = time.perf_counter()
    decode_weight_format = configure_decode_weight_format(model, str(args.decode_weight_format))
    maybe_sync_device(model.device)
    decode_weight_format["setup_s"] = float(time.perf_counter() - format_start)
    effective_decode_weight_format = str(decode_weight_format.get("effective_mode", "none"))
    rotary_start = time.perf_counter()
    decode_rotary_impl = configure_decode_rotary_impl(model, str(args.decode_rotary_impl))
    maybe_sync_device(model.device)
    decode_rotary_impl["setup_s"] = float(time.perf_counter() - rotary_start)
    effective_decode_rotary_impl = str(decode_rotary_impl.get("effective_mode", DECODE_ROTARY_IMPL_MANUAL))
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        use_fast=bool(args.use_fast),
        local_files_only=True,
    )
    prepared_crops = [prepare_crop(path=path, processor=processor, model=model) for path in crop_paths]
    maybe_sync_device(model.device)
    setup_s = time.perf_counter() - setup_start

    flat_decode = model.make_flat_static_decode_module(cache_length=int(args.cache_length)).eval()
    compile_wrapper_start = time.perf_counter()
    compiled_decode, compile_meta = compile_static_decode(
        flat_decode,
        device=model.device,
        cache_root=args.torchair_cache_dir,
        batch_size=int(args.batch_size),
        cache_length=int(args.cache_length),
        decode_weight_format=effective_decode_weight_format,
        decode_rotary_impl=effective_decode_rotary_impl,
    )
    maybe_sync_device(model.device)
    compile_wrapper_s = time.perf_counter() - compile_wrapper_start

    warm_state, warm_prefill_meta = build_batch_state(model, prepared_crops, cache_length=int(args.cache_length))
    maybe_sync_device(model.device)
    first_call_start = time.perf_counter()
    _first_logits, _ = run_compiled_steps(compiled_decode, warm_state, steps=1, collect_tokens=False)
    maybe_sync_device(model.device)
    first_call_s = time.perf_counter() - first_call_start

    remaining_warmup = max(0, int(args.warmup_steps) - 1)
    maybe_sync_device(model.device)
    warmup_start = time.perf_counter()
    if remaining_warmup:
        _warm_logits, _ = run_compiled_steps(compiled_decode, warm_state, steps=remaining_warmup, collect_tokens=False)
    maybe_sync_device(model.device)
    warmup_s = time.perf_counter() - warmup_start

    measure_state, measure_prefill_meta = build_batch_state(model, prepared_crops, cache_length=int(args.cache_length))
    maybe_sync_device(model.device)
    decode_start = time.perf_counter()
    _measured_logits, _ = run_compiled_steps(
        compiled_decode,
        measure_state,
        steps=int(args.measure_steps),
        collect_tokens=False,
    )
    maybe_sync_device(model.device)
    decode_s = time.perf_counter() - decode_start

    validation = validate_batched_decode(
        model=model,
        prepared_crops=prepared_crops,
        compiled_decode=compiled_decode,
        cache_length=int(args.cache_length),
        steps=int(args.validation_steps),
    )

    raw_batch_tokens = int(args.batch_size) * int(args.measure_steps)
    payload = {
        "experiment": "04_compiled_batch_decode",
        "scope": "Fixed-step compiled batched recognition decode only; sequential per-crop prefill is outside measured decode.",
        "model": str(model_dir),
        "model_identity": collect_model_identity(model_dir, hash_model_files=bool(args.hash_model_files)),
        "processor": str(processor_dir),
        "device": None if args.device is None else str(args.device),
        "dtype": str(args.dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "npu_conv3d_mode": str(args.npu_conv3d_mode),
        "use_fast": bool(args.use_fast),
        "batch_size": int(args.batch_size),
        "cache_length": int(args.cache_length),
        "measure_steps": int(args.measure_steps),
        "warmup_steps_requested": int(args.warmup_steps),
        "warmup_steps_after_compile_call": int(remaining_warmup),
        "validation_steps": int(args.validation_steps),
        "decode_weight_format": decode_weight_format,
        "decode_rotary_impl": decode_rotary_impl,
        "compile": {
            **dict(compile_meta),
            "compile_wrapper_s": float(compile_wrapper_s),
            "compiled_first_call_s": float(first_call_s),
            "first_call_counts_as_warmup_step": True,
        },
        "timing_s": {
            "setup_model_processor_inputs_s": float(setup_s),
            "warmup_prefill_total_s": float(warm_prefill_meta["prefill_total_s"]),
            "warmup_batch_assembly_s": float(warm_prefill_meta["batch_assembly_s"]),
            "post_compile_warmup_decode_s": float(warmup_s),
            "measure_prefill_total_s": float(measure_prefill_meta["prefill_total_s"]),
            "measure_batch_assembly_s": float(measure_prefill_meta["batch_assembly_s"]),
            "decode_s": float(decode_s),
        },
        "throughput": {
            "raw_batch_tokens": int(raw_batch_tokens),
            "decode_forward_calls": int(args.measure_steps),
            "decode_calls_per_s": float(int(args.measure_steps) / decode_s) if decode_s > 0 else 0.0,
            "raw_batch_tok_s": float(raw_batch_tokens / decode_s) if decode_s > 0 else 0.0,
            "per_item_tok_s_if_divided_evenly": float(int(args.measure_steps) / decode_s) if decode_s > 0 else 0.0,
            "eos_ignored": True,
        },
        "selected_crops": measure_prefill_meta["items"],
        "validation": validation,
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
