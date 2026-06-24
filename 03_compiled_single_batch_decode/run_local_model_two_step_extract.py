#!/usr/bin/env python3
"""Run the local MinerU2.5-Pro model implementation through the two-step protocol.

This script still uses the Hugging Face AutoProcessor for tokenization and image
preprocessing, but the model class itself is implemented locally in
local_modeling_mineru.py and does not import Transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image
import torch

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
DEFAULT_TORCHAIR_CACHE_DIR = Path("outputs") / "torchair_cache"
CANONICAL_CROP_01_RECOGNITION_TEXT = (
    "When an attempt is made to form the product BA, we discover that the dimensions are not compatible in this order "
    "because the rows of B are three-dimensional vectors and the columns of A are two-dimensional vectors. Hence the "
    "dot product of the jth row of B and the kth column of A is not defined."
)
CANONICAL_CROP_01_RECOGNITION_GENERATED_IDS = [
    4498,
    458,
    4774,
    374,
    1865,
    311,
    1352,
    279,
    1985,
    33489,
    11,
    582,
    6997,
    429,
    279,
    15336,
    525,
    537,
    18146,
    304,
    419,
    1973,
    1576,
    279,
    6978,
    315,
    425,
    525,
    2326,
    32420,
    22879,
    323,
    279,
    8147,
    315,
    362,
    525,
    1378,
    32420,
    22879,
    13,
    31040,
    279,
    12756,
    1985,
    315,
    279,
    502,
    339,
    2802,
    315,
    425,
    323,
    279,
    595,
    339,
    3250,
    315,
    362,
    374,
    537,
    4512,
    13,
    151645,
]
CANONICAL_CROP_01_RECOGNITION_FILTERED_IDS = CANONICAL_CROP_01_RECOGNITION_GENERATED_IDS[:-1]


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


def maybe_sync_device(device: Any) -> None:
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elif torch_device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


def import_torchair():
    try:
        import torchair

        CompilerConfig = torchair.CompilerConfig
    except Exception as direct_error:
        try:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
        except Exception as fallback_error:
            raise RuntimeError(
                "TorchAir is unavailable: direct `import torchair` failed with "
                f"{direct_error!r}, and `from torch_npu.dynamo import torchair` "
                f"failed with {fallback_error!r}."
            ) from fallback_error

    if not hasattr(torchair, "inference"):
        torchair.inference = importlib.import_module(f"{torchair.__name__}.inference")
    return torchair, CompilerConfig


def torchair_cache_dir_for_shape(cache_root: Path, *, batch_size: int, cache_length: int) -> Path:
    shape_key = f"mineru_manual_attention_bs{int(batch_size)}_cache{int(cache_length)}"
    return cache_root.expanduser().resolve() / shape_key


def compile_static_decode(
    flat_decode: Any,
    *,
    device: Any,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "npu":
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        shape_cache_dir = torchair_cache_dir_for_shape(
            cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        compiled_decode = torchair.inference.cache_compile(
            flat_decode.forward,
            config=config,
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        return compiled_decode, {
            "backend": "torchair",
            "compile_api": "torchair.inference.cache_compile",
            "fullgraph": True,
            "dynamic": False,
            "torchair_cache_dir": str(shape_cache_dir),
            "torchair_ge_cache": True,
            "batch_size": int(batch_size),
            "cache_length": int(cache_length),
            "decode_attention": "manual_eager_attention_ops",
        }

    compiled_decode = torch.compile(flat_decode, fullgraph=True, dynamic=False)
    return compiled_decode, {
        "backend": torch_device.type,
        "compile_api": "torch.compile",
        "fullgraph": True,
        "dynamic": False,
        "batch_size": int(batch_size),
        "cache_length": int(cache_length),
        "decode_attention": "manual_eager_attention_ops",
    }


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


def trim_ids_at_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    try:
        eos_index = token_ids.index(int(eos_token_id))
    except ValueError:
        return token_ids
    return token_ids[: eos_index + 1]


def first_mismatch_index(left: list[Any], right: list[Any]) -> int | None:
    for idx, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return int(idx)
    if len(left) != len(right):
        return int(min(len(left), len(right)))
    return None


def canonical_crop_01_reference_validation(image_path: Path, recognition: dict[str, Any]) -> dict[str, Any]:
    if image_path.name != "crop_01_text_block_en.png":
        return {
            "enabled": False,
            "reason": "canonical reference is defined only for crops/crop_01_text_block_en.png",
        }

    generated_ids = [int(value) for value in recognition.get("generated_token_ids", [])]
    filtered_ids = [int(value) for value in recognition.get("filtered_token_ids", [])]
    text = str(recognition.get("text", ""))
    generated_match = generated_ids == CANONICAL_CROP_01_RECOGNITION_GENERATED_IDS
    filtered_match = filtered_ids == CANONICAL_CROP_01_RECOGNITION_FILTERED_IDS
    text_match = text == CANONICAL_CROP_01_RECOGNITION_TEXT
    return {
        "enabled": True,
        "reference_source": "experiment_02_local_eager_cuda_outputs/local_model_crop_01_cuda.json",
        "reference_model_revision": "bff20d4ae2bf202df9f45284b4d43681555a97ed",
        "strict_match": bool(generated_match and filtered_match and text_match),
        "generated_ids_match": bool(generated_match),
        "filtered_ids_match": bool(filtered_match),
        "text_match": bool(text_match),
        "first_generated_id_mismatch_index": first_mismatch_index(
            generated_ids,
            CANONICAL_CROP_01_RECOGNITION_GENERATED_IDS,
        ),
        "first_filtered_id_mismatch_index": first_mismatch_index(
            filtered_ids,
            CANONICAL_CROP_01_RECOGNITION_FILTERED_IDS,
        ),
        "actual_generated_token_count": int(len(generated_ids)),
        "expected_generated_token_count": int(len(CANONICAL_CROP_01_RECOGNITION_GENERATED_IDS)),
        "actual_filtered_token_count": int(len(filtered_ids)),
        "expected_filtered_token_count": int(len(CANONICAL_CROP_01_RECOGNITION_FILTERED_IDS)),
        "actual_text": text,
        "expected_text": CANONICAL_CROP_01_RECOGNITION_TEXT,
    }


class CompiledSingleBatchRecognitionDecoder:
    def __init__(
        self,
        model: LocalMinerU2_5ForConditionalGeneration,
        *,
        cache_root: Path,
        cache_length: int | None,
    ) -> None:
        self.model = model
        self.cache_root = cache_root
        self.cache_length = None if cache_length is None else int(cache_length)
        self._flat_decode_by_shape: dict[tuple[int, int], Any] = {}
        self._compiled_by_shape: dict[tuple[int, int], tuple[Callable[..., Any], dict[str, Any]]] = {}
        self._warmup_by_shape: dict[tuple[int, int], dict[str, Any]] = {}

    def resolve_cache_length(self, input_ids: Any, max_new_tokens: int) -> int:
        return int(self.cache_length or (int(input_ids.shape[1]) + int(max_new_tokens)))

    @staticmethod
    def require_cache_capacity(input_ids: Any, cache_length: int, decode_steps: int) -> None:
        required = int(input_ids.shape[1]) + max(0, int(decode_steps))
        if int(cache_length) < required:
            raise ValueError(
                f"static cache_length={int(cache_length)} is too small for input_tokens={int(input_ids.shape[1])} "
                f"and decode_steps={int(decode_steps)}; need at least {required}"
            )

    @torch.inference_mode()
    def compiled_decode_for(self, *, batch_size: int, cache_length: int) -> tuple[Callable[..., Any], dict[str, Any]]:
        key = (int(batch_size), int(cache_length))
        if key not in self._compiled_by_shape:
            flat_decode = self._flat_decode_by_shape.get(key)
            if flat_decode is None:
                flat_decode = self.model.make_flat_static_decode_module(cache_length=int(cache_length)).eval()
                self._flat_decode_by_shape[key] = flat_decode
            compiled_decode, compile_meta = compile_static_decode(
                flat_decode,
                device=self.model.device,
                cache_root=self.cache_root,
                batch_size=int(batch_size),
                cache_length=int(cache_length),
            )
            self._compiled_by_shape[key] = (compiled_decode, compile_meta)
        return self._compiled_by_shape[key]

    @torch.inference_mode()
    def ensure_compiled_decode_warm(
        self,
        input_ids: Any,
        attention_mask: Any,
        pixel_values: Any,
        image_grid_thw: Any,
        *,
        cache_length: int,
    ) -> dict[str, Any]:
        key = (int(input_ids.shape[0]), int(cache_length))
        if key in self._warmup_by_shape:
            previous = dict(self._warmup_by_shape[key])
            previous["ran_this_call"] = False
            return previous

        compile_wrapper_start = time.perf_counter()
        compiled_decode, compile_meta = self.compiled_decode_for(
            batch_size=int(input_ids.shape[0]),
            cache_length=int(cache_length),
        )
        maybe_sync_device(self.model.device)
        compile_wrapper_s = time.perf_counter() - compile_wrapper_start

        maybe_sync_device(self.model.device)
        prefill_start = time.perf_counter()
        warm_prefill = self.model.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=int(cache_length),
            logits_to_keep=1,
        )
        maybe_sync_device(self.model.device)
        prefill_s = time.perf_counter() - prefill_start

        warm_next = torch.argmax(warm_prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        warm_cache_position = warm_prefill.next_cache_position.clone()
        flat_cache = warm_prefill.cache.flat_tensors()
        maybe_sync_device(self.model.device)
        first_call_start = time.perf_counter()
        _ = compiled_decode(warm_next, warm_cache_position, warm_prefill.rope_deltas, *flat_cache)
        maybe_sync_device(self.model.device)
        first_call_s = time.perf_counter() - first_call_start

        warmup_meta = {
            "ran": True,
            "ran_this_call": True,
            "batch_size": int(input_ids.shape[0]),
            "cache_length": int(cache_length),
            "compile_wrapper_s": float(compile_wrapper_s),
            "prefill_s": float(prefill_s),
            "first_call_s": float(first_call_s),
            "compile": dict(compile_meta),
        }
        self._warmup_by_shape[key] = dict(warmup_meta)
        return warmup_meta

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Any,
        attention_mask: Any,
        pixel_values: Any,
        image_grid_thw: Any,
        *,
        max_new_tokens: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> tuple[Any, dict[str, Any]]:
        if int(input_ids.shape[0]) != 1:
            raise ValueError(f"compiled single-batch decode expects batch size 1, got {int(input_ids.shape[0])}")
        cache_length = self.resolve_cache_length(input_ids, max_new_tokens)
        self.require_cache_capacity(input_ids, cache_length, int(max_new_tokens))
        compile_warmup_meta = self.ensure_compiled_decode_warm(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            cache_length=cache_length,
        )
        compiled_decode, compile_meta = self.compiled_decode_for(batch_size=1, cache_length=cache_length)

        maybe_sync_device(self.model.device)
        prefill_start = time.perf_counter()
        prefill = self.model.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        maybe_sync_device(self.model.device)
        prefill_s = time.perf_counter() - prefill_start

        next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == int(eos_token_id)
        cache_position = prefill.next_cache_position.clone()
        cache_position_start = int(cache_position[0].detach().cpu().item())
        flat_cache = prefill.cache.flat_tensors()
        decode_calls = 0

        maybe_sync_device(self.model.device)
        decode_start = time.perf_counter()
        for _step in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            logits = compiled_decode(next_token, cache_position, prefill.rope_deltas, *flat_cache)
            decode_calls += 1
            next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(
                finished.view(-1, 1),
                torch.full_like(next_token, int(pad_token_id)),
                next_token,
            )
            generated.append(next_token)
            finished = finished | (next_token.squeeze(1) == int(eos_token_id))
            cache_position.add_(1)
        maybe_sync_device(self.model.device)
        decode_s = time.perf_counter() - decode_start

        ids = torch.cat(generated, dim=1)
        return ids, {
            "enabled": True,
            "scope": "recognition_decode_only",
            "cache_length": int(cache_length),
            "cache_position_start": int(cache_position_start),
            "compile_warmup": compile_warmup_meta,
            "compile_wrapper_s": float(compile_warmup_meta.get("compile_wrapper_s", 0.0)),
            "compiled_first_call_s": float(compile_warmup_meta.get("first_call_s", 0.0)),
            "prefill_s": float(prefill_s),
            "decode_s": float(decode_s),
            "decode_calls": int(decode_calls),
            "generated_new_tokens": int(ids.shape[1]),
            "decode_tok_s": float(decode_calls / decode_s) if decode_s > 0 else 0.0,
            "stop_on_eos": True,
            "compile": dict(compile_meta),
        }

    @torch.inference_mode()
    def benchmark_decode(
        self,
        input_ids: Any,
        attention_mask: Any,
        pixel_values: Any,
        image_grid_thw: Any,
        *,
        warmup_steps: int,
        measure_steps: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> dict[str, Any]:
        del eos_token_id, pad_token_id
        if int(input_ids.shape[0]) != 1:
            raise ValueError(f"compiled single-batch decode expects batch size 1, got {int(input_ids.shape[0])}")
        measure_steps = max(0, int(measure_steps))
        warmup_steps = max(0, int(warmup_steps))
        cache_length = self.resolve_cache_length(input_ids, max(measure_steps + 1, warmup_steps + 1))
        self.require_cache_capacity(input_ids, cache_length, max(measure_steps, warmup_steps))
        compile_warmup_meta = self.ensure_compiled_decode_warm(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            cache_length=cache_length,
        )
        compiled_decode, compile_meta = self.compiled_decode_for(batch_size=1, cache_length=cache_length)

        warm_prefill = self.model.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        warm_next = torch.argmax(warm_prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        warm_cache_position = warm_prefill.next_cache_position.clone()
        warm_flat_cache = warm_prefill.cache.flat_tensors()
        for step in range(warmup_steps):
            logits = compiled_decode(warm_next, warm_cache_position, warm_prefill.rope_deltas, *warm_flat_cache)
            warm_next = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            warm_cache_position.add_(1)

        maybe_sync_device(self.model.device)
        prefill_start = time.perf_counter()
        prefill = self.model.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        maybe_sync_device(self.model.device)
        prefill_s = time.perf_counter() - prefill_start

        next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
        cache_position = prefill.next_cache_position.clone()
        flat_cache = prefill.cache.flat_tensors()

        maybe_sync_device(self.model.device)
        decode_start = time.perf_counter()
        for _step in range(measure_steps):
            logits = compiled_decode(next_token, cache_position, prefill.rope_deltas, *flat_cache)
            next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            cache_position.add_(1)
        maybe_sync_device(self.model.device)
        decode_s = time.perf_counter() - decode_start
        return {
            "enabled": True,
            "scope": "compiled_static_recognition_decode_only",
            "warmup_decode_steps": int(warmup_steps),
            "measured_decode_steps": int(measure_steps),
            "compile_warmup": compile_warmup_meta,
            "compile_first_call_s": None
            if not compile_warmup_meta.get("ran_this_call", False)
            else float(compile_warmup_meta.get("first_call_s", 0.0)),
            "prefill_s": float(prefill_s),
            "decode_s": float(decode_s),
            "decode_tok_s": float(measure_steps / decode_s) if decode_s > 0 else 0.0,
            "raw_decode_forward_calls": int(measure_steps),
            "stop_on_eos": False,
            "cache_length": int(cache_length),
            "compile": dict(compile_meta),
        }


class LocalMinerUModelPredictor:
    def __init__(
        self,
        model: LocalMinerU2_5ForConditionalGeneration,
        processor: Any,
        *,
        max_new_tokens: int,
        compiled_decode: CompiledSingleBatchRecognitionDecoder,
        benchmark_decode: bool = False,
        decode_warmup_steps: int = 8,
        decode_measure_steps: int = 64,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.compiled_decode = compiled_decode
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

    def predict(self, image: Image.Image, prompt: str, *, use_compiled_recognition_decode: bool = False) -> dict[str, Any]:
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
            eos_token_id = int(getattr(self.processor.tokenizer, "eos_token_id", self.model.config.eos_token_id))
            pad_token_id = int(getattr(self.processor.tokenizer, "pad_token_id", self.model.config.pad_token_id))
            if use_compiled_recognition_decode:
                generated_tensor, compiled_decode_meta = self.compiled_decode.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=getattr(inputs, "pixel_values", None),
                    image_grid_thw=getattr(inputs, "image_grid_thw", None),
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            else:
                generated_tensor = self.model.generate_ids(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=getattr(inputs, "pixel_values", None),
                    image_grid_thw=getattr(inputs, "image_grid_thw", None),
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                compiled_decode_meta = {"enabled": False}
        generate_s = time.perf_counter() - start
        decode_benchmark: dict[str, Any] = {"enabled": False}
        if self.benchmark_decode:
            if use_compiled_recognition_decode:
                decode_benchmark = self.compiled_decode.benchmark_decode(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=getattr(inputs, "pixel_values", None),
                    image_grid_thw=getattr(inputs, "image_grid_thw", None),
                    warmup_steps=self.decode_warmup_steps,
                    measure_steps=self.decode_measure_steps,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            else:
                decode_benchmark = self.model.benchmark_decode(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=getattr(inputs, "pixel_values", None),
                    image_grid_thw=getattr(inputs, "image_grid_thw", None),
                    warmup_steps=self.decode_warmup_steps,
                    measure_steps=self.decode_measure_steps,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

        validation: dict[str, Any] = {"enabled": False}
        if use_compiled_recognition_decode:
            with torch.inference_mode():
                eager_reference = self.model.generate_ids(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=getattr(inputs, "pixel_values", None),
                    image_grid_thw=getattr(inputs, "image_grid_thw", None),
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            compiled_ids = generated_tensor.detach().cpu().tolist()[0]
            eager_ids = eager_reference.detach().cpu().tolist()[0]
            compiled_trimmed = trim_ids_at_eos(compiled_ids, eos_token_id)
            eager_trimmed = trim_ids_at_eos(eager_ids, eos_token_id)
            validation = {
                "enabled": True,
                "reference": "dynamic_eager_generate_ids",
                "trimmed_token_match": compiled_trimmed == eager_trimmed,
                "compiled_trimmed_token_count": len(compiled_trimmed),
                "eager_trimmed_token_count": len(eager_trimmed),
                "compiled_generated_token_count": len(compiled_ids),
                "eager_generated_token_count": len(eager_ids),
                "first_mismatch_index": next(
                    (
                        idx
                        for idx, (left, right) in enumerate(zip(compiled_trimmed, eager_trimmed))
                        if left != right
                    ),
                    None if len(compiled_trimmed) == len(eager_trimmed) else min(len(compiled_trimmed), len(eager_trimmed)),
                ),
            }

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
            "compiled_decode": compiled_decode_meta,
            "validation": validation,
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
        prediction = self.predictor.predict(layout_image, prompt, use_compiled_recognition_decode=False)
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
            "compiled_decode": prediction["compiled_decode"],
            "validation": prediction["validation"],
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
        prediction = self.predictor.predict(crop, prompt, use_compiled_recognition_decode=True)
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
            "compiled_decode": prediction["compiled_decode"],
            "validation": prediction["validation"],
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
    parser.add_argument("--cache-length", type=int, default=None, help="Static KV cache length for compiled recognition decode; defaults to input tokens + max new tokens.")
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--benchmark-decode", action="store_true", help="Run a separate warmed decode-only tok/s benchmark for each generation call.")
    parser.add_argument("--decode-warmup-steps", type=int, default=8)
    parser.add_argument("--decode-measure-steps", type=int, default=64)
    parser.add_argument("--hash-model-files", action="store_true", help="Compute sha256 for safetensors files for model-version audit. This may add startup wall time.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-selected-crop", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
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
    compiled_decoder = CompiledSingleBatchRecognitionDecoder(
        model,
        cache_root=args.torchair_cache_dir,
        cache_length=args.cache_length,
    )
    predictor = LocalMinerUModelPredictor(
        model,
        processor,
        max_new_tokens=args.max_new_tokens,
        compiled_decode=compiled_decoder,
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
    canonical_reference = canonical_crop_01_reference_validation(image_path, recognition)
    payload = {
        "experiment": "03_compiled_single_batch_decode",
        "scope": "Manual local MinerU two-step protocol; layout stays dynamic eager, recognition uses static-cache compiled single-batch decode.",
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
        "recognition_compiled_decode": {
            "enabled": True,
            "batch_size": 1,
            "cache_length_arg": None if args.cache_length is None else int(args.cache_length),
            "torchair_cache_dir": str(args.torchair_cache_dir),
            "layout_decode": "dynamic_eager",
            "recognition_decode": "static_cache_compiled",
        },
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
            "compiled_decode": layout["compiled_decode"],
            "validation": layout["validation"],
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
            "compiled_decode": recognition["compiled_decode"],
            "validation": recognition["validation"],
            "canonical_reference": canonical_reference,
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
