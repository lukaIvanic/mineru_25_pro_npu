#!/usr/bin/env python3
"""Create a larger deterministic OmniDocBench crop set for batch/hot-swap tests."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


Image.MAX_IMAGE_PIXELS = None


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "remote_artifacts/aos_research_remote_shutdown_20260531"
    / "glm_ocr_portable_bundle/data/OmniDocBench"
)
DEFAULT_OUT_DIR = WORK_DIR / "crops"
DEFAULT_MANIFEST_NAME = "hotswap_100_manifest.json"
DEFAULT_SUMMARY_NAME = "hotswap_100_summary.json"
DEFAULT_CONTACT_SHEET_NAME = "hotswap_100_contact_sheet.jpg"


CATEGORY_QUOTAS = {
    "text_block": 32,
    "title": 12,
    "equation_isolated": 14,
    "table": 10,
    "figure_caption": 8,
    "table_caption": 5,
    "header": 5,
    "footer": 4,
    "table_footnote": 4,
    "page_footnote": 2,
    "reference": 2,
    "code_txt": 2,
}

PROMPT_BY_CATEGORY = {
    "equation_isolated": "Formula Recognition:",
    "equation_semantic": "Formula Recognition:",
    "table": "Table Recognition:",
    "table_mask": "Table Recognition:",
    "chart_mask": "Chart Recognition:",
}


@dataclass(frozen=True)
class Candidate:
    page_index: int
    page: dict[str, Any]
    det: dict[str, Any]
    source_image: Path
    bbox: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    ground_truth: str
    ground_truth_source: str


def clean_slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "item"


def ground_truth_for_det(det: dict[str, Any]) -> tuple[str, str]:
    for key in ("text", "latex", "html"):
        value = det.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return "", ""


def bbox_from_poly(poly: list[float], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    xs = poly[0::2]
    ys = poly[1::2]
    left = max(0, int(min(xs)) - pad)
    top = max(0, int(min(ys)) - pad)
    right = min(width, int(max(xs) + 0.999) + pad)
    bottom = min(height, int(max(ys) + 0.999) + pad)
    return left, top, right, bottom


def suggested_prompt(category: str) -> str:
    return PROMPT_BY_CATEGORY.get(category, "OCR:")


def load_dataset(dataset_dir: Path) -> list[dict[str, Any]]:
    json_path = dataset_dir / "OmniDocBench.json"
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_candidates(dataset: list[dict[str, Any]], dataset_dir: Path, *, pad: int) -> dict[str, list[Candidate]]:
    images_dir = dataset_dir / "images"
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    image_size_cache: dict[Path, tuple[int, int]] = {}

    for page_index, page in enumerate(dataset):
        page_info = page.get("page_info", {})
        image_path = images_dir / page_info.get("image_path", "")
        if not image_path.exists():
            continue
        if image_path not in image_size_cache:
            with Image.open(image_path) as image:
                image_size_cache[image_path] = image.size
        image_width, image_height = image_size_cache[image_path]

        for det in page.get("layout_dets", []):
            category = det.get("category_type")
            if category not in CATEGORY_QUOTAS:
                continue
            if det.get("ignore"):
                continue
            poly = det.get("poly")
            if not isinstance(poly, list) or len(poly) < 8:
                continue
            ground_truth, ground_truth_source = ground_truth_for_det(det)
            if not ground_truth:
                continue

            bbox = bbox_from_poly(poly, image_width, image_height, pad)
            crop_width = bbox[2] - bbox[0]
            crop_height = bbox[3] - bbox[1]
            if crop_width < 24 or crop_height < 24:
                continue
            by_category[category].append(
                Candidate(
                    page_index=page_index,
                    page=page,
                    det=det,
                    source_image=image_path,
                    bbox=bbox,
                    crop_size=(crop_width, crop_height),
                    ground_truth=ground_truth,
                    ground_truth_source=ground_truth_source,
                )
            )

    for category in by_category:
        by_category[category].sort(
            key=lambda item: (
                item.page_index,
                int(item.det.get("order", 0) or 0),
                str(item.det.get("anno_id", "")),
            )
        )
    return by_category


def evenly_select(candidates: list[Candidate], count: int) -> list[Candidate]:
    if count <= 0:
        return []
    if len(candidates) < count:
        raise ValueError(f"requested {count} candidates but only found {len(candidates)}")
    if count == 1:
        return [candidates[len(candidates) // 2]]
    selected = []
    used = set()
    last = len(candidates) - 1
    for idx in range(count):
        pos = round(idx * last / (count - 1))
        while pos in used and pos + 1 < len(candidates):
            pos += 1
        while pos in used and pos > 0:
            pos -= 1
        used.add(pos)
        selected.append(candidates[pos])
    return selected


def make_contact_sheet(
    out_dir: Path,
    manifest: list[dict[str, Any]],
    *,
    filename: str,
    columns: int = 10,
) -> None:
    cell_w, cell_h = 280, 230
    rows = (len(manifest) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, item in enumerate(manifest):
        path = out_dir / item["file"]
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w - 28, cell_h - 58))
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        sheet.paste(img, (x + 14, y + 14))
        label = f"{idx + 1:03d} {item['category_type']}"
        draw.text((x + 14, y + cell_h - 38), label, fill=(20, 20, 20), font=font)
        draw.text((x + 14, y + cell_h - 22), item["id"], fill=(70, 70, 70), font=font)

    sheet.save(out_dir / filename, quality=90)


def write_crops(selected: list[Candidate], out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("hotswap_*.png"):
        path.unlink()

    manifest = []
    for idx, candidate in enumerate(selected, start=1):
        category = str(candidate.det.get("category_type"))
        anno_id = clean_slug(str(candidate.det.get("anno_id", f"anno_{idx}")))
        crop_id = f"hotswap_{idx:03d}_{clean_slug(category)}_p{candidate.page_index:04d}_{anno_id}"
        out_path = out_dir / f"{crop_id}.png"

        with Image.open(candidate.source_image).convert("RGB") as image:
            crop = image.crop(candidate.bbox)
            crop.save(out_path)

        page_info = candidate.page.get("page_info", {})
        page_attribute = page_info.get("page_attribute", {})
        manifest.append(
            {
                "id": crop_id,
                "file": out_path.name,
                "source_image": str(candidate.source_image.relative_to(REPO_ROOT)),
                "page_index": candidate.page_index,
                "page_no": page_info.get("page_no"),
                "page_attribute": page_attribute,
                "category_type": category,
                "anno_id": candidate.det.get("anno_id"),
                "order": candidate.det.get("order"),
                "bbox_xyxy_with_padding": list(candidate.bbox),
                "crop_size": [crop.width, crop.height],
                "suggested_prompt": suggested_prompt(category),
                "ground_truth_source": candidate.ground_truth_source,
                "ground_truth": candidate.ground_truth,
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--contact-sheet-name", default=DEFAULT_CONTACT_SHEET_NAME)
    parser.add_argument("--pad", type=int, default=12)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    dataset = load_dataset(dataset_dir)
    by_category = collect_candidates(dataset, dataset_dir, pad=int(args.pad))

    selected = []
    for category, quota in CATEGORY_QUOTAS.items():
        selected.extend(evenly_select(by_category[category], quota))

    selected.sort(
        key=lambda item: (
            str(item.det.get("category_type")),
            item.page_index,
            int(item.det.get("order", 0) or 0),
            str(item.det.get("anno_id", "")),
        )
    )
    manifest = write_crops(selected, out_dir)

    with (out_dir / args.manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary = {
        "count": len(manifest),
        "category_counts": dict(Counter(item["category_type"] for item in manifest)),
        "prompt_counts": dict(Counter(item["suggested_prompt"] for item in manifest)),
        "dataset_dir": str(dataset_dir),
        "pad": int(args.pad),
        "quotas": CATEGORY_QUOTAS,
    }
    with (out_dir / args.summary_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_contact_sheet(out_dir, manifest, filename=args.contact_sheet_name)

    print(f"Wrote {len(manifest)} crops to {out_dir}")
    print("category_counts=" + json.dumps(summary["category_counts"], sort_keys=True))
    print("prompt_counts=" + json.dumps(summary["prompt_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
