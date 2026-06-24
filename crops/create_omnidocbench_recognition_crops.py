#!/usr/bin/env python3
"""Create a small set of element-level OmniDocBench recognition crops."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    REPO_ROOT
    / "remote_artifacts/aos_research_remote_shutdown_20260531"
    / "glm_ocr_portable_bundle/data/OmniDocBench"
)
JSON_PATH = DATASET_DIR / "OmniDocBench.json"
IMAGES_DIR = DATASET_DIR / "images"
OUT_DIR = WORK_DIR / "crops"


SELECTIONS = [
    {
        "id": "crop_01_text_block_en",
        "page_index": 0,
        "category_type": "text_block",
        "text_prefix": "When an attempt",
        "suggested_prompt": "OCR:",
    },
    {
        "id": "crop_02_equation_matrix",
        "page_index": 0,
        "category_type": "equation_isolated",
        "anno_id": "box_id_1",
        "suggested_prompt": "Formula Recognition:",
    },
    {
        "id": "crop_03_code_block",
        "page_index": 1,
        "category_type": "code_txt",
        "anno_id": "box_id_3",
        "suggested_prompt": "OCR:",
    },
    {
        "id": "crop_04_handwritten_title_zh",
        "page_index": 3,
        "category_type": "title",
        "anno_id": "box_id_0",
        "suggested_prompt": "OCR:",
    },
    {
        "id": "crop_05_table_rwkv_dims",
        "page_index": 10,
        "category_type": "table",
        "anno_id": "box_id_1",
        "suggested_prompt": "Table Recognition:",
    },
    {
        "id": "crop_06_chart_cubic_spline",
        "page_index": 19,
        "category_type": "chart_mask",
        "anno_id": "box_id_0",
        "suggested_prompt": "Chart Recognition:",
    },
    {
        "id": "crop_07_figure_caption",
        "page_index": 19,
        "category_type": "figure_caption",
        "anno_id": "box_id_1",
        "suggested_prompt": "OCR:",
    },
    {
        "id": "crop_08_table_footnote_zh",
        "page_index": 106,
        "category_type": "table_footnote",
        "anno_id": "box_id_5",
        "suggested_prompt": "OCR:",
    },
]


def bbox_from_poly(poly: list[float], width: int, height: int, pad: int = 12) -> tuple[int, int, int, int]:
    xs = poly[0::2]
    ys = poly[1::2]
    left = max(0, int(min(xs)) - pad)
    top = max(0, int(min(ys)) - pad)
    right = min(width, int(max(xs) + 0.999) + pad)
    bottom = min(height, int(max(ys) + 0.999) + pad)
    return left, top, right, bottom


def find_det(page: dict, spec: dict) -> dict:
    for det in page["layout_dets"]:
        if det.get("ignore"):
            continue
        if det.get("category_type") != spec["category_type"]:
            continue
        if "anno_id" in spec and str(det.get("anno_id")) != spec["anno_id"]:
            continue
        if "text_prefix" in spec and not (det.get("text") or "").startswith(spec["text_prefix"]):
            continue
        return det
    raise RuntimeError(f"No matching detection found for {spec['id']}")


def make_contact_sheet(crop_paths: list[Path]) -> None:
    thumbs = []
    for path in crop_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 220))
        thumbs.append((path, img.copy()))

    cell_w, cell_h = 360, 275
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 4), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (path, img) in enumerate(thumbs):
        x = (idx % 2) * cell_w
        y = (idx // 2) * cell_h
        sheet.paste(img, (x + 20, y + 20))
        draw.text((x + 20, y + 235), path.stem, fill=(20, 20, 20), font=font)

    sheet.save(OUT_DIR / "contact_sheet.jpg", quality=92)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_PATH.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    manifest = []
    crop_paths = []
    for spec in SELECTIONS:
        page = dataset[spec["page_index"]]
        det = find_det(page, spec)
        source_image = IMAGES_DIR / page["page_info"]["image_path"]
        image = Image.open(source_image).convert("RGB")
        bbox = bbox_from_poly(det["poly"], image.width, image.height)
        crop = image.crop(bbox)

        out_path = OUT_DIR / f"{spec['id']}.png"
        crop.save(out_path)
        crop_paths.append(out_path)

        manifest.append(
            {
                "id": spec["id"],
                "file": out_path.name,
                "source_image": str(source_image.relative_to(REPO_ROOT)),
                "page_index": spec["page_index"],
                "page_no": page["page_info"].get("page_no"),
                "category_type": det.get("category_type"),
                "anno_id": det.get("anno_id"),
                "bbox_xyxy_with_padding": list(bbox),
                "crop_size": [crop.width, crop.height],
                "suggested_prompt": spec["suggested_prompt"],
                "ground_truth": det.get("text") or det.get("latex") or det.get("html") or "",
            }
        )

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    make_contact_sheet(crop_paths)

    print(f"Wrote {len(crop_paths)} crops to {OUT_DIR}")
    for item in manifest:
        print(f"{item['file']}: {item['category_type']} {item['crop_size']}")


if __name__ == "__main__":
    main()
