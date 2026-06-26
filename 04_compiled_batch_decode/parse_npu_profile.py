#!/usr/bin/env python3
"""Parse torch_npu profiler output from the MinerU single-batch decode lane.

The parser intentionally stays model-agnostic: it summarizes the CANN CSVs and
trace JSON emitted by torch_npu.profiler/tensorboard_trace_handler rather than
using PaddleOCR-VL-specific bucket names.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SUSPECT_TERMS = (
    "transdata",
    "cast",
    "scatter",
    "incre",
    "flash",
    "attention",
    "matmul",
    "batchmatmul",
    "softmax",
    "rms",
    "norm",
    "gelu",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--topn", type=int, default=25)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--skip-trace", action="store_true", help="Skip trace_view.json parsing for very large traces.")
    return parser.parse_args()


def parse_float(raw: str | None) -> float:
    if raw is None:
        return 0.0
    cleaned = str(raw).replace("\t", "").strip()
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_int(raw: str | None) -> int:
    if raw is None:
        return 0
    try:
        return int(float(str(raw).strip() or "0"))
    except ValueError:
        return 0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def top_items(mapping: dict[str, dict[str, Any]], *, key: str, topn: int) -> list[dict[str, Any]]:
    return sorted(mapping.values(), key=lambda item: float(item.get(key, 0.0)), reverse=True)[:topn]


def add_sample(bucket: dict[str, Any], field: str, value: str | None, *, limit: int = 3) -> None:
    if not value:
        return
    samples = bucket.setdefault(field, [])
    if value not in samples and len(samples) < limit:
        samples.append(value)


def operator_duration_us(row: dict[str, str]) -> float:
    return max(
        parse_float(row.get("Device Total Duration(us)")),
        parse_float(row.get("Device Total Duration With AICore(us)")),
        parse_float(row.get("Device Self Duration(us)")),
        parse_float(row.get("Host Total Duration(us)")),
    )


def summarize_operator_details(path: Path, *, topn: int) -> dict[str, Any]:
    rows = read_csv_rows(path)
    by_name: dict[str, dict[str, Any]] = {}
    total_device_us = 0.0
    total_host_us = 0.0
    for row in rows:
        name = row.get("Name") or row.get("Op Name") or "unknown"
        device_us = operator_duration_us(row)
        host_us = parse_float(row.get("Host Total Duration(us)"))
        total_device_us += device_us
        total_host_us += host_us
        bucket = by_name.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "device_total_us": 0.0,
                "host_total_us": 0.0,
                "input_shape_samples": [],
                "call_stack_samples": [],
            },
        )
        bucket["count"] += 1
        bucket["device_total_us"] += device_us
        bucket["host_total_us"] += host_us
        add_sample(bucket, "input_shape_samples", row.get("Input Shapes"))
        add_sample(bucket, "call_stack_samples", row.get("Call Stack"), limit=2)
    return {
        "path": str(path),
        "row_count": len(rows),
        "total_device_us": total_device_us,
        "total_host_us": total_host_us,
        "top_by_device_total_us": top_items(by_name, key="device_total_us", topn=topn),
        "top_by_host_total_us": top_items(by_name, key="host_total_us", topn=topn),
    }


def kernel_duration_us(row: dict[str, str]) -> float:
    return max(parse_float(row.get("Duration(us)")), parse_float(row.get("Task Duration(us)")))


def weighted_avg(total_weighted: float, total_weight: float) -> float:
    return total_weighted / total_weight if total_weight > 0 else 0.0


def summarize_kernel_details(path: Path, *, topn: int) -> dict[str, Any]:
    rows = read_csv_rows(path)
    by_name: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    by_core: dict[str, dict[str, Any]] = {}
    by_format: dict[str, dict[str, Any]] = {}
    by_dtype: dict[str, dict[str, Any]] = {}
    shape_signatures: dict[str, dict[str, Any]] = {}
    matmul_names: dict[str, dict[str, Any]] = {}
    matmul_shape_signatures: dict[str, dict[str, Any]] = {}
    transdata_names: dict[str, dict[str, Any]] = {}
    transdata_shape_signatures: dict[str, dict[str, Any]] = {}
    suspect_rows: list[dict[str, Any]] = []
    total_duration_us = 0.0
    total_wait_us = 0.0
    total_aicore_us = 0.0
    cube_weighted = 0.0
    cube_weight = 0.0

    def add_group(mapping: dict[str, dict[str, Any]], key: str, duration_us: float, row: dict[str, str]) -> None:
        bucket = mapping.setdefault(
            key,
            {
                "name": key,
                "count": 0,
                "duration_us": 0.0,
                "wait_us": 0.0,
                "aicore_time_us": 0.0,
                "kernel_name_samples": [],
                "input_shape_samples": [],
                "output_shape_samples": [],
                "input_format_samples": [],
                "output_format_samples": [],
                "input_dtype_samples": [],
            },
        )
        bucket["count"] += 1
        bucket["duration_us"] += duration_us
        bucket["wait_us"] += parse_float(row.get("Wait Time(us)"))
        bucket["aicore_time_us"] += parse_float(row.get("aicore_time(us)"))
        add_sample(bucket, "kernel_name_samples", row.get("Name") or row.get("Task Name"))
        add_sample(bucket, "input_shape_samples", row.get("Input Shapes"))
        add_sample(bucket, "output_shape_samples", row.get("Output Shapes"))
        add_sample(bucket, "input_format_samples", row.get("Input Formats"))
        add_sample(bucket, "output_format_samples", row.get("Output Formats"))
        add_sample(bucket, "input_dtype_samples", row.get("Input Data Types"))

    for row in rows:
        duration_us = kernel_duration_us(row)
        total_duration_us += duration_us
        total_wait_us += parse_float(row.get("Wait Time(us)"))
        total_aicore_us += parse_float(row.get("aicore_time(us)"))
        cube = parse_float(row.get("cube_utilization(%)"))
        if cube > 0 and duration_us > 0:
            cube_weighted += cube * duration_us
            cube_weight += duration_us

        name = row.get("Name") or row.get("Task Name") or "unknown"
        op_type = row.get("Type") or row.get("Op Type") or row.get("Task Type") or "unknown"
        core = row.get("Accelerator Core") or row.get("Core Type") or "unknown"
        input_formats = row.get("Input Formats") or "unknown"
        input_dtypes = row.get("Input Data Types") or "unknown"
        shape_key = f"{op_type} | {row.get('Input Shapes') or ''} -> {row.get('Output Shapes') or ''}"
        shape_format_key = (
            f"{op_type} | {row.get('Input Shapes') or ''} -> {row.get('Output Shapes') or ''} | "
            f"{row.get('Input Formats') or ''} -> {row.get('Output Formats') or ''}"
        )

        add_group(by_name, name, duration_us, row)
        add_group(by_type, op_type, duration_us, row)
        add_group(by_core, core, duration_us, row)
        add_group(by_format, input_formats, duration_us, row)
        add_group(by_dtype, input_dtypes, duration_us, row)
        add_group(shape_signatures, shape_key, duration_us, row)

        haystack = " ".join(
            [name, op_type]
            + [str(row.get(field) or "") for field in ("Task Name", "Input Formats", "Output Formats")]
        ).lower()
        if "matmul" in haystack:
            add_group(matmul_names, name, duration_us, row)
            add_group(matmul_shape_signatures, shape_format_key, duration_us, row)
        if "transdata" in haystack:
            add_group(transdata_names, name, duration_us, row)
            add_group(transdata_shape_signatures, shape_format_key, duration_us, row)
        if any(term in haystack for term in SUSPECT_TERMS):
            suspect_rows.append(
                {
                    "name": name,
                    "type": op_type,
                    "count": 1,
                    "duration_us": duration_us,
                    "wait_us": parse_float(row.get("Wait Time(us)")),
                    "aicore_time_us": parse_float(row.get("aicore_time(us)")),
                    "input_shapes": row.get("Input Shapes"),
                    "input_dtypes": row.get("Input Data Types"),
                    "input_formats": row.get("Input Formats"),
                    "output_shapes": row.get("Output Shapes"),
                    "output_formats": row.get("Output Formats"),
                }
            )

    suspect_rows = sorted(suspect_rows, key=lambda item: float(item["duration_us"]), reverse=True)[:topn]
    return {
        "path": str(path),
        "row_count": len(rows),
        "total_duration_us": total_duration_us,
        "total_wait_us": total_wait_us,
        "total_aicore_time_us": total_aicore_us,
        "weighted_cube_utilization_pct": weighted_avg(cube_weighted, cube_weight),
        "top_kernel_names": top_items(by_name, key="duration_us", topn=topn),
        "top_kernel_types": top_items(by_type, key="duration_us", topn=topn),
        "top_core_types": top_items(by_core, key="duration_us", topn=topn),
        "top_input_formats": top_items(by_format, key="duration_us", topn=topn),
        "top_input_dtypes": top_items(by_dtype, key="duration_us", topn=topn),
        "top_shape_signatures": top_items(shape_signatures, key="duration_us", topn=topn),
        "top_matmul_names": top_items(matmul_names, key="duration_us", topn=topn),
        "top_matmul_shape_signatures": top_items(matmul_shape_signatures, key="duration_us", topn=topn),
        "top_transdata_names": top_items(transdata_names, key="duration_us", topn=topn),
        "top_transdata_shape_signatures": top_items(transdata_shape_signatures, key="duration_us", topn=topn),
        "suspect_kernel_rows": suspect_rows,
    }


def summarize_op_statistic(path: Path, *, topn: int) -> dict[str, Any]:
    rows = read_csv_rows(path)
    entries = []
    for row in rows:
        entries.append(
            {
                "op_type": row.get("OP Type") or row.get("Op Type") or row.get("Type") or "unknown",
                "core_type": row.get("Core Type") or "",
                "count": parse_int(row.get("Count")),
                "total_time_us": parse_float(row.get("Total Time(us)")),
                "avg_time_us": parse_float(row.get("Avg Time(us)")),
                "ratio_pct": parse_float(row.get("Ratio(%)")),
            }
        )
    return {
        "path": str(path),
        "row_count": len(rows),
        "top_op_types": sorted(entries, key=lambda item: item["total_time_us"], reverse=True)[:topn],
    }


def summarize_api_statistic(path: Path, *, topn: int) -> dict[str, Any]:
    rows = read_csv_rows(path)
    entries = []
    for row in rows:
        entries.append(
            {
                "level": row.get("Level") or "",
                "api_name": row.get("API Name") or "unknown",
                "count": parse_int(row.get("Count")),
                "time_us": parse_float(row.get("Time(us)")),
                "avg_us": parse_float(row.get("Avg(us)")),
                "max_us": parse_float(row.get("Max(us)")),
            }
        )
    return {
        "path": str(path),
        "row_count": len(rows),
        "top_apis": sorted(entries, key=lambda item: item["time_us"], reverse=True)[:topn],
    }


def summarize_step_trace(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    numeric_totals: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            if key in {"Device_id", "Step"}:
                continue
            numeric_totals[key] += parse_float(value)
    return {
        "path": str(path),
        "row_count": len(rows),
        "totals_us": dict(sorted(numeric_totals.items())),
    }


def summarize_trace_view(path: Path, *, topn: int) -> dict[str, Any]:
    raw = load_json(path)
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    by_name: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        duration_us = parse_float(str(event.get("dur") or 0.0))
        name = str(event.get("name") or "unknown")
        category = str(event.get("cat") or "unknown")
        for mapping, key in ((by_name, name), (by_category, category)):
            bucket = mapping.setdefault(key, {"name": key, "count": 0, "duration_us": 0.0})
            bucket["count"] += 1
            bucket["duration_us"] += duration_us
    return {
        "path": str(path),
        "event_count": len(events),
        "top_trace_events": top_items(by_name, key="duration_us", topn=topn),
        "top_trace_categories": top_items(by_category, key="duration_us", topn=topn),
    }


def find_run_roots(profile_dir: Path) -> list[Path]:
    roots = set()
    for output_dir in profile_dir.glob("**/ASCEND_PROFILER_OUTPUT"):
        roots.add(output_dir.parent)
    if (profile_dir / "ASCEND_PROFILER_OUTPUT").exists():
        roots.add(profile_dir)
    return sorted(roots)


def parse_run(run_root: Path, *, topn: int, skip_trace: bool) -> dict[str, Any]:
    output_dir = run_root / "ASCEND_PROFILER_OUTPUT"
    result: dict[str, Any] = {
        "run_root": str(run_root),
        "ascend_output_dir": str(output_dir),
        "files": {},
    }

    paths = {
        "operator_details": output_dir / "operator_details.csv",
        "kernel_details": output_dir / "kernel_details.csv",
        "op_statistic": output_dir / "op_statistic.csv",
        "api_statistic": output_dir / "api_statistic.csv",
        "step_trace_time": output_dir / "step_trace_time.csv",
        "trace_view": output_dir / "trace_view.json",
        "profiler_info": run_root / "profiler_info.json",
        "profiler_metadata": run_root / "profiler_metadata.json",
    }
    result["files"] = {key: str(path) for key, path in paths.items() if path.exists()}

    if paths["operator_details"].exists():
        result["operator_details"] = summarize_operator_details(paths["operator_details"], topn=topn)
    if paths["kernel_details"].exists():
        result["kernel_details"] = summarize_kernel_details(paths["kernel_details"], topn=topn)
    if paths["op_statistic"].exists():
        result["op_statistic"] = summarize_op_statistic(paths["op_statistic"], topn=topn)
    if paths["api_statistic"].exists():
        result["api_statistic"] = summarize_api_statistic(paths["api_statistic"], topn=topn)
    if paths["step_trace_time"].exists():
        result["step_trace_time"] = summarize_step_trace(paths["step_trace_time"])
    if paths["trace_view"].exists() and not skip_trace:
        result["trace_view"] = summarize_trace_view(paths["trace_view"], topn=topn)
    if paths["profiler_info"].exists():
        result["profiler_info"] = load_json(paths["profiler_info"])
    return result


def render_table(rows: list[dict[str, Any]], *, name_key: str, value_key: str, limit: int) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| name | count | total_us |", "|---|---:|---:|"]
    for row in rows[:limit]:
        name = str(row.get(name_key) or row.get("name") or "unknown").replace("\n", " ")[:120]
        count = int(row.get("count") or 0)
        value = float(row.get(value_key) or 0.0)
        lines.append(f"| `{name}` | {count} | {value:.3f} |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, summary: dict[str, Any], *, topn: int) -> None:
    lines = ["# NPU Profile Summary", ""]
    lines.append(f"profile_dir: `{summary['profile_dir']}`")
    lines.append(f"runs: `{len(summary['runs'])}`")
    lines.append("")
    for idx, run in enumerate(summary["runs"], start=1):
        lines.append(f"## Run {idx}")
        lines.append(f"run_root: `{run['run_root']}`")
        if "step_trace_time" in run:
            lines.append("")
            lines.append("### Step Trace Totals")
            for key, value in run["step_trace_time"]["totals_us"].items():
                lines.append(f"- `{key}`: `{float(value):.3f} us`")
        if "kernel_details" in run:
            kernel = run["kernel_details"]
            lines.append("")
            lines.append("### Kernel Types")
            lines.append(render_table(kernel["top_kernel_types"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### Kernel Names")
            lines.append(render_table(kernel["top_kernel_names"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### MatMul Names")
            lines.append(render_table(kernel["top_matmul_names"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### MatMul Shape And Format Signatures")
            lines.append(render_table(kernel["top_matmul_shape_signatures"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### TransData Names")
            lines.append(render_table(kernel["top_transdata_names"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### TransData Shape And Format Signatures")
            lines.append(render_table(kernel["top_transdata_shape_signatures"], name_key="name", value_key="duration_us", limit=topn))
            lines.append("### Suspect Kernels")
            lines.append(render_table(kernel["suspect_kernel_rows"], name_key="name", value_key="duration_us", limit=topn))
        if "operator_details" in run:
            operators = run["operator_details"]
            lines.append("")
            lines.append("### Operators")
            lines.append(render_table(operators["top_by_device_total_us"], name_key="name", value_key="device_total_us", limit=topn))
        if "api_statistic" in run:
            lines.append("")
            lines.append("### APIs")
            lines.append(render_table(run["api_statistic"]["top_apis"], name_key="api_name", value_key="time_us", limit=topn))
        if "trace_view" in run:
            lines.append("")
            lines.append("### Trace Events")
            lines.append(render_table(run["trace_view"]["top_trace_events"], name_key="name", value_key="duration_us", limit=topn))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.expanduser().resolve()
    run_roots = find_run_roots(profile_dir)
    if not run_roots:
        raise RuntimeError(f"No ASCEND_PROFILER_OUTPUT directories found under {profile_dir}")

    summary = {
        "profile_dir": str(profile_dir),
        "topn": int(args.topn),
        "skip_trace": bool(args.skip_trace),
        "runs": [parse_run(root, topn=int(args.topn), skip_trace=bool(args.skip_trace)) for root in run_roots],
    }
    out_json = args.out_json or (profile_dir / "profile_parse_summary.json")
    out_md = args.out_md or (profile_dir / "profile_parse_summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_md, summary, topn=int(args.topn))
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "runs": len(run_roots)}, indent=2))


if __name__ == "__main__":
    main()
