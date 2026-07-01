"""
tests/test_compare_reference.py

Batch runner for reference-guided vision comparison.
This file only coordinates batch image collection, reference lookup, analyzer calls, evaluation, and report writing. All model work is delegated to utils.current_state_analyzer.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.current_state_analyzer import (  # noqa: E402
    GROUND_TRUTH_DIR,
    IMAGE_EXTENSIONS,
    analyze_image,
    calculate_decision_level,
    max_confidence,
    parse_filename,
    safe_relative_path,
    save_json,
)

INPUT_DIR = PROJECT_ROOT / "input"
LOGS_DIR = PROJECT_ROOT / "logs"
COMPARE_PARSED_DIR = LOGS_DIR / "compare_parsed_json"
COMPARE_FAILED_DIR = LOGS_DIR / "compare_parse_failed"
SUMMARY_DIR = LOGS_DIR / "compare_summaries"


def collect_images(input_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def find_reference_image(filename_info: dict[str, str]) -> Path:
    model_id = filename_info.get("model_id", "")
    step_id = filename_info.get("step_id", "")
    view_angle = filename_info.get("view_angle", "")

    if not model_id or not step_id or not view_angle:
        raise ValueError("Cannot find reference image because filename info is incomplete.")

    normal_dir = INPUT_DIR / "normal" / f"{model_id}_{step_id}"
    pattern = f"{model_id}_{step_id}_correct-*_{view_angle}_*"
    candidates: list[Path] = []

    search_roots = [normal_dir, INPUT_DIR / "normal"]
    for root in search_roots:
        if not root.exists():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidates.extend(root.rglob(f"{pattern}{ext}"))
        if candidates:
            break

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "Correct reference image not found for "
            f"{model_id} {step_id} {view_angle}. Pattern: {pattern}"
        )

    for candidate in candidates:
        if "correct-01" in candidate.stem:
            return candidate
    return candidates[0]


def expected_state_path(filename_info: dict[str, str]) -> Path:
    model_id = filename_info.get("model_id", "")
    step_id = filename_info.get("step_id", "")
    path = GROUND_TRUTH_DIR / model_id / f"{step_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected state JSON not found: {path}")
    return path


def compare_log_path(directory: Path, image_path: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{image_path.stem}_{suffix}_{timestamp}.json"


def build_compare_result(
    image_path: Path,
    filename_info: dict[str, str],
    reference_image_path: Path,
    expected_path: Path,
    analyzer_result: dict[str, Any],
) -> dict[str, Any]:
    model_response = analyzer_result.get("model_response") or {}
    ground_truth = filename_info.get("ground_truth", "")
    is_error = model_response.get("is_error", "")
    decision_level = calculate_decision_level(ground_truth, is_error)

    return {
        "image_name": image_path.name,
        "image_path": safe_relative_path(image_path),
        "status": "success",
        "model_id": filename_info.get("model_id", ""),
        "step_id": filename_info.get("step_id", ""),
        "view_angle": filename_info.get("view_angle", ""),
        "ground_truth": ground_truth,
        "gpt_result": model_response.get("overall_error_type", ""),
        "is_error": is_error,
        "decision_level": decision_level,
        "confidence": max_confidence(model_response),
        "reference_image": reference_image_path.name,
        "reference_image_path": safe_relative_path(reference_image_path),
        "expected_state_path": safe_relative_path(expected_path),
        "model_response": model_response,
        "raw_response_path": analyzer_result.get("raw_response_path", ""),
        "current_parsed_json_path": analyzer_result.get("parsed_json_path", ""),
        "response_time_sec": analyzer_result.get("response_time_sec", 0),
        "attempt": analyzer_result.get("attempt", 0),
    }


def build_failed_result(
    image_path: Path,
    filename_info: dict[str, str],
    error: str,
    detail: str = "",
    reference_image_path: Path | None = None,
    expected_path: Path | None = None,
    analyzer_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analyzer_result = analyzer_result or {}
    return {
        "image_name": image_path.name,
        "image_path": safe_relative_path(image_path),
        "status": "failed",
        "model_id": filename_info.get("model_id", ""),
        "step_id": filename_info.get("step_id", ""),
        "view_angle": filename_info.get("view_angle", ""),
        "ground_truth": filename_info.get("ground_truth", ""),
        "gpt_result": "",
        "is_error": "",
        "decision_level": "",
        "confidence": 0.0,
        "reference_image": reference_image_path.name if reference_image_path else "",
        "reference_image_path": safe_relative_path(reference_image_path) if reference_image_path else "",
        "expected_state_path": safe_relative_path(expected_path) if expected_path else "",
        "model_response": analyzer_result.get("model_response"),
        "raw_response_path": analyzer_result.get("raw_response_path", ""),
        "current_failed_path": analyzer_result.get("failed_path", ""),
        "error": error,
        "detail": detail,
    }


def analyze_single_image(image_path: Path) -> dict[str, Any]:
    filename_info = parse_filename(image_path)
    if not filename_info.get("model_id") or not filename_info.get("step_id"):
        raise ValueError(f"Cannot parse model_id and step_id from filename: {image_path.name}")

    reference_image_path = find_reference_image(filename_info)
    expected_path = expected_state_path(filename_info)

    analyzer_result = analyze_image(
        image_path=str(image_path),
        reference_image_path=str(reference_image_path),
        expected_state_path=str(expected_path),
        filename_info=filename_info,
    )

    if analyzer_result.get("success"):
        result = build_compare_result(
            image_path=image_path,
            filename_info=filename_info,
            reference_image_path=reference_image_path,
            expected_path=expected_path,
            analyzer_result=analyzer_result,
        )
        compare_path = compare_log_path(COMPARE_PARSED_DIR, image_path, "compare_parsed")
        save_json(result, compare_path)
        result["compare_parsed_json_path"] = safe_relative_path(compare_path)
        return result

    result = build_failed_result(
        image_path=image_path,
        filename_info=filename_info,
        reference_image_path=reference_image_path,
        expected_path=expected_path,
        analyzer_result=analyzer_result,
        error=analyzer_result.get("error", "Analyzer failed"),
        detail=analyzer_result.get("detail", ""),
    )
    failed_path = compare_log_path(COMPARE_FAILED_DIR, image_path, "compare_failed")
    save_json(result, failed_path)
    result["compare_failed_path"] = safe_relative_path(failed_path)
    return result


def metric_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for result in results:
        level = result.get("decision_level")
        if level in counts:
            counts[level] += 1
    return counts


def save_summary_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "model_id",
        "step_id",
        "view_angle",
        "ground_truth",
        "gpt_result",
        "is_error",
        "decision_level",
        "confidence",
        "status",
        "reference_image",
        "raw_response_path",
        "parsed_or_failed_path",
        "error",
        "detail",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "image_name": result.get("image_name", ""),
                "model_id": result.get("model_id", ""),
                "step_id": result.get("step_id", ""),
                "view_angle": result.get("view_angle", ""),
                "ground_truth": result.get("ground_truth", ""),
                "gpt_result": result.get("gpt_result", ""),
                "is_error": result.get("is_error", ""),
                "decision_level": result.get("decision_level", ""),
                "confidence": result.get("confidence", ""),
                "status": result.get("status", ""),
                "reference_image": result.get("reference_image", ""),
                "raw_response_path": result.get("raw_response_path", ""),
                "parsed_or_failed_path": result.get("compare_parsed_json_path") or result.get("compare_failed_path", ""),
                "error": result.get("error", ""),
                "detail": result.get("detail", ""),
            })


def main() -> None:
    for directory in (COMPARE_PARSED_DIR, COMPARE_FAILED_DIR, SUMMARY_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(INPUT_DIR)
    if not image_files:
        raise FileNotFoundError(f"No jpg/png images found under {INPUT_DIR}")

    results: list[dict[str, Any]] = []
    print(f"Found {len(image_files)} images under input/.")

    for index, image_path in enumerate(image_files, start=1):
        print(f"[{index}/{len(image_files)}] {safe_relative_path(image_path)}")
        try:
            result = analyze_single_image(image_path)
        except Exception as exc:
            filename_info = parse_filename(image_path)
            result = build_failed_result(image_path, filename_info, error="Batch item failed", detail=str(exc))
            failed_path = compare_log_path(COMPARE_FAILED_DIR, image_path, "compare_failed")
            save_json(result, failed_path)
            result["compare_failed_path"] = safe_relative_path(failed_path)

        results.append(result)
        print(
            "  "
            f"status={result.get('status')} "
            f"ground_truth={result.get('ground_truth')} "
            f"gpt_result={result.get('gpt_result')} "
            f"decision={result.get('decision_level')} "
            f"confidence={result.get('confidence')}"
        )
        time.sleep(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counts = metric_counts(results)
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": safe_relative_path(INPUT_DIR),
        "total_images": len(results),
        "success_count": sum(1 for item in results if item.get("status") == "success"),
        "failed_count": sum(1 for item in results if item.get("status") != "success"),
        "metrics": counts,
        "results": results,
    }

    json_summary_path = SUMMARY_DIR / f"compare_summary_{timestamp}.json"
    csv_summary_path = SUMMARY_DIR / f"compare_summary_{timestamp}.csv"
    save_json(summary_payload, json_summary_path)
    save_summary_csv(results, csv_summary_path)

    print("Batch comparison complete.")
    print(f"JSON summary: {safe_relative_path(json_summary_path)}")
    print(f"CSV summary: {safe_relative_path(csv_summary_path)}")
    print(f"Metrics: {counts}")


if __name__ == "__main__":
    main()
