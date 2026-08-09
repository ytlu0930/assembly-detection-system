"""Build reproducible affected-part evaluation assets without API calls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_FIELDS = (
    "image_id", "file_name", "model_id", "step_id", "view_angle",
    "error_type", "case_id", "ground_truth_part_ids",
    "ground_truth_part_count", "is_composite", "review_status",
    "review_source", "notes",
)
PREDICTION_FIELDS = (
    "image_id", "case_id", "view_angle", "error_type",
    "ground_truth_part_ids", "predicted_part_ids", "predicted_confidence",
    "prediction_count", "has_unknown", "is_composite_prediction", "source_json",
)
PRIMARY_VIEWS = ("front", "back", "top", "bottom", "left", "right")


def split_ids(value: str) -> list[str]:
    return [item.strip().upper() for item in re.split(r"[;|]", value or "") if item.strip()]


def case_id_from_name(file_name: str) -> str:
    match = re.search(r"_(missingpart|extrapart|wrongpart)-([A-Z]\d+)_", file_name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    if "_correct-" in file_name:
        return "correct-control"
    raise ValueError(f"Cannot derive case_id from {file_name!r}")


def formal_error_type(value: str) -> str:
    return {"missingpart": "missing", "extrapart": "extra"}.get(value, value)


def build_ground_truth(review_csv: Path) -> list[dict[str, Any]]:
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if row.get("annotation_status") != "confirmed":
            continue
        part_ids = sorted(set(split_ids(row.get("affected_part_ids", ""))))
        result.append({
            "image_id": row["image_id"],
            "file_name": row["image_id"],
            "model_id": row["model_id"],
            "step_id": row["step_id"],
            "view_angle": row["view_angle"],
            "error_type": formal_error_type(row["overall_error_type"]),
            "case_id": case_id_from_name(row["image_id"]),
            "ground_truth_part_ids": "|".join(part_ids),
            "ground_truth_part_count": len(part_ids),
            "is_composite": str(row.get("is_composite_error", "false")).lower(),
            "review_status": "confirmed",
            "review_source": row.get("source_evidence", ""),
            "notes": row.get("review_notes", ""),
        })
    return result


def select_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one image per available case/view, capped at 24 confirmed rows."""
    by_case_view: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case_view[(row["case_id"], row["view_angle"])].append(row)
    selected = []
    for case_id in sorted({row["case_id"] for row in rows}):
        for view in PRIMARY_VIEWS:
            choices = sorted(by_case_view.get((case_id, view), []), key=lambda item: item["image_id"])
            if choices:
                selected.append(choices[0])
    return selected[:24]


def latest_baseline_json(parsed_dir: Path, image_id: str) -> Path | None:
    stem = Path(image_id).stem
    candidates = [path for path in parsed_dir.glob(f"{stem}*_parsed_20260701_*.json")]
    return max(candidates, key=lambda path: path.name) if candidates else None


def extract_prediction(path: Path) -> tuple[list[str], list[float], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = payload.get("model_response") or {}
    overall = formal_error_type(str(response.get("overall_error_type") or ""))
    if overall == "correct":
        return [], [], overall
    scores: dict[str, float] = {}
    for part in response.get("detected_parts") or []:
        part_id = str(part.get("part_id") or "").strip().upper()
        if not part_id:
            continue
        confidence = float(part.get("confidence") or 0.0)
        scores[part_id] = max(confidence, scores.get(part_id, 0.0))
    ids = sorted(scores)
    return ids, [scores[item] for item in ids], overall


def build_predictions(gt_rows: list[dict[str, Any]], parsed_dir: Path) -> list[dict[str, Any]]:
    result = []
    for gt in gt_rows:
        source = latest_baseline_json(parsed_dir, gt["image_id"])
        if source is None:
            continue
        ids, confidences, predicted_error_type = extract_prediction(source)
        result.append({
            "image_id": gt["image_id"],
            "case_id": gt["case_id"],
            "view_angle": gt["view_angle"],
            "error_type": predicted_error_type,
            "ground_truth_part_ids": gt["ground_truth_part_ids"],
            "predicted_part_ids": "|".join(ids),
            "predicted_confidence": "|".join(f"{value:.6f}" for value in confidences),
            "prediction_count": len(ids),
            "has_unknown": str(any(item.startswith(("UNKNOWN", "UNRESOLVED")) for item in ids)).lower(),
            "is_composite_prediction": str(len(ids) > 1).lower(),
            "source_json": source.as_posix(),
        })
    return result


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "analysis/affected_parts_review_template.csv")
    parser.add_argument("--parsed-dir", type=Path, default=PROJECT_ROOT / "logs/current_parsed_json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis")
    args = parser.parse_args()
    gt = build_ground_truth(args.review_csv)
    subset = select_subset(gt)
    predictions = build_predictions(gt, args.parsed_dir)
    write_csv(args.output_dir / "affected_part_eval_ground_truth.csv", GT_FIELDS, gt)
    subset_fields = ("image_id", "case_id", "view_angle", "error_type", "ground_truth_part_ids", "is_composite", "include_reason")
    subset_rows = [{**{key: row[key] for key in subset_fields if key != "include_reason"}, "include_reason": "confirmed; one image per available case/view"} for row in subset]
    write_csv(args.output_dir / "vision_ab_eval_subset.csv", subset_fields, subset_rows)
    write_csv(args.output_dir / "affected_part_baseline_predictions.csv", PREDICTION_FIELDS, predictions)
    print(json.dumps({"confirmed_ground_truth": len(gt), "evaluation_subset": len(subset), "baseline_predictions": len(predictions), "api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
