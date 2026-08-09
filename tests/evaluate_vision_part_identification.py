"""Offline evaluator for the 2026-07-01 Vision part-identification run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "current_parsed_json"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Supported by the handoff's explicit human observations and source-image review.
# Extrapart-A01 remains intentionally unlabeled because no canonical part id exists.
CASE_LABELS = {
    "missingpart-A01": (["PIN_RED_SHORT"], False, "handoff + source-image verification"),
    "missingpart-B01": (["WHEEL_BLUE_SMALL"], False, "handoff + source-image verification"),
    "wrongpart-A01": (["EYE_BALL"], False, "handoff + source-image verification; multiplicity needs review"),
    "wrongpart-B01": (["PIN_YELLOW", "PIN_RED_SHORT"], False, "handoff + source-image verification"),
    "extrapart-A01": ([], True, "extra red rod has no unambiguous canonical part id"),
}

FIELDS = [
    "image_id", "image_path", "model_id", "step_id", "view_angle",
    "ground_truth_error_type", "ground_truth_affected_parts",
    "predicted_overall_error_type", "predicted_part_ids",
    "predicted_part_descriptions", "predicted_error_types", "error_type_correct",
    "exact_part_match", "partial_part_match", "all_parts_detected",
    "hallucinated_parts", "unknown_part_count", "expected_part_count",
    "predicted_part_count", "is_composite_error", "schema_allows_multiple_parts",
    "prompt_requests_all_parts", "expected_state_contains_target",
    "part_library_contains_target", "reference_available", "failure_category", "notes",
]


def _latest_unique(paths: list[Path]) -> list[Path]:
    by_image: dict[str, Path] = {}
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        image = str(payload.get("file_info", {}).get("image_name") or path.name.split("_parsed_")[0])
        by_image[image] = path
    return sorted(by_image.values(), key=lambda item: item.name)


def _case_key(info: dict[str, Any]) -> str:
    gt = str(info.get("ground_truth", ""))
    target = str(info.get("target_part", ""))
    return f"{gt}-{target}" if target else gt


def _expected_parts(info: dict[str, Any]) -> tuple[list[str], bool, str]:
    if str(info.get("ground_truth")) == "correct":
        return [], False, "filename and source directory"
    return CASE_LABELS.get(_case_key(info), ([], True, "affected parts are not formally labeled"))


def build_rows(log_dir: Path = DEFAULT_LOG_DIR) -> list[dict[str, Any]]:
    paths = _latest_unique(list(log_dir.glob("*20260701*.json")))
    part_library = json.loads((PROJECT_ROOT / "config" / "part_library.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload.get("file_info", {})
        model = payload.get("model_response", {})
        detected = [item for item in model.get("detected_parts", []) if isinstance(item, dict)]
        errors = [item for item in detected if str(item.get("error_type", "")).lower() != "correct"]
        predicted = list(dict.fromkeys(str(item.get("part_id", "unknown_part")) for item in errors))
        expected, review_required, source_note = _expected_parts(info)
        expected_set, predicted_set = set(expected), set(predicted)
        evaluable = not review_required
        exact = predicted_set == expected_set if evaluable else None
        partial = bool(predicted_set & expected_set) if expected_set and evaluable else (exact if evaluable else None)
        all_detected = expected_set.issubset(predicted_set) if evaluable else None
        hallucinated = sorted(predicted_set - expected_set) if evaluable else []
        unknown_count = sum(part.lower().startswith("unknown") for part in predicted)
        image_path = PROJECT_ROOT / str(info.get("relative_path", ""))
        expected_path = PROJECT_ROOT / str(payload.get("expected_state_path", ""))
        expected_state = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.is_file() else {}
        expected_ids = {str(item.get("part_id")) for item in expected_state.get("expected_parts", []) if isinstance(item, dict)}
        key = _case_key(info)
        composite = key in {"wrongpart-A01", "wrongpart-B01"}
        error_correct = str(model.get("overall_error_type")) == str(info.get("ground_truth"))
        if not error_correct:
            category = "error_type_mismatch"
        elif review_required:
            category = "affected_parts_review_required"
        elif exact:
            category = "match"
        elif partial:
            category = "partial_part_match"
        elif unknown_count:
            category = "unknown_part"
        else:
            category = "wrong_part_identity"
        rows.append({
            "image_id": info.get("image_name"),
            "image_path": info.get("relative_path"),
            "model_id": info.get("model_id"),
            "step_id": info.get("step_id"),
            "view_angle": info.get("view_angle"),
            "ground_truth_error_type": info.get("ground_truth"),
            "ground_truth_affected_parts": ";".join(expected),
            "predicted_overall_error_type": model.get("overall_error_type"),
            "predicted_part_ids": ";".join(predicted),
            "predicted_part_descriptions": " | ".join(str(item.get("description", "")) for item in errors),
            "predicted_error_types": ";".join(str(item.get("error_type", "")) for item in errors),
            "error_type_correct": error_correct,
            "exact_part_match": exact,
            "partial_part_match": partial,
            "all_parts_detected": all_detected,
            "hallucinated_parts": ";".join(hallucinated),
            "unknown_part_count": unknown_count,
            "expected_part_count": len(expected),
            "predicted_part_count": len(predicted),
            "is_composite_error": composite,
            "schema_allows_multiple_parts": True,
            "prompt_requests_all_parts": False,
            "expected_state_contains_target": (all(part in expected_ids for part in expected) if expected else None),
            "part_library_contains_target": (all(part in part_library for part in expected) if expected else None),
            "reference_available": (PROJECT_ROOT / str(payload.get("reference_image", {}).get("relative_path", ""))).is_file(),
            "failure_category": category,
            "notes": f"label_source={source_note}; review_required={str(review_required).lower()}; source_log={path.name}",
        })
    return rows


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {"count": numerator, "denominator": denominator, "rate": round(numerator / denominator, 6) if denominator else None}


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["exact_part_match"] is not None]
    parts_expected = [row for row in evaluable if row["expected_part_count"] > 0]
    composites = [row for row in evaluable if row["is_composite_error"]]

    def composite_complete(row: dict[str, Any]) -> bool:
        if row["all_parts_detected"] is not True:
            return False
        predicted_types = set(str(row["predicted_error_types"]).split(";"))
        # A01 contains a wrong eye arrangement plus an extra eye; B01 is a
        # two-part swap and therefore requires both distinct identities.
        required_types = {"wrongpart", "extrapart"} if "wrongpart-A01" in str(row["image_id"]) else {"wrongpart"}
        return required_types.issubset(predicted_types)

    def group(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        return {
            name: {
                "samples": len(items),
                "error_type_accuracy": _ratio(sum(bool(item["error_type_correct"]) for item in items), len(items)),
                "affected_part_exact": _ratio(
                    sum(item["exact_part_match"] is True for item in items),
                    sum(item["exact_part_match"] is not None for item in items),
                ),
            }
            for name, items in sorted(grouped.items())
        }

    predicted_total = sum(int(row["predicted_part_count"]) for row in rows)
    unknown_total = sum(int(row["unknown_part_count"]) for row in rows)
    hallucination_rows = [row for row in evaluable if row["predicted_part_count"]]
    return {
        "sample_count": len(rows),
        "unique_image_policy": "latest filename timestamp per image_name",
        "error_type_accuracy": _ratio(sum(bool(row["error_type_correct"]) for row in rows), len(rows)),
        "affected_part_exact_match": _ratio(sum(row["exact_part_match"] is True for row in evaluable), len(evaluable)),
        "affected_part_at_least_one_match": _ratio(sum(row["partial_part_match"] is True for row in parts_expected), len(parts_expected)),
        "affected_part_all_detected": _ratio(sum(row["all_parts_detected"] is True for row in parts_expected), len(parts_expected)),
        "hallucinated_part_rate": _ratio(sum(bool(row["hallucinated_parts"]) for row in hallucination_rows), len(hallucination_rows)),
        "unknown_part_rate": _ratio(unknown_total, predicted_total),
        "composite_error_full_recall": _ratio(sum(composite_complete(row) for row in composites), len(composites)),
        "review_required_samples": sum("review_required=true" in row["notes"] for row in rows),
        "by_view_angle": group("view_angle"),
        "by_error_type": group("ground_truth_error_type"),
        "failure_categories": dict(Counter(str(row["failure_category"]) for row in rows)),
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path = DEFAULT_ANALYSIS_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vision_part_failure_analysis_20260701.csv"
    json_path = output_dir / "vision_part_failure_analysis_20260701.json"
    review_path = output_dir / "affected_parts_review_template.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = metrics(rows)
    json_path.write_text(json.dumps({"metrics": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    review_fields = [
        "image_id", "image_path", "ground_truth_error_type", "affected_parts",
        "label_source", "review_required", "reviewer", "review_notes",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for row in rows:
            review_required = "review_required=true" in str(row["notes"])
            source = str(row["notes"]).split(";", 1)[0].replace("label_source=", "")
            writer.writerow({
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "ground_truth_error_type": row["ground_truth_error_type"],
                "affected_parts": row["ground_truth_affected_parts"],
                "label_source": source,
                "review_required": str(review_required).lower(),
                "reviewer": "",
                "review_notes": "",
            })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--case", help="Substring filter for image id")
    parser.add_argument("--no-api", action="store_true", default=True, help="Retained for safe A/B CLI compatibility")
    parser.add_argument("--dry-run", action="store_true", help="Print planned offline sample count without writing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_rows()
    if args.case:
        rows = [row for row in rows if args.case in str(row["image_id"])]
    if args.max_images is not None:
        rows = rows[: max(0, args.max_images)]
    print(f"Offline samples selected: {len(rows)}; API calls: 0")
    if args.dry_run:
        return 0
    print(json.dumps(write_outputs(rows), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
