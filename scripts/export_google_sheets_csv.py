"""Export deterministic, Google Sheets-ready CSV files from formal Ground Truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ground_truth_loader import DEFAULT_GROUND_TRUTH_PATH, load_ground_truth
from utils.taxonomy import FORMAL_ERROR_TYPES


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "google_sheets_import"
GROUND_TRUTH_FIELDS = [
    "image_id",
    "file_name",
    "image_path",
    "source_split",
    "model_id",
    "step_id",
    "view_angle",
    "is_error",
    "error_type",
    "error_detail",
    "evaluation_scope",
    "filename_valid",
    "inventory_error_type",
    "inventory_filename_valid",
    "review_status",
    "reviewer",
    "review_notes",
]
SUMMARY_FIELDS = [
    "category",
    "image_count",
    "target_count",
    "target_applicable",
    "target_status",
    "evaluation_scope",
    "notes",
]
STEP_COVERAGE_FIELDS = [
    "error_type",
    "image_count",
    "covered_step_count",
    "covered_steps",
    "target_step_count",
    "target_applicable",
    "target_status",
    "notes",
]
BATCH_RESULTS_FIELDS = [
    "test_date",
    "run_id",
    "image_id",
    "file_name",
    "model_id",
    "step_id",
    "view_angle",
    "gt_is_error",
    "gt_error_type",
    "evaluation_scope",
    "pred_is_error",
    "pred_error_type",
    "confidence",
    "result_class",
    "type_match",
    "localization_success",
    "schema_valid",
    "parse_success",
    "response_time_sec",
    "status",
    "failure_stage",
    "failure_reason",
    "notes",
]
FAILURE_ANALYSIS_FIELDS = [
    "run_id",
    "image_id",
    "file_name",
    "gt_error_type",
    "pred_error_type",
    "result_class",
    "failure_stage",
    "failure_reason",
    "image_quality_issue",
    "localization_issue",
    "vision_issue",
    "parsing_issue",
    "recommended_action",
    "review_status",
    "reviewer",
    "review_notes",
]
OUTPUT_FILENAMES = (
    "01_ground_truth.csv",
    "02_dataset_summary.csv",
    "03_step_coverage.csv",
    "04_batch_results_template.csv",
    "05_failure_analysis_template.csv",
)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _csv_text(fields: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        clean = {
            field: "" if row.get(field) is None else row.get(field, "")
            for field in fields
        }
        if any(isinstance(value, (dict, list, tuple, set)) for value in clean.values()):
            raise ValueError("CSV cells must contain scalar values")
        writer.writerow(clean)
    return buffer.getvalue()


def build_ground_truth_export(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        image_id = str(row["image_id"])
        file_name = Path(image_id).name
        if image_id in seen_ids:
            raise ValueError(f"Duplicate formal image_id: {image_id}")
        if file_name != row.get("image_name"):
            raise ValueError(f"image_name does not match image_id: {image_id}")
        seen_ids.add(image_id)
        exported.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "image_path": row["image_path"],
                "source_split": row["source_split"],
                "model_id": row["model_id"],
                "step_id": row["step_id"],
                "view_angle": row["view_angle"],
                "is_error": _bool_text(bool(row["is_error"])),
                "error_type": row["error_type"],
                "error_detail": row.get("error_detail", ""),
                "evaluation_scope": row["evaluation_scope"],
                "filename_valid": str(row.get("filename_valid", "")).lower(),
                "inventory_error_type": row.get("raw_label", ""),
                "inventory_filename_valid": str(
                    row.get("inventory_filename_valid", "")
                ).lower(),
                "review_status": "",
                "reviewer": "",
                "review_notes": "",
            }
        )
    return exported


def build_dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row["error_type"] for row in rows)
    error_count = sum(bool(row["is_error"]) for row in rows)
    definitions = [
        ("correct", counts["correct"], 30, True, "in_scope", "Correct target: 30 images."),
        ("position", counts["position"], 20, True, "in_scope", "Per-class target: 20 images."),
        (
            "orientation",
            counts["orientation"],
            "",
            False,
            "out_of_scope",
            "No collected samples; excluded from evaluation.",
        ),
        ("missing", counts["missing"], 20, True, "in_scope", "Per-class target: 20 images."),
        ("extra", counts["extra"], 20, True, "in_scope", "Per-class target: 20 images."),
        (
            "wrongpart",
            counts["wrongpart"],
            "",
            False,
            "in_scope",
            "Formal taxonomy class; no numeric collection target was defined.",
        ),
        (
            "criticalerror",
            counts["criticalerror"],
            "",
            False,
            "in_scope",
            "Retained as its own schema-supported class.",
        ),
        ("all_error", error_count, 80, True, "in_scope", "All-error target: 80 images."),
        (
            "all_images",
            len(rows),
            "",
            False,
            "in_scope",
            "Includes input and regression_subset records.",
        ),
    ]
    result: list[dict[str, Any]] = []
    for category, count, target, applicable, scope, notes in definitions:
        if scope == "out_of_scope":
            status = "out_of_scope"
        elif applicable:
            status = "achieved" if count >= int(target) else "not_achieved"
        else:
            status = "not_applicable"
        result.append(
            {
                "category": category,
                "image_count": count,
                "target_count": target,
                "target_applicable": _bool_text(applicable),
                "target_status": status,
                "evaluation_scope": scope,
                "notes": notes,
            }
        )
    return result


def build_step_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_type"]].append(row)
    ordered_types = [
        "correct",
        "position",
        "orientation",
        "missing",
        "extra",
        "wrongpart",
        "criticalerror",
    ]
    if set(ordered_types) != set(FORMAL_ERROR_TYPES):
        raise ValueError("Exporter taxonomy order is out of sync")

    result: list[dict[str, Any]] = []
    target_types = {"position", "missing", "extra"}
    for error_type in ordered_types:
        matching = grouped[error_type]
        covered = sorted(
            {f"{row['model_id']}-{row['step_id']}" for row in matching}
        )
        applicable = error_type in target_types
        if error_type == "orientation":
            scope = "out_of_scope"
            status = "out_of_scope"
            notes = "No collected samples; no coverage target is applied."
        elif applicable:
            scope = "in_scope"
            status = "achieved" if len(covered) >= 3 else "not_achieved"
            notes = "Target: coverage across at least 3 unique model-step pairs."
        else:
            scope = "in_scope"
            status = "not_applicable"
            notes = "No 3-step coverage target was defined for this category."
        result.append(
            {
                "error_type": error_type,
                "image_count": len(matching),
                "covered_step_count": len(covered),
                "covered_steps": "; ".join(covered),
                "target_step_count": 3 if applicable else "",
                "target_applicable": _bool_text(applicable),
                "target_status": status,
                "notes": notes,
            }
        )
    return result


def build_export_texts(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "01_ground_truth.csv": _csv_text(
            GROUND_TRUTH_FIELDS, build_ground_truth_export(rows)
        ),
        "02_dataset_summary.csv": _csv_text(
            SUMMARY_FIELDS, build_dataset_summary(rows)
        ),
        "03_step_coverage.csv": _csv_text(
            STEP_COVERAGE_FIELDS, build_step_coverage(rows)
        ),
        "04_batch_results_template.csv": _csv_text(BATCH_RESULTS_FIELDS, []),
        "05_failure_analysis_template.csv": _csv_text(
            FAILURE_ANALYSIS_FIELDS, []
        ),
    }


def export_google_sheets_csv(
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    source = Path(ground_truth_path).expanduser().resolve()
    if source == (PROJECT_ROOT / "ground_truth.csv").resolve():
        raise ValueError("Legacy repository-root ground_truth.csv is not supported")
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = load_ground_truth(source)
    texts = build_export_texts(rows)
    destination = Path(output_dir).expanduser().resolve()

    if check_only:
        missing = [
            name for name, text in texts.items()
            if not (destination / name).is_file()
            or (destination / name).read_text(encoding="utf-8") != text
        ]
        if missing:
            raise ValueError(f"Missing or stale exported CSV files: {missing}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            (destination / name).write_text(
                text,
                encoding="utf-8",
                newline="",
            )

    after_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError("Formal Ground Truth changed during export")
    return {
        "source": str(source),
        "source_sha256": before_hash,
        "output_dir": str(destination),
        "check_only": check_only,
        "row_count": len(rows),
        "files": list(texts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_google_sheets_csv(
        args.ground_truth,
        args.output_dir,
        check_only=args.check_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
