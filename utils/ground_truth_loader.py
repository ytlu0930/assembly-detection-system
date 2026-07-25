"""Validated loader and minimal batch-test adapter for formal Ground Truth."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from utils.taxonomy import REQUIRED_GROUND_TRUTH_FIELDS, parse_bool, validate_ground_truth_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "ground_truth.csv"


def load_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH_PATH) -> list[dict[str, Any]]:
    """Load, type-normalize, and validate the formal Ground Truth CSV."""
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Ground Truth CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED_GROUND_TRUTH_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Ground Truth CSV is missing fields: {missing}")
        rows = list(reader)

    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        errors = validate_ground_truth_row(row)
        if errors:
            raise ValueError(f"Invalid Ground Truth row {index}: {errors}")
        image_id = row["image_id"]
        if image_id in seen:
            raise ValueError(f"Duplicate image_id at row {index}: {image_id}")
        seen.add(image_id)
        row["is_error"] = parse_bool(row["is_error"])
        # Compatibility aliases for the current filename-based batch runner.
        row["ground_truth"] = row.get("schema_error_type") or row["error_type"]
        row["expected_error_type"] = row["ground_truth"]
    return rows


def get_ground_truth_by_image_id(
    image_id: str,
    rows: Iterable[dict[str, Any]] | None = None,
    path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    """Look up a unique path-based image id, with safe filename fallback."""
    records = list(rows) if rows is not None else load_ground_truth(path)
    exact = [row for row in records if row["image_id"] == image_id]
    if len(exact) == 1:
        return exact[0]

    by_name = [row for row in records if row.get("image_name") == image_id]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise KeyError(
            f"Ambiguous image filename {image_id!r}; use the source-qualified image_id"
        )
    raise KeyError(f"Ground Truth image_id not found: {image_id}")


def validate_batch_test_compatibility(
    rows: Iterable[dict[str, Any]] | None = None,
    path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
) -> dict[str, Any]:
    """Check fields needed by the existing reference-comparison batch runner."""
    records = list(rows) if rows is not None else load_ground_truth(path)
    required = {
        "image_id",
        "is_error",
        "error_type",
        "schema_error_type",
        "model_id",
        "step_id",
        "view_angle",
        "evaluation_scope",
    }
    errors: list[str] = []
    for index, row in enumerate(records, start=2):
        missing = sorted(field for field in required if field not in row)
        if missing:
            errors.append(f"row {index} missing {missing}")
        if row.get("evaluation_scope") == "in_scope" and not row.get("schema_error_type"):
            errors.append(f"row {index} has no schema-compatible error type")
    return {
        "compatible": not errors,
        "row_count": len(records),
        "in_scope_count": sum(row.get("evaluation_scope") == "in_scope" for row in records),
        "out_of_scope_count": sum(row.get("evaluation_scope") == "out_of_scope" for row in records),
        "errors": errors,
    }
