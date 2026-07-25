"""Single source of truth for dataset and evaluation error taxonomy."""

from __future__ import annotations

import re
from typing import Any, Mapping


VIEW_ANGLES = frozenset({"top", "bottom", "front", "back", "left", "right"})
EVALUATION_SCOPES = frozenset({"in_scope", "out_of_scope"})
FORMAL_ERROR_TYPES = frozenset(
    {
        "correct",
        "position",
        "missing",
        "extra",
        "wrongpart",
        "criticalerror",
        "orientation",
    }
)

RAW_TO_FORMAL = {
    "correct": "correct",
    "position": "position",
    "positionerror": "position",
    "position_error": "position",
    "missing": "missing",
    "missingpart": "missing",
    "missing_part": "missing",
    "extra": "extra",
    "extrapart": "extra",
    "extra_part": "extra",
    "wrongpart": "wrongpart",
    "wrong_part": "wrongpart",
    "criticalerror": "criticalerror",
    "critical_error": "criticalerror",
    "orientation": "orientation",
    "orientationerror": "orientation",
    "orientation_error": "orientation",
}

# Names consumed by the existing Vision schema and batch comparison output.
FORMAL_TO_SCHEMA = {
    "correct": "correct",
    "position": "positionerror",
    "missing": "missingpart",
    "extra": "extrapart",
    "wrongpart": "wrongpart",
    "criticalerror": "criticalerror",
    "orientation": None,
}

# Orientation is intentionally excluded: the schema has no orientation enum and
# the frozen dataset contains no collected orientation samples.
IN_SCOPE_ERROR_TYPES = frozenset(
    {"correct", "position", "missing", "extra", "wrongpart", "criticalerror"}
)

REQUIRED_GROUND_TRUTH_FIELDS = (
    "image_id",
    "image_path",
    "model_id",
    "step_id",
    "view_angle",
    "is_error",
    "error_type",
    "error_detail",
    "evaluation_scope",
    "source_split",
)


def normalize_error_type(value: str) -> str:
    """Normalize a raw filename/schema label into the formal taxonomy."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized not in RAW_TO_FORMAL:
        raise ValueError(f"Unsupported error type: {value!r}")
    return RAW_TO_FORMAL[normalized]


def schema_error_type(error_type: str) -> str | None:
    """Return the matching existing Vision schema enum, if one exists."""
    formal = normalize_error_type(error_type)
    return FORMAL_TO_SCHEMA[formal]


def is_supported_for_evaluation(error_type: str) -> bool:
    """Return whether the formal type participates in this evaluation scope."""
    try:
        return normalize_error_type(error_type) in IN_SCOPE_ERROR_TYPES
    except ValueError:
        return False


def parse_bool(value: Any) -> bool:
    """Parse the CSV-compatible true/false representations used by this project."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected a boolean true/false value, got {value!r}")


def validate_ground_truth_row(row: Mapping[str, Any]) -> list[str]:
    """Return logical and format validation errors for one Ground Truth row."""
    errors: list[str] = []
    for field in REQUIRED_GROUND_TRUTH_FIELDS:
        if field == "error_detail":
            continue
        if field not in row or str(row[field]).strip() == "":
            errors.append(f"missing_{field}")

    try:
        formal = normalize_error_type(str(row.get("error_type", "")))
    except ValueError:
        formal = ""
        errors.append("invalid_error_type")

    try:
        is_error = parse_bool(row.get("is_error"))
    except ValueError:
        is_error = None
        errors.append("invalid_is_error")

    if formal == "correct" and is_error is True:
        errors.append("correct_must_not_be_error")
    if formal and formal != "correct" and is_error is False:
        errors.append("error_type_must_be_error")

    scope = str(row.get("evaluation_scope", "")).strip()
    if scope not in EVALUATION_SCOPES:
        errors.append("invalid_evaluation_scope")
    elif formal:
        expected_scope = (
            "in_scope" if is_supported_for_evaluation(formal) else "out_of_scope"
        )
        if scope != expected_scope:
            errors.append("evaluation_scope_mismatch")

    if not re.fullmatch(r"model\d+", str(row.get("model_id", ""))):
        errors.append("invalid_model_id")
    if not re.fullmatch(r"step\d+", str(row.get("step_id", ""))):
        errors.append("invalid_step_id")
    if str(row.get("view_angle", "")).strip().lower() not in VIEW_ANGLES:
        errors.append("invalid_view_angle")
    if str(row.get("source_split", "")) not in {"input", "regression_subset"}:
        errors.append("invalid_source_split")

    expected_schema = FORMAL_TO_SCHEMA.get(formal)
    if row.get("schema_error_type") not in {None, "", expected_schema}:
        errors.append("schema_error_type_mismatch")
    return errors
