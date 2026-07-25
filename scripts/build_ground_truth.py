"""Build the formal, stable Ground Truth CSV from a frozen dataset inventory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.taxonomy import (
    is_supported_for_evaluation,
    normalize_error_type,
    parse_bool,
    schema_error_type,
    validate_ground_truth_row,
)
from scripts.audit_dataset import parse_dataset_filename


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ground_truth.csv"
OUTPUT_FIELDS = [
    "image_id",
    "image_name",
    "image_path",
    "model_id",
    "step_id",
    "view_angle",
    "is_error",
    "error_type",
    "schema_error_type",
    "error_detail",
    "evaluation_scope",
    "source_split",
    "raw_label",
    "filename_valid",
    "inventory_filename_valid",
    "inventory_validation_errors",
    "duplicate_group_id",
    "sha256",
]
REQUIRED_INVENTORY_FIELDS = {
    "source_root",
    "relative_path",
    "filename",
    "model_id",
    "step_id",
    "label",
    "is_error",
    "error_variant",
    "view_angle",
    "filename_valid",
    "validation_errors",
    "duplicate_group_id",
    "sha256",
}


def find_latest_frozen_inventory(
    audit_root: str | Path = PROJECT_ROOT / "output" / "dataset_audit",
) -> Path:
    """Find the newest inventory whose run also contains a freeze manifest."""
    root = Path(audit_root).expanduser().resolve()
    candidates = sorted(
        (
            path
            for path in root.glob("*/dataset_inventory.csv")
            if (path.parent / "freeze_manifest.json").is_file()
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No frozen dataset inventory found under: {root}")
    return candidates[0]


def _read_inventory(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = sorted(REQUIRED_INVENTORY_FIELDS - set(fields))
        if missing:
            raise ValueError(f"Dataset inventory is missing fields: {missing}")
        return list(reader), fields


def build_ground_truth_rows(
    inventory_path: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> list[dict[str, str]]:
    """Build and validate deterministic rows without modifying source files."""
    inventory = Path(inventory_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    inventory_rows, _ = _read_inventory(inventory)
    output_rows: list[dict[str, str]] = []
    seen_image_ids: set[str] = set()

    for index, source in enumerate(inventory_rows, start=2):
        source_split = source["source_root"].strip()
        relative_path = Path(source["relative_path"]).as_posix()
        image_path = f"{source_split}/{relative_path}"
        image_id = image_path
        absolute_image = root / Path(image_path)
        if not absolute_image.is_file():
            raise FileNotFoundError(
                f"Inventory row {index} image does not exist: {absolute_image}"
            )
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image_id at inventory row {index}: {image_id}")
        seen_image_ids.add(image_id)

        formal_type = normalize_error_type(source["label"])
        expected_is_error = formal_type != "correct"
        inventory_is_error = parse_bool(source["is_error"])
        if inventory_is_error != expected_is_error:
            raise ValueError(
                f"Inventory row {index} has inconsistent is_error for {formal_type}"
            )

        scope = (
            "in_scope"
            if is_supported_for_evaluation(formal_type)
            else "out_of_scope"
        )
        current_filename_validation = parse_dataset_filename(source["filename"])
        row = {
            "image_id": image_id,
            "image_name": source["filename"],
            "image_path": image_path,
            "model_id": source["model_id"],
            "step_id": source["step_id"],
            "view_angle": source["view_angle"] or "unknown",
            "is_error": "true" if expected_is_error else "false",
            "error_type": formal_type,
            "schema_error_type": schema_error_type(formal_type) or "",
            "error_detail": (
                source["error_variant"] if expected_is_error else ""
            ),
            "evaluation_scope": scope,
            "source_split": source_split,
            "raw_label": source["label"],
            "filename_valid": (
                "true" if current_filename_validation["filename_valid"] else "false"
            ),
            "inventory_filename_valid": source["filename_valid"].strip().lower(),
            "inventory_validation_errors": source["validation_errors"],
            "duplicate_group_id": source["duplicate_group_id"],
            "sha256": source["sha256"],
        }
        errors = validate_ground_truth_row(row)
        if errors:
            raise ValueError(f"Invalid generated row for {image_id}: {errors}")
        output_rows.append(row)

    return sorted(output_rows, key=lambda row: row["image_path"])


def write_ground_truth(
    rows: list[dict[str, str]], output_path: str | Path = DEFAULT_OUTPUT
) -> Path:
    """Write a validated Ground Truth CSV in deterministic order."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_ground_truth(
    inventory_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the CSV and return a concise generation summary."""
    rows = build_ground_truth_rows(inventory_path, project_root=project_root)
    path = write_ground_truth(rows, output_path)
    type_counts = Counter(row["error_type"] for row in rows)
    scope_counts = Counter(row["evaluation_scope"] for row in rows)
    return {
        "inventory_path": str(Path(inventory_path).expanduser().resolve()),
        "output_path": str(path),
        "row_count": len(rows),
        "error_type_counts": dict(sorted(type_counts.items())),
        "evaluation_scope_counts": dict(sorted(scope_counts.items())),
        "orientation_count": type_counts.get("orientation", 0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inventory = args.inventory or find_latest_frozen_inventory()
    summary = build_ground_truth(inventory, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
