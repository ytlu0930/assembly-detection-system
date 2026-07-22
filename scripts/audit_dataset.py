"""Read-only dataset inventory, validation, duplicate audit, and freeze tool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_manager import resolve_run_output, write_run_summary


LEGAL_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIEW_ANGLES = {"top", "bottom", "front", "back", "left", "right"}
ERROR_TYPE_MAP = {
    "correct": "correct",
    "positionerror": "position",
    "orientationerror": "orientation",
    "missingpart": "missing",
    "extrapart": "extra",
    "wrongpart": "wrongpart",
}
TARGETS = {
    "correct": 30,
    "error": 80,
    "position": 20,
    "orientation": 20,
    "missing": 20,
    "extra": 20,
}
INVENTORY_FIELDS = [
    "source_root",
    "relative_path",
    "filename",
    "extension",
    "file_size_bytes",
    "sha256",
    "model_id",
    "step_id",
    "label",
    "is_error",
    "error_type",
    "error_variant",
    "view_angle",
    "sequence_index",
    "filename_valid",
    "validation_errors",
    "has_matching_reference",
    "matching_reference_path",
    "duplicate_group_id",
]


def parse_dataset_filename(path: str | Path) -> dict[str, Any]:
    """Parse and validate the project's five-field image filename format."""
    file_path = Path(path)
    extension = file_path.suffix.lower()
    stem = file_path.stem
    parts = stem.split("_")
    errors: list[str] = []

    model_id = parts[0] if len(parts) >= 1 else ""
    step_id = parts[1] if len(parts) >= 2 else ""
    label_field = parts[2] if len(parts) >= 3 else ""
    view_angle = parts[3].lower() if len(parts) >= 4 else ""
    sequence_index = parts[4] if len(parts) >= 5 else ""

    if len(parts) != 5:
        errors.append(f"expected_5_fields_found_{len(parts)}")
    if not model_id or not model_id.lower().startswith("model"):
        errors.append("missing_or_invalid_model_id")
    if not step_id or not step_id.lower().startswith("step"):
        errors.append("missing_or_invalid_step_id")
    if not label_field:
        errors.append("missing_label")

    if "-" in label_field:
        label, error_variant = label_field.split("-", 1)
    else:
        label, error_variant = label_field, ""
    label = label.lower()
    if label not in ERROR_TYPE_MAP:
        errors.append("unknown_label")

    if view_angle not in VIEW_ANGLES:
        errors.append("missing_or_invalid_view_angle")
    if not sequence_index or not sequence_index.isdigit():
        errors.append("missing_or_invalid_sequence_index")
    if extension not in LEGAL_EXTENSIONS:
        errors.append("unsupported_extension")

    error_type = ERROR_TYPE_MAP.get(label, "unknown")
    is_error: bool | None = False if label == "correct" else (True if label else None)
    return {
        "filename": file_path.name,
        "extension": extension,
        "model_id": model_id,
        "step_id": step_id,
        "label": label,
        "is_error": is_error,
        "error_type": error_type,
        "error_variant": error_variant,
        "view_angle": view_angle,
        "sequence_index": sequence_index,
        "filename_valid": not errors,
        "validation_errors": errors,
    }


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file without modifying it."""
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_references(
    parsed: dict[str, Any], input_root: Path
) -> tuple[list[Path], str]:
    model_id = parsed["model_id"]
    step_id = parsed["step_id"]
    view_angle = parsed["view_angle"]
    pattern = f"{model_id}_{step_id}_correct-*_{view_angle}_*"
    normal_dir = input_root / "normal" / f"{model_id}_{step_id}"
    for search_root in (normal_dir, input_root / "normal"):
        if not search_root.exists():
            continue
        candidates: list[Path] = []
        for extension in sorted(LEGAL_EXTENSIONS):
            candidates.extend(search_root.rglob(f"{pattern}{extension}"))
        candidates = sorted(set(candidates))
        if candidates:
            return candidates, pattern
    return [], pattern


def find_matching_reference(
    parsed: dict[str, Any], input_root: Path
) -> tuple[Path | None, list[Path], str]:
    """Apply the existing batch pipeline rule: prefer correct-01, else first."""
    candidates, pattern = _candidate_references(parsed, input_root)
    selected = next(
        (candidate for candidate in candidates if "correct-01" in candidate.stem),
        candidates[0] if candidates else None,
    )
    return selected, candidates, pattern


def classify_duplicate_group(rows: list[dict[str, Any]]) -> str:
    """Classify one identical-content group by its source membership."""
    sources = {row["source_root"] for row in rows}
    if sources == {"input"}:
        return "duplicate_within_input"
    if sources == {"regression_subset"}:
        return "duplicate_within_regression"
    if {"input", "regression_subset"}.issubset(sources):
        input_names = {
            row["filename"] for row in rows if row["source_root"] == "input"
        }
        regression_names = {
            row["filename"]
            for row in rows
            if row["source_root"] == "regression_subset"
        }
        if input_names & regression_names:
            return "expected_regression_copy"
        return "cross_source_duplicate"
    return "cross_source_duplicate"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if isinstance(output.get("validation_errors"), list):
                output["validation_errors"] = ";".join(output["validation_errors"])
            writer.writerow(output)


def build_target_status(counts: dict[str, int]) -> dict[str, str]:
    """Compare observed counts with explicit collection targets."""
    result = {
        name: "met" if counts.get(name, 0) >= target else "not_met"
        for name, target in TARGETS.items()
    }
    result["wrongpart"] = "not_applicable"
    result["unknown"] = "unknown" if counts.get("unknown", 0) else "not_applicable"
    return result


def audit_dataset(
    input_root: str | Path,
    regression_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    freeze: bool = False,
    hash_algorithm: str = "sha256",
    output_root: str | Path = PROJECT_ROOT / "output",
) -> dict[str, Any]:
    """Audit both sources and optionally write an immutable baseline manifest."""
    input_path = Path(input_root).expanduser().resolve()
    regression_path = Path(regression_root).expanduser().resolve()
    for root in (input_path, regression_path):
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")

    paths = resolve_run_output(
        "dataset_audit",
        "dataset_freeze" if freeze else "dataset_audit",
        output_dir=output_dir,
        output_root=output_root,
        image_subdirs=[],
        nest_experiment=False,
    )
    inventory_path = paths.run_dir / "dataset_inventory.csv"
    summary_path = paths.run_dir / "dataset_summary.json"
    invalid_path = paths.run_dir / "invalid_filenames.csv"
    duplicates_path = paths.run_dir / "duplicate_files.csv"
    missing_path = paths.run_dir / "missing_reference_files.csv"
    freeze_path = paths.run_dir / "freeze_manifest.json"

    roots = [("input", input_path), ("regression_subset", regression_path)]
    rows: list[dict[str, Any]] = []
    for source_name, root in roots:
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            parsed = parse_dataset_filename(file_path)
            matching_reference = None
            candidates: list[Path] = []
            expected_pattern = ""
            if parsed["is_error"] is True and parsed["model_id"] and parsed["step_id"]:
                matching_reference, candidates, expected_pattern = find_matching_reference(
                    parsed, input_path
                )
            row = {
                "source_root": source_name,
                "relative_path": file_path.relative_to(root).as_posix(),
                "filename": file_path.name,
                "extension": parsed["extension"],
                "file_size_bytes": file_path.stat().st_size,
                "sha256": file_digest(file_path, hash_algorithm),
                "model_id": parsed["model_id"],
                "step_id": parsed["step_id"],
                "label": parsed["label"],
                "is_error": parsed["is_error"],
                "error_type": parsed["error_type"],
                "error_variant": parsed["error_variant"],
                "view_angle": parsed["view_angle"],
                "sequence_index": parsed["sequence_index"],
                "filename_valid": parsed["filename_valid"],
                "validation_errors": parsed["validation_errors"],
                "has_matching_reference": matching_reference is not None,
                "matching_reference_path": (
                    matching_reference.relative_to(PROJECT_ROOT).as_posix()
                    if matching_reference
                    and matching_reference.is_relative_to(PROJECT_ROOT)
                    else str(matching_reference or "")
                ),
                "duplicate_group_id": "",
                "_absolute_path": file_path,
                "_reference_candidates": candidates,
                "_expected_reference_pattern": expected_pattern,
            }
            rows.append(row)

    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        hash_groups[row["sha256"]].append(row)
    duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
    duplicate_rows: list[dict[str, Any]] = []
    for index, group in enumerate(sorted(duplicate_groups, key=lambda x: x[0]["sha256"]), 1):
        group_id = f"dup_{index:04d}"
        classification = classify_duplicate_group(group)
        for row in group:
            row["duplicate_group_id"] = group_id
            duplicate_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "sha256": row["sha256"],
                    "file_count": len(group),
                    "classification": classification,
                    "source_root": row["source_root"],
                    "relative_path": row["relative_path"],
                    "filename": row["filename"],
                }
            )

    clean_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    invalid_rows = [
        {
            "source_root": row["source_root"],
            "relative_path": row["relative_path"],
            "filename": row["filename"],
            "validation_errors": row["validation_errors"],
        }
        for row in rows
        if not row["filename_valid"]
    ]
    missing_rows = []
    for row in rows:
        if row["is_error"] is True and not row["has_matching_reference"]:
            missing_rows.append(
                {
                    "error_image_path": f"{row['source_root']}/{row['relative_path']}",
                    "model_id": row["model_id"],
                    "step_id": row["step_id"],
                    "view_angle": row["view_angle"],
                    "expected_reference_pattern": row["_expected_reference_pattern"],
                    "candidate_count": len(row["_reference_candidates"]),
                    "selection_rule": "prefer correct-01, otherwise first sorted candidate",
                    "reason": "no matching correct reference",
                }
            )

    _write_csv(inventory_path, clean_rows, INVENTORY_FIELDS)
    _write_csv(
        invalid_path,
        invalid_rows,
        ["source_root", "relative_path", "filename", "validation_errors"],
    )
    _write_csv(
        duplicates_path,
        duplicate_rows,
        [
            "duplicate_group_id",
            "sha256",
            "file_count",
            "classification",
            "source_root",
            "relative_path",
            "filename",
        ],
    )
    _write_csv(
        missing_path,
        missing_rows,
        [
            "error_image_path",
            "model_id",
            "step_id",
            "view_angle",
            "expected_reference_pattern",
            "candidate_count",
            "selection_rule",
            "reason",
        ],
    )

    label_counts = Counter(row["label"] for row in rows)
    error_type_counts = Counter(
        row["error_type"] for row in rows if row["is_error"] is True
    )
    coverage: dict[str, Any] = {}
    for error_type in sorted(set(error_type_counts) | {"position", "orientation", "missing", "extra", "wrongpart", "unknown"}):
        matching = [row for row in rows if row["error_type"] == error_type]
        model_ids = sorted({row["model_id"] for row in matching if row["model_id"]})
        step_ids = sorted({row["step_id"] for row in matching if row["step_id"]})
        coverage[error_type] = {
            "file_count": len(matching),
            "model_ids": model_ids,
            "model_count": len(model_ids),
            "step_ids": step_ids,
            "step_count": len(step_ids),
            "covers_at_least_3_steps": len(step_ids) >= 3,
        }

    correct_count = label_counts.get("correct", 0)
    error_count = sum(1 for row in rows if row["is_error"] is True)
    target_counts = {
        "correct": correct_count,
        "error": error_count,
        "position": error_type_counts.get("position", 0),
        "orientation": error_type_counts.get("orientation", 0),
        "missing": error_type_counts.get("missing", 0),
        "extra": error_type_counts.get("extra", 0),
        "wrongpart": error_type_counts.get("wrongpart", 0),
        "unknown": error_type_counts.get("unknown", 0),
    }
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_status": "frozen" if freeze else "audited",
        "total_files": len(rows),
        "input_files": sum(row["source_root"] == "input" for row in rows),
        "regression_subset_files": sum(
            row["source_root"] == "regression_subset" for row in rows
        ),
        "valid_filename_count": sum(row["filename_valid"] for row in rows),
        "invalid_filename_count": len(invalid_rows),
        "correct_count": correct_count,
        "error_count": error_count,
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "model_counts": dict(sorted(Counter(row["model_id"] for row in rows if row["model_id"]).items())),
        "step_counts": dict(sorted(Counter(row["step_id"] for row in rows if row["step_id"]).items())),
        "view_angle_counts": dict(sorted(Counter(row["view_angle"] for row in rows if row["view_angle"]).items())),
        "error_type_step_coverage": coverage,
        "file_extension_counts": dict(sorted(Counter(row["extension"] or "<none>" for row in rows).items())),
        "duplicate_file_count": len(duplicate_rows),
        "duplicate_group_count": len(duplicate_groups),
        "missing_reference_count": len(missing_rows),
        "unknown_label_count": sum(row["label"] not in ERROR_TYPE_MAP for row in rows),
        "unknown_labels": sorted({row["label"] for row in rows if row["label"] not in ERROR_TYPE_MAP}),
        "target_comparison": {
            "correct_target": 30,
            "error_target": 80,
            "position_target": 20,
            "orientation_target": 20,
            "missing_target": 20,
            "extra_target": 20,
        },
        "target_status": build_target_status(target_counts),
        "limitations": [
            "No ground-truth bounding boxes are included in this audit.",
            "Unknown labels are preserved and reported instead of being remapped.",
            "Freeze records hashes but does not change Windows file permissions.",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if freeze:
        manifest = {
            "freeze_id": paths.run_id,
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "frozen",
            "source_roots": ["input", "regression_subset"],
            "total_files": len(rows),
            "total_bytes": sum(row["file_size_bytes"] for row in rows),
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "file_hash_algorithm": hash_algorithm,
            "files": [
                {
                    "source_root": row["source_root"],
                    "relative_path": row["relative_path"],
                    "sha256": row["sha256"],
                    "file_size_bytes": row["file_size_bytes"],
                }
                for row in clean_rows
            ],
            "limitations": summary["limitations"],
            "notes": [
                "No new samples were captured.",
                "No source file was renamed, moved, edited, or deleted.",
                "The frozen dataset represents the final available dataset for this project.",
            ],
        }
        with freeze_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    output_paths = {
        "dataset_inventory": str(inventory_path),
        "dataset_summary": str(summary_path),
        "invalid_filenames": str(invalid_path),
        "duplicate_files": str(duplicates_path),
        "missing_reference_files": str(missing_path),
    }
    if freeze:
        output_paths["freeze_manifest"] = str(freeze_path)
    write_run_summary(
        paths,
        status="completed",
        input_count=len(rows),
        success_count=len(rows),
        failure_count=0,
        parameters={
            "input_root": str(input_path),
            "regression_root": str(regression_path),
            "freeze": freeze,
            "hash_algorithm": hash_algorithm,
        },
        output_paths=output_paths,
        notes=["Dataset sources were read only."],
    )
    return {"run_dir": paths.run_dir, "summary": summary, "rows": clean_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "input")
    parser.add_argument(
        "--regression-root", type=Path, default=PROJECT_ROOT / "regression_subset"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--hash-algorithm", default="sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_dataset(
        args.input_root,
        args.regression_root,
        output_dir=args.output_dir,
        freeze=args.freeze,
        hash_algorithm=args.hash_algorithm,
    )
    print(f"run_dir: {result['run_dir']}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
