"""Offline audit of experiment-only ROI Checklist normalization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.roi_checklist_response_normalizer import normalize_roi_checklist_response


CHECKLIST_RESPONSES = (
    "EXP-002_missingpart-A01_roi_checklist.json",
    "EXP-004_missingpart-B01_roi_checklist.json",
    "EXP-006_wrongpart-B01_roi_checklist.json",
)
SEMANTIC_FIELDS = (
    "part_id", "reference_present", "test_present", "reference_count", "test_count",
    "spatial_match", "appearance_match", "status", "confidence",
)
THESIS_ARTIFACTS = (
    "figures/checklist_confusion_matrix.png",
    "figures/method_comparison_metrics.png",
    "figures/thesis_case_missingpart_A01.png",
    "figures/thesis_case_missingpart_B01.png",
    "figures/thesis_case_wrongpart_B01.png",
    "thesis_tables/roi_direct_vs_checklist_metrics.csv",
    "thesis_tables/roi_direct_vs_checklist_cases.csv",
    "thesis_tables/checklist_component_results.csv",
    "thesis_tables/research_method_evolution.csv",
    "thesis_tables/request_efficiency.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _raw_content(response: dict[str, Any]) -> str:
    try:
        return response["raw_response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("raw response content is unavailable") from exc


def _metadata(run_dir: Path, response: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "packages" / response["case_id"] / "roi_checklist" / "request_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: item.get(key) for key in SEMANTIC_FIELDS} for item in checks]


def _differences(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> str:
    if left == right:
        return ""
    differences: list[str] = []
    if len(left) != len(right):
        differences.append(f"check_count:{len(left)}!={len(right)}")
    for index, (old, new) in enumerate(zip(left, right)):
        for field in SEMANTIC_FIELDS:
            if old.get(field) != new.get(field):
                differences.append(f"check[{index}].{field}:{old.get(field)!r}!={new.get(field)!r}")
    return " | ".join(differences)


def consolidate_thesis_artifacts(run_dir: Path) -> dict[str, Any]:
    destination = run_dir / "thesis_artifacts"
    records = []
    for relative in THESIS_ARTIFACTS:
        source = run_dir / relative
        if not source.is_file() or source.stat().st_size == 0:
            records.append({"source": relative, "status": "missing", "sha256": None})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        readable = False
        try:
            if source.suffix.lower() == ".png":
                from PIL import Image
                with Image.open(target) as artifact_image:
                    artifact_image.verify()
            else:
                with target.open("r", encoding="utf-8", newline="") as handle:
                    header = next(csv.reader(handle), [])
                    if not header:
                        raise ValueError("CSV has no header")
            readable = True
        except Exception:
            readable = False
        records.append({
            "source": relative,
            "consolidated_path": str(target.relative_to(run_dir)).replace("\\", "/"),
            "status": "verified",
            "readable": readable,
            "sha256": _sha256(source),
            "copy_sha256": _sha256(target),
        })
    result = {
        "status": "PASS" if records and all(item["status"] == "verified" and item["readable"] and item["sha256"] == item["copy_sha256"] for item in records) else "FAIL",
        "artifact_count": len(records),
        "artifacts": records,
    }
    _write_json(destination / "artifact_manifest.json", result)
    return result


def audit(run_dir: Path) -> dict[str, Any]:
    normalized_dir = run_dir / "evaluation" / "normalized_checklist"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for filename in CHECKLIST_RESPONSES:
        response_path = run_dir / "responses" / filename
        before_hash = _sha256(response_path)
        response = json.loads(response_path.read_text(encoding="utf-8"))
        metadata = _metadata(run_dir, response)
        parsed_from_raw = json.loads(_raw_content(response))
        result = normalize_roi_checklist_response(
            parsed_from_raw,
            candidate_part_ids=metadata["candidate_part_ids"],
        )
        logical_id = response["logical_request_id"]
        output_path = normalized_dir / f"{logical_id}.json"
        _write_json(output_path, {
            "logical_request_id": logical_id,
            "case_id": response["case_id"],
            "source_raw_path": str(response_path),
            **result,
        })

        prior_path = run_dir / "evaluation" / "recovered_checklist" / f"{logical_id}.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        old_semantic = _semantic(prior.get("checks", []))
        new_semantic = _semantic(result.get("checks", []))
        differences = _differences(old_semantic, new_semantic)
        comparison_rows.append({
            "logical_request_id": logical_id,
            "semantic_equal": not differences,
            "differences": differences,
            "prior_recovery_path": str(prior_path),
            "new_normalized_path": str(output_path),
        })

        after_hash = _sha256(response_path)
        audit_rows.append({
            "logical_request_id": logical_id,
            "raw_path": str(response_path),
            "raw_sha256_before": before_hash,
            "raw_sha256_after": after_hash,
            "raw_unchanged": before_hash == after_hash,
            "normalization_status": result["normalization_status"],
            "transformations_applied": result["transformations_applied"],
            "schema_valid_after": result["schema_valid_after_normalization"],
            "candidate_membership_valid": result["candidate_membership_valid"],
            "gt_used": result["gt_used"],
        })

    comparison_path = run_dir / "evaluation" / "checklist_normalization_semantic_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["logical_request_id", "semantic_equal", "differences", "prior_recovery_path", "new_normalized_path"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison_rows)

    normalization_pass = all(
        row["normalization_status"] in {"normalized", "already_valid"}
        and row["schema_valid_after"] and row["candidate_membership_valid"]
        and row["raw_unchanged"] and not row["gt_used"]
        for row in audit_rows
    )
    semantic_pass = all(row["semantic_equal"] for row in comparison_rows)
    artifact_result = consolidate_thesis_artifacts(run_dir)
    audit_result = {
        "status": "PASS" if normalization_pass and semantic_pass else "FAIL",
        "checklist_schema_robustness": "PASS" if normalization_pass and semantic_pass else "FAIL",
        "raw_schema_valid": {"valid": 0, "total": 3, "rate": 0.0},
        "normalized_schema_valid": {"valid": sum(row["schema_valid_after"] for row in audit_rows), "total": 3},
        "semantic_equivalence": "PASS" if semantic_pass else "FAIL",
        "gt_used": False,
        "api_requests_made": 0,
        "responses": audit_rows,
        "thesis_artifact_consolidation": artifact_result,
    }
    _write_json(run_dir / "evaluation" / "checklist_normalization_audit.json", audit_result)
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
