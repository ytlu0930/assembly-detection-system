"""Offline forensic recovery and candidate-set audit for the partial A/B run."""

from __future__ import annotations

import ast
import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SEQUENCES = (2, 5, 8, 11, 14)


def recover_instance(error: str) -> dict[str, Any]:
    if "\nOn instance:\n" not in error:
        raise ValueError("Validation error does not retain an On instance payload")
    wrapped = ast.literal_eval(textwrap.dedent(error.split("\nOn instance:\n", 1)[1]).strip())
    required = {"model_id", "step_id", "view_angle", "is_error", "overall_error_type", "detected_parts", "summary"}
    nested = wrapped.get("properties")
    if isinstance(nested, dict) and required.issubset(nested):
        return nested
    schema_keys = {"$schema", "title", "type", "additionalProperties", "required", "properties"}
    return {key: value for key, value in wrapped.items() if key not in schema_keys}


def recover_reference_failures(results_dir: Path) -> list[dict[str, Any]]:
    raw_dir = results_dir / "raw"
    output = ROOT / "analysis/vision_prompt_ab/recovered/reference"
    output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "schema/vision_output_schema.json").read_text(encoding="utf-8"))
    rows = []
    for sequence in REFERENCE_SEQUENCES:
        source = next(raw_dir.glob(f"{sequence:02d}_*.json"))
        raw = json.loads(source.read_text(encoding="utf-8"))
        recovered = recover_instance(str(raw.get("error") or ""))
        validate(instance=recovered, schema=schema)
        payload = {
            "recovered_for_analysis": True, "excluded_from_primary_metrics": True,
            "source_result": source.resolve().relative_to(ROOT.resolve()).as_posix(),
            "case_id": raw["case_id"], "variant": raw["variant"],
            "original_raw_response_available": raw.get("raw_response") is not None,
            "original_parsed_output_available": raw.get("parsed_output") is not None,
            "recovery_source": "jsonschema ValidationError On instance payload",
            "recovered_schema_validation": "valid", "recovered_output": recovered,
        }
        target = output / f"{raw['case_id']}_reference_recovered.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append(payload)
    return rows


def write_candidate_audit(results_dir: Path) -> list[dict[str, Any]]:
    library_count = len(json.loads((ROOT / "config/part_library.json").read_text(encoding="utf-8")))
    rows = []
    for metadata_path in sorted((ROOT / "analysis/vision_prompt_ab/packages/reference_candidate").glob("*/request_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidates = metadata.get("candidate_part_ids") or []
        ratio = len([item for item in candidates if item != "UNKNOWN_EXTRA_PART"]) / library_count if library_count else 0.0
        effectiveness = "weak" if ratio >= 0.80 else "moderate"
        if metadata["error_type_hint"] == "correct":
            effectiveness = "not_applicable"
        rows.append({
            "case_id": metadata["case_id"], "error_type": metadata["error_type_hint"],
            "candidate_count": len(candidates), "canonical_library_count": library_count,
            "inventory_coverage_ratio": ratio, "candidate_ids": "|".join(candidates),
            "constraint_effectiveness": effectiveness,
        })
    path = results_dir / "candidate_set_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return rows


def main() -> int:
    results = ROOT / "analysis/vision_prompt_ab/results"
    recovered = recover_reference_failures(results)
    candidates = write_candidate_audit(results)
    print(json.dumps({"recovered_reference_failures": len(recovered), "candidate_packages": len(candidates), "api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
