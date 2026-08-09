"""Experiment-only normalization for ROI checklist responses.

This module repairs known response-contract aliases without consulting Ground
Truth, review labels, or production inference state.  The input object is never
mutated.  Invalid or ambiguous inputs fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCHEMA_PATH = PROJECT_ROOT / "experiments/schema/vision_roi_checklist_output_schema.json"
CONFIDENCE_WORDS = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3, "UNCERTAIN": 0.0}
VALID_STATUSES = {"PASS", "FAIL", "UNCERTAIN"}
EVIDENCE_TEXT_FIELDS = ("evidence", "observation", "observation_text", "evidence_text")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _invalid(
    error: str,
    raw: Any,
    transformations: list[str] | None = None,
    *,
    membership_valid: bool = False,
) -> dict[str, Any]:
    return {
        "normalization_status": "failed",
        "normalized_response": None,
        "checks": [],
        "transformations_applied": sorted(set(transformations or [])),
        "failure_reason": error,
        "raw_sha256": _json_sha256(raw),
        "normalized_sha256": None,
        "schema_valid_after_normalization": False,
        "gt_used": False,
        "candidate_membership_valid": membership_valid,
        "input_mutated": False,
    }


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean confidence is invalid")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        token = value.strip().upper()
        if token in CONFIDENCE_WORDS:
            number = CONFIDENCE_WORDS[token]
        else:
            try:
                number = float(token)
            except ValueError as exc:
                raise ValueError(f"unsupported confidence: {value!r}") from exc
    else:
        raise ValueError(f"unsupported confidence type: {type(value).__name__}")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"confidence outside [0, 1]: {number}")
    return number


def _status(item: dict[str, Any]) -> tuple[str, str | None]:
    present = [(key, item[key]) for key in ("status", "CHECK", "check_result") if key in item]
    if not present:
        raise ValueError("missing status/CHECK/check_result")
    normalized = {str(value).strip().upper() for _, value in present}
    if len(normalized) != 1:
        raise ValueError("conflicting status aliases")
    status = normalized.pop()
    if status not in VALID_STATUSES:
        raise ValueError(f"unsupported status: {status!r}")
    alias = present[0][0] if "status" not in item else None
    return status, alias


def _evidence_summary(item: dict[str, Any], status: str) -> str:
    supplied = item.get("evidence_summary")
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    for field in EVIDENCE_TEXT_FIELDS:
        evidence = item.get(field)
        if evidence is not None and str(evidence).strip():
            if isinstance(evidence, (dict, list)):
                return json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return str(evidence).strip()
    # The experiment schema requires a non-empty string.  This explicit marker
    # states only that text evidence was absent; it does not invent evidence.
    return "No explicit evidence summary supplied in the response."


def normalize_roi_checklist_response(
    parsed_response: Any,
    *,
    candidate_part_ids: list[str] | None = None,
    schema_path: Path = EXPERIMENT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Return a canonical experiment response or a fail-closed result.

    Candidate IDs may be supplied solely for membership enforcement. Ground
    Truth is deliberately absent from the interface.
    """
    before = copy.deepcopy(parsed_response)
    if not isinstance(parsed_response, dict):
        return _invalid("response must be an object", parsed_response)

    transformations: list[str] = []
    has_checks = "checks" in parsed_response
    has_results = "results" in parsed_response
    if has_checks and has_results:
        return _invalid("ambiguous response contains both checks and results", parsed_response)
    source_checks = parsed_response.get("checks") if has_checks else parsed_response.get("results")
    if has_results:
        transformations.append("results->checks")
    if not isinstance(source_checks, list) or not source_checks:
        return _invalid("checks/results must be a non-empty array", parsed_response, transformations)

    checks: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(source_checks):
            if not isinstance(item, dict):
                raise ValueError(f"check[{index}] must be an object")
            part_id = item.get("part_id")
            if not isinstance(part_id, str) or not part_id.strip():
                raise ValueError(f"check[{index}] has invalid part_id")
            status, alias = _status(item)
            if alias:
                transformations.append(f"{alias}->status")
            confidence = _confidence(item.get("confidence"))
            if not isinstance(item.get("confidence"), (int, float)) or isinstance(item.get("confidence"), bool):
                transformations.append("string_confidence->float")
            summary = _evidence_summary(item, status)
            if not item.get("evidence_summary"):
                transformations.append("missing_evidence_summary->deterministic_fallback")
            checks.append({
                "part_id": part_id.strip().upper(),
                "reference_present": item.get("reference_present"),
                "test_present": item.get("test_present"),
                "reference_count": item.get("reference_count"),
                "test_count": item.get("test_count"),
                "spatial_match": item.get("spatial_match"),
                "appearance_match": item.get("appearance_match"),
                "status": status,
                "confidence": confidence,
                "evidence_summary": summary,
            })
    except (TypeError, ValueError) as exc:
        return _invalid(str(exc), parsed_response, transformations)

    normalized = {"checks": checks}
    membership_valid = True
    if candidate_part_ids is not None:
        allowed = [str(value).strip().upper() for value in candidate_part_ids]
        normalized_ids = [item["part_id"] for item in checks]
        membership_valid = (
            len(normalized_ids) == len(allowed)
            and set(normalized_ids) == set(allowed)
            and len(normalized_ids) == len(set(normalized_ids))
        )
        if not membership_valid:
            return _invalid(
                "normalized part IDs do not exactly match the request candidate set",
                parsed_response,
                transformations,
                membership_valid=False,
            )
    try:
        from jsonschema import validate

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validate(instance=normalized, schema=schema)
    except Exception as exc:  # schema/path errors must also fail closed
        return _invalid(
            f"{type(exc).__name__}: {exc}", parsed_response, transformations,
            membership_valid=membership_valid,
        )

    status = "already_valid" if not transformations else "normalized"
    return {
        "normalization_status": status,
        "normalized_response": normalized,
        "checks": checks,
        "transformations_applied": sorted(set(transformations)),
        "failure_reason": None,
        "raw_sha256": _json_sha256(parsed_response),
        "normalized_sha256": _json_sha256(normalized),
        "schema_valid_after_normalization": True,
        "gt_used": False,
        "candidate_membership_valid": membership_valid,
        "input_mutated": parsed_response != before,
    }
