"""Deterministic decisions over ROI checklist observations.

The model observes candidate-level facts. This module alone derives the
affected-part conclusion and never receives Ground Truth or review labels.
"""

from __future__ import annotations

from typing import Any


def _mismatch(check: dict[str, Any], error_type: str) -> tuple[bool, str]:
    ref, test = check.get("reference_present"), check.get("test_present")
    ref_count, test_count = check.get("reference_count"), check.get("test_count")
    normalized = str(error_type).lower().replace("_", "")
    if normalized in {"missing", "missingpart"}:
        if ref is True and test is False:
            return True, "reference_present_test_absent"
        if isinstance(ref_count, int) and isinstance(test_count, int) and ref_count > test_count:
            return True, "reference_count_exceeds_test"
    elif normalized in {"extra", "extrapart"}:
        if ref is False and test is True:
            return True, "test_present_reference_absent"
        if isinstance(ref_count, int) and isinstance(test_count, int) and test_count > ref_count:
            return True, "test_count_exceeds_reference"
    elif normalized in {"wrong", "wrongpart"}:
        if (ref is True and test is False) or (ref is False and test is True):
            return True, "paired_presence_mismatch"
        if check.get("appearance_match") is False:
            return True, "appearance_mismatch"
        if isinstance(ref_count, int) and isinstance(test_count, int) and ref_count != test_count:
            return True, "count_mismatch"
    elif normalized in {"position", "positionerror"} and check.get("spatial_match") is False:
        return True, "spatial_mismatch"
    return False, "no_rule_match"


def evaluate_roi_checklist(
    *, checks: list[dict[str, Any]], candidate_part_ids: list[str],
    error_type: str, paired_roi_supported: bool = False,
) -> dict[str, Any]:
    allowed = [str(value).upper() for value in candidate_part_ids]
    by_id: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    for item in checks:
        part_id = str(item.get("part_id") or "").upper()
        if part_id not in allowed or part_id in by_id:
            violations.append(part_id or "<EMPTY>")
            continue
        by_id[part_id] = item
    missing_checks = [part_id for part_id in allowed if part_id not in by_id]
    affected: list[dict[str, Any]] = []
    uncertain = []
    for part_id in allowed:
        check = by_id.get(part_id)
        if check is None:
            continue
        if check.get("status") == "UNCERTAIN":
            uncertain.append(part_id)
            continue
        matched, reason = _mismatch(check, error_type)
        if check.get("status") == "FAIL" and matched:
            affected.append({
                "part_id": part_id,
                "confidence": max(0.0, min(1.0, float(check.get("confidence") or 0.0))),
                "rule": reason,
            })
    normalized = str(error_type).lower().replace("_", "")
    if normalized in {"wrong", "wrongpart"} and not paired_roi_supported:
        uncertain.extend(item["part_id"] for item in affected)
        affected = []
    confidence = min((item["confidence"] for item in affected), default=0.0)
    requires_review = bool(violations or missing_checks or uncertain or not affected)
    return {
        "error_type": error_type,
        "affected_parts": affected,
        "affected_part_ids": [item["part_id"] for item in affected],
        "confidence": confidence,
        "candidate_membership_status": "violation" if violations else "valid",
        "candidate_violations": sorted(set(violations)),
        "missing_checks": missing_checks,
        "uncertain_part_ids": sorted(set(uncertain)),
        "paired_roi_supported": bool(paired_roi_supported),
        "requires_manual_review": requires_review,
        "verifier_status": "accepted" if affected and not requires_review else "conflict" if affected else "unresolved",
    }
