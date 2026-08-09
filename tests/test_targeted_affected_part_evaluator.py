import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "analysis/vision_prompt_ab/targeted_run_20260809_111248/evaluation"


def test_targeted_request_audit_is_exactly_six_without_retry_or_incident():
    audit = json.loads((EVALUATION / "request_audit_summary.json").read_text(encoding="utf-8"))
    assert audit["logical_requests"] == 6
    assert audit["physical_requests"] == 6
    assert audit["automatic_retries"] == audit["retry_requests"] == 0
    assert audit["successful_http_requests"] == 6
    assert audit["incident"] is False
    assert all(item["raw_artifact_saved"] and item["parsed_output_saved"] for item in audit["requests"])


def test_targeted_primary_metrics_exclude_schema_invalid_reference_responses():
    metrics = json.loads((EVALUATION / "targeted_ab_metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_validity"]["reference"] == {"valid": 0, "total": 3, "rate": 0.0}
    assert metrics["schema_validity"]["reference_candidate"] == {"valid": 3, "total": 3, "rate": 1.0}
    assert metrics["variants"]["reference"]["summary"]["exact_set_match_accuracy"] is None
    candidate = metrics["variants"]["reference_candidate"]
    assert candidate["summary"]["exact_set_match_accuracy"] == 0.0
    assert candidate["summary"]["at_least_one_part_recall"] == 1 / 3
    assert candidate["verifier"]["wrong_identity_escaped_verifier_count"] == 0


def test_targeted_case_results_and_candidate_strength_are_regression_locked():
    with (EVALUATION / "targeted_ab_case_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidate = {row["case_id"]: row for row in rows if row["variant"] == "reference_candidate"}
    assert candidate["missingpart-A01"]["predicted_part_ids"] == "EYE_BALL"
    assert candidate["missingpart-B01"]["predicted_part_ids"] == "EYE_BALL"
    assert candidate["wrongpart-B01"]["predicted_part_ids"] == "PIN_RED_SHORT"
    with (EVALUATION / "candidate_effectiveness.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert all(row["candidate_count"] == "15" for row in candidates)
    assert all(row["constraint_strength"] == "weak" for row in candidates)


def test_targeted_decision_remains_fail_closed():
    decision = json.loads((EVALUATION / "targeted_ab_decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "NO_CLEAR_IMPROVEMENT"
    assert decision["recommended_variant"] == "NONE"
    assert decision["next_experiment"] == "LOCALIZATION_GUIDED_ROI"
    assert decision["phase_2b_recommendation"] == "BLOCK"
