import json

import pytest

from scripts.run_roi_direct_vs_checklist_experiment import (
    AUTOMATIC_RETRY, LOGICAL_REQUEST_LIMIT, PHYSICAL_REQUEST_HARD_CEILING,
    _sanitize_response, build_preflight, candidate_membership, gt_leakage_audit, validate_preflight,
)
from scripts.evaluate_roi_direct_vs_checklist import recover_checklist_for_analysis
from scripts.freeze_roi_experiment_responses import freeze_and_audit
from utils.experiment_request_guard import ExperimentRequestGuard, ExplicitRetryRequiredError, PhysicalBudgetExhaustedError


def test_exactly_six_frozen_packages_and_no_label_leakage(tmp_path):
    run_dir = tmp_path / "run"
    manifest = build_preflight(run_dir, run_uuid="00000000-0000-4000-8000-000000000001")
    assert len(manifest["planned_requests"]) == 6
    assert [(item["case_id"], item["method"]) for item in manifest["planned_requests"]] == [
        ("missingpart-A01", "roi_direct"), ("missingpart-A01", "roi_checklist"),
        ("missingpart-B01", "roi_direct"), ("missingpart-B01", "roi_checklist"),
        ("wrongpart-B01", "roi_direct"), ("wrongpart-B01", "roi_checklist"),
    ]
    assert validate_preflight(run_dir)["status"] == "PASS"
    assert gt_leakage_audit(run_dir)["status"] == "PASS"
    assert manifest["automatic_retry"] == 0
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    assert ledger["physical_request_counter"] == 0


def test_response_sanitizer_and_dynamic_candidate_membership():
    clean, removed = _sanitize_response({"$schema": "x", "title": "echo", "checks": []})
    assert clean == {"checks": []} and removed == ["$schema", "title"]
    direct = candidate_membership("roi_direct", {"affected_parts": [{"part_id": "UNKNOWN"}]}, ["A"])
    invalid = candidate_membership("roi_direct", {"affected_parts": [{"part_id": "OUTSIDE"}]}, ["A"])
    checklist = candidate_membership("roi_checklist", {"checks": [{"part_id": "A"}]}, ["A", "B"])
    assert direct["status"] == "valid"
    assert invalid["status"] == "violation"
    assert checklist["status"] == "violation" and checklist["missing_or_duplicate_candidate_checks"] == ["B"]


def test_six_request_ceiling_and_reserved_resume_are_fail_closed(tmp_path):
    guard = ExperimentRequestGuard(experiment_id="roi", lock_path=tmp_path / "lock", ledger_path=tmp_path / "ledger.json", max_physical_requests=6)
    guard.acquire()
    for index in range(6):
        reservation = guard.reserve(f"EXP-{index}")
        guard.finish(reservation, "completed")
    with pytest.raises(PhysicalBudgetExhaustedError):
        guard.reserve("EXP-007")
    guard.release()
    resumed = ExperimentRequestGuard(experiment_id="roi", lock_path=tmp_path / "lock", ledger_path=tmp_path / "ledger.json", max_physical_requests=6)
    resumed.acquire()
    assert resumed.reserve("EXP-0") is None
    resumed.release()
    assert (LOGICAL_REQUEST_LIMIT, PHYSICAL_REQUEST_HARD_CEILING, AUTOMATIC_RETRY) == (6, 6, 0)


def test_reserved_request_requires_explicit_retry(tmp_path):
    first = ExperimentRequestGuard(experiment_id="roi", lock_path=tmp_path / "lock", ledger_path=tmp_path / "ledger.json", max_physical_requests=6)
    first.acquire(); first.reserve("EXP-001"); first.release()
    resumed = ExperimentRequestGuard(experiment_id="roi", lock_path=tmp_path / "lock", ledger_path=tmp_path / "ledger.json", max_physical_requests=6)
    resumed.acquire()
    with pytest.raises(ExplicitRetryRequiredError): resumed.reserve("EXP-001")
    resumed.release()


def test_checklist_analysis_recovery_is_label_free_and_candidate_bound():
    recovered = recover_checklist_for_analysis({"results": [
        {"part_id": "A", "reference_present": True, "test_present": False, "reference_count": 1,
         "test_count": 0, "spatial_match": False, "appearance_match": False, "confidence": "high", "CHECK": "FAIL"},
    ]}, ["A"])
    assert recovered["status"] == "recovered"
    assert recovered["checks"][0]["status"] == "FAIL"
    assert recovered["checks"][0]["confidence"] == 0.9
    assert recovered["labels_used_for_recovery"] is False
    assert recovered["excluded_from_original_schema_valid_rate"] is True
    assert recover_checklist_for_analysis({"results": []}, ["A"])["status"] == "unrecoverable"


def test_freeze_audit_requires_exactly_six_and_loads_no_labels(tmp_path):
    run_dir = tmp_path / "run"
    manifest = build_preflight(run_dir, run_uuid="00000000-0000-4000-8000-000000000002")
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    ledger["physical_request_counter"] = 6
    ledger["reservations"] = []
    for item in manifest["planned_requests"]:
        reservation_id = "reservation-" + item["logical_request_id"]
        ledger["reservations"].append({"logical_request_id": item["logical_request_id"], "reservation_id": reservation_id, "status": "completed", "explicit_retry": False})
        response = {"logical_request_id": item["logical_request_id"], "case_id": item["case_id"], "method": item["method"],
                    "request_id": reservation_id, "api_request_id": "api-" + item["logical_request_id"],
                    "raw_response": {"content": "{}"}, "parsed_response": {},
                    "schema_validation_result": {"status": "invalid"}, "candidate_membership_result": {"status": "violation"},
                    "rule_engine_result": None, "verifier_result": {"verifier_status": "unresolved"},
                    "request_duration_seconds": 1.0, "http_api_error_type": None}
        (run_dir / "responses" / f"{item['logical_request_id']}.json").write_text(json.dumps(response), encoding="utf-8")
    (run_dir / "request_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    audit = freeze_and_audit(run_dir)
    assert audit["status"] == "PASS" and audit["labels_loaded"] is False
    frozen = json.loads((run_dir / "evaluation/frozen_responses/frozen_manifest.json").read_text(encoding="utf-8"))
    assert frozen["response_count"] == 6 and frozen["labels_loaded"] is False
