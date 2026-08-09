import json

import pytest
from jsonschema import ValidationError, validate

import scripts.run_targeted_affected_part_ab as targeted
from utils.experiment_request_guard import ExperimentRequestGuard, PhysicalBudgetExhaustedError


def test_targeted_preflight_builds_exact_fixed_matrix_and_new_zero_ledger(tmp_path):
    run_dir = tmp_path / "targeted"
    manifest = targeted.build_targeted_preflight(run_dir, run_uuid="00000000-0000-4000-8000-000000000001")
    checked = targeted.validate_preflight(run_dir)
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    assert checked["status"] == "PASS"
    assert manifest["logical_request_limit"] == 6
    assert manifest["physical_request_hard_ceiling"] == 6
    assert manifest["automatic_retry"] == 0
    assert ledger["physical_request_counter"] == 0
    assert ledger["reservations"] == []
    assert ledger["run_uuid"] == manifest["run_uuid"]
    assert {(item["case_id"], item["variant"]) for item in manifest["planned_requests"]} == {
        (case, variant)
        for case in ("missingpart-A01", "missingpart-B01", "wrongpart-B01")
        for variant in ("reference", "reference_candidate")
    }
    assert (run_dir / "requests").is_dir()
    assert (run_dir / "responses").is_dir()
    assert (run_dir / "evaluation").is_dir()


def test_targeted_candidates_are_runtime_reproducible_and_review_independent(tmp_path):
    run_dir = tmp_path / "targeted"
    manifest = targeted.build_targeted_preflight(run_dir)
    for item in manifest["planned_requests"]:
        if item["variant"] == "reference_candidate":
            assert item["candidate_metadata"]["source"] == "expected_state+part_library"
            assert item["candidate_metadata"]["human_review_source_used"] is False
        else:
            assert item["candidate_part_ids"] == []
    source = (targeted.PROJECT_ROOT / "utils/affected_part_candidate_builder.py").read_text(encoding="utf-8")
    assert "affected_parts_review_template" not in source
    assert "affected_part_eval_ground_truth" not in source
    assert "PIN_RED_SHORT" not in source
    assert "missingpart-A01" not in source


def test_targeted_seventh_physical_request_fails_closed_after_six_persisted_reservations(tmp_path):
    item = ExperimentRequestGuard(
        experiment_id="targeted-affected-part-ab", run_uuid="fixed-run",
        lock_path=tmp_path / ".lock", ledger_path=tmp_path / "ledger.json",
        max_physical_requests=6,
    )
    item.acquire()
    for number in range(1, 7):
        request_id = item.reserve(f"TGT-{number:03d}")
        persisted = item.read_ledger()
        assert persisted["physical_request_counter"] == number
        assert persisted["reservations"][-1]["request_id"] == request_id
        assert persisted["reservations"][-1]["run_uuid"] == "fixed-run"
    with pytest.raises(PhysicalBudgetExhaustedError):
        item.reserve("TGT-007")
    assert item.read_ledger()["physical_request_counter"] == 6
    item.release()


def test_reference_response_parser_and_current_schema_validation():
    schema = json.loads(targeted.CURRENT_SCHEMA.read_text(encoding="utf-8"))
    payload = {
        "model_id": "model03", "step_id": "step03", "view_angle": "front",
        "is_error": True, "overall_error_type": "missingpart",
        "detected_parts": [{
            "part_id": "UNKNOWN", "error_type": "missingpart",
            "description": "localized difference", "confidence": 0.5,
        }],
        "summary": "A difference requires review.",
    }
    parsed = targeted._extract_json("```json\n" + json.dumps(payload) + "\n```")
    validate(instance=parsed, schema=schema)
    with pytest.raises(ValidationError):
        validate(instance={**parsed, "$schema": "echoed"}, schema=schema)


class _Message:
    content = json.dumps({
        "$schema": "echoed", "model_id": "model03", "step_id": "step03", "view_angle": "front",
        "is_error": True, "overall_error_type": "missingpart",
        "detected_parts": [{
            "part_id": "UNKNOWN", "error_type": "missingpart",
            "description": "localized difference", "confidence": 0.5,
        }],
        "summary": "A difference requires review.",
    })


class _Response:
    id = "fake-api-id"
    choices = [type("Choice", (), {"message": _Message()})()]
    usage = None

    def model_dump(self):
        return {"id": self.id, "choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeClient:
    def __init__(self):
        self.calls = 0
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    def create(self, **_kwargs):
        self.calls += 1
        return _Response()


def test_schema_failure_is_persisted_and_never_automatically_retried(tmp_path, monkeypatch):
    run_dir = tmp_path / "targeted"
    manifest = targeted.build_targeted_preflight(run_dir)
    manifest["planned_requests"] = manifest["planned_requests"][:1]
    manifest["logical_request_limit"] = 6
    # Keep the fixed six-package manifest on disk; execute only the first via a controlled in-memory validation result.
    validation = {"status": "PASS", "failures": [], "manifest": manifest}
    monkeypatch.setattr(targeted, "validate_preflight", lambda _path: validation)
    monkeypatch.setattr(targeted, "safe_environment_preflight", lambda **_kwargs: {
        "ready": True, "model": "offline-fake", "api_version": "offline", "provider": "fake",
    })
    client = _FakeClient()
    result = targeted.execute_targeted(run_dir, confirmed_run_uuid=manifest["run_uuid"], client=client, offline_verifier=object())
    assert result["api_requests_made"] == 1
    saved = json.loads(next((run_dir / "responses").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["schema_validation_result"]["status"] == "invalid"
    assert saved["raw_response"] is not None
    assert saved["parsed_response"] is not None
    for field in (
        "candidate_membership_result", "verifier_result", "request_duration_seconds",
        "http_api_error_type", "request_id", "logical_request_id", "api_request_id",
    ):
        assert field in saved
    targeted.execute_targeted(run_dir, confirmed_run_uuid=manifest["run_uuid"], client=client, offline_verifier=object())
    assert client.calls == 1
