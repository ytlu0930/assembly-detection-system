import copy
import inspect
import json
from pathlib import Path

import pytest

from utils.roi_checklist_response_normalizer import normalize_roi_checklist_response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / "analysis/roi_direct_vs_checklist/run_20260809_preflight"


def check(**overrides):
    value = {
        "part_id": "PART_A", "reference_present": True, "test_present": False,
        "reference_count": 1, "test_count": 0, "spatial_match": False,
        "appearance_match": False, "status": "FAIL", "confidence": 0.9,
        "evidence_summary": "observed only in reference",
    }
    value.update(overrides)
    return value


def normalize(value, candidates=None):
    return normalize_roi_checklist_response(value, candidate_part_ids=candidates or ["PART_A"])


def test_valid_schema_input_is_already_valid():
    result = normalize({"checks": [check()]})
    assert result["normalization_status"] == "already_valid"
    assert result["transformations_applied"] == []
    assert result["schema_valid_after_normalization"] is True


@pytest.mark.parametrize("alias", ["CHECK", "check_result"])
def test_results_and_status_alias_are_normalized(alias):
    item = check()
    item.pop("status")
    item[alias] = "FAIL"
    result = normalize({"results": [item]})
    assert result["normalization_status"] == "normalized"
    assert result["checks"][0]["status"] == "FAIL"
    assert "results->checks" in result["transformations_applied"]
    assert f"{alias}->status" in result["transformations_applied"]


def test_numeric_string_confidence_becomes_float():
    result = normalize({"checks": [check(confidence="0.90")]})
    assert result["checks"][0]["confidence"] == 0.9
    assert "string_confidence->float" in result["transformations_applied"]


def test_missing_evidence_summary_gets_nonsemantic_fallback():
    item = check(); item.pop("evidence_summary")
    result = normalize({"checks": [item]})
    assert result["schema_valid_after_normalization"] is True
    assert result["checks"][0]["evidence_summary"] == "No explicit evidence summary supplied in the response."


def test_existing_observation_text_is_preserved_as_evidence_summary():
    item = check(observation="visible only in the reference ROI"); item.pop("evidence_summary")
    result = normalize({"checks": [item]})
    assert result["checks"][0]["evidence_summary"] == "visible only in the reference ROI"


@pytest.mark.parametrize(
    "payload",
    [
        {"checks": [check(status="MAYBE")]},
        {"checks": [check(confidence="very high")]},
        {"checks": []},
    ],
)
def test_malformed_input_fails_closed(payload):
    result = normalize(payload)
    assert result["normalization_status"] == "failed"
    assert result["normalized_response"] is None
    assert result["schema_valid_after_normalization"] is False
    assert result["failure_reason"]


def test_candidate_outside_allowed_set_fails_closed():
    result = normalize_roi_checklist_response({"checks": [check(part_id="OUTSIDE")]}, candidate_part_ids=["PART_A"])
    assert result["normalization_status"] == "failed"
    assert result["candidate_membership_valid"] is False


def test_no_ground_truth_dependency_and_raw_object_not_mutated():
    assert "ground_truth" not in inspect.signature(normalize_roi_checklist_response).parameters
    raw = {"results": [check(CHECK="FAIL", status="FAIL", confidence="HIGH")]}
    original = copy.deepcopy(raw)
    result = normalize(raw)
    assert raw == original
    assert result["gt_used"] is False
    assert result["input_mutated"] is False


def test_deterministic_and_idempotent():
    item = check(confidence="HIGH"); item.pop("evidence_summary")
    first = normalize({"results": [item]})
    repeated = normalize({"results": [item]})
    second = normalize(first["normalized_response"])
    assert first == repeated
    assert second["normalization_status"] == "already_valid"
    assert second["normalized_response"] == first["normalized_response"]
    assert second["normalized_sha256"] == first["normalized_sha256"]


def test_all_three_real_frozen_responses_normalize_successfully():
    filenames = (
        "EXP-002_missingpart-A01_roi_checklist.json",
        "EXP-004_missingpart-B01_roi_checklist.json",
        "EXP-006_wrongpart-B01_roi_checklist.json",
    )
    for filename in filenames:
        response = json.loads((RUN_DIR / "responses" / filename).read_text(encoding="utf-8"))
        content = response["raw_response"]["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        metadata = json.loads((RUN_DIR / "packages" / response["case_id"] / "roi_checklist/request_metadata.json").read_text(encoding="utf-8"))
        result = normalize_roi_checklist_response(parsed, candidate_part_ids=metadata["candidate_part_ids"])
        assert result["normalization_status"] == "normalized"
        assert result["schema_valid_after_normalization"] is True
        assert result["candidate_membership_valid"] is True
        assert result["gt_used"] is False
