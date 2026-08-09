from __future__ import annotations

import inspect

from utils.roi_candidate_builder import build_roi_candidates


LIBRARY = {
    "EYE_BALL": ["white ball with black pupil"],
    "PIN_RED_SHORT": ["short red cylinder stick"],
    "PIN_YELLOW": ["yellow pin with flat head"],
    "WHEEL_BLUE_SMALL": ["small blue wheel"],
}
EXPECTED = {
    "expected_parts": [
        {"part_id": "EYE_BALL", "color": "WHITE", "position": "TOP"},
        {"part_id": "PIN_RED_SHORT", "color": "RED", "position": "LEFT"},
        {"part_id": "PIN_YELLOW", "color": "YELLOW", "position": "CENTER"},
        {"part_id": "WHEEL_BLUE_SMALL", "color": "BLUE", "position": "RIGHT"},
    ]
}


def _evidence(color: str, family: str, *, score: float = 0.8, relation: str = "reference_only"):
    return {
        "color": color,
        "shape_family": family,
        "position": "LEFT",
        "score": score,
        "status": "success",
        "bbox": [10, 10, 30, 30],
        "reference_bbox": [10, 10, 30, 30] if relation == "reference_only" else None,
        "test_bbox": [10, 10, 30, 30] if relation == "test_only" else None,
        "difference_relation": relation,
    }


def test_candidate_reduction_and_deterministic_ordering():
    kwargs = dict(
        expected_state=EXPECTED,
        part_library=LIBRARY,
        roi_evidence=[_evidence("YELLOW", "pin"), _evidence("RED", "pin")],
        error_type="wrongpart",
        view_angle="top",
    )
    first = build_roi_candidates(**kwargs)
    second = build_roi_candidates(**kwargs)
    assert first == second
    assert first["candidate_part_ids"] == ["PIN_RED_SHORT", "PIN_YELLOW"]
    assert first["candidate_count"] < first["full_candidate_count"]
    assert {"PIN_RED_SHORT", "PIN_YELLOW"}.issubset(first["candidate_part_ids"])
    assert "EYE_BALL" not in first["candidate_part_ids"]


def test_low_localization_fails_closed():
    result = build_roi_candidates(
        expected_state=EXPECTED,
        part_library=LIBRARY,
        roi_evidence=[_evidence("RED", "pin", score=0.2)],
        error_type="missingpart",
        view_angle="front",
    )
    assert result["status"] == "localization_insufficient"
    assert result["candidate_part_ids"] == []


def test_no_ground_truth_or_review_input_surface():
    signature = inspect.signature(build_roi_candidates)
    forbidden = {"ground_truth", "review_csv", "case_id", "confirmed_gt_parts"}
    assert forbidden.isdisjoint(signature.parameters)
    source = inspect.getsource(build_roi_candidates).lower()
    assert "review_csv" not in source
    assert "affected_part_eval_ground_truth" not in source
