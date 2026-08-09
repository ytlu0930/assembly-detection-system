import pytest

from utils.affected_part_identity_verifier import (
    AffectedPartIdentityVerifier,
    VALID_IDENTITY_STATUSES,
)


LIBRARY = {
    "PIN_RED_SHORT": ["short red cylinder stick"],
    "EYE_BALL": ["white ball with black pupil"],
    "PIN_YELLOW": ["yellow pin"],
}
EXPECTED = {
    "expected_parts": [
        {"part_id": "PIN_RED_SHORT", "position": "LEFT"},
        {"part_id": "EYE_BALL", "position": "TOP"},
        {"part_id": "PIN_YELLOW", "position": "CENTER"},
    ]
}


def _loc(count, score=0.95):
    return {
        "status": "success",
        "estimated_count": count,
        "selected_detection_score": score,
        "selected_bbox": [1, 2, 30, 40] if count else None,
    }


def _verify(part="PIN_RED_SHORT", error="missingpart", evidence=None, expected=EXPECTED):
    return AffectedPartIdentityVerifier().verify(
        error_report={"part_id": part, "error_type": error, "confidence": 0.99},
        test_image_metadata={"image_name": "test.jpg", "view_angle": "front"},
        reference_image_metadata={"image_name": "reference.jpg"},
        expected_state=expected,
        localization_evidence=evidence or {},
        part_library=LIBRARY,
    )


def test_valid_and_correct_prediction_is_verified_only_with_difference_evidence():
    result = _verify(evidence={"reference_localization": _loc(2), "test_localization": _loc(1)})
    assert result.identity_status == "verified"
    assert result.verified_part_id == "PIN_RED_SHORT"
    assert result.requires_manual_review is False


def test_valid_but_wrong_high_confidence_prediction_is_conflict():
    result = _verify(
        part="EYE_BALL",
        evidence={"reference_localization": _loc(2), "test_localization": _loc(2)},
    )
    assert result.identity_status == "conflict"
    assert result.verified_part_id is None
    assert result.requires_manual_review is True


def test_unknown_part_is_unresolved():
    result = _verify(part="UNKNOWN_WIDGET", evidence={"reference_localization": _loc(1), "test_localization": _loc(0)})
    assert result.identity_status == "unresolved" and result.verified_part_id is None


def test_insufficient_evidence_is_uncertain():
    result = _verify(evidence={"reference_localization": _loc(1)})
    assert result.identity_status == "uncertain" and result.requires_manual_review


def test_conflicting_expected_inventory_is_conflict():
    result = _verify(expected={"expected_parts": []}, evidence={"reference_localization": _loc(1), "test_localization": _loc(0)})
    assert result.identity_status == "conflict" and result.verified_part_id is None


def test_missingpart_relation_uses_reference_minus_test_count():
    supported = _verify(evidence={"reference_localization": _loc(1), "test_localization": _loc(0)})
    refuted = _verify(evidence={"reference_localization": _loc(1), "test_localization": _loc(1)})
    assert supported.identity_status == "verified"
    assert refuted.identity_status == "conflict"


def test_extrapart_relation_uses_test_minus_reference_count():
    supported = _verify(error="extrapart", evidence={"reference_localization": _loc(1), "test_localization": _loc(2)})
    refuted = _verify(error="extrapart", evidence={"reference_localization": _loc(1), "test_localization": _loc(1)})
    assert supported.identity_status == "verified"
    assert refuted.identity_status == "conflict"


def test_wrongpart_requires_explicit_relation_or_cross_view_support():
    result = _verify(
        error="wrongpart",
        evidence={"relation_supported": True, "relation_confidence": 0.91},
    )
    assert result.identity_status == "verified"
    assert result.verified_part_id == "PIN_RED_SHORT"


def test_swap_composite_candidates_are_ranked_and_thresholded():
    result = _verify(
        part="EYE_BALL",
        error="wrongpart",
        evidence={
            "relation_supported": False,
            "relation_confidence": 0.95,
            "candidate_evidence": [
                {
                    "part_id": "PIN_RED_SHORT",
                    "score": 0.93,
                    "difference_supported": True,
                    "evidence": ["independent reference/test delta"],
                },
                {
                    "part_id": "PIN_YELLOW",
                    "score": 0.70,
                    "difference_supported": True,
                    "evidence": ["weaker candidate"],
                },
            ],
        },
    )
    # Explicit relation conflict returns before alternative substitution.  The
    # ranked candidates remain advisory for swap/composite manual review.
    assert result.identity_status == "conflict"
    assert result.verified_part_id is None
    assert [item["part_id"] for item in result.alternative_candidates] == ["PIN_RED_SHORT", "PIN_YELLOW"]
    assert result.identity_status in VALID_IDENTITY_STATUSES


@pytest.mark.parametrize("status", sorted(VALID_IDENTITY_STATUSES))
def test_result_status_vocabulary_is_closed(status):
    assert status in {"verified", "conflict", "uncertain", "unresolved"}
