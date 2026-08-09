from utils.error_report_adapter import adapt_vision_result, attach_bbox


def test_correct_case_returns_no_reports():
    assert adapt_vision_result({"is_error": False, "overall_error_type": "correct", "detected_parts": []}) == []


def test_multiple_and_composite_parts_are_preserved():
    payload = {
        "is_error": True,
        "overall_error_type": "wrongpart",
        "error_components": ["wrongpart", "extrapart"],
        "detected_parts": [
            {"part_id": "EYE_BALL", "error_type": "wrongpart", "confidence": .9},
            {"part_id": "unknown_extra_part", "error_type": "extrapart", "confidence": .7},
        ],
    }
    reports = adapt_vision_result(payload)
    assert [item["part_id"] for item in reports] == ["EYE_BALL", "unknown_extra_part"]
    assert reports[1]["unresolved"] is True
    assert reports[0]["error_components"] == ["wrongpart", "extrapart"]


def test_bbox_is_attached_without_mutating_source():
    reports = adapt_vision_result({"is_error": True, "overall_error_type": "missingpart", "detected_parts": []})
    enriched = attach_bbox(reports, 0, [1, 2, 3, 4])
    assert reports[0]["bbox"] is None
    assert enriched[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
