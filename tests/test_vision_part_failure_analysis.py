from tests.evaluate_vision_part_identification import build_rows, metrics


def test_july_first_analysis_is_deduplicated_and_separates_metrics():
    rows = build_rows()
    summary = metrics(rows)
    assert len(rows) == 58
    assert summary["error_type_accuracy"]["denominator"] == 58
    assert summary["affected_part_exact_match"]["denominator"] < 58
    assert summary["review_required_samples"] == 6


def test_requested_cases_and_views_are_present():
    rows = build_rows()
    names = {row["image_id"] for row in rows}
    assert any("missingpart-A01" in str(name) for name in names)
    assert any("missingpart-B01" in str(name) for name in names)
    assert any("wrongpart-A01" in str(name) for name in names)
    assert any("wrongpart-B01" in str(name) for name in names)
    assert {row["view_angle"] for row in rows} == {"top", "bottom", "front", "back", "left", "right"}
