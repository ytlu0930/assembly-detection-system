from utils.affected_part_candidate_builder import build_affected_part_candidates


EXPECTED = {"model_id": "model03", "step_id": "step03", "expected_parts": [{"part_id": "B"}, {"part_id": "A"}, {"part_id": "A"}]}
LIBRARY = {"A": ["a"], "B": ["b"], "C": ["c"]}


def build(error_type, **kwargs):
    return build_affected_part_candidates(model_id="model03", step_id="step03", expected_state=EXPECTED, part_library=LIBRARY, error_type=error_type, **kwargs)


def test_missing_candidates_are_deterministic_stable_and_unique():
    first = build("missingpart")
    second = build("missingpart")
    assert first == second
    assert first["candidate_part_ids"] == ["A", "B"]
    assert len(first["candidate_part_ids"]) == len(set(first["candidate_part_ids"]))


def test_extra_allows_unknown_extra_after_canonical_candidates():
    assert build("extrapart")["candidate_part_ids"] == ["A", "B", "UNKNOWN_EXTRA_PART"]


def test_wrongpart_adds_only_canonical_observed_and_swap_candidates():
    result = build("wrongpart", observed_part_ids=["C", "FAKE"], swap_candidate_pairs=[["B", "C"]])
    assert result["candidate_part_ids"] == ["A", "B", "C"]


def test_candidate_builder_has_no_human_ground_truth_source_or_case_hardcode():
    result = build("missingpart")
    assert result["candidate_metadata"]["human_review_source_used"] is False
    assert "PIN_RED_SHORT" not in result["candidate_part_ids"]
    assert "A01" not in repr(result)


def test_mismatched_expected_state_is_rejected():
    bad = {**EXPECTED, "step_id": "step99"}
    try:
        build_affected_part_candidates(model_id="model03", step_id="step03", expected_state=bad, part_library=LIBRARY, error_type="missingpart")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched expected state must fail")
