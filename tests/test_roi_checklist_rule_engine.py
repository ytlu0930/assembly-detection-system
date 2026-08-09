from utils.roi_checklist_rule_engine import evaluate_roi_checklist


def check(part_id, *, ref, test, status="FAIL", ref_count=None, test_count=None, spatial=True, appearance=True, confidence=.9):
    return {"part_id": part_id, "reference_present": ref, "test_present": test, "reference_count": ref_count,
            "test_count": test_count, "spatial_match": spatial, "appearance_match": appearance,
            "status": status, "confidence": confidence, "evidence_summary": "observable ROI difference"}


def test_missing_rule_and_count_rule():
    result = evaluate_roi_checklist(
        checks=[check("A", ref=True, test=False), check("B", ref=True, test=True, ref_count=2, test_count=1)],
        candidate_part_ids=["A", "B"], error_type="missingpart",
    )
    assert result["affected_part_ids"] == ["A", "B"]
    assert result["candidate_membership_status"] == "valid"


def test_extra_and_position_rules():
    extra = evaluate_roi_checklist(checks=[check("A", ref=False, test=True)], candidate_part_ids=["A"], error_type="extrapart")
    position = evaluate_roi_checklist(checks=[check("A", ref=True, test=True, spatial=False)], candidate_part_ids=["A"], error_type="positionerror")
    assert extra["affected_part_ids"] == ["A"]
    assert position["affected_part_ids"] == ["A"]


def test_wrongpart_swap_requires_paired_roi_and_keeps_multiple_parts():
    checks = [check("EXPECTED", ref=True, test=False), check("OBSERVED", ref=False, test=True)]
    blocked = evaluate_roi_checklist(checks=checks, candidate_part_ids=["EXPECTED", "OBSERVED"], error_type="wrongpart", paired_roi_supported=False)
    paired = evaluate_roi_checklist(checks=checks, candidate_part_ids=["EXPECTED", "OBSERVED"], error_type="wrongpart", paired_roi_supported=True)
    assert blocked["affected_part_ids"] == [] and blocked["requires_manual_review"]
    assert paired["affected_part_ids"] == ["EXPECTED", "OBSERVED"]


def test_uncertain_and_candidate_violation_fail_closed():
    result = evaluate_roi_checklist(
        checks=[check("A", ref=None, test=None, status="UNCERTAIN"), check("OUTSIDE", ref=True, test=False)],
        candidate_part_ids=["A"], error_type="missingpart",
    )
    assert result["affected_part_ids"] == []
    assert result["candidate_membership_status"] == "violation"
    assert result["requires_manual_review"]
