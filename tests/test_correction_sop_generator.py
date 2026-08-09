from utils.correction_sop_generator import generate_correction_sop


def _report(part, error, **extra):
    return {"part_id": part, "error_type": error, **extra}


def test_missing_part_is_local_repair_with_visual_steps():
    sop = generate_correction_sop([_report("PIN_RED_SHORT", "missingpart")], {}, "step03")
    assert sop["repair_scope"] == "local"
    assert [step["action"] for step in sop["steps"]] == ["locate", "insert", "verify"]
    assert all(step["visual_instruction"] for step in sop["steps"])


def test_pair_swap_is_one_consolidated_plan():
    sop = generate_correction_sop([
        _report("PIN_YELLOW", "wrongpart"), _report("PIN_RED_SHORT", "wrongpart")
    ], {}, "step03")
    assert [step["action"] for step in sop["steps"]] == ["locate", "remove", "swap", "verify"]


def test_partial_and_full_rollback_scopes():
    partial = generate_correction_sop([_report("BASE", "positionerror", requires_rollback=True)], {}, "step03")
    full = generate_correction_sop([_report("BASE", "criticalerror")], {}, "step03")
    assert partial["repair_scope"] == "partial_rollback"
    assert full["repair_scope"] == "full_rollback"
