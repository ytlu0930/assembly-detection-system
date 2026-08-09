import json
from pathlib import Path

import pytest

import scripts.run_affected_part_prompt_ab as ab_runner
from scripts.build_affected_part_baseline import build_ground_truth, select_subset
from scripts.evaluate_affected_part_prompt_ab import compare
from scripts.run_affected_part_prompt_ab import CURRENT_SCHEMA, build_packages, main
from scripts.run_affected_part_prompt_ab import enforce_candidate_constraint
from scripts.finalize_affected_part_prompt_ab import quarantine_candidate_violation

ROOT = Path(__file__).resolve().parents[1]


def test_ab_dry_run_request_count_and_current_schema(tmp_path):
    plan = build_packages(variants=["baseline", "reference", "reference_candidate"], case_limit=6, output_dir=tmp_path)
    assert plan["number_of_cases"] == 6
    assert plan["number_of_variants"] == 3
    assert plan["estimated_requests"] == 18
    assert plan["api_calls_performed"] == 0
    assert plan["current_schema"] == "schema/vision_output_schema.json"
    assert len(plan["package_paths"]) == 18


def test_packages_have_no_key_or_ground_truth_label_leakage(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "TOP_SECRET_SENTINEL")
    build_packages(variants=["reference_candidate"], case_limit=1, output_dir=tmp_path)
    metadata = next(tmp_path.glob("packages/reference_candidate/*/request_metadata.json"))
    combined = metadata.read_text(encoding="utf-8") + metadata.with_name("prompt.txt").read_text(encoding="utf-8")
    assert "TOP_SECRET_SENTINEL" not in combined
    assert "ground_truth_part_ids" not in combined
    assert json.loads(metadata.read_text(encoding="utf-8"))["contains_api_key"] is False


def test_default_cli_is_dry_run_and_never_calls_api(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_affected_part_prompt_ab.py", "--case-limit", "1", "--output-dir", str(tmp_path)])
    assert main() == 0
    output = capsys.readouterr().out
    assert '"api_calls_performed": 0' in output


def test_execute_requires_cost_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_affected_part_prompt_ab.py", "--execute-api", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="requires --confirm-cost"):
        main()


def test_confirmed_execute_uses_audited_adapter_only_after_both_gates(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(ab_runner, "execute_packages", lambda **kwargs: called.append(kwargs) or {"physical_requests": 0})
    monkeypatch.setattr("sys.argv", ["run_affected_part_prompt_ab.py", "--execute-api", "--confirm-cost", "--output-dir", str(tmp_path)])
    assert main() == 0
    assert len(called) == 1


def test_execution_lock_blocks_duplicate_process(tmp_path, monkeypatch):
    lock = tmp_path / "results" / ".execution.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("existing-pid", encoding="utf-8")
    called = []
    monkeypatch.setattr(ab_runner, "execute_packages", lambda **kwargs: called.append(kwargs))
    monkeypatch.setattr("sys.argv", ["run_affected_part_prompt_ab.py", "--execute-api", "--confirm-cost", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="experiment lock"):
        main()
    assert called == []


def test_ab_evaluator_uses_shared_metrics_and_delta(tmp_path):
    gt_path = tmp_path / "gt.csv"
    gt_path.write_text(
        "image_id,ground_truth_part_ids,review_status,is_composite,error_type,view_angle,case_id\n"
        "one,A,confirmed,false,missing,front,one\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    header = "image_id,predicted_part_ids,predicted_confidence\n"
    (predictions / "baseline.csv").write_text(header + "one,X,0.95\n", encoding="utf-8")
    (predictions / "reference.csv").write_text(header + "one,A,0.95\n", encoding="utf-8")
    payload = compare(gt_path, predictions)
    exact = next(row for row in payload["comparison"] if row["metric"] == "exact_set_match_accuracy")
    assert exact["Baseline"] == 0.0
    assert exact["Reference"] == 1.0
    assert exact["Reference delta_vs_baseline"] == 1.0
    assert exact["Reference+Candidate"] is None


def test_confirmed_subset_never_promotes_unreviewed_rows():
    gt = build_ground_truth(ROOT / "analysis/affected_parts_review_template.csv")
    subset = select_subset(gt)
    assert gt and all(row["review_status"] == "confirmed" for row in gt)
    assert all(row["case_id"] not in {"extrapart-A01", "wrongpart-A01"} for row in subset)
    assert 12 <= len(subset) <= 24
    assert CURRENT_SCHEMA == ROOT / "schema/vision_output_schema.json"


def test_candidate_runtime_enforcement_is_deterministic_and_never_maps_violation():
    assert enforce_candidate_constraint("baseline", ["A"], ["X"])["candidate_constraint_status"] == "not_applicable"
    assert enforce_candidate_constraint("reference_candidate", ["A"], ["A"])["candidate_constraint_status"] == "valid"
    unknown = enforce_candidate_constraint("reference_candidate", ["A"], ["UNKNOWN_EXTRA_PART"])
    assert unknown["candidate_constraint_status"] == "valid"
    violation = enforce_candidate_constraint("reference_candidate", ["A"], ["X"])
    assert violation == {"candidate_constraint_status": "violation", "candidate_constraint_violations": ["X"]}
    quarantined = quarantine_candidate_violation(
        {"verifier_status": "verified", "verified_part_id": "A", "requires_manual_review": False},
        violation["candidate_constraint_status"],
    )
    assert quarantined["verified_part_id"] is None
    assert quarantined["requires_manual_review"] is True


def test_candidate_set_audit_marks_full_inventory_missing_constraint_weak():
    rows = list(__import__("csv").DictReader((ROOT / "analysis/vision_prompt_ab/results/candidate_set_audit.csv").open(encoding="utf-8")))
    a01 = next(row for row in rows if row["case_id"] == "missingpart-A01")
    assert a01["candidate_count"] == a01["canonical_library_count"] == "15"
    assert a01["constraint_effectiveness"] == "weak"
