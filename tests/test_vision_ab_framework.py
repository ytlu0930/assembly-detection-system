import json
from pathlib import Path

from scripts.run_vision_prompt_schema_ab import METRIC_NAMES, build_plan, evaluate_records


ROOT = Path(__file__).resolve().parents[1]


def test_default_plan_is_six_cases_by_three_variants_and_offline_safe():
    plan = build_plan()
    assert len(plan["cases"]) == 6
    assert len(plan["variants"]) == 3
    assert plan["estimated_api_calls"] == 18
    assert set(plan["metrics"]) == set(METRIC_NAMES)


def test_filters_and_max_cases_are_deterministic():
    plan = build_plan(["missingpart-A01", "correct-control"], ["improved_prompt_current_schema"], 1)
    assert [job["case"] for job in plan["jobs"]] == ["missingpart-A01"]


def test_offline_metrics_include_composite_and_unknown_rates():
    metrics = evaluate_records([
        {
            "expected_part_ids": ["A", "B"], "predicted_part_ids": ["A", "unknown_part"],
            "expected_error_components": ["wrongpart", "extrapart"], "predicted_error_components": ["wrongpart"],
            "expected_error_type": "wrongpart", "predicted_error_type": "wrongpart", "is_composite_error": True,
        }
    ])
    assert metrics["affected_part_exact_match"] == 0.0
    assert metrics["at_least_one_part_recall"] == 1.0
    assert metrics["all_parts_recall"] == 0.0
    assert metrics["composite_recall"] == 0.0
    assert metrics["unknown_part_rate"] == 0.5
    assert metrics["error_type_accuracy"] == 1.0


def test_candidate_schema_is_experimental_and_excludes_bbox():
    path = ROOT / "experiments" / "schema" / "vision_output_schema_vnext_candidate.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "EXPERIMENTAL" in payload["$comment"]
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()
    assert "bbox" not in keys(payload)
    assert payload["properties"]["detected_parts"]["type"] == "array"
