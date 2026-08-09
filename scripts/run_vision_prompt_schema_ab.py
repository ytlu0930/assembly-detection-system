"""Safe planning and offline evaluation framework for Vision Prompt/Schema A/B."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "missingpart-A01": "input/missingpart/model03_step03/model03_step03_missingpart-A01_front_01.jpg",
    "missingpart-B01": "input/missingpart/model03_step03/model03_step03_missingpart-B01_front_01.jpg",
    "extrapart-A01": "input/extrapart/model03_step03/model03_step03_extrapart-A01_front_01.jpg",
    "wrongpart-A01": "input/wrongpart/model03_step03/model03_step03_wrongpart-A01_front_01.jpg",
    "wrongpart-B01": "input/wrongpart/model03_step03/model03_step03_wrongpart-B01_front_01.jpg",
    "correct-control": "input/normal/model03_step01/model03_step01_correct-01_front_01.jpg",
}

VARIANTS = {
    "baseline_prompt_current_schema": {
        "prompt": "prompts/vision_v2.txt",
        "schema": "schema/vision_output_schema.json",
    },
    "improved_prompt_current_schema": {
        "prompt": "experiments/prompts/vision_affected_parts_candidate.txt",
        "schema": "schema/vision_output_schema.json",
    },
    "improved_prompt_schema_vnext": {
        "prompt": "experiments/prompts/vision_affected_parts_candidate.txt",
        "schema": "experiments/schema/vision_output_schema_vnext_candidate.json",
    },
}

METRIC_NAMES = (
    "affected_part_exact_match",
    "at_least_one_part_recall",
    "all_parts_recall",
    "composite_recall",
    "unknown_part_rate",
    "error_type_accuracy",
)


def build_plan(case_names: list[str] | None = None, variants: list[str] | None = None, max_cases: int | None = None) -> dict[str, Any]:
    selected_cases = case_names or list(CASES)
    selected_variants = variants or list(VARIANTS)
    unknown_cases = set(selected_cases) - set(CASES)
    unknown_variants = set(selected_variants) - set(VARIANTS)
    if unknown_cases or unknown_variants:
        raise ValueError(f"Unknown cases={sorted(unknown_cases)} variants={sorted(unknown_variants)}")
    if max_cases is not None:
        selected_cases = selected_cases[: max(0, max_cases)]
    jobs = []
    for variant in selected_variants:
        for case in selected_cases:
            image = CASES[case]
            step = "step01" if case == "correct-control" else "step03"
            jobs.append({
                "case": case,
                "variant": variant,
                "test_image": image,
                "reference_image": f"input/normal/model03_{step}/model03_{step}_correct-01_front_01.jpg",
                "expected_state": f"ground_truth/model03/{step}.json",
                **VARIANTS[variant],
            })
    return {"cases": selected_cases, "variants": selected_variants, "jobs": jobs, "estimated_api_calls": len(jobs), "metrics": list(METRIC_NAMES)}


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Evaluate normalized confirmed-label records without calling an API."""
    if not records:
        return {name: None for name in METRIC_NAMES}
    exact = partial = all_parts = composite = type_correct = 0
    affected_denom = composite_denom = predicted_total = unknown_total = 0
    for record in records:
        expected = set(record.get("expected_part_ids", []))
        predicted = set(record.get("predicted_part_ids", []))
        expected_components = set(record.get("expected_error_components", []))
        predicted_components = set(record.get("predicted_error_components", []))
        exact += predicted == expected
        type_correct += record.get("expected_error_type") == record.get("predicted_error_type")
        if expected:
            affected_denom += 1
            partial += bool(expected & predicted)
            all_parts += expected.issubset(predicted)
        if len(expected_components) > 1 or record.get("is_composite_error"):
            composite_denom += 1
            composite += expected.issubset(predicted) and expected_components.issubset(predicted_components)
        predicted_total += len(predicted)
        unknown_total += sum(value.lower().startswith(("unknown", "unresolved")) for value in predicted)
    return {
        "affected_part_exact_match": exact / len(records),
        "at_least_one_part_recall": partial / affected_denom if affected_denom else None,
        "all_parts_recall": all_parts / affected_denom if affected_denom else None,
        "composite_recall": composite / composite_denom if composite_denom else None,
        "unknown_part_rate": unknown_total / predicted_total if predicted_total else 0.0,
        "error_type_accuracy": type_correct / len(records),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--no-api", action="store_true", help="Explicitly retain offline mode")
    result.add_argument("--execute-api", action="store_true", help="Requires a separately approved API adapter; never sufficient by itself")
    result.add_argument("--max-cases", type=int)
    result.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    result.add_argument("--case", action="append", choices=sorted(CASES))
    result.add_argument("--output-dir", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    plan = build_plan(args.case, args.variant, args.max_cases)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or PROJECT_ROOT / "output" / "vision_prompt_schema_ab" / run_id
    print(json.dumps({**plan, "mode": "dry-run", "output_dir": str(output)}, ensure_ascii=False, indent=2))
    if args.execute_api:
        raise RuntimeError("API execution is intentionally unimplemented until budget approval and an audited adapter are provided")
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print("API calls executed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
