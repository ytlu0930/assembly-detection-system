"""Compare affected-part identity metrics for Prompt variants A/B/C."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_affected_part_identity import _read, evaluate

VARIANTS = ("baseline", "reference", "reference_candidate")
LABELS = {"baseline": "Baseline", "reference": "Reference", "reference_candidate": "Reference+Candidate"}
METRICS = (
    "exact_set_match_accuracy", "at_least_one_part_recall", "all_parts_recall",
    "part_level_precision", "part_level_recall", "part_level_f1", "unknown_part_rate",
    "composite_full_recall", "correct_control_false_positive_rate",
)


def compare(ground_truth: Path, predictions_dir: Path) -> dict[str, Any]:
    gt = _read(ground_truth)
    results = {}
    for variant in VARIANTS:
        path = predictions_dir / f"{variant}.csv"
        results[variant] = evaluate(gt, _read(path)) if path.exists() else None
    rows = []
    baseline = results["baseline"]
    for metric in METRICS:
        base_value = baseline["summary"].get(metric) if baseline else None
        row = {"metric": metric}
        for variant in VARIANTS:
            value = results[variant]["summary"].get(metric) if results[variant] else None
            row[LABELS[variant]] = value
            row[f"{LABELS[variant]} delta_vs_baseline"] = None if value is None or base_value is None else value - base_value
        rows.append(row)
    threshold_metrics = [
        (f"false_confident_identity_rate@{threshold}", threshold, "false_confident_identity_rate")
        for threshold in ("0.70", "0.80", "0.90")
    ] + [("false_confident_case_rate@0.80", "0.80", "false_confident_case_rate")]
    for metric, threshold, source in threshold_metrics:
        base_value = baseline["false_confident"][threshold][source] if baseline else None
        row = {"metric": metric}
        for variant in VARIANTS:
            value = results[variant]["false_confident"][threshold][source] if results[variant] else None
            row[LABELS[variant]] = value
            row[f"{LABELS[variant]} delta_vs_baseline"] = None if value is None or base_value is None else value - base_value
        rows.append(row)
    for source in ("acceptance_rate", "conflict_rate", "unresolved_rate", "wrong_identity_escaped_verifier_count"):
        base_value = baseline["verifier"][source] if baseline else None
        row = {"metric": f"verifier_{source}"}
        for variant in VARIANTS:
            value = results[variant]["verifier"][source] if results[variant] else None
            row[LABELS[variant]] = value
            row[f"{LABELS[variant]} delta_vs_baseline"] = None if value is None or base_value is None else value - base_value
        rows.append(row)
    return {"variants": results, "comparison": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.ground_truth, args.predictions_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "affected_part_prompt_ab_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["metric", *sum(([LABELS[v], f"{LABELS[v]} delta_vs_baseline"] for v in VARIANTS), [])]
    with (args.output_dir / "affected_part_prompt_ab_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(payload["comparison"])
    print(json.dumps(payload["comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
