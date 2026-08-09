"""Evaluate affected-part identity predictions against confirmed labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

THRESHOLDS = (0.70, 0.80, 0.90)
BINS = ((0.0, 0.5, "0.00-0.49"), (0.5, 0.7, "0.50-0.69"), (0.7, 0.8, "0.70-0.79"), (0.8, 0.9, "0.80-0.89"), (0.9, 1.0000001, "0.90-1.00"))
UNKNOWN_PREFIXES = ("UNKNOWN", "UNRESOLVED")


def split_ids(value: Any) -> list[str]:
    return [item.strip().upper() for item in str(value or "").split("|") if item.strip() and item.strip().upper() != "NONE"]


def split_confidences(value: Any, count: int) -> list[float]:
    values = [float(item) for item in str(value or "").split("|") if item.strip()]
    if len(values) == 1 and count > 1:
        values *= count
    return (values + [0.0] * count)[:count]


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(ground_truth_rows: Iterable[dict[str, Any]], prediction_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    pred_by_id = {row["image_id"]: row for row in prediction_rows}
    gt_by_id = {
        row["image_id"]: row
        for row in ground_truth_rows
        if row.get("review_status", "confirmed") == "confirmed" and row["image_id"] in pred_by_id
    }
    cases = []
    identity_records = []
    for image_id, gt in gt_by_id.items():
        prediction = pred_by_id.get(image_id, {})
        expected = set(split_ids(gt.get("ground_truth_part_ids")))
        predicted_ids = split_ids(prediction.get("predicted_part_ids"))
        confidences = split_confidences(prediction.get("predicted_confidence"), len(predicted_ids))
        predicted = set(predicted_ids)
        for part_id, confidence in zip(predicted_ids, confidences):
            identity_records.append({"image_id": image_id, "part_id": part_id, "confidence": confidence, "correct": part_id in expected})
        cases.append({
            "image_id": image_id, "case_id": gt.get("case_id"), "view": gt.get("view_angle"),
            "error_type": gt.get("error_type"), "expected": expected, "predicted": predicted,
            "exact": expected == predicted, "composite": str(gt.get("is_composite", "false")).lower() == "true",
            "verifier_status": str(prediction.get("verifier_status") or "").lower(),
            "verified_part_ids": set(split_ids(prediction.get("verified_part_ids"))),
        })
    affected = [case for case in cases if case["expected"]]
    composite = [case for case in cases if case["composite"]]
    correct_controls = [case for case in cases if not case["expected"] and case["error_type"] == "correct"]
    true_positive = sum(len(case["expected"] & case["predicted"]) for case in cases)
    predicted_total = sum(len(case["predicted"]) for case in cases)
    expected_total = sum(len(case["expected"]) for case in cases)
    precision = safe_ratio(true_positive, predicted_total)
    recall = safe_ratio(true_positive, expected_total)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    unknown_count = sum(record["part_id"].startswith(UNKNOWN_PREFIXES) for record in identity_records)
    summary = {
        "evaluated_case_count": len(cases),
        "exact_set_match_accuracy": safe_ratio(sum(case["exact"] for case in cases), len(cases)),
        "at_least_one_part_recall": safe_ratio(sum(bool(case["expected"] & case["predicted"]) for case in affected), len(affected)),
        "all_parts_recall": safe_ratio(sum(case["expected"].issubset(case["predicted"]) for case in affected), len(affected)),
        "part_level_precision": precision,
        "part_level_recall": recall,
        "part_level_f1": f1,
        "unknown_part_rate": safe_ratio(unknown_count, predicted_total),
        "composite_full_recall": safe_ratio(sum(case["expected"].issubset(case["predicted"]) for case in composite), len(composite)),
        "correct_control_false_positive_rate": safe_ratio(sum(bool(case["predicted"]) for case in correct_controls), len(correct_controls)),
    }
    candidate_rows = [row for row in pred_by_id.values() if row.get("candidate_constraint_status") in {"valid", "violation"}]
    candidate_violations = [row for row in candidate_rows if row.get("candidate_constraint_status") == "violation"]
    high_candidate_rows = [row for row in candidate_rows if any(value >= 0.80 for value in split_confidences(row.get("predicted_confidence"), len(split_ids(row.get("predicted_part_ids")))))]
    high_candidate_violations = [row for row in high_candidate_rows if row.get("candidate_constraint_status") == "violation"]
    summary.update({
        "candidate_constraint_evaluated_count": len(candidate_rows),
        "candidate_violation_count": len(candidate_violations),
        "candidate_violation_rate": safe_ratio(len(candidate_violations), len(candidate_rows)),
        "high_confidence_candidate_violation_count": len(high_candidate_violations),
        "high_confidence_candidate_violation_rate": safe_ratio(len(high_candidate_violations), len(high_candidate_rows)),
    })
    false_confident = {}
    for threshold in THRESHOLDS:
        high = [record for record in identity_records if record["confidence"] >= threshold]
        false = [record for record in high if not record["correct"]]
        false_case_ids = {record["image_id"] for record in false}
        false_confident[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "false_confident_count": len(false),
            "high_confidence_prediction_count": len(high),
            "false_confident_identity_rate": safe_ratio(len(false), len(high)),
            "false_confident_case_count": len(false_case_ids),
            "evaluated_case_count": len(cases),
            "false_confident_case_rate": safe_ratio(len(false_case_ids), len(cases)),
        }
    calibration = []
    for low, high, label in BINS:
        records = [record for record in identity_records if low <= record["confidence"] < high]
        correct = sum(record["correct"] for record in records)
        calibration.append({"bin": label, "prediction_count": len(records), "correct_count": correct, "incorrect_count": len(records) - correct, "empirical_accuracy": safe_ratio(correct, len(records))})
    dimensions = {}
    for dimension, key in (("per_error_type", "error_type"), ("per_view", "view")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in cases:
            groups[str(case[key])].append(case)
        dimensions[dimension] = {name: {"case_count": len(group), "exact_set_match_accuracy": safe_ratio(sum(case["exact"] for case in group), len(group))} for name, group in sorted(groups.items())}
    valid_statuses = {"verified", "conflict", "uncertain", "unresolved"}
    verifier_cases = [case for case in cases if case["verifier_status"] in valid_statuses]
    statuses = Counter(case["verifier_status"] for case in verifier_cases)
    accepted = statuses.get("verified", 0)
    verified_wrong = sum(case["verifier_status"] == "verified" and bool(case["verified_part_ids"] - case["expected"]) for case in verifier_cases)
    verified_correct = sum(case["verifier_status"] == "verified" and bool(case["verified_part_ids"]) and case["verified_part_ids"].issubset(case["expected"]) for case in verifier_cases)
    wrong_blocked = sum(bool(case["predicted"] - case["expected"]) and case["verifier_status"] != "verified" for case in verifier_cases)
    correct_blocked = sum(case["predicted"] == case["expected"] and case["verifier_status"] != "verified" for case in verifier_cases)
    verifier_denominator = len(verifier_cases)
    verifier = {
        "evaluated_case_count": verifier_denominator,
        "acceptance_rate": safe_ratio(accepted, verifier_denominator),
        "conflict_rate": safe_ratio(statuses.get("conflict", 0), verifier_denominator),
        "unresolved_rate": safe_ratio(statuses.get("unresolved", 0) + statuses.get("uncertain", 0), verifier_denominator),
        "verified_correct_count": verified_correct,
        "verified_wrong_count": verified_wrong,
        "wrong_identity_escaped_verifier_count": verified_wrong,
        "wrong_identity_blocked_count": wrong_blocked,
        "correct_identity_blocked_count": correct_blocked,
    }
    return {"summary": summary, "false_confident": false_confident, "confidence_bins": calibration, **dimensions, "verifier": verifier}


def write_outputs(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "affected_part_baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for name, value in metrics["summary"].items(): rows.append({"section": "summary", "name": name, "value": value, "denominator": ""})
    for threshold, values in metrics["false_confident"].items():
        for name, value in values.items(): rows.append({"section": f"false_confident@{threshold}", "name": name, "value": value, "denominator": ""})
    for item in metrics["confidence_bins"]: rows.append({"section": "confidence_bin", "name": item["bin"], "value": item["empirical_accuracy"], "denominator": item["prediction_count"]})
    for section in ("per_error_type", "per_view"):
        for name, item in metrics[section].items(): rows.append({"section": section, "name": name, "value": item["exact_set_match_accuracy"], "denominator": item["case_count"]})
    with (output_dir / "affected_part_baseline_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "name", "value", "denominator"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate(_read(args.ground_truth), _read(args.predictions))
    write_outputs(metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
