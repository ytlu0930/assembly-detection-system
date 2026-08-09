"""Offline-only evaluation for the fixed six-request targeted affected-part A/B run."""

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

VARIANTS = ("reference", "reference_candidate")
LABELS = {"reference": "Reference", "reference_candidate": "Reference+Candidate"}
HISTORICAL = {
    "denominator": 25,
    "exact_set_match_accuracy": 0.08,
    "at_least_one_part_recall": 0.125,
    "all_parts_recall": 0.0417,
    "part_level_f1": 0.105263,
    "false_confident_identity_rate@0.80": 0.88,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ids(parts: list[dict[str, Any]]) -> list[str]:
    return sorted({str(part.get("part_id") or "").upper() for part in parts if str(part.get("part_id") or "").strip()})


def _confidences(parts: list[dict[str, Any]], ids: list[str]) -> list[float]:
    scores = {part_id: 0.0 for part_id in ids}
    for part in parts:
        part_id = str(part.get("part_id") or "").upper()
        if part_id in scores:
            scores[part_id] = max(scores[part_id], float(part.get("confidence") or 0.0))
    return [scores[part_id] for part_id in ids]


def evaluate_run(run_dir: Path, ground_truth_path: Path) -> dict[str, Any]:
    evaluation_dir = run_dir / "evaluation"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "request_ledger.json").read_text(encoding="utf-8"))
    planned = {item["logical_request_id"]: item for item in manifest["planned_requests"]}
    reservations = {item["logical_request_id"]: item for item in ledger["reservations"]}
    gt_rows = _read(ground_truth_path)
    gt_by_image = {
        row["image_id"]: row for row in gt_rows
        if row.get("review_status") == "confirmed"
    }

    audit_rows = []
    prediction_rows = []
    response_files = sorted((run_dir / "responses").glob("*.json"))
    for response_path in response_files:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        logical_id = response["logical_request_id"]
        metadata = planned[logical_id]
        reservation = reservations.get(logical_id, {})
        image_id = Path(metadata["test_image"]).name
        gt = gt_by_image.get(image_id)
        if not gt:
            raise RuntimeError(f"No confirmed frozen Ground Truth for {image_id}")
        parsed = response.get("parsed_response") or {}
        parts = [part for part in parsed.get("detected_parts", []) if isinstance(part, dict) and part.get("error_type") != "correct"]
        predicted = _ids(parts)
        confidences = _confidences(parts, predicted)
        expected = sorted({item for item in str(gt.get("ground_truth_part_ids") or "").upper().split("|") if item})
        predicted_set, expected_set = set(predicted), set(expected)
        schema_valid = response.get("schema_validation_result", {}).get("status") == "valid"
        verifier = response.get("verifier_result") or {}
        verifier_status = verifier.get("verifier_status") or verifier.get("status") or "not_run"
        membership = response.get("candidate_membership_result") or {}
        candidate_ids = metadata.get("candidate_part_ids") or []
        audit_rows.append({
            "logical_request_id": logical_id,
            "reservation_id": reservation.get("reservation_id"),
            "reserved": bool(reservation),
            "sent": bool(response.get("api_request_id") or response.get("raw_response")),
            "response_received": response.get("raw_response") is not None,
            "raw_artifact_saved": response.get("raw_response") is not None,
            "parsed_output_saved": response.get("parsed_response") is not None,
            "schema_validation": response.get("schema_validation_result", {}).get("status"),
            "candidate_constraint_status": membership.get("candidate_constraint_status"),
            "verifier_result": verifier_status,
            "request_id": response.get("request_id"),
            "api_request_id": response.get("api_request_id"),
            "http_api_error_type": response.get("http_api_error_type"),
            "ledger_status": reservation.get("status"),
        })
        prediction_rows.append({
            "logical_request_id": logical_id,
            "image_id": image_id,
            "case_id": metadata["case_id"],
            "variant": metadata["variant"],
            "ground_truth_part_ids": "|".join(expected),
            "predicted_part_ids": "|".join(predicted),
            "predicted_confidence": "|".join(f"{value:.6f}" for value in confidences),
            "confidence": "|".join(f"{value:.6f}" for value in confidences),
            "error_type": parsed.get("overall_error_type"),
            "schema_valid": str(schema_valid).lower(),
            "candidate_part_ids": "|".join(candidate_ids),
            "candidate_count": len(candidate_ids),
            "candidate_constraint_status": membership.get("candidate_constraint_status"),
            "verifier_status": verifier_status,
            "verified_part_id": verifier.get("verified_part_id"),
            "verified_part_ids": verifier.get("verified_part_id"),
            "requires_manual_review": verifier.get("requires_manual_review"),
            "latency_seconds": response.get("request_duration_seconds"),
            "exact_match": predicted_set == expected_set,
            "at_least_one_match": bool(predicted_set & expected_set),
            "all_parts_match": expected_set.issubset(predicted_set),
            "source_response": response_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        })

    retries = sum(bool(item.get("explicit_retry")) for item in ledger["reservations"])
    audit_summary = {
        "run_uuid": manifest["run_uuid"],
        "logical_requests": len(manifest["planned_requests"]),
        "physical_requests": ledger["physical_request_counter"],
        "automatic_retries": manifest["automatic_retry"],
        "retry_requests": retries,
        "successful_http_requests": sum(row["response_received"] and not row["http_api_error_type"] for row in audit_rows),
        "schema_valid_requests": sum(row["schema_validation"] == "valid" for row in audit_rows),
        "all_reservations_completed": all(row["ledger_status"] == "completed" for row in audit_rows),
        "incident": not (
            len(manifest["planned_requests"]) == 6
            and ledger["physical_request_counter"] == 6
            and retries == 0
            and len(response_files) == 6
        ),
        "requests": audit_rows,
        "api_requests_during_offline_evaluation": 0,
    }
    _write_json(evaluation_dir / "request_audit_summary.json", audit_summary)

    prediction_fields = [
        "logical_request_id", "image_id", "case_id", "variant", "ground_truth_part_ids",
        "predicted_part_ids", "confidence", "error_type", "schema_valid", "candidate_part_ids",
        "candidate_count", "candidate_constraint_status", "verifier_status", "verified_part_id",
        "requires_manual_review", "latency_seconds", "source_response",
    ]
    _write_csv(evaluation_dir / "targeted_ab_predictions.csv", prediction_fields, prediction_rows)
    comparison_fields = [
        "case_id", "variant", "ground_truth_part_ids", "predicted_part_ids", "confidence",
        "exact_match", "at_least_one_match", "all_parts_match", "candidate_count",
        "candidate_constraint_status", "verifier_status", "verified_part_id", "requires_manual_review",
        "schema_valid",
    ]
    _write_csv(evaluation_dir / "targeted_ab_case_comparison.csv", comparison_fields, prediction_rows)

    variant_metrics = {}
    for variant in VARIANTS:
        primary = [row for row in prediction_rows if row["variant"] == variant and row["schema_valid"] == "true"]
        variant_metrics[variant] = evaluate(gt_rows, primary)
    metrics_payload = {
        "primary_metric_rule": "schema-valid responses with exact-image confirmed frozen Ground Truth only",
        "schema_validity": {
            variant: {
                "valid": sum(row["variant"] == variant and row["schema_valid"] == "true" for row in prediction_rows),
                "total": sum(row["variant"] == variant for row in prediction_rows),
                "rate": (
                    sum(row["variant"] == variant and row["schema_valid"] == "true" for row in prediction_rows)
                    / sum(row["variant"] == variant for row in prediction_rows)
                ),
            }
            for variant in VARIANTS
        },
        "variants": variant_metrics,
        "historical_baseline": HISTORICAL,
        "historical_comparison_caveat": "Directional only: historical denominator=25, targeted denominator=3.",
    }
    _write_json(evaluation_dir / "targeted_ab_metrics.json", metrics_payload)

    metric_sources = [
        ("Exact Set Match", "summary", "exact_set_match_accuracy"),
        ("At-least-one Recall", "summary", "at_least_one_part_recall"),
        ("All-parts Recall", "summary", "all_parts_recall"),
        ("Part Precision", "summary", "part_level_precision"),
        ("Part Recall", "summary", "part_level_recall"),
        ("Part F1", "summary", "part_level_f1"),
        ("Unknown Rate", "summary", "unknown_part_rate"),
        ("False-confident Identity Rate @0.70", "false:0.70", "false_confident_identity_rate"),
        ("False-confident Identity Rate @0.80", "false:0.80", "false_confident_identity_rate"),
        ("False-confident Identity Rate @0.90", "false:0.90", "false_confident_identity_rate"),
        ("False-confident Case Rate @0.80", "false:0.80", "false_confident_case_rate"),
        ("Candidate Violation Rate", "summary", "candidate_violation_rate"),
        ("Verifier Acceptance Rate", "verifier", "acceptance_rate"),
        ("Verifier Conflict Rate", "verifier", "conflict_rate"),
        ("Verifier Unresolved Rate", "verifier", "unresolved_rate"),
        ("Wrong Identity Escaped Verifier Count", "verifier", "wrong_identity_escaped_verifier_count"),
        ("Wrong Identity Blocked Count", "verifier", "wrong_identity_blocked_count"),
        ("Correct Identity Blocked Count", "verifier", "correct_identity_blocked_count"),
    ]
    comparison_rows = []
    for label, section, key in metric_sources:
        row = {"metric": label}
        for variant in VARIANTS:
            metric = variant_metrics[variant]
            if section.startswith("false:"):
                value = metric["false_confident"][section.split(":", 1)[1]][key]
            else:
                value = metric[section][key]
            row[LABELS[variant]] = value
        comparison_rows.append(row)
    _write_csv(
        evaluation_dir / "targeted_ab_metrics_comparison.csv",
        ["metric", "Reference", "Reference+Candidate"], comparison_rows,
    )

    bin_rows = []
    for variant in VARIANTS:
        for item in variant_metrics[variant]["confidence_bins"]:
            bin_rows.append({"variant": LABELS[variant], **item})
    _write_csv(
        evaluation_dir / "confidence_bins.csv",
        ["variant", "bin", "prediction_count", "correct_count", "incorrect_count", "empirical_accuracy"],
        bin_rows,
    )

    library = json.loads((PROJECT_ROOT / "config/part_library.json").read_text(encoding="utf-8"))
    canonical_count = len(library)
    candidate_rows = []
    for row in prediction_rows:
        if row["variant"] != "reference_candidate":
            continue
        candidates = [item for item in row["candidate_part_ids"].split("|") if item]
        gt_ids = [item for item in row["ground_truth_part_ids"].split("|") if item]
        predicted_ids = [item for item in row["predicted_part_ids"].split("|") if item]
        coverage = len([item for item in candidates if item in library]) / canonical_count if canonical_count else None
        strength = "weak" if coverage is not None and coverage >= 0.8 else "moderate" if coverage is not None and coverage >= 0.4 else "strong"
        candidate_rows.append({
            "case_id": row["case_id"], "candidate_count": len(candidates),
            "candidate_ids": "|".join(candidates), "ground_truth_in_candidate_set": set(gt_ids).issubset(candidates),
            "prediction_in_candidate_set": set(predicted_ids).issubset(candidates),
            "candidate_constraint_status": row["candidate_constraint_status"],
            "inventory_coverage_ratio": coverage, "constraint_strength": strength,
        })
    _write_csv(
        evaluation_dir / "candidate_effectiveness.csv",
        ["case_id", "candidate_count", "candidate_ids", "ground_truth_in_candidate_set",
         "prediction_in_candidate_set", "candidate_constraint_status", "inventory_coverage_ratio", "constraint_strength"],
        candidate_rows,
    )

    candidate_metric = variant_metrics["reference_candidate"]
    decision = {
        "decision": "NO_CLEAR_IMPROVEMENT",
        "recommended_variant": "NONE",
        "next_experiment": "LOCALIZATION_GUIDED_ROI",
        "phase_2b_recommendation": "BLOCK",
        "reasons": [
            "Reference primary denominator is zero because all three responses echoed schema metadata and failed validation.",
            "Reference+Candidate exact match and all-parts recall are both zero.",
            "Reference+Candidate retained high-confidence EYE_BALL errors for both missing-part cases.",
            "wrongpart-B01 recovered only one of the two Ground Truth swap identities.",
            "Candidate violation rate and verifier wrong-identity escape are zero, but all candidate sets cover the full inventory and are weak constraints.",
        ],
        "candidate_success_checks": {
            "candidate_violation_rate_zero": candidate_metric["summary"]["candidate_violation_rate"] == 0,
            "wrong_identity_escape_zero": candidate_metric["verifier"]["wrong_identity_escaped_verifier_count"] == 0,
            "exact_match_nonzero": bool(candidate_metric["summary"]["exact_set_match_accuracy"]),
            "all_parts_recall_nonzero": bool(candidate_metric["summary"]["all_parts_recall"]),
        },
        "api_requests_during_offline_evaluation": 0,
    }
    _write_json(evaluation_dir / "targeted_ab_decision.json", decision)
    return {
        "audit": audit_summary,
        "predictions": prediction_rows,
        "metrics": metrics_payload,
        "candidate_effectiveness": candidate_rows,
        "decision": decision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "analysis/affected_part_eval_ground_truth.csv")
    args = parser.parse_args()
    result = evaluate_run(args.run_dir, args.ground_truth)
    print(json.dumps({
        "logical_requests": result["audit"]["logical_requests"],
        "physical_requests": result["audit"]["physical_requests"],
        "schema_valid_requests": result["audit"]["schema_valid_requests"],
        "decision": result["decision"]["decision"],
        "api_requests_during_offline_evaluation": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
