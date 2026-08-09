"""Offline evaluation of frozen ROI Direct vs Checklist responses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_checklist_confusion_matrix import confusion_counts, render as render_confusion
from scripts.render_roi_thesis_case_figure import render_case_figure
from utils.deterministic_correction_annotator import annotate_correction
from utils.roi_checklist_rule_engine import evaluate_roi_checklist
from utils.roi_checklist_response_normalizer import normalize_roi_checklist_response

METHODS = ("roi_direct", "roi_checklist")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def recover_checklist_for_analysis(parsed: dict[str, Any], candidate_part_ids: list[str]) -> dict[str, Any]:
    """Normalize observable checklist fields and enforce candidate membership."""
    normalized = normalize_roi_checklist_response(parsed, candidate_part_ids=candidate_part_ids)
    checks = normalized["checks"]
    ids = [item["part_id"] for item in checks]
    allowed = [str(item).upper() for item in candidate_part_ids]
    membership_valid = bool(checks) and not any(item not in allowed for item in ids) and all(ids.count(item) == 1 for item in allowed)
    schema_valid = normalized["schema_valid_after_normalization"]
    return {
        "status": "recovered" if schema_valid and membership_valid else "unrecoverable",
        "checks": checks, "changes": normalized["transformations_applied"],
        "membership_status": "valid" if membership_valid else "violation",
        "schema_valid": schema_valid, "schema_error": normalized["failure_reason"],
        "recovered_for_analysis": True, "excluded_from_original_schema_valid_rate": True,
        "labels_used_for_recovery": False, "input_mutated": normalized["input_mutated"],
    }


def _method_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows); tp = fp = fn = exact = at_least = all_parts = unknown = manual = 0
    high_predictions = {0.70: [0, 0], 0.80: [0, 0], 0.90: [0, 0]}
    high_cases = {0.70: [0, 0], 0.80: [0, 0], 0.90: [0, 0]}
    verifier = {"accepted": 0, "conflict": 0, "unresolved": 0, "wrong_identity_escaped": 0, "wrong_identity_blocked": 0, "correct_identity_blocked": 0}
    for row in rows:
        expected, predicted = set(row["ground_truth_parts"]), set(row["predicted_parts"])
        is_exact = predicted == expected
        exact += is_exact; at_least += bool(predicted & expected); all_parts += expected.issubset(predicted)
        unknown += bool(row["unknown"]); manual += bool(row["manual_review"])
        tp += len(predicted & expected); fp += len(predicted - expected); fn += len(expected - predicted)
        for threshold, counts in high_predictions.items():
            qualifying = [(part, score) for part, score in row["part_confidences"].items() if score >= threshold]
            counts[0] += sum(part not in expected for part, _ in qualifying); counts[1] += len(qualifying)
            if qualifying:
                high_cases[threshold][1] += 1
                high_cases[threshold][0] += any(part not in expected for part, _ in qualifying)
        status = row["verifier_status"] if row["verifier_status"] in {"accepted", "conflict", "unresolved"} else "unresolved"
        verifier[status] += 1
        if predicted != expected and status == "accepted": verifier["wrong_identity_escaped"] += 1
        if predicted != expected and status != "accepted": verifier["wrong_identity_blocked"] += 1
        if is_exact and status != "accepted": verifier["correct_identity_blocked"] += 1
    precision, recall = _safe_ratio(tp, tp + fp), _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
    return {
        "denominator": total, "exact_set_match": _safe_ratio(exact, total),
        "at_least_one_recall": _safe_ratio(at_least, total), "all_parts_recall": _safe_ratio(all_parts, total),
        "part_precision": precision, "part_recall": recall, "part_f1": f1,
        "unknown_rate": _safe_ratio(unknown, total), "manual_review_rate": _safe_ratio(manual, total),
        "false_confident_identity": {f"{key:.2f}": _safe_ratio(value[0], value[1]) for key, value in high_predictions.items()},
        "false_confident_case": {f"{key:.2f}": _safe_ratio(value[0], value[1]) for key, value in high_cases.items()},
        "verifier_acceptance": _safe_ratio(verifier["accepted"], total),
        "verifier_conflict": _safe_ratio(verifier["conflict"], total),
        "verifier_unresolved": _safe_ratio(verifier["unresolved"], total),
        **{key: verifier[key] for key in ("wrong_identity_escaped", "wrong_identity_blocked", "correct_identity_blocked")},
    }


def _render_method_chart(metrics: dict[str, Any], output: Path) -> None:
    labels = ["Exact Match", "At-least-one", "All-parts", "Part F1", "False-confident @.80"]
    keys = ("exact_set_match", "at_least_one_recall", "all_parts_recall", "part_f1")
    values = {}
    for method in METHODS:
        values[method] = [(metrics[method].get(key) or 0) for key in keys] + [(metrics[method]["false_confident_identity"].get("0.80") or 0)]
    try:
        import matplotlib.pyplot as plt
        positions = range(len(labels)); fig, axis = plt.subplots(figsize=(10, 5.5))
        axis.bar([item - .2 for item in positions], values["roi_direct"], .4, label="ROI Direct")
        axis.bar([item + .2 for item in positions], values["roi_checklist"], .4, label="ROI Checklist")
        axis.set_xticks(list(positions), labels, rotation=15, ha="right"); axis.set_ylim(0, 1); axis.set_ylabel("Rate"); axis.legend(); fig.tight_layout()
        fig.savefig(output, dpi=300); plt.close(fig)
    except ModuleNotFoundError:
        import cv2
        import numpy as np
        canvas = np.full((1650, 3000, 3), 255, dtype=np.uint8); baseline = 1350
        cv2.putText(canvas, "ROI Direct vs Checklist", (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 4)
        cv2.rectangle(canvas, (1030, 150), (1110, 205), (220, 120, 40), -1)
        cv2.putText(canvas, "ROI Direct", (1130, 197), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
        cv2.rectangle(canvas, (1530, 150), (1610, 205), (40, 150, 220), -1)
        cv2.putText(canvas, "ROI Checklist", (1630, 197), cv2.FONT_HERSHEY_SIMPLEX, .9, (20, 20, 20), 2)
        for index, label in enumerate(labels):
            x = 250 + index * 540
            direct_height = int(1000 * values["roi_direct"][index]); checklist_height = int(1000 * values["roi_checklist"][index])
            cv2.rectangle(canvas, (x, baseline - direct_height), (x + 150, baseline), (220, 120, 40), -1)
            cv2.rectangle(canvas, (x + 170, baseline - checklist_height), (x + 320, baseline), (40, 150, 220), -1)
            cv2.putText(canvas, f"{values['roi_direct'][index]:.2f}", (x + 18, baseline - direct_height - 20), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 2)
            cv2.putText(canvas, f"{values['roi_checklist'][index]:.2f}", (x + 188, baseline - checklist_height - 20), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 2)
            cv2.putText(canvas, label, (x - 45, 1430), cv2.FONT_HERSHEY_SIMPLEX, .58, (20, 20, 20), 2)
        cv2.imwrite(str(output), canvas)


def evaluate_run(run_dir: Path, label_path: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    frozen_manifest_path = run_dir / "evaluation/frozen_responses/frozen_manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    frozen_rows = frozen_manifest.get("responses") or []
    if frozen_manifest.get("snapshot_status") != "PASS" or len(frozen_rows) != 6 or frozen_manifest.get("labels_loaded") is not False:
        raise RuntimeError("Six-response label-free frozen snapshot is required before label join")
    for item in frozen_rows:
        frozen_path = Path(item["frozen_response_path"])
        if _sha256(frozen_path) != item["frozen_sha256"]:
            raise RuntimeError(f"Frozen response hash changed: {item['logical_request_id']}")
    planned = {item["logical_request_id"]: item for item in manifest["planned_requests"]}
    responses = [json.loads(Path(item["frozen_response_path"]).read_text(encoding="utf-8")) for item in frozen_rows]

    # Label join starts only after all snapshot/hash assertions above pass.
    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        labels = {
            row["image_id"]: sorted(item for item in row["ground_truth_part_ids"].upper().split("|") if item)
            for row in csv.DictReader(handle) if row.get("review_status") == "confirmed"
        }

    evaluation, thesis, figures = run_dir / "evaluation", run_dir / "thesis_tables", run_dir / "figures"
    recovered_dir = evaluation / "recovered_checklist"; recovered_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []; checklist_rows: list[dict[str, Any]] = []; efficiency_rows = []
    annotated_by_case: dict[str, str] = {}; localization_by_case: dict[str, str] = {}
    metadata_by_case_method = {}
    for response in responses:
        plan = planned[response["logical_request_id"]]
        metadata_path = run_dir / plan["package_metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")); metadata_by_case_method[(metadata["case_id"], metadata["method"])] = (metadata_path, metadata)
        image_id = Path(metadata["test_image"]).name
        if image_id not in labels: raise RuntimeError(f"No confirmed exact-image label: {image_id}")
        expected = labels[image_id]
        original_schema_valid = (response.get("schema_validation_result") or {}).get("status") == "valid"
        analysis_recovered = False; analysis_schema_valid = original_schema_valid; rule_result = response.get("rule_engine_result")
        parsed = response.get("parsed_response") or {}; checks = parsed.get("checks") or []
        membership_status = (response.get("candidate_membership_result") or {}).get("status")
        if metadata["method"] == "roi_checklist" and not original_schema_valid:
            recovery = recover_checklist_for_analysis(parsed, metadata["candidate_part_ids"])
            _write_json(recovered_dir / f"{response['logical_request_id']}.json", {
                "logical_request_id": response["logical_request_id"], "case_id": metadata["case_id"],
                "original_schema_valid": False, **recovery,
            })
            if recovery["status"] == "recovered":
                analysis_recovered = True; analysis_schema_valid = True; checks = recovery["checks"]; membership_status = recovery["membership_status"]
                rule_result = evaluate_roi_checklist(checks=checks, candidate_part_ids=metadata["candidate_part_ids"], error_type=metadata["error_type"], paired_roi_supported=metadata["paired_roi_supported"])
        if metadata["method"] == "roi_direct":
            parts = parsed.get("affected_parts") or []
        else:
            parts = (rule_result or {}).get("affected_parts") or []
        clean_parts = []
        for item in parts:
            part_id = str(item.get("part_id") or "").upper()
            if part_id and not part_id.startswith(("UNKNOWN", "UNRESOLVED")):
                clean_parts.append({"part_id": part_id, "confidence": float(item.get("confidence") or 0.0)})
        predicted = sorted({item["part_id"] for item in clean_parts})
        confidence_by_id = {part_id: max(item["confidence"] for item in clean_parts if item["part_id"] == part_id) for part_id in predicted}
        analysis_verifier = "conflict" if predicted else "unresolved"
        manual = True  # Every frozen ROI PoC package requires manual review.
        row = {
            "logical_request_id": response["logical_request_id"], "case_id": metadata["case_id"], "method": metadata["method"],
            "error_type": metadata["error_type"], "ground_truth_parts": expected, "candidate_ids": metadata["candidate_part_ids"],
            "candidate_count": metadata["candidate_count"], "candidate_reduction": metadata["candidate_reduction_ratio"],
            "localization_score": metadata["localization_score"], "predicted_parts": predicted,
            "part_confidences": confidence_by_id, "prediction_confidence": min(confidence_by_id.values(), default=0.0),
            "unknown": not predicted, "original_schema_valid": original_schema_valid,
            "analysis_recovered": analysis_recovered, "analysis_schema_valid": analysis_schema_valid,
            "candidate_constraint_status": membership_status, "verifier_status": analysis_verifier,
            "source_verifier_status": (response.get("verifier_result") or {}).get("verifier_status"),
            "verified_part_id": None, "manual_review": manual,
            "latency_seconds": response.get("request_duration_seconds"),
            "source_result": next(item["frozen_response_path"] for item in frozen_rows if item["logical_request_id"] == response["logical_request_id"]),
            "checklist_fail_parts": [item["part_id"] for item in checks if item.get("status") == "FAIL"],
            "checklist_uncertain_parts": [item["part_id"] for item in checks if item.get("status") == "UNCERTAIN"],
            "rule_engine_prediction": (rule_result or {}).get("affected_part_ids", []),
        }
        case_rows.append(row)
        usage = response.get("usage") or {}
        efficiency_rows.append({"method": metadata["method"], "case_id": metadata["case_id"], "request_count": 1, "latency_seconds": response.get("request_duration_seconds"), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "total_tokens": usage.get("total_tokens")})
        if metadata["method"] == "roi_checklist":
            for check in checks:
                part_id = check["part_id"]; status = check["status"]; gt_status = "MISMATCH" if part_id in expected else "NORMAL"
                checklist_rows.append({
                    "case_id": metadata["case_id"], "part_id": part_id, "gt_status": gt_status,
                    "reference_present": check.get("reference_present"), "test_present": check.get("test_present"),
                    "reference_count": check.get("reference_count"), "test_count": check.get("test_count"),
                    "spatial_match": check.get("spatial_match"), "appearance_match": check.get("appearance_match"),
                    "predicted_status": status, "confidence": check.get("confidence"),
                    "correct": ((status == "FAIL") == (gt_status == "MISMATCH")) if status != "UNCERTAIN" else None,
                    "uncertain": status == "UNCERTAIN", "analysis_recovered": analysis_recovered,
                })
            output = figures / "cases" / f"{metadata['case_id'].replace('-', '_')}_result.png"
            verified_parts = clean_parts if analysis_verifier == "accepted" else []
            annotated_by_case[metadata["case_id"]] = annotate_correction(test_image=metadata["test_image"], bbox_evidence=metadata["bbox_evidence"], affected_parts=verified_parts, error_type=metadata["error_type"], output_path=output, requires_manual_review=manual)
            localization_output = figures / "cases" / f"{metadata['case_id'].replace('-', '_')}_roi_localization.png"
            localization_by_case[metadata["case_id"]] = annotate_correction(
                test_image=metadata["reference_image"], bbox_evidence=metadata["bbox_evidence"],
                affected_parts=clean_parts, error_type=metadata["error_type"], output_path=localization_output,
                requires_manual_review=manual, evidence_role="reference", label_prefix="UNVERIFIED_ROI",
            )

    metrics = {method: _method_metrics([row for row in case_rows if row["method"] == method]) for method in METHODS}
    strict_rows = []
    for row in case_rows:
        strict = dict(row)
        if row["method"] == "roi_checklist" and not row["original_schema_valid"]:
            strict.update({"predicted_parts": [], "part_confidences": {}, "unknown": True, "verifier_status": "unresolved"})
        strict_rows.append(strict)
    strict_metrics = {method: _method_metrics([row for row in strict_rows if row["method"] == method]) for method in METHODS}
    schema_validity = {method: {"valid": sum(row["method"] == method and row["original_schema_valid"] for row in case_rows), "total": 3} for method in METHODS}
    for value in schema_validity.values(): value["rate"] = value["valid"] / value["total"]
    normalized_analysis_validity = {method: {"valid": sum(row["method"] == method and row["analysis_schema_valid"] for row in case_rows), "total": 3} for method in METHODS}
    for value in normalized_analysis_validity.values(): value["rate"] = value["valid"] / value["total"]
    metrics_payload = {
        "primary_semantic_rule": "experiment-only deterministic checklist normalization; exact-image confirmed labels joined after response freeze",
        "original_schema_validity": schema_validity,
        "normalized_analysis_schema_validity": normalized_analysis_validity,
        "semantic_metrics": metrics, "strict_original_schema_metrics": strict_metrics,
        "api_requests_during_evaluation": 0,
    }
    _write_json(evaluation / "roi_direct_vs_checklist_metrics.json", metrics_payload)

    prediction_fields = ["case_id", "error_type", "method", "ground_truth_parts", "candidate_ids", "candidate_count", "candidate_reduction", "localization_score", "predicted_parts", "prediction_confidence", "exact_match", "at_least_one_match", "all_parts_match", "original_schema_valid", "analysis_recovered", "analysis_schema_valid", "candidate_constraint_status", "verifier_status", "verified_part_id", "manual_review_required", "latency_seconds", "source_result", "checklist_fail_parts", "checklist_uncertain_parts", "rule_engine_prediction"]
    prediction_csv_rows = []
    for row in case_rows:
        expected, predicted = set(row["ground_truth_parts"]), set(row["predicted_parts"])
        prediction_csv_rows.append({
            **row, "ground_truth_parts": "|".join(row["ground_truth_parts"]), "candidate_ids": "|".join(row["candidate_ids"]),
            "predicted_parts": "|".join(row["predicted_parts"]), "exact_match": predicted == expected,
            "at_least_one_match": bool(predicted & expected), "all_parts_match": expected.issubset(predicted),
            "manual_review_required": row["manual_review"], "checklist_fail_parts": "|".join(row["checklist_fail_parts"]),
            "checklist_uncertain_parts": "|".join(row["checklist_uncertain_parts"]), "rule_engine_prediction": "|".join(row["rule_engine_prediction"]),
        })
    _write_csv(evaluation / "roi_direct_vs_checklist_predictions.csv", prediction_fields, prediction_csv_rows)

    checklist_fields = ["case_id", "part_id", "gt_status", "reference_present", "test_present", "reference_count", "test_count", "spatial_match", "appearance_match", "predicted_status", "confidence", "correct", "uncertain", "analysis_recovered"]
    _write_csv(evaluation / "checklist_component_results.csv", checklist_fields, checklist_rows)
    _write_csv(thesis / "checklist_component_results.csv", checklist_fields, checklist_rows)
    counts = confusion_counts(checklist_rows); resolved = counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"]
    component_precision = _safe_ratio(counts["TP"], counts["TP"] + counts["FP"]); component_recall = _safe_ratio(counts["TP"], counts["TP"] + counts["FN"])
    component_metrics = {
        **counts, "resolved_count": resolved, "total_count": len(checklist_rows),
        "accuracy": _safe_ratio(counts["TP"] + counts["TN"], resolved), "precision": component_precision, "recall": component_recall,
        "f1": _safe_ratio(2 * component_precision * component_recall, component_precision + component_recall) if component_precision is not None and component_recall is not None else None,
        "uncertain_rate": _safe_ratio(counts["UNCERTAIN"], len(checklist_rows)),
    }
    _write_json(evaluation / "checklist_component_metrics.json", component_metrics)
    render_confusion(checklist_rows, figures / "checklist_confusion_matrix.png", figures / "checklist_confusion_matrix.csv")

    metric_defs = [
        ("Exact Set Match", "exact_set_match", "higher"), ("At-least-one Recall", "at_least_one_recall", "higher"),
        ("All-parts Recall", "all_parts_recall", "higher"), ("Part Precision", "part_precision", "higher"),
        ("Part Recall", "part_recall", "higher"), ("Part F1", "part_f1", "higher"), ("Unknown Rate", "unknown_rate", "lower"),
        ("False-confident @0.80", "false_confident_identity.0.80", "lower"), ("Manual Review Rate", "manual_review_rate", "lower"),
        ("Wrong Identity Escaped Verifier", "wrong_identity_escaped", "lower"),
    ]
    def metric(method: str, key: str):
        if key.startswith("false_confident_identity."):
            return metrics[method]["false_confident_identity"].get(key.split(".", 1)[1])
        value: Any = metrics[method]
        for part in key.split("."): value = value.get(part) if isinstance(value, dict) else None
        return value
    comparison_rows = []
    for label, key, direction in metric_defs:
        direct, checklist = metric("roi_direct", key), metric("roi_checklist", key)
        delta = checklist - direct if direct is not None and checklist is not None else None
        winner = "N/A" if delta is None else "tie" if delta == 0 else ("roi_checklist" if (delta > 0) == (direction == "higher") else "roi_direct")
        comparison_rows.append({"metric": label, "roi_direct": direct, "roi_checklist": checklist, "delta": delta, "preferred_direction": direction, "winner": winner})
    comparison_fields = ["metric", "roi_direct", "roi_checklist", "delta", "preferred_direction", "winner"]
    _write_csv(evaluation / "roi_direct_vs_checklist_metrics.csv", comparison_fields, comparison_rows)
    _write_csv(thesis / "roi_direct_vs_checklist_metrics.csv", comparison_fields, comparison_rows)
    _render_method_chart(metrics, figures / "method_comparison_metrics.png")

    case_table = []
    for case_id in manifest["cases"]:
        direct = next(row for row in case_rows if row["case_id"] == case_id and row["method"] == "roi_direct")
        checklist = next(row for row in case_rows if row["case_id"] == case_id and row["method"] == "roi_checklist")
        case_table.append({
            "case_id": case_id, "error_type": direct["error_type"], "ground_truth_parts": "|".join(direct["ground_truth_parts"]),
            "full_candidate_count": 15, "roi_candidate_count": direct["candidate_count"], "candidate_reduction": direct["candidate_reduction"], "localization_score": direct["localization_score"],
            "direct_prediction": "|".join(direct["predicted_parts"]), "direct_confidence": direct["prediction_confidence"], "direct_exact_match": direct["predicted_parts"] == direct["ground_truth_parts"], "direct_verifier_status": direct["verifier_status"],
            "checklist_prediction": "|".join(checklist["predicted_parts"]), "checklist_confidence": checklist["prediction_confidence"], "checklist_exact_match": checklist["predicted_parts"] == checklist["ground_truth_parts"], "checklist_verifier_status": checklist["verifier_status"],
            "checklist_uncertain_count": len(checklist["checklist_uncertain_parts"]), "annotated_result_path": annotated_by_case[case_id],
        })
    case_fields = ["case_id", "error_type", "ground_truth_parts", "full_candidate_count", "roi_candidate_count", "candidate_reduction", "localization_score", "direct_prediction", "direct_confidence", "direct_exact_match", "direct_verifier_status", "checklist_prediction", "checklist_confidence", "checklist_exact_match", "checklist_verifier_status", "checklist_uncertain_count", "annotated_result_path"]
    _write_csv(thesis / "roi_direct_vs_checklist_cases.csv", case_fields, case_table)
    _write_csv(thesis / "request_efficiency.csv", ["method", "case_id", "request_count", "latency_seconds", "input_tokens", "output_tokens", "total_tokens"], efficiency_rows)
    evolution = [
        {"stage": "Stage 1: Free-form VLM Baseline", "exact_match": .08, "at_least_one": None, "all_parts": None, "part_f1": .105263, "false_confident_0_80": .88, "candidate_reduction": None, "gt_coverage": None, "eye_ball_retained": None, "denominator": 25},
        {"stage": "Stage 2: Prompt/Candidate Constraint", "exact_match": 0, "at_least_one": .3333, "all_parts": 0, "part_f1": .2857, "false_confident_0_80": .6667, "candidate_reduction": None, "gt_coverage": None, "eye_ball_retained": None, "denominator": 3},
        {"stage": "Stage 3: ROI Candidate Reduction", "exact_match": None, "at_least_one": None, "all_parts": None, "part_f1": None, "false_confident_0_80": None, "candidate_reduction": .6444, "gt_coverage": 1.0, "eye_ball_retained": "0/3", "denominator": 3},
        {"stage": "Stage 4: ROI Direct Classification", "exact_match": metrics["roi_direct"]["exact_set_match"], "at_least_one": metrics["roi_direct"]["at_least_one_recall"], "all_parts": metrics["roi_direct"]["all_parts_recall"], "part_f1": metrics["roi_direct"]["part_f1"], "false_confident_0_80": metrics["roi_direct"]["false_confident_identity"]["0.80"], "candidate_reduction": .6444, "gt_coverage": 1.0, "eye_ball_retained": "0/3", "denominator": 3},
        {"stage": "Stage 5: ROI Checklist Verification (recovered analysis)", "exact_match": metrics["roi_checklist"]["exact_set_match"], "at_least_one": metrics["roi_checklist"]["at_least_one_recall"], "all_parts": metrics["roi_checklist"]["all_parts_recall"], "part_f1": metrics["roi_checklist"]["part_f1"], "false_confident_0_80": metrics["roi_checklist"]["false_confident_identity"]["0.80"], "candidate_reduction": .6444, "gt_coverage": 1.0, "eye_ball_retained": "0/3", "denominator": 3},
    ]
    evolution_fields = ["stage", "exact_match", "at_least_one", "all_parts", "part_f1", "false_confident_0_80", "candidate_reduction", "gt_coverage", "eye_ball_retained", "denominator"]
    _write_csv(thesis / "research_method_evolution.csv", evolution_fields, evolution)

    for case_id in manifest["cases"]:
        metadata_path, metadata = metadata_by_case_method[(case_id, "roi_checklist")]
        render_case_figure(reference_image=Path(metadata["reference_image"]), test_image=Path(metadata["test_image"]), roi_image=Path(localization_by_case[case_id]), annotated_image=Path(annotated_by_case[case_id]), output_path=figures / f"thesis_case_{case_id.replace('-', '_')}.png", title=f"ROI affected-part result: {case_id}")

    checks = {
        "exact_not_worse": metrics["roi_checklist"]["exact_set_match"] >= metrics["roi_direct"]["exact_set_match"],
        "at_least_not_worse": metrics["roi_checklist"]["at_least_one_recall"] >= metrics["roi_direct"]["at_least_one_recall"],
        "all_parts_not_worse": metrics["roi_checklist"]["all_parts_recall"] >= metrics["roi_direct"]["all_parts_recall"],
        "f1_better": metrics["roi_checklist"]["part_f1"] > metrics["roi_direct"]["part_f1"],
        "false_confident_better": metrics["roi_checklist"]["false_confident_identity"]["0.80"] < metrics["roi_direct"]["false_confident_identity"]["0.80"],
        "wrong_identity_escape_zero": metrics["roi_checklist"]["wrong_identity_escaped"] == 0,
    }
    semantic_promising = all(checks.values())
    decision = "PROMISING" if semantic_promising and schema_validity["roi_checklist"]["rate"] == 1.0 else "NO_CLEAR_IMPROVEMENT" if semantic_promising else "REGRESSION"
    decision_payload = {
        "decision": decision, "semantic_recovered_checklist_signal": "PROMISING" if semantic_promising else "NOT_PROMISING",
        "success_checks": checks, "blocking_reason": "original checklist schema-valid rate is 0/3" if semantic_promising and decision != "PROMISING" else None,
        "recommended_production_method": "ROI_CHECKLIST" if decision == "PROMISING" else "NONE",
        "final_output_strategy": "DETERMINISTIC_ANNOTATION", "gpt_image_recommendation": "DO_NOT_USE_FOR_THIS_OUTPUT",
        "phase_2b_recommendation": "BLOCK", "api_requests_during_evaluation": 0,
    }
    _write_json(evaluation / "decision.json", decision_payload)
    _write_json(evaluation / "case_forensics.json", {row["case_id"] + ":" + row["method"]: {
        "ground_truth_parts": row["ground_truth_parts"], "predicted_parts": row["predicted_parts"],
        "confidence": row["prediction_confidence"], "verifier_status": row["verifier_status"],
        "checklist_fail_parts": row["checklist_fail_parts"], "checklist_uncertain_parts": row["checklist_uncertain_parts"],
        "rule_engine_prediction": row["rule_engine_prediction"], "manual_review": row["manual_review"],
    } for row in case_rows})
    audit = json.loads((evaluation / "request_audit_summary.json").read_text(encoding="utf-8"))
    _write_json(evaluation / "final_evaluation_report.json", {
        "logical_requests": audit["logical_requests"], "physical_requests": audit["physical_requests"],
        "retry_requests": audit["retry_requests"], "successful_requests": audit["successful_requests"],
        "schema_valid_requests": audit["schema_valid_requests"], "original_schema_validity": schema_validity,
        "normalized_analysis_schema_validity": normalized_analysis_validity,
        "semantic_metrics": metrics, "strict_original_schema_metrics": strict_metrics,
        "checklist_component_metrics": component_metrics, "cases": case_table,
        "candidate_reduction_recap": {"counts": [5, 5, 6], "mean_reduction": 0.6444444444444445, "confirmed_gt_coverage": 1.0, "eye_ball_retained": "0/3"},
        "figures": {
            "confusion_matrix": str((figures / "checklist_confusion_matrix.png").resolve()),
            "method_comparison": str((figures / "method_comparison_metrics.png").resolve()),
            "annotated_results": annotated_by_case,
            "thesis_case_figures": {case_id: str((figures / f"thesis_case_{case_id.replace('-', '_')}.png").resolve()) for case_id in manifest["cases"]},
        },
        "thesis_tables": {name: str((thesis / name).resolve()) for name in (
            "roi_direct_vs_checklist_metrics.csv", "roi_direct_vs_checklist_cases.csv",
            "checklist_component_results.csv", "research_method_evolution.csv", "request_efficiency.csv",
        )},
        "decision": decision_payload, "production_prompt_changed": False, "production_schema_changed": False,
        "ground_truth_changed": False, "source_images_changed": False,
        "api_requests_during_offline_evaluation": 0, "gpt_image_requests": 0, "phase_2b_executed": False,
    })
    return {"methods": metrics, "strict_methods": strict_metrics, "component_metrics": component_metrics, "decision": decision_payload, "case_rows": case_rows, "api_requests_during_evaluation": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "analysis/affected_part_eval_ground_truth.csv")
    args = parser.parse_args(); result = evaluate_run(args.run_dir, args.labels)
    print(json.dumps({"semantic_metrics": result["methods"], "component_metrics": result["component_metrics"], "decision": result["decision"], "api_requests_during_evaluation": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
