"""Normalize, verify, evaluate, and report completed affected-part A/B results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_affected_part_identity import _read, evaluate, split_ids
from scripts.evaluate_affected_part_prompt_ab import LABELS, METRICS, VARIANTS
from utils.affected_part_identity_verifier import AffectedPartIdentityVerifier

PREDICTION_FIELDS = (
    "variant", "case_id", "image_id", "view_angle", "error_type", "gt_status",
    "ground_truth_part_ids", "predicted_part_ids", "predicted_confidence",
    "prediction_count", "has_unknown", "is_composite_prediction",
    "verifier_status", "verified_part_id", "verified_part_ids",
    "requires_manual_review", "candidate_constraint_status", "candidate_constraint_violations",
    "latency_seconds", "source_result",
)


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _english_prompt(part_id: str, library: dict[str, Any]) -> str:
    aliases = [str(value) for value in library.get(part_id, []) if str(value).isascii()]
    return max(aliases, key=lambda value: (len(value.split()), len(value))) if aliases else part_id.replace("_", " ").lower()


def quarantine_candidate_violation(verification: dict[str, Any], constraint_status: str) -> dict[str, Any]:
    if constraint_status != "violation":
        return verification
    return {
        **verification,
        "verifier_status": "unresolved",
        "verified_part_id": None,
        "requires_manual_review": True,
    }


class OfflineProductionVerifier:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.verifier = AffectedPartIdentityVerifier()
        self.library = json.loads((PROJECT_ROOT / "config/part_library.json").read_text(encoding="utf-8"))
        self.pipeline: Any | None = None
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _evidence(self, raw: dict[str, Any], part_id: str, error_type: str, expected: dict[str, Any]) -> dict[str, Any]:
        if error_type not in {"missingpart", "extrapart"} or part_id not in self.library:
            return {}
        key = (raw["case_id"], part_id, error_type)
        if key in self.cache:
            return self.cache[key]
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if self.pipeline is None:
            from utils.localization_pipeline import LocalizationPipeline
            self.pipeline = LocalizationPipeline()
        expected_part = next((part for part in expected.get("expected_parts", []) if part.get("part_id") == part_id), {})
        target_position = {"LEFT": "left", "RIGHT": "right", "TOP": "top", "BOTTOM": "bottom"}.get(str(expected_part.get("position") or "").upper(), "center")
        prompt = _english_prompt(part_id, self.library)
        pair = {}
        for role, image_key in (("reference_localization", "reference_image"), ("test_localization", "test_image")):
            image = PROJECT_ROOT / raw[image_key]
            pair[role] = self.pipeline.localize(
                image_path=str(image), text_prompt=prompt, box_threshold=0.15,
                text_threshold=0.10, target_position=target_position, max_detections=10,
                output_dir=str(self.output_dir / raw["case_id"] / part_id / role),
            )
        self.cache[key] = pair
        return pair

    def verify(self, raw: dict[str, Any]) -> dict[str, Any]:
        parsed = raw.get("parsed_output") or {}
        parts = [] if parsed.get("overall_error_type") == "correct" else [part for part in parsed.get("detected_parts", []) if isinstance(part, dict) and part.get("error_type") != "correct"]
        if not parts:
            return {"verifier_status": "not_applicable", "verified_part_id": "", "requires_manual_review": False, "details": []}
        expected_path = PROJECT_ROOT / f"ground_truth/{parsed.get('model_id')}/{parsed.get('step_id')}.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        details = []
        for part in parts:
            part_id = str(part.get("part_id") or "UNKNOWN").upper()
            error_type = str(part.get("error_type") or parsed.get("overall_error_type") or "uncertain")
            evidence = self._evidence(raw, part_id, error_type, expected)
            details.append(self.verifier.verify(
                error_report={"part_id": part_id, "error_type": error_type, "confidence": part.get("confidence")},
                test_image_metadata={"image_name": raw["image_id"], "relative_path": raw["test_image"], "view_angle": raw["view_angle"]},
                reference_image_metadata={"relative_path": raw["reference_image"]},
                expected_state=expected, localization_evidence=evidence, part_library=self.library,
            ).to_dict())
        statuses = {item["identity_status"] for item in details}
        status = next((value for value in ("conflict", "unresolved", "uncertain") if value in statuses), "verified")
        verified = sorted({str(item["verified_part_id"]) for item in details if item.get("verified_part_id")})
        return {"verifier_status": status, "verified_part_id": "|".join(verified), "requires_manual_review": status != "verified", "details": details}


def normalize(raw_dir: Path, ground_truth: Path, review_csv: Path, results_dir: Path) -> list[dict[str, Any]]:
    gt_rows = _read(ground_truth)
    gt = {row["image_id"]: row for row in gt_rows if row.get("review_status") == "confirmed"}
    with review_csv.open(encoding="utf-8-sig", newline="") as handle:
        review = {row["image_id"]: row for row in csv.DictReader(handle)}
    verifier = OfflineProductionVerifier(results_dir / "verifier_localization")
    rows = []
    for source in sorted(raw_dir.glob("*.json")):
        raw = json.loads(source.read_text(encoding="utf-8"))
        if raw.get("status") != "success":
            continue
        raw["test_image"] = next(
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in PROJECT_ROOT.glob(f"input/*/*/{raw['image_id']}")
        )
        raw["reference_image"] = f"input/normal/{raw['parsed_output']['model_id']}_{raw['parsed_output']['step_id']}/{raw['parsed_output']['model_id']}_{raw['parsed_output']['step_id']}_correct-01_{raw['view_angle']}_01.jpg"
        verification = verifier.verify(raw)
        constraint_status = raw.get("candidate_constraint_status")
        if constraint_status not in {"valid", "violation", "not_applicable"}:
            from scripts.run_affected_part_prompt_ab import enforce_candidate_constraint
            constraint = enforce_candidate_constraint(raw["variant"], raw.get("candidate_part_ids") or [], raw.get("predicted_part_ids") or [])
            constraint_status = constraint["candidate_constraint_status"]
            raw.update(constraint)
        verification = quarantine_candidate_violation(verification, constraint_status)
        frozen = gt.get(raw["image_id"])
        reviewed = review.get(raw["image_id"], {})
        gt_status = "confirmed" if frozen else reviewed.get("annotation_status", "not_in_frozen_evaluation")
        ids = [str(value) for value in raw.get("predicted_part_ids") or []]
        confidences = [float(value) for value in raw.get("predicted_confidence") or []]
        row = {
            "variant": raw["variant"], "case_id": raw["case_id"], "image_id": raw["image_id"],
            "view_angle": raw["view_angle"], "error_type": raw.get("predicted_error_type") or "uncertain",
            "gt_status": gt_status, "ground_truth_part_ids": frozen.get("ground_truth_part_ids", "") if frozen else "",
            "predicted_part_ids": "|".join(ids), "predicted_confidence": "|".join(f"{value:.6f}" for value in confidences),
            "prediction_count": len(ids), "has_unknown": str(bool(raw.get("unknown_flag"))).lower(),
            "is_composite_prediction": str(len(ids) > 1).lower(),
            "verifier_status": verification["verifier_status"], "verified_part_id": verification["verified_part_id"],
            "verified_part_ids": verification["verified_part_id"],
            "requires_manual_review": str(verification["requires_manual_review"]).lower(),
            "candidate_constraint_status": constraint_status,
            "candidate_constraint_violations": "|".join(raw.get("candidate_constraint_violations") or []),
            "latency_seconds": raw.get("latency_seconds"), "source_result": source.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        }
        rows.append(row)
        raw["verifier"] = verification
        source.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(results_dir / "affected_part_prompt_ab_predictions.csv", PREDICTION_FIELDS, rows)
    predictions_dir = results_dir / "predictions"
    for variant in VARIANTS:
        _write_csv(predictions_dir / f"{variant}.csv", PREDICTION_FIELDS, [row for row in rows if row["variant"] == variant])
    return rows


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = [(name, "summary", name) for name in METRICS]
    definitions += [(f"False-confident Identity @{threshold}", f"false:{threshold}", "false_confident_identity_rate") for threshold in ("0.70", "0.80", "0.90")]
    definitions += [("False-confident Case @0.80", "false:0.80", "false_confident_case_rate")]
    definitions += [(f"Verifier {name.replace('_rate','').title()}", "verifier", name) for name in ("acceptance_rate", "conflict_rate", "unresolved_rate", "wrong_identity_escaped_verifier_count")]
    rows = []
    for label, section, key in definitions:
        values = {}
        for variant in VARIANTS:
            result = metrics[variant]
            if section == "summary": value = result["summary"].get(key)
            elif section.startswith("false:"): value = result["false_confident"][section.split(":")[1]].get(key)
            else: value = result["verifier"].get(key)
            values[variant] = value
        base = values["baseline"]
        rows.append({
            "Metric": label, "Baseline": base, "Reference": values["reference"],
            "Reference+Candidate": values["reference_candidate"],
            "Reference_Delta": None if base is None or values["reference"] is None else values["reference"] - base,
            "ReferenceCandidate_Delta": None if base is None or values["reference_candidate"] is None else values["reference_candidate"] - base,
        })
    return rows


def _choose(metrics: dict[str, Any]) -> dict[str, Any]:
    base = metrics["baseline"]
    candidates = []
    for variant in ("reference", "reference_candidate"):
        item = metrics[variant]
        if (
            item["summary"]["exact_set_match_accuracy"] is None
            or item["false_confident"]["0.80"]["false_confident_identity_rate"] is None
            or item["verifier"]["wrong_identity_escaped_verifier_count"] is None
            or item["summary"]["evaluated_case_count"] != base["summary"]["evaluated_case_count"]
        ):
            continue
        safe = (
            item["summary"]["exact_set_match_accuracy"] >= base["summary"]["exact_set_match_accuracy"]
            and item["summary"]["at_least_one_part_recall"] > base["summary"]["at_least_one_part_recall"]
            and item["false_confident"]["0.80"]["false_confident_identity_rate"] <= base["false_confident"]["0.80"]["false_confident_identity_rate"]
            and item["verifier"]["wrong_identity_escaped_verifier_count"] <= base["verifier"]["wrong_identity_escaped_verifier_count"]
        )
        if safe:
            candidates.append(variant)
    if not candidates:
        return {"recommended_variant": None, "decision": "NO_CLEAR_IMPROVEMENT", "reasons": ["No experimental variant met all safety gates on the confirmed subset."], "safety_tradeoffs": ["Keep production prompt unchanged."]}
    winner = max(candidates, key=lambda value: (metrics[value]["summary"]["exact_set_match_accuracy"], -metrics[value]["false_confident"]["0.80"]["false_confident_identity_rate"], metrics[value]["summary"]["at_least_one_part_recall"]))
    return {"recommended_variant": winner, "decision": "PROMISING_NEEDS_MORE_DATA", "reasons": ["Exact Match did not decline, false-confident identity did not increase, and verifier escapes did not increase."], "safety_tradeoffs": ["Confirmed execution denominator is only three front-view error cases; correct-control FP is N/A."]}


def finalize(results_dir: Path, ground_truth: Path, review_csv: Path) -> dict[str, Any]:
    rows = normalize(results_dir / "raw", ground_truth, review_csv, results_dir)
    gt = _read(ground_truth)
    metrics = {variant: evaluate(gt, [row for row in rows if row["variant"] == variant]) for variant in VARIANTS}
    (results_dir / "affected_part_prompt_ab_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison = _metric_rows(metrics)
    _write_csv(results_dir / "prompt_ab_metrics_comparison.csv", list(comparison[0]), comparison)
    decision = _choose(metrics)
    (results_dir / "variant_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raw_results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((results_dir / "raw").glob("*.json"))]
    row_index = {(row["variant"], row["case_id"]): row for row in rows}
    a01 = []
    for raw in raw_results:
        if raw["case_id"] == "missingpart-A01":
            row = row_index.get((raw["variant"], raw["case_id"]))
            expected = {"PIN_RED_SHORT"}
            predicted = set(split_ids(row["predicted_part_ids"])) if row else set()
            a01.append({"variant": raw["variant"], "status": raw["status"], "predicted_part_ids": row["predicted_part_ids"] if row else "", "confidence": row["predicted_confidence"] if row else "", "identity_correct": expected == predicted if row else None, "unknown": row["has_unknown"] if row else None, "verifier_status": row["verifier_status"] if row else "not_evaluated", "verified_part_id": row["verified_part_id"] if row else ""})
    _write_csv(results_dir / "missingpart_A01_comparison.csv", list(a01[0]), a01)
    qualitative = []
    for raw in raw_results:
        if raw["case_id"] not in {"extrapart-A01", "wrongpart-A01"}:
            continue
        row = row_index.get((raw["variant"], raw["case_id"]))
        qualitative.append({"variant": raw["variant"], "case_id": raw["case_id"], "status": raw["status"], "prediction": row["predicted_part_ids"] if row else "", "confidence": row["predicted_confidence"] if row else "", "candidate_list": raw.get("candidate_part_ids", []), "verifier_result": row["verifier_status"] if row else "not_evaluated", "EXCLUDED_FROM_PRIMARY_METRICS": True})
    _write_csv(results_dir / "qualitative_unconfirmed_cases.csv", list(qualitative[0]), qualitative)
    executive = []
    for variant in VARIANTS:
        m = metrics[variant]
        executive.append({"Variant": LABELS[variant], "Exact Match": m["summary"]["exact_set_match_accuracy"], "At-least-one Recall": m["summary"]["at_least_one_part_recall"], "Part F1": m["summary"]["part_level_f1"], "False-confident @0.80": m["false_confident"]["0.80"]["false_confident_identity_rate"], "Unknown Rate": m["summary"]["unknown_part_rate"], "Wrong Escaped Verifier": m["verifier"]["wrong_identity_escaped_verifier_count"], "Decision": decision["decision"] if decision["recommended_variant"] == variant else "NOT_SELECTED"})
    _write_csv(results_dir / "prompt_ab_executive_summary.csv", list(executive[0]), executive)
    return {"rows": rows, "metrics": metrics, "comparison": comparison, "decision": decision, "a01": a01, "qualitative": qualitative}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "analysis/vision_prompt_ab/results")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "analysis/affected_part_eval_ground_truth.csv")
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "analysis/affected_parts_review_template.csv")
    args = parser.parse_args()
    payload = finalize(args.results_dir, args.ground_truth, args.review_csv)
    print(json.dumps({"prediction_rows": len(payload["rows"]), "decision": payload["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
