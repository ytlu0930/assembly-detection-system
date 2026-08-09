"""Run the offline localization-guided ROI identity PoC for the three targeted cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.grounding_detector import GroundingDetector
from utils.roi_identity_pipeline import ROIIdentityPipeline

CASES = (
    ("missingpart-A01", "missingpart", "missingpart"),
    ("missingpart-B01", "missingpart", "missingpart"),
    ("wrongpart-B01", "wrongpart", "wrongpart"),
)
VIEW_ORDER = ("front", "left", "right", "top", "back", "bottom")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ground_truth() -> dict[str, list[str]]:
    path = PROJECT_ROOT / "analysis/affected_part_eval_ground_truth.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[str]] = {}
    for case_id, _, _ in CASES:
        row = next(item for item in rows if item["case_id"] == case_id and item["view_angle"] == "front" and item["review_status"] == "confirmed")
        result[case_id] = sorted(item for item in row["ground_truth_part_ids"].split("|") if item)
    return result


def run_poc(analysis_dir: Path, output_dir: Path, *, enable_dino: bool) -> dict[str, Any]:
    expected = PROJECT_ROOT / "ground_truth/model03/step03.json"
    library = PROJECT_ROOT / "config/part_library.json"
    detector = None
    if enable_dino:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        detector = GroundingDetector(device="cpu")
    basic_pipeline = ROIIdentityPipeline(detector=None)
    dino_pipeline = ROIIdentityPipeline(detector=detector) if detector is not None else basic_pipeline
    inventory_rows: list[dict[str, Any]] = []
    packages: dict[str, Any] = {}

    for case_id, error_type, folder in CASES:
        view_results = []
        for view in VIEW_ORDER:
            test = PROJECT_ROOT / f"input/{folder}/model03_step03/model03_step03_{case_id}_{view}_01.jpg"
            reference = PROJECT_ROOT / f"input/normal/model03_step03/model03_step03_correct-01_{view}_01.jpg"
            if not test.exists() or not reference.exists():
                continue
            pipeline = dino_pipeline if view == "front" else basic_pipeline
            result = pipeline.run(
                test_image=test, reference_image=reference,
                model_id="model03", step_id="step03", view_angle=view,
                error_type=error_type, expected_state=expected, part_library=library,
                output_dir=output_dir / case_id / view,
            )
            view_results.append(result)
            inventory_rows.append({
                "case_id": case_id, "view": view, "error_type": error_type,
                "full_candidate_count": result["full_candidate_count"],
                "view_candidate_count": result["candidate_count"],
                "view_candidate_ids": "|".join(result["candidate_part_ids"]),
                "localization_score": result["localization_score"],
                "localization_status": result["localization_status"],
                "bbox": json.dumps(result["bbox"], ensure_ascii=False),
                "manual_review": result["requires_manual_review"],
                "dino_status": result["dino_corroboration"]["status"],
                "human_ground_truth_used": result["human_ground_truth_used"],
            })
        score_by_candidate: dict[str, list[float]] = defaultdict(list)
        evidence_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in view_results:
            for item in result["candidate_evidence"]:
                score_by_candidate[item["part_id"]].append(float(item["score"]))
                evidence_by_candidate[item["part_id"]].append({"view": result["view_angle"], **item})
        ranked = sorted(
            score_by_candidate,
            key=lambda part_id: (-sum(score_by_candidate[part_id]), -len(score_by_candidate[part_id]), part_id),
        )
        maximum = 6 if error_type == "wrongpart" else 5
        candidates = ranked[:maximum]
        identity_rois = []
        for part_id in candidates:
            supporting = sorted(evidence_by_candidate[part_id], key=lambda item: (-float(item["score"]), item["view"]))
            roi = next((item for item in supporting if item.get("roi_evidence")), None)
            if roi is not None:
                identity_rois.append({
                    "part_id": part_id, "view": roi["view"], "score": roi["score"],
                    "roi_evidence": roi["roi_evidence"],
                })
        front = next((item for item in view_results if item["view_angle"] == "front"), None)
        selected = front if front and front["localization_status"] == "success" else max(view_results, key=lambda item: item["localization_score"])
        package = {
            "case_id": case_id, "model_id": "model03", "step_id": "step03",
            "primary_view": selected["view_angle"], "available_views": [item["view_angle"] for item in view_results],
            "error_type": error_type, "full_candidate_count": 15,
            "candidate_part_ids": candidates, "candidate_count": len(candidates),
            "reduction_ratio": 1.0 - len(candidates) / 15.0,
            "localization_score": selected["localization_score"],
            "localization_status": "success" if candidates and selected["localization_status"] == "success" else "insufficient_evidence",
            "primary_bbox": selected["bbox"],
            "primary_test_roi": selected["test_roi"], "primary_reference_roi": selected["reference_roi"],
            "identity_rois": identity_rois,
            "supports_paired_swap_roi": error_type == "wrongpart" and any(item["test_roi"] for item in view_results) and any(item["reference_roi"] for item in view_results),
            "candidate_evidence": {part_id: evidence_by_candidate[part_id] for part_id in candidates},
            "view_results": view_results,
            "requires_manual_review": True,
            "human_ground_truth_used": False,
        }
        packages[case_id] = package
        _write_json(analysis_dir / "packages" / f"{case_id}.json", package)

    _write_csv(
        analysis_dir / "roi_inventory.csv",
        ["case_id", "view", "error_type", "full_candidate_count", "view_candidate_count", "view_candidate_ids",
         "localization_score", "localization_status", "bbox", "manual_review", "dino_status", "human_ground_truth_used"],
        inventory_rows,
    )

    ground_truth = _ground_truth()  # Evaluation-only join after every inference package is frozen.
    case_rows = []
    for case_id, package in packages.items():
        gt = ground_truth[case_id]
        coverage = set(gt).issubset(package["candidate_part_ids"])
        case_rows.append({
            "case_id": case_id, "view": package["primary_view"], "error_type": package["error_type"],
            "full_candidate_count": package["full_candidate_count"], "roi_candidate_count": package["candidate_count"],
            "roi_candidate_ids": "|".join(package["candidate_part_ids"]),
            "reduction_ratio": package["reduction_ratio"], "confirmed_gt_parts": "|".join(gt),
            "gt_coverage": coverage, "localization_score": package["localization_score"],
            "localization_status": package["localization_status"],
            "bbox": json.dumps(package["primary_bbox"], ensure_ascii=False),
            "manual_review": package["requires_manual_review"],
            "swap_pair_coverage": coverage if package["error_type"] == "wrongpart" else "not_applicable",
        })
    fields = [
        "case_id", "view", "error_type", "full_candidate_count", "roi_candidate_count", "roi_candidate_ids",
        "reduction_ratio", "confirmed_gt_parts", "gt_coverage", "localization_score", "localization_status",
        "bbox", "manual_review", "swap_pair_coverage",
    ]
    _write_csv(analysis_dir / "candidate_reduction.csv", fields, case_rows)
    _write_csv(analysis_dir / "case_summary.csv", fields, case_rows)
    summary = {
        "case_count": len(case_rows),
        "mean_candidate_reduction": sum(float(row["reduction_ratio"]) for row in case_rows) / len(case_rows),
        "confirmed_gt_coverage": sum(bool(row["gt_coverage"]) for row in case_rows) / len(case_rows),
        "localization_failure_rate": sum(row["localization_status"] != "success" for row in case_rows) / len(case_rows),
        "api_requests": 0,
        "ground_truth_usage": "evaluation_only_after_package_freeze",
    }
    _write_json(analysis_dir / "poc_summary.json", summary)
    return {"summary": summary, "cases": case_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT / "analysis/roi_identity_poc")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/roi_identity_poc")
    parser.add_argument("--enable-dino", action="store_true")
    args = parser.parse_args()
    result = run_poc(args.analysis_dir, args.output_dir, enable_dino=args.enable_dino)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
