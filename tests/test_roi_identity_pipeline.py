from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from utils.roi_identity_pipeline import ROIIdentityPipeline


LIBRARY = {
    "EYE_BALL": ["white ball with black pupil"],
    "PIN_RED_SHORT": ["short red cylinder stick"],
    "PIN_YELLOW": ["yellow pin with flat head"],
}
EXPECTED = {
    "expected_parts": [
        {"part_id": "EYE_BALL", "color": "WHITE", "position": "TOP"},
        {"part_id": "PIN_RED_SHORT", "color": "RED", "position": "LEFT"},
        {"part_id": "PIN_YELLOW", "color": "YELLOW", "position": "LEFT"},
    ]
}


def _write(path: Path, *, red: bool = False, yellow: bool = False) -> None:
    image = np.full((300, 300, 3), 235, dtype=np.uint8)
    # Stable context components make the synthetic assembly extent larger than
    # the target ROI, mirroring a small part on a larger real construction.
    cv2.rectangle(image, (200, 35), (250, 75), (220, 80, 20), -1)
    cv2.rectangle(image, (200, 225), (250, 265), (220, 80, 20), -1)
    if red:
        cv2.rectangle(image, (35, 120), (105, 155), (0, 0, 230), -1)
    if yellow:
        cv2.rectangle(image, (35, 120), (105, 155), (0, 225, 225), -1)
    assert cv2.imwrite(str(path), image)


def test_missing_reference_only_roi_reduces_candidates(tmp_path):
    reference, test = tmp_path / "reference.png", tmp_path / "test.png"
    _write(reference, red=True)
    _write(test)
    result = ROIIdentityPipeline(working_size=300).run(
        test_image=test,
        reference_image=reference,
        model_id="model03",
        step_id="step03",
        view_angle="top",
        error_type="missingpart",
        expected_state=EXPECTED,
        part_library=LIBRARY,
        output_dir=tmp_path / "output",
    )
    assert result["localization_status"] == "success"
    assert result["candidate_part_ids"] == ["PIN_RED_SHORT"]
    assert result["candidate_count"] < result["full_candidate_count"]
    assert all(item["difference_relation"] == "reference_only" for item in result["roi_evidence"])
    assert result["human_ground_truth_used"] is False


def test_wrongpart_preserves_paired_swap_evidence(tmp_path):
    reference, test = tmp_path / "reference.png", tmp_path / "test.png"
    _write(reference, red=True)
    _write(test, yellow=True)
    result = ROIIdentityPipeline(working_size=300).run(
        test_image=test,
        reference_image=reference,
        model_id="model03",
        step_id="step03",
        view_angle="top",
        error_type="wrongpart",
        expected_state=EXPECTED,
        part_library=LIBRARY,
        output_dir=tmp_path / "output",
    )
    assert result["supports_paired_roi"] is True
    assert {"PIN_RED_SHORT", "PIN_YELLOW"}.issubset(result["candidate_part_ids"])
    assert {item["difference_relation"] for item in result["roi_evidence"]} == {"reference_only", "test_only"}
    assert result["test_roi"] and result["reference_roi"]


def test_blank_pair_fails_closed(tmp_path):
    reference, test = tmp_path / "reference.png", tmp_path / "test.png"
    _write(reference)
    _write(test)
    result = ROIIdentityPipeline(working_size=300).run(
        test_image=test,
        reference_image=reference,
        model_id="model03",
        step_id="step03",
        view_angle="front",
        error_type="missingpart",
        expected_state=EXPECTED,
        part_library=LIBRARY,
    )
    assert result["localization_status"] == "insufficient_evidence"
    assert result["candidate_part_ids"] == []
    assert result["bbox"] == []
    assert result["requires_manual_review"] is True


def test_frozen_poc_preserves_confirmed_gt_for_offline_evaluation():
    artifact = Path(__file__).parents[1] / "analysis" / "roi_identity_poc" / "candidate_reduction.csv"
    with artifact.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["case_id"] for row in rows} == {
        "missingpart-A01", "missingpart-B01", "wrongpart-B01",
    }
    assert all(row["gt_coverage"].lower() == "true" for row in rows)
    assert all(int(row["roi_candidate_count"]) < int(row["full_candidate_count"]) for row in rows)
