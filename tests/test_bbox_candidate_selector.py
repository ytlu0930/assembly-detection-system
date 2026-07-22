from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.bbox_candidate_selector import BBoxCandidateSelector


class BBoxCandidateSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = BBoxCandidateSelector(expected_area_range=(0.01, 0.08))

    def test_central_small_box_beats_higher_confidence_large_box(self) -> None:
        result = self.selector.select(
            [
                {"bbox": [100, 300, 900, 700], "score": 0.90, "label": "whole"},
                {"bbox": [420, 360, 580, 640], "score": 0.45, "label": "part"},
            ],
            1000,
            1000,
            target_position="center",
        )
        self.assertEqual(result["selected"]["candidate_index"], 1)

    def test_single_legal_bbox_is_selected(self) -> None:
        result = self.selector.select(
            [{"bbox": [10, 20, 110, 120], "score": 0.2}], 500, 500
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["selected"]["candidate_index"], 0)
        self.assertEqual(result["selected"]["label"], "unknown")

    def test_empty_detections_returns_no_detection(self) -> None:
        result = self.selector.select([], 100, 100)
        self.assertEqual(result["status"], "no_detection")
        self.assertIsNone(result["selected"])

    def test_nearly_full_image_box_gets_oversized_penalty(self) -> None:
        result = self.selector.select(
            [{"bbox": [1, 1, 999, 999], "score": 0.9}], 1000, 1000
        )
        self.assertEqual(result["candidates"][0]["oversized_penalty"], 1.0)

    def test_target_position_rewards_nearby_bbox(self) -> None:
        detections = [
            {"bbox": [50, 50, 150, 150], "score": 0.5},
            {"bbox": [800, 800, 900, 900], "score": 0.5},
        ]
        result = self.selector.select(
            detections, 1000, 1000, target_position="top_left"
        )
        self.assertEqual(result["selected"]["candidate_index"], 0)
        self.assertGreater(
            result["candidates"][0]["position_score"],
            result["candidates"][1]["position_score"],
        )

    def test_out_of_bounds_bbox_is_clipped(self) -> None:
        result = self.selector.select(
            [{"bbox": [-20, -10, 120, 110], "score": 0.5}], 100, 100
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["bbox"], [0.0, 0.0, 100.0, 100.0])
        self.assertTrue(candidate["bbox_was_clipped"])

    def test_invalid_bboxes_are_rejected_without_crashing(self) -> None:
        result = self.selector.select(
            [
                {"bbox": [50, 10, 20, 30], "score": 0.9},
                {"bbox": [1, 2, 3], "score": 0.8},
                {"bbox": [10, 10, 40, 40], "score": 0.1},
            ],
            100,
            100,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["selected"]["candidate_index"], 2)
        self.assertEqual(len(result["rejected_candidates"]), 2)

    def test_weight_change_can_change_selection(self) -> None:
        detections = [
            {"bbox": [100, 100, 300, 300], "score": 0.95},
            {"bbox": [450, 450, 550, 550], "score": 0.20},
        ]
        confidence_only = self.selector.select(
            detections,
            1000,
            1000,
            target_position="center",
            weights={
                "confidence": 1.0,
                "position": 0.0,
                "area": 0.0,
                "oversized": 0.0,
                "boundary": 0.0,
            },
        )
        position_only = self.selector.select(
            detections,
            1000,
            1000,
            target_position="center",
            weights={
                "confidence": 0.0,
                "position": 1.0,
                "area": 0.0,
                "oversized": 0.0,
                "boundary": 0.0,
            },
        )
        self.assertEqual(confidence_only["selected"]["candidate_index"], 0)
        self.assertEqual(position_only["selected"]["candidate_index"], 1)

    def test_any_position_does_not_prefer_center(self) -> None:
        result = self.selector.select(
            [
                {"bbox": [50, 50, 150, 150], "score": 0.6},
                {"bbox": [450, 450, 550, 550], "score": 0.5},
            ],
            1000,
            1000,
            target_position="any",
        )
        self.assertEqual(
            result["candidates"][0]["position_score"],
            result["candidates"][1]["position_score"],
        )
        self.assertEqual(result["selected"]["candidate_index"], 0)

    def test_equal_candidates_are_deterministic(self) -> None:
        detections = [
            {"bbox": [400, 400, 500, 500], "score": 0.5},
            {"bbox": [500, 500, 600, 600], "score": 0.5},
        ]
        selections = {
            self.selector.select(detections, 1000, 1000, "any")["selected"][
                "candidate_index"
            ]
            for _ in range(10)
        }
        self.assertEqual(selections, {0})

    def test_result_is_json_serializable(self) -> None:
        result = self.selector.select(
            [{"bbox": [1, 1, 20, 20], "score": 0.3}], 100, 100
        )
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
