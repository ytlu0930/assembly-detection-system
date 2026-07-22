from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_dataset import (
    audit_dataset,
    build_target_status,
    classify_duplicate_group,
    find_matching_reference,
    parse_dataset_filename,
)


class DatasetFilenameTests(unittest.TestCase):
    def test_correct_filename(self) -> None:
        result = parse_dataset_filename("model03_step01_correct-01_left_01.jpg")
        self.assertTrue(result["filename_valid"])
        self.assertFalse(result["is_error"])
        self.assertEqual(result["error_type"], "correct")

    def test_error_filename_and_mappings(self) -> None:
        cases = {
            "positionerror": "position",
            "orientationerror": "orientation",
            "missingpart": "missing",
            "extrapart": "extra",
            "wrongpart": "wrongpart",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                result = parse_dataset_filename(
                    f"model03_step03_{label}-A01_front_01.jpg"
                )
                self.assertTrue(result["filename_valid"])
                self.assertTrue(result["is_error"])
                self.assertEqual(result["error_type"], expected)

    def test_all_six_view_angles(self) -> None:
        for view in ("top", "bottom", "front", "back", "left", "right"):
            with self.subTest(view=view):
                result = parse_dataset_filename(
                    f"model03_step01_correct-01_{view}_01.jpg"
                )
                self.assertTrue(result["filename_valid"])

    def test_invalid_view_angle(self) -> None:
        result = parse_dataset_filename("model03_step01_correct-01_diagonal_01.jpg")
        self.assertFalse(result["filename_valid"])
        self.assertIn("missing_or_invalid_view_angle", result["validation_errors"])

    def test_missing_model_and_step(self) -> None:
        missing_model = parse_dataset_filename("_step01_correct-01_front_01.jpg")
        missing_step = parse_dataset_filename("model03__correct-01_front_01.jpg")
        self.assertIn("missing_or_invalid_model_id", missing_model["validation_errors"])
        self.assertIn("missing_or_invalid_step_id", missing_step["validation_errors"])


class DatasetAuditTests(unittest.TestCase):
    def _roots(self, temp: str) -> tuple[Path, Path]:
        input_root = Path(temp) / "input"
        regression = Path(temp) / "regression_subset"
        input_root.mkdir()
        regression.mkdir()
        return input_root, regression

    def test_duplicate_hash_and_expected_regression_copy(self) -> None:
        rows = [
            {"source_root": "input", "filename": "same.jpg"},
            {"source_root": "regression_subset", "filename": "same.jpg"},
        ]
        self.assertEqual(classify_duplicate_group(rows), "expected_regression_copy")
        digest = hashlib.sha256(b"same").hexdigest()
        self.assertEqual(digest, hashlib.sha256(b"same").hexdigest())

    def test_missing_correct_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_root, _ = self._roots(temp)
            parsed = parse_dataset_filename(
                "model03_step03_wrongpart-A01_right_01.jpg"
            )
            selected, candidates, pattern = find_matching_reference(parsed, input_root)
            self.assertIsNone(selected)
            self.assertEqual(candidates, [])
            self.assertIn("model03_step03_correct", pattern)

    def test_reference_rule_prefers_correct_01(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_root, _ = self._roots(temp)
            folder = input_root / "normal" / "model03_step03"
            folder.mkdir(parents=True)
            (folder / "model03_step03_correct-02_front_01.jpg").write_bytes(b"two")
            preferred = folder / "model03_step03_correct-01_front_01.jpg"
            preferred.write_bytes(b"one")
            parsed = parse_dataset_filename(
                "model03_step03_wrongpart-A01_front_01.jpg"
            )
            selected, candidates, _ = find_matching_reference(parsed, input_root)
            self.assertEqual(selected, preferred)
            self.assertEqual(len(candidates), 2)

    def test_freeze_outputs_and_does_not_modify_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_root, regression = self._roots(temp)
            normal = input_root / "normal" / "model03_step01"
            normal.mkdir(parents=True)
            correct = normal / "model03_step01_correct-01_front_01.jpg"
            correct.write_bytes(b"correct")
            error_dir = input_root / "wrongpart" / "model03_step01"
            error_dir.mkdir(parents=True)
            error = error_dir / "model03_step01_wrongpart-A01_front_01.jpg"
            error.write_bytes(b"error")
            regression_copy = regression / error.name
            regression_copy.write_bytes(b"error")
            before = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (correct, error, regression_copy)
            }
            output_dir = Path(temp) / "audit-output"
            result = audit_dataset(
                input_root, regression, output_dir=output_dir, freeze=True
            )
            after = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in before
            }
            self.assertEqual(before, after)
            self.assertTrue((output_dir / "freeze_manifest.json").exists())
            self.assertTrue((output_dir / "dataset_inventory.csv").exists())
            manifest = json.loads(
                (output_dir / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "frozen")
            self.assertEqual(manifest["total_files"], 3)
            self.assertEqual(result["summary"]["missing_reference_count"], 0)
            self.assertEqual(result["summary"]["duplicate_file_count"], 2)

    def test_summary_target_status(self) -> None:
        status = build_target_status(
            {
                "correct": 30,
                "error": 79,
                "position": 20,
                "orientation": 0,
                "missing": 21,
                "extra": 19,
                "unknown": 1,
            }
        )
        self.assertEqual(status["correct"], "met")
        self.assertEqual(status["error"], "not_met")
        self.assertEqual(status["position"], "met")
        self.assertEqual(status["orientation"], "not_met")
        self.assertEqual(status["missing"], "met")
        self.assertEqual(status["extra"], "not_met")
        self.assertEqual(status["wrongpart"], "not_applicable")
        self.assertEqual(status["unknown"], "unknown")


if __name__ == "__main__":
    unittest.main()
