from __future__ import annotations

import unittest

from utils.taxonomy import (
    is_supported_for_evaluation,
    normalize_error_type,
    schema_error_type,
    validate_ground_truth_row,
)


class TaxonomyTests(unittest.TestCase):
    def test_raw_labels_normalize_to_formal_taxonomy(self) -> None:
        cases = {
            "correct": "correct",
            "positionerror": "position",
            "missingpart": "missing",
            "extrapart": "extra",
            "wrongpart": "wrongpart",
            "criticalerror": "criticalerror",
            "orientationerror": "orientation",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_error_type(raw), expected)

    def test_schema_and_evaluation_scope_decisions(self) -> None:
        self.assertEqual(schema_error_type("criticalerror"), "criticalerror")
        self.assertTrue(is_supported_for_evaluation("criticalerror"))
        self.assertIsNone(schema_error_type("orientation"))
        self.assertFalse(is_supported_for_evaluation("orientation"))

    def test_logical_validation(self) -> None:
        base = {
            "image_id": "input/normal/example.jpg",
            "image_path": "input/normal/example.jpg",
            "model_id": "model01",
            "step_id": "step01",
            "view_angle": "front",
            "is_error": "false",
            "error_type": "correct",
            "schema_error_type": "correct",
            "error_detail": "",
            "evaluation_scope": "in_scope",
            "source_split": "input",
        }
        self.assertEqual(validate_ground_truth_row(base), [])

        contradictory = dict(base, is_error="true")
        self.assertIn(
            "correct_must_not_be_error",
            validate_ground_truth_row(contradictory),
        )

        wrong_scope = dict(
            base,
            is_error="true",
            error_type="orientation",
            schema_error_type="",
        )
        self.assertIn(
            "evaluation_scope_mismatch",
            validate_ground_truth_row(wrong_scope),
        )

    def test_unknown_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_error_type("unknown-new-label")


if __name__ == "__main__":
    unittest.main()
