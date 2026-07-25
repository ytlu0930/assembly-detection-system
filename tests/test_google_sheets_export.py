from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.export_google_sheets_csv import (
    BATCH_RESULTS_FIELDS,
    FAILURE_ANALYSIS_FIELDS,
    GROUND_TRUTH_FIELDS,
    OUTPUT_FILENAMES,
    STEP_COVERAGE_FIELDS,
    SUMMARY_FIELDS,
    export_google_sheets_csv,
)
from utils.ground_truth_loader import DEFAULT_GROUND_TRUTH_PATH, load_ground_truth


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_GROUND_TRUTH = PROJECT_ROOT / "ground_truth.csv"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


class GoogleSheetsExportTests(unittest.TestCase):
    def test_export_uses_formal_ground_truth_and_preserves_source_hash(self) -> None:
        before = hashlib.sha256(DEFAULT_GROUND_TRUTH_PATH.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            summary = export_google_sheets_csv(output_dir=temp)
        after = hashlib.sha256(DEFAULT_GROUND_TRUTH_PATH.read_bytes()).hexdigest()
        self.assertEqual(summary["source"], str(DEFAULT_GROUND_TRUTH_PATH.resolve()))
        self.assertEqual(summary["row_count"], 158)
        self.assertEqual(before, after)
        self.assertEqual(summary["source_sha256"], before)

    def test_legacy_ground_truth_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Legacy"):
                export_google_sheets_csv(
                    ground_truth_path=LEGACY_GROUND_TRUTH,
                    output_dir=temp,
                )

    def test_ground_truth_export_schema_counts_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export_google_sheets_csv(output_dir=output)
            fields, rows = read_csv(output / "01_ground_truth.csv")
        self.assertEqual(fields, GROUND_TRUTH_FIELDS)
        self.assertEqual(len(rows), 158)
        self.assertEqual(len({row["image_id"] for row in rows}), 158)
        self.assertTrue(
            all(Path(row["image_id"]).name == row["file_name"] for row in rows)
        )
        self.assertEqual(
            {row["source_split"] for row in rows},
            {"input", "regression_subset"},
        )
        self.assertEqual(
            Counter(row["error_type"] for row in rows),
            {
                "correct": 61,
                "criticalerror": 6,
                "extra": 15,
                "missing": 36,
                "position": 12,
                "wrongpart": 28,
            },
        )
        self.assertEqual(
            sum(row["error_type"] == "orientation" for row in rows),
            0,
        )
        self.assertTrue(
            all(
                row["is_error"] == ("false" if row["error_type"] == "correct" else "true")
                for row in rows
            )
        )

    def test_no_nan_none_or_container_literals(self) -> None:
        forbidden = {"nan", "none"}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export_google_sheets_csv(output_dir=output)
            for name in OUTPUT_FILENAMES:
                fields, rows = read_csv(output / name)
                self.assertTrue(fields)
                for row in rows:
                    for value in row.values():
                        self.assertNotIn(value.strip().lower(), forbidden)
                        self.assertFalse(value.startswith("[") or value.startswith("{"))

    def test_summary_and_step_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export_google_sheets_csv(output_dir=output)
            summary_fields, summary_rows = read_csv(
                output / "02_dataset_summary.csv"
            )
            coverage_fields, coverage_rows = read_csv(
                output / "03_step_coverage.csv"
            )
        self.assertEqual(summary_fields, SUMMARY_FIELDS)
        self.assertEqual(coverage_fields, STEP_COVERAGE_FIELDS)
        summary = {row["category"]: row for row in summary_rows}
        self.assertEqual(summary["correct"]["target_status"], "achieved")
        self.assertEqual(summary["all_error"]["target_status"], "achieved")
        self.assertEqual(summary["position"]["target_status"], "not_achieved")
        self.assertEqual(summary["extra"]["target_status"], "not_achieved")
        self.assertEqual(summary["orientation"]["target_status"], "out_of_scope")
        coverage = {row["error_type"]: row for row in coverage_rows}
        self.assertEqual(coverage["orientation"]["image_count"], "0")
        self.assertEqual(coverage["orientation"]["target_status"], "out_of_scope")
        self.assertEqual(coverage["wrongpart"]["target_status"], "not_applicable")
        self.assertEqual(
            coverage["criticalerror"]["target_status"],
            "not_applicable",
        )
        self.assertFalse(
            any(row["covered_steps"].startswith("[") for row in coverage_rows)
        )

    def test_templates_are_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            export_google_sheets_csv(output_dir=output)
            batch_fields, batch_rows = read_csv(
                output / "04_batch_results_template.csv"
            )
            failure_fields, failure_rows = read_csv(
                output / "05_failure_analysis_template.csv"
            )
        self.assertEqual(batch_fields, BATCH_RESULTS_FIELDS)
        self.assertEqual(failure_fields, FAILURE_ANALYSIS_FIELDS)
        self.assertEqual(batch_rows, [])
        self.assertEqual(failure_rows, [])

    def test_export_is_deterministic_and_check_only_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first)
            second_dir = Path(second)
            export_google_sheets_csv(output_dir=first_dir)
            export_google_sheets_csv(output_dir=second_dir)
            first_bytes = {
                name: (first_dir / name).read_bytes() for name in OUTPUT_FILENAMES
            }
            second_bytes = {
                name: (second_dir / name).read_bytes() for name in OUTPUT_FILENAMES
            }
            self.assertEqual(first_bytes, second_bytes)
            before = {
                name: hashlib.sha256(content).hexdigest()
                for name, content in first_bytes.items()
            }
            result = export_google_sheets_csv(
                output_dir=first_dir,
                check_only=True,
            )
            after = {
                name: hashlib.sha256((first_dir / name).read_bytes()).hexdigest()
                for name in OUTPUT_FILENAMES
            }
            self.assertTrue(result["check_only"])
            self.assertEqual(before, after)

    def test_source_images_still_match_formal_sha256(self) -> None:
        rows = load_ground_truth()
        mismatches = []
        for row in rows:
            image = PROJECT_ROOT / Path(row["image_path"])
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                mismatches.append(row["image_id"])
        self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
