from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_ground_truth import (
    OUTPUT_FIELDS,
    build_ground_truth,
    build_ground_truth_rows,
)
from utils.ground_truth_loader import (
    DEFAULT_GROUND_TRUTH_PATH,
    get_ground_truth_by_image_id,
    load_ground_truth,
    validate_batch_test_compatibility,
)
from utils.taxonomy import REQUIRED_GROUND_TRUTH_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_INVENTORY = (
    PROJECT_ROOT
    / "output"
    / "dataset_audit"
    / "20260722_170328"
    / "dataset_inventory.csv"
)


class FormalGroundTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_ground_truth()

    def test_formal_csv_matches_frozen_inventory(self) -> None:
        with FROZEN_INVENTORY.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            inventory_rows = list(csv.DictReader(handle))
        self.assertEqual(len(self.rows), 158)
        self.assertEqual(len(self.rows), len(inventory_rows))
        self.assertTrue(set(REQUIRED_GROUND_TRUTH_FIELDS) <= set(self.rows[0]))

    def test_ids_are_unique_stable_paths_and_files_exist(self) -> None:
        image_ids = [row["image_id"] for row in self.rows]
        image_paths = [row["image_path"] for row in self.rows]
        self.assertEqual(len(image_ids), len(set(image_ids)))
        self.assertEqual(image_ids, image_paths)
        self.assertEqual(image_paths, sorted(image_paths))
        self.assertTrue(
            all((PROJECT_ROOT / Path(path)).is_file() for path in image_paths)
        )

    def test_counts_and_logical_consistency(self) -> None:
        counts = Counter(row["error_type"] for row in self.rows)
        self.assertEqual(
            counts,
            {
                "correct": 61,
                "criticalerror": 6,
                "extra": 15,
                "missing": 36,
                "position": 12,
                "wrongpart": 28,
            },
        )
        self.assertEqual(counts["orientation"], 0)
        self.assertTrue(
            all(not row["is_error"] for row in self.rows if row["error_type"] == "correct")
        )
        self.assertTrue(
            all(row["is_error"] for row in self.rows if row["error_type"] != "correct")
        )
        critical = [
            row for row in self.rows if row["error_type"] == "criticalerror"
        ]
        self.assertEqual(len(critical), 6)
        self.assertTrue(
            all(
                row["schema_error_type"] == "criticalerror"
                and row["evaluation_scope"] == "in_scope"
                and row["filename_valid"] == "true"
                for row in critical
            )
        )
        self.assertEqual(
            sum(row["filename_valid"] == "false" for row in self.rows),
            2,
        )

    def test_source_qualified_ids_preserve_duplicate_filenames(self) -> None:
        by_name = Counter(row["image_name"] for row in self.rows)
        duplicate_names = {name for name, count in by_name.items() if count > 1}
        self.assertEqual(len(duplicate_names), 10)
        sample_name = sorted(duplicate_names)[0]
        matching = [row for row in self.rows if row["image_name"] == sample_name]
        self.assertEqual(
            {row["source_split"] for row in matching},
            {"input", "regression_subset"},
        )
        with self.assertRaisesRegex(KeyError, "Ambiguous"):
            get_ground_truth_by_image_id(sample_name, self.rows)
        self.assertEqual(
            get_ground_truth_by_image_id(matching[0]["image_id"], self.rows),
            matching[0],
        )

    def test_existing_batch_contract_is_compatible(self) -> None:
        result = validate_batch_test_compatibility(self.rows)
        self.assertTrue(result["compatible"], result["errors"])
        self.assertEqual(result["row_count"], 158)
        self.assertEqual(result["in_scope_count"], 158)
        self.assertEqual(result["out_of_scope_count"], 0)

    def test_builder_is_deterministic(self) -> None:
        first = build_ground_truth_rows(FROZEN_INVENTORY)
        second = build_ground_truth_rows(FROZEN_INVENTORY)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "ground_truth.csv"
            summary = build_ground_truth(FROZEN_INVENTORY, output)
            self.assertEqual(summary["row_count"], 158)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, OUTPUT_FIELDS)
                self.assertEqual(len(list(reader)), 158)

    def test_default_location_is_data_directory(self) -> None:
        self.assertEqual(
            DEFAULT_GROUND_TRUTH_PATH,
            PROJECT_ROOT / "data" / "ground_truth.csv",
        )


if __name__ == "__main__":
    unittest.main()
