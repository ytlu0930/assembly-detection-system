from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from utils.output_manager import (
    create_run_output,
    generate_run_id,
    resolve_run_output,
)


class OutputManagerTests(unittest.TestCase):
    def test_creates_standard_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = create_run_output(
                "localization",
                "phase08_bbox_selection",
                run_id="20260722_153000",
                output_root=temp,
            )
            self.assertTrue(paths.run_dir.is_dir())
            self.assertTrue(paths.images_dir.is_dir())
            self.assertEqual(paths.json_path.name, "results.json")
            self.assertEqual(paths.csv_path.name, "results.csv")
            self.assertEqual(paths.summary_path.name, "run_summary.json")

    def test_run_id_format(self) -> None:
        run_id = generate_run_id(datetime(2026, 7, 22, 15, 30, 0))
        self.assertEqual(run_id, "20260722_153000")
        self.assertRegex(run_id, r"^\d{8}_\d{6}$")

    def test_duplicate_run_id_uses_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            first = create_run_output("pipeline", "demo", "20260722_153000", temp)
            second = create_run_output("pipeline", "demo", "20260722_153000", temp)
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertEqual(second.run_id, "20260722_153000_01")
            self.assertTrue(first.run_dir.exists())

    def test_image_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = create_run_output(
                "localization",
                "selector",
                output_root=temp,
                image_subdirs=["detections", "selected_bbox"],
            )
            self.assertTrue(paths.image_subdirs["detections"].is_dir())
            self.assertTrue(paths.image_subdirs["selected_bbox"].is_dir())

    def test_sanitized_components_are_windows_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = create_run_output(
                "local:ization", "phase?08*test", output_root=temp
            )
            self.assertIsInstance(paths.run_dir, Path)
            self.assertIsNone(re.search(r'[<>:"/\\|?*]', paths.category))
            self.assertIsNone(re.search(r'[<>:"/\\|?*]', paths.experiment))

    def test_blocks_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                create_run_output("../escape", "test", output_root=temp)
            with self.assertRaises(ValueError):
                create_run_output("safe", "..\\escape", output_root=temp)

    def test_custom_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "custom"
            paths = create_run_output("vision", "annotations", output_root=root)
            self.assertTrue(paths.run_dir.is_relative_to(root.resolve()))

    def test_explicit_output_dir_is_used_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            explicit = Path(temp) / "legacy-compatible"
            paths = resolve_run_output(
                "pipeline", "localization_pipeline", output_dir=explicit
            )
            self.assertEqual(paths.run_dir, explicit.resolve())
            self.assertTrue((explicit / "images").is_dir())

    def test_category_can_contain_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = create_run_output(
                "dataset_audit",
                "dataset_freeze",
                output_root=temp,
                image_subdirs=[],
                nest_experiment=False,
            )
            self.assertEqual(paths.run_dir.parent, Path(temp).resolve() / "dataset_audit")
            self.assertFalse(paths.images_dir.exists())


if __name__ == "__main__":
    unittest.main()
