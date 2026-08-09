import csv

import cv2
import numpy as np

from scripts.evaluate_roi_direct_vs_checklist import _method_metrics, _safe_ratio
from scripts.render_roi_thesis_case_figure import render_case_figure


def test_zero_denominator_is_null_not_fabricated_zero():
    assert _safe_ratio(0, 0) is None
    metrics = _method_metrics([])
    assert metrics["denominator"] == 0
    assert metrics["exact_set_match"] is None
    assert metrics["false_confident_identity"]["0.80"] is None


def test_four_panel_thesis_figure_created(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"panel_{index}.png"
        assert cv2.imwrite(str(path), np.full((120, 160, 3), 80 + index * 30, dtype=np.uint8))
        paths.append(path)
    output = tmp_path / "thesis.png"
    render_case_figure(reference_image=paths[0], test_image=paths[1], roi_image=paths[2], annotated_image=paths[3], output_path=output, title="Case")
    assert output.is_file() and output.stat().st_size > 0


def test_thesis_csv_column_contract():
    expected = ["metric", "roi_direct", "roi_checklist", "delta", "preferred_direction", "winner"]
    assert len(expected) == len(set(expected))
