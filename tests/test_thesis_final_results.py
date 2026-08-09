import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/thesis_final_results"


def test_all_thesis_figures_are_readable_and_validated():
    rows = list(csv.DictReader((OUT / "figure_validation.csv").open(encoding="utf-8")))
    assert len(rows) == 23
    assert all(row["status"] == "PASS" for row in rows)
    for row in rows:
        path = OUT / row["file"]
        with Image.open(path) as image:
            image.verify()
        assert int(row["width"]) > 0 and int(row["height"]) > 0 and path.stat().st_size > 0


def test_required_thesis_tables_have_stable_csv_contracts():
    names = [
        "01_baseline_metrics.csv", "02_prompt_candidate_metrics.csv",
        "03_roi_candidate_reduction.csv", "04_direct_vs_checklist_metrics.csv",
        "05_checklist_component_results.csv", "06_case_results.csv",
        "07_research_method_evolution.csv", "08_request_efficiency.csv",
        "master_experiment_summary.csv",
    ]
    for name in names:
        text = (OUT / "thesis_tables" / name).read_text(encoding="utf-8")
        assert text.splitlines()[0]
        assert "NaN" not in text and "None" not in text


def test_manifest_covers_every_consolidated_artifact_and_hashes_match():
    manifest = json.loads((OUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    by_path = {item["file_path"]: item for item in manifest}
    expected = {
        path.relative_to(OUT).as_posix()
        for path in OUT.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    assert set(by_path) == expected
    assert all(item["source_artifacts"] for item in manifest)


def test_master_summary_preserves_stage_denominator_caveat():
    rows = list(csv.DictReader((OUT / "thesis_tables/master_experiment_summary.csv").open(encoding="utf-8")))
    assert [row["sample_size"] for row in rows] == ["25", "3", "3", "3", "3"]
    checklist = rows[-1]
    assert checklist["method"] == "ROI Checklist (normalized)"
    assert "original schema validity 0/3" in checklist["notes"]
