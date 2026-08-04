"""Replay representative July-1 results through the mock full pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.integration_pipeline import PROJECT_ROOT, run_full_pipeline
from utils.ui_pipeline_adapter import run_analysis_for_ui

CASES = ["missingpart-A01", "missingpart-B01", "wrongpart-A01", "wrongpart-B01"]


def no_detection_localizer(**kwargs):
    return {"status": "no_detection", "selected_bbox": None, "error_message": None}


def latest(case: str) -> Path:
    files = sorted((PROJECT_ROOT / "logs" / "current_parsed_json").glob(f"*{case}*20260701*.json"))
    if not files:
        raise FileNotFoundError(case)
    front = [path for path in files if "_front_01_" in path.name]
    return (front or files)[-1]


def main() -> int:
    output = PROJECT_ROOT / "output" / "offline_integration_smoke"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in CASES:
        source = latest(case)
        payload = json.loads(source.read_text(encoding="utf-8"))
        info = payload["file_info"]
        result = run_full_pipeline(
            str(PROJECT_ROOT / info["relative_path"]),
            str(PROJECT_ROOT / payload["reference_image"]["relative_path"]),
            str(PROJECT_ROOT / payload["expected_state_path"]),
            info["model_id"], info["step_id"], info["view_angle"],
            analysis_result=payload,
            localizer=no_detection_localizer,
            output_dir=output / case,
        )
        records.append({
            "case": case,
            "source_log": source.name,
            "success": result["success"],
            "error_report_count": len(result["error_reports"]),
            "sop_step_count": len((result["correction_sop"] or {}).get("steps", [])),
            "mock_image_count": sum(bool(step.get("generated_image")) for step in (result["correction_sop"] or {}).get("steps", [])),
            "flowchart": bool(result["flowchart_image"]),
            "warnings": result["warnings"],
        })

    first = json.loads(latest("missingpart-A01").read_text(encoding="utf-8"))
    info = first["file_info"]
    ui = run_analysis_for_ui(
        str(PROJECT_ROOT / info["relative_path"]), info["model_id"], info["step_id"], info["view_angle"],
        reference_image_path=str(PROJECT_ROOT / first["reference_image"]["relative_path"]),
        expected_state_path=str(PROJECT_ROOT / first["expected_state_path"]),
        analysis_result=first, localizer=no_detection_localizer,
        output_dir=output / "ui_missingpart-A01",
    )
    report = {"cases": records, "ui_smoke": {"success": ui["success"], "gallery_count": len(ui["sop_gallery"]), "contract_keys": sorted(ui)}}
    (output / "smoke_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["success"] for item in records) and ui["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
