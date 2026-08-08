import json
from pathlib import Path
from types import SimpleNamespace

from app import run_analysis
from batch_pipeline import run_batch, select_latest_per_image
from main import run_pipeline


ROOT = Path(__file__).resolve().parents[1]


class FakeLocalizer:
    def __init__(self):
        self.calls = []

    def localize(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "success",
            "selected_bbox": [10, 10, 40, 40],
            "selected_detection_score": 0.95,
            "selected_selection_score": 0.95,
            "annotated_image_path": kwargs["image_path"],
            "error_message": None,
        }


def _parsed(tmp_path: Path, parts=None) -> Path:
    payload = {
        "success": True,
        "file_info": {
            "image_name": "model03_step03_wrongpart-A01_front_01.jpg",
            "relative_path": "input/wrongpart/model03_step03/model03_step03_wrongpart-A01_front_01.jpg",
            "model_id": "model03", "step_id": "step03", "view_angle": "front",
        },
        "test_image": {"relative_path": "input/wrongpart/model03_step03/model03_step03_wrongpart-A01_front_01.jpg"},
        "reference_image": {"relative_path": "input/normal/model03_step03/model03_step03_correct-01_front_01.jpg"},
        "expected_state_path": "ground_truth/model03/step03.json",
        "model_response": {
            "model_id": "model03", "step_id": "step03", "is_error": True,
            "overall_error_type": "wrongpart", "summary": "two parts swapped",
            "error_components": ["identity", "position"],
            "detected_parts": parts or [
                {"part_id": "PIN_YELLOW", "error_type": "wrongpart", "confidence": 0.95},
                {"part_id": "PIN_RED_SHORT", "error_type": "wrongpart", "confidence": 0.94},
            ],
        },
    }
    path = tmp_path / "model03_step03_wrongpart-A01_front_01_parsed_20260808_120000_000001.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_main_localizes_all_parts_builds_swap_and_instruction_book(tmp_path):
    parsed = _parsed(tmp_path)
    localizer = FakeLocalizer()
    manifest = run_pipeline(parsed_json_path=parsed, output_dir=tmp_path / "out", localizer=localizer)
    assert manifest.status in {"success", "partial"}
    assert len(localizer.calls) == 2
    assert Path(manifest.final_instruction_path).is_file()
    assert Path(manifest.results_path).is_file()
    assert Path(manifest.correction_sop_path).is_file()
    assert Path(manifest.step_prompts_path).is_file()
    assert Path(manifest.generated_steps_dir, "generation_manifest_v2.json").is_file()
    sop = json.loads(Path(manifest.correction_sop_path).read_text(encoding="utf-8"))
    assert sop["repair_scope"] == "local"
    assert set(sop["target_parts"]) >= {"PIN_YELLOW", "PIN_RED_SHORT"}
    assert any(step["action"] == "swap_parts" for step in sop["steps"])
    prompt = json.loads(Path(manifest.step_prompts_path).read_text(encoding="utf-8"))["step_prompts"][0]["prompt_en"]
    assert "Image 1 is the current/source" in prompt
    assert "Preserve every non-target brick" in prompt


def test_batch_selects_latest_limits_three_and_reuses_pipeline(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for name in [
        "a_parsed_20260801_010101_000001.json", "a_parsed_20260802_010101_000001.json",
        "b_parsed_20260801_010101_000001.json", "c_parsed_20260801_010101_000001.json",
        "d_parsed_20260801_010101_000001.json",
    ]:
        (parsed_dir / name).write_text("{}", encoding="utf-8")
    latest = select_latest_per_image(list(parsed_dir.glob("*.json")))
    assert len(latest) == 4 and "20260802" in next(path.name for path in latest if path.name.startswith("a_"))
    calls = []
    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="success", final_instruction_path=str(Path(kwargs["output_dir"]) / "book.png"), warnings=[], errors=[], execute_image_api=False)
    summary = run_batch(parsed_dir=parsed_dir, output_dir=tmp_path / "batch", limit=3, pipeline_runner=fake_runner)
    assert summary.requested_count == 3 and summary.completed_count == 3
    assert len(calls) == 3 and len({str(call["output_dir"]) for call in calls}) == 3
    assert all(call["image_provider"] == "mock" and not call["execute_image_api"] for call in calls)


def test_app_callback_displays_manifest_final_instruction_and_has_no_flowchart(tmp_path):
    parsed = _parsed(tmp_path)
    book = tmp_path / "book.png"
    book.write_bytes(b"book")
    calls = []
    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            annotated_image_path="annotated.png", final_instruction_path=str(book),
            warnings=[], errors=[], status="success",
        )
    annotated, final, message, status = run_analysis(str(parsed), pipeline_runner=fake_runner)
    assert calls and final == str(book) and annotated == "annotated.png"
    assert status == {"status": "success"}
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from main import run_pipeline" in source
    assert "generate_flowchart" not in source and "flowchart_generator" not in source
