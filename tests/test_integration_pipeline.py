import json
from pathlib import Path

from PIL import Image

from utils.integration_pipeline import run_full_pipeline
from utils.openai_image_provider import DISABLED_MESSAGE, OpenAIImageProvider


def _inputs(tmp_path):
    test = tmp_path / "bad.jpg"
    reference = tmp_path / "good.jpg"
    Image.new("RGB", (100, 100), "white").save(test)
    Image.new("RGB", (100, 100), "white").save(reference)
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps({"expected_parts": [{"part_id": "PIN_RED_SHORT"}]}), encoding="utf-8")
    return test, reference, expected


def _analysis(parts=None, error="missingpart"):
    return {"success": True, "model_response": {
        "is_error": error != "correct", "overall_error_type": error,
        "detected_parts": parts if parts is not None else [
            {"part_id": "PIN_RED_SHORT", "error_type": error, "confidence": .9, "description": "missing"}
        ], "summary": "result",
    }}


def test_pipeline_supports_multiple_reports_without_loading_detector(tmp_path):
    test, reference, expected = _inputs(tmp_path)
    analysis = _analysis([
        {"part_id": "PIN_YELLOW", "error_type": "wrongpart", "confidence": .8},
        {"part_id": "PIN_RED_SHORT", "error_type": "wrongpart", "confidence": .8},
    ], "wrongpart")
    result = run_full_pipeline(str(test), str(reference), str(expected), "model03", "step03", "front", analysis_result=analysis, localizer=lambda **kwargs: {"status": "no_detection", "selected_bbox": None}, generate_flowchart=False, output_dir=tmp_path / "out")
    assert result["success"] is True
    assert len(result["error_reports"]) == 2
    assert len(result["correction_sop"]["steps"]) == 4


def test_localization_and_flowchart_failures_fall_back(tmp_path):
    test, reference, expected = _inputs(tmp_path)
    def broken_localizer(**kwargs):
        raise RuntimeError("localizer down")
    def broken_flowchart(*args, **kwargs):
        raise RuntimeError("flowchart down")
    result = run_full_pipeline(str(test), str(reference), str(expected), "model03", "step03", "front", analysis_result=_analysis(), localizer=broken_localizer, flowchart_builder=broken_flowchart, output_dir=tmp_path / "out")
    assert result["success"] is True
    assert result["correction_sop"]["steps"]
    assert any("Localization failed" in item for item in result["warnings"])
    assert any("Flowchart failed" in item for item in result["warnings"])


def test_correct_case_has_no_repair_steps(tmp_path):
    test, reference, expected = _inputs(tmp_path)
    result = run_full_pipeline(str(test), str(reference), str(expected), "model03", "step03", "front", analysis_result=_analysis([], "correct"), localizer=lambda **kwargs: {}, generate_flowchart=False, output_dir=tmp_path / "out")
    assert result["error_reports"] == []
    assert result["correction_sop"]["steps"] == []


def test_disabled_openai_provider_preserves_text_outputs(tmp_path):
    test, reference, expected = _inputs(tmp_path)
    result = run_full_pipeline(
        str(test), str(reference), str(expected), "model03", "step03", "front",
        analysis_result=_analysis(), localizer=lambda **kwargs: {"selected_bbox": None},
        image_provider=OpenAIImageProvider(), generate_flowchart=False, output_dir=tmp_path / "out",
    )
    steps = result["correction_sop"]["steps"]
    assert result["success"] is True and steps
    assert steps[0]["image_generation_status"] == "disabled"
    assert all(step["generated_image"] is None for step in steps)
    assert any(DISABLED_MESSAGE in warning for warning in result["warnings"])


def test_pipeline_named_provider_remains_disabled_without_execute_flag(tmp_path):
    test, reference, expected = _inputs(tmp_path)
    result = run_full_pipeline(
        str(test), str(reference), str(expected), "model03", "step03", "front",
        analysis_result=_analysis(), localizer=lambda **kwargs: {"selected_bbox": None},
        step_image_provider="openai", execute_image_api=False,
        generate_flowchart=False, output_dir=tmp_path / "out",
    )
    assert result["success"] is True
    assert result["correction_sop"]["steps"][0]["image_generation_status"] == "disabled"
