from pathlib import Path

from utils.step_image_generator import MockStepImageProvider, generate_step_images


def _task():
    return {"step_number": 1, "action": "insert", "instruction": "insert", "affected_parts": ["P1"], "output_filename": "step_01.png", "prompt": "p"}


def test_mock_provider_generates_offline_image(tmp_path):
    result = generate_step_images([_task()], tmp_path, MockStepImageProvider())
    assert result[0]["status"] == "success"
    assert Path(result[0]["output_path"]).is_file()


def test_provider_failure_is_recorded(tmp_path):
    class Broken:
        name = "broken"
        def generate(self, task, output_path):
            raise RuntimeError("no provider")
    result = generate_step_images([_task()], tmp_path, Broken())
    assert result[0]["status"] == "failed"
    assert "no provider" in result[0]["error"]
