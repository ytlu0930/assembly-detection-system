from pathlib import Path

from utils.step_image_generator import MockStepImageProvider, generate_step_images
from utils.step_image_provider_contract import StepImageResult


def _task():
    return {"step_number": 1, "action": "insert", "instruction": "insert", "affected_parts": ["P1"], "output_filename": "step_01.png", "prompt": "p"}


def test_step_image_result_keeps_legacy_positional_order():
    result = StepImageResult(False, None, "legacy", 1.5, "warning", "error", {"old": True})
    assert result.warning == "warning" and result.error == "error"
    assert result.metadata == {"old": True}


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


def test_sequential_provider_uses_previous_successful_output(tmp_path):
    class Recording:
        name = "recording"
        def __init__(self):
            self.sources = []
        def generate_step_image(self, source_image_path, reference_image_path, prompt, output_path, metadata=None, *, execute_api=False):
            self.sources.append(source_image_path)
            Path(output_path).write_bytes(b"image")
            return StepImageResult(True, str(Path(output_path).resolve()), self.name, status="success")
    tasks = [_task(), {**_task(), "step_number": 2, "output_filename": "step_02.png"}]
    provider = Recording()
    result = generate_step_images(tasks, tmp_path, provider)
    assert [item["status"] for item in result] == ["success", "success"]
    assert provider.sources == ["", str((tmp_path / "step_01.png").resolve())]


def test_external_step_budget_and_failure_stop_remaining_steps(tmp_path):
    class Disabled:
        name = "openai"
        def generate_step_image(self, *args, **kwargs):
            return StepImageResult(False, None, self.name, status="disabled", warning="off")
    tasks = [{**_task(), "step_number": number, "output_filename": f"step_{number:02d}.png"} for number in range(1, 4)]
    result = generate_step_images(tasks, tmp_path, Disabled(), execute_api=False)
    assert [item["status"] for item in result] == ["disabled", "skipped", "skipped"]
    assert all(item["output_path"] is None for item in result)
