import hashlib
import json
from pathlib import Path

from main import run_pipeline
from correction_sop_generator import CorrectionSOPGenerator
from utils.step_image_provider_contract import StepImageResult


ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "logs/current_parsed_json/model03_step03_missingpart-A01_front_01_parsed_20260701_160358_542567.json"


class ContrastLocalizer:
    """Offline evidence: the predicted eye count is equal in test/reference."""

    def __init__(self):
        self.calls = []

    def localize(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "success",
            "selected_bbox": [100, 100, 200, 200],
            "selected_detection_score": 0.95,
            "selected_selection_score": 0.95,
            "estimated_count": 2,
            "annotated_image_path": kwargs["image_path"],
            "error_message": None,
        }


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def generate_step_image(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return StepImageResult(success=True, output_path=None, provider=self.name, request_count=1)


def test_missingpart_a01_conflict_blocks_eye_sop_and_provider(tmp_path):
    before = hashlib.sha256(PARSED.read_bytes()).hexdigest()
    parsed_payload = json.loads(PARSED.read_text(encoding="utf-8"))
    assert parsed_payload["model_response"]["detected_parts"][0]["part_id"] == "EYE_BALL"

    localizer = ContrastLocalizer()
    provider = RecordingProvider()
    manifest = run_pipeline(
        parsed_json_path=PARSED,
        output_dir=tmp_path / "identity_regression",
        generate_images=True,
        provider=provider,
        localizer=localizer,
        allow_manual_review=True,
        image_max_tasks=5,
        image_max_requests=5,
    )

    results = json.loads(Path(manifest.results_path).read_text(encoding="utf-8"))
    report = results["error_reports"][0]
    assert report["part_id"] == "EYE_BALL"
    assert report["identity_status"] == "conflict"
    assert report["verified_part_id"] is None
    assert report["requires_manual_review"] is True

    sop = json.loads(Path(manifest.correction_sop_path).read_text(encoding="utf-8"))
    assert sop["identity_verification_blocked"] is True
    assert sop["requires_manual_review"] is True
    assert "EYE_BALL" not in sop["target_parts"]
    assert not any(
        step["requires_image_generation"] and step.get("target_part_id") == "EYE_BALL"
        for step in sop["steps"]
    )

    prompts = json.loads(Path(manifest.step_prompts_path).read_text(encoding="utf-8"))
    assert prompts["identity_verification_blocked"] is True
    assert prompts["generation_allowed"] is False
    assert prompts["step_prompts"] == []
    assert "white ball with black pupil" not in json.dumps(prompts, ensure_ascii=False)

    generation = json.loads(
        Path(manifest.generated_steps_dir, "generation_manifest_v2.json").read_text(encoding="utf-8")
    )
    assert generation["identity_verification_blocked"] is True
    assert generation["requested_task_count"] == 0
    assert provider.calls == []
    assert manifest.manual_review_required is True
    assert len(localizer.calls) == 2
    assert hashlib.sha256(PARSED.read_bytes()).hexdigest() == before


def test_pre_verifier_results_are_fail_closed(tmp_path):
    parsed_payload = json.loads(PARSED.read_text(encoding="utf-8"))
    legacy_results = {
        "vision_result": parsed_payload,
        "model_response": parsed_payload["model_response"],
        "test_image_path": str(ROOT / parsed_payload["test_image"]["relative_path"]),
        "reference_image_path": str(ROOT / parsed_payload["reference_image"]["relative_path"]),
        "expected_state_path": str(ROOT / parsed_payload["expected_state_path"]),
        "error_reports": [
            {
                "part_id": "EYE_BALL",
                "error_type": "missingpart",
                "confidence": 0.95,
                "localization": {"status": "success", "selected_detection_score": 0.95, "selected_selection_score": 0.95},
                "localization_strategy": {},
            }
        ],
        "localization": {"status": "success", "selected_detection_score": 0.95, "selected_selection_score": 0.95},
        "localization_strategy": {},
    }
    path = tmp_path / "legacy_results.json"
    path.write_text(json.dumps(legacy_results), encoding="utf-8")
    sop = CorrectionSOPGenerator().generate_from_results(path).to_dict()
    assert sop["identity_verification_blocked"] is True
    assert sop["requires_manual_review"] is True
    assert "EYE_BALL" not in sop["target_parts"]
    assert not any(step["requires_image_generation"] for step in sop["steps"])
