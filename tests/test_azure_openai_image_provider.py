import base64
import io
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from batch_pipeline import run_batch
from main import build_parser as main_parser
from scripts.run_azure_image_smoke_test import main as smoke_main
from step_image_generator_v2 import StepImageGeneratorV2
from utils.azure_openai_image_provider import (
    AzureOpenAIImageProvider,
    MALFORMED_KEY_MESSAGE,
    build_azure_image_edit_endpoint,
)
from utils.step_image_provider_contract import StepImageResult


AUTH = {
    "AZURE_OPENAI_ENDPOINT": "https://undergraduateproject2eastus2.cognitiveservices.azure.com/",
    "AZURE_OPENAI_API_KEY": "azure-secret-value",
    "AZURE_IMAGE_DEPLOYMENT": "gpt-image-2",
    "AZURE_IMAGE_API_VERSION": "2024-02-01",
    "AZURE_IMAGE_AUTH_MODE": "bearer",
    "ENABLE_OPENAI_IMAGE_API": "true",
    "CONFIRM_OPENAI_IMAGE_API_EXECUTION": "true",
}


def _png_b64() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "blue").save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _images(tmp_path: Path):
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 6), "white").save(source)
    Image.new("RGB", (8, 6), "black").save(reference)
    Image.new("RGBA", (8, 6), (255, 255, 255, 0)).save(mask)
    return source, reference, mask


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": [{"b64_json": _png_b64()}]}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class Client:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [Response()])
        self.calls = []

    def post(self, url, **kwargs):
        files = kwargs["files"]
        self.calls.append({
            "url": url,
            "headers": dict(kwargs["headers"]),
            "data": dict(kwargs["data"]),
            "file_keys": list(files),
            "file_names": {key: value[0] for key, value in files.items()},
            "file_bytes": {key: value[1].read() for key, value in files.items()},
            "timeout": kwargs["timeout"],
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider(client=None, **kwargs):
    return AzureOpenAIImageProvider(
        enabled=True, environment=AUTH, http_client=client or Client(), **kwargs
    )


def test_endpoint_builder_encodes_and_rejects_malformed_endpoint():
    endpoint = build_azure_image_edit_endpoint(
        "https://example.cognitiveservices.azure.com///", "gpt image/2", "2024-02-01 preview"
    )
    assert endpoint == "https://example.cognitiveservices.azure.com/openai/deployments/gpt%20image%2F2/images/edits?api-version=2024-02-01%20preview"
    with pytest.raises(ValueError):
        build_azure_image_edit_endpoint("http://example.com?api-key=secret", "gpt-image-2", "2024-02-01")


@pytest.mark.parametrize("auth_mode,expected,absent", [
    ("bearer", "Authorization", "api-key"),
    ("api_key", "api-key", "Authorization"),
])
def test_auth_mode_single_header_and_single_source_image(auth_mode, expected, absent, tmp_path):
    source, reference, _ = _images(tmp_path)
    client = Client()
    provider = AzureOpenAIImageProvider(enabled=True, environment={**AUTH, "AZURE_IMAGE_AUTH_MODE": auth_mode}, http_client=client)
    result = provider.generate_step_image(str(source), str(reference), "exact prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "success"
    call = client.calls[0]
    assert expected in call["headers"] and absent not in call["headers"]
    assert call["file_keys"] == ["image"]
    assert call["file_names"]["image"] == "source.png"
    assert call["data"] == {"prompt": "exact prompt"}
    assert "reference" not in call["file_keys"]
    assert result.metadata["reference_image"] == str(reference)
    assert result.metadata["supports_multi_image_reference"] is False
    assert "azure-secret-value" not in json.dumps(result.to_dict())


def test_optional_mask_is_the_only_additional_binary(tmp_path):
    source, reference, mask = _images(tmp_path)
    client = Client()
    result = _provider(client).generate_step_image(
        str(source), str(reference), "prompt", str(tmp_path / "out.png"),
        execute_api=True, mask_path=mask,
    )
    assert result.success and client.calls[0]["file_keys"] == ["image", "mask"]
    assert result.metadata["mask_used"] is True


@pytest.mark.parametrize("bad_key", ["", "   ", "=abc", " key", "key ", "key\n", "your_api_key_here", "https://key", "AZURE_OPENAI_API_KEY=value", "[key](https://x)"])
def test_key_preflight_stops_before_network_without_leak(tmp_path, bad_key):
    source, reference, _ = _images(tmp_path)
    client = Client()
    provider = AzureOpenAIImageProvider(enabled=True, environment={**AUTH, "AZURE_OPENAI_API_KEY": bad_key}, http_client=client)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "not_configured" and result.error == MALFORMED_KEY_MESSAGE
    assert client.calls == []
    if bad_key:
        assert bad_key not in json.dumps(result.to_dict())


@pytest.mark.parametrize("enabled,execute,env_enabled,env_confirmed", [
    (False, True, "true", "true"), (True, False, "true", "true"),
    (True, True, "false", "true"), (True, True, "true", "false"),
])
def test_triple_gate_prevents_http(enabled, execute, env_enabled, env_confirmed, tmp_path):
    source, reference, _ = _images(tmp_path)
    client = Client()
    provider = AzureOpenAIImageProvider(
        enabled=enabled,
        environment={**AUTH, "ENABLE_OPENAI_IMAGE_API": env_enabled, "CONFIRM_OPENAI_IMAGE_API_EXECUTION": env_confirmed},
        http_client=client,
    )
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=execute)
    assert result.status == "disabled" and client.calls == []


@pytest.mark.parametrize("code,status", [
    (400, "invalid_request"), (401, "authentication_error"),
    (403, "permission_error"), (404, "deployment_or_endpoint_not_found"),
])
def test_nonretryable_http_mapping(code, status, tmp_path):
    source, reference, _ = _images(tmp_path)
    client = Client([Response(code, {"error": {"message": "azure-secret-value"}})])
    waits = []
    provider = _provider(client, max_retries=2, max_requests_per_run=3, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == status and result.request_count == 1 and waits == []
    assert "azure-secret-value" not in result.error


@pytest.mark.parametrize("first,status", [(Response(429), "rate_limited"), (Response(500), "service_error"), (TimeoutError("slow"), "timeout")])
def test_retryable_errors_then_success(first, status, tmp_path):
    source, reference, _ = _images(tmp_path)
    client = Client([first, Response()])
    waits = []
    provider = _provider(client, max_retries=2, max_requests_per_run=3, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "success" and result.request_count == 2 and result.retry_count == 1
    assert waits == [2]


def test_two_retries_use_two_and_four_seconds(tmp_path):
    source, reference, _ = _images(tmp_path)
    waits = []
    provider = _provider(Client([TimeoutError(), TimeoutError(), Response()]), max_retries=2, max_requests_per_run=3, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.success and waits == [2, 4]


@pytest.mark.parametrize("payload,status", [
    ({}, "invalid_response"), ({"data": []}, "invalid_response"),
    ({"data": [{}]}, "invalid_response"), ({"data": [{"b64_json": "bad"}]}, "invalid_response"),
    ({"data": [{"b64_json": base64.b64encode(b"not image").decode()}]}, "output_validation_failed"),
])
def test_response_and_output_validation(payload, status, tmp_path):
    source, reference, _ = _images(tmp_path)
    provider = _provider(Client([Response(200, payload)]))
    output = tmp_path / "out.png"
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(output), execute_api=True)
    assert result.status == status and not output.exists()


def test_request_budget_zero_stops_before_http(tmp_path):
    source, reference, _ = _images(tmp_path)
    client = Client()
    result = _provider(client, max_requests_per_run=0).generate_step_image(
        str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True
    )
    assert result.status == "invalid_request" and client.calls == []


def test_confirm_cost_gate_in_v2_generator_keeps_azure_disabled():
    provider = AzureOpenAIImageProvider(enabled=True, environment=AUTH, http_client=Client())
    generator = StepImageGeneratorV2(provider=provider, execute_api=True, confirm_cost=False)
    assert generator.execute_api is False


def test_main_parser_selects_azure_without_implying_execution():
    args = main_parser().parse_args(["--image-provider", "azure_openai"])
    assert args.image_provider == "azure_openai"
    assert args.execute_image_api is False and args.confirm_cost is False


def test_v2_sequential_mode_keeps_reference_metadata(tmp_path):
    source, reference, _ = _images(tmp_path)
    calls = []
    class RecordingProvider:
        name = "azure_openai"
        def generate_step_image(self, source_image_path, reference_image_path, prompt, output_path, metadata=None, **kwargs):
            calls.append((source_image_path, reference_image_path))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), "green").save(output_path)
            return StepImageResult(True, str(Path(output_path).resolve()), self.name, status="success")
    package = {
        "model_id": "model03", "step_id": "step03", "image_name": source.name,
        "generation_allowed": True, "requires_manual_review": False,
        "step_prompts": [
            {"sequence_index": 1, "sop_step_no": 1, "action": "insert_part", "title": "one", "prompt_en": "p1", "output_filename": "one.png", "image_task": {"api_mode": "edit", "branch": "assembly", "use_previous_output": False}, "image_inputs": [{"role": "base_image", "path": str(source)}, {"role": "reference_image", "path": str(reference)}]},
            {"sequence_index": 2, "sop_step_no": 2, "action": "verify_local_result", "title": "two", "prompt_en": "p2", "output_filename": "two.png", "image_task": {"api_mode": "edit", "branch": "assembly", "use_previous_output": True}, "image_inputs": [{"role": "reference_image", "path": str(reference)}]},
        ],
    }
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(package), encoding="utf-8")
    manifest = StepImageGeneratorV2(provider=RecordingProvider(), execute_api=True, confirm_cost=True).run(prompts_json_path=prompts, output_dir=tmp_path / "generated")
    assert manifest.successful_task_count == 2
    assert calls[0] == (str(source), str(reference))
    assert calls[1][0].endswith("one.png") and calls[1][1] == str(reference)


def test_azure_smoke_cli_dry_run_and_missing_cost_never_network(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", AUTH["AZURE_OPENAI_ENDPOINT"])
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", AUTH["AZURE_OPENAI_API_KEY"])
    monkeypatch.setenv("ENABLE_OPENAI_IMAGE_API", "true")
    monkeypatch.setenv("CONFIRM_OPENAI_IMAGE_API_EXECUTION", "true")
    with patch.object(socket, "create_connection") as connect:
        assert smoke_main(["--dry-run", "--output-dir", str(tmp_path / "dry")]) == 0
        dry = json.loads(capsys.readouterr().out)
        assert dry["status"] == "dry-run" and dry["reference_binary_supported"] is False
        assert smoke_main(["--execute-api", "--output-dir", str(tmp_path / "blocked")]) == 2
        blocked_output = capsys.readouterr().out
    connect.assert_not_called()
    assert AUTH["AZURE_OPENAI_API_KEY"] not in blocked_output


def test_batch_azure_needs_all_live_flags(tmp_path):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "a_parsed_20260808_010101_000001.json").write_text("{}", encoding="utf-8")
    calls = []
    def runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="success", final_instruction_path=None, warnings=[], errors=[], execute_image_api=False)
    summary = run_batch(
        parsed_dir=parsed_dir, output_dir=tmp_path / "batch", limit=1,
        generate_images=True, image_provider="azure_openai", execute_image_api=True,
        confirm_cost=False, pipeline_runner=runner,
    )
    assert summary.execute_image_api is False and calls[0]["execute_image_api"] is False
