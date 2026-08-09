import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from utils.openai_image_provider import DEFAULT_MODEL, DISABLED_MESSAGE, OpenAIImageProvider


AUTH = {
    "ENABLE_OPENAI_IMAGE_API": "true",
    "CONFIRM_OPENAI_IMAGE_API_EXECUTION": "true",
    "OPENAI_API_KEY": "sk-secret-test-value",
}


def _png_b64() -> str:
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _images(tmp_path: Path):
    source, reference = tmp_path / "source.png", tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "white").save(source)
    Image.new("RGB", (8, 8), "black").save(reference)
    return source, reference


class RecordingImages:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])])
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append({**kwargs, "image_names": [Path(item.name).name for item in kwargs["image"]]})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _provider(images=None, **kwargs):
    client = SimpleNamespace(images=images or RecordingImages())
    return OpenAIImageProvider(client=client, enabled=True, environment=AUTH, **kwargs), client


def test_defaults_are_disabled_and_do_not_construct_client():
    called = []
    provider = OpenAIImageProvider(client_factory=lambda **kwargs: called.append(kwargs))
    result = provider.generate_step_image("missing", "missing", "prompt", "out.png", execute_api=True)
    assert provider.model == DEFAULT_MODEL == "gpt-image-2"
    assert result.status == "disabled" and result.warning == DISABLED_MESSAGE
    assert called == []


def test_missing_key_is_not_configured_without_client(tmp_path):
    source, reference = _images(tmp_path)
    called = []
    provider = OpenAIImageProvider(enabled=True, environment={**AUTH, "OPENAI_API_KEY": ""}, client_factory=lambda **kw: called.append(kw))
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "not_configured"
    assert called == []


@pytest.mark.parametrize("bad_key", ["=abc123", " sk-test", "sk-test ", "sk-test\n", "your_api_key_here", "sk-..."])
def test_malformed_key_preflight_rejects_without_leaking_or_client(tmp_path, bad_key):
    source, reference = _images(tmp_path)
    called = []
    provider = OpenAIImageProvider(
        enabled=True,
        environment={**AUTH, "OPENAI_API_KEY": bad_key},
        client_factory=lambda **kwargs: called.append(kwargs),
    )
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "not_configured"
    assert result.error == "OPENAI_API_KEY appears malformed. Check the local .env file."
    assert bad_key not in result.error and called == []


def test_success_uses_source_then_reference_and_validates_output(tmp_path):
    source, reference = _images(tmp_path)
    provider, client = _provider()
    output = tmp_path / "out.png"
    result = provider.generate_step_image(str(source), str(reference), "exact prompt", str(output), {"step": 1}, execute_api=True)
    assert result.status == "success" and output.is_file()
    Image.open(output).verify()
    call = client.images.calls[0]
    assert call["image_names"] == ["source.png", "reference.png"]
    assert call["prompt"] == "exact prompt"
    assert call["model"] == "gpt-image-2"
    assert "input_fidelity" not in call
    assert result.request_count == 1


@pytest.mark.parametrize("response,status", [
    (SimpleNamespace(data=[]), "invalid_response"),
    (SimpleNamespace(data=[SimpleNamespace(b64_json="not base64")]), "invalid_response"),
    (SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"no image").decode())]), "output_validation_failed"),
])
def test_invalid_responses(response, status, tmp_path):
    source, reference = _images(tmp_path)
    provider, _ = _provider(RecordingImages([response]))
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == status
    assert not (tmp_path / "out.png").exists()


def test_timeout_retries_with_finite_backoff(tmp_path):
    source, reference = _images(tmp_path)
    images = RecordingImages([TimeoutError("slow"), TimeoutError("slow"), SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])])
    waits = []
    provider, _ = _provider(images, max_retries=2, max_requests_per_run=3, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "success"
    assert result.request_count == 3 and result.retry_count == 2
    assert waits == [2, 4]


def test_rate_limit_retries_within_request_budget(tmp_path):
    class RateLimitError(Exception):
        status_code = 429
    source, reference = _images(tmp_path)
    images = RecordingImages([RateLimitError("busy"), SimpleNamespace(data=[SimpleNamespace(b64_json=_png_b64())])])
    waits = []
    provider, _ = _provider(images, max_retries=2, max_requests_per_run=2, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "success" and result.request_count == 2
    assert waits == [2]


def test_auth_error_is_not_retried_and_key_is_redacted(tmp_path):
    class AuthenticationError(Exception):
        status_code = 401
    source, reference = _images(tmp_path)
    images = RecordingImages([AuthenticationError("bad sk-secret-test-value")])
    waits = []
    provider, _ = _provider(images, max_retries=2, max_requests_per_run=3, sleep_fn=waits.append)
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "api_error" and result.request_count == 1
    assert waits == [] and "sk-secret" not in result.error


def test_invalid_input_and_existing_output_never_call_client(tmp_path):
    source, reference = _images(tmp_path)
    provider, client = _provider()
    output = tmp_path / "out.png"
    output.write_bytes(b"owned")
    result = provider.generate_step_image(str(source), str(reference), "prompt", str(output), execute_api=True)
    assert result.status == "invalid_input"
    assert output.read_bytes() == b"owned" and client.images.calls == []


def test_request_budget_and_output_format_are_checked_before_client(tmp_path):
    source, reference = _images(tmp_path)
    provider, client = _provider(max_requests_per_run=0)
    budget = provider.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert budget.status == "invalid_input"
    mismatch = OpenAIImageProvider(client=client, enabled=True, environment=AUTH, output_format="webp")
    result = mismatch.generate_step_image(str(source), str(reference), "prompt", str(tmp_path / "out.png"), execute_api=True)
    assert result.status == "invalid_input" and client.images.calls == []
