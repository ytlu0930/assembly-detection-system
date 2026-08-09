"""Guarded GPT Image 2 Images Edit provider.

The adapter is usable with an injected client, but live execution is disabled
unless all three authorization gates are explicitly enabled.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import re
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Mapping

from PIL import Image, UnidentifiedImageError

from utils.step_image_provider_contract import StepImageResult


DEFAULT_MODEL = "gpt-image-2"
DISABLED_MESSAGE = "OpenAI Image API execution is disabled by the required authorization gates."
SUPPORTED_INPUT_FORMATS = {"JPEG", "PNG", "WEBP"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]+")


class ProviderNotConfiguredError(RuntimeError):
    """Retained for compatibility with callers importing the old stub symbol."""


class OpenAIImageProvider:
    """Generate one corrected SOP image using ``client.images.edit``."""

    name = "openai"
    mode = "image_edit"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        quality: str = "low",
        size: str = "1536x1024",
        output_format: str = "png",
        timeout_seconds: float = 120,
        max_retries: int = 2,
        max_requests_per_run: int = 1,
        enabled: bool = False,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        environment: Mapping[str, str] | None = None,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self.model = str(model)
        self.quality = str(quality)
        self.size = str(size)
        self.output_format = str(output_format).lower().lstrip(".")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.max_requests_per_run = max(0, int(max_requests_per_run))
        self.enabled = bool(enabled)
        self._client = client
        self._client_factory = client_factory
        self._environment = environment
        self._sleep = sleep_fn
        self._request_count = 0

    @property
    def provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "mode": self.mode,
            "enabled": self.enabled,
            "quality": self.quality,
            "size": self.size,
            "output_format": self.output_format,
        }

    def _env(self, name: str) -> str:
        source = self._environment if self._environment is not None else os.environ
        return str(source.get(name, ""))

    @staticmethod
    def _valid_api_key(raw: str) -> bool:
        """Reject clearly malformed or placeholder credentials without exposing them."""
        if not raw or raw != raw.strip() or raw.startswith("=") or "\n" in raw or "\r" in raw:
            return False
        normalized = raw.lower()
        placeholders = {"your_api_key_here", "replace_me", "changeme", "sk-...", "<api_key>"}
        return normalized not in placeholders and "placeholder" not in normalized

    @staticmethod
    def _is_true(value: str) -> bool:
        return value.strip().lower() == "true"

    def _result(
        self,
        started: float,
        *,
        status: str,
        output_path: str | None = None,
        warning: str | None = None,
        error: str | None = None,
        retry_count: int = 0,
        last_error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StepImageResult:
        duration = perf_counter() - started
        return StepImageResult(
            success=status == "success",
            status=status,
            output_path=output_path,
            provider=self.name,
            model=self.model,
            mode=self.mode,
            duration=duration,
            duration_seconds=duration,
            request_count=self._request_count,
            quality=self.quality,
            size=self.size,
            output_format=self.output_format,
            retry_count=retry_count,
            last_error_type=last_error_type,
            warning=warning,
            error=error,
            metadata={**(metadata or {}), **self.provider_metadata},
        )

    def _sanitize_error(self, exc: BaseException, api_key: str) -> str:
        text = f"{type(exc).__name__}: {exc}"
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
        return _TOKEN_PATTERN.sub("[REDACTED]", text)

    def _validate_inputs(self, source: Path, reference: Path, prompt: str, output: Path) -> str | None:
        if not prompt.strip():
            return "Prompt must not be empty."
        for label, path in (("source", source), ("reference", reference)):
            if not path.is_file():
                return f"Missing {label} image."
            try:
                with Image.open(path) as image:
                    image.verify()
                    if (image.format or "").upper() not in SUPPORTED_INPUT_FORMATS:
                        return f"Unsupported {label} image format."
            except (OSError, UnidentifiedImageError):
                return f"Invalid {label} image."
        if output.suffix.lower() != f".{self.output_format}":
            return "Output extension does not match output_format."
        if output.exists():
            return "Output path already exists; overwriting is not allowed."
        if output in {source, reference}:
            return "Output path must differ from both input images."
        resolved = output.resolve()
        for protected in (PROJECT_ROOT / "input", PROJECT_ROOT / "regression_subset"):
            try:
                resolved.relative_to(protected.resolve())
                return "Output path must not be inside protected input directories."
            except ValueError:
                pass
        return None

    def _client_instance(self, api_key: str) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(api_key=api_key)
        else:
            from openai import OpenAI  # lazy import: never runs on disabled/dry paths

            self._client = OpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def _classify(exc: BaseException) -> tuple[str, bool]:
        name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, TimeoutError) or "timeout" in name:
            return "timeout", True
        if status_code == 429 or "ratelimit" in name or "rate_limit" in name:
            return "rate_limited", True
        if status_code is not None and 500 <= int(status_code) <= 599:
            return "api_error", True
        if "connection" in name:
            return "api_error", True
        return "api_error", False

    def generate_step_image(
        self,
        source_image_path: str,
        reference_image_path: str,
        prompt: str,
        output_path: str,
        metadata: dict[str, Any] | None = None,
        *,
        execute_api: bool = False,
        mask_path: str | Path | None = None,
    ) -> StepImageResult:
        """Execute one image edit only after code and both environment gates agree."""
        started = perf_counter()
        details = dict(metadata or {})
        authorized = (
            self.enabled
            and execute_api is True
            and self._is_true(self._env("ENABLE_OPENAI_IMAGE_API"))
            and self._is_true(self._env("CONFIRM_OPENAI_IMAGE_API_EXECUTION"))
        )
        if not authorized:
            return self._result(started, status="disabled", warning=DISABLED_MESSAGE, error=DISABLED_MESSAGE, metadata=details)
        raw_api_key = self._env("OPENAI_API_KEY")
        if not self._valid_api_key(raw_api_key):
            message = "OPENAI_API_KEY appears malformed. Check the local .env file."
            return self._result(started, status="not_configured", warning=message, error=message, metadata=details)
        api_key = raw_api_key

        source = Path(source_image_path).expanduser().resolve()
        reference = Path(reference_image_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        invalid = self._validate_inputs(source, reference, prompt, output)
        if invalid:
            return self._result(started, status="invalid_input", error=invalid, metadata=details)
        if self._request_count >= self.max_requests_per_run:
            return self._result(started, status="invalid_input", error="OpenAI image request budget exhausted.", metadata=details)

        retry_count = 0
        last_error_type: str | None = None
        for attempt in range(self.max_retries + 1):
            if self._request_count >= self.max_requests_per_run:
                return self._result(started, status=last_error_type or "api_error", error="OpenAI image request budget exhausted.", retry_count=retry_count, last_error_type=last_error_type, metadata=details)
            try:
                client = self._client_instance(api_key)
                with source.open("rb") as source_file, reference.open("rb") as reference_file:
                    self._request_count += 1
                    response = client.images.edit(
                        model=self.model,
                        image=[source_file, reference_file],
                        prompt=prompt,
                        quality=self.quality,
                        size=self.size,
                        output_format=self.output_format,
                        timeout=self.timeout_seconds,
                    )
                data = getattr(response, "data", None)
                encoded = getattr(data[0], "b64_json", None) if data else None
                if not isinstance(encoded, str) or not encoded:
                    return self._result(started, status="invalid_response", error="Image response did not contain b64_json.", retry_count=retry_count, metadata=details)
                try:
                    payload = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError):
                    return self._result(started, status="invalid_response", error="Image response contained invalid base64.", retry_count=retry_count, metadata=details)
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError):
                    return self._result(started, status="output_validation_failed", error="Decoded output is not a valid image.", retry_count=retry_count, metadata=details)
                if not payload:
                    return self._result(started, status="output_validation_failed", error="Decoded output is empty.", retry_count=retry_count, metadata=details)
                try:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(payload)
                    if output.stat().st_size == 0:
                        raise OSError("Written output is empty")
                    with Image.open(output) as written:
                        written.verify()
                except (OSError, UnidentifiedImageError) as exc:
                    if output.is_file():
                        output.unlink()
                    return self._result(
                        started,
                        status="output_validation_failed",
                        error=f"Output validation failed: {type(exc).__name__}.",
                        retry_count=retry_count,
                        last_error_type=type(exc).__name__,
                        metadata=details,
                    )
                return self._result(started, status="success", output_path=str(output), retry_count=retry_count, metadata=details)
            except Exception as exc:
                status, retryable = self._classify(exc)
                last_error_type = type(exc).__name__
                message = self._sanitize_error(exc, api_key)
                can_retry = retryable and attempt < self.max_retries and self._request_count < self.max_requests_per_run
                if not can_retry:
                    return self._result(started, status=status, error=message, retry_count=retry_count, last_error_type=last_error_type, metadata=details)
                retry_count += 1
                self._sleep(2**retry_count)
        return self._result(started, status="api_error", error="Image request failed.", retry_count=retry_count, last_error_type=last_error_type, metadata=details)
