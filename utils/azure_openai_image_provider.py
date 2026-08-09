"""Guarded Azure-hosted GPT Image 2 single-image edit provider."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlencode, urlparse

from PIL import Image, UnidentifiedImageError

from utils.step_image_provider_contract import StepImageResult


DEFAULT_DEPLOYMENT = "gpt-image-2"
DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_AUTH_MODE = "bearer"
DISABLED_MESSAGE = "Azure OpenAI Image API execution is disabled by the required authorization gates."
MALFORMED_KEY_MESSAGE = "AZURE_OPENAI_API_KEY appears malformed. Check the local .env file."
SUPPORTED_INPUT_FORMATS = {"JPEG", "PNG", "WEBP"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_azure_image_edit_endpoint(endpoint: str, deployment: str, api_version: str) -> str:
    """Build the Azure deployment edit endpoint without credentials."""
    raw_endpoint = str(endpoint).strip().rstrip("/")
    parsed = urlparse(raw_endpoint)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("AZURE_OPENAI_ENDPOINT is malformed; use an HTTPS Azure resource base URL without query parameters.")
    deployment_value = str(deployment).strip()
    version_value = str(api_version).strip()
    if not deployment_value or not version_value:
        raise ValueError("Azure image deployment and API version must be configured.")
    return (
        f"{raw_endpoint}/openai/deployments/{quote(deployment_value, safe='')}/images/edits?"
        f"{urlencode({'api-version': version_value}, quote_via=quote)}"
    )


class AzureOpenAIImageProvider:
    """Use Azure's documented single-image multipart edit contract."""

    name = "azure_openai"
    mode = "image_edit"
    supports_multi_image_reference = False

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        auth_mode: str | None = None,
        model: str = DEFAULT_DEPLOYMENT,
        quality: str = "low",
        size: str = "1536x1024",
        output_format: str = "png",
        timeout_seconds: float = 120,
        max_retries: int = 2,
        max_requests_per_run: int = 1,
        enabled: bool = False,
        http_client: Any | None = None,
        http_client_factory: Callable[[], Any] | None = None,
        environment: Mapping[str, str] | None = None,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        self._endpoint = endpoint
        self._deployment = deployment
        self._api_version = api_version
        self._auth_mode = auth_mode
        self.model = str(model)
        self.quality = str(quality)
        self.size = str(size)
        self.output_format = str(output_format).lower().lstrip(".")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.max_requests_per_run = max(0, int(max_requests_per_run))
        self.enabled = bool(enabled)
        self._http_client = http_client
        self._http_client_factory = http_client_factory
        self._environment = environment
        self._sleep = sleep_fn
        self._request_count = 0

    def _env(self, name: str) -> str:
        source = self._environment if self._environment is not None else os.environ
        return str(source.get(name, ""))

    @staticmethod
    def _is_true(value: str) -> bool:
        return value.strip().lower() == "true"

    @property
    def endpoint_base(self) -> str:
        return self._endpoint if self._endpoint is not None else self._env("AZURE_OPENAI_ENDPOINT")

    @property
    def deployment(self) -> str:
        return self._deployment if self._deployment is not None else (self._env("AZURE_IMAGE_DEPLOYMENT") or DEFAULT_DEPLOYMENT)

    @property
    def api_version(self) -> str:
        return self._api_version if self._api_version is not None else (self._env("AZURE_IMAGE_API_VERSION") or DEFAULT_API_VERSION)

    @property
    def auth_mode(self) -> str:
        return (self._auth_mode if self._auth_mode is not None else (self._env("AZURE_IMAGE_AUTH_MODE") or DEFAULT_AUTH_MODE)).strip().lower()

    @property
    def edit_endpoint(self) -> str:
        return build_azure_image_edit_endpoint(self.endpoint_base, self.deployment, self.api_version)

    @staticmethod
    def _valid_api_key(raw: str) -> bool:
        if not raw or not raw.strip() or raw != raw.strip() or raw.startswith("=") or "\n" in raw or "\r" in raw:
            return False
        lower = raw.lower()
        if lower.startswith(("http://", "https://")) or "placeholder" in lower:
            return False
        if lower in {"your_api_key_here", "replace_me", "changeme", "<api_key>", "azure_openai_api_key"}:
            return False
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", raw) or re.search(r"\[[^\]]+\]\([^\)]+\)", raw):
            return False
        return True

    def _base_metadata(
        self,
        source: str,
        reference: str,
        mask_used: bool,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            **(metadata or {}),
            "provider": self.name,
            "model": self.model,
            "deployment": self.deployment,
            "mode": self.mode,
            "api_version": self.api_version,
            "auth_mode": self.auth_mode,
            "quality": self.quality,
            "size": self.size,
            "output_format": self.output_format,
            "source_image": source,
            "reference_image": reference,
            "mask_used": mask_used,
            "supports_multi_image_reference": self.supports_multi_image_reference,
            "request_count": self._request_count,
        }

    def _result(
        self,
        started: float,
        *,
        status: str,
        source: str,
        reference: str,
        mask_used: bool,
        metadata: dict[str, Any] | None = None,
        output_path: str | None = None,
        warning: str | None = None,
        error: str | None = None,
        retry_count: int = 0,
        last_error_type: str | None = None,
    ) -> StepImageResult:
        duration = perf_counter() - started
        details = self._base_metadata(source, reference, mask_used, metadata)
        details.update({"request_count": self._request_count, "retry_count": retry_count, "duration_seconds": duration})
        return StepImageResult(
            success=status == "success", status=status, output_path=output_path,
            provider=self.name, model=self.model, mode=self.mode,
            duration=duration, duration_seconds=duration, request_count=self._request_count,
            quality=self.quality, size=self.size, output_format=self.output_format,
            retry_count=retry_count, last_error_type=last_error_type,
            warning=warning, error=error, metadata=details,
        )

    def _validate_inputs(self, source: Path, prompt: str, output: Path, mask: Path | None) -> str | None:
        if not prompt.strip():
            return "Prompt must not be empty."
        for label, path in (("source", source), ("mask", mask)):
            if path is None:
                continue
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
        if output == source or (mask is not None and output == mask):
            return "Output path must differ from input images."
        for protected in (PROJECT_ROOT / "input", PROJECT_ROOT / "regression_subset"):
            try:
                output.resolve().relative_to(protected.resolve())
                return "Output path must not be inside protected input directories."
            except ValueError:
                pass
        return None

    def _client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        if self._http_client_factory is not None:
            self._http_client = self._http_client_factory()
        else:
            import httpx

            self._http_client = httpx.Client()
        return self._http_client

    @staticmethod
    def _http_status(status_code: int) -> tuple[str, bool]:
        mapping = {
            400: "invalid_request", 401: "authentication_error",
            403: "permission_error", 404: "deployment_or_endpoint_not_found",
            408: "timeout", 429: "rate_limited",
        }
        if status_code in mapping:
            return mapping[status_code], status_code in {408, 429}
        if 500 <= status_code <= 599:
            return "service_error", True
        return "service_error", False

    @staticmethod
    def _exception_status(exc: BaseException) -> tuple[str, bool]:
        name = type(exc).__name__.lower()
        if isinstance(exc, TimeoutError) or "timeout" in name:
            return "timeout", True
        if "connection" in name or "connect" in name:
            return "connection_error", True
        return "connection_error", False

    @staticmethod
    def _mime(path: Path) -> str:
        return {".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")

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
        started = perf_counter()
        source_text, reference_text = str(source_image_path), str(reference_image_path)
        raw_mask = mask_path or (metadata or {}).get("mask_path")
        mask = Path(raw_mask).expanduser().resolve() if raw_mask else None
        mask_used = mask is not None

        authorized = (
            self.enabled and execute_api is True
            and self._is_true(self._env("ENABLE_OPENAI_IMAGE_API"))
            and self._is_true(self._env("CONFIRM_OPENAI_IMAGE_API_EXECUTION"))
        )
        if not authorized:
            return self._result(started, status="disabled", source=source_text, reference=reference_text, mask_used=mask_used, warning=DISABLED_MESSAGE, error=DISABLED_MESSAGE, metadata=metadata)

        try:
            endpoint = self.edit_endpoint
        except ValueError as exc:
            return self._result(started, status="invalid_configuration", source=source_text, reference=reference_text, mask_used=mask_used, error=str(exc), last_error_type=type(exc).__name__, metadata=metadata)
        if self.auth_mode not in {"bearer", "api_key"}:
            return self._result(started, status="invalid_configuration", source=source_text, reference=reference_text, mask_used=mask_used, error="AZURE_IMAGE_AUTH_MODE must be bearer or api_key.", metadata=metadata)
        api_key = self._env("AZURE_OPENAI_API_KEY")
        if not self._valid_api_key(api_key):
            return self._result(started, status="not_configured", source=source_text, reference=reference_text, mask_used=mask_used, warning=MALFORMED_KEY_MESSAGE, error=MALFORMED_KEY_MESSAGE, metadata=metadata)

        source = Path(source_image_path).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        invalid = self._validate_inputs(source, prompt, output, mask)
        if invalid:
            return self._result(started, status="invalid_input", source=source_text, reference=reference_text, mask_used=mask_used, error=invalid, metadata=metadata)
        if self._request_count >= self.max_requests_per_run:
            return self._result(started, status="invalid_request", source=source_text, reference=reference_text, mask_used=mask_used, error="Azure image request budget exhausted.", metadata=metadata)

        headers = {"Authorization": f"Bearer {api_key}"} if self.auth_mode == "bearer" else {"api-key": api_key}
        retry_count = 0
        last_error_type: str | None = None
        for attempt in range(self.max_retries + 1):
            if self._request_count >= self.max_requests_per_run:
                return self._result(started, status="invalid_request", source=source_text, reference=reference_text, mask_used=mask_used, error="Azure image request budget exhausted.", retry_count=retry_count, last_error_type=last_error_type, metadata=metadata)
            try:
                with source.open("rb") as source_file:
                    files: dict[str, Any] = {"image": (source.name, source_file, self._mime(source))}
                    if mask is not None:
                        with mask.open("rb") as mask_file:
                            files["mask"] = (mask.name, mask_file, self._mime(mask))
                            self._request_count += 1
                            response = self._client().post(endpoint, headers=headers, data={"prompt": prompt}, files=files, timeout=self.timeout_seconds)
                    else:
                        self._request_count += 1
                        response = self._client().post(endpoint, headers=headers, data={"prompt": prompt}, files=files, timeout=self.timeout_seconds)
                status_code = int(getattr(response, "status_code", 0))
                if not 200 <= status_code <= 299:
                    status, retryable = self._http_status(status_code)
                    last_error_type = f"HTTP{status_code}"
                    can_retry = retryable and attempt < self.max_retries and self._request_count < self.max_requests_per_run
                    if can_retry:
                        retry_count += 1
                        self._sleep(2**retry_count)
                        continue
                    return self._result(started, status=status, source=source_text, reference=reference_text, mask_used=mask_used, error=f"Azure image request failed with HTTP {status_code}.", retry_count=retry_count, last_error_type=last_error_type, metadata=metadata)
                try:
                    payload = response.json()
                except Exception:
                    try:
                        payload = json.loads(response.text)
                    except Exception:
                        payload = None
                data = payload.get("data") if isinstance(payload, dict) else None
                encoded = data[0].get("b64_json") if isinstance(data, list) and data and isinstance(data[0], dict) else None
                if not isinstance(encoded, str) or not encoded:
                    return self._result(started, status="invalid_response", source=source_text, reference=reference_text, mask_used=mask_used, error="Azure image response did not contain data[0].b64_json.", retry_count=retry_count, metadata=metadata)
                try:
                    image_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError):
                    return self._result(started, status="invalid_response", source=source_text, reference=reference_text, mask_used=mask_used, error="Azure image response contained invalid base64.", retry_count=retry_count, metadata=metadata)
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        width, height = image.size
                        image_format = (image.format or "").upper()
                        image.verify()
                    if width <= 0 or height <= 0 or image_format not in SUPPORTED_INPUT_FORMATS:
                        raise UnidentifiedImageError("Invalid image properties")
                except (OSError, UnidentifiedImageError):
                    return self._result(started, status="output_validation_failed", source=source_text, reference=reference_text, mask_used=mask_used, error="Decoded Azure output is not a valid image.", retry_count=retry_count, metadata=metadata)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(image_bytes)
                try:
                    if output.stat().st_size <= 0:
                        raise OSError("Empty output")
                    with Image.open(output) as written:
                        if written.width <= 0 or written.height <= 0:
                            raise OSError("Invalid dimensions")
                        written.verify()
                except (OSError, UnidentifiedImageError):
                    if output.is_file():
                        output.unlink()
                    return self._result(started, status="output_validation_failed", source=source_text, reference=reference_text, mask_used=mask_used, error="Written Azure output failed validation.", retry_count=retry_count, metadata=metadata)
                return self._result(started, status="success", source=source_text, reference=reference_text, mask_used=mask_used, output_path=str(output), retry_count=retry_count, metadata=metadata)
            except Exception as exc:
                status, retryable = self._exception_status(exc)
                last_error_type = type(exc).__name__
                can_retry = retryable and attempt < self.max_retries and self._request_count < self.max_requests_per_run
                if can_retry:
                    retry_count += 1
                    self._sleep(2**retry_count)
                    continue
                return self._result(started, status=status, source=source_text, reference=reference_text, mask_used=mask_used, error=f"{type(exc).__name__}: Azure image request failed.", retry_count=retry_count, last_error_type=last_error_type, metadata=metadata)
        return self._result(started, status="service_error", source=source_text, reference=reference_text, mask_used=mask_used, error="Azure image request failed.", retry_count=retry_count, last_error_type=last_error_type, metadata=metadata)
