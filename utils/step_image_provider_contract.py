"""Provider contract for one correction-SOP instruction image."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class StepImageResult:
    success: bool
    output_path: str | None
    provider: str
    duration: float = 0.0
    warning: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    model: str | None = None
    mode: str | None = None
    duration_seconds: float = 0.0
    request_count: int = 0
    quality: str | None = None
    size: str | None = None
    output_format: str | None = None
    retry_count: int = 0
    last_error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepImageProvider(Protocol):
    name: str

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
        """Generate exactly one step image and return a structured result."""
