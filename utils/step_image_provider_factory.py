"""Explicit factory for offline and guarded external step-image providers."""

from __future__ import annotations

from utils.openai_image_provider import OpenAIImageProvider
from utils.step_image_generator import MockStepImageProvider
from utils.step_image_provider_contract import StepImageProvider


def create_step_image_provider(
    provider_name: str = "mock",
    *,
    enable_external_api: bool = False,
    **provider_options,
) -> StepImageProvider:
    """Create a provider without reading secrets or making a request."""
    normalized = str(provider_name).strip().lower()
    if normalized == "mock":
        return MockStepImageProvider()
    if normalized == "openai":
        return OpenAIImageProvider(enabled=enable_external_api, **provider_options)
    raise ValueError(f"Unknown step-image provider: {provider_name!r}. Expected 'mock' or 'openai'.")
