import pytest

from utils.openai_image_provider import OpenAIImageProvider
from utils.step_image_generator import MockStepImageProvider
from utils.step_image_provider_factory import create_step_image_provider


def test_factory_defaults_to_mock():
    assert isinstance(create_step_image_provider(), MockStepImageProvider)


def test_factory_creates_only_disabled_openai_stub():
    provider = create_step_image_provider("openai")
    assert isinstance(provider, OpenAIImageProvider)
    assert provider.enabled is False


def test_explicit_flag_only_arms_provider_without_executing():
    provider = create_step_image_provider("openai", enable_external_api=True)
    assert isinstance(provider, OpenAIImageProvider)
    assert provider.enabled is True


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown step-image provider"):
        create_step_image_provider("other")
