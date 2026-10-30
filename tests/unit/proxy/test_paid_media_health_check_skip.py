"""Paid-media health-check modes are skipped unless the operator opts in (#49089)."""

from litellm.proxy.health_check import (
    PAID_MEDIA_HEALTH_CHECK_MODES,
    _health_check_eligible,
    _skips_paid_media_health_checks,
    health_check_filter_kwargs_from_general_settings,
    is_paid_media_health_check_mode,
)


def test_paid_media_modes_constant_covers_synthesis_modes():
    assert "image_generation" in PAID_MEDIA_HEALTH_CHECK_MODES
    assert "image_edit" in PAID_MEDIA_HEALTH_CHECK_MODES
    assert "video_generation" in PAID_MEDIA_HEALTH_CHECK_MODES
    assert "audio_speech" in PAID_MEDIA_HEALTH_CHECK_MODES
    assert "chat" not in PAID_MEDIA_HEALTH_CHECK_MODES
    assert "embedding" not in PAID_MEDIA_HEALTH_CHECK_MODES


def test_is_paid_media_health_check_mode():
    assert is_paid_media_health_check_mode("image_generation") is True
    assert is_paid_media_health_check_mode("chat") is False
    assert is_paid_media_health_check_mode(None) is False


def test_skips_paid_media_by_default():
    deployment = {
        "model_name": "gemini-image",
        "litellm_params": {"model": "gemini/gemini-3-pro-image"},
        "model_info": {"mode": "image_generation"},
    }
    assert _skips_paid_media_health_checks(deployment, allow_paid_media=False) is True
    assert _skips_paid_media_health_checks(deployment, allow_paid_media=True) is False


def test_per_deployment_opt_in_overrides_global_default():
    deployment = {
        "model_name": "gemini-image",
        "litellm_params": {"model": "gemini/gemini-3-pro-image"},
        "model_info": {"mode": "image_generation", "health_check_paid_media": True},
    }
    assert _skips_paid_media_health_checks(deployment, allow_paid_media=False) is False


def test_chat_deployments_are_never_skipped_as_paid_media():
    deployment = {
        "model_name": "gpt",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"mode": "chat"},
    }
    assert _skips_paid_media_health_checks(deployment, allow_paid_media=False) is False


def test_health_check_eligible_drops_paid_media_unless_allowed():
    chat = {
        "model_name": "gpt",
        "litellm_params": {"model": "gpt-4o"},
        "model_info": {"id": "chat-1", "mode": "chat"},
    }
    image = {
        "model_name": "img",
        "litellm_params": {"model": "dall-e-3"},
        "model_info": {"id": "img-1", "mode": "image_generation"},
    }
    eligible_default = _health_check_eligible([chat, image], skip_disabled=False)
    assert [d["model_info"]["id"] for d in eligible_default] == ["chat-1"]

    eligible_allowed = _health_check_eligible(
        [chat, image], skip_disabled=False, allow_paid_media=True
    )
    assert [d["model_info"]["id"] for d in eligible_allowed] == ["chat-1", "img-1"]


def test_filter_kwargs_include_paid_media_flag():
    assert health_check_filter_kwargs_from_general_settings(None) == {
        "health_check_skip_disabled_background_models": False,
        "health_check_paid_media": False,
    }
    assert health_check_filter_kwargs_from_general_settings(
        {"health_check_paid_media": True}
    )["health_check_paid_media"] is True
