"""Claude Platform on AWS must price via Bedrock anthropic.* cost-map keys (#48045)."""

from __future__ import annotations

import litellm
from litellm.llms.bedrock.common_utils import BedrockModelInfo, strip_bedrock_routing_prefix


def test_strip_bedrock_routing_prefix_removes_claude_platform():
    assert strip_bedrock_routing_prefix("claude_platform/claude-opus-5-5") == "claude-opus-5-5"
    assert (
        strip_bedrock_routing_prefix("bedrock/claude_platform/claude-opus-5-5")
        == "claude-opus-5-5"
    )


def test_bedrock_model_info_maps_claude_platform_api_id_to_cost_key():
    info = BedrockModelInfo()
    assert info.get_model_cost_key("claude_platform/claude-opus-5-5") == "anthropic.claude-opus-5-5"
    assert info.get_model_cost_key("claude-opus-5-5") == "anthropic.claude-opus-5-5"
    # already a Bedrock id — no remapping needed / not a Claude API id
    assert info.get_model_cost_key("anthropic.claude-opus-5-5") is None


def test_claude_platform_cost_per_token_matches_bedrock_anthropic_id():
    expected = litellm.cost_per_token(
        model="anthropic.claude-opus-5-5",
        custom_llm_provider="bedrock",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert expected == (4.0, 20.0)

    for model in (
        "bedrock/claude_platform/claude-opus-5-5",
        "claude_platform/claude-opus-5-5",
    ):
        got = litellm.cost_per_token(
            model=model,
            custom_llm_provider="bedrock",
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        assert got == expected, (model, got, expected)
