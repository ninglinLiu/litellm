"""Regression: Vertex AI rerank must reuse one config so credentials stay cached."""

import litellm
from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_reuses_vertex_ai_rerank_config_instance():
    ProviderConfigManager._get_vertex_ai_rerank_config.cache_clear()
    try:
        first = ProviderConfigManager.get_provider_rerank_config(
            model="semantic-ranker-default@latest",
            provider=litellm.LlmProviders.VERTEX_AI,
            api_base=None,
            present_version_params=[],
        )
        second = ProviderConfigManager.get_provider_rerank_config(
            model="semantic-ranker-default-004",
            provider=litellm.LlmProviders.VERTEX_AI,
            api_base=None,
            present_version_params=[],
        )
        assert isinstance(first, VertexAIRerankConfig)
        assert first is second
        assert hasattr(first, "_credentials_project_mapping")
        first._credentials_project_mapping["probe"] = ("tok", "proj")  # type: ignore[index]
        assert "probe" in second._credentials_project_mapping
    finally:
        ProviderConfigManager._get_vertex_ai_rerank_config.cache_clear()
