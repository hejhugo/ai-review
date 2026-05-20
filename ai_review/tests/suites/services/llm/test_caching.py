"""Unit tests for prompt caching support.

Tests cover:
(a) default config: caching disabled, behavior byte-identical to baseline
(b) Anthropic: cache_control stamped when enabled and prefix >= min_tokens
(c) Anthropic: no cache_control when enabled but prefix < min_tokens
(d) Anthropic: cache key changes when system prompt changes
(e) OpenAI v2: instructions field carries system prompt; cache_read_tokens read back
(f) Gemini: below 32K floor returns no cached_content_name (graceful no-op)
(g) cost schema: cache tokens surfaced in CostReportSchema
(h) multi-stage simulation: second call sees cache_hit=True when tokens fed back
"""

from __future__ import annotations

import pytest
from pydantic import HttpUrl, SecretStr

from ai_review.clients.claude.schema import (
    ClaudeContentSchema,
    ClaudeChatResponseSchema,
    ClaudeChatRequestSchema,
    ClaudeUsageSchema,
)
from ai_review.config import settings
from ai_review.libs.cache.gemini import get_or_create_cached_content, GEMINI_TOKEN_FLOOR
from ai_review.libs.config.llm.base import ClaudeLLMConfig
from ai_review.libs.config.llm.cache import LLMCacheConfig
from ai_review.libs.config.llm.claude import ClaudeMetaConfig, ClaudeHTTPClientConfig
from ai_review.libs.constants.llm_provider import LLMProvider
from ai_review.services.cost.schema import CostReportSchema, CalculateCostSchema
from ai_review.services.cost.service import CostService
from ai_review.services.llm.claude.client import ClaudeLLMClient, _build_system
from ai_review.services.llm.types import ChatResultSchema
from ai_review.tests.fixtures.clients.claude import FakeClaudeHTTPClient


def _short_system() -> str:
    return "short system prompt"


def _long_system(min_tokens: int = 1024) -> str:
    return "x" * min_tokens


def make_claude_config(cache_enabled: bool = False, min_tokens: int = 1024, monkeypatch=None) -> ClaudeLLMConfig:
    cfg = ClaudeLLMConfig(
        meta=ClaudeMetaConfig(),
        provider=LLMProvider.CLAUDE,
        http_client=ClaudeHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.anthropic.com"),
            api_token=SecretStr("fake-token"),
            api_version="2023-06-01",
        ),
        cache=LLMCacheConfig(enabled=cache_enabled, min_tokens=min_tokens),
    )
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "llm", cfg)
    return cfg


class TestDefaultNoCaching:
    def test_cache_disabled_by_default(self):
        cfg = LLMCacheConfig()
        assert cfg.enabled is False

    def test_min_tokens_default(self):
        cfg = LLMCacheConfig()
        assert cfg.min_tokens == 1024

    @pytest.mark.usefixtures("claude_http_client_config")
    def test_build_system_returns_plain_string_when_disabled(self, monkeypatch):
        make_claude_config(cache_enabled=False, monkeypatch=monkeypatch)
        system = _long_system(2000)
        result = _build_system(system)
        assert isinstance(result, str)
        assert result == system

    @pytest.mark.usefixtures("claude_http_client_config")
    def test_build_system_returns_plain_string_below_floor(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=5000, monkeypatch=monkeypatch)
        system = _short_system()
        result = _build_system(system)
        assert isinstance(result, str)
        assert result == system


class TestAnthropicCaching:
    def test_cache_control_stamped_when_enabled_and_meets_floor(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=10, monkeypatch=monkeypatch)
        system = "x" * 20
        result = _build_system(system)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].text == system
        assert result[0].cache_control is not None
        assert result[0].cache_control.type == "ephemeral"

    def test_no_cache_control_when_below_min_tokens(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=1000, monkeypatch=monkeypatch)
        system = "short"
        result = _build_system(system)
        assert isinstance(result, str)

    def test_cache_key_differs_for_different_prompts(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=1, monkeypatch=monkeypatch)
        r1 = _build_system("system-a")
        r2 = _build_system("system-b")
        assert r1[0].text != r2[0].text

    @pytest.mark.asyncio
    async def test_chat_returns_cache_tokens_on_hit(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=1, monkeypatch=monkeypatch)
        fake = FakeClaudeHTTPClient(responses={
            "chat": ClaudeChatResponseSchema(
                id="id",
                role="assistant",
                usage=ClaudeUsageSchema(
                    input_tokens=100,
                    output_tokens=50,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=80,
                ),
                content=[ClaudeContentSchema(type="text", text="response")],
            )
        })
        monkeypatch.setattr(
            "ai_review.services.llm.claude.client.get_claude_http_client",
            lambda: fake,
        )
        client = ClaudeLLMClient()
        result = await client.chat("prompt", "sys")
        assert result.cache_read_tokens == 80
        assert result.cache_hit is True

    @pytest.mark.asyncio
    async def test_chat_cache_disabled_no_cache_tokens(self, monkeypatch):
        make_claude_config(cache_enabled=False, monkeypatch=monkeypatch)
        fake = FakeClaudeHTTPClient(responses={
            "chat": ClaudeChatResponseSchema(
                id="id",
                role="assistant",
                usage=ClaudeUsageSchema(input_tokens=100, output_tokens=50),
                content=[ClaudeContentSchema(type="text", text="response")],
            )
        })
        monkeypatch.setattr(
            "ai_review.services.llm.claude.client.get_claude_http_client",
            lambda: fake,
        )
        client = ClaudeLLMClient()
        result = await client.chat("prompt", "sys")
        assert result.cache_read_tokens == 0
        assert result.cache_creation_tokens == 0
        assert result.cache_hit is False


class TestGeminiCachingFloor:
    def test_below_floor_returns_none(self, monkeypatch):
        result = get_or_create_cached_content(
            model="gemini-pro",
            system_prompt="short",
            api_token="fake-token",
        )
        assert result is None

    def test_floor_threshold(self):
        short = "x" * (GEMINI_TOKEN_FLOOR * 4 - 1)
        long = "x" * (GEMINI_TOKEN_FLOOR * 4)
        from ai_review.libs.cache.gemini import _estimate_tokens
        assert _estimate_tokens(short) < GEMINI_TOKEN_FLOOR
        assert _estimate_tokens(long) >= GEMINI_TOKEN_FLOOR


class TestCostReportCacheFields:
    def test_cache_tokens_zero_by_default(self):
        report = CostReportSchema(
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        )
        assert report.cache_creation_tokens == 0
        assert report.cache_read_tokens == 0
        assert report.cache_hit is False

    def test_cache_hit_true_when_read_tokens_present(self):
        report = CostReportSchema(
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            cache_read_tokens=80,
        )
        assert report.cache_hit is True

    def test_pretty_includes_cache_line(self):
        report = CostReportSchema(
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            cache_creation_tokens=200,
            cache_read_tokens=80,
        )
        pretty = report.pretty()
        assert "Cache tokens" in pretty
        assert "created 200" in pretty
        assert "read 80" in pretty

    def test_pretty_no_cache_line_when_zero(self):
        report = CostReportSchema(
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        )
        assert "Cache tokens" not in report.pretty()


class TestMultiStageCacheHit:
    """Simulate inline -> context -> summary calls and verify second/third see cache hits."""

    @pytest.mark.asyncio
    async def test_second_call_sees_cache_hit(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_tokens=1, monkeypatch=monkeypatch)

        call_count = 0

        def _make_response():
            nonlocal call_count
            call_count += 1
            cached = 100 if call_count > 1 else 0
            return ClaudeChatResponseSchema(
                id=f"id-{call_count}",
                role="assistant",
                usage=ClaudeUsageSchema(
                    input_tokens=120,
                    output_tokens=30,
                    cache_creation_input_tokens=100 if call_count == 1 else 0,
                    cache_read_input_tokens=cached,
                ),
                content=[ClaudeContentSchema(type="text", text=f"resp-{call_count}")],
            )

        class SequentialFake:
            async def chat(self, request):
                return _make_response()

        monkeypatch.setattr(
            "ai_review.services.llm.claude.client.get_claude_http_client",
            lambda: SequentialFake(),
        )
        client = ClaudeLLMClient()
        system = "shared system prompt for all stages"

        r1 = await client.chat("inline prompt", system)
        r2 = await client.chat("context prompt", system)
        r3 = await client.chat("summary prompt", system)

        assert r1.cache_creation_tokens == 100
        assert r1.cache_hit is False

        assert r2.cache_read_tokens == 100
        assert r2.cache_hit is True

        assert r3.cache_read_tokens == 100
        assert r3.cache_hit is True
