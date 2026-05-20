"""Unit tests for prompt caching support.

Covers:
(a) Default config: caching disabled, byte-identical request shape across providers
(b) Anthropic: cache_control stamped when enabled and prefix >= min_chars; cache tokens flow back
(c) OpenAI v2: instructions field used when enabled; system role kept in input when disabled
(d) Gemini: below 32K floor returns no cached_content_name; in-process map; model-bound key
(e) CostService: cache tokens priced via dedicated rates (no $0 leak), with default = input rate
(f) Multi-stage simulation: second call sees cache_hit=True
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from ai_review.clients.claude.schema import (
    ClaudeContentSchema,
    ClaudeChatResponseSchema,
    ClaudeUsageSchema,
)
from ai_review.clients.openai.v2.schema import (
    OpenAIInputTokensDetailsSchema,
    OpenAIResponseContentSchema,
    OpenAIResponseOutputSchema,
    OpenAIResponseUsageSchema,
    OpenAIResponsesRequestSchema,
    OpenAIResponsesResponseSchema,
)
from ai_review.config import settings
from ai_review.libs.cache.gemini import (
    GEMINI_TOKEN_FLOOR,
    _cache_key,
    _estimate_tokens,
    _reset_cache_state,
    get_or_create_cached_content,
)
from ai_review.libs.config.llm.base import ClaudeLLMConfig, LLMPricingConfig, OpenAILLMConfig
from ai_review.libs.config.llm.cache import LLMCacheConfig
from ai_review.libs.config.llm.claude import ClaudeMetaConfig, ClaudeHTTPClientConfig
from ai_review.libs.config.llm.openai import OpenAIMetaConfig, OpenAIHTTPClientConfig
from ai_review.libs.constants.llm_provider import LLMProvider
from ai_review.services.cost.schema import CalculateCostSchema, CostReportSchema
from ai_review.services.cost.service import CostService
from ai_review.services.llm.claude.client import ClaudeLLMClient, _build_system
from ai_review.services.llm.openai.client import OpenAILLMClient
from ai_review.tests.fixtures.clients.claude import FakeClaudeHTTPClient

GEMINI_FLOOR_CHARS = GEMINI_TOKEN_FLOOR * 4


def make_claude_config(cache_enabled: bool = False, min_chars: int = 4096, monkeypatch=None) -> ClaudeLLMConfig:
    cfg = ClaudeLLMConfig(
        meta=ClaudeMetaConfig(),
        provider=LLMProvider.CLAUDE,
        http_client=ClaudeHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.anthropic.com"),
            api_token=SecretStr("fake-token"),
            api_version="2023-06-01",
        ),
        cache=LLMCacheConfig(enabled=cache_enabled, min_chars=min_chars),
    )
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "llm", cfg)
    return cfg


def make_openai_v2_config(cache_enabled: bool = False, min_chars: int = 4096, monkeypatch=None) -> OpenAILLMConfig:
    cfg = OpenAILLMConfig(
        meta=OpenAIMetaConfig(model="gpt-5"),
        provider=LLMProvider.OPENAI,
        http_client=OpenAIHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.openai.com"),
            api_token=SecretStr("fake-token"),
        ),
        cache=LLMCacheConfig(enabled=cache_enabled, min_chars=min_chars),
    )
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "llm", cfg)
    return cfg


class _FakeOpenAIV2Client:
    def __init__(self, usage_kwargs: dict[str, Any] | None = None) -> None:
        self.requests: list[OpenAIResponsesRequestSchema] = []
        self.usage_kwargs = usage_kwargs or {"total_tokens": 200, "input_tokens": 150, "output_tokens": 50}

    async def chat(self, request: OpenAIResponsesRequestSchema) -> OpenAIResponsesResponseSchema:
        self.requests.append(request)
        usage = OpenAIResponseUsageSchema(**self.usage_kwargs)
        return OpenAIResponsesResponseSchema(
            usage=usage,
            output=[
                OpenAIResponseOutputSchema(
                    type="message",
                    role="assistant",
                    content=[OpenAIResponseContentSchema(type="output_text", text="resp")],
                )
            ],
        )


class TestDefaultsAreOptIn:
    def test_cache_disabled_by_default(self):
        assert LLMCacheConfig().enabled is False

    def test_min_chars_default(self):
        assert LLMCacheConfig().min_chars == 4096

    @pytest.mark.usefixtures("claude_http_client_config")
    def test_claude_build_system_returns_plain_string_when_disabled(self, monkeypatch):
        make_claude_config(cache_enabled=False, monkeypatch=monkeypatch)
        system = "x" * 8000
        result = _build_system(system)
        assert isinstance(result, str)
        assert result == system

    @pytest.mark.usefixtures("claude_http_client_config")
    def test_claude_build_system_returns_plain_string_below_floor(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_chars=5000, monkeypatch=monkeypatch)
        result = _build_system("short")
        assert isinstance(result, str)


class TestAnthropicCaching:
    def test_cache_control_stamped_when_enabled_and_meets_floor(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_chars=10, monkeypatch=monkeypatch)
        result = _build_system("x" * 20)
        assert isinstance(result, list)
        assert result[0].cache_control is not None
        assert result[0].cache_control.type == "ephemeral"

    def test_no_cache_control_when_below_min_chars(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_chars=1000, monkeypatch=monkeypatch)
        result = _build_system("short")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_chat_returns_cache_tokens_on_hit(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_chars=1, monkeypatch=monkeypatch)
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


class TestOpenAIDefaultOffParity:
    """OpenAI v2 must keep the system role in `input` when caching is disabled."""

    @pytest.mark.asyncio
    async def test_default_off_keeps_system_in_input_array(self, monkeypatch):
        make_openai_v2_config(cache_enabled=False, monkeypatch=monkeypatch)
        fake = _FakeOpenAIV2Client()
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        await client.chat("user prompt", "system prompt")

        request = fake.requests[0]
        assert request.instructions is None
        assert len(request.input) == 2
        assert request.input[0].role == "system"
        assert request.input[0].content == "system prompt"
        assert request.input[1].role == "user"

    @pytest.mark.asyncio
    async def test_enabled_below_floor_keeps_system_in_input_array(self, monkeypatch):
        make_openai_v2_config(cache_enabled=True, min_chars=1_000_000, monkeypatch=monkeypatch)
        fake = _FakeOpenAIV2Client()
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        await client.chat("user prompt", "short system")

        request = fake.requests[0]
        assert request.instructions is None
        assert request.input[0].role == "system"

    @pytest.mark.asyncio
    async def test_enabled_above_floor_uses_instructions_field(self, monkeypatch):
        make_openai_v2_config(cache_enabled=True, min_chars=10, monkeypatch=monkeypatch)
        fake = _FakeOpenAIV2Client(
            usage_kwargs={
                "total_tokens": 250,
                "input_tokens": 200,
                "output_tokens": 50,
                "input_tokens_details": OpenAIInputTokensDetailsSchema(cached_tokens=120),
            }
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        result = await client.chat("user prompt", "x" * 100)

        request = fake.requests[0]
        assert request.instructions == "x" * 100
        assert len(request.input) == 1
        assert request.input[0].role == "user"
        # OpenAI input_tokens includes cached; client must subtract before returning.
        assert result.prompt_tokens == 80
        assert result.cache_read_tokens == 120


class TestGeminiCacheLib:
    def setup_method(self) -> None:
        _reset_cache_state()

    def test_cache_key_includes_model(self):
        a = _cache_key("gemini-2.5-pro", "system text")
        b = _cache_key("gemini-2.5-flash", "system text")
        assert a != b

    def test_estimate_tokens(self):
        assert _estimate_tokens("x" * 100) == 25
        assert _estimate_tokens("") == 0

    @pytest.mark.asyncio
    async def test_below_floor_returns_none(self):
        async with httpx.AsyncClient() as client:
            name, creation = await get_or_create_cached_content(
                client=client,
                model="gemini-2.5-pro",
                system_prompt="short",
            )
        assert name is None
        assert creation == 0

    @pytest.mark.asyncio
    async def test_create_then_reuse_from_in_process_map(self):
        prompt = "x" * GEMINI_FLOOR_CHARS
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                json={"name": "cachedContents/abc123", "usageMetadata": {"totalTokenCount": 33000}},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://example.test") as client:
            name1, creation1 = await get_or_create_cached_content(client, "gemini-2.5-pro", prompt)
            name2, creation2 = await get_or_create_cached_content(client, "gemini-2.5-pro", prompt)

        assert len(calls) == 1
        assert name1 == "cachedContents/abc123"
        assert creation1 == 33000
        assert name2 == "cachedContents/abc123"
        assert creation2 == 0

    @pytest.mark.asyncio
    async def test_model_mismatch_creates_separate_entry(self):
        prompt = "x" * GEMINI_FLOOR_CHARS
        responses_iter = iter([
            {"name": "cachedContents/pro", "usageMetadata": {"totalTokenCount": 33000}},
            {"name": "cachedContents/flash", "usageMetadata": {"totalTokenCount": 33000}},
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(responses_iter))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://example.test") as client:
            name_pro, _ = await get_or_create_cached_content(client, "gemini-2.5-pro", prompt)
            name_flash, _ = await get_or_create_cached_content(client, "gemini-2.5-flash", prompt)

        assert name_pro == "cachedContents/pro"
        assert name_flash == "cachedContents/flash"

    @pytest.mark.asyncio
    async def test_http_error_returns_none_no_cache_stored(self):
        prompt = "x" * GEMINI_FLOOR_CHARS

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "model not supported"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://example.test") as client:
            name, creation = await get_or_create_cached_content(client, "gemini-2.5-pro", prompt)
        assert name is None
        assert creation == 0


class TestGeminiCacheCreateBody:
    """Verify create body does not include warm-up user content."""

    @pytest.mark.asyncio
    async def test_create_body_has_no_dummy_contents(self):
        prompt = "x" * GEMINI_FLOOR_CHARS
        _reset_cache_state()
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json
            captured.append(_json.loads(request.content))
            return httpx.Response(
                200,
                json={"name": "cachedContents/abc", "usageMetadata": {"totalTokenCount": 33000}},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport, base_url="https://example.test") as client:
            await get_or_create_cached_content(client, "gemini-2.5-pro", prompt)

        assert len(captured) == 1
        body = captured[0]
        assert "contents" not in body
        assert "systemInstruction" in body


class TestCostServiceCacheMath:
    """CostService must price cache tokens at non-zero rates."""

    def _calculate(self, monkeypatch, pricing: LLMPricingConfig, **calc) -> CostReportSchema | None:
        cfg = make_claude_config(monkeypatch=monkeypatch)
        monkeypatch.setattr(
            ClaudeLLMConfig,
            "load_pricing",
            lambda self: {cfg.meta.model: pricing},
        )
        service = CostService()
        return service.calculate(CalculateCostSchema(**calc))

    def test_cache_tokens_priced_at_input_rate_by_default(self, monkeypatch):
        pricing = LLMPricingConfig(input=1.0e-6, output=2.0e-6)
        report = self._calculate(
            monkeypatch,
            pricing,
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_tokens=200,
            cache_read_tokens=300,
        )
        assert report is not None
        # base 100 + creation 200 + read 300 = 600 input-equivalent tokens at 1e-6
        assert report.input_cost == pytest.approx(600 * 1.0e-6)
        assert report.output_cost == pytest.approx(50 * 2.0e-6)

    def test_cache_creation_uses_dedicated_rate_when_set(self, monkeypatch):
        pricing = LLMPricingConfig(
            input=1.0e-6,
            output=2.0e-6,
            cache_creation_input=1.25e-6,
            cache_read_input=0.1e-6,
        )
        report = self._calculate(
            monkeypatch,
            pricing,
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_tokens=200,
            cache_read_tokens=300,
        )
        assert report is not None
        expected_input = (100 * 1.0e-6) + (200 * 1.25e-6) + (300 * 0.1e-6)
        assert report.input_cost == pytest.approx(expected_input)


class TestMultiStageCacheHit:
    @pytest.mark.asyncio
    async def test_second_call_sees_cache_hit(self, monkeypatch):
        make_claude_config(cache_enabled=True, min_chars=1, monkeypatch=monkeypatch)

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
