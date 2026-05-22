"""Split-stamp prompt caching: cache the stable user-provided prefix, leave the
per-stage default uncached.

Covers:
- PromptService inserts the boundary between user files (cacheable) and the
  per-stage default (variable).
- Claude `_build_system` emits a two-block system with cache_control on the
  prefix block only.
- OpenAI `chat_v2` routes the cached prefix into `instructions` and keeps the
  variable suffix as a per-call system message in `input`.
- Gemini caches only the prefix and sends the variable suffix as the per-call
  system_instruction.
- Non-caching clients (Ollama, Bedrock, OpenRouter, Azure OpenAI) strip the
  boundary before sending.
- Direct gateway hands hooks and artifacts a boundary-free prompt.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import HttpUrl, SecretStr

from ai_review.clients.bedrock.schema import BedrockChatRequestSchema
from ai_review.clients.claude.schema import (
    ClaudeContentSchema,
    ClaudeChatResponseSchema,
    ClaudeUsageSchema,
)
from ai_review.clients.gemini.schema import GeminiChatRequestSchema
from ai_review.clients.ollama.schema import OllamaChatRequestSchema
from ai_review.clients.openai.v1.schema import OpenAIChatRequestSchema
from ai_review.clients.openai.v2.schema import (
    OpenAIInputTokensDetailsSchema,
    OpenAIResponseContentSchema,
    OpenAIResponseOutputSchema,
    OpenAIResponseUsageSchema,
    OpenAIResponsesRequestSchema,
    OpenAIResponsesResponseSchema,
)
from ai_review.clients.openrouter.schema import OpenRouterChatRequestSchema
from ai_review.clients.azure_openai.schema import AzureOpenAIChatRequestSchema
from ai_review.config import settings
from ai_review.libs.cache.gemini import _reset_cache_state
from ai_review.libs.config.llm.base import ClaudeLLMConfig, OpenAILLMConfig
from ai_review.libs.config.llm.cache import LLMCacheConfig
from ai_review.libs.config.llm.claude import ClaudeMetaConfig, ClaudeHTTPClientConfig
from ai_review.libs.config.llm.openai import OpenAIMetaConfig, OpenAIHTTPClientConfig
from ai_review.libs.config.prompt import PromptConfig
from ai_review.libs.constants.llm_provider import LLMProvider
from ai_review.services.llm.azure_openai.client import AzureOpenAILLMClient
from ai_review.services.llm.bedrock.client import BedrockLLMClient
from ai_review.services.llm.claude.client import ClaudeLLMClient, _build_system
from ai_review.services.llm.gemini.client import GeminiLLMClient
from ai_review.services.llm.ollama.client import OllamaLLMClient
from ai_review.services.llm.openai.client import OpenAILLMClient
from ai_review.services.llm.openrouter.client import OpenRouterLLMClient
from ai_review.services.prompt.schema import (
    SYSTEM_PROMPT_CACHE_BOUNDARY,
    PromptContextSchema,
    split_system_prompt,
    strip_system_prompt_boundary,
)
from ai_review.services.prompt.service import PromptService


class _RecordingFake:
    """Captures requests sent to a fake HTTP client."""

    def __init__(self, response: Any) -> None:
        self.requests: list[Any] = []
        self.response = response

    async def chat(self, request: Any) -> Any:
        self.requests.append(request)
        return self.response


def _openai_v2_response(input_tokens: int = 100, cached_tokens: int = 0) -> OpenAIResponsesResponseSchema:
    return OpenAIResponsesResponseSchema(
        id="resp_x",
        usage=OpenAIResponseUsageSchema(
            total_tokens=input_tokens + 10,
            input_tokens=input_tokens,
            output_tokens=10,
            input_tokens_details=OpenAIInputTokensDetailsSchema(cached_tokens=cached_tokens),
        ),
        output=[OpenAIResponseOutputSchema(
            type="message",
            role="assistant",
            content=[OpenAIResponseContentSchema(text="ok", type="output_text")],
        )],
    )


def _claude_response() -> ClaudeChatResponseSchema:
    return ClaudeChatResponseSchema(
        id="id",
        role="assistant",
        usage=ClaudeUsageSchema(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=80,
            cache_read_input_tokens=0,
        ),
        content=[ClaudeContentSchema(type="text", text="ok")],
    )


def _make_claude_cfg(monkeypatch) -> None:
    cfg = ClaudeLLMConfig(
        meta=ClaudeMetaConfig(),
        provider=LLMProvider.CLAUDE,
        http_client=ClaudeHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.anthropic.com"),
            api_token=SecretStr("fake-token"),
            api_version="2023-06-01",
        ),
        cache=LLMCacheConfig(enabled=True, min_chars=10),
    )
    monkeypatch.setattr(settings, "llm", cfg)


def _make_openai_cfg(monkeypatch) -> None:
    cfg = OpenAILLMConfig(
        meta=OpenAIMetaConfig(model="gpt-5"),
        provider=LLMProvider.OPENAI,
        http_client=OpenAIHTTPClientConfig(
            timeout=10,
            api_url=HttpUrl("https://api.openai.com"),
            api_token=SecretStr("fake-token"),
        ),
        cache=LLMCacheConfig(enabled=True, min_chars=10),
    )
    monkeypatch.setattr(settings, "llm", cfg)


# ---------- helpers ----------

class TestBoundaryHelpers:
    def test_split_no_boundary_returns_empty_prefix(self) -> None:
        assert split_system_prompt("no marker") == ("", "no marker")

    def test_split_with_boundary(self) -> None:
        prompt = f"PREFIX{SYSTEM_PROMPT_CACHE_BOUNDARY}VARIABLE"
        assert split_system_prompt(prompt) == ("PREFIX", "VARIABLE")

    def test_strip_restores_legacy_default_first_order(self) -> None:
        # Caching layout is `user_prefix || BOUNDARY || stage_default`. When
        # caching is off we strip the boundary AND swap back to v0.67.0's
        # `stage_default + user_prefix` ordering so disabled-mode byte
        # output matches upstream.
        prompt = f"USER{SYSTEM_PROMPT_CACHE_BOUNDARY}DEFAULT"
        result = strip_system_prompt_boundary(prompt)
        assert "AI_REVIEW_CACHE_BOUNDARY" not in result
        assert result == "DEFAULT\n\nUSER"

    def test_strip_passthrough_when_no_boundary(self) -> None:
        assert strip_system_prompt_boundary("plain") == "plain"

    def test_strip_with_one_side_empty(self) -> None:
        assert strip_system_prompt_boundary(f"{SYSTEM_PROMPT_CACHE_BOUNDARY}DEFAULT") == "DEFAULT"
        assert strip_system_prompt_boundary(f"USER{SYSTEM_PROMPT_CACHE_BOUNDARY}") == "USER"


# ---------- PromptService boundary emission ----------

@pytest.mark.usefixtures("fake_prompts")
class TestPromptServiceBoundary:
    def test_no_boundary_when_prefix_empty(self, fake_prompt_context: PromptContextSchema) -> None:
        # Default fake_prompts puts everything in variable -> no boundary expected.
        result = PromptService.build_system_inline_request(fake_prompt_context)
        assert SYSTEM_PROMPT_CACHE_BOUNDARY not in result

    def test_boundary_inserted_when_both_buckets_populated(
            self,
            monkeypatch: pytest.MonkeyPatch,
            fake_prompt_context: PromptContextSchema,
    ) -> None:
        monkeypatch.setattr(
            PromptConfig,
            "load_system_inline",
            lambda self: (["STABLE_DOCS"], ["PER_STAGE_INSTRUCTIONS"]),
        )
        result = PromptService.build_system_inline_request(fake_prompt_context)
        prefix, variable = split_system_prompt(result)
        assert prefix == "STABLE_DOCS"
        assert variable == "PER_STAGE_INSTRUCTIONS"

    def test_only_prefix_emits_no_boundary(
            self,
            monkeypatch: pytest.MonkeyPatch,
            fake_prompt_context: PromptContextSchema,
    ) -> None:
        monkeypatch.setattr(
            PromptConfig,
            "load_system_inline",
            lambda self: (["ONLY_PREFIX"], []),
        )
        result = PromptService.build_system_inline_request(fake_prompt_context)
        assert SYSTEM_PROMPT_CACHE_BOUNDARY not in result
        assert result == "ONLY_PREFIX"


# ---------- Claude split-stamp ----------

class TestClaudeSplitStamp:
    @pytest.mark.usefixtures("claude_http_client_config")
    def test_two_blocks_when_boundary_present(self, monkeypatch) -> None:
        _make_claude_cfg(monkeypatch)
        prompt = f"STABLE_PREFIX_OK{SYSTEM_PROMPT_CACHE_BOUNDARY}VARIABLE_TAIL"
        result = _build_system(prompt)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].text == "STABLE_PREFIX_OK"
        assert result[0].cache_control is not None
        assert result[0].cache_control.type == "ephemeral"
        # The variable block must NOT have cache_control: caching the variable
        # part would fragment the cache key per stage and defeat the purpose.
        assert result[1].text == "VARIABLE_TAIL"
        assert result[1].cache_control is None

    @pytest.mark.usefixtures("claude_http_client_config")
    def test_disabled_strips_boundary(self, monkeypatch) -> None:
        cfg = ClaudeLLMConfig(
            meta=ClaudeMetaConfig(),
            provider=LLMProvider.CLAUDE,
            http_client=ClaudeHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("https://api.anthropic.com"),
                api_token=SecretStr("fake-token"),
                api_version="2023-06-01",
            ),
            cache=LLMCacheConfig(enabled=False),
        )
        monkeypatch.setattr(settings, "llm", cfg)
        prompt = f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B"
        result = _build_system(prompt)
        assert isinstance(result, str)
        assert "AI_REVIEW_CACHE_BOUNDARY" not in result


# ---------- OpenAI split-stamp ----------

class TestOpenAISplitStamp:
    @pytest.mark.asyncio
    async def test_split_routes_prefix_to_instructions_and_variable_to_input_system(
            self, monkeypatch,
    ) -> None:
        _make_openai_cfg(monkeypatch)
        fake = _RecordingFake(_openai_v2_response(input_tokens=200, cached_tokens=120))
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        prompt_system = f"STABLE_DOCS_LONG_ENOUGH{SYSTEM_PROMPT_CACHE_BOUNDARY}PER_STAGE_VAR"
        result = await client.chat("user prompt", prompt_system)

        request: OpenAIResponsesRequestSchema = fake.requests[0]
        assert request.instructions == "STABLE_DOCS_LONG_ENOUGH"
        assert len(request.input) == 2
        assert request.input[0].role == "system"
        assert request.input[0].content == "PER_STAGE_VAR"
        assert request.input[1].role == "user"
        assert result.cache_read_tokens == 120

    @pytest.mark.asyncio
    async def test_disabled_strips_boundary_in_input_system(self, monkeypatch) -> None:
        cfg = OpenAILLMConfig(
            meta=OpenAIMetaConfig(model="gpt-5"),
            provider=LLMProvider.OPENAI,
            http_client=OpenAIHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("https://api.openai.com"),
                api_token=SecretStr("fake-token"),
            ),
            cache=LLMCacheConfig(enabled=False),
        )
        monkeypatch.setattr(settings, "llm", cfg)
        fake = _RecordingFake(_openai_v2_response())
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        await client.chat("user prompt", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: OpenAIResponsesRequestSchema = fake.requests[0]
        assert request.instructions is None
        # Disabled-mode strips the boundary AND restores legacy default-first
        # ordering (B comes before A).
        assert request.input[0].content == "B\n\nA"


# ---------- Gemini split-stamp ----------

class TestGeminiSplitStamp:
    def setup_method(self) -> None:
        _reset_cache_state()

    @pytest.mark.asyncio
    async def test_caches_prefix_only_passes_variable_per_call(self, monkeypatch) -> None:
        from ai_review.libs.cache.gemini import GEMINI_TOKEN_FLOOR

        prefix = "x" * (GEMINI_TOKEN_FLOOR * 4)
        prompt_system = f"{prefix}{SYSTEM_PROMPT_CACHE_BOUNDARY}PER_STAGE"

        # Patch the cache helper so we can observe what got cached.
        captured: dict[str, str] = {}

        async def fake_cache(client, model, system_prompt):
            captured["cached"] = system_prompt
            return "cachedContents/abc", 33000

        monkeypatch.setattr(
            "ai_review.services.llm.gemini.client.get_or_create_cached_content",
            fake_cache,
        )

        # Patch the gemini http client.
        from ai_review.clients.gemini.schema import (
            GeminiCandidateSchema,
            GeminiChatResponseSchema,
            GeminiContentSchema,
            GeminiPartSchema,
            GeminiUsageSchema,
        )

        response = GeminiChatResponseSchema(
            candidates=[GeminiCandidateSchema(
                content=GeminiContentSchema(parts=[GeminiPartSchema(text="ok")]),
            )],
            usage=GeminiUsageSchema(
                prompt_token_count=180,
                total_tokens_count=200,
                candidates_token_count=20,
                cached_content_token_count=170,
            ),
        )

        class _FakeGeminiHTTP:
            def __init__(self) -> None:
                self.client = object()
                self.requests: list[GeminiChatRequestSchema] = []

            async def chat(self, request: GeminiChatRequestSchema) -> Any:
                self.requests.append(request)
                return response

        fake = _FakeGeminiHTTP()
        monkeypatch.setattr(
            "ai_review.services.llm.gemini.client.get_gemini_http_client",
            lambda: fake,
        )

        # Set up cache config.
        from ai_review.libs.config.llm.base import GeminiLLMConfig
        from ai_review.libs.config.llm.gemini import GeminiMetaConfig, GeminiHTTPClientConfig

        cfg = GeminiLLMConfig(
            meta=GeminiMetaConfig(),
            provider=LLMProvider.GEMINI,
            http_client=GeminiHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("https://generativelanguage.googleapis.com"),
                api_token=SecretStr("fake-token"),
            ),
            cache=LLMCacheConfig(enabled=True, min_chars=10),
        )
        monkeypatch.setattr(settings, "llm", cfg)

        client = GeminiLLMClient()
        result = await client.chat("user prompt", prompt_system)

        # The cache helper was given ONLY the stable prefix, not the variable.
        assert captured["cached"] == prefix
        assert SYSTEM_PROMPT_CACHE_BOUNDARY not in captured["cached"]

        # Gemini rejects requests that combine `cachedContent` with a
        # `system_instruction`, so when the cache hits the per-stage variable
        # is folded into the user prompt and `system_instruction` is None.
        request: GeminiChatRequestSchema = fake.requests[0]
        assert request.cached_content == "cachedContents/abc"
        assert request.system_instruction is None
        assert request.contents[0].parts[0].text == "PER_STAGE\n\nuser prompt"
        assert result.cache_read_tokens == 170

    @pytest.mark.asyncio
    async def test_stale_cached_content_is_evicted_and_retried_uncached(self, monkeypatch) -> None:
        # When Gemini returns 404/400 for a `cachedContent` name we hold in
        # the in-process map (TTL expiry or out-of-band deletion), the
        # client must evict the dead entry and retry the same call without
        # the stale reference so the stage still produces output.
        from ai_review.clients.gemini.client import GeminiHTTPClientError
        from ai_review.clients.gemini.schema import (
            GeminiCandidateSchema,
            GeminiChatResponseSchema,
            GeminiContentSchema,
            GeminiPartSchema,
            GeminiUsageSchema,
        )
        from ai_review.libs.cache.gemini import (
            GEMINI_TOKEN_FLOOR,
            _cache_key,
            _cache_name_by_key,
        )
        from ai_review.libs.config.llm.base import GeminiLLMConfig
        from ai_review.libs.config.llm.gemini import GeminiMetaConfig, GeminiHTTPClientConfig

        prefix = "x" * (GEMINI_TOKEN_FLOOR * 4)
        prompt_system = f"{prefix}{SYSTEM_PROMPT_CACHE_BOUNDARY}PER_STAGE"

        # Pretend the cache helper still returns the (now stale) name.
        async def fake_cache(client, model, system_prompt):
            return "cachedContents/stale", 0

        monkeypatch.setattr(
            "ai_review.services.llm.gemini.client.get_or_create_cached_content",
            fake_cache,
        )

        # Seed the in-process map so we can assert eviction afterwards.
        _cache_name_by_key[_cache_key("gemini-2.0-pro", prefix)] = "cachedContents/stale"

        success_response = GeminiChatResponseSchema(
            candidates=[GeminiCandidateSchema(
                content=GeminiContentSchema(parts=[GeminiPartSchema(text="ok")]),
            )],
            usage=GeminiUsageSchema(
                prompt_token_count=300,
                total_tokens_count=320,
                candidates_token_count=20,
                cached_content_token_count=0,
            ),
        )

        class _FakeGeminiHTTP:
            def __init__(self) -> None:
                self.client = object()
                self.requests: list[GeminiChatRequestSchema] = []

            async def chat(self, request: GeminiChatRequestSchema) -> Any:
                self.requests.append(request)
                if request.cached_content == "cachedContents/stale":
                    raise GeminiHTTPClientError(
                        client="GeminiHTTPClient",
                        details="CachedContent not found",
                        status_code=404,
                    )
                return success_response

        fake = _FakeGeminiHTTP()
        monkeypatch.setattr(
            "ai_review.services.llm.gemini.client.get_gemini_http_client",
            lambda: fake,
        )

        cfg = GeminiLLMConfig(
            meta=GeminiMetaConfig(),
            provider=LLMProvider.GEMINI,
            http_client=GeminiHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("https://generativelanguage.googleapis.com"),
                api_token=SecretStr("fake-token"),
            ),
            cache=LLMCacheConfig(enabled=True, min_chars=10),
        )
        monkeypatch.setattr(settings, "llm", cfg)

        client = GeminiLLMClient()
        result = await client.chat("user prompt", prompt_system)

        # Two attempts: first with cached_content, second uncached.
        assert len(fake.requests) == 2
        assert fake.requests[0].cached_content == "cachedContents/stale"
        assert fake.requests[1].cached_content is None
        # Retry sends the full (boundary-stripped) system prompt as
        # system_instruction, and the user prompt is the original (no
        # variable folded in).
        assert fake.requests[1].system_instruction is not None
        assert fake.requests[1].contents[0].parts[0].text == "user prompt"
        # Stale entry was evicted from the in-process map.
        assert _cache_key("gemini-2.0-pro", prefix) not in _cache_name_by_key
        # The retry returns a normal response.
        assert result.text == "ok"


# ---------- Non-caching clients strip the boundary ----------

class TestNonCachingClientsStripBoundary:
    @pytest.mark.asyncio
    async def test_ollama_strips_boundary(self, monkeypatch) -> None:
        from ai_review.clients.ollama.schema import OllamaChatResponseSchema, OllamaMessageSchema as OllamaMsg
        from ai_review.libs.config.llm.base import OllamaLLMConfig
        from ai_review.libs.config.llm.ollama import OllamaMetaConfig, OllamaHTTPClientConfig
        cfg = OllamaLLMConfig(
            meta=OllamaMetaConfig(),
            provider=LLMProvider.OLLAMA,
            http_client=OllamaHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("http://localhost:11434"),
            ),
        )
        monkeypatch.setattr(settings, "llm", cfg)
        response = OllamaChatResponseSchema(
            model="x",
            message=OllamaMsg(role="assistant", content="ok"),
            done=True,
        )
        fake = _RecordingFake(response)
        monkeypatch.setattr(
            "ai_review.services.llm.ollama.client.get_ollama_http_client",
            lambda: fake,
        )
        client = OllamaLLMClient()
        await client.chat("p", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: OllamaChatRequestSchema = fake.requests[0]
        system_msg = request.messages[0]
        assert system_msg.role == "system"
        assert "AI_REVIEW_CACHE_BOUNDARY" not in system_msg.content
        assert system_msg.content == "B\n\nA"

    @pytest.mark.asyncio
    async def test_bedrock_strips_boundary(self, monkeypatch) -> None:
        from ai_review.clients.bedrock.schema import (
            BedrockChatResponseSchema,
            BedrockContentSchema,
            BedrockUsageSchema,
        )
        response = BedrockChatResponseSchema(
            id="id",
            type="message",
            role="assistant",
            content=[BedrockContentSchema(type="text", text="ok")],
            usage=BedrockUsageSchema(input_tokens=1, output_tokens=1),
        )
        fake = _RecordingFake(response)
        monkeypatch.setattr(
            "ai_review.services.llm.bedrock.client.get_bedrock_http_client",
            lambda: fake,
        )
        client = BedrockLLMClient()
        await client.chat("p", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: BedrockChatRequestSchema = fake.requests[0]
        assert request.messages[0].content == "B\n\nA"

    @pytest.mark.asyncio
    async def test_openrouter_strips_boundary(self, monkeypatch) -> None:
        from ai_review.clients.openrouter.schema import (
            OpenRouterChatResponseSchema,
            OpenRouterChoiceSchema,
            OpenRouterMessageSchema as OpenRouterMsg,
            OpenRouterUsageSchema,
        )
        response = OpenRouterChatResponseSchema(
            id="id",
            choices=[OpenRouterChoiceSchema(
                index=0,
                message=OpenRouterMsg(role="assistant", content="ok"),
                finish_reason="stop",
            )],
            usage=OpenRouterUsageSchema(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        fake = _RecordingFake(response)
        monkeypatch.setattr(
            "ai_review.services.llm.openrouter.client.get_openrouter_http_client",
            lambda: fake,
        )
        client = OpenRouterLLMClient()
        await client.chat("p", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: OpenRouterChatRequestSchema = fake.requests[0]
        assert request.messages[0].content == "B\n\nA"

    @pytest.mark.asyncio
    async def test_azure_openai_strips_boundary(self, monkeypatch) -> None:
        from ai_review.clients.azure_openai.schema import (
            AzureOpenAIChatResponseSchema,
            AzureOpenAIChoice,
            AzureOpenAIMessage as AzureMsg,
            AzureOpenAIUsage,
        )
        response = AzureOpenAIChatResponseSchema(
            id="id",
            choices=[AzureOpenAIChoice(
                index=0,
                message=AzureMsg(role="assistant", content="ok"),
                finish_reason="stop",
            )],
            usage=AzureOpenAIUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        fake = _RecordingFake(response)
        monkeypatch.setattr(
            "ai_review.services.llm.azure_openai.client.get_azure_openai_http_client",
            lambda: fake,
        )
        client = AzureOpenAILLMClient()
        await client.chat("p", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: AzureOpenAIChatRequestSchema = fake.requests[0]
        assert request.messages[0].content == "B\n\nA"

    @pytest.mark.asyncio
    async def test_openai_v1_strips_boundary(self, monkeypatch) -> None:
        cfg = OpenAILLMConfig(
            meta=OpenAIMetaConfig(model="gpt-4o"),  # v1 path
            provider=LLMProvider.OPENAI,
            http_client=OpenAIHTTPClientConfig(
                timeout=10,
                api_url=HttpUrl("https://api.openai.com"),
                api_token=SecretStr("fake-token"),
            ),
            cache=LLMCacheConfig(enabled=False),
        )
        monkeypatch.setattr(settings, "llm", cfg)

        from ai_review.clients.openai.v1.schema import (
            OpenAIChatResponseSchema,
            OpenAIChoiceSchema,
            OpenAIMessageSchema as OpenAIMsg,
            OpenAIUsageSchema,
        )
        response = OpenAIChatResponseSchema(
            id="id",
            choices=[OpenAIChoiceSchema(
                index=0,
                message=OpenAIMsg(role="assistant", content="ok"),
                finish_reason="stop",
            )],
            usage=OpenAIUsageSchema(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        fake = _RecordingFake(response)
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v1_http_client",
            lambda: fake,
        )
        monkeypatch.setattr(
            "ai_review.services.llm.openai.client.get_openai_v2_http_client",
            lambda: object(),
        )
        client = OpenAILLMClient()
        await client.chat("p", f"A{SYSTEM_PROMPT_CACHE_BOUNDARY}B")

        request: OpenAIChatRequestSchema = fake.requests[0]
        assert request.messages[0].content == "B\n\nA"
