from ai_review.clients.openai.v1.client import get_openai_v1_http_client
from ai_review.clients.openai.v1.schema import OpenAIChatRequestSchema, OpenAIMessageSchema
from ai_review.clients.openai.v2.client import get_openai_v2_http_client
from ai_review.clients.openai.v2.schema import OpenAIInputMessageSchema, OpenAIResponsesRequestSchema
from ai_review.config import settings
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema
from ai_review.services.prompt.schema import (
    split_system_prompt,
    strip_system_prompt_boundary,
)


class OpenAILLMClient(LLMClientProtocol):
    def __init__(self):
        self.meta = settings.llm.meta

        self.http_client_v1 = get_openai_v1_http_client()
        self.http_client_v2 = get_openai_v2_http_client()

    async def chat_v1(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        prompt_system = strip_system_prompt_boundary(prompt_system)
        request = OpenAIChatRequestSchema(
            model=self.meta.model,
            messages=[
                OpenAIMessageSchema(role="system", content=prompt_system),
                OpenAIMessageSchema(role="user", content=prompt),
            ],
            max_tokens=self.meta.max_tokens,
            temperature=self.meta.temperature,
        )
        response = await self.http_client_v1.chat(request)
        # Chat Completions implicitly caches stable prefixes >= 1024 tokens
        # and reports the hit count in `usage.prompt_tokens_details.cached_tokens`.
        # Subtract it so cost math doesn't double-bill cached input.
        cache_read = response.usage.cached_tokens
        prompt_tokens = max(response.usage.prompt_tokens - cache_read, 0)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cache_read_tokens=cache_read,
        )

    async def chat_v2(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        cache = settings.llm.cache
        cached_prefix, variable = split_system_prompt(prompt_system) if cache.enabled else ("", "")
        # Gate `instructions` on the cacheable prefix size, not the whole
        # system prompt — otherwise a short user prefix + long stage default
        # crosses the threshold but OpenAI auto-cache still misses (its
        # floor is 1024 tokens of stable prefix).
        use_instructions = cache.enabled and len(cached_prefix) >= cache.min_chars

        if use_instructions:
            if cached_prefix and variable:
                # Split-stamp: cached prefix in `instructions` (auto-cached by
                # OpenAI), variable suffix as a per-call system message in
                # `input`. The system message is sent after instructions so
                # the model still sees the per-stage rules.
                request = OpenAIResponsesRequestSchema(
                    model=self.meta.model,
                    instructions=cached_prefix,
                    input=[
                        OpenAIInputMessageSchema(role="system", content=variable),
                        OpenAIInputMessageSchema(role="user", content=prompt),
                    ],
                    temperature=self.meta.temperature,
                    max_output_tokens=self.meta.max_tokens,
                )
            else:
                request = OpenAIResponsesRequestSchema(
                    model=self.meta.model,
                    instructions=cached_prefix or variable,
                    input=[OpenAIInputMessageSchema(role="user", content=prompt)],
                    temperature=self.meta.temperature,
                    max_output_tokens=self.meta.max_tokens,
                )
        else:
            request = OpenAIResponsesRequestSchema(
                model=self.meta.model,
                input=[
                    OpenAIInputMessageSchema(
                        role="system",
                        content=strip_system_prompt_boundary(prompt_system),
                    ),
                    OpenAIInputMessageSchema(role="user", content=prompt),
                ],
                temperature=self.meta.temperature,
                max_output_tokens=self.meta.max_tokens,
            )

        response = await self.http_client_v2.chat(request)
        # OpenAI's Responses API automatically caches any prefix >= 1024
        # tokens, regardless of whether `instructions` is used. Always
        # report the cached_tokens it returns so cost math reflects real
        # savings even when the opt-in caching path is disabled.
        cache_read = response.usage.cached_tokens
        prompt_tokens = max(response.usage.input_tokens - cache_read, 0)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=response.usage.output_tokens,
            cache_read_tokens=cache_read,
        )

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        if self.meta.is_v2_model:
            return await self.chat_v2(prompt, prompt_system)

        return await self.chat_v1(prompt, prompt_system)
