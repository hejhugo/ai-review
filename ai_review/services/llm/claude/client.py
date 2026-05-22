from ai_review.clients.claude.client import get_claude_http_client
from ai_review.clients.claude.schema import (
    ClaudeCacheControlSchema,
    ClaudeChatRequestSchema,
    ClaudeMessageSchema,
    ClaudeSystemBlockSchema,
)
from ai_review.config import settings
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema
from ai_review.services.prompt.schema import (
    split_system_prompt,
    strip_system_prompt_boundary,
)


def _build_system(prompt_system: str) -> str | list[ClaudeSystemBlockSchema]:
    cache = settings.llm.cache
    if not cache.enabled:
        return strip_system_prompt_boundary(prompt_system)

    cached_prefix, variable = split_system_prompt(prompt_system)
    # Gate on the cacheable prefix size, not the whole system prompt.
    # A tiny stable prefix + long stage default would clear the threshold
    # on total length but produce a too-small cache_control block that
    # Anthropic's cache floor would reject silently.
    prefix_for_cache = cached_prefix or prompt_system
    if len(prefix_for_cache) < cache.min_chars:
        return strip_system_prompt_boundary(prompt_system)

    if cached_prefix and variable:
        return [
            ClaudeSystemBlockSchema(
                text=cached_prefix,
                cache_control=ClaudeCacheControlSchema(),
            ),
            ClaudeSystemBlockSchema(text=variable),
        ]

    # No boundary marker: cache the whole prompt as one block.
    return [ClaudeSystemBlockSchema(
        text=prefix_for_cache,
        cache_control=ClaudeCacheControlSchema(),
    )]


class ClaudeLLMClient(LLMClientProtocol):
    def __init__(self):
        self.http_client = get_claude_http_client()

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        meta = settings.llm.meta
        request = ClaudeChatRequestSchema(
            model=meta.model,
            system=_build_system(prompt_system),
            messages=[ClaudeMessageSchema(role="user", content=prompt)],
            max_tokens=meta.max_tokens,
            temperature=meta.temperature,
        )
        response = await self.http_client.chat(request)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cache_creation_tokens=response.usage.cache_creation_input_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens,
        )
