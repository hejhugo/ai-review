from ai_review.clients.gemini.client import GeminiHTTPClientError, get_gemini_http_client
from ai_review.clients.gemini.schema import (
    GeminiPartSchema,
    GeminiContentSchema,
    GeminiChatRequestSchema,
    GeminiGenerationConfigSchema,
)
from ai_review.config import settings
from ai_review.libs.cache.gemini import evict_cached_content, get_or_create_cached_content
from ai_review.libs.logger import get_logger
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema
from ai_review.services.prompt.schema import (
    split_system_prompt,
    strip_system_prompt_boundary,
)

logger = get_logger("GEMINI_LLM")


class GeminiLLMClient(LLMClientProtocol):
    def __init__(self):
        self.http_client = get_gemini_http_client()

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        meta = settings.llm.meta
        cache = settings.llm.cache

        cached_content_name: str | None = None
        cache_creation = 0
        cached_prefix, variable = split_system_prompt(prompt_system)
        prefix_for_cache = cached_prefix or prompt_system
        per_call_system = variable if cached_prefix else ""

        if cache.enabled and len(prefix_for_cache) >= cache.min_chars:
            cached_content_name, cache_creation = await get_or_create_cached_content(
                client=self.http_client.client,
                model=meta.model,
                system_prompt=prefix_for_cache,
            )

        if cached_content_name:
            # Gemini rejects requests that combine `cachedContent` with a
            # `system_instruction` (400 INVALID_ARGUMENT). The cached resource
            # already owns the system instruction, so fold the per-stage
            # variable into the user prompt instead.
            system_instruction = None
            user_text = f"{per_call_system}\n\n{prompt}" if per_call_system else prompt
        else:
            system_instruction = GeminiContentSchema(
                parts=[GeminiPartSchema(text=strip_system_prompt_boundary(prompt_system))]
            )
            user_text = prompt

        request = GeminiChatRequestSchema(
            contents=[GeminiContentSchema(parts=[GeminiPartSchema(text=user_text)])],
            cached_content=cached_content_name,
            generation_config=GeminiGenerationConfigSchema(
                temperature=meta.temperature,
                max_output_tokens=meta.max_tokens,
            ),
            system_instruction=system_instruction,
        )
        try:
            response = await self.http_client.chat(request)
        except GeminiHTTPClientError as exc:
            # The in-process cache map can hold a name whose `cachedContent`
            # resource has since expired (1h TTL) or been deleted out-of-band.
            # Evict the dead entry and retry once with a normal uncached
            # request so the stage still produces output.
            if cached_content_name is None or exc.status_code not in (400, 404):
                raise
            logger.warning(
                f"Gemini rejected cached_content={cached_content_name!r} "
                f"with {exc.status_code}; evicting and retrying uncached"
            )
            evict_cached_content(meta.model, prefix_for_cache)
            uncached_request = GeminiChatRequestSchema(
                contents=[GeminiContentSchema(parts=[GeminiPartSchema(text=prompt)])],
                cached_content=None,
                generation_config=GeminiGenerationConfigSchema(
                    temperature=meta.temperature,
                    max_output_tokens=meta.max_tokens,
                ),
                system_instruction=GeminiContentSchema(
                    parts=[GeminiPartSchema(text=strip_system_prompt_boundary(prompt_system))]
                ),
            )
            response = await self.http_client.chat(uncached_request)
            cache_creation = 0
        cache_read = response.usage.cached_content_token_count or 0
        prompt_tokens = max((response.usage.prompt_tokens or 0) - cache_read, 0)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )
