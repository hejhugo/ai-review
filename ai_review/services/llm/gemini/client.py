from ai_review.clients.gemini.client import get_gemini_http_client
from ai_review.clients.gemini.schema import (
    GeminiPartSchema,
    GeminiContentSchema,
    GeminiChatRequestSchema,
    GeminiGenerationConfigSchema,
)
from ai_review.config import settings
from ai_review.libs.cache.gemini import get_or_create_cached_content
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema


class GeminiLLMClient(LLMClientProtocol):
    def __init__(self):
        self.http_client = get_gemini_http_client()

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        meta = settings.llm.meta
        cache = settings.llm.cache

        cached_content_name: str | None = None
        cache_creation = 0
        if cache.enabled and len(prompt_system) >= cache.min_chars:
            cached_content_name, cache_creation = await get_or_create_cached_content(
                client=self.http_client.client,
                model=meta.model,
                system_prompt=prompt_system,
            )

        request = GeminiChatRequestSchema(
            contents=[GeminiContentSchema(parts=[GeminiPartSchema(text=prompt)])],
            cached_content=cached_content_name,
            generation_config=GeminiGenerationConfigSchema(
                temperature=meta.temperature,
                max_output_tokens=meta.max_tokens,
            ),
            system_instruction=(
                None
                if cached_content_name
                else GeminiContentSchema(parts=[GeminiPartSchema(text=prompt_system)])
            ),
        )
        response = await self.http_client.chat(request)
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
