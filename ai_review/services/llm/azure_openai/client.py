from ai_review.clients.azure_openai.client import get_azure_openai_http_client
from ai_review.clients.azure_openai.schema import AzureOpenAIMessage, AzureOpenAIChatRequestSchema
from ai_review.config import settings
from ai_review.services.llm.types import LLMClientProtocol, ChatResultSchema
from ai_review.services.prompt.schema import strip_system_prompt_boundary


class AzureOpenAILLMClient(LLMClientProtocol):
    def __init__(self):
        self.http_client = get_azure_openai_http_client()

    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        prompt_system = strip_system_prompt_boundary(prompt_system)
        request = AzureOpenAIChatRequestSchema(
            messages=[
                AzureOpenAIMessage(role="system", content=prompt_system),
                AzureOpenAIMessage(role="user", content=prompt),
            ],
            temperature=settings.llm.meta.temperature,
            max_tokens=settings.llm.meta.max_tokens,
        )
        response = await self.http_client.chat(request)
        return ChatResultSchema(
            text=response.first_text,
            total_tokens=response.usage.total_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )
