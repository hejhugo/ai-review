from typing import Protocol

from pydantic import BaseModel


class ChatResultSchema(BaseModel):
    text: str
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0


class LLMClientProtocol(Protocol):
    async def chat(self, prompt: str, prompt_system: str) -> ChatResultSchema:
        ...
