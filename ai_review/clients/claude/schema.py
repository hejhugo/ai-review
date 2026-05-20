from typing import Literal, Any

from pydantic import BaseModel


class ClaudeMessageSchema(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ClaudeCacheControlSchema(BaseModel):
    type: Literal["ephemeral"] = "ephemeral"


class ClaudeSystemBlockSchema(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: ClaudeCacheControlSchema | None = None


class ClaudeChatRequestSchema(BaseModel):
    model: str
    system: str | list[ClaudeSystemBlockSchema] | None = None
    messages: list[ClaudeMessageSchema]
    max_tokens: int | None = None
    temperature: float | None = None


class ClaudeContentSchema(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeCacheUsageSchema(BaseModel):
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ClaudeUsageSchema(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ClaudeChatResponseSchema(BaseModel):
    id: str
    role: str
    usage: ClaudeUsageSchema
    content: list[ClaudeContentSchema]

    @property
    def first_text(self) -> str:
        if not self.content:
            return ""

        return self.content[0].text.strip()
