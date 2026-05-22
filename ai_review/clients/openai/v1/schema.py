from typing import Literal

from pydantic import BaseModel


class OpenAIPromptTokensDetailsSchema(BaseModel):
    cached_tokens: int = 0


class OpenAIUsageSchema(BaseModel):
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: OpenAIPromptTokensDetailsSchema = OpenAIPromptTokensDetailsSchema()

    @property
    def cached_tokens(self) -> int:
        return self.prompt_tokens_details.cached_tokens


class OpenAIMessageSchema(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIChoiceSchema(BaseModel):
    message: OpenAIMessageSchema


class OpenAIChatRequestSchema(BaseModel):
    model: str
    stream: bool = False
    messages: list[OpenAIMessageSchema]
    max_tokens: int | None = None
    temperature: float | None = None


class OpenAIChatResponseSchema(BaseModel):
    usage: OpenAIUsageSchema
    choices: list[OpenAIChoiceSchema]

    @property
    def first_text(self) -> str:
        if not self.choices:
            return ""

        return (self.choices[0].message.content or "").strip()
