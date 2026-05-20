from pydantic import BaseModel

DEFAULT_MIN_TOKENS = 1024


class LLMCacheConfig(BaseModel):
    enabled: bool = False
    min_tokens: int = DEFAULT_MIN_TOKENS
