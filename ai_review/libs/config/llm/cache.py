from pydantic import BaseModel

DEFAULT_MIN_CHARS = 4096


class LLMCacheConfig(BaseModel):
    enabled: bool = False
    min_chars: int = DEFAULT_MIN_CHARS
