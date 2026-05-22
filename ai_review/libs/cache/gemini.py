"""Gemini explicit prompt caching.

When caching is enabled and the system prompt is large enough to clear the
Gemini provider floor (estimated as char_count / 4), this module creates a
``cachedContent`` resource via the Gemini REST API and returns its
resource name.

The floor is the smallest documented per-model minimum: 1,024 tokens for
``gemini-2.5-flash`` / ``gemini-3.5-flash`` and 4,096 tokens for
``gemini-2.5-pro`` / ``gemini-3-pro-preview``. We use 4,096 as a single
conservative threshold across models -- Pro calls below 4,096 would be
rejected by the API anyway, and Flash callers giving up the 1,024-4,095
sweet spot is an acceptable trade for not encoding model-specific logic.

Within a single process, the (model, system_prompt) mapping is kept in
memory. This matches the intended scope -- inline/context/summary stages
share a system prompt within one run -- and avoids relying on the
``cachedContents.list`` endpoint, which does not support server-side
filtering in ``v1beta`` and can return duplicates under pagination.

If the prompt is below the floor or any API call fails, the function
returns ``None`` so the caller falls back to a normal (uncached) request.

The cache key is ``sha256(f"{model}:{system_prompt}")`` so that
model-bound resources never alias across models.
"""

from __future__ import annotations

import asyncio
import hashlib

import httpx

from ai_review.libs.logger import get_logger

GEMINI_CACHE_PATH = "/v1beta/cachedContents"
GEMINI_TOKEN_FLOOR = 4_096
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_TTL = "3600s"

logger = get_logger("GEMINI_CACHE")

_cache_name_by_key: dict[str, str] = {}
_creation_locks: dict[str, asyncio.Lock] = {}


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def _cache_key(model: str, system_prompt: str) -> str:
    payload = f"{model}:{system_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reset_cache_state() -> None:
    """Clear in-process cache map. Test-only helper."""
    _cache_name_by_key.clear()
    _creation_locks.clear()


def evict_cached_content(model: str, system_prompt: str) -> None:
    """Drop a (model, system_prompt) entry from the in-process map.

    Use when Gemini rejects a request that references a stale or
    invalidated ``cachedContent`` name so the next call recreates the
    resource instead of looping on the dead reference.
    """
    _cache_name_by_key.pop(_cache_key(model, system_prompt), None)


async def get_or_create_cached_content(
    client: httpx.AsyncClient,
    model: str,
    system_prompt: str,
) -> tuple[str | None, int]:
    """Return ``(cached_content_name, cache_creation_tokens)``.

    ``cached_content_name`` is ``None`` when caching is unavailable:
    - The estimated token count is below ``GEMINI_TOKEN_FLOOR``
    - The Gemini API rejects or fails the create request

    ``cache_creation_tokens`` is the token count reported by Gemini at the
    moment a new cache entry is created (and ``0`` on cache hits or
    failures).
    """
    if _estimate_tokens(system_prompt) < GEMINI_TOKEN_FLOOR:
        return None, 0

    key = _cache_key(model, system_prompt)
    if key in _cache_name_by_key:
        return _cache_name_by_key[key], 0

    lock = _creation_locks.setdefault(key, asyncio.Lock())
    async with lock:
        # Re-check after acquiring the lock: a concurrent caller may have
        # finished the POST while we were waiting, so the cache entry is
        # already populated.
        if key in _cache_name_by_key:
            return _cache_name_by_key[key], 0

        body = {
            "model": f"models/{model}",
            "displayName": key,
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": system_prompt}],
            },
            "ttl": DEFAULT_TTL,
        }

        try:
            response = await client.post(GEMINI_CACHE_PATH, json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"Gemini cache create failed: {exc!r}; falling back to uncached")
            return None, 0

        payload = response.json()
        cached_name = payload.get("name")
        if not cached_name:
            return None, 0

        usage = payload.get("usageMetadata") or {}
        creation_tokens = int(usage.get("totalTokenCount") or 0)

        _cache_name_by_key[key] = cached_name
        return cached_name, creation_tokens
