"""Gemini explicit prompt caching.

When caching is enabled and the system prompt is large enough (>= provider
floor of roughly 32 768 tokens), this module looks up or creates a
``cachedContent`` resource via the Gemini REST API and returns its resource
name for use in the generation request.

If the system prompt is below the floor, or if any API call fails, the
function returns ``None`` so the caller falls back to a normal (uncached)
request — no error is raised.

The token floor is estimated cheaply as ``len(text) // 4``.  This is a
deliberate under-estimate; the real tokeniser would be more accurate, but
adding a tokeniser dependency just to gate caching is not worth it.  If the
estimate is wrong the Gemini API will return a 400 and we handle that by
returning ``None``.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

GEMINI_CACHE_API = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
GEMINI_TOKEN_FLOOR = 32_768
CHARS_PER_TOKEN_ESTIMATE = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN_ESTIMATE


def _cache_key(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def _api_request(url: str, method: str, body: dict | None, api_token: str) -> dict | None:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_token,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            return None
        return None
    except Exception:
        return None


def get_or_create_cached_content(
    model: str,
    system_prompt: str,
    api_token: str,
) -> str | None:
    """Return the ``cachedContent`` resource name, or ``None`` if unavailable.

    Returns ``None`` when:
    - Estimated token count is below ``GEMINI_TOKEN_FLOOR``
    - Any network or API error occurs
    - The provider returns a 400 (e.g. model does not support caching)
    """
    if _estimate_tokens(system_prompt) < GEMINI_TOKEN_FLOOR:
        return None

    display_name = _cache_key(system_prompt)

    list_resp = _api_request(
        f"{GEMINI_CACHE_API}?filter=display_name%3D{display_name}",
        "GET",
        None,
        api_token,
    )
    if list_resp:
        for item in list_resp.get("cachedContents", []):
            if item.get("displayName") == display_name:
                return item.get("name")

    body = {
        "model": f"models/{model}",
        "displayName": display_name,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "(cache warm-up — discard this content)"}],
            }
        ],
        "systemInstruction": {
            "role": "user",
            "parts": [{"text": system_prompt}],
        },
        "ttl": "3600s",
    }
    create_resp = _api_request(GEMINI_CACHE_API, "POST", body, api_token)
    if create_resp:
        return create_resp.get("name")

    return None
