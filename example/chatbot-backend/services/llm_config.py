from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "-"),
    )


def get_model() -> str:
    return os.environ.get("LLM_MODEL", "gpt-4o")


def get_extra_body() -> dict[str, Any] | None:
    """Return extra_body for vLLM models that need thinking disabled."""
    raw = os.environ.get("LLM_EXTRA_BODY")
    if raw:
        return json.loads(raw)
    return None
