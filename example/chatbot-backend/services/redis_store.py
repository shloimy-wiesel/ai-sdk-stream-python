from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6312")

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(_REDIS_URL, decode_responses=True)
    return _redis


async def get_document(doc_id: str) -> dict[str, Any] | None:
    r = _get_redis()
    raw = await r.get(f"doc:{doc_id}")
    if raw is None:
        return None
    return json.loads(raw)


async def save_document(
    doc_id: str,
    title: str,
    kind: str,
    content: str,
    user_id: str = "anonymous",
) -> dict[str, Any]:
    r = _get_redis()
    doc: dict[str, Any] = {
        "id": doc_id,
        "title": title,
        "kind": kind,
        "content": content,
        "userId": user_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    await r.set(f"doc:{doc_id}", json.dumps(doc))
    return doc


async def get_chat_messages(chat_id: str) -> list[dict[str, Any]]:
    r = _get_redis()
    raw = await r.get(f"chat:{chat_id}:messages")
    if raw is None:
        return []
    return json.loads(raw)


async def save_chat_message(chat_id: str, message: dict[str, Any]) -> None:
    r = _get_redis()
    messages = await get_chat_messages(chat_id)
    existing_ids = {m.get("id") for m in messages}
    if message.get("id") not in existing_ids:
        messages.append(message)
        await r.set(f"chat:{chat_id}:messages", json.dumps(messages))


async def save_chat(
    chat_id: str,
    title: str,
    visibility: str = "private",
) -> dict[str, Any]:
    r = _get_redis()
    chat: dict[str, Any] = {
        "id": chat_id,
        "title": title,
        "visibility": visibility,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    await r.set(f"chat:{chat_id}", json.dumps(chat))
    return chat


async def get_chat(chat_id: str) -> dict[str, Any] | None:
    r = _get_redis()
    raw = await r.get(f"chat:{chat_id}")
    if raw is None:
        return None
    return json.loads(raw)
