from __future__ import annotations

import contextlib
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services import llm_service, redis_store

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.types import UIMessage

router = APIRouter()


class ChatRequestBody(BaseModel):
    id: str
    message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None
    selectedChatModel: str = "gpt-4o"
    selectedVisibilityType: str = "private"


def _parse_ui_messages(raw_messages: list[dict[str, Any]]) -> list[UIMessage]:
    result: list[UIMessage] = []
    for raw in raw_messages:
        with contextlib.suppress(Exception):
            result.append(UIMessage.model_validate(raw))
    return result


async def _save_assistant_message(chat_id: str, ctx: StreamContext) -> None:
    parts = await ctx.store.get("assistant_parts", default=[])
    if not parts:
        return
    assistant_msg: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "parts": parts,
    }
    await redis_store.save_chat_message(chat_id, assistant_msg)


@router.post("/chat")
async def chat(body: ChatRequestBody) -> StreamingResponse:
    chat_id = body.id

    stored_messages = await redis_store.get_chat_messages(chat_id)

    if body.messages is not None:
        incoming_raw = body.messages
    elif body.message is not None:
        incoming_raw = [body.message]
    else:
        incoming_raw = []

    incoming_ids = {m.get("id") for m in incoming_raw if m.get("id")}
    stored_not_incoming = [
        m for m in stored_messages if m.get("id") not in incoming_ids
    ]

    all_raw = stored_not_incoming + incoming_raw

    for msg in incoming_raw:
        await redis_store.save_chat_message(chat_id, msg)

    all_messages = _parse_ui_messages(all_raw)

    existing_chat = await redis_store.get_chat(chat_id)
    is_first_message = existing_chat is None
    if is_first_message:
        await redis_store.save_chat(
            chat_id, "New Conversation", body.selectedVisibilityType
        )

    ctx = StreamContext()

    async def _work(ctx: StreamContext) -> None:
        await llm_service.chat(
            all_messages,
            chat_id=chat_id,
            ctx=ctx,
            is_first_message=is_first_message,
        )
        await _save_assistant_message(chat_id, ctx)

    await ctx.run(_work)

    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
