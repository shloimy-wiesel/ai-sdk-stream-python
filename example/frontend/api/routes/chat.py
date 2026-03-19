"""
routes/chat.py — The /api/chat streaming endpoint.

Accepts a full conversation history (all messages, not just the latest) so
the backend is completely stateless: every request carries everything the
LLM needs to produce a contextual reply.

Pattern:
1. Create a ``StreamContext`` at the start of the request.
2. Spin up a background asyncio task that calls llm_service.chat().
3. Return ``StreamingResponse(ctx.stream(), ...)`` immediately so the
   client starts receiving SSE events as they are produced.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_sdk_stream_python import StreamContext

from ..services import llm_service

router = APIRouter()


class ContentPart(BaseModel):
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    parts: list[ContentPart] | None = None

    def to_llm_message(self) -> dict:
        if self.content is not None:
            return {"role": self.role, "content": self.content}
        text = " ".join(
            p.text for p in (self.parts or []) if p.type == "text" and p.text
        )
        return {"role": self.role, "content": text}


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint. Returns a UIMessageStream (SSE) response.

    The full conversation history is forwarded to the LLM so responses
    are contextually aware of previous turns.
    """
    ctx = StreamContext()

    async def _work() -> None:
        try:
            messages = [m.to_llm_message() for m in request.messages]
            await llm_service.chat(messages, ctx=ctx)
        except Exception as exc:  # noqa: BLE001
            await ctx.write_text(f"\n\n_(Error: {exc})_")
        finally:
            await ctx.finish()

    asyncio.create_task(_work())

    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
