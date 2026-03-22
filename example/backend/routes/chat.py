"""
routes/chat.py — The /chat streaming endpoint.

Accepts a full conversation history (all messages, not just the latest) so
the backend is completely stateless: every request carries everything the
LLM needs to produce a contextual reply.

Pattern:
1. Create a ``StreamContext`` at the start of the request.
2. Spin up a background asyncio task that calls llm_service.chat().
3. Return ``StreamingResponse(ctx.stream(), ...)`` immediately so the
   client starts receiving SSE events as they are produced.

``llm_service`` handles the full tool-calling loop, streaming text via
``ctx.write_text()`` and executing tools via ``db_service`` (which emits
its own tool-input / source-url events through the same ``ctx``).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_sdk_stream_python import StreamContext

from ..services import llm_service

router = APIRouter()
_background_tasks: set[asyncio.Task] = set()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint.  Returns a UIMessageStream (SSE) response.

    The full conversation history is forwarded to the LLM so responses
    are contextually aware of previous turns.
    """
    ctx = StreamContext()

    async def _work() -> None:
        try:
            messages = [m.model_dump() for m in request.messages]
            await llm_service.chat(messages, ctx=ctx)
        except Exception as exc:
            await ctx.write_text(f"\n\n_(Error: {exc})_")
        finally:
            await ctx.finish()

    _task = asyncio.create_task(_work())
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
