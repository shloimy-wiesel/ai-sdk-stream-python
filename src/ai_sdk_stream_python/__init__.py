"""
ai-sdk-stream-python
====================

A Python library for building Vercel AI SDK v6 UIMessageStream-compatible
streaming backends.

Core idea (inspired by llama-index-workflows):
  - ``StreamContext`` holds shared **state** (``ctx.store.get/set``) and
    emits **typed stream events** (``ctx.write_text``, ``ctx.write_reasoning``,
    ``ctx.begin_tool_call``, …).
  - The context tracks the stream lifecycle so you never have to manually emit
    ``start``, ``start-step``, ``text-start`` etc. — they are auto-emitted.
  - All wire-protocol events are **Pydantic models** (``TextDeltaEvent``, …).

Quickstart::

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from ai_sdk_stream_python import StreamContext
    import asyncio

    app = FastAPI()


    @app.post("/chat")
    async def chat():
        ctx = StreamContext()

        async def _work():
            try:
                await ctx.write_text("Hello ")
                await ctx.write_text("world!")
            finally:
                await ctx.finish()

        asyncio.create_task(_work())
        return StreamingResponse(
            ctx.stream(),
            media_type="text/event-stream",
            headers=ctx.response_headers,
        )
"""

from .collect import SourceRecord as SourceRecord
from .collect import StreamRecord as StreamRecord
from .collect import ToolCallRecord as ToolCallRecord
from .context import StreamContext, ToolCallHandle
from .events import (
    BaseEvent,
    FinishEvent,
    FinishStepEvent,
    ReasoningDeltaEvent,
    ReasoningEndEvent,
    ReasoningStartEvent,
    SourceUrlEvent,
    StartEvent,
    StartStepEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInputAvailableEvent,
    ToolInputDeltaEvent,
    ToolInputStartEvent,
    ToolOutputAvailableEvent,
    ToolOutputErrorEvent,
    UIMessageStreamEvent,
)
from .state import StateStore

__all__ = [
    # Core
    "StreamContext",
    "ToolCallHandle",
    "StateStore",
    # Collection
    "StreamRecord",
    "ToolCallRecord",
    "SourceRecord",
    # Base / union
    "BaseEvent",
    "UIMessageStreamEvent",
    # Lifecycle
    "StartEvent",
    "StartStepEvent",
    "FinishStepEvent",
    "FinishEvent",
    # Reasoning
    "ReasoningStartEvent",
    "ReasoningDeltaEvent",
    "ReasoningEndEvent",
    # Text
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    # Tools
    "ToolInputStartEvent",
    "ToolInputDeltaEvent",
    "ToolInputAvailableEvent",
    "ToolOutputAvailableEvent",
    "ToolOutputErrorEvent",
    # Sources
    "SourceUrlEvent",
]
