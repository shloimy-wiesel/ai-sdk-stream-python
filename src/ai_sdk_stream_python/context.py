"""
StreamContext — the main API for building Vercel AI SDK v6 compatible streams.

Inspired by llama-index-workflows ``Context``:
  - ``ctx.store`` for typed, async state (read/write via dot-paths)
  - ``ctx.write_event_to_stream(ev)`` to push a protocol event (sync)
  - High-level helpers (``write_text``, ``write_reasoning``, ``begin_tool_call``…)
    that automatically emit the correct lifecycle events in the right order

Lifecycle state machine
-----------------------
The stream protocol requires events in a strict ordering::

    start
      start-step
        [reasoning-start … reasoning-delta* … reasoning-end]
        [tool-input-start … tool-input-available … tool-output-available]
        [text-start … text-delta* … text-end]
        [source-url*]
      finish-step
    finish
    [DONE]

The context tracks which lifecycle events have already been emitted and
auto-emits any missing prefix events before writing content.

Usage example (FastAPI)::

    from fastapi.responses import StreamingResponse
    from ai_sdk_stream_python import StreamContext


    @app.post("/chat")
    async def chat(request: ChatRequest):
        ctx = StreamContext()

        async def _work():
            try:
                await ctx.store.set("query", request.message)
                async for chunk in my_llm.stream(request.message):
                    await ctx.write_text(chunk)
            finally:
                await ctx.finish()

        asyncio.create_task(_work())
        return StreamingResponse(
            ctx.stream(),
            media_type="text/event-stream",
            headers=ctx.response_headers,
        )
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, ClassVar

from .collect import SourceRecord, StreamRecord, ToolCallRecord
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
    ToolInputStartEvent,
    ToolOutputAvailableEvent,
    ToolOutputErrorEvent,
)
from .state import StateStore


@dataclass
class ToolCallHandle:
    """Returned by ``begin_tool_call`` so the caller has the generated IDs."""

    toolCallId: str
    toolName: str


class StreamContext:
    """
    A stateful context for producing a Vercel AI SDK v6 UIMessageStream.

    One ``StreamContext`` maps to one assistant message.  Create it at the
    start of a request, spin up a background task that writes events through
    the helpers, and pass ``ctx.stream()`` to ``StreamingResponse``.

    Attributes
    ----------
    store : StateStore
        Async key-value store shared across the background task and any caller.
    """

    response_headers: ClassVar[dict[str, str]] = {
        "x-vercel-ai-ui-message-stream": "v1",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    def __init__(
        self,
        message_id: str | None = None,
        *,
        collect: bool = False,
    ) -> None:
        self._message_id: str = message_id or str(uuid.uuid4())
        self._queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        self.store: StateStore = StateStore()

        # Lifecycle state
        self._started: bool = False
        self._step_open: bool = False
        self._text_id: str | None = None
        self._reasoning_id: str | None = None
        self._finished: bool = False

        # Collection
        self._record: StreamRecord | None = (
            StreamRecord(message_id=self._message_id) if collect else None
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def message_id(self) -> str:
        return self._message_id

    @property
    def current_text_id(self) -> str | None:
        """ID of the currently open text part, or ``None``."""
        return self._text_id

    @property
    def current_reasoning_id(self) -> str | None:
        """ID of the currently open reasoning part, or ``None``."""
        return self._reasoning_id

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def record(self) -> StreamRecord | None:
        """
        The accumulated stream record, or ``None`` if ``collect=False``.

        Populated incrementally during the stream; fully available after
        ``finish()`` has been called.
        """
        return self._record

    # ── Low-level sync emit ────────────────────────────────────────────────────

    def write_event_to_stream(self, ev: BaseEvent) -> None:
        """
        Push a pre-constructed event onto the internal queue (sync).

        This is the lowest-level emission method, matching the LlamaIndex
        ``Context.write_event_to_stream`` API.  Prefer the async helpers
        (``write_text``, ``write_reasoning``, etc.) which auto-handle lifecycle
        ordering.

        Raises ``RuntimeError`` if the stream has already been finished.
        """
        if self._finished:
            raise RuntimeError("Cannot write to a finished StreamContext")
        self._queue.put_nowait(ev)

    # ── Lifecycle ensure-helpers (async) ──────────────────────────────────────

    async def _ensure_started(self) -> None:
        if not self._started:
            self._started = True
            self._queue.put_nowait(StartEvent(messageId=self._message_id))

    async def _ensure_step_open(self) -> None:
        await self._ensure_started()
        if not self._step_open:
            self._step_open = True
            if self._record is not None:
                self._record.step_count += 1
            self._queue.put_nowait(StartStepEvent())

    async def _ensure_text_closed(self) -> None:
        if self._text_id is not None:
            self._queue.put_nowait(TextEndEvent(id=self._text_id))
            self._text_id = None

    async def _ensure_reasoning_closed(self) -> None:
        if self._reasoning_id is not None:
            self._queue.put_nowait(ReasoningEndEvent(id=self._reasoning_id))
            self._reasoning_id = None

    async def _ensure_step_closed(self) -> None:
        await self._ensure_text_closed()
        await self._ensure_reasoning_closed()
        if self._step_open:
            self._queue.put_nowait(FinishStepEvent())
            self._step_open = False

    # ── High-level async write helpers ────────────────────────────────────────

    async def write_text(self, delta: str) -> None:
        """
        Stream a text delta.

        Auto-emits ``start``, ``start-step``, and ``text-start`` if not yet
        open.  Also closes any open reasoning part first.
        """
        await self._ensure_step_open()
        await self._ensure_reasoning_closed()
        if self._text_id is None:
            self._text_id = str(uuid.uuid4())
            self._queue.put_nowait(TextStartEvent(id=self._text_id))
        self._queue.put_nowait(TextDeltaEvent(id=self._text_id, delta=delta))
        if self._record is not None:
            self._record.text += delta

    async def write_reasoning(self, delta: str) -> None:
        """
        Stream a reasoning / chain-of-thought delta.

        Auto-emits ``start``, ``start-step``, and ``reasoning-start`` if not
        yet open.  Also closes any open text part first.
        """
        await self._ensure_step_open()
        await self._ensure_text_closed()
        if self._reasoning_id is None:
            self._reasoning_id = str(uuid.uuid4())
            self._queue.put_nowait(ReasoningStartEvent(id=self._reasoning_id))
        self._queue.put_nowait(ReasoningDeltaEvent(id=self._reasoning_id, delta=delta))
        if self._record is not None:
            self._record.reasoning += delta

    async def new_step(self) -> None:
        """
        Close any open text/reasoning/step and open a new step.

        Use this when the assistant moves from one logical phase to another
        (e.g. from reasoning to tool use, or from tool use to final answer).
        """
        await self._ensure_step_closed()
        await self._ensure_step_open()

    async def begin_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        tool_call_id: str | None = None,
    ) -> ToolCallHandle:
        """
        Emit ``tool-input-start`` + ``tool-input-available`` and return a
        handle containing the generated ``toolCallId``.

        Call ``complete_tool_call`` (or ``fail_tool_call``) with the returned
        handle once the tool has executed.
        """
        await self._ensure_step_open()
        await self._ensure_text_closed()
        await self._ensure_reasoning_closed()
        tcid = tool_call_id or str(uuid.uuid4())
        self._queue.put_nowait(ToolInputStartEvent(toolCallId=tcid, toolName=tool_name))
        self._queue.put_nowait(
            ToolInputAvailableEvent(
                toolCallId=tcid, toolName=tool_name, input=tool_input
            )
        )
        if self._record is not None:
            self._record.tool_calls.append(
                ToolCallRecord(tool_call_id=tcid, tool_name=tool_name, input=tool_input)
            )
        return ToolCallHandle(toolCallId=tcid, toolName=tool_name)

    async def complete_tool_call(self, tool_call_id: str, output: Any) -> None:
        """Emit ``tool-output-available`` with the tool result."""
        if self._record is not None:
            for tc in self._record.tool_calls:
                if tc.tool_call_id == tool_call_id:
                    tc.output = output
                    break
        self.write_event_to_stream(
            ToolOutputAvailableEvent(toolCallId=tool_call_id, output=output)
        )

    async def fail_tool_call(self, tool_call_id: str, error: str) -> None:
        """Emit ``tool-output-error`` when a tool call fails."""
        if self._record is not None:
            for tc in self._record.tool_calls:
                if tc.tool_call_id == tool_call_id:
                    tc.error = error
                    break
        self.write_event_to_stream(
            ToolOutputErrorEvent(toolCallId=tool_call_id, error=error)
        )

    async def write_source(
        self,
        source_id: str,
        url: str,
        title: str | None = None,
    ) -> None:
        """Emit a ``source-url`` event (document / citation reference)."""
        await self._ensure_started()
        if self._record is not None:
            self._record.sources.append(
                SourceRecord(source_id=source_id, url=url, title=title)
            )
        self.write_event_to_stream(
            SourceUrlEvent(sourceId=source_id, url=url, title=title)
        )

    async def write(self, event: BaseEvent) -> None:
        """
        Low-level: push any pre-constructed ``BaseEvent`` directly.

        Only auto-emits ``start`` if not yet started.  All other ordering
        is the caller's responsibility.
        """
        await self._ensure_started()
        self.write_event_to_stream(event)

    async def finish(
        self,
        finish_reason: str = "stop",
        message_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Close all open parts/steps, emit ``finish``, and terminate the stream.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._finished:
            return
        await self._ensure_started()
        await self._ensure_step_closed()
        self._finished = True
        if self._record is not None:
            self._record.finish_reason = finish_reason
        self._queue.put_nowait(
            FinishEvent(finishReason=finish_reason, messageMetadata=message_metadata)
        )
        self._queue.put_nowait(None)  # sentinel → stream() yields [DONE]

    async def abort(self) -> None:
        """
        Terminate the stream immediately without a proper finish event.

        Use in error-handling paths where the normal ``finish()`` flow
        cannot be reached.
        """
        if self._finished:
            return
        self._finished = True
        self._queue.put_nowait(None)

    # ── SSE async generator ───────────────────────────────────────────────────

    async def stream(self) -> AsyncGenerator[str, None]:
        """
        Async generator of SSE-encoded strings.

        Pass this directly to ``fastapi.responses.StreamingResponse``::

            StreamingResponse(
                ctx.stream(),
                media_type="text/event-stream",
                headers=ctx.response_headers,
            )

        The generator runs until ``finish()`` (or ``abort()``) is called by
        the background task.
        """
        while True:
            ev = await self._queue.get()
            if ev is None:
                yield "data: [DONE]\n\n"
                return
            yield ev.encode()


__all__ = ["StreamContext", "ToolCallHandle"]
