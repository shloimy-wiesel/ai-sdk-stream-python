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
import inspect
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from .collect import (
    DataPartRecord,
    FileRecord,
    SourceRecord,
    StreamRecord,
    ToolCallRecord,
)
from .events import (
    AbortEvent,
    BaseEvent,
    DataEvent,
    ErrorEvent,
    FileEvent,
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
)
from .state import StateStore

logger = logging.getLogger(__name__)

_InfoT = TypeVar("_InfoT", bound=BaseModel)

#: Type alias for the ``on_finish`` callback.
#: May be a plain synchronous callable or an async callable.
OnFinishCallback = Callable[["StreamRecord"], Any]


@dataclass
class ToolCallHandle:
    """Returned by ``begin_tool_call`` so the caller has the generated IDs."""

    toolCallId: str
    toolName: str


class StreamContext(Generic[_InfoT]):
    """
    A stateful context for producing a Vercel AI SDK v6 UIMessageStream.

    One ``StreamContext`` maps to one assistant message.  Create it at the
    start of a request, spin up a background task that writes events through
    the helpers, and pass ``ctx.stream()`` to ``StreamingResponse``.

    Attributes
    ----------
    store : StateStore
        Async key-value store shared across the background task and any caller.
    info : _InfoT | None
        Static, read-only metadata supplied at construction time (e.g. user_id,
        rate_limit).  Pass any Pydantic ``BaseModel`` instance as
        ``custom_information=`` — it is available unchanged for the entire
        lifetime of the context.  Defaults to ``None`` when not provided.
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
        count_func: Callable[[str], int] | None = None,
        custom_information: _InfoT | None = None,
        on_finish: OnFinishCallback | None = None,
        start_metadata: dict[str, Any] | None = None,
        stream_exclude: list[str] | None = None,
        store_exclude: list[str] | None = None,
    ) -> None:
        self._message_id: str = message_id or str(uuid.uuid4())
        self._queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        self.store: StateStore = StateStore()
        self._info: _InfoT | None = custom_information
        self._on_finish: OnFinishCallback | None = on_finish
        self._start_metadata: dict[str, Any] | None = start_metadata
        self._stream_exclude: tuple[str, ...] = (
            tuple(stream_exclude) if stream_exclude else ()
        )
        self._store_exclude: tuple[str, ...] = (
            tuple(store_exclude) if store_exclude else ()
        )

        # Lifecycle state
        self._started: bool = False
        self._step_open: bool = False
        self._text_id: str | None = None
        self._reasoning_id: str | None = None
        self._finished: bool = False

        # Collection — auto-enabled when on_finish is provided
        self._collect: bool = collect or on_finish is not None
        self._record: StreamRecord | None = (
            StreamRecord(message_id=self._message_id)
            if (collect or on_finish is not None)
            else None
        )
        self._count: Callable[[str], int] = (
            count_func if count_func is not None else len
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
    def info(self) -> _InfoT | None:
        """
        Static, read-only metadata supplied at construction time.

        Returns the ``custom_information`` value passed to ``__init__``, or
        ``None`` if none was provided.  Useful for carrying request-scoped
        data (e.g. ``user_id``, ``rate_limit``) through service layers without
        threading extra function arguments.
        """
        return self._info

    @property
    def record(self) -> StreamRecord | None:
        """
        The accumulated stream record, or ``None`` if ``collect=False``.

        Populated incrementally during the stream; fully available after
        ``finish()`` has been called.
        """
        return self._record

    def _should_collect(self, collect: bool | None) -> bool:
        """Resolve per-call *collect* against context-level collection.

        - ``None`` (default) → collect if context-level collection is enabled.
        - ``True`` → require collection; raise if context has no record.
        - ``False`` → skip collection even if context-level is enabled.
        """
        if collect is True and self._record is None:
            raise RuntimeError(
                "collect=True was passed but the StreamContext was created "
                "with collect=False. Either enable collection on the context "
                "(StreamContext(collect=True)) or omit the per-call collect "
                "parameter."
            )
        if collect is False:
            return False
        # collect is None → follow context-level collect flag
        return self._collect and self._record is not None

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
            self._queue.put_nowait(
                StartEvent(
                    messageId=self._message_id, messageMetadata=self._start_metadata
                )
            )

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

    async def write_text(self, delta: str, *, collect: bool | None = None) -> None:
        """
        Stream a text delta.

        Auto-emits ``start``, ``start-step``, and ``text-start`` if not yet
        open.  Also closes any open reasoning part first.

        Pass ``collect=False`` to stream the delta to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        stream_delta = delta
        for s in self._stream_exclude:
            stream_delta = stream_delta.replace(s, "")

        store_delta = delta
        for s in self._store_exclude:
            store_delta = store_delta.replace(s, "")

        if stream_delta:
            await self._ensure_step_open()
            await self._ensure_reasoning_closed()
            if self._text_id is None:
                self._text_id = str(uuid.uuid4())
                self._queue.put_nowait(TextStartEvent(id=self._text_id))
            self._queue.put_nowait(TextDeltaEvent(id=self._text_id, delta=stream_delta))

        if self._should_collect(collect) and self._record is not None and store_delta:
            self._record.text += store_delta
            self._record.answer_tokens += self._count(store_delta)

    async def write_reasoning(self, delta: str, *, collect: bool | None = None) -> None:
        """
        Stream a reasoning / chain-of-thought delta.

        Auto-emits ``start``, ``start-step``, and ``reasoning-start`` if not
        yet open.  Also closes any open text part first.

        Pass ``collect=False`` to stream the delta to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        stream_delta = delta
        for s in self._stream_exclude:
            stream_delta = stream_delta.replace(s, "")

        store_delta = delta
        for s in self._store_exclude:
            store_delta = store_delta.replace(s, "")

        if stream_delta:
            await self._ensure_step_open()
            await self._ensure_text_closed()
            if self._reasoning_id is None:
                self._reasoning_id = str(uuid.uuid4())
                self._queue.put_nowait(ReasoningStartEvent(id=self._reasoning_id))
            self._queue.put_nowait(
                ReasoningDeltaEvent(id=self._reasoning_id, delta=stream_delta)
            )

        if self._should_collect(collect) and self._record is not None and store_delta:
            self._record.reasoning += store_delta
            self._record.reasoning_tokens += self._count(store_delta)

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
        collect: bool | None = None,
    ) -> ToolCallHandle:
        """
        Emit ``tool-input-start`` + ``tool-input-available`` and return a
        handle containing the generated ``toolCallId``.

        Call ``complete_tool_call`` (or ``fail_tool_call``) with the returned
        handle once the tool has executed.

        Pass ``collect=False`` to stream the tool call to the frontend without
        recording it in ``ctx.record``.  Subsequent ``complete_tool_call`` /
        ``fail_tool_call`` calls will naturally skip the update since no
        matching record exists.  Passing ``collect=True`` when the context
        was created with ``collect=False`` raises ``RuntimeError``.
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
        if self._should_collect(collect) and self._record is not None:
            self._record.tool_calls.append(
                ToolCallRecord(tool_call_id=tcid, tool_name=tool_name, input=tool_input)
            )
        return ToolCallHandle(toolCallId=tcid, toolName=tool_name)

    async def start_tool_input(
        self,
        tool_name: str,
        *,
        tool_call_id: str | None = None,
        collect: bool | None = None,
    ) -> ToolCallHandle:
        """
        Emit ``tool-input-start`` and return a handle for streaming deltas.

        Use this when tool arguments arrive incrementally (e.g. from an LLM
        stream).  Follow with :meth:`stream_tool_input_delta` calls and
        finish with :meth:`finish_tool_input`.

        Pass ``collect=False`` to stream the tool call to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        await self._ensure_step_open()
        await self._ensure_text_closed()
        await self._ensure_reasoning_closed()
        tcid = tool_call_id or str(uuid.uuid4())
        self._queue.put_nowait(ToolInputStartEvent(toolCallId=tcid, toolName=tool_name))
        if self._should_collect(collect) and self._record is not None:
            self._record.tool_calls.append(
                ToolCallRecord(tool_call_id=tcid, tool_name=tool_name, input={})
            )
        return ToolCallHandle(toolCallId=tcid, toolName=tool_name)

    async def stream_tool_input_delta(
        self, tool_call_id: str, input_text_delta: str
    ) -> None:
        """Emit a ``tool-input-delta`` for an in-progress tool call."""
        self.write_event_to_stream(
            ToolInputDeltaEvent(
                toolCallId=tool_call_id, inputTextDelta=input_text_delta
            )
        )

    async def finish_tool_input(
        self,
        tool_call_id: str,
        tool_name: str,
        input: dict[str, Any],
    ) -> None:
        """
        Emit ``tool-input-available`` to close a streaming tool call.

        Updates the collected :class:`~collect.ToolCallRecord` with the
        final *input* if collection is enabled.
        """
        if self._record is not None:
            for tc in self._record.tool_calls:
                if tc.tool_call_id == tool_call_id:
                    tc.input = input
                    break
        self.write_event_to_stream(
            ToolInputAvailableEvent(
                toolCallId=tool_call_id, toolName=tool_name, input=input
            )
        )

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

    async def write_data(
        self,
        name: str,
        data: Any,
        *,
        id: str | None = None,
        transient: bool = False,
        collect: bool | None = None,
    ) -> None:
        """
        Emit a custom data part (``data-{name}`` type).

        *name* is validated: it must be non-empty and contain only
        alphanumeric characters, hyphens, or underscores.
        Non-transient parts are collected in ``ctx.record.data_parts``.
        Transient parts are only available through the ``onData`` callback
        on the frontend; they are not stored in message history.

        Pass ``collect=False`` to stream the data part to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(
                f"Invalid data part name {name!r}. "
                "Use only alphanumeric characters, hyphens, or underscores."
            )
        await self._ensure_started()
        event = DataEvent(
            type=f"data-{name}",
            data=data,
            id=id,
            transient=transient or None,
        )
        if self._should_collect(collect) and self._record is not None and not transient:
            self._record.data_parts.append(DataPartRecord(name=name, data=data, id=id))
        self.write_event_to_stream(event)

    async def write_file(
        self, url: str, media_type: str, *, collect: bool | None = None
    ) -> None:
        """
        Emit a ``file`` event (image, PDF, or other file content).

        Auto-emits ``start`` and ``start-step`` if not yet open.
        On the frontend this produces a ``FileUIPart`` in ``message.parts``.

        Pass ``collect=False`` to stream the file to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        await self._ensure_step_open()
        if self._should_collect(collect) and self._record is not None:
            self._record.files.append(FileRecord(url=url, media_type=media_type))
        self.write_event_to_stream(FileEvent(url=url, mediaType=media_type))

    async def write_source(
        self,
        source_id: str,
        url: str,
        title: str | None = None,
        *,
        collect: bool | None = None,
    ) -> None:
        """
        Emit a ``source-url`` event (document / citation reference).

        Pass ``collect=False`` to stream the source to the frontend without
        recording it in ``ctx.record``.  Passing ``collect=True`` when the
        context was created with ``collect=False`` raises ``RuntimeError``.
        """
        await self._ensure_started()
        if self._should_collect(collect) and self._record is not None:
            self._record.sources.append(
                SourceRecord(source_id=source_id, url=url, title=title)
            )
        self.write_event_to_stream(
            SourceUrlEvent(sourceId=source_id, url=url, title=title)
        )

    async def set_usage(
        self,
        *,
        prompt_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        answer_tokens: int | None = None,
    ) -> None:
        """
        Override auto-counted token values with exact LLM-reported counts.

        Use this to replace character-count approximations with exact values
        when the LLM provides them (e.g. OpenAI's ``stream_options={"include_usage": True}``
        in the final chunk, or Anthropic's ``usage`` block).

        Only fields that are explicitly passed are updated; omitted fields
        retain their current values.  Has no effect when ``collect=False``.
        """
        if self._record is not None:
            if prompt_tokens is not None:
                self._record.prompt_tokens = prompt_tokens
            if reasoning_tokens is not None:
                self._record.reasoning_tokens = reasoning_tokens
            if answer_tokens is not None:
                self._record.answer_tokens = answer_tokens

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
        *,
        on_finish: OnFinishCallback | None = None,
    ) -> None:
        """
        Close all open parts/steps, emit ``finish``, and terminate the stream.

        If an ``on_finish`` callback is provided (either here or at construction
        time), it is invoked with the fully populated :class:`~collect.StreamRecord`
        after the finish event is queued.  Both sync and async callables are
        accepted.  Any exception raised by the callback is logged and swallowed
        so that stream termination is never blocked.

        The ``on_finish`` parameter passed here takes priority over the one
        supplied at construction time.  Providing a callback here also
        auto-enables collection for this call (the record will be non-``None``).

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._finished:
            return
        # If a per-call on_finish is given, ensure we have a record to pass it.
        effective_callback = on_finish if on_finish is not None else self._on_finish
        if effective_callback is not None and self._record is None:
            self._record = StreamRecord(message_id=self._message_id)
        await self._ensure_started()
        await self._ensure_step_closed()
        self._finished = True
        if self._record is not None:
            self._record.finish_reason = finish_reason
            self._record.finished_at = datetime.now(timezone.utc)
        self._queue.put_nowait(
            FinishEvent(finishReason=finish_reason, messageMetadata=message_metadata)
        )
        # Invoke the callback before the sentinel so the record is fully
        # populated.  Exceptions are caught and logged to avoid blocking
        # stream termination.
        if effective_callback is not None and self._record is not None:
            try:
                if inspect.iscoroutinefunction(effective_callback):
                    await effective_callback(self._record)
                else:
                    result = effective_callback(self._record)
                    if inspect.isawaitable(result):
                        await result
            except Exception:
                logger.exception("on_finish callback raised an exception")
        self._queue.put_nowait(None)  # sentinel → stream() yields [DONE]

    async def abort(self, reason: str | None = None) -> None:
        """
        Emit an ``abort`` event and terminate the stream.

        Use in error-handling paths where the normal ``finish()`` flow
        cannot be reached.  The optional *reason* string is forwarded to
        the frontend so ``useChat`` can surface why the stream stopped.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._finished:
            return
        self._finished = True
        self._queue.put_nowait(AbortEvent(reason=reason))
        self._queue.put_nowait(None)

    async def error(self, error_text: str) -> None:
        """
        Emit an ``error`` event and terminate the stream.

        The AI SDK v6 ``useChat`` hook surfaces this via its ``error`` object.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._finished:
            return
        await self._ensure_started()
        self._finished = True
        self._queue.put_nowait(ErrorEvent(errorText=error_text))
        self._queue.put_nowait(None)

    async def run(
        self, coro: Callable[[StreamContext[_InfoT]], Awaitable[None]]
    ) -> asyncio.Task[None]:
        """
        Run *coro* as a background task with automatic error/finish handling.

        This is the **recommended way** to wire up a streaming endpoint.
        It provides three safety guarantees:

        1. **Auto-finish** — ``finish()`` is called in a ``finally`` block so
           the stream is always closed, even if *coro* returns early.
        2. **Auto-error** — unhandled exceptions are caught and emitted as an
           ``error`` event so the frontend receives a proper error response
           instead of hanging indefinitely.
        3. **Task GC prevention** — the background task is stored on the
           context so Python's garbage collector cannot silently discard it.

        Recommended FastAPI pattern::

            @router.post("/chat")
            async def chat(request: ChatRequest) -> StreamingResponse:
                ctx = StreamContext()
                await ctx.run(lambda ctx: my_service.chat(request, ctx=ctx))
                return StreamingResponse(
                    ctx.stream(),
                    media_type="text/event-stream",
                    headers=ctx.response_headers,
                )
        """

        async def _safe() -> None:
            try:
                await coro(self)
            except Exception as exc:
                if not self.is_finished:
                    await self.error(str(exc))
            finally:
                if not self.is_finished:
                    await self.finish()

        task: asyncio.Task[None] = asyncio.create_task(_safe())
        self._task = task  # prevent GC
        return task

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


__all__ = ["StreamContext", "ToolCallHandle", "OnFinishCallback"]
