"""
Tests for StreamContext — lifecycle state machine and edge cases.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.events import (
    StartEvent,
    TextDeltaEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_stream(ctx: StreamContext) -> list[dict]:
    """Drain ctx.stream() and return parsed event dicts (skipping [DONE])."""
    events = []
    async for chunk in ctx.stream():
        if chunk.strip() == "data: [DONE]":
            break
        raw = chunk.removeprefix("data: ").strip()
        if raw:
            events.append(json.loads(raw))
    return events


async def run_and_collect(producer, **kwargs) -> list[dict]:
    """Run producer(ctx) as a background task and collect all events."""
    ctx = StreamContext(**kwargs)
    asyncio.create_task(producer(ctx))
    return await collect_stream(ctx)


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


class TestBasicLifecycle:
    async def test_simple_text(self):
        async def work(ctx):
            await ctx.write_text("hello world")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        assert types == [
            "start",
            "start-step",
            "text-start",
            "text-delta",
            "text-end",
            "finish-step",
            "finish",
        ]
        assert events[3]["delta"] == "hello world"

    async def test_multiple_text_deltas_share_id(self):
        async def work(ctx):
            await ctx.write_text("foo")
            await ctx.write_text("bar")
            await ctx.finish()

        events = await run_and_collect(work)
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        assert len(text_deltas) == 2
        assert text_deltas[0]["id"] == text_deltas[1]["id"]

    async def test_start_emitted_once(self):
        async def work(ctx):
            await ctx.write_text("a")
            await ctx.write_text("b")
            await ctx.finish()

        events = await run_and_collect(work)
        assert sum(1 for e in events if e["type"] == "start") == 1

    async def test_finish_is_idempotent(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()
            await ctx.finish()  # second call must be a no-op

        events = await run_and_collect(work)
        assert sum(1 for e in events if e["type"] == "finish") == 1

    async def test_empty_stream_finish(self):
        """finish() without any writes still produces a valid stream."""

        async def work(ctx):
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        assert "start" in types
        assert "finish" in types
        # No step should be opened if nothing was written
        assert "start-step" not in types


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    async def test_reasoning_then_text(self):
        async def work(ctx):
            await ctx.write_reasoning("thinking…")
            await ctx.write_text("answer")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        # reasoning-end must come before text-start
        assert types.index("reasoning-end") < types.index("text-start")

    async def test_reasoning_auto_closes_when_text_starts(self):
        async def work(ctx):
            await ctx.write_reasoning("thought")
            await ctx.write_text("answer")
            await ctx.finish()

        events = await run_and_collect(work)
        assert any(e["type"] == "reasoning-end" for e in events)

    async def test_text_auto_closes_when_reasoning_starts(self):
        """Switching from text → reasoning should close text first."""

        async def work(ctx):
            await ctx.write_text("partial")
            await ctx.write_reasoning("second thought")
            await ctx.finish()

        events = await run_and_collect(work)
        text_end_idx = next(i for i, e in enumerate(events) if e["type"] == "text-end")
        reasoning_start_idx = next(
            i for i, e in enumerate(events) if e["type"] == "reasoning-start"
        )
        assert text_end_idx < reasoning_start_idx


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class TestStreamingToolInput:
    async def test_start_stream_finish_sequence(self):
        """start_tool_input → stream_tool_input_delta* → finish_tool_input ordering."""

        async def work(ctx):
            handle = await ctx.start_tool_input("search")
            await ctx.stream_tool_input_delta(handle.toolCallId, '{"q":')
            await ctx.stream_tool_input_delta(handle.toolCallId, '"cats"}')
            await ctx.finish_tool_input(handle.toolCallId, "search", {"q": "cats"})
            await ctx.complete_tool_call(handle.toolCallId, ["cat1"])
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        ti_start = types.index("tool-input-start")
        deltas = [i for i, t in enumerate(types) if t == "tool-input-delta"]
        ti_avail = types.index("tool-input-available")
        assert ti_start < deltas[0] < deltas[1] < ti_avail

    async def test_delta_ids_match_start(self):
        async def work(ctx):
            handle = await ctx.start_tool_input("calc")
            await ctx.stream_tool_input_delta(handle.toolCallId, '{"x": 1}')
            await ctx.finish_tool_input(handle.toolCallId, "calc", {"x": 1})
            await ctx.finish()

        events = await run_and_collect(work)
        start = next(e for e in events if e["type"] == "tool-input-start")
        delta = next(e for e in events if e["type"] == "tool-input-delta")
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert start["toolCallId"] == delta["toolCallId"] == avail["toolCallId"]

    async def test_delta_carries_input_text(self):
        async def work(ctx):
            handle = await ctx.start_tool_input("fn")
            await ctx.stream_tool_input_delta(handle.toolCallId, "chunk1")
            await ctx.finish_tool_input(handle.toolCallId, "fn", {})
            await ctx.finish()

        events = await run_and_collect(work)
        delta = next(e for e in events if e["type"] == "tool-input-delta")
        assert delta["inputTextDelta"] == "chunk1"

    async def test_begin_tool_call_still_works(self):
        """begin_tool_call() backward-compat: no deltas, input known upfront."""

        async def work(ctx):
            handle = await ctx.begin_tool_call("lookup", {"id": 1})
            await ctx.complete_tool_call(handle.toolCallId, "result")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-input-delta" not in types


class TestToolCalls:
    async def test_tool_call_complete(self):
        async def work(ctx):
            handle = await ctx.begin_tool_call("search", {"q": "test"})
            await ctx.complete_tool_call(handle.toolCallId, {"results": [1, 2]})
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-output-available" in types

    async def test_tool_call_fail(self):
        async def work(ctx):
            handle = await ctx.begin_tool_call("search", {"q": "test"})
            await ctx.fail_tool_call(handle.toolCallId, "timeout")
            await ctx.finish()

        events = await run_and_collect(work)
        error_ev = next(e for e in events if e["type"] == "tool-output-error")
        assert error_ev["error"] == "timeout"

    async def test_tool_call_ids_match(self):
        async def work(ctx):
            handle = await ctx.begin_tool_call("lookup", {"id": "x"})
            await ctx.complete_tool_call(handle.toolCallId, "done")
            await ctx.finish()

        events = await run_and_collect(work)
        start = next(e for e in events if e["type"] == "tool-input-start")
        available = next(e for e in events if e["type"] == "tool-input-available")
        output = next(e for e in events if e["type"] == "tool-output-available")
        assert start["toolCallId"] == available["toolCallId"] == output["toolCallId"]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class TestSteps:
    async def test_new_step_closes_text_opens_new(self):
        async def work(ctx):
            await ctx.write_text("step 1 text")
            await ctx.new_step()
            await ctx.write_text("step 2 text")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        # Two start-step and two finish-step events
        assert types.count("start-step") == 2
        assert types.count("finish-step") == 2

    async def test_full_flow_ordering(self):
        """reasoning → tool → text ordering matches the UIMessageStream spec."""

        async def work(ctx):
            await ctx.write_reasoning("thinking")
            await ctx.new_step()
            handle = await ctx.begin_tool_call("search", {"q": "x"})
            await ctx.complete_tool_call(handle.toolCallId, {})
            await ctx.new_step()
            await ctx.write_text("answer")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]

        r_start = types.index("reasoning-start")
        r_end = types.index("reasoning-end")
        fs1 = types.index("finish-step")
        ti_start = types.index("tool-input-start")
        to_avail = types.index("tool-output-available")
        fs2 = [i for i, t in enumerate(types) if t == "finish-step"][1]
        txt_start = types.index("text-start")
        txt_end = types.index("text-end")
        finish = types.index("finish")

        assert (
            r_start
            < r_end
            < fs1
            < ti_start
            < to_avail
            < fs2
            < txt_start
            < txt_end
            < finish
        )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestSources:
    async def test_write_source(self):
        async def work(ctx):
            await ctx.write_text("answer")
            await ctx.write_source("s1", "https://example.com", "Example")
            await ctx.finish()

        events = await run_and_collect(work)
        source_ev = next(e for e in events if e["type"] == "source-url")
        assert source_ev["sourceId"] == "s1"
        assert source_ev["url"] == "https://example.com"
        assert source_ev["title"] == "Example"

    async def test_source_without_title(self):
        async def work(ctx):
            await ctx.write_source("s2", "https://example.com")
            await ctx.finish()

        events = await run_and_collect(work)
        source_ev = next(e for e in events if e["type"] == "source-url")
        assert "title" not in source_ev  # excluded when None


# ---------------------------------------------------------------------------
# Custom data parts
# ---------------------------------------------------------------------------


class TestWriteData:
    async def test_write_data_emits_event(self):
        async def work(ctx):
            await ctx.write_data("weather", {"city": "SF", "temp": 72})
            await ctx.finish()

        events = await run_and_collect(work)
        data_ev = next(e for e in events if e["type"] == "data-weather")
        assert data_ev["data"] == {"city": "SF", "temp": 72}

    async def test_write_data_type_includes_name(self):
        async def work(ctx):
            await ctx.write_data("status", {"state": "loading"})
            await ctx.finish()

        events = await run_and_collect(work)
        assert any(e["type"] == "data-status" for e in events)

    async def test_write_data_with_id(self):
        async def work(ctx):
            await ctx.write_data("progress", {"pct": 50}, id="prog-1")
            await ctx.finish()

        events = await run_and_collect(work)
        data_ev = next(e for e in events if e["type"] == "data-progress")
        assert data_ev["id"] == "prog-1"

    async def test_write_data_transient_omits_from_collection(self):
        """Transient parts send the event but are not collected."""

        async def work(ctx):
            await ctx.write_data("ping", {"ts": 1}, transient=True)
            await ctx.finish()

        events = await run_and_collect(work)
        data_ev = next(e for e in events if e["type"] == "data-ping")
        assert data_ev["transient"] is True

    async def test_write_data_non_transient_omits_transient_field(self):
        async def work(ctx):
            await ctx.write_data("info", {"x": 1})
            await ctx.finish()

        events = await run_and_collect(work)
        data_ev = next(e for e in events if e["type"] == "data-info")
        assert "transient" not in data_ev

    async def test_write_data_invalid_name_raises(self):
        ctx = StreamContext()
        await ctx.store.set("x", 1)  # start context
        with pytest.raises(ValueError, match="Invalid data part name"):
            await ctx.write_data("bad name!", {"x": 1})
        await ctx.finish()

    async def test_write_data_auto_emits_start(self):
        async def work(ctx):
            await ctx.write_data("info", {"x": 1})
            await ctx.finish()

        events = await run_and_collect(work)
        assert events[0]["type"] == "start"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class TestFiles:
    async def test_write_file_emits_event(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/img.png", "image/png")
            await ctx.finish()

        events = await run_and_collect(work)
        file_ev = next(e for e in events if e["type"] == "file")
        assert file_ev["url"] == "https://example.com/img.png"
        assert file_ev["mediaType"] == "image/png"

    async def test_write_file_auto_emits_step(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/doc.pdf", "application/pdf")
            await ctx.finish()

        events = await run_and_collect(work)
        types = [e["type"] for e in events]
        assert "start-step" in types

    async def test_multiple_files(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/a.png", "image/png")
            await ctx.write_file("https://example.com/b.jpg", "image/jpeg")
            await ctx.finish()

        events = await run_and_collect(work)
        file_evs = [e for e in events if e["type"] == "file"]
        assert len(file_evs) == 2
        assert file_evs[0]["url"] == "https://example.com/a.png"
        assert file_evs[1]["url"] == "https://example.com/b.jpg"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_write_after_finish_raises(self):
        ctx = StreamContext()

        async def work():
            await ctx.finish()
            with pytest.raises(RuntimeError, match="finished"):
                ctx.write_event_to_stream(StartEvent(messageId="x"))

        await work()

    async def test_abort_terminates_stream(self):
        ctx = StreamContext()

        async def work():
            await ctx.write_text("hello")
            await ctx.abort()  # should end stream without finish event

        asyncio.create_task(work())
        events = await collect_stream(ctx)
        assert not any(e["type"] == "finish" for e in events)

    async def test_abort_emits_abort_event(self):
        """abort() must emit an abort event per the v6 spec."""

        async def work(ctx):
            await ctx.write_text("partial")
            await ctx.abort()

        events = await run_and_collect(work)
        assert any(e["type"] == "abort" for e in events)
        assert not any(e["type"] == "finish" for e in events)

    async def test_abort_with_reason(self):
        async def work(ctx):
            await ctx.abort(reason="user cancelled")

        events = await run_and_collect(work)
        abort_ev = next(e for e in events if e["type"] == "abort")
        assert abort_ev["reason"] == "user cancelled"

    async def test_abort_without_reason_omits_field(self):
        async def work(ctx):
            await ctx.abort()

        events = await run_and_collect(work)
        abort_ev = next(e for e in events if e["type"] == "abort")
        assert "reason" not in abort_ev

    async def test_abort_is_idempotent(self):
        async def work(ctx):
            await ctx.abort("first")
            await ctx.abort("second")  # no-op

        events = await run_and_collect(work)
        assert sum(1 for e in events if e["type"] == "abort") == 1

    async def test_error_emits_error_event_and_terminates(self):
        """ctx.error() emits an error event then [DONE] — no finish event."""

        async def work(ctx):
            await ctx.write_text("partial")
            await ctx.error("something went wrong")

        events = await run_and_collect(work)
        error_ev = next(e for e in events if e["type"] == "error")
        assert error_ev["errorText"] == "something went wrong"
        assert not any(e["type"] == "finish" for e in events)

    async def test_error_is_idempotent(self):
        """Calling ctx.error() twice only emits one error event."""

        async def work(ctx):
            await ctx.error("first")
            await ctx.error("second")  # no-op

        events = await run_and_collect(work)
        assert sum(1 for e in events if e["type"] == "error") == 1

    async def test_error_auto_emits_start(self):
        """ctx.error() without any prior writes still emits start first."""

        async def work(ctx):
            await ctx.error("oops")

        events = await run_and_collect(work)
        assert events[0]["type"] == "start"
        assert any(e["type"] == "error" for e in events)

    async def test_message_id_propagated(self):
        async def work(ctx):
            await ctx.finish()

        events = await run_and_collect(work, message_id="custom-msg-id")
        start = next(e for e in events if e["type"] == "start")
        assert start["messageId"] == "custom-msg-id"

    async def test_finish_reason_propagated(self):
        async def work(ctx):
            await ctx.finish(finish_reason="length")

        events = await run_and_collect(work)
        finish = next(e for e in events if e["type"] == "finish")
        assert finish["finishReason"] == "length"

    async def test_no_step_opened_if_nothing_written(self):
        async def work(ctx):
            await ctx.finish()

        events = await run_and_collect(work)
        assert not any(e["type"] == "start-step" for e in events)

    async def test_low_level_write_requires_started(self):
        """ctx.write(ev) auto-emits start before the raw event."""

        async def work(ctx):
            await ctx.write(TextDeltaEvent(id="x", delta="raw"))
            await ctx.finish()

        events = await run_and_collect(work)
        assert events[0]["type"] == "start"
        assert any(e["type"] == "text-delta" for e in events)


# ---------------------------------------------------------------------------
# StateStore integration
# ---------------------------------------------------------------------------


class TestStateStore:
    async def test_store_set_get(self):
        ctx = StreamContext()
        await ctx.store.set("user.name", "Alice")
        name = await ctx.store.get("user.name")
        assert name == "Alice"

    async def test_store_default(self):
        ctx = StreamContext()
        val = await ctx.store.get("missing.key", default=42)
        assert val == 42

    async def test_store_missing_raises(self):
        ctx = StreamContext()
        with pytest.raises(KeyError):
            await ctx.store.get("does.not.exist")

    async def test_store_shared_across_services(self):
        """Simulates db_service writing and llm_service reading from ctx.store."""
        ctx = StreamContext()

        async def db_service(c: StreamContext):
            await c.store.set("user.name", "Bob")
            await c.store.set("user.plan", "pro")

        async def llm_service(c: StreamContext) -> str:
            name = await c.store.get("user.name")
            plan = await c.store.get("user.plan")
            return f"Hello {name} ({plan})"

        await db_service(ctx)
        greeting = await llm_service(ctx)
        assert greeting == "Hello Bob (pro)"


# ---------------------------------------------------------------------------
# custom_information
# ---------------------------------------------------------------------------


class _RequestInfo(BaseModel):
    user_id: str
    rate_limit: int


class TestCustomInformation:
    async def test_default_is_none(self):
        """ctx.info is None when no custom_information is passed."""
        ctx = StreamContext()
        assert ctx.info is None

    async def test_stored_and_accessible(self):
        """ctx.info returns the exact Pydantic model passed at construction."""
        info = _RequestInfo(user_id="u_42", rate_limit=100)
        ctx: StreamContext[_RequestInfo] = StreamContext(custom_information=info)
        assert ctx.info is info
        assert ctx.info is not None
        assert ctx.info.user_id == "u_42"
        assert ctx.info.rate_limit == 100

    async def test_is_immutable(self):
        """ctx.info has no setter — attempts to assign raise AttributeError."""
        info = _RequestInfo(user_id="u_1", rate_limit=50)
        ctx: StreamContext[_RequestInfo] = StreamContext(custom_information=info)
        with pytest.raises(AttributeError):
            ctx.info = _RequestInfo(user_id="u_2", rate_limit=10)  # type: ignore[misc]

    async def test_survives_full_stream(self):
        """ctx.info remains accessible after the stream has been finished."""
        info = _RequestInfo(user_id="u_99", rate_limit=200)
        ctx: StreamContext[_RequestInfo] = StreamContext(custom_information=info)

        async def _work():
            await ctx.write_text("hello")
            await ctx.finish()

        asyncio.create_task(_work())
        await collect_stream(ctx)

        assert ctx.info is not None
        assert ctx.info.user_id == "u_99"

    async def test_available_in_service_layer(self):
        """Demonstrates the intended usage: info travels through service layers."""

        async def _service(c: StreamContext[_RequestInfo]) -> str:
            assert c.info is not None
            return f"Processing for user {c.info.user_id}"

        info = _RequestInfo(user_id="u_7", rate_limit=10)
        ctx: StreamContext[_RequestInfo] = StreamContext(custom_information=info)
        result = await _service(ctx)
        assert result == "Processing for user u_7"


# ---------------------------------------------------------------------------
# on_finish callback
# ---------------------------------------------------------------------------


class TestOnFinish:
    async def test_sync_callback_called_with_record(self):
        """A sync on_finish callback receives the fully populated StreamRecord."""
        received = []

        async def work(ctx):
            await ctx.write_text("hello")
            await ctx.finish()

        ctx = StreamContext(on_finish=lambda rec: received.append(rec))
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert len(received) == 1
        assert received[0].text == "hello"
        assert received[0].finish_reason == "stop"

    async def test_async_callback_called_with_record(self):
        """An async on_finish callback is awaited and receives the record."""
        received = []

        async def _on_finish(rec):
            received.append(rec)

        async def work(ctx):
            await ctx.write_text("async cb")
            await ctx.finish()

        ctx = StreamContext(on_finish=_on_finish)
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert len(received) == 1
        assert received[0].text == "async cb"

    async def test_on_finish_auto_enables_collect(self):
        """Passing on_finish without collect=True still populates ctx.record."""
        record_holder = []

        async def work(ctx):
            await ctx.write_text("captured")
            await ctx.finish()

        ctx = StreamContext(on_finish=lambda rec: record_holder.append(rec))
        assert ctx.record is not None, "record should be created when on_finish is set"
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert ctx.record is not None
        assert ctx.record.text == "captured"

    async def test_callback_exception_does_not_prevent_stream_termination(self):
        """If the callback raises, the stream still terminates normally."""

        def bad_callback(rec):
            raise ValueError("db write failed")

        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = StreamContext(on_finish=bad_callback)
        asyncio.create_task(work(ctx))
        # collect_stream should complete without raising
        events = await collect_stream(ctx)
        assert any(e["type"] == "finish" for e in events)

    async def test_async_callback_exception_swallowed(self):
        """Async callback exception is also swallowed."""

        async def bad_async_callback(rec):
            raise RuntimeError("async failure")

        async def work(ctx):
            await ctx.write_text("test")
            await ctx.finish()

        ctx = StreamContext(on_finish=bad_async_callback)
        asyncio.create_task(work(ctx))
        events = await collect_stream(ctx)
        assert any(e["type"] == "finish" for e in events)

    async def test_on_finish_parameter_on_finish_method(self):
        """on_finish passed directly to finish() is called after the stream.

        When on_finish is provided only at finish()-call time (not construction),
        collection wasn't enabled during the stream, so the record holds only
        the finish_reason — text already emitted won't be captured.
        """
        received = []

        async def work(ctx):
            await ctx.write_text("per-call")
            await ctx.finish(on_finish=lambda rec: received.append(rec))

        ctx = StreamContext()
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert len(received) == 1
        # Record was created lazily inside finish(); text was not collected.
        assert received[0].finish_reason == "stop"
        assert received[0].text == ""

    async def test_finish_on_finish_takes_priority_over_constructor(self):
        """on_finish on finish() overrides the constructor-level callback."""
        constructor_calls = []
        finish_calls = []

        async def work(ctx):
            await ctx.finish(on_finish=lambda rec: finish_calls.append(rec))

        ctx = StreamContext(on_finish=lambda rec: constructor_calls.append(rec))
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert len(finish_calls) == 1
        assert len(constructor_calls) == 0

    async def test_on_finish_called_only_once_on_repeated_finish(self):
        """Calling finish() twice must not invoke the callback twice."""
        calls = []

        async def work(ctx):
            await ctx.finish()
            await ctx.finish()  # no-op

        ctx = StreamContext(on_finish=lambda rec: calls.append(1))
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert len(calls) == 1

    async def test_on_finish_record_contains_finish_reason(self):
        """Record passed to callback has the finish_reason populated."""
        received = []

        async def work(ctx):
            await ctx.finish(finish_reason="length")

        ctx = StreamContext(on_finish=lambda rec: received.append(rec))
        asyncio.create_task(work(ctx))
        await collect_stream(ctx)

        assert received[0].finish_reason == "length"


# ---------------------------------------------------------------------------
# pytest-asyncio config
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio
