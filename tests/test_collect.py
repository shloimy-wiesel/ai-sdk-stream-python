"""
Tests for StreamContext collect=True — stream data collection feature.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python import StreamContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def run_collecting(producer, **ctx_kwargs) -> StreamContext:
    """Run producer(ctx) as a background task, drain the stream, return ctx."""
    ctx = StreamContext(**ctx_kwargs)
    asyncio.create_task(producer(ctx))
    async for _ in ctx.stream():
        pass
    return ctx


# ---------------------------------------------------------------------------
# collect disabled
# ---------------------------------------------------------------------------


class TestCollectDisabled:
    async def test_record_is_none_by_default(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work)
        assert ctx.record is None

    async def test_record_is_none_explicit_false(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work, collect=False)
        assert ctx.record is None


# ---------------------------------------------------------------------------
# Text collection
# ---------------------------------------------------------------------------


class TestCollectText:
    async def test_text_accumulated(self):
        async def work(ctx):
            await ctx.write_text("hello ")
            await ctx.write_text("world")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.text == "hello world"

    async def test_text_empty_when_no_writes(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.text == ""

    async def test_text_is_live_during_stream(self):
        """Collection happens incrementally, not only after finish."""
        snapshots: list[str] = []

        async def work(ctx):
            await ctx.write_text("a")
            assert ctx.record is not None
            snapshots.append(ctx.record.text)
            await ctx.write_text("b")
            snapshots.append(ctx.record.text)
            await ctx.finish()

        await run_collecting(work, collect=True)
        assert snapshots == ["a", "ab"]


# ---------------------------------------------------------------------------
# Reasoning collection
# ---------------------------------------------------------------------------


class TestCollectReasoning:
    async def test_reasoning_accumulated(self):
        async def work(ctx):
            await ctx.write_reasoning("think ")
            await ctx.write_reasoning("more")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.reasoning == "think more"

    async def test_reasoning_separate_from_text(self):
        async def work(ctx):
            await ctx.write_reasoning("thought")
            await ctx.write_text("answer")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.reasoning == "thought"
        assert ctx.record.text == "answer"


# ---------------------------------------------------------------------------
# Tool call collection
# ---------------------------------------------------------------------------


class TestCollectToolCalls:
    async def test_tool_call_recorded_on_begin(self):
        async def work(ctx):
            await ctx.begin_tool_call("search", {"query": "cats"})
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.tool_calls) == 1
        tc = ctx.record.tool_calls[0]
        assert tc.tool_name == "search"
        assert tc.input == {"query": "cats"}
        assert tc.output is None
        assert tc.error is None

    async def test_complete_tool_call_fills_output(self):
        async def work(ctx):
            handle = await ctx.begin_tool_call("search", {"query": "cats"})
            await ctx.complete_tool_call(handle.toolCallId, ["cat1", "cat2"])
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        tc = ctx.record.tool_calls[0]
        assert tc.output == ["cat1", "cat2"]
        assert tc.error is None

    async def test_fail_tool_call_fills_error(self):
        async def work(ctx):
            handle = await ctx.begin_tool_call("search", {"query": "cats"})
            await ctx.fail_tool_call(handle.toolCallId, "timeout")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        tc = ctx.record.tool_calls[0]
        assert tc.error == "timeout"
        assert tc.output is None

    async def test_multiple_tool_calls_independent(self):
        async def work(ctx):
            h1 = await ctx.begin_tool_call("search", {"q": "a"})
            h2 = await ctx.begin_tool_call("lookup", {"id": 42})
            await ctx.complete_tool_call(h1.toolCallId, "result-a")
            await ctx.complete_tool_call(h2.toolCallId, "result-b")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.tool_calls) == 2
        assert ctx.record.tool_calls[0].tool_name == "search"
        assert ctx.record.tool_calls[0].output == "result-a"
        assert ctx.record.tool_calls[1].tool_name == "lookup"
        assert ctx.record.tool_calls[1].output == "result-b"

    async def test_completing_b_does_not_overwrite_a(self):
        async def work(ctx):
            await ctx.begin_tool_call("toolA", {"x": 1})
            h2 = await ctx.begin_tool_call("toolB", {"x": 2})
            await ctx.complete_tool_call(h2.toolCallId, "out-b")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        a_rec = next(tc for tc in ctx.record.tool_calls if tc.tool_name == "toolA")
        b_rec = next(tc for tc in ctx.record.tool_calls if tc.tool_name == "toolB")
        assert a_rec.output is None
        assert b_rec.output == "out-b"


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------


class TestCollectSources:
    async def test_source_recorded(self):
        async def work(ctx):
            await ctx.write_source("src-1", "https://example.com", title="Example")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.sources) == 1
        s = ctx.record.sources[0]
        assert s.source_id == "src-1"
        assert s.url == "https://example.com"
        assert s.title == "Example"

    async def test_source_without_title(self):
        async def work(ctx):
            await ctx.write_source("src-1", "https://example.com")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.sources[0].title is None

    async def test_multiple_sources(self):
        async def work(ctx):
            await ctx.write_source("src-1", "https://a.com")
            await ctx.write_source("src-2", "https://b.com")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.sources) == 2
        assert ctx.record.sources[0].source_id == "src-1"
        assert ctx.record.sources[1].source_id == "src-2"


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


class TestCollectFiles:
    async def test_file_recorded(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/img.png", "image/png")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.files) == 1
        f = ctx.record.files[0]
        assert f.url == "https://example.com/img.png"
        assert f.media_type == "image/png"

    async def test_multiple_files_recorded(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/a.png", "image/png")
            await ctx.write_file("https://example.com/b.pdf", "application/pdf")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.files) == 2

    async def test_files_in_to_dict(self):
        async def work(ctx):
            await ctx.write_file("https://example.com/img.png", "image/png")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        d = ctx.record.to_dict()
        assert len(d["files"]) == 1
        assert d["files"][0]["url"] == "https://example.com/img.png"
        assert d["files"][0]["media_type"] == "image/png"

    async def test_files_empty_by_default(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.files == []


# ---------------------------------------------------------------------------
# Step count
# ---------------------------------------------------------------------------


class TestCollectSteps:
    async def test_single_step_count(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.step_count == 1

    async def test_multi_step_count(self):
        async def work(ctx):
            await ctx.write_text("step 1")
            await ctx.new_step()
            await ctx.write_text("step 2")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.step_count == 2

    async def test_no_step_when_nothing_written(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.step_count == 0


# ---------------------------------------------------------------------------
# Finish reason
# ---------------------------------------------------------------------------


class TestCollectFinishReason:
    async def test_finish_reason_default_stop(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.finish_reason == "stop"

    async def test_finish_reason_length(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish(finish_reason="length")

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.finish_reason == "length"

    async def test_finish_reason_none_before_finish(self):
        """finish_reason starts as None and is set only when finish() is called."""
        ctx = StreamContext(collect=True)
        assert ctx.record is not None
        assert ctx.record.finish_reason is None
        await ctx.write_text("hi")
        assert ctx.record.finish_reason is None
        await ctx.finish()
        assert ctx.record.finish_reason == "stop"
        # drain queue
        async for _ in ctx.stream():
            pass


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


class TestCollectToDict:
    async def test_to_dict_structure(self):
        async def work(ctx):
            await ctx.write_reasoning("think")
            await ctx.write_text("answer")
            handle = await ctx.begin_tool_call("calc", {"expr": "1+1"})
            await ctx.complete_tool_call(handle.toolCallId, 2)
            await ctx.write_source("s1", "https://example.com", title="Ex")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        d = ctx.record.to_dict()

        assert d["text"] == "answer"
        assert d["reasoning"] == "think"
        assert d["finish_reason"] == "stop"
        assert d["step_count"] == 1

        assert len(d["tool_calls"]) == 1
        tc = d["tool_calls"][0]
        assert tc["tool_name"] == "calc"
        assert tc["input"] == {"expr": "1+1"}
        assert tc["output"] == 2
        assert tc["error"] is None

        assert len(d["sources"]) == 1
        assert d["sources"][0]["url"] == "https://example.com"
        assert d["sources"][0]["title"] == "Ex"

    async def test_to_dict_empty(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        d = ctx.record.to_dict()
        assert d["text"] == ""
        assert d["reasoning"] == ""
        assert d["tool_calls"] == []
        assert d["sources"] == []
        assert d["step_count"] == 0


# ---------------------------------------------------------------------------
# Message ID
# ---------------------------------------------------------------------------


class TestCollectMessageId:
    async def test_record_carries_message_id(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, message_id="test-msg-id", collect=True)
        assert ctx.record is not None
        assert ctx.record.message_id == "test-msg-id"


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------


class TestCollectAbort:
    async def test_abort_does_not_set_finish_reason(self):
        async def work(ctx):
            await ctx.write_text("partial")
            await ctx.abort()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.finish_reason is None
        assert ctx.record.text == "partial"

    async def test_abort_preserves_collected_text(self):
        async def work(ctx):
            await ctx.write_text("hello")
            await ctx.abort()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.text == "hello"
