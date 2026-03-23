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

    async def test_streaming_tool_input_recorded(self):
        """start_tool_input / finish_tool_input updates the record correctly."""

        async def work(ctx):
            handle = await ctx.start_tool_input("search")
            await ctx.stream_tool_input_delta(handle.toolCallId, '{"q":"cats"}')
            await ctx.finish_tool_input(handle.toolCallId, "search", {"q": "cats"})
            await ctx.complete_tool_call(handle.toolCallId, ["cat1"])
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.tool_calls) == 1
        tc = ctx.record.tool_calls[0]
        assert tc.tool_name == "search"
        assert tc.input == {"q": "cats"}  # filled in by finish_tool_input
        assert tc.output == ["cat1"]

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
# Custom data part collection
# ---------------------------------------------------------------------------


class TestCollectDataParts:
    async def test_data_part_recorded(self):
        async def work(ctx):
            await ctx.write_data("weather", {"city": "SF"})
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.data_parts) == 1
        dp = ctx.record.data_parts[0]
        assert dp.name == "weather"
        assert dp.data == {"city": "SF"}
        assert dp.id is None

    async def test_data_part_with_id_recorded(self):
        async def work(ctx):
            await ctx.write_data("progress", {"pct": 80}, id="prog-1")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        dp = ctx.record.data_parts[0]
        assert dp.id == "prog-1"

    async def test_transient_data_part_not_collected(self):
        async def work(ctx):
            await ctx.write_data("ping", {"ts": 1}, transient=True)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.data_parts == []

    async def test_data_parts_in_to_dict(self):
        async def work(ctx):
            await ctx.write_data("status", {"state": "done"}, id="s-1")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        d = ctx.record.to_dict()
        assert len(d["data_parts"]) == 1
        assert d["data_parts"][0]["name"] == "status"
        assert d["data_parts"][0]["data"] == {"state": "done"}
        assert d["data_parts"][0]["id"] == "s-1"

    async def test_data_parts_empty_by_default(self):
        async def work(ctx):
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.data_parts == []


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
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCounting:
    async def test_answer_tokens_default_len(self):
        async def work(ctx):
            await ctx.write_text("hello")  # len=5
            await ctx.write_text(" world")  # len=6
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.answer_tokens == 11

    async def test_reasoning_tokens_default_len(self):
        async def work(ctx):
            await ctx.write_reasoning("think")  # len=5
            await ctx.write_reasoning("!")  # len=1
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.reasoning_tokens == 6

    async def test_custom_count_func(self):
        def word_count(s: str) -> int:
            return len(s.split())

        async def work(ctx):
            await ctx.write_text("one two three")  # 3 words
            await ctx.write_reasoning("four five")  # 2 words
            await ctx.finish()

        ctx = await run_collecting(work, collect=True, count_func=word_count)
        assert ctx.record is not None
        assert ctx.record.answer_tokens == 3
        assert ctx.record.reasoning_tokens == 2

    async def test_total_output_tokens(self):
        async def work(ctx):
            await ctx.write_reasoning("abc")  # 3
            await ctx.write_text("de")  # 2
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.total_output_tokens == 5

    async def test_total_tokens_none_without_prompt(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.total_tokens is None

    async def test_total_tokens_with_prompt(self):
        async def work(ctx):
            await ctx.write_text("hi")  # len=2
            await ctx.set_usage(prompt_tokens=10)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.prompt_tokens == 10
        assert ctx.record.total_tokens == 12

    async def test_set_usage_overrides_auto_counted(self):
        async def work(ctx):
            await ctx.write_text("hello world")  # auto: 11 chars
            await ctx.set_usage(answer_tokens=3)  # override with exact
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.answer_tokens == 3

    async def test_set_usage_partial_update(self):
        async def work(ctx):
            await ctx.write_reasoning("think")  # auto: 5
            await ctx.write_text("answer")  # auto: 6
            await ctx.set_usage(prompt_tokens=20, answer_tokens=2)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.prompt_tokens == 20
        assert ctx.record.answer_tokens == 2
        assert ctx.record.reasoning_tokens == 5  # unchanged

    async def test_set_usage_noop_when_not_collecting(self):
        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.set_usage(prompt_tokens=10)
            await ctx.finish()

        ctx = await run_collecting(work, collect=False)
        assert ctx.record is None  # no error raised

    async def test_tokens_in_to_dict(self):
        async def work(ctx):
            await ctx.write_reasoning("ab")  # 2
            await ctx.write_text("cde")  # 3
            await ctx.set_usage(prompt_tokens=7)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        d = ctx.record.to_dict()
        assert d["reasoning_tokens"] == 2
        assert d["answer_tokens"] == 3
        assert d["prompt_tokens"] == 7
        assert d["total_output_tokens"] == 5
        assert d["total_tokens"] == 12

    async def test_no_counting_when_not_collecting(self):
        """count_func is ignored when collect=False — no errors."""

        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.finish()

        ctx = await run_collecting(work, collect=False, count_func=lambda s: len(s))
        assert ctx.record is None


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-call collect=False opt-out
# ---------------------------------------------------------------------------


class TestPerCallCollectFalse:
    async def test_text_not_recorded_when_collect_false(self):
        async def work(ctx):
            await ctx.write_text("recorded")
            await ctx.write_text("not recorded", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.text == "recorded"

    async def test_text_collect_false_still_streamed(self):
        """The delta is emitted to the SSE stream even when collect=False."""
        events: list[str] = []

        async def work(ctx):
            await ctx.write_text("hidden", collect=False)
            await ctx.finish()

        ctx = StreamContext(collect=True)
        asyncio.create_task(work(ctx))
        async for chunk in ctx.stream():
            events.append(chunk)

        # Should contain text-delta with "hidden"
        assert any("hidden" in e for e in events)
        # But record should NOT contain it
        assert ctx.record is not None
        assert ctx.record.text == ""

    async def test_text_collect_false_no_token_counting(self):
        async def work(ctx):
            await ctx.write_text("counted", collect=True)
            await ctx.write_text("not counted", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.answer_tokens == len("counted")

    async def test_reasoning_not_recorded_when_collect_false(self):
        async def work(ctx):
            await ctx.write_reasoning("recorded")
            await ctx.write_reasoning("not recorded", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.reasoning == "recorded"

    async def test_reasoning_collect_false_no_token_counting(self):
        async def work(ctx):
            await ctx.write_reasoning("counted")
            await ctx.write_reasoning("not counted", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.reasoning_tokens == len("counted")

    async def test_source_not_recorded_when_collect_false(self):
        async def work(ctx):
            await ctx.write_source("s1", "https://a.com", collect=True)
            await ctx.write_source("s2", "https://b.com", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.sources) == 1
        assert ctx.record.sources[0].source_id == "s1"

    async def test_file_not_recorded_when_collect_false(self):
        async def work(ctx):
            await ctx.write_file("https://a.com/img.png", "image/png")
            await ctx.write_file("https://b.com/tmp.png", "image/png", collect=False)
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.files) == 1
        assert ctx.record.files[0].url == "https://a.com/img.png"

    async def test_begin_tool_call_not_recorded_when_collect_false(self):
        async def work(ctx):
            h1 = await ctx.begin_tool_call("visible", {"q": "a"})
            await ctx.complete_tool_call(h1.toolCallId, "result-a")
            h2 = await ctx.begin_tool_call("hidden", {"q": "b"}, collect=False)
            await ctx.complete_tool_call(h2.toolCallId, "result-b")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.tool_calls) == 1
        assert ctx.record.tool_calls[0].tool_name == "visible"
        assert ctx.record.tool_calls[0].output == "result-a"

    async def test_start_tool_input_not_recorded_when_collect_false(self):
        async def work(ctx):
            h = await ctx.start_tool_input("hidden", collect=False)
            await ctx.stream_tool_input_delta(h.toolCallId, '{"q":"x"}')
            await ctx.finish_tool_input(h.toolCallId, "hidden", {"q": "x"})
            await ctx.complete_tool_call(h.toolCallId, "out")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert len(ctx.record.tool_calls) == 0

    async def test_collect_false_noop_when_collection_disabled(self):
        """collect=False doesn't crash when ctx.record is None."""

        async def work(ctx):
            await ctx.write_text("hi", collect=False)
            await ctx.write_reasoning("think", collect=False)
            await ctx.write_source("s1", "https://a.com", collect=False)
            await ctx.write_file("https://a.com/f.png", "image/png", collect=False)
            h = await ctx.begin_tool_call("t", {"x": 1}, collect=False)
            await ctx.complete_tool_call(h.toolCallId, "out")
            await ctx.finish()

        ctx = await run_collecting(work, collect=False)
        assert ctx.record is None


# ---------------------------------------------------------------------------
# Per-call collect=True on non-collecting context raises RuntimeError
# ---------------------------------------------------------------------------


class TestPerCallCollectTrueOnNonCollecting:
    async def test_write_text_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.write_text("hi", collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_write_reasoning_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.write_reasoning("think", collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_write_source_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.write_source("s1", "https://a.com", collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_write_file_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.write_file("https://a.com/f.png", "image/png", collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_begin_tool_call_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.begin_tool_call("t", {"x": 1}, collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_start_tool_input_collect_true_raises(self):
        ctx = StreamContext(collect=False)
        import pytest

        with pytest.raises(RuntimeError, match="collect=True was passed"):
            await ctx.start_tool_input("t", collect=True)
        await ctx.finish()
        async for _ in ctx.stream():
            pass

    async def test_collect_true_ok_when_context_collecting(self):
        """Explicit collect=True is fine when context has collect=True."""

        async def work(ctx):
            await ctx.write_text("hi", collect=True)
            await ctx.write_reasoning("think", collect=True)
            await ctx.write_source("s1", "https://a.com", collect=True)
            await ctx.write_file("https://a.com/f.png", "image/png", collect=True)
            h = await ctx.begin_tool_call("t", {"x": 1}, collect=True)
            await ctx.complete_tool_call(h.toolCallId, "out")
            await ctx.finish()

        ctx = await run_collecting(work, collect=True)
        assert ctx.record is not None
        assert ctx.record.text == "hi"
        assert ctx.record.reasoning == "think"
        assert len(ctx.record.sources) == 1
        assert len(ctx.record.files) == 1
        assert len(ctx.record.tool_calls) == 1

    async def test_default_none_no_error_when_not_collecting(self):
        """Default collect=None silently skips when context has no record."""

        async def work(ctx):
            await ctx.write_text("hi")
            await ctx.write_reasoning("think")
            await ctx.write_source("s1", "https://a.com")
            await ctx.write_file("https://a.com/f.png", "image/png")
            h = await ctx.begin_tool_call("t", {"x": 1})
            await ctx.complete_tool_call(h.toolCallId, "out")
            await ctx.finish()

        ctx = await run_collecting(work, collect=False)
        assert ctx.record is None


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
