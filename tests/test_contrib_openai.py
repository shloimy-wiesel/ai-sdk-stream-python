"""
Tests for contrib.openai.consume_openai_stream().

Uses simple duck-typed mock objects — no openai package required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.contrib.openai import ConsumeResult, consume_openai_stream

# ---------------------------------------------------------------------------
# Mock helpers — duck-typed OpenAI ChatCompletionChunk objects
# ---------------------------------------------------------------------------


def _fn(name: str | None = None, arguments: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


def _tc(
    index: int,
    id: str | None = None,
    function: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=id, function=function)


def _delta(
    content: str | None = None,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )


def _choice(
    delta: SimpleNamespace,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _chunk(
    choices: list | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(choices=choices or [], usage=usage)


def _usage(prompt: int = 10, completion: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


async def _make_stream(*chunks: SimpleNamespace):
    """Async generator of mock chunks."""
    for chunk in chunks:
        yield chunk


async def run_consume(
    chunks: list,
    *,
    stream_tool_deltas: bool = False,
    collect: bool = False,
) -> tuple[ConsumeResult, list[dict]]:
    """Run consume_openai_stream and return (result, list of SSE event dicts)."""
    ctx = StreamContext(collect=collect)
    events: list[dict] = []

    async def work(ctx: StreamContext) -> None:
        result = await consume_openai_stream(
            _make_stream(*chunks),
            ctx,
            stream_tool_deltas=stream_tool_deltas,
        )
        ctx._result = result  # stash for inspection
        await ctx.finish(finish_reason=result.finish_reason or "stop")

    asyncio.create_task(work(ctx))

    import json as _json

    async for raw in ctx.stream():
        stripped = raw.strip()
        if stripped == "data: [DONE]":
            break
        body = stripped.removeprefix("data: ")
        if body:
            events.append(_json.loads(body))

    return ctx._result, events  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TestText:
    async def test_single_text_chunk(self):
        chunks = [
            _chunk([_choice(_delta(content="hello"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        types = [e["type"] for e in events]
        assert "text-delta" in types
        td = next(e for e in events if e["type"] == "text-delta")
        assert td["delta"] == "hello"

    async def test_multiple_text_deltas_concatenated(self):
        chunks = [
            _chunk([_choice(_delta(content="foo"))]),
            _chunk([_choice(_delta(content="bar"))]),
            _chunk([_choice(_delta(content=None), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "foobar"
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        assert len(text_deltas) == 2

    async def test_empty_content_not_written(self):
        chunks = [
            _chunk([_choice(_delta(content=None), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == ""
        assert not any(e["type"] == "text-delta" for e in events)

    async def test_chunk_with_no_choices_skipped(self):
        chunks = [
            _chunk([]),  # empty choices
            _chunk([_choice(_delta(content="hi"), finish_reason="stop")]),
        ]
        result, _ = await run_consume(chunks)
        assert result.content == "hi"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    async def test_reasoning_field(self):
        chunks = [
            _chunk([_choice(_delta(reasoning="think"))]),
            _chunk([_choice(_delta(content="answer"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "answer"
        reasoning_deltas = [e for e in events if e["type"] == "reasoning-delta"]
        assert len(reasoning_deltas) == 1
        assert reasoning_deltas[0]["delta"] == "think"

    async def test_reasoning_content_field(self):
        """OpenAI o-models use delta.reasoning_content."""
        chunks = [
            _chunk([_choice(_delta(reasoning_content="step 1"))]),
            _chunk([_choice(_delta(content="done"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "done"
        assert any(e["type"] == "reasoning-delta" for e in events)

    async def test_reasoning_content_preferred_over_reasoning(self):
        """reasoning_content takes priority when both fields are present."""
        d = SimpleNamespace(
            content=None,
            reasoning="fallback",
            reasoning_content="preferred",
            tool_calls=None,
        )
        chunks = [
            _chunk([_choice(d, finish_reason="stop")]),
        ]
        _, events = await run_consume(chunks)
        reasoning_deltas = [e for e in events if e["type"] == "reasoning-delta"]
        assert reasoning_deltas[0]["delta"] == "preferred"


# ---------------------------------------------------------------------------
# Finish reason & usage
# ---------------------------------------------------------------------------


class TestMetadata:
    async def test_finish_reason_extracted(self):
        chunks = [
            _chunk([_choice(_delta(content="hi"))]),
            _chunk([_choice(_delta(), finish_reason="length")]),
        ]
        result, _ = await run_consume(chunks)
        assert result.finish_reason == "length"

    async def test_finish_reason_none_when_absent(self):
        chunks = [
            _chunk([_choice(_delta(content="hi"))]),
        ]
        result, _ = await run_consume(chunks)
        assert result.finish_reason is None

    async def test_usage_extracted(self):
        chunks = [
            _chunk([_choice(_delta(content="x"), finish_reason="stop")]),
            _chunk([], usage=_usage(prompt=5, completion=10)),
        ]
        result, _ = await run_consume(chunks)
        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        }

    async def test_usage_none_when_not_provided(self):
        chunks = [_chunk([_choice(_delta(content="hi"), finish_reason="stop")])]
        result, _ = await run_consume(chunks)
        assert result.usage is None


# ---------------------------------------------------------------------------
# Tool calls — non-streaming (stream_tool_deltas=False)
# ---------------------------------------------------------------------------


class TestToolCallsNonStreaming:
    async def test_single_tool_call_emitted(self):
        chunks = [
            # First chunk: id + name
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_1",
                                    function=_fn(name="search", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Argument chunks
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments='{"q":'))]))]
            ),
            _chunk(
                [
                    _choice(
                        _delta(tool_calls=[_tc(0, function=_fn(arguments='"cats"}'))])
                    )
                ]
            ),
            # Finish
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["name"] == "search"
        assert json.loads(tc["arguments"]) == {"q": "cats"}

        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-input-delta" not in types

        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["toolName"] == "search"
        assert avail["input"] == {"q": "cats"}

    async def test_multiple_tool_calls_emitted_in_order(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_0",
                                    function=_fn(name="toolA", arguments=""),
                                ),
                                _tc(
                                    1,
                                    id="call_1",
                                    function=_fn(name="toolB", arguments=""),
                                ),
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, function=_fn(arguments='{"x":1}')),
                                _tc(1, function=_fn(arguments='{"y":2}')),
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "toolA"
        assert result.tool_calls[1]["name"] == "toolB"

        avail_events = [e for e in events if e["type"] == "tool-input-available"]
        assert len(avail_events) == 2
        assert avail_events[0]["toolName"] == "toolA"
        assert avail_events[1]["toolName"] == "toolB"

    async def test_tool_call_uses_provided_id(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="myid",
                                    function=_fn(name="fn", arguments="{}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.tool_calls[0]["id"] == "myid"
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["toolCallId"] == "myid"

    async def test_malformed_json_args_fallback_to_empty_dict(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="x",
                                    function=_fn(name="fn", arguments="not-json"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks)
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["input"] == {}

    async def test_zero_arg_tool(self):
        """Tool with empty arguments `{}` should emit input={}."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0, id="z", function=_fn(name="noop", arguments="{}")
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks)
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["input"] == {}


# ---------------------------------------------------------------------------
# Tool calls — streaming (stream_tool_deltas=True)
# ---------------------------------------------------------------------------


class TestToolCallsStreaming:
    async def test_streaming_emits_deltas(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_1",
                                    function=_fn(name="search", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments='{"q":'))]))]
            ),
            _chunk(
                [
                    _choice(
                        _delta(tool_calls=[_tc(0, function=_fn(arguments='"cats"}'))])
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-delta" in types
        assert "tool-input-available" in types

        deltas = [e for e in events if e["type"] == "tool-input-delta"]
        assert len(deltas) == 2
        combined = "".join(d["inputTextDelta"] for d in deltas)
        assert json.loads(combined) == {"q": "cats"}

    async def test_streaming_tool_call_ids_consistent(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, id="cid", function=_fn(name="fn", arguments=""))
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments="{}"))]))]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)

        start = next(e for e in events if e["type"] == "tool-input-start")
        deltas = [e for e in events if e["type"] == "tool-input-delta"]
        avail = next(e for e in events if e["type"] == "tool-input-available")

        cid = start["toolCallId"]
        assert all(d["toolCallId"] == cid for d in deltas)
        assert avail["toolCallId"] == cid

    async def test_streaming_fallback_for_no_arguments(self):
        """Tool call with no arguments chunks falls back to begin_tool_call."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0, id="z", function=_fn(name="noop", arguments=None)
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-input-delta" not in types


# ---------------------------------------------------------------------------
# Mixed text + tool calls
# ---------------------------------------------------------------------------


class TestMixed:
    async def test_text_before_tool_calls(self):
        chunks = [
            _chunk([_choice(_delta(content="Let me search."))]),
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="c1",
                                    function=_fn(name="search", arguments='{"q":"x"}'),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        types = [e["type"] for e in events]
        assert "text-delta" in types
        assert "tool-input-available" in types
