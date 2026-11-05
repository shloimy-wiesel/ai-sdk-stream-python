"""
Tests for ai_sdk_stream_python.testing — the ready-made fake stream helpers.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.testing import (
    DEFAULT_REASONING,
    DEFAULT_TEXT,
    DEFAULT_TOOL_CALLS,
    FakeToolCall,
    test_stream,
    test_stream_response,
)


async def _events(ctx: StreamContext, **kwargs) -> list[dict]:
    await ctx.run(lambda c: test_stream(c, delay=0, **kwargs))
    events = []
    async for chunk in ctx.stream():
        if chunk.strip() == "data: [DONE]":
            break
        events.append(json.loads(chunk.removeprefix("data: ").strip()))
    return events


def _joined(events: list[dict], type_: str) -> str:
    return "".join(e["delta"] for e in events if e["type"] == type_)


async def test_text_only_by_default():
    events = await _events(StreamContext())
    types = [e["type"] for e in events]

    assert types[0] == "start"
    assert types[-1] == "finish"
    assert _joined(events, "text-delta") == DEFAULT_TEXT
    assert "reasoning-start" not in types
    assert "tool-input-start" not in types


async def test_with_reasoning():
    events = await _events(StreamContext(), reasoning=True)
    types = [e["type"] for e in events]

    assert _joined(events, "reasoning-delta") == DEFAULT_REASONING
    assert types.index("reasoning-end") < types.index("text-start")


async def test_with_tool_calls():
    events = await _events(StreamContext(), tool_calls=True)
    types = [e["type"] for e in events]

    available = [e for e in events if e["type"] == "tool-input-available"]
    outputs = [e for e in events if e["type"] == "tool-output-available"]
    assert [e["toolName"] for e in available] == [
        t.tool_name for t in DEFAULT_TOOL_CALLS
    ]
    assert [e["input"] for e in available] == [t.input for t in DEFAULT_TOOL_CALLS]
    assert [e["output"] for e in outputs] == [t.output for t in DEFAULT_TOOL_CALLS]

    # Streamed input deltas reassemble into the final JSON input.
    first_id = available[0]["toolCallId"]
    streamed = "".join(
        e["inputTextDelta"]
        for e in events
        if e["type"] == "tool-input-delta" and e["toolCallId"] == first_id
    )
    assert json.loads(streamed) == DEFAULT_TOOL_CALLS[0].input

    # Answer text comes in a new step after the tools.
    assert types.count("start-step") == 2
    assert types.index("tool-output-available") < types.index("text-start")


async def test_all_options_order():
    events = await _events(StreamContext(), reasoning=True, tool_calls=True)
    types = [e["type"] for e in events]

    assert (
        types.index("reasoning-start")
        < types.index("tool-input-start")
        < types.index("text-start")
        < types.index("finish")
    )


async def test_custom_content_and_collect():
    tool = FakeToolCall(tool_name="ping", input={"host": "x"}, output="pong")
    ctx = StreamContext(collect=True)
    await _events(
        ctx,
        reasoning=True,
        tool_calls=True,
        text="Hi there",
        reasoning_text="Thinking hard",
        tools=[tool],
    )

    assert ctx.record is not None
    assert ctx.record.text == "Hi there"
    assert ctx.record.reasoning == "Thinking hard"
    assert len(ctx.record.tool_calls) == 1
    assert ctx.record.tool_calls[0].input == {"host": "x"}
    assert ctx.record.tool_calls[0].output == "pong"


async def test_finish_false_leaves_stream_open():
    ctx = StreamContext()
    await test_stream(ctx, delay=0, finish=False)
    assert not ctx.is_finished
    await ctx.write_text(" more")
    await ctx.finish()


async def test_response_helper():
    response = await test_stream_response(delay=0, reasoning=True)
    assert response.media_type == "text/event-stream"
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"

    body = ""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, str) else bytes(chunk).decode()
    assert '"type":"reasoning-start"' in body.replace(" ", "")
    assert body.rstrip().endswith("data: [DONE]")


def test_not_collected_by_pytest():
    assert test_stream.__test__ is False  # type: ignore[attr-defined]
    assert test_stream_response.__test__ is False  # type: ignore[attr-defined]


async def test_context_method():
    ctx = StreamContext(collect=True)
    await ctx.run(lambda c: c.test_stream(delay=0, reasoning=True, tool_calls=True))
    types = []
    async for chunk in ctx.stream():
        if chunk.strip() == "data: [DONE]":
            break
        types.append(json.loads(chunk.removeprefix("data: ").strip())["type"])

    assert types[-1] == "finish"
    assert "reasoning-start" in types
    assert "tool-output-available" in types
    assert ctx.record is not None
    assert ctx.record.text == DEFAULT_TEXT
