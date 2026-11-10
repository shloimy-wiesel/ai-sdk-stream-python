"""Preliminary tool outputs (AI SDK ``tool-output-available`` + ``preliminary``)."""

from __future__ import annotations

import json

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.events import ToolOutputAvailableEvent


async def collect(ctx: StreamContext) -> list[dict]:
    events = []
    async for raw in ctx.stream():
        body = raw.strip().removeprefix("data: ")
        if body == "[DONE]":
            break
        events.append(json.loads(body))
    return events


def outputs(events: list[dict]) -> list[tuple[object, bool | None]]:
    return [
        (e["output"], e.get("preliminary"))
        for e in events
        if e["type"] == "tool-output-available"
    ]


def test_event_preliminary_field_is_optional_on_the_wire():
    final = ToolOutputAvailableEvent(toolCallId="t", output=1)
    assert "preliminary" not in final.encode()
    prelim = ToolOutputAvailableEvent(toolCallId="t", output=1, preliminary=True)
    assert '"preliminary":true' in prelim.encode()


async def test_preliminary_then_final_snapshots():
    ctx = StreamContext(collect=True)
    h = await ctx.begin_tool_call("count", {})
    await ctx.write_preliminary_tool_output(h.toolCallId, {"n": 1})
    await ctx.write_preliminary_tool_output(h.toolCallId, {"n": 2})
    await ctx.complete_tool_call(h.toolCallId, {"n": 3})
    await ctx.finish()
    events = await collect(ctx)
    assert outputs(events) == [({"n": 1}, True), ({"n": 2}, True), ({"n": 3}, None)]

    assert ctx.record is not None
    tc = ctx.record.tool_calls[0]
    assert tc.state == "output-available"
    assert tc.output == {"n": 3}
    assert tc.preliminary_output == {"n": 2}


async def test_partial_output_is_not_recorded_as_success():
    ctx = StreamContext(collect=True)
    h = await ctx.begin_tool_call("count", {})
    await ctx.write_preliminary_tool_output(h.toolCallId, {"n": 1})
    assert ctx.record is not None
    tc = ctx.record.tool_calls[0]
    assert tc.state == "input-available"
    assert tc.output is None
    assert not tc.is_complete
    assert tc.preliminary_output == {"n": 1}


async def test_failure_keeps_partial_output_and_emits_error():
    ctx = StreamContext(collect=True)
    h = await ctx.begin_tool_call("count", {})
    await ctx.write_preliminary_tool_output(h.toolCallId, {"n": 1})
    await ctx.fail_tool_call(h.toolCallId, "boom")
    await ctx.finish()
    events = await collect(ctx)
    assert [e["errorText"] for e in events if e["type"] == "tool-output-error"] == [
        "boom"
    ]
    assert ctx.record is not None
    tc = ctx.record.tool_calls[0]
    assert tc.state == "output-error"
    assert tc.is_complete
    assert tc.output is None
    assert tc.preliminary_output == {"n": 1}
    d = ctx.record.to_dict()["tool_calls"][0]
    assert d["state"] == "output-error"
    assert d["preliminary_output"] == {"n": 1}


async def test_recorded_preliminary_output_is_not_aliased():
    ctx = StreamContext(collect=True)
    h = await ctx.begin_tool_call("t", {})
    snapshot = {"parts": ["a"]}
    await ctx.write_preliminary_tool_output(h.toolCallId, snapshot)
    snapshot["parts"].append("b")
    assert ctx.record is not None
    assert ctx.record.tool_calls[0].preliminary_output == {"parts": ["a"]}


async def test_streaming_input_states():
    ctx = StreamContext(collect=True)
    h = await ctx.start_tool_input("t")
    assert ctx.record is not None
    assert ctx.record.tool_calls[0].state == "input-streaming"
    await ctx.finish_tool_input(h.toolCallId, "t", {"a": 1})
    assert ctx.record.tool_calls[0].state == "input-available"
