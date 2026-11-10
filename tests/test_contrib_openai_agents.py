"""
Tests for contrib.openai_agents.

Scripted event sequences built from real ``openai-agents`` types — no network.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("agents")

from agents import Agent, MaxTurnsExceeded, RunConfig, Runner
from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.models.interface import Model
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import OutputTokensDetails

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.contrib.openai_agents import (
    SDK_DEFAULT_TOOL_ERROR,
    NestedAgentTranscripts,
    agent_tool_stream_handler,
    consume_agent_stream,
)

MAIN = Agent(name="Main")
HELPER = Agent(name="Helper")
RESEARCHER = Agent(name="Researcher")

# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def text(delta: str, item_id: str = "msg_1") -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseTextDeltaEvent(
            content_index=0,
            delta=delta,
            item_id=item_id,
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )
    )


def reasoning(delta: str, item_id: str = "rs_1") -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseReasoningSummaryTextDeltaEvent(
            delta=delta,
            item_id=item_id,
            output_index=0,
            sequence_number=0,
            summary_index=0,
            type="response.reasoning_summary_text.delta",
        )
    )


def reasoning_done(item_id: str, summary: str) -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseOutputItemDoneEvent(
            item=ResponseReasoningItem(
                id=item_id,
                summary=[Summary(text=summary, type="summary_text")],
                type="reasoning",
            ),
            output_index=0,
            sequence_number=0,
            type="response.output_item.done",
        )
    )


def reasoning_item(item_id: str, summary: str, agent: Agent = MAIN):
    return RunItemStreamEvent(
        name="reasoning_item_created",
        item=ReasoningItem(
            agent=agent,
            raw_item=ResponseReasoningItem(
                id=item_id,
                summary=[Summary(text=summary, type="summary_text")],
                type="reasoning",
            ),
        ),
    )


def fn_call(
    call_id: str, name: str, args: dict[str, Any] | None = None
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(args or {}),
        call_id=call_id,
        name=name,
        type="function_call",
        id=f"fc_{call_id}",
    )


def tool_called(
    call_id: str, name: str, args: dict[str, Any], agent: Agent = MAIN
) -> RunItemStreamEvent:
    return RunItemStreamEvent(
        name="tool_called",
        item=ToolCallItem(agent=agent, raw_item=fn_call(call_id, name, args)),
    )


def tool_output(call_id: str, output: Any, agent: Agent = MAIN) -> RunItemStreamEvent:
    return RunItemStreamEvent(
        name="tool_output",
        item=ToolCallOutputItem(
            agent=agent,
            raw_item={
                "call_id": call_id,
                "output": str(output),
                "type": "function_call_output",
            },
            output=output,
        ),
    )


def message(text_: str, agent: Agent) -> RunItemStreamEvent:
    return RunItemStreamEvent(
        name="message_output_created",
        item=MessageOutputItem(
            agent=agent,
            raw_item=ResponseOutputMessage(
                id="msg_x",
                content=[
                    ResponseOutputText(annotations=[], text=text_, type="output_text")
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ),
    )


def handoff(source: Agent, target: Agent) -> list[RunItemStreamEvent]:
    call = ResponseFunctionToolCall(
        arguments="{}",
        call_id="call_h",
        name=f"transfer_to_{target.name.lower()}",
        type="function_call",
    )
    return [
        RunItemStreamEvent(
            name="handoff_requested", item=HandoffCallItem(agent=source, raw_item=call)
        ),
        RunItemStreamEvent(
            name="handoff_occured",
            item=HandoffOutputItem(
                agent=source,
                raw_item={
                    "call_id": "call_h",
                    "output": "{}",
                    "type": "function_call_output",
                },
                source_agent=source,
                target_agent=target,
            ),
        ),
    ]


def agent_updated(agent: Agent) -> AgentUpdatedStreamEvent:
    return AgentUpdatedStreamEvent(new_agent=agent)


# ---------------------------------------------------------------------------
# Fake RunResultStreaming
# ---------------------------------------------------------------------------


class FakeRun:
    """Stands in for ``RunResultStreaming``.

    *script* items are stream events, or async callables that run in-between
    (simulating the SDK invoking ``on_stream`` while a tool executes).
    """

    def __init__(
        self,
        script: list[Any],
        *,
        usage: Usage | None = None,
        raise_exc: BaseException | None = None,
        hang: bool = False,
    ) -> None:
        self.script = script
        self.raise_exc = raise_exc
        self.hang = hang
        self.is_complete = False
        self.cancelled = False
        self.context_wrapper = type("CW", (), {"usage": usage or Usage()})()
        self.yielded = asyncio.Event()

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True
        self.is_complete = True

    async def stream_events(self) -> AsyncIterator[Any]:
        for step in self.script:
            if callable(step):
                await step()  # type: ignore[misc]
                continue
            yield step
            self.yielded.set()
        if self.raise_exc is not None:
            self.is_complete = True
            raise self.raise_exc
        if self.hang:
            await asyncio.Event().wait()
        self.is_complete = True


async def collect(ctx: StreamContext) -> list[dict]:
    events = []
    async for raw in ctx.stream():
        body = raw.strip().removeprefix("data: ")
        if body == "[DONE]":
            break
        events.append(json.loads(body))
    return events


async def run_agent(run: FakeRun, **kwargs: Any) -> tuple[list[dict], StreamContext]:
    ctx = StreamContext(collect=True)

    async def work(ctx: StreamContext) -> None:
        await consume_agent_stream(run, ctx, **kwargs)  # type: ignore[arg-type]

    await ctx.run(work)
    return await collect(ctx), ctx


def types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_text_only_run():
    events, ctx = await run_agent(
        FakeRun([agent_updated(MAIN), text("Hel"), text("lo")])
    )
    assert types(events) == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    assert ctx.record is not None
    assert ctx.record.text == "Hello"


async def test_reasoning_then_text():
    events, ctx = await run_agent(
        FakeRun(
            [
                agent_updated(MAIN),
                reasoning("think"),
                reasoning("ing"),
                reasoning_item("rs_1", "thinking"),  # already streamed → skipped
                text("answer"),
            ]
        )
    )
    assert types(events)[2:9] == [
        "reasoning-start",
        "reasoning-delta",
        "reasoning-delta",
        "reasoning-end",
        "text-start",
        "text-delta",
        "text-end",
    ]
    assert ctx.record is not None
    assert ctx.record.reasoning == "thinking"


async def test_reasoning_item_written_before_text_when_run_item_arrives_late():
    """The SDK emits reasoning_item_created after the item's text streamed."""
    events, ctx = await run_agent(
        FakeRun(
            [
                agent_updated(MAIN),
                reasoning_done("rs_1", "summary only"),
                text("answer"),
                reasoning_item("rs_1", "summary only"),  # late; must not duplicate
            ]
        )
    )
    t = types(events)
    assert t.index("reasoning-delta") < t.index("text-delta")
    assert t.count("reasoning-delta") == 1
    assert ctx.record is not None
    assert ctx.record.reasoning == "summary only"


async def test_unstreamed_reasoning_item_written_from_summary():
    events, _ = await run_agent(FakeRun([reasoning_item("rs_9", "summary only")]))
    deltas = [e["delta"] for e in events if e["type"] == "reasoning-delta"]
    assert deltas == ["summary only"]


async def test_parallel_tool_calls_interleaved_outputs():
    events, ctx = await run_agent(
        FakeRun(
            [
                agent_updated(MAIN),
                tool_called("call_a", "weather", {"city": "Paris"}),
                tool_called("call_b", "weather", {"city": "Rome"}),
                tool_output("call_b", "sunny"),
                tool_output("call_a", "rainy"),
                text("done"),
            ]
        )
    )
    inputs = {
        e["toolCallId"]: e["input"]
        for e in events
        if e["type"] == "tool-input-available"
    }
    outputs = [
        (e["toolCallId"], e["output"])
        for e in events
        if e["type"] == "tool-output-available"
    ]
    assert inputs == {"call_a": {"city": "Paris"}, "call_b": {"city": "Rome"}}
    assert outputs == [("call_b", "sunny"), ("call_a", "rainy")]
    assert ctx.record is not None
    assert {tc.tool_call_id: tc.output for tc in ctx.record.tool_calls} == {
        "call_a": "rainy",
        "call_b": "sunny",
    }


async def test_tool_error_default_message_and_exception():
    events, _ = await run_agent(
        FakeRun(
            [
                tool_called("c1", "flaky", {}),
                tool_called("c2", "boom", {}),
                tool_output("c1", SDK_DEFAULT_TOOL_ERROR),
                tool_output("c2", ValueError("bad input")),
            ]
        )
    )
    errors = {
        e["toolCallId"]: e["errorText"]
        for e in events
        if e["type"] == "tool-output-error"
    }
    assert errors == {"c1": SDK_DEFAULT_TOOL_ERROR, "c2": "bad input"}
    assert "tool-output-available" not in types(events)


async def test_tool_error_custom_detector():
    def detect(item: Any) -> str | None:
        out = item.output
        return (
            out.removeprefix("ERR:")
            if isinstance(out, str) and out.startswith("ERR:")
            else None
        )

    events, _ = await run_agent(
        FakeRun([tool_called("c1", "t", {}), tool_output("c1", "ERR:nope")]),
        detect_tool_error=detect,
    )
    assert [e["errorText"] for e in events if e["type"] == "tool-output-error"] == [
        "nope"
    ]


async def test_hosted_tool_completed_immediately():
    web = RunItemStreamEvent(
        name="tool_called",
        item=ToolCallItem(
            agent=MAIN,
            raw_item=ResponseFunctionWebSearch(
                id="ws_1",
                action=ActionSearch(type="search", query="news"),
                status="completed",
                type="web_search_call",
            ),
        ),
    )
    events, _ = await run_agent(FakeRun([web]))
    start = next(e for e in events if e["type"] == "tool-input-start")
    out = next(e for e in events if e["type"] == "tool-output-available")
    assert start["toolCallId"] == out["toolCallId"] == "ws_1"
    assert out["output"]["status"] == "completed"


async def test_handoff_note_and_new_step():
    events, _ = await run_agent(
        FakeRun(
            [
                agent_updated(MAIN),
                text("let me transfer you"),
                *handoff(MAIN, HELPER),
                agent_updated(HELPER),
                text("hi, helper here"),
            ]
        )
    )
    notes = [e["delta"] for e in events if e["type"] == "reasoning-delta"]
    assert notes == ["Handing off to Helper"]
    assert types(events).count("start-step") == 2
    assert types(events).count("finish-step") == 2
    # handoff_requested does not create a tool part
    assert "tool-input-start" not in types(events)


async def test_handoff_notes_opt_out():
    events, _ = await run_agent(
        FakeRun([*handoff(MAIN, HELPER), text("x")]), handoff_notes=False
    )
    assert "reasoning-delta" not in types(events)


async def test_mcp_events_ignored_by_default_and_hook_opt_in():
    from agents.items import MCPListToolsItem
    from openai.types.responses.response_output_item import McpListTools

    mcp = RunItemStreamEvent(
        name="mcp_list_tools",
        item=MCPListToolsItem(
            agent=MAIN,
            raw_item=McpListTools(
                id="m1", server_label="srv", tools=[], type="mcp_list_tools"
            ),
        ),
    )
    events, _ = await run_agent(FakeRun([mcp, text("x")]))
    assert types(events) == [
        "start",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]

    seen: list[str] = []

    async def hook(event: Any, ctx: StreamContext) -> None:
        seen.append(event.name)
        await ctx.write_data("mcp", {"server": event.item.raw_item.server_label})

    events, _ = await run_agent(FakeRun([mcp]), on_mcp_event=hook)
    assert seen == ["mcp_list_tools"]
    assert any(e["type"] == "data-mcp" for e in events)


async def test_usage_copied_to_record():
    usage = Usage(
        requests=1,
        input_tokens=100,
        output_tokens=50,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=20),
        total_tokens=150,
    )
    _, ctx = await run_agent(FakeRun([text("hi")], usage=usage))
    assert ctx.record is not None
    assert ctx.record.prompt_tokens == 100
    assert ctx.record.reasoning_tokens == 20
    assert ctx.record.answer_tokens == 30


async def test_stream_exception_goes_through_ctx_run_error_path():
    run = FakeRun(
        [text("partial")], raise_exc=MaxTurnsExceeded("Max turns (3) exceeded")
    )
    events, _ = await run_agent(run)
    assert events[-1] == {"type": "error", "errorText": "Max turns (3) exceeded"}
    assert not run.cancelled  # the run had already ended on its own


async def test_cancellation_mid_run_cancels_agent_run():
    run = FakeRun([text("partial")], hang=True)
    ctx = StreamContext()
    task = asyncio.create_task(consume_agent_stream(run, ctx))  # type: ignore[arg-type]
    await run.yielded.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.cancelled


async def test_ctx_aborted_mid_run_cancels_agent_run():
    ctx = StreamContext()

    async def abort() -> None:
        await ctx.abort("client disconnected")

    run = FakeRun([text("partial"), abort, text("never")])
    await consume_agent_stream(run, ctx)  # type: ignore[arg-type]
    assert run.cancelled
    events = await collect(ctx)
    assert [e["delta"] for e in events if e["type"] == "text-delta"] == ["partial"]
    assert events[-1]["type"] == "abort"


async def test_nested_as_tool_run_with_on_stream():
    """Two sub-agents run in parallel as tools; their activity becomes
    reasoning notes and never leaks into the top-level text part."""
    ctx = StreamContext(collect=True)
    on_stream = agent_tool_stream_handler(ctx)
    parent_a = fn_call("call_a", "research", {"q": "a"})
    parent_b = fn_call("call_b", "research", {"q": "b"})

    def nested(event: Any, agent: Agent, parent: ResponseFunctionToolCall):
        async def emit() -> None:
            await on_stream({"event": event, "agent": agent, "tool_call": parent})

        return emit

    run = FakeRun(
        [
            agent_updated(MAIN),
            tool_called("call_a", "research", {"q": "a"}),
            tool_called("call_b", "research", {"q": "b"}),
            nested(agent_updated(RESEARCHER), RESEARCHER, parent_a),
            nested(text("inner text A"), RESEARCHER, parent_a),
            nested(tool_called("in_1", "search", {}, RESEARCHER), RESEARCHER, parent_a),
            nested(text("inner text B"), HELPER, parent_b),
            nested(tool_output("in_1", "hits", RESEARCHER), RESEARCHER, parent_a),
            nested(message("found A", RESEARCHER), RESEARCHER, parent_a),
            nested(message("found B", HELPER), HELPER, parent_b),
            tool_output("call_b", "B"),
            tool_output("call_a", "A"),
            text("final answer"),
        ]
    )

    async def work(ctx: StreamContext) -> None:
        await consume_agent_stream(run, ctx)  # type: ignore[arg-type]

    await ctx.run(work)
    events = await collect(ctx)

    text_deltas = [e["delta"] for e in events if e["type"] == "text-delta"]
    assert text_deltas == ["final answer"]
    reasoning_text = "".join(
        e["delta"] for e in events if e["type"] == "reasoning-delta"
    )
    assert reasoning_text == (
        "[Researcher] Calling search\n\n[Researcher] found A\n\n[Helper] found B"
    )
    # Only the two parent tool parts exist, resolved by call_id.
    assert [e["toolCallId"] for e in events if e["type"] == "tool-input-start"] == [
        "call_a",
        "call_b",
    ]
    assert [
        (e["toolCallId"], e["output"])
        for e in events
        if e["type"] == "tool-output-available"
    ] == [("call_b", "B"), ("call_a", "A")]
    # Nested agent_updated must not open a new top-level step.
    assert types(events).count("start-step") == 1


async def test_nested_tool_calls_as_tool_parts():
    ctx = StreamContext()
    on_stream = agent_tool_stream_handler(ctx, nested_tool_calls=True)
    parent = fn_call("call_a", "research")

    async def work(ctx: StreamContext) -> None:
        await ctx.begin_tool_call("research", {}, tool_call_id="call_a")
        for ev in [
            tool_called("in_1", "search", {"q": "x"}, RESEARCHER),
            tool_output("in_1", "hits", RESEARCHER),
            tool_called("in_2", "fetch", {}, RESEARCHER),
            tool_output("in_2", SDK_DEFAULT_TOOL_ERROR, RESEARCHER),
        ]:
            await on_stream({"event": ev, "agent": RESEARCHER, "tool_call": parent})
        await ctx.complete_tool_call("call_a", "done")

    await ctx.run(work)
    events = await collect(ctx)
    assert [e["toolCallId"] for e in events if e["type"] == "tool-input-start"] == [
        "call_a",
        "in_1",
        "in_2",
    ]
    assert ("in_1", "hits") in [
        (e["toolCallId"], e["output"])
        for e in events
        if e["type"] == "tool-output-available"
    ]
    assert [e["toolCallId"] for e in events if e["type"] == "tool-output-error"] == [
        "in_2"
    ]
    assert "reasoning-delta" not in types(events)


async def test_nested_handler_noop_after_finish():
    ctx = StreamContext()
    on_stream = agent_tool_stream_handler(ctx)
    await ctx.finish()
    await on_stream(
        {"event": message("late", RESEARCHER), "agent": RESEARCHER, "tool_call": None}
    )
    assert types(await collect(ctx)) == ["start", "finish"]


# ---------------------------------------------------------------------------
# NestedAgentTranscripts — sub-agent transcripts as preliminary tool results
# ---------------------------------------------------------------------------


def nested_via(
    transcripts: NestedAgentTranscripts,
    agent: Agent,
    parent: ResponseFunctionToolCall | None,
):
    """Return ``ev -> async callable`` that feeds *ev* to the handler."""

    def wrap(event: Any):
        async def emit() -> None:
            await transcripts.on_stream(
                {"event": event, "agent": agent, "tool_call": parent}  # type: ignore[typeddict-item]
            )

        return emit

    return wrap


def tool_outputs(events: list[dict], call_id: str) -> list[dict]:
    return [
        e
        for e in events
        if e["type"] in ("tool-output-available", "tool-output-error")
        and e["toolCallId"] == call_id
    ]


def child_texts(output: dict) -> list[str]:
    return [p["text"] for p in output["message"]["parts"] if p["type"] == "text"]


async def run_nested(
    script_fn: Any, **kwargs: Any
) -> tuple[list[dict], StreamContext, FakeRun]:
    ctx = StreamContext(collect=True)
    transcripts = NestedAgentTranscripts(ctx, throttle=0, **kwargs)
    run = FakeRun(script_fn(transcripts))

    async def work(ctx: StreamContext) -> None:
        await consume_agent_stream(run, ctx, nested=transcripts)  # type: ignore[arg-type]

    await ctx.run(work)
    return await collect(ctx), ctx, run


async def test_child_text_streams_into_parent_card_before_return():
    parent = fn_call("call_a", "research", {"q": "x"})

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {"q": "x"}),
            child(text("child ", item_id="msg_x")),
            child(text("answer", item_id="msg_x")),
            child(message("child answer", RESEARCHER)),
            tool_output("call_a", "concise result"),
            text("final"),
        ]

    events, ctx, _ = await run_nested(script)
    outs = tool_outputs(events, "call_a")
    assert [o.get("preliminary") for o in outs] == [True, True, True, None]
    # Snapshots accumulate (not aliased to later mutable state).
    assert [child_texts(o["output"]) for o in outs] == [
        ["child "],
        ["child answer"],
        ["child answer"],
        ["child answer"],
    ]
    first = outs[0]["output"]
    assert first["kind"] == "subagent"
    assert first["version"] == 1
    assert first["agentName"] == "Researcher"
    assert first["message"]["id"] == "child_call_a"
    assert first["message"]["parts"][0] == {"type": "step-start"}
    assert first["message"]["parts"][1]["state"] == "streaming"
    final = outs[-1]["output"]
    assert final["result"] == "concise result"
    assert final["message"]["parts"][1]["state"] == "done"
    # Child text never reaches the parent text part.
    assert [e["delta"] for e in events if e["type"] == "text-delta"] == ["final"]
    assert "reasoning-delta" not in types(events)
    # Parent input comes before any output; exactly one finish.
    idx = types(events)
    assert idx.index("tool-input-available") < idx.index("tool-output-available")
    assert idx.count("finish") == 1
    assert ctx.record is not None
    tc = ctx.record.tool_calls[0]
    assert tc.state == "output-available"
    assert tc.output["result"] == "concise result"


async def test_parallel_same_agent_with_colliding_child_ids_stay_separate():
    pa = fn_call("call_a", "research", {"q": "a"})
    pb = fn_call("call_b", "research", {"q": "b"})

    def script(t: NestedAgentTranscripts) -> list[Any]:
        a = nested_via(t, RESEARCHER, pa)
        b = nested_via(t, RESEARCHER, pb)
        return [
            tool_called("call_a", "research", {"q": "a"}),
            tool_called("call_b", "research", {"q": "b"}),
            a(text("A1", item_id="msg_1")),
            b(text("B1", item_id="msg_1")),
            a(text("A2", item_id="msg_1")),
            b(text("B2", item_id="msg_1")),
            tool_output("call_b", "B"),
            tool_output("call_a", "A"),
        ]

    events, _, _ = await run_nested(script)
    final_a = tool_outputs(events, "call_a")[-1]["output"]
    final_b = tool_outputs(events, "call_b")[-1]["output"]
    assert child_texts(final_a) == ["A1A2"]
    assert child_texts(final_b) == ["B1B2"]
    assert final_a["result"] == "A"
    assert final_b["result"] == "B"


async def test_nested_tools_render_inside_child_transcript():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(tool_called("in_1", "search", {"q": "x"}, RESEARCHER)),
            child(tool_output("in_1", "hits", RESEARCHER)),
            child(tool_called("in_2", "fetch", {}, RESEARCHER)),
            child(tool_output("in_2", SDK_DEFAULT_TOOL_ERROR, RESEARCHER)),
            tool_output("call_a", "done"),
        ]

    events, _, _ = await run_nested(script)
    # Nested tools are not top-level tool parts.
    assert [e["toolCallId"] for e in events if e["type"] == "tool-input-start"] == [
        "call_a"
    ]
    parts = tool_outputs(events, "call_a")[-1]["output"]["message"]["parts"]
    tools = [p for p in parts if p["type"] == "dynamic-tool"]
    assert tools == [
        {
            "type": "dynamic-tool",
            "toolName": "search",
            "toolCallId": "in_1",
            "state": "output-available",
            "input": {"q": "x"},
            "output": "hits",
        },
        {
            "type": "dynamic-tool",
            "toolName": "fetch",
            "toolCallId": "in_2",
            "state": "output-error",
            "input": {},
            "errorText": SDK_DEFAULT_TOOL_ERROR,
        },
    ]


async def test_completed_message_does_not_duplicate_streamed_text():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(text("hello", item_id="msg_x")),
            child(message("hello", RESEARCHER)),  # id msg_x
            child(reasoning("thinking", item_id="rs_9")),
            child(reasoning_item("rs_9", "thinking", RESEARCHER)),
            child(message("hello", RESEARCHER)),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script)
    parts = tool_outputs(events, "call_a")[-1]["output"]["message"]["parts"]
    assert [(p["type"], p.get("text")) for p in parts] == [
        ("step-start", None),
        ("text", "hello"),
        ("reasoning", "thinking"),
    ]


def response_created() -> Any:
    return SimpleNamespace(
        type="raw_response_event", data=SimpleNamespace(type="response.created")
    )


async def test_fake_item_ids_are_scoped_to_one_model_response():
    """Chat Completions models reuse ``__fake_id__`` for every item."""
    parent = fn_call("call_a", "research")
    fake = "__fake_id__"

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(response_created()),
            child(text("first", item_id=fake)),
            child(tool_called("in_1", "search", {}, RESEARCHER)),
            child(tool_output("in_1", "hits", RESEARCHER)),
            child(response_created()),
            child(text("second", item_id=fake)),
            child(message("second", RESEARCHER)),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script)
    parts = tool_outputs(events, "call_a")[-1]["output"]["message"]["parts"]
    assert [(p["type"], p.get("text")) for p in parts] == [
        ("step-start", None),
        ("text", "first"),
        ("dynamic-tool", None),
        ("step-start", None),
        ("text", "second"),
    ]


async def test_non_streamed_message_is_added_once():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(reasoning_item("rs_1", "plan", RESEARCHER)),
            child(message("whole", RESEARCHER)),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script, include_reasoning=False)
    parts = tool_outputs(events, "call_a")[-1]["output"]["message"]["parts"]
    assert [(p["type"], p.get("text")) for p in parts] == [
        ("step-start", None),
        ("text", "whole"),
    ]


async def test_callback_before_parent_tool_called_is_buffered():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            child(text("early")),
            tool_called("call_a", "research", {}),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script)
    ts = types(events)
    assert ts.index("tool-input-available") < ts.index("tool-output-available")
    outs = tool_outputs(events, "call_a")
    assert [o.get("preliminary") for o in outs] == [True, None]
    assert child_texts(outs[0]["output"]) == ["early"]


async def test_failed_subagent_keeps_partial_transcript():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(text("partial")),
            tool_output("call_a", SDK_DEFAULT_TOOL_ERROR),
        ]

    events, ctx, _ = await run_nested(script)
    outs = tool_outputs(events, "call_a")
    assert outs[-1]["type"] == "tool-output-error"
    assert outs[-1]["errorText"] == SDK_DEFAULT_TOOL_ERROR
    assert ctx.record is not None
    tc = ctx.record.tool_calls[0]
    assert tc.state == "output-error"
    assert tc.output is None
    assert child_texts(tc.preliminary_output) == ["partial"]


async def test_cancellation_stops_late_writes_and_timers():
    ctx = StreamContext()
    transcripts = NestedAgentTranscripts(ctx, throttle=10)
    parent = fn_call("call_a", "research")
    child = nested_via(transcripts, RESEARCHER, parent)
    run = FakeRun(
        [tool_called("call_a", "research", {}), child(text("a")), child(text("b"))],
        hang=True,
    )
    task = asyncio.create_task(consume_agent_stream(run, ctx, nested=transcripts))  # type: ignore[arg-type]
    await run.yielded.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert run.cancelled
    # The throttled trailing snapshot was cancelled; late callbacks are ignored.
    assert all(r.timer is None for r in transcripts._runs.values())
    await child(text("late"))()
    await ctx.finish()
    events = await collect(ctx)
    outs = tool_outputs(events, "call_a")
    assert [child_texts(o["output"]) for o in outs] == [["a"]]


async def test_throttle_coalesces_deltas_and_flushes_trailing_edge():
    ctx = StreamContext()
    transcripts = NestedAgentTranscripts(ctx, throttle=0.05)
    parent = fn_call("call_a", "research")
    child = nested_via(transcripts, RESEARCHER, parent)

    async def pause() -> None:
        await asyncio.sleep(0.1)

    run = FakeRun(
        [
            tool_called("call_a", "research", {}),
            child(text("a")),
            child(text("b")),
            child(text("c")),
            pause,  # trailing-edge timer fires here
            tool_output("call_a", "ok"),
        ]
    )

    async def work(ctx: StreamContext) -> None:
        await consume_agent_stream(run, ctx, nested=transcripts)  # type: ignore[arg-type]

    await ctx.run(work)
    events = await collect(ctx)
    outs = tool_outputs(events, "call_a")
    assert [(child_texts(o["output"]), o.get("preliminary")) for o in outs] == [
        (["a"], True),
        (["abc"], True),
        (["abc"], None),
    ]


async def test_missing_originating_tool_call_is_dropped():
    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, None)
        return [child(text("orphan")), text("final")]

    events, _, _ = await run_nested(script)
    assert "tool-output-available" not in types(events)
    assert [e["delta"] for e in events if e["type"] == "text-delta"] == ["final"]


async def test_adapter_failure_surfaces_as_request_error():
    def boom(item: Any) -> str | None:
        raise ValueError("detector exploded")

    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            child(tool_called("in_1", "search", {}, RESEARCHER)),
            child(tool_output("in_1", "hits", RESEARCHER)),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script, detect_tool_error=boom)
    assert events[-1] == {"type": "error", "errorText": "detector exploded"}
    assert not [
        e for e in tool_outputs(events, "call_a") if e.get("preliminary") is None
    ]


async def test_handoff_inside_child_is_recorded_as_data_part():
    parent = fn_call("call_a", "research")

    def script(t: NestedAgentTranscripts) -> list[Any]:
        child = nested_via(t, RESEARCHER, parent)
        return [
            tool_called("call_a", "research", {}),
            *[child(e) for e in handoff(RESEARCHER, HELPER)],
            child(agent_updated(HELPER)),
            child(text("helper says")),
            tool_output("call_a", "ok"),
        ]

    events, _, _ = await run_nested(script)
    parts = tool_outputs(events, "call_a")[-1]["output"]["message"]["parts"]
    assert parts == [
        {"type": "step-start"},
        {"type": "data-handoff", "data": {"from": "Researcher", "to": "Helper"}},
        {"type": "step-start"},
        {"type": "text", "text": "helper says", "state": "done"},
    ]
    assert types(events).count("start-step") == 1


async def test_tools_without_transcript_are_unchanged():
    def script(t: NestedAgentTranscripts) -> list[Any]:
        return [tool_called("call_w", "weather", {}), tool_output("call_w", "sunny")]

    events, _, _ = await run_nested(script)
    assert [
        (e["output"], e.get("preliminary")) for e in tool_outputs(events, "call_w")
    ] == [("sunny", None)]


# ---------------------------------------------------------------------------
# End-to-end with the real SDK Runner and a scripted (network-free) model
# ---------------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "nested_agent_transcript.sse"


def _response(output: list[Any]) -> Response:
    return Response(
        id="resp",
        created_at=0,
        model="scripted",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _out_message(text_: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_1",
        role="assistant",
        status="completed",
        type="message",
        content=[ResponseOutputText(annotations=[], text=text_, type="output_text")],
    )


class ScriptedModel(Model):
    """Streams ``turn(input) -> (text deltas, output items)``; item ids repeat."""

    def __init__(self, turn: Any) -> None:
        self.turn = turn
        self.inputs: list[Any] = []

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def stream_response(  # type: ignore[override]
        self, system_instructions: Any, input: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.inputs.append(input)
        deltas, items = self.turn(input)
        yield ResponseCreatedEvent(
            response=_response([]), sequence_number=0, type="response.created"
        )
        for delta in deltas:
            yield ResponseTextDeltaEvent(
                content_index=0,
                delta=delta,
                item_id="msg_1",
                logprobs=[],
                output_index=0,
                sequence_number=0,
                type="response.output_text.delta",
            )
        yield ResponseCompletedEvent(
            response=_response(items), sequence_number=0, type="response.completed"
        )


def _child_turn(input: Any) -> tuple[list[str], list[Any]]:
    tag = "A" if "alpha" in str(input) else "B"
    return [f"{tag}-one ", f"{tag}-two"], [_out_message(f"{tag}-one {tag}-two")]


def _parent_turn(input: Any) -> tuple[list[str], list[Any]]:
    if any(
        isinstance(i, dict) and i.get("type") == "function_call_output" for i in input
    ):
        return ["done"], [_out_message("done")]
    return [], [
        fn_call("call_a", "research", {"input": "alpha"}),
        fn_call("call_b", "research", {"input": "beta"}),
    ]


def _normalize(lines: list[str]) -> list[str]:
    """Replace random text-part ids so the stream is reproducible."""
    ids: dict[str, str] = {}
    out = []
    for line in lines:
        body = line.strip().removeprefix("data: ")
        if body != "[DONE]":
            chunk = json.loads(body)
            if chunk["type"].startswith("text-"):
                chunk["id"] = ids.setdefault(chunk["id"], f"text-{len(ids)}")
            body = json.dumps(chunk, separators=(",", ":"))
        out.append(f"data: {body}")
    return out


async def test_real_runner_parallel_subagents_end_to_end():
    from test_wire_schema import validate_chunk

    ctx = StreamContext(message_id="msg_fixture", collect=True)
    transcripts = NestedAgentTranscripts(ctx, throttle=0)
    child = Agent(name="Researcher", model=ScriptedModel(_child_turn))
    parent_model = ScriptedModel(_parent_turn)
    parent = Agent(
        name="Main",
        model=parent_model,
        tools=[
            child.as_tool(
                tool_name="research",
                tool_description="Research a topic",
                on_stream=transcripts.on_stream,
            )
        ],
    )

    async def work(ctx: StreamContext) -> None:
        result = Runner.run_streamed(
            parent, "go", run_config=RunConfig(tracing_disabled=True)
        )
        await consume_agent_stream(result, ctx, nested=transcripts)

    await ctx.run(work)
    lines = [line async for line in ctx.stream()]
    events = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    for event in events:
        validate_chunk(event)

    for call_id, tag in (("call_a", "A"), ("call_b", "B")):
        outs = tool_outputs(events, call_id)
        assert [o.get("preliminary") for o in outs][-1] is None
        assert all(o.get("preliminary") for o in outs[:-1])
        assert len(outs) > 1  # visible before the child returned
        assert child_texts(outs[0]["output"]) == [f"{tag}-one "]
        assert child_texts(outs[-1]["output"]) == [f"{tag}-one {tag}-two"]
        assert outs[-1]["output"]["result"] == f"{tag}-one {tag}-two"

    # The parent model got the concise SDK result, not the UI transcript.
    tool_results = {
        i["call_id"]: i["output"]
        for i in parent_model.inputs[-1]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    }
    assert tool_results == {"call_a": "A-one A-two", "call_b": "B-one B-two"}
    assert [e["delta"] for e in events if e["type"] == "text-delta"] == ["done"]
    assert types(events).count("finish") == 1
    assert lines[-1] == "data: [DONE]\n\n"

    # Protocol fixture for AI SDK v6 (replay: fixtures/verify_nested_fixture.mjs).
    normalized = "\n\n".join(_normalize(lines)) + "\n\n"
    if os.environ.get("UPDATE_FIXTURES"):
        FIXTURE.write_text(normalized)
    assert normalized == FIXTURE.read_text()
