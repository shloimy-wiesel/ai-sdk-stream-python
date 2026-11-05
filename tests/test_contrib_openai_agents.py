"""
Tests for contrib.openai_agents.

Scripted event sequences built from real ``openai-agents`` types — no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("agents")

from agents import Agent, MaxTurnsExceeded
from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)
from agents.usage import Usage
from openai.types.responses import (
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
