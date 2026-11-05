"""
contrib.openai_agents — OpenAI Agents SDK (``openai-agents``) adapter.

Maps the high-level event stream of ``Runner.run_streamed()`` onto
``StreamContext`` writes so an Agents SDK backend can serve the Vercel AI SDK
v6 UIMessageStream without writing its own encoder.

Two entry points:

* ``consume_agent_stream(result, ctx)`` — iterates
  ``RunResultStreaming.stream_events()`` and writes text, reasoning, tool
  calls, handoffs and usage to *ctx*.
* ``agent_tool_stream_handler(ctx)`` — returns a callback for
  ``Agent.as_tool(on_stream=...)`` so events from nested agents-as-tools are
  rendered as reasoning notes (or their own tool parts) instead of leaking into
  the top-level answer text.

Install with the ``agents`` extra::

    pip install "ai-sdk-stream-python[agents]"

The ``openai-agents`` package is only needed for building the agents; this
module reads events by duck typing and never imports it at runtime.

Usage::

    from agents import Agent, Runner
    from ai_sdk_stream_python import StreamContext
    from ai_sdk_stream_python.contrib.openai_agents import (
        agent_tool_stream_handler,
        consume_agent_stream,
    )


    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        ctx = StreamContext()
        researcher_tool = researcher.as_tool(
            tool_name="research",
            tool_description="Research a topic",
            on_stream=agent_tool_stream_handler(ctx),
        )
        agent = Agent(name="Assistant", tools=[researcher_tool])

        async def work(ctx: StreamContext) -> None:
            result = Runner.run_streamed(agent, request.messages[-1].text)
            await consume_agent_stream(result, ctx)

        await ctx.run(work)
        return StreamingResponse(
            ctx.stream(),
            media_type="text/event-stream",
            headers=ctx.response_headers,
        )

``consume_agent_stream`` does **not** call ``ctx.finish()``; use it inside
``ctx.run(...)`` (which finishes, and turns exceptions into ``error`` events)
or finish the context yourself.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ..context import StreamContext

if TYPE_CHECKING:
    from agents import AgentToolStreamEvent, RunResultStreaming
    from agents.items import ToolCallOutputItem
    from agents.stream_events import RunItemStreamEvent

#: Hook for ``mcp_list_tools`` / ``mcp_approval_*`` run-item events.
#: Receives the ``RunItemStreamEvent`` and the context; may be sync or async.
McpEventHook = Callable[["RunItemStreamEvent", StreamContext[Any]], Any]

#: Returns the error text when a tool output represents a failure, else ``None``.
ToolErrorDetector = Callable[["ToolCallOutputItem"], "str | None"]

#: Text returned by the SDK's ``default_tool_error_function`` when a function
#: tool raises.  The SDK reports tool failures as ordinary string outputs.
SDK_DEFAULT_TOOL_ERROR = "An error occurred while running the tool. Please try again."

_MCP_EVENTS = frozenset(
    {"mcp_list_tools", "mcp_approval_requested", "mcp_approval_response"}
)
#: Hosted tools whose result is embedded in the call item itself — no
#: ``tool_output`` event follows, so the tool part is completed right away.
_HOSTED_TOOL_TYPES = frozenset(
    {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
    }
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read *name* from a pydantic model / dataclass or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _call_id(raw: Any) -> str | None:
    return _get(raw, "call_id") or _get(raw, "id")


def _tool_name(item: Any) -> str:
    name = getattr(item, "tool_name", None) or _get(item.raw_item, "name")
    return name or _get(item.raw_item, "type") or "tool"


def _tool_input(raw: Any) -> dict[str, Any]:
    """Parse a tool call's arguments into a dict (``{}`` when unavailable)."""
    args = _get(raw, "arguments")
    if args is None:
        args = _get(raw, "input")
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args:
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"arguments": args}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _dump(raw: Any) -> Any:
    dump = getattr(raw, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else raw


def _message_text(raw: Any) -> str:
    parts = _get(raw, "content") or []
    return "".join(
        _get(p, "text") or ""
        for p in parts
        if _get(p, "type") in ("output_text", "text", None)
    )


def _default_tool_error(item: ToolCallOutputItem) -> str | None:
    output = item.output
    if isinstance(output, BaseException):
        return str(output) or type(output).__name__
    if output == SDK_DEFAULT_TOOL_ERROR:
        return output
    return None


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


async def _write_note(ctx: StreamContext[Any], note: str) -> None:
    """Write a reasoning note, separated from preceding reasoning text."""
    sep = "\n\n" if ctx.current_reasoning_id is not None else ""
    await ctx.write_reasoning(f"{sep}{note}")


async def _begin_tool(ctx: StreamContext[Any], item: Any) -> str:
    """Emit a tool part for a ``tool_called`` item; return its ``toolCallId``."""
    raw = item.raw_item
    handle = await ctx.begin_tool_call(
        _tool_name(item), _tool_input(raw), tool_call_id=_call_id(raw)
    )
    if _get(raw, "type") in _HOSTED_TOOL_TYPES:
        error = _get(raw, "error")
        if error:
            await ctx.fail_tool_call(handle.toolCallId, str(error))
        else:
            output = _get(raw, "output")
            await ctx.complete_tool_call(
                handle.toolCallId, output if output is not None else _dump(raw)
            )
    return handle.toolCallId


async def _end_tool(
    ctx: StreamContext[Any], item: Any, detect_tool_error: ToolErrorDetector
) -> None:
    """Emit the output (or error) for a ``tool_output`` item."""
    call_id = _call_id(item.raw_item)
    if call_id is None:
        return
    error = detect_tool_error(item)
    if error is not None:
        await ctx.fail_tool_call(call_id, error)
    else:
        await ctx.complete_tool_call(call_id, item.output)


async def _write_usage(ctx: StreamContext[Any], result: Any) -> None:
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return
    output_tokens = getattr(usage, "output_tokens", None)
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    answer_tokens = output_tokens
    if output_tokens is not None and reasoning_tokens:
        answer_tokens = output_tokens - reasoning_tokens
    await ctx.set_usage(
        prompt_tokens=getattr(usage, "input_tokens", None),
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
    )


# ── Public API ─────────────────────────────────────────────────────────────────


async def consume_agent_stream(
    result: RunResultStreaming,
    ctx: StreamContext[Any],
    *,
    handoff_notes: bool = True,
    detect_tool_error: ToolErrorDetector | None = None,
    on_mcp_event: McpEventHook | None = None,
) -> None:
    """
    Consume ``RunResultStreaming.stream_events()`` and write it to *ctx*.

    Event mapping:

    ==========================================  ==================================
    Agents SDK event                            ``StreamContext`` write
    ==========================================  ==================================
    raw ``response.output_text.delta``          ``write_text``
    raw ``response.reasoning_text.delta`` /     ``write_reasoning``
    ``response.reasoning_summary_text.delta``
    ``reasoning_item_created`` (not streamed)   ``write_reasoning`` (summary text)
    ``tool_called``                             ``begin_tool_call`` (id = call_id)
    ``tool_output``                             ``complete_tool_call`` /
                                                ``fail_tool_call``
    ``handoff_occured``                         reasoning note "Handing off to X"
    ``agent_updated_stream_event``              ``new_step``
    ``mcp_list_tools`` / ``mcp_approval_*``     *on_mcp_event* hook (else ignored)
    end of stream                               ``set_usage`` from
                                                ``result.context_wrapper.usage``
    ==========================================  ==================================

    Tool calls are tracked by ``call_id``, so parallel calls whose outputs
    arrive interleaved are each resolved on the right tool part.

    Exceptions raised by ``stream_events()`` (``MaxTurnsExceeded``, guardrail
    tripwires, model errors) propagate unchanged, so ``ctx.run()`` emits them
    as an ``error`` event.  If this coroutine is cancelled, or *ctx* is
    finished by someone else (e.g. aborted after a client disconnect), the run
    is stopped with ``result.cancel()``.

    Does **not** call ``ctx.finish()``.

    Parameters
    ----------
    result:
        The object returned by ``Runner.run_streamed(...)``.
    ctx:
        The active ``StreamContext``.
    handoff_notes:
        Write a short reasoning note when a handoff occurs.  Default ``True``.
    detect_tool_error:
        ``(ToolCallOutputItem) -> str | None`` returning the error text when a
        tool output is a failure.  The SDK reports tool exceptions as ordinary
        outputs produced by the tool's ``failure_error_function``, so the
        default only recognises exception objects and the SDK's default
        failure message (:data:`SDK_DEFAULT_TOOL_ERROR`).  Supply your own
        when you use a custom ``failure_error_function``.
    on_mcp_event:
        Called with ``(event, ctx)`` for MCP list-tools / approval run items,
        which are ignored by default.  May be sync or async.
    """
    detect = detect_tool_error or _default_tool_error
    streamed_reasoning: set[str] = set()

    try:
        async for event in result.stream_events():
            if ctx.is_finished:
                result.cancel()
                break

            if event.type == "raw_response_event":
                data = event.data
                if data.type == "response.output_text.delta":
                    await ctx.write_text(data.delta)
                elif (
                    data.type == "response.reasoning_text.delta"
                    or data.type == "response.reasoning_summary_text.delta"
                ):
                    streamed_reasoning.add(data.item_id)
                    await ctx.write_reasoning(data.delta)

            elif event.type == "agent_updated_stream_event":
                await ctx.new_step()

            elif event.type == "run_item_stream_event":
                name, item = event.name, event.item
                if name == "tool_called":
                    await _begin_tool(ctx, item)
                elif name == "tool_output":
                    await _end_tool(ctx, item, detect)
                elif name == "reasoning_item_created":
                    raw = item.raw_item
                    if _get(raw, "id") not in streamed_reasoning:
                        summary = "\n\n".join(
                            _get(s, "text") or "" for s in _get(raw, "summary") or []
                        )
                        if summary:
                            await ctx.write_reasoning(summary)
                elif name == "handoff_occured":
                    if handoff_notes:
                        target = getattr(item, "target_agent", None)
                        target_name = getattr(target, "name", None) or "another agent"
                        await _write_note(ctx, f"Handing off to {target_name}")
                elif name in _MCP_EVENTS and on_mcp_event is not None:
                    await _maybe_await(on_mcp_event(event, ctx))
                # message_output_created: already streamed via text deltas.
                # handoff_requested: the handoff_occured note covers it.
    except BaseException:
        # Cancellation (client gone) or a failed ctx write: stop the run so its
        # background model/tool tasks don't keep going.  Errors raised by the
        # run itself leave it complete, so this is a no-op for them.
        if not getattr(result, "is_complete", True):
            result.cancel()
        raise

    await _write_usage(ctx, result)


def agent_tool_stream_handler(
    ctx: StreamContext[Any],
    *,
    nested_tool_calls: bool = False,
    detect_tool_error: ToolErrorDetector | None = None,
) -> Callable[[AgentToolStreamEvent], Awaitable[None]]:
    """
    Build an ``on_stream`` callback for ``Agent.as_tool(...)``.

    Events from a nested agent run (a sub-agent called as a tool, possibly
    several in parallel) never touch the top-level text part.  The parent
    tool part (keyed by the originating ``tool_call.call_id``) is created and
    resolved by :func:`consume_agent_stream` as usual; this handler only adds
    the nested activity.  By default it is rendered as whole-item reasoning
    notes, each prefixed with the sub-agent's name:

    * a nested tool call → ``"[Researcher] Calling search"``
    * a nested final message → ``"[Researcher] <message text>"``
    * a nested handoff → ``"[Researcher] Handing off to X"``

    Per-token deltas from nested runs are not streamed, so parallel sub-agents
    cannot interleave characters inside one reasoning part.

    Parameters
    ----------
    ctx:
        The same ``StreamContext`` passed to :func:`consume_agent_stream`.
    nested_tool_calls:
        When ``True``, nested tool calls are rendered as their own tool parts
        (input, then output or error) instead of reasoning notes.
    detect_tool_error:
        Same as in :func:`consume_agent_stream`; used when
        *nested_tool_calls* is ``True``.
    """
    detect = detect_tool_error or _default_tool_error

    async def on_stream(payload: AgentToolStreamEvent) -> None:
        if ctx.is_finished:
            return
        event = payload["event"]
        if event.type != "run_item_stream_event":
            return
        agent_name = getattr(payload.get("agent"), "name", None) or "agent"
        name, item = event.name, event.item

        if name == "tool_called":
            if nested_tool_calls:
                await _begin_tool(ctx, item)
            else:
                await _write_note(ctx, f"[{agent_name}] Calling {_tool_name(item)}")
        elif name == "tool_output":
            if nested_tool_calls:
                await _end_tool(ctx, item, detect)
        elif name == "message_output_created":
            text = _message_text(item.raw_item)
            if text:
                await _write_note(ctx, f"[{agent_name}] {text}")
        elif name == "handoff_occured":
            target = getattr(getattr(item, "target_agent", None), "name", None)
            await _write_note(
                ctx, f"[{agent_name}] Handing off to {target or 'another agent'}"
            )

    return on_stream


__all__ = [
    "SDK_DEFAULT_TOOL_ERROR",
    "McpEventHook",
    "ToolErrorDetector",
    "agent_tool_stream_handler",
    "consume_agent_stream",
]
