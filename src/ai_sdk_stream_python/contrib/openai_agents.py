"""
contrib.openai_agents — OpenAI Agents SDK (``openai-agents``) adapter.

Maps the high-level event stream of ``Runner.run_streamed()`` onto
``StreamContext`` writes so an Agents SDK backend can serve the Vercel AI SDK
v6 UIMessageStream without writing its own encoder.

Entry points:

* ``consume_agent_stream(result, ctx)`` — iterates
  ``RunResultStreaming.stream_events()`` and writes text, reasoning, tool
  calls, handoffs and usage to *ctx*.
* ``agent_tool_stream_handler(ctx)`` — returns a callback for
  ``Agent.as_tool(on_stream=...)`` so events from nested agents-as-tools are
  rendered as reasoning notes (or their own tool parts) instead of leaking into
  the top-level answer text.
* ``NestedAgentTranscripts(ctx)`` — opt-in, request-scoped alternative to
  ``agent_tool_stream_handler``: streams each sub-agent's live transcript
  (text, reasoning, nested tools) into its *own* parent tool card as AI SDK
  preliminary tool results.  Pass ``transcripts.on_stream`` to ``as_tool`` and
  ``nested=transcripts`` to ``consume_agent_stream``.

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

import asyncio
import copy
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..context import StreamContext

if TYPE_CHECKING:
    from agents import AgentToolStreamEvent, RunResultStreaming
    from agents.items import ToolCallOutputItem
    from agents.stream_events import RunItemStreamEvent

logger = logging.getLogger(__name__)

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


def _reasoning_summary(raw: Any) -> str:
    return "\n\n".join(_get(s, "text") or "" for s in _get(raw, "summary") or [])


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
    nested: NestedAgentTranscripts | None = None,
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
    nested:
        A :class:`NestedAgentTranscripts` whose ``on_stream`` is wired into
        the run's agents-as-tools.  Its sub-agent transcripts are published
        once the parent tool part exists and finalized from the parent's
        ``tool_output``.  Exceptions raised inside its callback surface here.
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
                elif (
                    data.type == "response.output_item.done"
                    and _get(data.item, "type") == "reasoning"
                    and _get(data.item, "id") not in streamed_reasoning
                ):
                    # The SDK emits ``reasoning_item_created`` only after the
                    # turn's text has streamed, so write the summary here to
                    # keep reasoning ahead of the answer text.
                    streamed_reasoning.add(_get(data.item, "id"))
                    summary = _reasoning_summary(data.item)
                    if summary:
                        await ctx.write_reasoning(summary)

            elif event.type == "agent_updated_stream_event":
                await ctx.new_step()

            elif event.type == "run_item_stream_event":
                name, item = event.name, event.item
                if name == "tool_called":
                    call_id = await _begin_tool(ctx, item)
                    if nested is not None:
                        nested._parent_started(call_id)
                elif name == "tool_output":
                    if nested is None or not await nested._finalize(item, detect):
                        await _end_tool(ctx, item, detect)
                elif name == "reasoning_item_created":
                    raw = item.raw_item
                    if _get(raw, "id") not in streamed_reasoning:
                        summary = _reasoning_summary(raw)
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
    finally:
        if nested is not None:
            nested.close()

    if nested is not None:
        nested._raise_pending()
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


# ── Nested agent transcripts (preliminary tool results) ──────────────────────

#: ``output["kind"]`` of the envelope published by :class:`NestedAgentTranscripts`.
SUBAGENT_OUTPUT_KIND = "subagent"
#: ``output["version"]`` of that envelope.
SUBAGENT_OUTPUT_VERSION = 1


@dataclass
class _ChildRun:
    """Accumulated UI message of one sub-agent invocation (one parent call)."""

    call_id: str
    agent_name: str
    parts: list[dict[str, Any]] = field(default_factory=list)
    # Child-local ids → part; scoped to this call so ids that repeat across
    # parallel child runs (e.g. two runs that both produce ``msg_1``) are
    # kept apart.
    texts: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasonings: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    new_step: bool = True
    dirty: bool = False
    last_publish: float | None = None
    timer: asyncio.TimerHandle | None = None
    error: BaseException | None = None


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:  # an uncopyable nested tool output — copy the shell
        return {
            **value,
            "message": {
                **value["message"],
                "parts": [dict(p) for p in value["message"]["parts"]],
            },
        }


class NestedAgentTranscripts:
    """
    Stream sub-agent transcripts into their parent tool cards.

    A request-scoped, opt-in alternative to :func:`agent_tool_stream_handler`.
    Each ``Agent.as_tool(on_stream=...)`` invocation is identified by its
    originating ``tool_call.call_id`` (never by agent name), so parallel calls
    of the same agent stay separate.  Its events are accumulated into a child
    UI message and published as AI SDK *preliminary tool results* on that
    parent tool part::

        {
            "type": "tool-output-available",
            "toolCallId": "call_research_1",
            "preliminary": true,
            "output": {
                "kind": "subagent",
                "version": 1,
                "agentName": "Researcher",
                "message": {
                    "id": "child_call_research_1",
                    "role": "assistant",
                    "parts": [
                        {"type": "step-start"},
                        {
                            "type": "text",
                            "text": "Findings so far…",
                            "state": "streaming",
                        },
                    ],
                },
            },
        }

    Every output is a complete snapshot (the AI SDK replaces the previous
    one).  When the parent run reports the tool's output, the card is
    finalized once with the same envelope plus ``"result"`` — the value the
    parent model actually received, which is left untouched — and without
    ``preliminary``.  If the tool failed (per *detect_tool_error*), a
    ``tool-output-error`` is sent instead; the last partial transcript stays
    in ``ToolCallRecord.preliminary_output``.

    Child ``message.parts`` use AI SDK ``UIMessage`` part shapes:
    ``step-start``, ``text`` / ``reasoning`` (``state`` ``"streaming"`` or
    ``"done"``), ``dynamic-tool`` (``input-available`` / ``output-available``
    / ``output-error``) and ``data-handoff`` (``{"from", "to"}``).

    Nothing is ever written to the parent's text or reasoning parts, and no
    child ``finish`` reaches the outer stream.

    Usage::

        transcripts = NestedAgentTranscripts(ctx)
        tool = researcher.as_tool(
            tool_name="research",
            tool_description="Research a topic",
            on_stream=transcripts.on_stream,
        )
        ...
        await consume_agent_stream(result, ctx, nested=transcripts)

    Behaviour details:

    * **Ordering** — snapshots are buffered until ``consume_agent_stream`` has
      emitted the parent tool part, so a callback that races the parent's
      ``tool_called`` event never produces output before input.
    * **Throttling** — at most one snapshot per *throttle* seconds per call
      while tokens stream (full snapshots on every token cost quadratic
      bandwidth); pending changes are flushed on the trailing edge, on every
      completed run item, and at finalization.  ``throttle=0`` publishes on
      every event.
    * **Missing tool call** — events without ``payload["tool_call"]`` cannot be
      routed and are dropped (logged once per agent).
    * **Failures** — an exception inside the callback (which the SDK would
      log and swallow) is stored and re-raised from ``consume_agent_stream``
      when that call is finalized, so the request fails visibly instead of
      showing an incomplete "successful" card.
    * **Cancellation** — ``consume_agent_stream`` calls :meth:`close` when it
      ends for any reason: pending timers are cancelled and late callbacks
      are ignored.

    Parameters
    ----------
    ctx:
        The same ``StreamContext`` passed to :func:`consume_agent_stream`.
    throttle:
        Minimum seconds between two snapshots of the same call.  Default 0.1.
    include_reasoning:
        Include the sub-agent's reasoning parts.  Default ``True``.
    detect_tool_error:
        Detects failed *nested* tool outputs; see
        :func:`consume_agent_stream`.
    """

    def __init__(
        self,
        ctx: StreamContext[Any],
        *,
        throttle: float = 0.1,
        include_reasoning: bool = True,
        detect_tool_error: ToolErrorDetector | None = None,
    ) -> None:
        self._ctx = ctx
        self._throttle = throttle
        self._include_reasoning = include_reasoning
        self._detect = detect_tool_error or _default_tool_error
        self._runs: dict[str, _ChildRun] = {}
        self._begun: set[str] = set()
        self._done: set[str] = set()
        self._closed = False
        self._unrouted: set[str] = set()

    # ── SDK callback ──────────────────────────────────────────────────────

    async def on_stream(self, payload: AgentToolStreamEvent) -> None:
        """``Agent.as_tool(on_stream=...)`` callback."""
        if self._closed or self._ctx.is_finished:
            return
        agent_name = getattr(payload.get("agent"), "name", None) or "agent"
        call_id = _call_id(payload.get("tool_call"))
        if call_id is None:
            if agent_name not in self._unrouted:
                self._unrouted.add(agent_name)
                logger.warning(
                    "Dropping nested events from %r: no originating tool call",
                    agent_name,
                )
            return
        if call_id in self._done:
            return
        run = self._runs.get(call_id)
        if run is None:
            run = self._runs[call_id] = _ChildRun(call_id, agent_name)
        if run.error is not None:
            return
        try:
            flush = self._apply(run, payload["event"])
            self._touch(run, flush=flush)
        except Exception as exc:
            logger.exception("Failed to render nested agent event")
            run.error = exc

    # ── Accumulation ──────────────────────────────────────────────────────

    def _add(self, run: _ChildRun, part: dict[str, Any]) -> dict[str, Any]:
        if run.new_step:
            run.new_step = False
            run.parts.append({"type": "step-start"})
        run.parts.append(part)
        return part

    def _add_reasoning(self, run: _ChildRun, item_id: Any, text: str) -> None:
        if not self._include_reasoning or item_id in run.reasonings:
            return
        if text:
            run.reasonings[item_id] = self._add(
                run, {"type": "reasoning", "text": text, "state": "done"}
            )

    def _apply(self, run: _ChildRun, event: Any) -> bool:
        """Fold *event* into *run*; return ``True`` to publish immediately."""
        etype = event.type
        if etype == "raw_response_event":
            data = event.data
            dtype = _get(data, "type")
            if dtype == "response.created":
                # Item ids are only unique within one model response (Chat
                # Completions models reuse a fake id for every item).
                run.new_step = True
                run.texts.clear()
                run.reasonings.clear()
            elif dtype == "response.output_text.delta":
                part = run.texts.get(data.item_id)
                if part is None:
                    part = run.texts[data.item_id] = self._add(
                        run, {"type": "text", "text": "", "state": "streaming"}
                    )
                part["text"] += data.delta
                run.dirty = True
            elif dtype in (
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                if self._include_reasoning:
                    part = run.reasonings.get(data.item_id)
                    if part is None:
                        part = run.reasonings[data.item_id] = self._add(
                            run,
                            {"type": "reasoning", "text": "", "state": "streaming"},
                        )
                    part["text"] += data.delta
                    run.dirty = True
            elif (
                dtype == "response.output_item.done"
                and _get(data.item, "type") == "reasoning"
            ):
                item_id = _get(data.item, "id")
                if item_id in run.reasonings:
                    run.reasonings[item_id]["state"] = "done"
                else:
                    self._add_reasoning(run, item_id, _reasoning_summary(data.item))
                run.dirty = True
            return False

        if etype == "agent_updated_stream_event":
            run.new_step = True
            return False

        if etype != "run_item_stream_event":
            return False

        name, item = event.name, event.item
        raw = getattr(item, "raw_item", None)
        if name == "message_output_created":
            msg_id = _get(raw, "id")
            text = _message_text(raw)
            part = run.texts.get(msg_id) if msg_id is not None else None
            if part is None:
                # Streamed under a different item id: match by content.
                part = next(
                    (
                        p
                        for p in run.texts.values()
                        if p["state"] == "streaming" and p["text"] == text
                    ),
                    None,
                )
            if part is not None:
                # Already streamed as deltas: the completed item is
                # authoritative, but never duplicated.
                part["text"] = text or part["text"]
                part["state"] = "done"
            elif text:
                part = self._add(run, {"type": "text", "text": text, "state": "done"})
                if msg_id is not None:
                    run.texts[msg_id] = part
        elif name == "reasoning_item_created":
            item_id = _get(raw, "id")
            if item_id in run.reasonings:
                run.reasonings[item_id]["state"] = "done"
            else:
                self._add_reasoning(run, item_id, _reasoning_summary(raw))
        elif name == "tool_called":
            tcid = _call_id(raw) or f"tool_{len(run.tools)}"
            part = self._add(
                run,
                {
                    "type": "dynamic-tool",
                    "toolName": _tool_name(item),
                    "toolCallId": tcid,
                    "state": "input-available",
                    "input": _tool_input(raw),
                },
            )
            run.tools[tcid] = part
            if _get(raw, "type") in _HOSTED_TOOL_TYPES:
                error = _get(raw, "error")
                if error:
                    part.update(state="output-error", errorText=str(error))
                else:
                    output = _get(raw, "output")
                    part.update(
                        state="output-available",
                        output=output if output is not None else _dump(raw),
                    )
        elif name == "tool_output":
            tcid = _call_id(raw)
            part = run.tools.get(tcid) if tcid is not None else None
            if part is not None:
                error = self._detect(item)
                if error is not None:
                    part.update(state="output-error", errorText=error)
                else:
                    part.update(state="output-available", output=item.output)
        elif name == "handoff_occured":
            source = getattr(getattr(item, "source_agent", None), "name", None)
            target = getattr(getattr(item, "target_agent", None), "name", None)
            self._add(
                run,
                {"type": "data-handoff", "data": {"from": source, "to": target}},
            )
        else:
            return False
        run.dirty = True
        return True

    # ── Publishing ────────────────────────────────────────────────────────

    def _envelope(self, run: _ChildRun) -> dict[str, Any]:
        return {
            "kind": SUBAGENT_OUTPUT_KIND,
            "version": SUBAGENT_OUTPUT_VERSION,
            "agentName": run.agent_name,
            "message": {
                "id": f"child_{run.call_id}",
                "role": "assistant",
                "parts": run.parts,
            },
        }

    def _can_publish(self, run: _ChildRun) -> bool:
        return (
            not self._closed
            and run.call_id in self._begun
            and run.call_id not in self._done
            and not self._ctx.is_finished
        )

    def _touch(self, run: _ChildRun, *, flush: bool = False) -> None:
        if not run.dirty or not self._can_publish(run):
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        wait = (
            0.0
            if flush or run.last_publish is None
            else self._throttle - (now - run.last_publish)
        )
        if wait <= 0:
            self._publish(run)
        elif run.timer is None:
            run.timer = loop.call_later(wait, self._publish_from_timer, run)

    def _publish_from_timer(self, run: _ChildRun) -> None:
        run.timer = None
        try:
            self._publish(run)
        except Exception as exc:
            logger.exception("Failed to publish nested agent transcript")
            run.error = exc

    def _publish(self, run: _ChildRun) -> None:
        if run.timer is not None:
            run.timer.cancel()
            run.timer = None
        if not run.dirty or not self._can_publish(run):
            return
        run.dirty = False
        run.last_publish = asyncio.get_running_loop().time()
        self._ctx._write_preliminary_tool_output(
            run.call_id, _snapshot(self._envelope(run))
        )

    # ── Parent lifecycle (called by consume_agent_stream) ─────────────────

    def _parent_started(self, call_id: str) -> None:
        self._begun.add(call_id)
        run = self._runs.get(call_id)
        if run is not None and run.error is None:
            self._touch(run, flush=True)

    async def _finalize(self, item: Any, detect: ToolErrorDetector) -> bool:
        """Resolve the parent tool part; ``False`` if *item* isn't ours."""
        call_id = _call_id(item.raw_item)
        run = self._runs.pop(call_id, None) if call_id is not None else None
        if run is None or call_id is None:
            return False
        self._done.add(call_id)
        if run.timer is not None:
            run.timer.cancel()
            run.timer = None
        if run.error is not None:
            raise run.error
        for part in run.parts:
            if part.get("state") == "streaming":
                part["state"] = "done"
        error = detect(item)
        if error is not None:
            if run.dirty and call_id in self._begun:
                self._ctx._write_preliminary_tool_output(
                    call_id, _snapshot(self._envelope(run))
                )
            await self._ctx.fail_tool_call(call_id, error)
        else:
            final = self._envelope(run)
            final["result"] = item.output
            await self._ctx.complete_tool_call(call_id, _snapshot(final))
        return True

    def close(self) -> None:
        """Stop publishing: cancel pending timers and ignore late callbacks."""
        self._closed = True
        for run in self._runs.values():
            if run.timer is not None:
                run.timer.cancel()
                run.timer = None

    def _raise_pending(self) -> None:
        for run in self._runs.values():
            if run.error is not None:
                raise run.error


__all__ = [
    "SDK_DEFAULT_TOOL_ERROR",
    "SUBAGENT_OUTPUT_KIND",
    "SUBAGENT_OUTPUT_VERSION",
    "McpEventHook",
    "NestedAgentTranscripts",
    "ToolErrorDetector",
    "agent_tool_stream_handler",
    "consume_agent_stream",
]
