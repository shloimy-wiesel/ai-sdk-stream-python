"""
contrib.openai — OpenAI integration utilities.

Two complementary functions:

* ``convert_to_openai_messages()`` — **input side**: converts AI SDK v6
  ``UIMessage`` objects (parts-based format) into
  ``list[ChatCompletionMessageParam]`` ready to pass to the OpenAI API.

* ``consume_openai_stream()`` — **output side**: consumes an OpenAI-compatible
  async stream and maps each chunk to the appropriate ``StreamContext`` call.

No hard dependency on the ``openai`` package: both functions use duck typing so
they work with any compatible objects.  The import does NOT fail if ``openai``
is not installed.

Usage (full round-trip)::

    from openai import AsyncOpenAI
    from ai_sdk_stream_python import StreamContext
    from ai_sdk_stream_python.contrib.openai import (
        consume_openai_stream,
        convert_to_openai_messages,
    )
    from ai_sdk_stream_python.types import ChatRequest

    client = AsyncOpenAI()


    async def chat(request: ChatRequest, ctx: StreamContext) -> None:
        messages = convert_to_openai_messages(request.messages)
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        )
        result = await consume_openai_stream(stream, ctx)
        await ctx.finish(finish_reason=result.finish_reason or "stop")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..context import StreamContext
from ..types import (
    FileUIPart,
    MessagePart,
    ReasoningUIPart,
    TextUIPart,
    ToolUIPart,
    UIMessage,
)

# ── convert_to_openai_messages helpers ────────────────────────────────────────


def _get_tool_name(part: ToolUIPart) -> str:
    """Return the tool name for a ToolUIPart."""
    if part.type == "dynamic-tool":
        return part.toolName or ""
    return part.type.removeprefix("tool-")


def _user_content(parts: list[MessagePart]) -> str | list[dict[str, Any]]:
    """Build the ``content`` value for a user message.

    Returns a plain string when only text parts are present.  Returns a list
    of content blocks when image file parts are included.

    Non-image files, reasoning, source, step-start, and data parts are skipped.
    """
    blocks: list[dict[str, Any]] = []
    has_image = False

    for part in parts:
        if isinstance(part, TextUIPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, FileUIPart) and part.mediaType.startswith("image/"):
            has_image = True
            blocks.append({"type": "image_url", "image_url": {"url": part.url}})

    if not has_image:
        return "".join(b["text"] for b in blocks if b["type"] == "text")
    return blocks


def _assistant_messages(
    parts: list[MessagePart],
    include_reasoning: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the assistant message dict and any following tool result messages.

    Returns a tuple of:
    - The ``role: "assistant"`` message dict (with optional ``tool_calls``).
    - A list of ``role: "tool"`` message dicts for completed tool invocations.
    """
    reasoning_chunks: list[str] = []
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for part in parts:
        if isinstance(part, ReasoningUIPart):
            if include_reasoning:
                reasoning_chunks.append(part.text)
        elif isinstance(part, TextUIPart):
            text_chunks.append(part.text)
        elif isinstance(part, ToolUIPart):
            tool_name = _get_tool_name(part)
            args = json.dumps(part.input) if part.input is not None else "{}"
            tool_calls.append(
                {
                    "id": part.toolCallId,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }
            )
            if part.state == "output-available":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.toolCallId,
                        "content": (
                            json.dumps(part.output) if part.output is not None else ""
                        ),
                    }
                )
            elif part.state == "output-error":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.toolCallId,
                        "content": part.errorText or "Tool execution failed",
                    }
                )

    content_parts: list[str] = []
    if reasoning_chunks:
        content_parts.append(f"<reasoning>\n{''.join(reasoning_chunks)}\n</reasoning>")
    content_parts.extend(text_chunks)
    content = "".join(content_parts) or None  # None → JSON null (valid with tool_calls)

    assistant: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        assistant["tool_calls"] = tool_calls

    return assistant, tool_results


# ── Public API ─────────────────────────────────────────────────────────────────


def convert_to_openai_messages(
    messages: list[UIMessage],
    *,
    include_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Convert AI SDK v6 ``UIMessage`` objects to OpenAI chat message params.

    Transforms the parts-based ``UIMessage`` format sent by ``useChat`` into
    ``list[ChatCompletionMessageParam]`` suitable for the OpenAI API.

    Part mapping:

    ========================  ================================================
    AI SDK v6 part            OpenAI mapping
    ========================  ================================================
    ``TextUIPart``            ``content`` string / ``{"type": "text"}`` block
    ``FileUIPart`` (image)    ``{"type": "image_url", "image_url": {...}}``
    ``FileUIPart`` (other)    *skipped*
    ``ToolUIPart``            assistant ``tool_calls`` entry
    ``ToolUIPart`` (result)   separate ``role: "tool"`` message
    ``ReasoningUIPart``       *skipped* unless *include_reasoning* is ``True``
    All other parts           *skipped*
    ========================  ================================================

    For ``ToolUIPart`` state semantics:

    * ``"output-available"`` → assistant ``tool_calls`` **and** a ``role:
      "tool"`` result message.
    * ``"output-error"`` → assistant ``tool_calls`` and a ``role: "tool"``
      message containing the ``errorText``.
    * Any other state (``"input-streaming"``, ``"input-available"``, etc.) →
      assistant ``tool_calls`` only; no result message.

    Parameters
    ----------
    messages:
        The ``messages`` list from a ``ChatRequest`` (or any sequence of
        ``UIMessage`` objects).
    include_reasoning:
        When ``True``, ``ReasoningUIPart`` text is prepended to the assistant
        message content inside ``<reasoning>…</reasoning>`` tags.  Defaults
        to ``False`` (reasoning parts are dropped).

    Returns
    -------
    list[dict[str, Any]]
        A list of message dicts compatible with ``ChatCompletionMessageParam``.
        The list may be longer than *messages* when assistant messages contain
        completed tool calls (each adds a ``role: "tool"`` entry).
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            text = "".join(
                part.text for part in msg.parts if isinstance(part, TextUIPart)
            )
            result.append({"role": "system", "content": text})

        elif msg.role == "user":
            content = _user_content(msg.parts)
            result.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            assistant_msg, tool_msgs = _assistant_messages(msg.parts, include_reasoning)
            result.append(assistant_msg)
            result.extend(tool_msgs)

    return result


@dataclass
class ConsumeResult:
    """Accumulated data from a consumed OpenAI stream.

    Attributes
    ----------
    content:
        All text deltas concatenated in order.
    tool_calls:
        One entry per tool call: ``{"id": str, "name": str, "arguments": str}``.
        ``arguments`` is the raw JSON string as sent by the model.
    finish_reason:
        The finish reason from the last chunk (e.g. ``"stop"``,
        ``"tool_calls"``, ``"length"``).  ``None`` if no chunk carried one.
    usage:
        Token counts when the stream was created with
        ``stream_options={"include_usage": True}``, otherwise ``None``.
        Keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
    """

    content: str = ""
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


async def consume_openai_stream(
    stream: Any,
    ctx: StreamContext,
    *,
    stream_tool_deltas: bool = False,
) -> ConsumeResult:
    """
    Consume an OpenAI-compatible async chat completion stream.

    Iterates over *stream* and maps each chunk to the appropriate
    ``StreamContext`` call:

    - ``delta.content`` → ``ctx.write_text()``
    - ``delta.reasoning`` / ``delta.reasoning_content`` →
      ``ctx.write_reasoning()`` (supports both OpenAI o-models and
      Anthropic-via-OpenAI field names)
    - ``delta.tool_calls`` → buffered by index; emitted at end, or streamed
      incrementally when *stream_tool_deltas* is ``True``

    This function only consumes the stream — it does **not** call
    ``ctx.finish()``.  The caller is responsible for finishing the context.

    Parameters
    ----------
    stream:
        An ``openai.AsyncStream[ChatCompletionChunk]`` or any async iterable
        whose items expose ``choices`` and optionally ``usage``.
    ctx:
        The active ``StreamContext`` to write events to.
    stream_tool_deltas:
        When ``True``, emit ``tool-input-delta`` events as argument chunks
        arrive (requires ``ctx.start_tool_input`` / ``ctx.stream_tool_input_delta``
        / ``ctx.finish_tool_input``).  Defaults to ``False``, where only
        ``tool-input-start`` + ``tool-input-available`` are emitted after all
        chunks have been received.

    Returns
    -------
    ConsumeResult
        Accumulated ``content``, ``tool_calls``, ``finish_reason``, and
        ``usage``.
    """
    content = ""
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    # index → {id, name, arguments, handle}
    # handle is a ToolCallHandle (set only when stream_tool_deltas=True)
    tool_buffer: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        # Usage — present on the final chunk when stream_options={"include_usage": True}
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = {
                "prompt_tokens": chunk_usage.prompt_tokens,
                "completion_tokens": chunk_usage.completion_tokens,
                "total_tokens": chunk_usage.total_tokens,
            }

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.delta

        # Track finish_reason (non-None value from the last chunk with choices)
        fr = getattr(choice, "finish_reason", None)
        if fr is not None:
            finish_reason = fr

        # ── Text ────────────────────────────────────────────────────────────
        text_delta = getattr(delta, "content", None)
        if text_delta:
            content += text_delta
            await ctx.write_text(text_delta)

        # ── Reasoning ───────────────────────────────────────────────────────
        # OpenAI o-models use `reasoning_content`; Anthropic via OpenAI uses `reasoning`
        reasoning_delta = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reasoning_delta:
            await ctx.write_reasoning(reasoning_delta)

        # ── Tool calls ──────────────────────────────────────────────────────
        tool_calls = getattr(delta, "tool_calls", None)
        if not tool_calls:
            continue

        for tc in tool_calls:
            idx: int = tc.index
            if idx not in tool_buffer:
                tool_buffer[idx] = {
                    "id": "",
                    "name": "",
                    "arguments": "",
                    "handle": None,
                }
            buf = tool_buffer[idx]

            tc_id = getattr(tc, "id", None)
            if tc_id:
                buf["id"] = tc_id

            fn = getattr(tc, "function", None)
            if fn is None:
                continue

            fn_name = getattr(fn, "name", None)
            if fn_name:
                buf["name"] += fn_name

            fn_args = getattr(fn, "arguments", None)
            if not fn_args:
                continue

            buf["arguments"] += fn_args

            if stream_tool_deltas:
                # Start streaming once the tool name is known
                if buf["handle"] is None and buf["name"]:
                    buf["handle"] = await ctx.start_tool_input(
                        buf["name"],
                        tool_call_id=buf["id"] or None,
                    )
                if buf["handle"] is not None:
                    await ctx.stream_tool_input_delta(
                        buf["handle"].toolCallId,
                        fn_args,
                    )

    # ── Finalize tool calls ──────────────────────────────────────────────────
    result_tool_calls: list[dict[str, str]] = []
    for idx in sorted(tool_buffer):
        buf = tool_buffer[idx]
        try:
            parsed_input: dict[str, Any] = json.loads(buf["arguments"] or "{}")
        except json.JSONDecodeError:
            parsed_input = {}

        if stream_tool_deltas:
            if buf["handle"] is not None:
                await ctx.finish_tool_input(
                    buf["handle"].toolCallId,
                    buf["name"],
                    parsed_input,
                )
            else:
                # No arguments were streamed (zero-arg tool or empty chunks);
                # fall back to the non-streaming path.
                await ctx.begin_tool_call(
                    buf["name"],
                    parsed_input,
                    tool_call_id=buf["id"] or None,
                )
        else:
            await ctx.begin_tool_call(
                buf["name"],
                parsed_input,
                tool_call_id=buf["id"] or None,
            )

        result_tool_calls.append(
            {
                "id": buf["id"],
                "name": buf["name"],
                "arguments": buf["arguments"],
            }
        )

    return ConsumeResult(
        content=content,
        tool_calls=result_tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


__all__ = ["ConsumeResult", "consume_openai_stream", "convert_to_openai_messages"]
