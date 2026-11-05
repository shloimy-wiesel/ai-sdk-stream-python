"""
testing — ready-made fake streams for trying out a frontend without an LLM.

``test_stream()`` writes canned text (and, optionally, canned reasoning and
tool calls) into a :class:`StreamContext`, chunk by chunk with a small delay,
so it looks like a real model response in ``useChat``.

Usage (FastAPI)::

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from ai_sdk_stream_python import StreamContext
    from ai_sdk_stream_python.testing import test_stream

    app = FastAPI()


    @app.post("/chat")
    async def chat() -> StreamingResponse:
        ctx = StreamContext()
        await ctx.run(lambda c: test_stream(c, reasoning=True, tool_calls=True))
        return StreamingResponse(
            ctx.stream(),
            media_type="text/event-stream",
            headers=ctx.response_headers,
        )

Or in one line with :func:`test_stream_response`::

    @app.post("/chat")
    async def chat() -> StreamingResponse:
        return await test_stream_response(reasoning=True, tool_calls=True)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from fastapi.responses import StreamingResponse

from .context import StreamContext

__all__ = [
    "DEFAULT_REASONING",
    "DEFAULT_TEXT",
    "DEFAULT_TOOL_CALLS",
    "FakeToolCall",
    "test_stream",
    "test_stream_response",
]


@dataclass(frozen=True)
class FakeToolCall:
    """A canned tool call: its name, input arguments, and the result it returns."""

    tool_name: str
    input: dict[str, Any]
    output: Any


DEFAULT_REASONING = (
    "The user is asking about the weather in Paris and what to do there. "
    "I should first look up the current weather, then search for a few "
    "activities that fit the conditions, and finally summarize everything "
    "in a short, friendly answer."
)

DEFAULT_TOOL_CALLS: tuple[FakeToolCall, ...] = (
    FakeToolCall(
        tool_name="get_weather",
        input={"city": "Paris", "unit": "celsius"},
        output={"city": "Paris", "temperature": 18, "condition": "Partly cloudy"},
    ),
    FakeToolCall(
        tool_name="search_activities",
        input={"city": "Paris", "weather": "partly cloudy", "limit": 3},
        output=[
            {"name": "Louvre Museum", "type": "indoor"},
            {"name": "Seine river cruise", "type": "outdoor"},
            {"name": "Montmartre walk", "type": "outdoor"},
        ],
    ),
)

DEFAULT_TEXT = (
    "Here's a quick overview for **Paris** today:\n\n"
    "- **Weather:** 18°C and partly cloudy — a light jacket is enough.\n"
    "- **Things to do:**\n"
    "  1. Visit the Louvre Museum if it starts to rain.\n"
    "  2. Take a Seine river cruise in the afternoon.\n"
    "  3. Walk around Montmartre for the views at sunset.\n\n"
    "Enjoy your day! 🌤️"
)


def _chunks(text: str) -> list[str]:
    """Split text into word-sized chunks that keep their trailing whitespace."""
    return re.findall(r"\S+\s*|\s+", text)


async def _sleep(delay: float) -> None:
    if delay > 0:
        await asyncio.sleep(delay)


async def test_stream(
    ctx: StreamContext[Any],
    *,
    reasoning: bool = False,
    tool_calls: bool = False,
    delay: float = 0.03,
    text: str | None = None,
    reasoning_text: str | None = None,
    tools: list[FakeToolCall] | tuple[FakeToolCall, ...] | None = None,
    finish: bool = True,
) -> None:
    """
    Write a fake, realistic-looking assistant response into *ctx*.

    The order is: reasoning (if enabled) → tool calls (if enabled) → a new
    step → text → ``finish``.  Text and reasoning are streamed word by word,
    and tool inputs are streamed as JSON deltas, with *delay* seconds between
    chunks.

    Parameters
    ----------
    ctx:
        The context to write into.
    reasoning:
        Also stream a reasoning (thinking) block before the answer.
    tool_calls:
        Also stream tool calls (input + output) before the answer.
    delay:
        Seconds to wait between chunks.  Use ``0`` in unit tests.
    text, reasoning_text, tools:
        Override the canned content.  Defaults are :data:`DEFAULT_TEXT`,
        :data:`DEFAULT_REASONING` and :data:`DEFAULT_TOOL_CALLS`.
    finish:
        Call ``ctx.finish()`` at the end.  Set ``False`` to keep writing
        to the context afterwards.
    """
    if reasoning:
        for chunk in _chunks(reasoning_text or DEFAULT_REASONING):
            await ctx.write_reasoning(chunk)
            await _sleep(delay)

    if tool_calls:
        for tool in DEFAULT_TOOL_CALLS if tools is None else tools:
            handle = await ctx.start_tool_input(tool.tool_name)
            for chunk in _chunks(json.dumps(tool.input)):
                await ctx.stream_tool_input_delta(handle.toolCallId, chunk)
                await _sleep(delay)
            await ctx.finish_tool_input(handle.toolCallId, tool.tool_name, tool.input)
            await _sleep(delay * 5)  # pretend the tool is running
            await ctx.complete_tool_call(handle.toolCallId, tool.output)
        await ctx.new_step()

    for chunk in _chunks(text or DEFAULT_TEXT):
        await ctx.write_text(chunk)
        await _sleep(delay)

    if finish:
        await ctx.finish()


# Stop pytest from collecting this as a test when users import it in test files.
test_stream.__test__ = False  # type: ignore[attr-defined]


async def test_stream_response(
    *,
    reasoning: bool = False,
    tool_calls: bool = False,
    delay: float = 0.03,
    text: str | None = None,
    reasoning_text: str | None = None,
    tools: list[FakeToolCall] | tuple[FakeToolCall, ...] | None = None,
) -> StreamingResponse:
    """
    Return a ready-to-serve ``StreamingResponse`` that plays :func:`test_stream`.

    Takes the same content options as :func:`test_stream`.
    """
    ctx: StreamContext[Any] = StreamContext()

    async def _work(c: StreamContext[Any]) -> None:
        await test_stream(
            c,
            reasoning=reasoning,
            tool_calls=tool_calls,
            delay=delay,
            text=text,
            reasoning_text=reasoning_text,
            tools=tools,
        )

    await ctx.run(_work)
    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )


test_stream_response.__test__ = False  # type: ignore[attr-defined]
