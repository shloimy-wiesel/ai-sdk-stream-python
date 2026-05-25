"""
Tests for stream_exclude and store_exclude filtering in StreamContext.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python import StreamContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def collect_stream(ctx: StreamContext) -> list[dict]:
    """Drain ctx.stream() and return parsed event dicts (skipping [DONE])."""
    events = []
    async for chunk in ctx.stream():
        if chunk.strip() == "data: [DONE]":
            break
        raw = chunk.removeprefix("data: ").strip()
        if raw:
            events.append(json.loads(raw))
    return events


# ---------------------------------------------------------------------------
# stream_exclude tests
# ---------------------------------------------------------------------------


async def test_stream_exclude_single_string():
    """A single string in stream_exclude is stripped from text deltas."""
    ctx = StreamContext(stream_exclude=["<think>"])

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("Hello <think>world")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"] == "Hello world"


async def test_stream_exclude_multiple_strings():
    """Multiple strings in stream_exclude are all stripped."""
    ctx = StreamContext(stream_exclude=["<think>", "</think>"])

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>Let me reason</think> answer")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"] == "Let me reason answer"


async def test_stream_exclude_empty_delta_skipped():
    """A delta that becomes empty after filtering produces no SSE event."""
    ctx = StreamContext(stream_exclude=["<think>"])

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>")
        await ctx.write_text("hello")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert len(text_deltas) == 1
    assert text_deltas[0]["delta"] == "hello"


async def test_stream_exclude_reasoning():
    """stream_exclude also applies to write_reasoning deltas."""
    ctx = StreamContext(stream_exclude=["<internal>"])

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_reasoning("Think <internal>carefully")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    reasoning_deltas = [e for e in events if e.get("type") == "reasoning-delta"]
    assert len(reasoning_deltas) == 1
    assert reasoning_deltas[0]["delta"] == "Think carefully"


# ---------------------------------------------------------------------------
# store_exclude tests
# ---------------------------------------------------------------------------


async def test_store_exclude_single_string():
    """A single string in store_exclude is stripped from collected text."""
    ctx = StreamContext(store_exclude=["<think>"], collect=True)

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("Hello <think>world")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    await collect_stream(ctx)

    assert ctx.record is not None
    assert ctx.record.text == "Hello world"


async def test_store_exclude_empty_delta_not_recorded():
    """A delta that becomes empty after store filtering is not appended."""
    ctx = StreamContext(store_exclude=["<think>"], collect=True)

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>")
        await ctx.write_text("hello")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    await collect_stream(ctx)

    assert ctx.record is not None
    assert ctx.record.text == "hello"
    assert ctx.record.answer_tokens == 5


async def test_store_exclude_reasoning():
    """store_exclude applies to write_reasoning as well."""
    ctx = StreamContext(store_exclude=["<think>", "</think>"], collect=True)

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_reasoning("<think>internal</think>")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    await collect_stream(ctx)

    assert ctx.record is not None
    assert ctx.record.reasoning == "internal"


# ---------------------------------------------------------------------------
# Independence tests
# ---------------------------------------------------------------------------


async def test_stream_only_filtering():
    """stream_exclude filters stream but store keeps original."""
    ctx = StreamContext(
        stream_exclude=["<think>", "</think>"],
        store_exclude=None,
        collect=True,
    )

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>reasoning</think>answer")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "reasoninganswer"

    assert ctx.record is not None
    assert ctx.record.text == "<think>reasoning</think>answer"


async def test_store_only_filtering():
    """store_exclude filters record but stream keeps original."""
    ctx = StreamContext(
        stream_exclude=None,
        store_exclude=["<think>", "</think>"],
        collect=True,
    )

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>reasoning</think>answer")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "<think>reasoning</think>answer"

    assert ctx.record is not None
    assert ctx.record.text == "reasoninganswer"


async def test_different_filters_for_stream_and_store():
    """stream_exclude and store_exclude can have different strings."""
    ctx = StreamContext(
        stream_exclude=["<think>"],
        store_exclude=["</think>"],
        collect=True,
    )

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>data</think>")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "data</think>"

    assert ctx.record is not None
    assert ctx.record.text == "<think>data"


# ---------------------------------------------------------------------------
# No-op / default behaviour
# ---------------------------------------------------------------------------


async def test_no_filtering_default():
    """None (default) produces no filtering — same as current behaviour."""
    ctx = StreamContext(collect=True)

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>hello</think>")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "<think>hello</think>"

    assert ctx.record is not None
    assert ctx.record.text == "<think>hello</think>"


async def test_empty_list_no_filtering():
    """An empty list produces no filtering."""
    ctx = StreamContext(stream_exclude=[], store_exclude=[], collect=True)

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("<think>hello</think>")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "<think>hello</think>"

    assert ctx.record is not None
    assert ctx.record.text == "<think>hello</think>"


async def test_string_mid_delta():
    """Filter string appearing in the middle of a delta."""
    ctx = StreamContext(
        stream_exclude=["SECRET"], collect=True, store_exclude=["SECRET"]
    )

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("before SECRET after")
        await ctx.finish()

    asyncio.create_task(_work(ctx))
    events = await collect_stream(ctx)

    text_deltas = [e for e in events if e.get("type") == "text-delta"]
    assert text_deltas[0]["delta"] == "before  after"

    assert ctx.record is not None
    assert ctx.record.text == "before  after"
