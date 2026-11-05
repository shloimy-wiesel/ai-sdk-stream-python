"""
Conformance tests: every chunk we emit must validate against a mirror of the
AI SDK v6 ``uiMessageChunkSchema`` (``ai`` package, ``src/ui-message-stream``).

The mirror forbids unknown keys so that a misnamed field (e.g. ``error``
instead of ``errorText``) fails here instead of in the browser.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ai_sdk_stream_python import StreamContext

ProviderMetadata = dict[str, dict[str, Any]]
FinishReason = Literal[
    "stop", "length", "content-filter", "tool-calls", "error", "other"
]


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _TextStart(_Chunk):
    type: Literal["text-start"]
    id: str
    providerMetadata: ProviderMetadata | None = None


class _TextDelta(_Chunk):
    type: Literal["text-delta"]
    id: str
    delta: str
    providerMetadata: ProviderMetadata | None = None


class _TextEnd(_Chunk):
    type: Literal["text-end"]
    id: str
    providerMetadata: ProviderMetadata | None = None


class _Error(_Chunk):
    type: Literal["error"]
    errorText: str


class _ToolInputStart(_Chunk):
    type: Literal["tool-input-start"]
    toolCallId: str
    toolName: str
    inputEncoding: Literal["json", "text"] | None = None
    providerExecuted: bool | None = None
    providerMetadata: ProviderMetadata | None = None
    toolMetadata: Any = None
    dynamic: bool | None = None
    title: str | None = None


class _ToolInputDelta(_Chunk):
    type: Literal["tool-input-delta"]
    toolCallId: str
    inputTextDelta: str


class _ToolInputAvailable(_Chunk):
    type: Literal["tool-input-available"]
    toolCallId: str
    toolName: str
    input: Any
    providerExecuted: bool | None = None
    providerMetadata: ProviderMetadata | None = None
    toolMetadata: Any = None
    dynamic: bool | None = None
    title: str | None = None


class _ToolOutputAvailable(_Chunk):
    type: Literal["tool-output-available"]
    toolCallId: str
    output: Any
    providerExecuted: bool | None = None
    providerMetadata: ProviderMetadata | None = None
    toolMetadata: Any = None
    dynamic: bool | None = None
    preliminary: bool | None = None


class _ToolOutputError(_Chunk):
    type: Literal["tool-output-error"]
    toolCallId: str
    errorText: str
    providerExecuted: bool | None = None
    providerMetadata: ProviderMetadata | None = None
    toolMetadata: Any = None
    dynamic: bool | None = None


class _ReasoningStart(_Chunk):
    type: Literal["reasoning-start"]
    id: str
    providerMetadata: ProviderMetadata | None = None


class _ReasoningDelta(_Chunk):
    type: Literal["reasoning-delta"]
    id: str
    delta: str
    providerMetadata: ProviderMetadata | None = None


class _ReasoningEnd(_Chunk):
    type: Literal["reasoning-end"]
    id: str
    providerMetadata: ProviderMetadata | None = None


class _SourceUrl(_Chunk):
    type: Literal["source-url"]
    sourceId: str
    url: str
    title: str | None = None
    providerMetadata: ProviderMetadata | None = None


class _File(_Chunk):
    type: Literal["file"]
    url: str
    mediaType: str
    providerMetadata: ProviderMetadata | None = None


class _StartStep(_Chunk):
    type: Literal["start-step"]


class _FinishStep(_Chunk):
    type: Literal["finish-step"]


class _Start(_Chunk):
    type: Literal["start"]
    messageId: str | None = None
    messageMetadata: Any = None


class _Finish(_Chunk):
    type: Literal["finish"]
    finishReason: FinishReason | None = None
    messageMetadata: Any = None


class _Abort(_Chunk):
    type: Literal["abort"]
    reason: str | None = None


class _Data(_Chunk):
    type: str
    id: str | None = None
    data: Any
    transient: bool | None = None


_CHUNK_MODELS: dict[str, type[_Chunk]] = {
    m.model_fields["type"].annotation.__args__[0]: m  # type: ignore[union-attr]
    for m in (
        _TextStart,
        _TextDelta,
        _TextEnd,
        _Error,
        _ToolInputStart,
        _ToolInputDelta,
        _ToolInputAvailable,
        _ToolOutputAvailable,
        _ToolOutputError,
        _ReasoningStart,
        _ReasoningDelta,
        _ReasoningEnd,
        _SourceUrl,
        _File,
        _StartStep,
        _FinishStep,
        _Start,
        _Finish,
        _Abort,
    )
}


def validate_chunk(chunk: dict[str, Any]) -> None:
    """Raise if *chunk* does not match the AI SDK v6 UI message chunk schema."""
    ctype = chunk.get("type")
    assert isinstance(ctype, str), chunk
    if ctype.startswith("data-"):
        TypeAdapter(_Data).validate_python(chunk)
        assert "data" in chunk, chunk
        return
    model = _CHUNK_MODELS.get(ctype)
    assert model is not None, f"unknown chunk type {ctype!r}"
    model.model_validate(chunk)


async def _run(producer) -> list[dict[str, Any]]:
    ctx = StreamContext()
    asyncio.create_task(producer(ctx))
    events = []
    async for line in ctx.stream():
        if line.strip() == "data: [DONE]":
            break
        events.append(json.loads(line.removeprefix("data: ")))
    return events


async def test_tool_output_error_uses_error_text():
    async def work(ctx: StreamContext):
        h = await ctx.begin_tool_call("t", {})
        await ctx.fail_tool_call(h.toolCallId, "boom")
        await ctx.finish()

    events = await _run(work)
    err = next(e for e in events if e["type"] == "tool-output-error")
    assert err == {
        "type": "tool-output-error",
        "toolCallId": err["toolCallId"],
        "errorText": "boom",
    }
    validate_chunk(err)


async def test_every_emitted_chunk_matches_v6_schema():
    async def work(ctx: StreamContext):
        await ctx.write_reasoning("thinking")
        await ctx.write_text("hello")
        h = await ctx.begin_tool_call("search", {"q": "x"})
        await ctx.complete_tool_call(h.toolCallId, {"hits": 1})
        h2 = await ctx.start_tool_input("calc")
        await ctx.stream_tool_input_delta(h2.toolCallId, '{"a":')
        await ctx.finish_tool_input(h2.toolCallId, "calc", {"a": 1})
        await ctx.fail_tool_call(h2.toolCallId, "division by zero")
        await ctx.write_source("s1", "https://example.com", title="Example")
        await ctx.write_file("https://example.com/a.png", "image/png")
        await ctx.write_data("status", {"ok": True})
        await ctx.write_data("empty", None, transient=True)
        await ctx.finish(message_metadata={"k": "v"})

    events = await _run(work)
    assert {e["type"] for e in events} >= {
        "start",
        "start-step",
        "reasoning-start",
        "reasoning-delta",
        "reasoning-end",
        "text-start",
        "text-delta",
        "text-end",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-available",
        "tool-output-available",
        "tool-output-error",
        "source-url",
        "file",
        "data-status",
        "finish-step",
        "finish",
    }
    for e in events:
        validate_chunk(e)


@pytest.mark.parametrize("terminal", ["error", "abort"])
async def test_terminal_chunks_match_v6_schema(terminal: str):
    async def work(ctx: StreamContext):
        await ctx.write_text("hi")
        if terminal == "error":
            await ctx.error("bad")
        else:
            await ctx.abort("stopped")

    for e in await _run(work):
        validate_chunk(e)


def test_validator_rejects_legacy_error_key():
    with pytest.raises(ValidationError):
        validate_chunk({"type": "tool-output-error", "toolCallId": "c", "error": "x"})
