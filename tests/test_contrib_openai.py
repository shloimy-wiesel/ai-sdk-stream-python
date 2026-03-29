"""
Tests for contrib.openai.consume_openai_stream().

Uses simple duck-typed mock objects — no openai package required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.contrib.openai import (
    ConsumeResult,
    consume_openai_stream,
    convert_to_openai_messages,
)
from ai_sdk_stream_python.types import UIMessage

# ---------------------------------------------------------------------------
# Mock helpers — duck-typed OpenAI ChatCompletionChunk objects
# ---------------------------------------------------------------------------


def _fn(name: str | None = None, arguments: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


def _tc(
    index: int,
    id: str | None = None,
    function: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=id, function=function)


def _delta(
    content: str | None = None,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
    )


def _choice(
    delta: SimpleNamespace,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _chunk(
    choices: list | None = None,
    usage: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(choices=choices or [], usage=usage)


def _usage(prompt: int = 10, completion: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


async def _make_stream(*chunks: SimpleNamespace):
    """Async generator of mock chunks."""
    for chunk in chunks:
        yield chunk


async def run_consume(
    chunks: list,
    *,
    stream_tool_deltas: bool = False,
    collect: bool = False,
) -> tuple[ConsumeResult, list[dict]]:
    """Run consume_openai_stream and return (result, list of SSE event dicts)."""
    ctx = StreamContext(collect=collect)
    events: list[dict] = []

    async def work(ctx: StreamContext) -> None:
        result = await consume_openai_stream(
            _make_stream(*chunks),
            ctx,
            stream_tool_deltas=stream_tool_deltas,
        )
        ctx._result = result  # stash for inspection
        await ctx.finish(finish_reason=result.finish_reason or "stop")

    asyncio.create_task(work(ctx))

    import json as _json

    async for raw in ctx.stream():
        stripped = raw.strip()
        if stripped == "data: [DONE]":
            break
        body = stripped.removeprefix("data: ")
        if body:
            events.append(_json.loads(body))

    return ctx._result, events  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TestText:
    async def test_single_text_chunk(self):
        chunks = [
            _chunk([_choice(_delta(content="hello"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "hello"
        assert result.finish_reason == "stop"
        types = [e["type"] for e in events]
        assert "text-delta" in types
        td = next(e for e in events if e["type"] == "text-delta")
        assert td["delta"] == "hello"

    async def test_multiple_text_deltas_concatenated(self):
        chunks = [
            _chunk([_choice(_delta(content="foo"))]),
            _chunk([_choice(_delta(content="bar"))]),
            _chunk([_choice(_delta(content=None), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "foobar"
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        assert len(text_deltas) == 2

    async def test_empty_content_not_written(self):
        chunks = [
            _chunk([_choice(_delta(content=None), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == ""
        assert not any(e["type"] == "text-delta" for e in events)

    async def test_chunk_with_no_choices_skipped(self):
        chunks = [
            _chunk([]),  # empty choices
            _chunk([_choice(_delta(content="hi"), finish_reason="stop")]),
        ]
        result, _ = await run_consume(chunks)
        assert result.content == "hi"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


class TestReasoning:
    async def test_reasoning_field(self):
        chunks = [
            _chunk([_choice(_delta(reasoning="think"))]),
            _chunk([_choice(_delta(content="answer"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "answer"
        reasoning_deltas = [e for e in events if e["type"] == "reasoning-delta"]
        assert len(reasoning_deltas) == 1
        assert reasoning_deltas[0]["delta"] == "think"

    async def test_reasoning_content_field(self):
        """OpenAI o-models use delta.reasoning_content."""
        chunks = [
            _chunk([_choice(_delta(reasoning_content="step 1"))]),
            _chunk([_choice(_delta(content="done"), finish_reason="stop")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "done"
        assert any(e["type"] == "reasoning-delta" for e in events)

    async def test_reasoning_content_preferred_over_reasoning(self):
        """reasoning_content takes priority when both fields are present."""
        d = SimpleNamespace(
            content=None,
            reasoning="fallback",
            reasoning_content="preferred",
            tool_calls=None,
        )
        chunks = [
            _chunk([_choice(d, finish_reason="stop")]),
        ]
        _, events = await run_consume(chunks)
        reasoning_deltas = [e for e in events if e["type"] == "reasoning-delta"]
        assert reasoning_deltas[0]["delta"] == "preferred"


# ---------------------------------------------------------------------------
# Finish reason & usage
# ---------------------------------------------------------------------------


class TestMetadata:
    async def test_finish_reason_extracted(self):
        chunks = [
            _chunk([_choice(_delta(content="hi"))]),
            _chunk([_choice(_delta(), finish_reason="length")]),
        ]
        result, _ = await run_consume(chunks)
        assert result.finish_reason == "length"

    async def test_finish_reason_none_when_absent(self):
        chunks = [
            _chunk([_choice(_delta(content="hi"))]),
        ]
        result, _ = await run_consume(chunks)
        assert result.finish_reason is None

    async def test_usage_extracted(self):
        chunks = [
            _chunk([_choice(_delta(content="x"), finish_reason="stop")]),
            _chunk([], usage=_usage(prompt=5, completion=10)),
        ]
        result, _ = await run_consume(chunks)
        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        }

    async def test_usage_none_when_not_provided(self):
        chunks = [_chunk([_choice(_delta(content="hi"), finish_reason="stop")])]
        result, _ = await run_consume(chunks)
        assert result.usage is None


# ---------------------------------------------------------------------------
# Tool calls — non-streaming (stream_tool_deltas=False)
# ---------------------------------------------------------------------------


class TestToolCallsNonStreaming:
    async def test_single_tool_call_emitted(self):
        chunks = [
            # First chunk: id + name
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_1",
                                    function=_fn(name="search", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Argument chunks
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments='{"q":'))]))]
            ),
            _chunk(
                [
                    _choice(
                        _delta(tool_calls=[_tc(0, function=_fn(arguments='"cats"}'))])
                    )
                ]
            ),
            # Finish
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["name"] == "search"
        assert json.loads(tc["arguments"]) == {"q": "cats"}

        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-input-delta" not in types

        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["toolName"] == "search"
        assert avail["input"] == {"q": "cats"}

    async def test_multiple_tool_calls_emitted_in_order(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_0",
                                    function=_fn(name="toolA", arguments=""),
                                ),
                                _tc(
                                    1,
                                    id="call_1",
                                    function=_fn(name="toolB", arguments=""),
                                ),
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, function=_fn(arguments='{"x":1}')),
                                _tc(1, function=_fn(arguments='{"y":2}')),
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "toolA"
        assert result.tool_calls[1]["name"] == "toolB"

        avail_events = [e for e in events if e["type"] == "tool-input-available"]
        assert len(avail_events) == 2
        assert avail_events[0]["toolName"] == "toolA"
        assert avail_events[1]["toolName"] == "toolB"

    async def test_tool_call_uses_provided_id(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="myid",
                                    function=_fn(name="fn", arguments="{}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.tool_calls[0]["id"] == "myid"
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["toolCallId"] == "myid"

    async def test_malformed_json_args_fallback_to_empty_dict(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="x",
                                    function=_fn(name="fn", arguments="not-json"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks)
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["input"] == {}

    async def test_zero_arg_tool(self):
        """Tool with empty arguments `{}` should emit input={}."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0, id="z", function=_fn(name="noop", arguments="{}")
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks)
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["input"] == {}

    async def test_repeated_tool_name_not_duplicated(self):
        """Name sent on multiple chunks should not be concatenated."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_1",
                                    function=_fn(name="search", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Second chunk repeats the name
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, function=_fn(name="search", arguments='{"q":'))
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [
                    _choice(
                        _delta(tool_calls=[_tc(0, function=_fn(arguments='"x"}'))])
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.tool_calls[0]["name"] == "search"  # not "searchsearch"
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert avail["toolName"] == "search"

    async def test_missing_id_uses_generated_id(self):
        """When no tc.id is provided, ConsumeResult should use the generated ID."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    # No id provided
                                    function=_fn(name="fn", arguments="{}"),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        # The result id should match the id emitted in the SSE events
        avail = next(e for e in events if e["type"] == "tool-input-available")
        assert result.tool_calls[0]["id"] == avail["toolCallId"]
        # Should be a non-empty UUID, not an empty string
        assert result.tool_calls[0]["id"] != ""


# ---------------------------------------------------------------------------
# Tool calls — streaming (stream_tool_deltas=True)
# ---------------------------------------------------------------------------


class TestToolCallsStreaming:
    async def test_streaming_emits_deltas(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="call_1",
                                    function=_fn(name="search", arguments=""),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments='{"q":'))]))]
            ),
            _chunk(
                [
                    _choice(
                        _delta(tool_calls=[_tc(0, function=_fn(arguments='"cats"}'))])
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-delta" in types
        assert "tool-input-available" in types

        deltas = [e for e in events if e["type"] == "tool-input-delta"]
        assert len(deltas) == 2
        combined = "".join(d["inputTextDelta"] for d in deltas)
        assert json.loads(combined) == {"q": "cats"}

    async def test_streaming_tool_call_ids_consistent(self):
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, id="cid", function=_fn(name="fn", arguments=""))
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments="{}"))]))]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)

        start = next(e for e in events if e["type"] == "tool-input-start")
        deltas = [e for e in events if e["type"] == "tool-input-delta"]
        avail = next(e for e in events if e["type"] == "tool-input-available")

        cid = start["toolCallId"]
        assert all(d["toolCallId"] == cid for d in deltas)
        assert avail["toolCallId"] == cid

    async def test_streaming_fallback_for_no_arguments(self):
        """Tool call with no arguments chunks falls back to begin_tool_call."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0, id="z", function=_fn(name="noop", arguments=None)
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        _, events = await run_consume(chunks, stream_tool_deltas=True)
        types = [e["type"] for e in events]
        assert "tool-input-start" in types
        assert "tool-input-available" in types
        assert "tool-input-delta" not in types

    async def test_streaming_missing_id_uses_generated_id(self):
        """When no tc.id is provided in streaming mode, ConsumeResult should use the generated ID."""
        chunks = [
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(0, function=_fn(name="fn", arguments=""))
                            ]
                        )
                    )
                ]
            ),
            _chunk(
                [_choice(_delta(tool_calls=[_tc(0, function=_fn(arguments="{}"))]))]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks, stream_tool_deltas=True)
        start = next(e for e in events if e["type"] == "tool-input-start")
        avail = next(e for e in events if e["type"] == "tool-input-available")
        # Result ID should match emitted events and not be empty
        assert result.tool_calls[0]["id"] == start["toolCallId"]
        assert result.tool_calls[0]["id"] == avail["toolCallId"]
        assert result.tool_calls[0]["id"] != ""


# ---------------------------------------------------------------------------
# Mixed text + tool calls
# ---------------------------------------------------------------------------


class TestMixed:
    async def test_text_before_tool_calls(self):
        chunks = [
            _chunk([_choice(_delta(content="Let me search."))]),
            _chunk(
                [
                    _choice(
                        _delta(
                            tool_calls=[
                                _tc(
                                    0,
                                    id="c1",
                                    function=_fn(name="search", arguments='{"q":"x"}'),
                                )
                            ]
                        )
                    )
                ]
            ),
            _chunk([_choice(_delta(), finish_reason="tool_calls")]),
        ]
        result, events = await run_consume(chunks)
        assert result.content == "Let me search."
        assert len(result.tool_calls) == 1
        types = [e["type"] for e in events]
        assert "text-delta" in types
        assert "tool-input-available" in types


# ---------------------------------------------------------------------------
# TestConvertToOpenAIMessages
# ---------------------------------------------------------------------------


def _msg(role: str, parts: list[dict[str, Any]]) -> UIMessage:
    return UIMessage.model_validate({"id": "m1", "role": role, "parts": parts})


class TestConvertToOpenAIMessages:
    # ── System messages ──────────────────────────────────────────────────────

    def test_system_message(self):
        msgs = [_msg("system", [{"type": "text", "text": "You are helpful."}])]
        result = convert_to_openai_messages(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_system_multiple_text_parts(self):
        msgs = [
            _msg(
                "system",
                [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": " Be concise."},
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert (
            result[0]["content"] == "You are helpful. Be concise."
        )  # parts concatenated as-is

    # ── User messages ────────────────────────────────────────────────────────

    def test_user_text_only(self):
        msgs = [_msg("user", [{"type": "text", "text": "Hello"}])]
        result = convert_to_openai_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_user_multiple_text_parts_concatenated(self):
        msgs = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "What is "},
                    {"type": "text", "text": "the weather?"},
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert result[0]["content"] == "What is the weather?"

    def test_user_image_file_returns_content_blocks(self):
        msgs = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "file",
                        "mediaType": "image/png",
                        "url": "https://example.com/img.png",
                    },
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Describe this"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }

    def test_user_non_image_file_skipped(self):
        msgs = [
            _msg(
                "user",
                [
                    {"type": "text", "text": "See attachment"},
                    {
                        "type": "file",
                        "mediaType": "application/pdf",
                        "url": "data:application/pdf;base64,abc",
                    },
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        # Non-image file dropped; falls back to string content
        assert result[0]["content"] == "See attachment"

    def test_user_skips_step_start_and_source_parts(self):
        msgs = [
            _msg(
                "user",
                [
                    {"type": "step-start"},
                    {"type": "text", "text": "Hello"},
                    {"type": "source-url", "sourceId": "s1", "url": "https://x.com"},
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert result[0]["content"] == "Hello"

    # ── Assistant messages ───────────────────────────────────────────────────

    def test_assistant_text_only(self):
        msgs = [_msg("assistant", [{"type": "text", "text": "Hello there."}])]
        result = convert_to_openai_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hello there."}]

    def test_assistant_tool_call_no_result(self):
        """input-available state → tool_calls only, no tool result message."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-search",
                        "toolCallId": "tc1",
                        "state": "input-available",
                        "input": {"query": "weather NYC"},
                    }
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == [
            {
                "id": "tc1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "weather NYC"}',
                },
            }
        ]

    def test_assistant_tool_call_with_result(self):
        """output-available state → tool_calls + separate tool message."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-search",
                        "toolCallId": "tc1",
                        "state": "output-available",
                        "input": {"query": "weather NYC"},
                        "output": {"temp": 72},
                    }
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "tc1"
        assert result[1] == {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": '{"temp": 72}',
        }

    def test_assistant_tool_call_output_error(self):
        """output-error state → tool_calls + tool message with errorText."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-search",
                        "toolCallId": "tc2",
                        "state": "output-error",
                        "errorText": "Network timeout",
                    }
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert len(result) == 2
        assert result[1] == {
            "role": "tool",
            "tool_call_id": "tc2",
            "content": "Network timeout",
        }

    def test_assistant_dynamic_tool(self):
        """dynamic-tool uses toolName field for the function name."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "dynamic-tool",
                        "toolCallId": "tc3",
                        "toolName": "fetch",
                        "state": "input-available",
                        "input": {"url": "https://example.com"},
                    }
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        fn = result[0]["tool_calls"][0]["function"]
        assert fn["name"] == "fetch"

    def test_assistant_text_and_tool_call(self):
        """Text content and tool_calls coexist."""
        msgs = [
            _msg(
                "assistant",
                [
                    {"type": "text", "text": "Let me look that up."},
                    {
                        "type": "tool-search",
                        "toolCallId": "tc4",
                        "state": "input-available",
                        "input": {},
                    },
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert result[0]["content"] == "Let me look that up."
        assert len(result[0]["tool_calls"]) == 1

    def test_assistant_no_content_with_tool_call(self):
        """When no text, content is None (JSON null) alongside tool_calls."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-search",
                        "toolCallId": "tc5",
                        "state": "input-available",
                        "input": {},
                    }
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1

    def test_assistant_multiple_tool_calls_with_results(self):
        """Multiple tool calls in one assistant message → multiple tool msgs."""
        msgs = [
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-search",
                        "toolCallId": "tc6",
                        "state": "output-available",
                        "input": {"q": "a"},
                        "output": {"r": 1},
                    },
                    {
                        "type": "tool-calc",
                        "toolCallId": "tc7",
                        "state": "output-available",
                        "input": {"x": 2},
                        "output": {"v": 4},
                    },
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 2
        assert result[1]["tool_call_id"] == "tc6"
        assert result[2]["tool_call_id"] == "tc7"

    # ── Reasoning ────────────────────────────────────────────────────────────

    def test_reasoning_dropped_by_default(self):
        msgs = [
            _msg(
                "assistant",
                [
                    {"type": "reasoning", "text": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            )
        ]
        result = convert_to_openai_messages(msgs)
        assert result[0]["content"] == "The answer is 42."

    def test_reasoning_included_when_flag_set(self):
        msgs = [
            _msg(
                "assistant",
                [
                    {"type": "reasoning", "text": "Let me think..."},
                    {"type": "text", "text": "The answer is 42."},
                ],
            )
        ]
        result = convert_to_openai_messages(msgs, include_reasoning=True)
        content = result[0]["content"]
        assert "<reasoning>" in content
        assert "Let me think..." in content
        assert "The answer is 42." in content

    # ── Multi-turn conversation ───────────────────────────────────────────────

    def test_multi_turn_conversation(self):
        """Full round-trip: user → assistant (with tool) → user."""
        msgs = [
            _msg("user", [{"type": "text", "text": "What's the weather?"}]),
            _msg(
                "assistant",
                [
                    {
                        "type": "tool-getWeather",
                        "toolCallId": "tc1",
                        "state": "output-available",
                        "input": {"city": "NYC"},
                        "output": {"temp": 72},
                    },
                    {"type": "text", "text": "It's 72°F in NYC."},
                ],
            ),
            _msg("user", [{"type": "text", "text": "Thanks!"}]),
        ]
        result = convert_to_openai_messages(msgs)

        assert len(result) == 4  # user, assistant, tool, user
        assert result[0] == {"role": "user", "content": "What's the weather?"}
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "It's 72°F in NYC."
        assert result[1]["tool_calls"][0]["function"]["name"] == "getWeather"
        assert result[2] == {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": '{"temp": 72}',
        }
        assert result[3] == {"role": "user", "content": "Thanks!"}

    def test_empty_messages(self):
        assert convert_to_openai_messages([]) == []
