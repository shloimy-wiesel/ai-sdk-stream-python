"""
Tests for the incoming request Pydantic models (types.py).
"""

from __future__ import annotations

import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_sdk_stream_python.types import (
    ChatRequest,
    DataUIPart,
    FileUIPart,
    ReasoningUIPart,
    SourceDocumentUIPart,
    SourceUrlUIPart,
    StepStartUIPart,
    TextUIPart,
    ToolUIPart,
    UIMessage,
)

# ── Part models ─────────────────────────────────────────────────────────────────


class TestTextUIPart:
    def test_basic(self):
        part = TextUIPart(text="Hello")
        assert part.type == "text"
        assert part.text == "Hello"
        assert part.state is None

    def test_with_state(self):
        part = TextUIPart(text="Hi", state="done")
        assert part.state == "done"

    def test_extra_fields_allowed(self):
        part = TextUIPart.model_validate({"text": "Hi", "futureField": "x"})
        assert part.model_extra["futureField"] == "x"


class TestReasoningUIPart:
    def test_basic(self):
        part = ReasoningUIPart(text="Let me think...")
        assert part.type == "reasoning"
        assert part.text == "Let me think..."

    def test_extra_fields_allowed(self):
        part = ReasoningUIPart.model_validate({"text": "...", "futureField": True})
        assert part.model_extra["futureField"] is True


class TestFileUIPart:
    def test_basic(self):
        part = FileUIPart(mediaType="image/png", url="https://example.com/img.png")
        assert part.type == "file"
        assert part.mediaType == "image/png"
        assert part.filename is None

    def test_with_filename(self):
        part = FileUIPart(
            mediaType="application/pdf", url="data:...", filename="doc.pdf"
        )
        assert part.filename == "doc.pdf"

    def test_extra_fields_allowed(self):
        part = FileUIPart.model_validate(
            {
                "mediaType": "image/png",
                "url": "https://example.com/x.png",
                "futureField": 1,
            }
        )
        assert part.model_extra["futureField"] == 1


class TestSourceUrlUIPart:
    def test_basic(self):
        part = SourceUrlUIPart(sourceId="s1", url="https://example.com")
        assert part.type == "source-url"
        assert part.title is None

    def test_with_title(self):
        part = SourceUrlUIPart(
            sourceId="s1", url="https://example.com", title="Example"
        )
        assert part.title == "Example"

    def test_extra_fields_allowed(self):
        part = SourceUrlUIPart.model_validate(
            {"sourceId": "s1", "url": "https://example.com", "futureField": "y"}
        )
        assert part.model_extra["futureField"] == "y"


class TestSourceDocumentUIPart:
    def test_basic(self):
        part = SourceDocumentUIPart(
            sourceId="d1", mediaType="application/pdf", title="Report"
        )
        assert part.type == "source-document"
        assert part.filename is None

    def test_extra_fields_allowed(self):
        part = SourceDocumentUIPart.model_validate(
            {
                "sourceId": "d1",
                "mediaType": "application/pdf",
                "title": "Report",
                "futureField": "z",
            }
        )
        assert part.model_extra["futureField"] == "z"


class TestStepStartUIPart:
    def test_basic(self):
        part = StepStartUIPart()
        assert part.type == "step-start"

    def test_extra_fields_allowed(self):
        part = StepStartUIPart.model_validate({"futureField": "x"})
        assert part.model_extra["futureField"] == "x"


class TestDataUIPart:
    def test_basic(self):
        part = DataUIPart(type="data-users", data=[{"id": 1}])
        assert part.type == "data-users"
        assert part.data == [{"id": 1}]
        assert part.type.removeprefix("data-") == "users"

    def test_with_id(self):
        part = DataUIPart(type="data-scores", id="abc", data={"score": 42})
        assert part.id == "abc"

    def test_null_data(self):
        part = DataUIPart(type="data-empty", data=None)
        assert part.data is None

    def test_transient_field(self):
        part = DataUIPart(type="data-tmp", data={"x": 1}, transient=True)
        assert part.transient is True

    def test_transient_defaults_none(self):
        part = DataUIPart(type="data-tmp", data={"x": 1})
        assert part.transient is None


class TestToolUIPart:
    def test_static_input_available(self):
        part = ToolUIPart(
            type="tool-search",
            toolCallId="tc1",
            state="input-available",
            input={"query": "hello"},
        )
        assert part.type == "tool-search"
        assert part.state == "input-available"
        assert part.input == {"query": "hello"}
        assert part.output is None

    def test_static_output_available(self):
        part = ToolUIPart(
            type="tool-search",
            toolCallId="tc1",
            state="output-available",
            input={"query": "hello"},
            output={"results": []},
        )
        assert part.state == "output-available"
        assert part.output == {"results": []}

    def test_output_error(self):
        part = ToolUIPart(
            type="tool-search",
            toolCallId="tc1",
            state="output-error",
            errorText="Network timeout",
        )
        assert part.state == "output-error"
        assert part.errorText == "Network timeout"

    def test_dynamic(self):
        part = ToolUIPart(
            type="dynamic-tool",
            toolCallId="tc2",
            toolName="fetch",
            state="input-streaming",
        )
        assert part.type == "dynamic-tool"
        assert part.toolName == "fetch"

    def test_input_streaming(self):
        part = ToolUIPart(
            type="tool-calculator",
            toolCallId="tc3",
            state="input-streaming",
        )
        assert part.state == "input-streaming"

    def test_dynamic_tool_requires_tool_name(self):
        with pytest.raises(ValidationError):
            ToolUIPart(type="dynamic-tool", toolCallId="tc4", state="input-available")


# ── UIMessage discriminated union ───────────────────────────────────────────────


class TestUIMessageDiscriminatedUnion:
    def test_text_part(self):
        msg = UIMessage(
            id="m1",
            role="user",
            parts=[{"type": "text", "text": "Hello"}],
        )
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, TextUIPart)
        assert part.text == "Hello"

    def test_file_part(self):
        msg = UIMessage(
            id="m2",
            role="user",
            parts=[
                {"type": "text", "text": "See this image"},
                {
                    "type": "file",
                    "mediaType": "image/png",
                    "url": "https://example.com/a.png",
                },
            ],
        )
        assert len(msg.parts) == 2
        assert isinstance(msg.parts[1], FileUIPart)

    def test_tool_part(self):
        msg = UIMessage(
            id="m3",
            role="assistant",
            parts=[
                {
                    "type": "tool-search",
                    "toolCallId": "tc1",
                    "state": "output-available",
                    "input": {"q": "test"},
                    "output": {"hits": 3},
                }
            ],
        )
        part = msg.parts[0]
        assert isinstance(part, ToolUIPart)
        assert part.state == "output-available"

    def test_dynamic_tool_part(self):
        msg = UIMessage(
            id="m4",
            role="assistant",
            parts=[
                {
                    "type": "dynamic-tool",
                    "toolCallId": "tc2",
                    "toolName": "fetch",
                    "state": "input-available",
                    "input": {"url": "https://example.com"},
                }
            ],
        )
        assert isinstance(msg.parts[0], ToolUIPart)
        assert msg.parts[0].toolName == "fetch"

    def test_data_part(self):
        msg = UIMessage(
            id="m5",
            role="assistant",
            parts=[{"type": "data-scores", "data": {"score": 99}}],
        )
        assert isinstance(msg.parts[0], DataUIPart)
        assert msg.parts[0].type == "data-scores"

    def test_reasoning_part(self):
        msg = UIMessage(
            id="m6",
            role="assistant",
            parts=[{"type": "reasoning", "text": "Thinking..."}],
        )
        assert isinstance(msg.parts[0], ReasoningUIPart)

    def test_source_url_part(self):
        msg = UIMessage(
            id="m7",
            role="assistant",
            parts=[
                {"type": "source-url", "sourceId": "s1", "url": "https://example.com"}
            ],
        )
        assert isinstance(msg.parts[0], SourceUrlUIPart)

    def test_step_start_part(self):
        msg = UIMessage(
            id="m8",
            role="assistant",
            parts=[{"type": "step-start"}],
        )
        assert isinstance(msg.parts[0], StepStartUIPart)

    def test_source_document_part(self):
        msg = UIMessage(
            id="m9",
            role="assistant",
            parts=[
                {
                    "type": "source-document",
                    "sourceId": "d1",
                    "mediaType": "application/pdf",
                    "title": "Report Q1",
                }
            ],
        )
        assert isinstance(msg.parts[0], SourceDocumentUIPart)

    def test_empty_parts(self):
        msg = UIMessage(id="m10", role="user", parts=[])
        assert msg.parts == []

    def test_metadata(self):
        msg = UIMessage(
            id="m11",
            role="user",
            parts=[],
            metadata={"custom": "value"},
        )
        assert msg.metadata == {"custom": "value"}

    def test_extra_fields_allowed(self):
        """UIMessage passes through unknown fields for forward-compat."""
        msg = UIMessage.model_validate(
            {"id": "m12", "role": "user", "parts": [], "createdAt": "2026-01-01"}
        )
        assert msg.id == "m12"
        assert msg.model_extra["createdAt"] == "2026-01-01"

    def test_unknown_part_type_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            UIMessage.model_validate(
                {
                    "id": "m13",
                    "role": "user",
                    "parts": [{"type": "future-unknown-part"}],
                }
            )


# ── ChatRequest ─────────────────────────────────────────────────────────────────


class TestChatRequest:
    def test_basic(self):
        req = ChatRequest(
            id="chat1",
            messages=[
                {
                    "id": "m1",
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hi"}],
                }
            ],
            trigger="submit-message",
        )
        assert req.id == "chat1"
        assert len(req.messages) == 1
        assert req.trigger == "submit-message"

    def test_regenerate_trigger(self):
        req = ChatRequest(
            id="chat2",
            messages=[],
            trigger="regenerate-message",
        )
        assert req.trigger == "regenerate-message"

    def test_extra_fields(self):
        """App-specific fields (e.g. selectedChatModel) are preserved."""
        req = ChatRequest.model_validate(
            {
                "id": "chat3",
                "messages": [],
                "trigger": "submit-message",
                "selectedChatModel": "gpt-4o",
                "selectedVisibilityType": "private",
            }
        )
        assert req.model_extra["selectedChatModel"] == "gpt-4o"
        assert req.model_extra["selectedVisibilityType"] == "private"

    def test_multiple_messages(self):
        req = ChatRequest(
            id="chat4",
            messages=[
                {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "Q"}]},
                {
                    "id": "m2",
                    "role": "assistant",
                    "parts": [{"type": "text", "text": "A"}],
                },
            ],
            trigger="submit-message",
        )
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(messages=[], trigger="submit-message")  # type: ignore[call-arg]

    def test_missing_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(id="c", trigger="submit-message")  # type: ignore[call-arg]

    def test_unknown_trigger_accepted(self):
        """trigger is str — any value from future SDK versions is accepted."""
        req = ChatRequest(id="c", messages=[], trigger="future-trigger")
        assert req.trigger == "future-trigger"
