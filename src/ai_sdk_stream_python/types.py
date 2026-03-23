"""
Pydantic models for deserialising incoming AI SDK v6 ``useChat`` requests.

Backends receive a ``ChatRequest`` body; messages carry polymorphic ``parts``
arrays that this module models as a discriminated union.

Usage::

    from ai_sdk_stream_python.types import ChatRequest, UIMessage


    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        for msg in request.messages:
            for part in msg.parts:
                if part.type == "text":
                    print(part.text)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

# ── Part models ────────────────────────────────────────────────────────────────


class TextUIPart(BaseModel):
    """Plain-text content part (type: ``"text"``)."""

    type: Literal["text"] = "text"
    text: str
    state: Literal["streaming", "done"] | None = None


class ReasoningUIPart(BaseModel):
    """Chain-of-thought reasoning part (type: ``"reasoning"``)."""

    type: Literal["reasoning"] = "reasoning"
    text: str
    state: Literal["streaming", "done"] | None = None


class FileUIPart(BaseModel):
    """File attachment part (type: ``"file"``)."""

    type: Literal["file"] = "file"
    mediaType: str
    url: str
    filename: str | None = None


class SourceUrlUIPart(BaseModel):
    """URL citation / source part (type: ``"source-url"``)."""

    type: Literal["source-url"] = "source-url"
    sourceId: str
    url: str
    title: str | None = None


class SourceDocumentUIPart(BaseModel):
    """Document source part (type: ``"source-document"``)."""

    type: Literal["source-document"] = "source-document"
    sourceId: str
    mediaType: str
    title: str
    filename: str | None = None


class StepStartUIPart(BaseModel):
    """Step boundary marker (type: ``"step-start"``, no additional fields)."""

    type: Literal["step-start"] = "step-start"


class DataUIPart(BaseModel):
    """Custom data part (type: ``"data-{name}"``).

    The ``type`` field is dynamic (e.g. ``"data-users"``).  Check it with
    ``part.type.removeprefix("data-")`` to get the data name.
    """

    model_config = ConfigDict(extra="allow")

    type: str  # "data-{name}"
    id: str | None = None
    data: Any = None


class ToolUIPart(BaseModel):
    """Tool invocation part.

    Handles both *static* tool parts (type ``"tool-{name}"``) and *dynamic*
    tool parts (type ``"dynamic-tool"``).

    Static tools — whose name is known at development time — encode the tool
    name in the ``type`` field (e.g. ``"tool-search"``).  Dynamic tools carry
    an explicit ``toolName`` field and use ``type = "dynamic-tool"``.

    Tool call lifecycle states (AI SDK v6):

    * ``"input-streaming"`` — tool input is being streamed
    * ``"input-available"``  — tool input is complete, awaiting execution
    * ``"approval-requested"`` — waiting for user approval before execution
    * ``"approval-responded"`` — user has responded to approval request
    * ``"output-available"`` — tool has returned a result
    * ``"output-error"``     — tool execution failed (see ``errorText``)
    * ``"output-denied"``    — tool was denied by user approval
    """

    model_config = ConfigDict(extra="allow")

    type: str  # "tool-{name}" or "dynamic-tool"
    toolCallId: str
    toolName: str | None = None  # explicit name for "dynamic-tool" parts
    title: str | None = None
    providerExecuted: bool | None = None
    state: Literal[
        "input-streaming",
        "input-available",
        "approval-requested",
        "approval-responded",
        "output-available",
        "output-error",
        "output-denied",
    ]
    input: Any = None
    output: Any = None
    errorText: str | None = None


# ── Discriminated union ────────────────────────────────────────────────────────


def _discriminate_part(v: Any) -> str:
    """Map a raw part value to a discriminator tag."""
    if isinstance(v, dict):
        t: str = v.get("type", "")
    else:
        t = getattr(v, "type", "")
    if t in (
        "text",
        "reasoning",
        "file",
        "source-url",
        "source-document",
        "step-start",
    ):
        return t
    if t == "dynamic-tool" or t.startswith("tool-"):
        return "tool"
    if t.startswith("data-"):
        return "data"
    return "text"  # safe fallback


MessagePart = Annotated[
    Annotated[TextUIPart, Tag("text")]
    | Annotated[ReasoningUIPart, Tag("reasoning")]
    | Annotated[FileUIPart, Tag("file")]
    | Annotated[SourceUrlUIPart, Tag("source-url")]
    | Annotated[SourceDocumentUIPart, Tag("source-document")]
    | Annotated[StepStartUIPart, Tag("step-start")]
    | Annotated[ToolUIPart, Tag("tool")]
    | Annotated[DataUIPart, Tag("data")],
    Discriminator(_discriminate_part),
]
"""Discriminated union of all AI SDK v6 ``UIMessage`` part types."""


# ── UIMessage ──────────────────────────────────────────────────────────────────


class UIMessage(BaseModel):
    """An AI SDK v6 ``UIMessage`` as sent in ``useChat`` request bodies.

    ``parts`` is a polymorphic array; each element is one of the concrete
    ``*UIPart`` types above.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart] = Field(default_factory=list)
    metadata: Any = None


# ── ChatRequest ────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body sent by AI SDK v6 ``useChat``.

    ``extra = "allow"`` lets application-specific fields (e.g.
    ``selectedChatModel``, ``selectedVisibilityType``) pass through without
    requiring the library to enumerate them.  Access them via
    ``request.model_extra``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    messages: list[UIMessage]
    trigger: Literal["submit-message", "regenerate-message"] | str


__all__ = [
    # Parts
    "TextUIPart",
    "ReasoningUIPart",
    "FileUIPart",
    "SourceUrlUIPart",
    "SourceDocumentUIPart",
    "StepStartUIPart",
    "DataUIPart",
    "ToolUIPart",
    # Union
    "MessagePart",
    # Messages
    "UIMessage",
    "ChatRequest",
]
