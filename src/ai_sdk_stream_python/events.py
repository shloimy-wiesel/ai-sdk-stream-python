"""
Pydantic models for the Vercel AI SDK v6 UIMessageStream wire protocol.

Every event has a `type` discriminator field (Literal string).
The discriminated union `UIMessageStreamEvent` can be used for type-narrowing.

Wire format: each event is serialised as ``data: {json}\n\n`` (SSE).
The stream is terminated with ``data: [DONE]\n\n``.
Response header: ``x-vercel-ai-ui-message-stream: v1``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    def encode(self) -> str:
        """Serialise as a single SSE data line."""
        return "data: " + self.model_dump_json(exclude_none=True) + "\n\n"


# ── Message lifecycle ──────────────────────────────────────────────────────────


class StartEvent(BaseEvent):
    type: Literal["start"] = "start"
    messageId: str
    messageMetadata: dict[str, Any] | None = None


class StartStepEvent(BaseEvent):
    type: Literal["start-step"] = "start-step"


class FinishStepEvent(BaseEvent):
    type: Literal["finish-step"] = "finish-step"


class FinishEvent(BaseEvent):
    type: Literal["finish"] = "finish"
    finishReason: str = "stop"
    messageMetadata: dict[str, Any] | None = None


# ── Reasoning (chain-of-thought) ───────────────────────────────────────────────


class ReasoningStartEvent(BaseEvent):
    type: Literal["reasoning-start"] = "reasoning-start"
    id: str


class ReasoningDeltaEvent(BaseEvent):
    type: Literal["reasoning-delta"] = "reasoning-delta"
    id: str
    delta: str


class ReasoningEndEvent(BaseEvent):
    type: Literal["reasoning-end"] = "reasoning-end"
    id: str


# ── Text ───────────────────────────────────────────────────────────────────────


class TextStartEvent(BaseEvent):
    type: Literal["text-start"] = "text-start"
    id: str


class TextDeltaEvent(BaseEvent):
    type: Literal["text-delta"] = "text-delta"
    id: str
    delta: str


class TextEndEvent(BaseEvent):
    type: Literal["text-end"] = "text-end"
    id: str


# ── Tool calls ─────────────────────────────────────────────────────────────────


class ToolInputStartEvent(BaseEvent):
    type: Literal["tool-input-start"] = "tool-input-start"
    toolCallId: str
    toolName: str


class ToolInputDeltaEvent(BaseEvent):
    type: Literal["tool-input-delta"] = "tool-input-delta"
    toolCallId: str
    inputTextDelta: str


class ToolInputAvailableEvent(BaseEvent):
    type: Literal["tool-input-available"] = "tool-input-available"
    toolCallId: str
    toolName: str
    input: dict[str, Any]


class ToolOutputAvailableEvent(BaseEvent):
    type: Literal["tool-output-available"] = "tool-output-available"
    toolCallId: str
    output: Any


class ToolOutputErrorEvent(BaseEvent):
    type: Literal["tool-output-error"] = "tool-output-error"
    toolCallId: str
    error: str


# ── Sources ────────────────────────────────────────────────────────────────────


class SourceUrlEvent(BaseEvent):
    type: Literal["source-url"] = "source-url"
    sourceId: str
    url: str
    title: str | None = None


# ── Files ──────────────────────────────────────────────────────────────────────


class FileEvent(BaseEvent):
    type: Literal["file"] = "file"
    url: str
    mediaType: str


# ── Error ──────────────────────────────────────────────────────────────────────


class ErrorEvent(BaseEvent):
    type: Literal["error"] = "error"
    errorText: str


# ── Abort ──────────────────────────────────────────────────────────────────────


class AbortEvent(BaseEvent):
    type: Literal["abort"] = "abort"
    reason: str | None = None


# ── Discriminated union ────────────────────────────────────────────────────────

UIMessageStreamEvent = Annotated[
    StartEvent
    | StartStepEvent
    | ReasoningStartEvent
    | ReasoningDeltaEvent
    | ReasoningEndEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ToolInputStartEvent
    | ToolInputDeltaEvent
    | ToolInputAvailableEvent
    | ToolOutputAvailableEvent
    | ToolOutputErrorEvent
    | SourceUrlEvent
    | FileEvent
    | ErrorEvent
    | AbortEvent
    | FinishStepEvent
    | FinishEvent,
    Field(discriminator="type"),
]

__all__ = [
    "BaseEvent",
    "UIMessageStreamEvent",
    "StartEvent",
    "StartStepEvent",
    "FinishStepEvent",
    "FinishEvent",
    "ReasoningStartEvent",
    "ReasoningDeltaEvent",
    "ReasoningEndEvent",
    "TextStartEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ToolInputStartEvent",
    "ToolInputDeltaEvent",
    "ToolInputAvailableEvent",
    "ToolOutputAvailableEvent",
    "ToolOutputErrorEvent",
    "SourceUrlEvent",
    "FileEvent",
    "ErrorEvent",
    "AbortEvent",
]
