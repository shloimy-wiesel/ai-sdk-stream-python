"""
collect.py — StreamRecord and supporting dataclasses for stream data collection.

When StreamContext is created with ``collect=True``, all emitted content is
accumulated here and exposed via ``ctx.record`` after the stream finishes.
Useful for persisting the conversation turn to a database or audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolCallRecord:
    """A single tool call with its input and resolved output (or error)."""

    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    output: Any = None
    error: str | None = None


@dataclass
class SourceRecord:
    """A cited source URL emitted via ``write_source``."""

    source_id: str
    url: str
    title: str | None = None


@dataclass
class FileRecord:
    """A file/image emitted via ``write_file``."""

    url: str
    media_type: str


@dataclass
class DataPartRecord:
    """A non-transient custom data part emitted via ``write_data``."""

    name: str
    data: dict[str, Any]
    id: str | None = None


@dataclass
class StreamRecord:
    """
    Accumulated content from one assistant stream turn.

    Populated incrementally as events are emitted; fully available
    after ``ctx.finish()`` returns.

    Attributes
    ----------
    message_id:
        The message ID of the stream (matches ``ctx.message_id``).
    text:
        All text deltas concatenated in order.
    reasoning:
        All reasoning deltas concatenated in order.
    tool_calls:
        One entry per ``begin_tool_call`` call, with ``output`` or ``error``
        filled in once ``complete_tool_call`` / ``fail_tool_call`` is called.
    sources:
        All source URLs emitted via ``write_source``.
    finish_reason:
        The finish reason passed to ``finish()``.  ``None`` if the stream
        was terminated via ``abort()`` without a proper finish.
    step_count:
        Number of steps opened during the stream (including the first
        implicit step).
    reasoning_tokens:
        Auto-counted tokens from ``write_reasoning`` deltas.
    answer_tokens:
        Auto-counted tokens from ``write_text`` deltas.
    prompt_tokens:
        Prompt/input tokens — only available if set via ``ctx.set_usage()``.
    """

    message_id: str
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    files: list[FileRecord] = field(default_factory=list)
    data_parts: list[DataPartRecord] = field(default_factory=list)
    finish_reason: str | None = None
    step_count: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    prompt_tokens: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def total_output_tokens(self) -> int:
        return self.reasoning_tokens + self.answer_tokens

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None:
            return None
        return self.prompt_tokens + self.total_output_tokens

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or None if not yet finished."""
        if self.finished_at is not None:
            return (self.finished_at - self.created_at).total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for DB persistence."""
        return {
            "message_id": self.message_id,
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": [
                {
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "input": tc.input,
                    "output": tc.output,
                    "error": tc.error,
                }
                for tc in self.tool_calls
            ],
            "sources": [
                {
                    "source_id": s.source_id,
                    "url": s.url,
                    "title": s.title,
                }
                for s in self.sources
            ],
            "files": [
                {
                    "url": f.url,
                    "media_type": f.media_type,
                }
                for f in self.files
            ],
            "data_parts": [
                {
                    "name": dp.name,
                    "data": dp.data,
                    "id": dp.id,
                }
                for dp in self.data_parts
            ],
            "finish_reason": self.finish_reason,
            "step_count": self.step_count,
            "reasoning_tokens": self.reasoning_tokens,
            "answer_tokens": self.answer_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "DataPartRecord",
    "FileRecord",
    "SourceRecord",
    "StreamRecord",
    "ToolCallRecord",
]
