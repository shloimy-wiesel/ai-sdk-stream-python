"""
collect.py — StreamRecord and supporting dataclasses for stream data collection.

When StreamContext is created with ``collect=True``, all emitted content is
accumulated here and exposed via ``ctx.record`` after the stream finishes.
Useful for persisting the conversation turn to a database or audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """

    message_id: str
    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    finish_reason: str | None = None
    step_count: int = 0

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
            "finish_reason": self.finish_reason,
            "step_count": self.step_count,
        }


__all__ = ["SourceRecord", "StreamRecord", "ToolCallRecord"]
