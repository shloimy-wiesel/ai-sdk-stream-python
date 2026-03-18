"""
StateStore — a dict-backed async key-value store with dot-path access.

Inspired by the llama-index-workflows InMemoryStateStore.  Provides
coroutine-safe get/set via an asyncio.Lock and supports dot-separated paths
for reading/writing into nested dicts, lists, and object attributes.

Example::

    store = StateStore()
    await store.set("user.name", "Alice")
    name = await store.get("user.name")           # "Alice"
    await store.set("scores.0", 99)               # list index
    await store.get("missing.key", default=None)  # None
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

_SENTINEL = object()
_MAX_DEPTH: int = 100


def _traverse(obj: Any, segment: str) -> Any:
    """Follow one path segment into *obj* (dict key, list index, or attribute)."""
    if isinstance(obj, dict):
        return obj[segment]
    try:
        return obj[int(segment)]
    except (ValueError, TypeError, IndexError):
        pass
    return getattr(obj, segment)


def _assign(obj: Any, segment: str, value: Any) -> None:
    """Assign *value* to *segment* of *obj* (dict key, list index, or attribute)."""
    if isinstance(obj, dict):
        obj[segment] = value
        return
    try:
        obj[int(segment)] = value
        return
    except (ValueError, TypeError, IndexError):
        pass
    setattr(obj, segment, value)


class StateStore:
    """
    Async key-value store with dot-path access.

    All reads and writes are protected by an ``asyncio.Lock`` so they are
    safe to call from concurrent coroutines within the same event loop.
    """

    _MAX_DEPTH: ClassVar[int] = _MAX_DEPTH

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get(self, path: str, default: Any = _SENTINEL) -> Any:
        """
        Return the value at *path*.

        Raises ``KeyError`` when the path does not exist and no *default* was
        supplied.
        """
        segments = path.split(".") if path else []
        if len(segments) > self._MAX_DEPTH:
            raise ValueError(f"Path exceeds maximum depth of {self._MAX_DEPTH}")

        async with self._lock:
            try:
                value: Any = self._data
                for seg in segments:
                    value = _traverse(value, seg)
                return value
            except (KeyError, IndexError, AttributeError, TypeError):
                if default is not _SENTINEL:
                    return default
                raise KeyError(f"Path '{path}' not found in state")

    async def set(self, path: str, value: Any) -> None:
        """
        Write *value* at *path*, creating intermediate dicts as needed.
        """
        if not path:
            raise ValueError("path cannot be empty")

        segments = path.split(".")
        if len(segments) > self._MAX_DEPTH:
            raise ValueError(f"Path exceeds maximum depth of {self._MAX_DEPTH}")

        async with self._lock:
            current: Any = self._data
            for seg in segments[:-1]:
                try:
                    nxt = _traverse(current, seg)
                except (KeyError, AttributeError, TypeError):
                    nxt: dict[str, Any] = {}
                    _assign(current, seg, nxt)
                current = nxt
            _assign(current, segments[-1], value)

    async def delete(self, path: str) -> None:
        """Remove the key at *path* (only works on dict nodes)."""
        if not path:
            raise ValueError("path cannot be empty")

        segments = path.split(".")
        async with self._lock:
            current: Any = self._data
            for seg in segments[:-1]:
                current = _traverse(current, seg)
            last = segments[-1]
            if isinstance(current, dict):
                current.pop(last, None)
            else:
                try:
                    delattr(current, last)
                except AttributeError:
                    pass

    async def clear(self) -> None:
        """Remove all stored state."""
        async with self._lock:
            self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the top-level state dict (not lock-protected)."""
        return dict(self._data)


__all__ = ["StateStore"]
