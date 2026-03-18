"""
db_service.py — Simulated database operations.

Key pattern: receives ``ctx: StreamContext`` as a parameter and writes
reasoning / status updates directly into the stream while doing its work.
The caller does not have to know anything about the DB internals — it just
passes the context through.
"""

from __future__ import annotations

import asyncio

from ai_sdk_stream_python import StreamContext

# ---------------------------------------------------------------------------
# Fake data store
# ---------------------------------------------------------------------------

_DOCUMENTS: list[dict] = [
    {
        "id": "doc-1",
        "title": "Python Async Patterns",
        "snippet": "asyncio.Queue is the canonical way to build producer-consumer pipelines…",
        "url": "https://docs.python.org/3/library/asyncio-queue.html",
    },
    {
        "id": "doc-2",
        "title": "Vercel AI SDK v6 Streaming",
        "snippet": "The UIMessageStream protocol uses SSE with typed JSON events…",
        "url": "https://sdk.vercel.ai/docs/streaming",
    },
    {
        "id": "doc-3",
        "title": "LlamaIndex Workflows",
        "snippet": "Context.write_event_to_stream() enqueues an event for the handler…",
        "url": "https://docs.llamaindex.ai/en/stable/module_guides/workflow/",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def search_documents(
    query: str, *, ctx: StreamContext, tool_call_id: str | None = None
) -> list[dict]:
    """
    Full-text search over the (simulated) document store.

    Emits a tool call event so the UI can render a "Searching knowledge
    base…" card, then returns the matching documents and emits source citations.
    """
    handle = await ctx.begin_tool_call(
        "searchDocuments",
        {"query": query, "limit": 3},
        tool_call_id=tool_call_id,
    )

    await asyncio.sleep(0.15)  # simulate search latency

    q = query.lower()
    results = [
        doc
        for doc in _DOCUMENTS
        if q in doc["title"].lower() or q in doc["snippet"].lower()
    ]
    if not results:
        results = _DOCUMENTS[:2]  # always return something for the demo

    await ctx.complete_tool_call(
        handle.toolCallId,
        {"count": len(results), "results": [{"id": d["id"], "title": d["title"]} for d in results]},
    )

    for doc in results:
        await ctx.write_source(
            source_id=doc["id"],
            url=doc["url"],
            title=doc["title"],
        )

    return results
