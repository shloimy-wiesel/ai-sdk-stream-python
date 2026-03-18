"""
db_service.py — Simulated database operations.

Key pattern: receives ``ctx: StreamContext`` as a parameter and writes
reasoning / status updates directly into the stream while doing its work.
The caller does not have to know anything about the DB internals — it just
passes the context through.

The service also uses ``ctx.store`` to persist loaded data so that downstream
services (e.g. llm_service) can read it without having to pass it explicitly.
"""

from __future__ import annotations

import asyncio

from ai_sdk_stream_python import StreamContext

# ---------------------------------------------------------------------------
# Fake data store
# ---------------------------------------------------------------------------

_USERS: dict[str, dict] = {
    "u1": {"name": "Alice", "plan": "pro", "history_count": 42},
    "u2": {"name": "Bob", "plan": "free", "history_count": 7},
}

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
# Public API — all functions accept ctx as a named parameter
# ---------------------------------------------------------------------------


async def load_user(user_id: str, *, ctx: StreamContext) -> dict:
    """
    Load user record from the (simulated) database.

    Writes a brief reasoning step to the stream so the UI can show
    "Looking up user…" while the query runs.  Stores the result in
    ``ctx.store`` so other services can access it without extra params.
    """
    await ctx.write_reasoning(f"Looking up user profile for '{user_id}'…")
    await asyncio.sleep(0.05)  # simulate DB latency

    user = _USERS.get(user_id, {"name": "Guest", "plan": "free", "history_count": 0})

    # Persist into shared state so llm_service can personalise the answer
    await ctx.store.set("user.name", user["name"])
    await ctx.store.set("user.plan", user["plan"])

    await ctx.write_reasoning(f"Loaded user: {user['name']} ({user['plan']} plan).")
    return user


async def search_documents(
    query: str, *, ctx: StreamContext, tool_call_id: str | None = None
) -> list[dict]:
    """
    Full-text search over the (simulated) document store.

    Emits a tool call event so the UI can render a "Searching knowledge
    base…" card, then returns the matching documents and emits the results.
    The tool call ID is returned by ``ctx.begin_tool_call`` — we pass it to
    ``ctx.complete_tool_call`` once the results are ready.

    If *tool_call_id* is provided it will be used for the stream events so
    that the ID matches the one issued by the upstream LLM.
    """
    handle = await ctx.begin_tool_call(
        "searchDocuments",
        {"query": query, "limit": 3},
        tool_call_id=tool_call_id,
    )

    await asyncio.sleep(0.15)  # simulate search latency

    # Simple substring filter (real impl would use vector search etc.)
    q = query.lower()
    results = [doc for doc in _DOCUMENTS if q in doc["title"].lower() or q in doc["snippet"].lower()]
    if not results:
        results = _DOCUMENTS[:2]  # always return something for the demo

    await ctx.complete_tool_call(
        handle.toolCallId,
        {"count": len(results), "results": [{"id": d["id"], "title": d["title"]} for d in results]},
    )

    # Also write source-url events so the frontend can show citations
    for doc in results:
        await ctx.write_source(
            source_id=doc["id"],
            url=doc["url"],
            title=doc["title"],
        )

    return results
