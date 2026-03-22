"""
llm_service.py — Real LLM integration via OpenAI-compatible API.

Reads configuration from environment variables:
  LLM_BASE_URL  — OpenAI-compatible base URL
                  e.g. http://10.111.117.4:8070/v1
  LLM_API_KEY   — API key ("-" for unauthenticated endpoints)
  LLM_MODEL     — Model ID

Handles multi-turn tool calling: when the LLM requests ``search_documents``,
the tool is executed via db_service (which streams tool events to the UI via
``ctx``), the result is appended to the message history, and the LLM is called
again for the final answer.
"""

from __future__ import annotations

import json
import os

from openai import AsyncOpenAI

from ai_sdk_stream_python import StreamContext

from . import db_service

# ---------------------------------------------------------------------------
# Client & configuration (read at import time so errors surface early)
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_API_KEY", "-"),
)

_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "You have access to a document search tool — use it when you need to look up information."
)

# ---------------------------------------------------------------------------
# Tool definitions sent to the LLM
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the knowledge base for relevant documents and information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def chat(messages: list[dict], *, ctx: StreamContext) -> None:
    """
    Stream a response for the given message history using the real LLM.

    *messages* should be a list of ``{"role": ..., "content": ...}`` dicts
    (user and assistant turns only — the system prompt is prepended here).

    Handles multi-turn tool calling transparently:
    1. Call LLM with tool definitions.
    2. If the LLM returns a ``tool_calls`` finish reason, execute each tool
       via db_service (which streams tool-input/output events to the UI).
    3. Append the tool results and loop back to step 1 for the continuation.
    4. When the LLM returns ``stop``, the text has already been streamed.
    """
    current_messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *messages,
    ]

    # Tool-calling loop — runs until the model stops calling tools.
    while True:
        stream = await _client.chat.completions.create(
            model=_MODEL,
            messages=current_messages,
            tools=_TOOLS,
            stream=True,
        )

        # Accumulate the full streamed response for the message history.
        content_parts: list[str] = []
        tool_calls_buffer: dict[int, dict] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta

            if delta.content:
                await ctx.write_text(delta.content)
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["arguments"] += tc.function.arguments

        # No tool calls → final answer already streamed, we are done.
        if finish_reason != "tool_calls" or not tool_calls_buffer:
            break

        # ── Execute tool calls ──────────────────────────────────────────
        # Record the assistant's tool call request in the message history.
        assistant_msg: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls_buffer.values()
            ],
        }
        current_messages.append(assistant_msg)

        # Move to a new step so each tool call gets its own UI card.
        await ctx.new_step()

        tool_results: list[dict] = []
        for tc_data in tool_calls_buffer.values():
            if tc_data["name"] == "search_documents":
                try:
                    args = json.loads(tc_data["arguments"])
                    query = args.get("query", "")
                    # db_service streams tool-input-* and source-url events.
                    docs = await db_service.search_documents(
                        query,
                        ctx=ctx,
                        tool_call_id=tc_data["id"],
                    )
                    result: dict = {
                        "results": [
                            {"title": d["title"], "snippet": d["snippet"]} for d in docs
                        ],
                    }
                except Exception as exc:
                    await ctx.fail_tool_call(tc_data["id"], str(exc))
                    result = {"error": str(exc)}
            else:
                result = {"error": f"Unknown tool: {tc_data['name']}"}

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_data["id"],
                    "content": json.dumps(result),
                }
            )

        current_messages.extend(tool_results)

        # Open a fresh step for the continuation text.
        await ctx.new_step()
