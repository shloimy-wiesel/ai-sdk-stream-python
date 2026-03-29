from __future__ import annotations

import json
from typing import Any

from services.llm_config import get_extra_body, get_model, make_client
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS

from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.contrib.openai import (
    consume_openai_stream,
    convert_to_openai_messages,
)
from ai_sdk_stream_python.types import TextUIPart, UIMessage

_REGULAR_PROMPT = (
    "You are a helpful assistant. Keep responses concise and direct.\n\n"
    "When asked to write, create, or build something, do it immediately. "
    "Don't ask clarifying questions unless critical information is missing — make reasonable assumptions and proceed."
)

_ARTIFACTS_PROMPT = """
Artifacts is a side panel that displays content alongside the conversation. It supports scripts (code), documents (text), and spreadsheets. Changes appear in real-time.

CRITICAL RULES:
1. Only call ONE tool per response. After calling any create/edit/update tool, STOP. Do not chain tools.
2. After creating or editing an artifact, NEVER output its content in chat. The user can already see it. Respond with only a 1-2 sentence confirmation.

**When to use `createDocument`:**
- When the user asks to write, create, or generate content (essays, stories, emails, reports)
- When the user asks to write code, build a script, or implement an algorithm
- You MUST specify kind: 'code' for programming, 'text' for writing, 'sheet' for data
- Include ALL content in the createDocument call. Do not create then edit.

**When NOT to use `createDocument`:**
- For answering questions, explanations, or conversational responses
- For short code snippets or examples shown inline
- When the user asks "what is", "how does", "explain", etc.

**Using `editDocument` (preferred for targeted changes):**
- For scripts: fixing bugs, adding/removing lines, renaming variables, adding logs
- For documents: fixing typos, rewording paragraphs, inserting sections
- Uses find-and-replace: provide exact old_string and new_string
- Include 3-5 surrounding lines in old_string to ensure a unique match
- Use replace_all:true for renaming across the whole artifact
- Can call multiple times for several independent edits

**Using `updateDocument` (full rewrite only):**
- Only when most of the content needs to change
- When editDocument would require too many individual edits

**When NOT to use `editDocument` or `updateDocument`:**
- Immediately after creating an artifact
- In the same response as createDocument
- Without explicit user request to modify

**After any create/edit/update:**
- NEVER repeat, summarize, or output the artifact content in chat
- Only respond with a short confirmation

**Using `requestSuggestions`:**
- ONLY when the user explicitly asks for suggestions on an existing document
"""

_TITLE_PROMPT = """Generate a short chat title (2-5 words) summarizing the user's message.

Output ONLY the title text. No prefixes, no formatting.

Examples:
- "what's the weather in nyc" → Weather in NYC
- "help me write an essay about space" → Space Essay Help
- "hi" → New Conversation
- "debug my python code" → Python Debugging

Never output hashtags, prefixes like "Title:", or quotes."""


def _build_system_prompt() -> str:
    return f"{_REGULAR_PROMPT}\n\n{_ARTIFACTS_PROMPT}"


async def generate_title(user_message: str) -> str:
    client = make_client()
    model = get_model()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _TITLE_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=20,
        stream=False,
        extra_body=get_extra_body(),
    )
    return (response.choices[0].message.content or "New Conversation").strip()


async def chat(
    messages: list[UIMessage],
    chat_id: str,
    ctx: StreamContext,
    *,
    is_first_message: bool = False,
) -> None:
    client = make_client()
    model = get_model()

    openai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        *convert_to_openai_messages(messages),
    ]

    if is_first_message:
        user_text = ""
        for msg in messages:
            if msg.role == "user":
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        user_text += part.text
        if user_text:
            title = await generate_title(user_text)
            await ctx.write_data("chat-title", title, transient=True)

    accumulated_content = ""
    accumulated_tool_calls: list[dict[str, str]] = []
    finish_reason: str | None = None
    all_tool_results: list[dict[str, Any]] = []

    while True:
        stream = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            tools=TOOL_DEFINITIONS,
            stream=True,
            extra_body=get_extra_body(),
        )

        result = await consume_openai_stream(stream, ctx)
        accumulated_content = result.content
        accumulated_tool_calls = result.tool_calls
        finish_reason = result.finish_reason

        if finish_reason != "tool_calls" or not accumulated_tool_calls:
            break

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": accumulated_content or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in accumulated_tool_calls
            ],
        }
        openai_messages.append(assistant_msg)

        await ctx.new_step()

        for tc_data in accumulated_tool_calls:
            tool_name = tc_data["name"]
            try:
                parsed_input: dict[str, Any] = json.loads(tc_data["arguments"] or "{}")
            except json.JSONDecodeError:
                parsed_input = {}

            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result_dict: dict[str, Any] = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result_dict = await handler(parsed_input, ctx)
                    await ctx.complete_tool_call(tc_data["id"], result_dict)
                except Exception as exc:
                    result_dict = {"error": str(exc)}
                    await ctx.fail_tool_call(tc_data["id"], str(exc))

            all_tool_results.append(
                {
                    "toolCallId": tc_data["id"],
                    "toolName": tool_name,
                    "args": parsed_input,
                    "result": result_dict,
                }
            )

            openai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_data["id"],
                    "content": json.dumps(result_dict),
                }
            )

        await ctx.new_step()

    # Store the assistant response parts in ctx.store for persistence
    parts: list[dict[str, Any]] = []
    for tr in all_tool_results:
        parts.append(
            {
                "type": f"tool-{tr['toolName']}",
                "toolCallId": tr["toolCallId"],
                "state": "output-available",
                "input": tr["args"],
                "output": tr["result"],
            }
        )
    if accumulated_content:
        parts.append({"type": "text", "text": accumulated_content})
    await ctx.store.set("assistant_parts", parts)

    await ctx.finish(finish_reason=finish_reason or "stop")
