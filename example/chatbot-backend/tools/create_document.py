from __future__ import annotations

import re
import uuid
from typing import Any

from services import redis_store
from services.llm_config import get_extra_body, get_model, make_client

from ai_sdk_stream_python import StreamContext

_CODE_PROMPT = """
You are a code generator that creates self-contained, executable code snippets. When writing code:

1. Each snippet must be complete and runnable on its own
2. Use print/console.log to display outputs
3. Keep snippets concise and focused
4. Prefer standard library over external dependencies
5. Handle potential errors gracefully
6. Return meaningful output that demonstrates functionality
7. Don't use interactive input functions
8. Don't access files or network resources
9. Don't use infinite loops
"""

_SHEET_PROMPT = """
You are a spreadsheet creation assistant. Create a spreadsheet in CSV format based on the given prompt.

Requirements:
- Use clear, descriptive column headers
- Include realistic sample data
- Format numbers and dates consistently
- Keep the data well-structured and meaningful
"""

_TEXT_SYSTEM = (
    "You are a helpful writing assistant. Write clear, well-structured content."
)

_DELTA_KEYS = {
    "code": "codeDelta",
    "text": "textDelta",
    "sheet": "sheetDelta",
}


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```[\w]*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


async def handle_create_document(
    input: dict[str, Any], ctx: StreamContext
) -> dict[str, Any]:
    title: str = input["title"]
    kind: str = input.get("kind", "text")
    doc_id = str(uuid.uuid4())

    await ctx.write_data("kind", kind, transient=True)
    await ctx.write_data("id", doc_id, transient=True)
    await ctx.write_data("title", title, transient=True)
    await ctx.write_data("clear", None, transient=True)

    client = make_client()
    model = get_model()

    system = (
        _CODE_PROMPT
        if kind == "code"
        else (_SHEET_PROMPT if kind == "sheet" else _TEXT_SYSTEM)
    )
    delta_key = _DELTA_KEYS.get(kind, "textDelta")

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": title},
        ],
        stream=True,
        extra_body=get_extra_body(),
    )

    accumulated = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            accumulated += delta
            if kind == "text":
                # Text artifact appends: draftArtifact.content + streamPart.data
                await ctx.write_data(delta_key, delta, transient=True)
            else:
                # Code/sheet artifacts replace: content = streamPart.data
                cleaned = _strip_fences(accumulated) if kind == "code" else accumulated
                await ctx.write_data(delta_key, cleaned, transient=True)

    final_content = _strip_fences(accumulated) if kind == "code" else accumulated
    await ctx.write_data("finish", None, transient=True)

    await redis_store.save_document(doc_id, title, kind, final_content)

    label = (
        "A script was created and is now visible to the user."
        if kind == "code"
        else "A document was created and is now visible to the user."
    )
    return {"id": doc_id, "title": title, "kind": kind, "content": label}
