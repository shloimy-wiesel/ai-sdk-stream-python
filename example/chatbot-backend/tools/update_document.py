from __future__ import annotations

import re
from typing import Any

from services import redis_store
from services.llm_config import get_extra_body, get_model, make_client

from ai_sdk_stream_python import StreamContext

_DELTA_KEYS = {
    "code": "codeDelta",
    "text": "textDelta",
    "sheet": "sheetDelta",
}

_MEDIA_TYPES = {
    "code": "script",
    "sheet": "spreadsheet",
}


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```[\w]*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def _update_prompt(current_content: str | None, kind: str, description: str) -> str:
    media_type = _MEDIA_TYPES.get(kind, "document")
    base = f"Rewrite the following {media_type} based on the given description: {description}\n\n"
    if current_content:
        base += current_content
    return base


async def handle_update_document(
    input: dict[str, Any], ctx: StreamContext
) -> dict[str, Any]:
    doc_id: str = input["id"]
    description: str = input.get("description", "Improve the content")

    doc = await redis_store.get_document(doc_id)
    if doc is None:
        return {"error": "Document not found"}

    await ctx.write_data("clear", None, transient=True)

    client = make_client()
    model = get_model()
    kind = doc["kind"]
    delta_key = _DELTA_KEYS.get(kind, "textDelta")
    user_prompt = _update_prompt(doc.get("content"), kind, description)

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
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

    await redis_store.save_document(doc_id, doc["title"], kind, final_content)

    label = (
        "The script has been updated successfully."
        if kind == "code"
        else "The document has been updated successfully."
    )
    return {
        "id": doc_id,
        "title": doc["title"],
        "kind": kind,
        "content": label,
    }
