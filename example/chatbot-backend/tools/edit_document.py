from __future__ import annotations

from typing import Any

from services import redis_store

from ai_sdk_stream_python import StreamContext

_DELTA_KEYS = {
    "code": "codeDelta",
    "text": "textDelta",
    "sheet": "sheetDelta",
}


async def handle_edit_document(
    input: dict[str, Any], ctx: StreamContext
) -> dict[str, Any]:
    doc_id: str = input["id"]
    old_string: str = input["old_string"]
    new_string: str = input["new_string"]
    replace_all: bool = input.get("replace_all", False)

    doc = await redis_store.get_document(doc_id)
    if doc is None:
        return {"error": "Document not found"}

    content: str = doc.get("content") or ""
    if not content:
        return {"error": "Document has no content"}

    if old_string not in content:
        return {"error": "old_string not found in document"}

    updated = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )

    await redis_store.save_document(doc_id, doc["title"], doc["kind"], updated)

    await ctx.write_data("clear", None, transient=True)
    delta_key = _DELTA_KEYS.get(doc["kind"], "textDelta")
    await ctx.write_data(delta_key, updated, transient=True)
    await ctx.write_data("finish", None, transient=True)

    label = (
        "The script has been edited successfully."
        if doc["kind"] == "code"
        else "The document has been edited successfully."
    )
    return {
        "id": doc_id,
        "title": doc["title"],
        "kind": doc["kind"],
        "content": label,
    }
