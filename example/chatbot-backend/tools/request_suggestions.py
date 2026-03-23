from __future__ import annotations

import json
import uuid
from typing import Any

from services import redis_store
from services.llm_config import get_extra_body, get_model, make_client

from ai_sdk_stream_python import StreamContext

_SUGGESTIONS_SYSTEM = (
    "You are a writing assistant. Given a piece of writing, offer up to 5 suggestions to improve it. "
    "Each suggestion must contain full sentences, not just individual words. "
    "Describe what changed and why. "
    "Return a JSON array of objects with keys: originalSentence, suggestedSentence, description."
)


async def handle_request_suggestions(
    input: dict[str, Any], ctx: StreamContext
) -> dict[str, Any]:
    document_id: str = input["documentId"]

    doc = await redis_store.get_document(document_id)
    if doc is None or not doc.get("content"):
        return {"error": "Document not found"}

    client = make_client()
    model = get_model()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUGGESTIONS_SYSTEM},
            {"role": "user", "content": doc["content"]},
        ],
        response_format={"type": "json_object"},
        stream=False,
        extra_body=get_extra_body(),
    )

    raw = response.choices[0].message.content or "[]"
    try:
        parsed = json.loads(raw)
        items: list[dict[str, Any]] = (
            parsed if isinstance(parsed, list) else parsed.get("suggestions", [])
        )
    except (json.JSONDecodeError, AttributeError):
        items = []

    for item in items:
        original = item.get("originalSentence") or item.get("original_sentence") or ""
        suggested = (
            item.get("suggestedSentence") or item.get("suggested_sentence") or ""
        )
        description = item.get("description") or ""
        if not original or not suggested:
            continue

        suggestion: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "documentId": document_id,
            "originalText": original,
            "suggestedText": suggested,
            "description": description,
            "isResolved": False,
        }
        await ctx.write_data("suggestion", suggestion, transient=True)

    return {
        "id": document_id,
        "title": doc.get("title", ""),
        "kind": doc.get("kind", "text"),
        "message": "Suggestions have been added to the document",
    }
