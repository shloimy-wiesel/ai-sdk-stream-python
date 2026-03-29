from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from ai_sdk_stream_python import StreamContext

from .create_document import handle_create_document
from .edit_document import handle_edit_document
from .get_weather import handle_get_weather
from .request_suggestions import handle_request_suggestions
from .update_document import handle_update_document

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "getWeather",
            "description": "Get the current weather at a location. You can provide either coordinates or a city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate",
                    },
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'San Francisco', 'New York', 'London')",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "createDocument",
            "description": "Create an artifact. You MUST specify kind: use 'code' for any programming/algorithm request (creates a script), 'text' for essays/writing (creates a document), 'sheet' for spreadsheets/data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the artifact",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["code", "text", "sheet"],
                        "description": "REQUIRED. 'code' for programming/algorithms, 'text' for essays/writing, 'sheet' for spreadsheets",
                    },
                },
                "required": ["title", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "editDocument",
            "description": "Make a targeted edit to an existing artifact by finding and replacing an exact string. Preferred over updateDocument for small changes. The old_string must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The ID of the artifact to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find. Include 3-5 surrounding lines for uniqueness.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of just the first (default false)",
                    },
                },
                "required": ["id", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateDocument",
            "description": "Full rewrite of an existing artifact. Only use for major changes where most content needs replacing. Prefer editDocument for targeted changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The ID of the artifact to rewrite",
                    },
                    "description": {
                        "type": "string",
                        "description": "The description of changes that need to be made",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "requestSuggestions",
            "description": "Request writing suggestions for an existing document artifact. Only use this when the user explicitly asks to improve or get suggestions for a document they have already created. Never use for general questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documentId": {
                        "type": "string",
                        "description": "The UUID of an existing document artifact that was previously created with createDocument",
                    },
                },
                "required": ["documentId"],
            },
        },
    },
]

ToolHandler = Callable[
    [dict[str, Any], StreamContext], Coroutine[Any, Any, dict[str, Any]]
]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    "getWeather": handle_get_weather,
    "createDocument": handle_create_document,
    "editDocument": handle_edit_document,
    "updateDocument": handle_update_document,
    "requestSuggestions": handle_request_suggestions,
}

__all__ = ["TOOL_DEFINITIONS", "TOOL_HANDLERS"]
