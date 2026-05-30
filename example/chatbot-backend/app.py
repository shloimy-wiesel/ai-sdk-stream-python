"""
app.py — FastAPI application entry point for the chatbot backend.

Loads environment variables, configures CORS, mounts the chat router,
and exposes a health-check endpoint.

Run with:
    uv run uvicorn app:app --reload --port 8001
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from routes.chat import router as chat_router  # noqa: E402
from routes.document import router as document_router  # noqa: E402

app = FastAPI(title="Chatbot Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(document_router, prefix="/api")


@app.get("/api")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
