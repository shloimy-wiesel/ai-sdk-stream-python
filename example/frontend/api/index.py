"""
api/index.py — FastAPI app.

Vercel serves this file as a Python serverless function (at /api/).
In development, run it locally with:

    cd example/frontend
    uv run uvicorn api.index:app --reload --port 8000

next.config.ts rewrites /api/* → this server in dev and → Vercel's
Python runtime in prod, so the Next.js frontend never needs a proxy
route — useChat hits /api/chat on the same origin in both environments.
"""

from dotenv import load_dotenv

load_dotenv(".env.local")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .routes.chat import router as chat_router  # noqa: E402

app = FastAPI(
    title="ai-sdk-stream-python example",
    description="Demonstrates StreamContext passed across multiple service modules",
    version="0.1.0",
)

# Allow the Next.js dev server (port 3000) to call this backend (port 8000)
# during local development. In production both are on the same origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")


@app.get("/api")
def root() -> dict:
    return {"status": "ok", "docs": "/api/docs"}
