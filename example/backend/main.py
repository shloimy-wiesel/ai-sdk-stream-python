"""
main.py — Example FastAPI application.

Run with:
    cd example/backend
    pip install fastapi uvicorn
    pip install -e ../../   # install ai-sdk-stream-python from source
    uvicorn main:app --reload --port 8000

The backend exposes a single streaming endpoint:
    POST /chat  →  UIMessageStream (SSE, AI SDK v6 compatible)
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.chat import router as chat_router

app = FastAPI(
    title="ai-sdk-stream-python example",
    description="Demonstrates StreamContext passed across multiple service modules",
    version="0.1.0",
)

# Allow the Next.js dev server to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "docs": "/docs"}
