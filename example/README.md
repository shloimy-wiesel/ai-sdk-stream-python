# ai-sdk-stream-python · Example App

Full-stack example demonstrating **[ai-sdk-stream-python](../)** — a Python library for streaming [Vercel AI SDK v6](https://sdk.vercel.ai) UIMessageStream SSE events from a FastAPI backend.

The Python backend lives **inside the Next.js project** (`frontend/api/`) so the whole example deploys as a single Vercel project.

```
useChat (Next.js)
      │  POST /api/chat
      │  ┌─────────────────────────────────────────────────────────┐
      │  │  dev: Next.js rewrite → uvicorn :8000                   │
      │  │  prod: Vercel routes → api/index.py (Python function)   │
      │  └─────────────────────────────────────────────────────────┘
      ▼
FastAPI + ai-sdk-stream-python (frontend/api/)
      │  OpenAI-compatible API (any model)
      ▼
LLM (Mistral / OpenAI / etc.)
```

## Features

- **Real LLM** via any OpenAI-compatible endpoint
- **Tool calling** — the LLM can call `search_documents` and the result streams back to the UI
- **Stateless backend** — the full conversation history is sent on every request
- **Streaming UI** — text streams token-by-token using AI SDK v6 `useChat`
- **Single Vercel project** — Python function + Next.js frontend in one repo

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| [uv](https://docs.astral.sh/uv/) | latest |
| Node.js | ≥ 18 |
| npm / pnpm / yarn | any |

---

## Local development

### 1 — Configure environment

```bash
cd example/frontend
cp .env.local.example .env.local
# Edit .env.local with your LLM endpoint, API key, and model ID
```

`.env.local` variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_BASE_URL` | Base URL of the OpenAI-compatible API | `http://10.111.117.4:8070/v1` |
| `LLM_API_KEY` | API key (`-` for unauthenticated) | `-` |
| `LLM_MODEL` | Model ID | `mistralai/Mistral-Small-4-119B-2603-NVFP4` |

### 2 — Install Python dependencies

```bash
cd example/frontend
uv sync   # installs from pyproject.toml, uses local ai-sdk-stream-python source
```

### 3 — Run both servers

Open two terminals:

```bash
# Terminal 1 — Python FastAPI (port 8000)
cd example/frontend
uv run uvicorn api.index:app --reload --port 8000

# Terminal 2 — Next.js dev server (port 3000)
cd example/frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Next.js automatically rewrites `/api/*` → `http://127.0.0.1:8000/api/*` in development, so `useChat` hits the Python server transparently.

---

## Vercel deployment

The project is structured for a single Vercel deployment:

1. Set the **Root Directory** to `example/frontend` in your Vercel project settings.
2. Add the environment variables (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) in the Vercel dashboard.
3. Deploy — Vercel auto-detects:
   - `package.json` → Next.js frontend
   - `requirements.txt` + `api/index.py` → Python serverless function at `/api/`

In production, Next.js rewrites `/api/*` → `/api/` which Vercel routes to the Python function.

---

## E2E Tests (Playwright)

```bash
cd example/frontend
npm run test:e2e          # headless
npm run test:e2e -- --ui  # interactive UI mode
```

---

## Project Structure

```
example/
└── frontend/                        # Single Vercel project root
    ├── api/                         # Python serverless function (Vercel) / uvicorn app (dev)
    │   ├── index.py                 # FastAPI app entry point
    │   ├── routes/
    │   │   └── chat.py              # POST /api/chat — streaming endpoint
    │   └── services/
    │       ├── llm_service.py       # Real LLM via OpenAI SDK + tool calling
    │       └── db_service.py        # Document search tool (simulated DB)
    ├── app/
    │   ├── page.tsx                 # Chat UI (useChat + ai-elements)
    │   └── layout.tsx               # Root layout
    ├── tests/
    │   ├── chat.spec.ts             # Playwright E2E tests
    │   └── helpers/sse-stream.ts
    ├── next.config.ts               # Rewrites /api/* → Python server
    ├── pyproject.toml               # Python deps for local dev (uv, editable source)
    ├── requirements.txt             # Python deps for Vercel deployment (PyPI)
    └── .env.local.example           # Environment variable template
```
