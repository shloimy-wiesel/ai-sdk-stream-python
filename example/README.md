# ai-sdk-stream-python · Example App

Full-stack example demonstrating **[ai-sdk-stream-python](../)** — a Python library for streaming [Vercel AI SDK v6](https://sdk.vercel.ai) UIMessageStream SSE events from a FastAPI backend.

```
frontend (Next.js + AI SDK v6)
      │  POST /api/chat  { messages: [{role, content}] }
      ▼
backend (FastAPI + ai-sdk-stream-python)
      │  OpenAI-compatible API (any model)
      ▼
LLM (Mistral / OpenAI / etc.)
```

## Features

- **Real LLM** via any OpenAI-compatible endpoint
- **Tool calling** — the LLM can call `search_documents` and the result streams back to the UI
- **Stateless backend** — the full conversation history is sent on every request
- **Streaming UI** — text streams token-by-token using AI SDK v6 `useChat`
- **Dark mode** by default

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| [uv](https://docs.astral.sh/uv/) | latest |
| Node.js | ≥ 18 |
| pnpm / npm / yarn | any |

---

## Backend

### 1 — Configure environment

```bash
cd example/backend
cp .env.example .env
# Edit .env with your LLM endpoint, API key, and model ID
```

`.env` variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_BASE_URL` | Base URL of the OpenAI-compatible API | `http://10.111.117.4:8070/v1` |
| `LLM_API_KEY` | API key (`-` for unauthenticated) | `-` |
| `LLM_MODEL` | Model ID | `mistralai/Mistral-Small-4-119B-2603-NVFP4` |

### 2 — Install dependencies & run

```bash
# From the example/ directory
uv sync
uv run uvicorn backend.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.
Swagger docs: `http://localhost:8000/docs`

---

## Frontend

### 1 — Install dependencies

```bash
cd example/frontend
npm install        # or pnpm install / yarn
```

### 2 — Run the dev server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

> The frontend proxies `/api/chat` → `http://localhost:8000/chat`.
> Override the backend URL via the `BACKEND_URL` environment variable:
> ```bash
> BACKEND_URL=http://my-backend:8000 npm run dev
> ```

---

## Running both together

Open two terminals:

```bash
# Terminal 1 — backend
cd example
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd example/frontend
npm run dev
```

---

## E2E Tests (Playwright)

The frontend has a Playwright test suite that mocks the SSE stream and verifies the UI without a real backend.

```bash
cd example/frontend
npm run test:e2e          # headless
npm run test:e2e -- --ui  # interactive UI mode
```

---

## Project Structure

```
example/
├── backend/
│   ├── main.py                  # FastAPI app + CORS
│   ├── routes/
│   │   └── chat.py              # POST /chat — streaming endpoint
│   ├── services/
│   │   ├── llm_service.py       # Real LLM via OpenAI SDK + tool calling
│   │   └── db_service.py        # Document search tool (simulated DB)
│   ├── .env.example             # Environment variable template
│   └── pyproject.toml
└── frontend/
    ├── app/
    │   ├── page.tsx             # Chat UI (useChat + ai-elements)
    │   ├── layout.tsx           # Root layout
    │   └── api/chat/route.ts    # Proxy → backend
    ├── tests/
    │   ├── chat.spec.ts         # Playwright E2E tests
    │   └── helpers/sse-stream.ts
    └── package.json
```
