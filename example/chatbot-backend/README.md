# Vercel Chatbot Python Backend

A production-ready Python (FastAPI) backend that implements the Vercel AI SDK v6 UIMessageStream protocol, designed to replace the TypeScript backend of the [Vercel Chatbot](https://github.com/vercel/chatbot).

**Purpose**: Demonstrate that building AI SDK-compatible backends in Python is easy when you have [`ai-sdk-stream-python`](../../) — the same library powering this example.

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| **StreamContext** | ✅ | All responses use `ctx.run()` + `ctx.stream()` |
| **Text artifacts** | ✅ | Essays, docs, reports → `createDocument(kind="text")` |
| **Code artifacts** | ✅ | Scripts with syntax highlighting → `createDocument(kind="code")` |
| **Weather tool** | ✅ | `getWeather` tool renders full widget |
| **Update document** | ✅ | `updateDocument` replaces entire content |
| **Edit document** | ✅ | `editDocument` search-and-replace (though LLM prefers create) |
| **Request suggestions** | ✅ | `requestSuggestions` for text documents |
| **Session persistence** | ✅ | Redis stores chat history + documents |
| **Streaming tool calls** | ✅ | OpenAI-compatible streaming + context propagation |

## Architecture

```
Next.js (useChat)
      │  POST /api/chat
      ▼
FastAPI + ai-sdk-stream-python
      │  LLM (Qwen via vLLM)
      ▼
Redis (chat history, documents)
```

### Key patterns

1. **StreamContext everywhere** — No raw SSE strings anywhere
2. **ctx flows as parameter** — Passed from routes → services → tools
3. **Redis for session state** — Each chat has `chat:{id}:messages` + `doc:{id}` keys
4. **vLLM extra_body** — Model-specific config via `extra_body=get_extra_body()`

## Project Structure

```
chatbot-backend/
├── app.py                  # FastAPI entry point, CORS, router mounting
├── routes/
│   ├── chat.py             # POST /api/chat — streaming endpoint
│   └── document.py         # GET /api/document?id=... — fetch document
├── services/
│   ├── redis_store.py      # Async Redis client for docs & messages
│   ├── llm_config.py       # Shared: make_client(), get_model(), get_extra_body()
│   └── llm_service.py      # Chat/LLM streaming with tool loop
├── tools/
│   ├── __init__.py         # TOOL_DEFINITIONS, TOOL_HANDLERS
│   ├── get_weather.py      # OpenWeatherMap integration
│   ├── create_document.py  # Text/code/sheet creation
│   ├── update_document.py  # Full content replacement
│   ├── edit_document.py    # Search-and-replace edits
│   └── request_suggestions.py  # Text document suggestions
├── .env                    # LLM endpoint, Redis URL, extra_body
└── pyproject.toml          # Dependencies (fastapi, uvicorn, redis)
```

## Installation

```bash
# 1. Install Python deps
uv pip install fastapi uvicorn redis

# 2. Install ai-sdk-stream-python from source
pip install -e ../../  # Adjust path to library root

# 3. Configure environment
cp .env.example .env
# Edit .env with your vLLM endpoint + Redis URL
```

## Running

### Option 1: Local dev (uvicorn)

```bash
# Terminal 1 — Python backend
cd chatbot-backend
uvicorn app:app --reload --port 8001

# Terminal 2 — Next.js frontend (in chatbot/)
cd ..
PORT=3002 pnpm dev
# Edit next.config.ts: rewrites `/api/chat` → http://localhost:8001
```

### Option 2: Docker (optional)

```bash
# Python backend container
docker run -p 8001:8000 -e REDIS_URL=redis://host.docker.internal:6312 \
  -v $(pwd):/app -w /app image name
```

## Environment Variables

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `LLM_BASE_URL` | ✅ | `http://10.111.117.4:8070/v1` | vLLM endpoint |
| `LLM_API_KEY` | ⚠️ | `-` | API key (use `-` for unauthenticated) |
| `LLM_MODEL` | ✅ | `Qwen/Qwen3.5-35B-A3B-FP8` | Model ID |
| `LLM_EXTRA_BODY` | ⚠️ | `{"chat_template_kwargs": {"enable_thinking": false}}` | Model-specific config |
| `REDIS_URL` | ✅ | `redis://localhost:6312` | Redis connection string |

**CRITICAL**: The Qwen/Qwen3.5-35B-A3B-FP8 model at `10.111.117.4:8070` **requires** `enable_thinking: false` or all output goes to the `reasoning` field with `content: null`.

## API Reference

### `/api/chat` (POST)

**Request body**: Vercel AI SDK `ChatRequest` format
```json
{
  "id": "chat-uuid",
  "message": { "id": "msg-uuid", "role": "user", "parts": [{ "type": "text", "text": "hello" }] },
  "selectedChatModel": "gpt-4o",
  "selectedVisibilityType": "private"
}
```

**Response**: `text/event-stream` with UIMessageStream events

### `/api/document` (GET)

**Query params**: `id` (document UUID)

**Response**: JSON document object
```json
{
  "id": "doc-uuid",
  "title": "My Document",
  "kind": "text",
  "content": "Full document content..."
}
```

## Tool Definitions

The backend implements 5 tools matching the Vercel Chatbot frontend:

| Tool | Parameters | Description |
|------|------------|-------------|
| `getWeather` | `city: str` | OpenWeatherMap API integration |
| `createDocument` | `title: str, kind: str, content: str` | Creates text/code/sheet artifact |
| `updateDocument` | `id: str, title: str, kind: str, content: str` | Replaces entire document |
| `editDocument` | `id: str, old_string: str, new_string: str, replace_all: bool` | Search-and-replace |
| `requestSuggestions` | `id: str` | Requests writing suggestions for text docs |

## Known Gaps

### Library Limitation: Delta Strategy Per Artifact Kind

The frontend expects:
- `text` artifacts: **incremental deltas** (content appended)
- `code`/`sheet` artifacts: **full content** (replaced)

Our implementation handles this correctly in `create_document.py` and `update_document.py`:

```python
if kind == "text":
    await ctx.write_data(delta_key, delta, transient=True)
else:
    cleaned = _strip_fences(accumulated) if kind == "code" else accumulated
    await ctx.write_data(delta_key, cleaned, transient=True)
```

### LLM Tool Selection Behavior

The LLM may prefer `createDocument` over `editDocument` even for simple edits. This is expected LLM behavior — `editDocument` requires precise `old_string`/`new_string` values that are harder to generate than a full rewrite.

**Impact**: Minor — users get the desired result either way.

## Testing

All integration tests passed:

- ✅ Basic Q&A → assistant text response
- ✅ Weather tool → full widget rendered
- ✅ Code artifact → Dijkstra's algorithm with syntax highlighting
- ✅ Text artifact → Silicon Valley essay with headings
- ✅ Update document → full rewrite with A* algorithm
- ✅ Request suggestions → suggestions tool invoked successfully
- ⚠️ Edit document → LLM prefers `createDocument`; `editDocument` exists but rarely used

## Debugging

### Check Redis keys

```bash
docker exec redis-new-backend redis-cli KEYS "chat:*"
docker exec redis-new-backend redis-cli KEYS "doc:*"
```

### View backend logs

```bash
tmux capture-pane -t chatbot-be -p | tail -50
```

### Verify SSE stream

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"id":"test","message":{"id":"1","role":"user","parts":[{"type":"text","text":"hi"}]},"selectedChatModel":"gpt-4o","selectedVisibilityType":"private"}' \
  --no-buffer | head -20
```

## License

MIT — same as `ai-sdk-stream-python`
