# ai-sdk-stream-python

[![PyPI version](https://img.shields.io/pypi/v/ai-sdk-stream-python)](https://pypi.org/project/ai-sdk-stream-python/)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/ai-sdk-stream-python/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A Python library for building [Vercel AI SDK v6](https://sdk.vercel.ai/) **UIMessageStream**-compatible streaming backends.

## Installation

```bash
pip install ai-sdk-stream-python
```

**Inspired by [llama-index-workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)** — the same idea of a `Context` object that holds shared state and can write events to a stream, applied to the Vercel AI SDK wire protocol.

---

## Concept

Normally you have to manually yield raw SSE strings and track protocol ordering yourself:

```python
# Before — error-prone, no type safety, no shared state
yield "data: " + json.dumps({"type": "start", "messageId": id}) + "\n\n"
yield "data: " + json.dumps({"type": "start-step"}) + "\n\n"
yield "data: " + json.dumps({"type": "text-start", "id": part_id}) + "\n\n"
for chunk in llm.stream():
    yield "data: " + json.dumps({"type": "text-delta", "id": part_id, "delta": chunk}) + "\n\n"
# ... remember to close every part and step ...
yield "data: [DONE]\n\n"
```

With `StreamContext`:

```python
# After — typed, lifecycle-safe, state-sharing
ctx = StreamContext()
asyncio.create_task(my_work(ctx))
return StreamingResponse(ctx.stream(), headers=ctx.response_headers)

async def my_work(ctx):
    try:
        await ctx.write_text("Hello world!")   # auto-emits start/start-step/text-start
    finally:
        await ctx.finish()                      # auto-closes everything, emits [DONE]
```

### Key features

| Feature | Detail |
|---|---|
| **Typed events** | All 16 v6 protocol events as Pydantic models |
| **Lifecycle auto-management** | `start`, `start-step`, `text-start` etc. are emitted automatically |
| **Shared state** | `ctx.store.get/set()` — dot-path key-value store shared across modules |
| **Pass as parameter** | `ctx` flows through your services like a logger or DB session |
| **Low-level escape hatch** | `ctx.write(event)` / `ctx.write_event_to_stream(ev)` for raw control |
| **Abort support** | `ctx.abort()` terminates the stream safely on errors |

---

## Installation

```bash
uv add ai-sdk-stream-python
# or
pip install ai-sdk-stream-python
```

---

## Quick start (FastAPI)

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ai_sdk_stream_python import StreamContext

app = FastAPI()

@app.post("/chat")
async def chat():
    ctx = StreamContext()

    async def _work():
        try:
            await ctx.write_text("Hello ")
            await ctx.write_text("world!")
        finally:
            await ctx.finish()

    asyncio.create_task(_work())
    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
```

---

## `StreamContext` API

### State store

```python
await ctx.store.set("user.name", "Alice")        # dot-path write
name = await ctx.store.get("user.name")          # → "Alice"
plan = await ctx.store.get("user.plan", default="free")  # with default
```

The store uses an `asyncio.Lock` — safe to use across concurrent coroutines.

### Writing to the stream

```python
# Reasoning (chain-of-thought)
await ctx.write_reasoning("Let me think…")

# Text answer
await ctx.write_text("Here is the answer…")

# Tool calls
handle = await ctx.begin_tool_call("searchDocs", {"query": "hello"})
result = await my_search(query)
await ctx.complete_tool_call(handle.toolCallId, result)
# or on error:
await ctx.fail_tool_call(handle.toolCallId, "timeout")

# Source citations
await ctx.write_source("doc-1", "https://example.com/doc", "My Doc")

# Finish (closes all open parts/steps, emits finish + [DONE])
await ctx.finish(finish_reason="stop")
```

### `new_step()` — when to use it

`new_step()` closes any open text/reasoning part and the current step (`finish-step`), then immediately opens a new step (`start-step`). Use it when you want an explicit step boundary in the stream — for example in a multi-turn agentic flow:

```python
# Step 1: reasoning
await ctx.write_reasoning("Let me think…")

# Step 2: tool call
await ctx.new_step()
handle = await ctx.begin_tool_call("search", {"q": query})
await ctx.complete_tool_call(handle.toolCallId, results)

# Step 3: final answer
await ctx.new_step()
await ctx.write_text("Based on the results…")
```

**You don't need `new_step()` for simple responses.** The high-level helpers (`write_text`, `write_reasoning`, `begin_tool_call`) already auto-close each other within the same step, and `finish()` closes everything. Only call `new_step()` when you want the frontend to see distinct steps — e.g. separate reasoning, tool-use, and answer phases.

### Low-level / raw events

```python
from ai_sdk_stream_python import TextDeltaEvent

# Sync push (like LlamaIndex's write_event_to_stream)
ctx.write_event_to_stream(TextDeltaEvent(id=ctx.current_text_id, delta="!"))

# Async push (auto-ensures 'start' was emitted first)
await ctx.write(TextDeltaEvent(id="my-id", delta="raw"))
```

### Abort

```python
await ctx.abort()  # terminates stream immediately without finish event
```

### Properties

```python
ctx.message_id           # the message ID in the start event
ctx.current_text_id      # ID of open text part, or None
ctx.current_reasoning_id # ID of open reasoning part, or None
ctx.is_finished          # True after finish()/abort()
ctx.response_headers     # dict with x-vercel-ai-ui-message-stream: v1
```

---

## Passing `ctx` across modules

The key pattern — `ctx` flows as a parameter to any module that needs to write events:

```python
# routes/chat.py
@app.post("/chat")
async def chat(req: ChatRequest):
    ctx = StreamContext()

    async def _work():
        try:
            # db_service writes reasoning + stores user data in ctx.store
            user = await db_service.load_user(req.user_id, ctx=ctx)

            # search_service emits tool call events via ctx
            await ctx.new_step()
            docs = await search_service.search(req.query, ctx=ctx)

            # llm_service reads ctx.store + streams text via ctx
            await ctx.new_step()
            await llm_service.generate(req.query, docs, ctx=ctx)
        finally:
            await ctx.finish()

    asyncio.create_task(_work())
    return StreamingResponse(ctx.stream(), headers=ctx.response_headers)

# services/db_service.py
async def load_user(user_id: str, *, ctx: StreamContext) -> dict:
    await ctx.write_reasoning(f"Loading user {user_id}…")
    user = await db.get(user_id)
    await ctx.store.set("user.name", user["name"])  # share with downstream
    return user

# services/llm_service.py
async def generate(query: str, docs: list, *, ctx: StreamContext) -> None:
    name = await ctx.store.get("user.name", default="there")  # read from store
    async for chunk in llm.stream(query, docs):
        await ctx.write_text(chunk)
```

See `example/` for a full runnable example.

---

## Wire protocol

The library targets the **Vercel AI SDK v6 UIMessageStream** protocol — SSE events with typed JSON payloads:

```
data: {"type":"start","messageId":"..."}

data: {"type":"start-step"}

data: {"type":"reasoning-start","id":"..."}
data: {"type":"reasoning-delta","id":"...","delta":"thinking…"}
data: {"type":"reasoning-end","id":"..."}

data: {"type":"finish-step"}

data: {"type":"start-step"}

data: {"type":"tool-input-start","toolCallId":"...","toolName":"search"}
data: {"type":"tool-input-available","toolCallId":"...","input":{...}}
data: {"type":"tool-output-available","toolCallId":"...","output":{...}}

data: {"type":"finish-step"}

data: {"type":"start-step"}

data: {"type":"text-start","id":"..."}
data: {"type":"text-delta","id":"...","delta":"Hello "}
data: {"type":"text-end","id":"..."}

data: {"type":"source-url","sourceId":"s1","url":"https://...","title":"Doc"}

data: {"type":"finish-step"}

data: {"type":"finish","finishReason":"stop"}

data: [DONE]
```

Response header: `x-vercel-ai-ui-message-stream: v1`

---

## Example app

`example/` contains a complete runnable demo:

```
example/
├── backend/          # FastAPI app — shows ctx passed across 3 service modules
│   ├── main.py
│   ├── routes/chat.py
│   └── services/
│       ├── db_service.py     # writes reasoning + stores data in ctx.store
│       ├── llm_service.py    # reads ctx.store + streams text
│       └── search_service.py
└── frontend/         # Next.js + AI SDK v6 + ai-elements chat UI
    ├── app/
    │   ├── page.tsx           # Conversation/Message/PromptInput from ai-elements
    │   └── api/chat/route.ts  # proxies useChat → Python backend
    └── package.json
```

### Run the backend

```bash
cd example/backend
uv pip install fastapi uvicorn
uv pip install -e ../../   # install ai-sdk-stream-python from source
uvicorn main:app --reload --port 8000
```

### Run the frontend

```bash
cd example/frontend
npm install
# Install ai-elements components (shadcn/ui registry):
npx ai-elements@latest add conversation message prompt-input
npm run dev
# Open http://localhost:3000
```

---

## Tests

```bash
uv run pytest tests/ -v
```

25 tests covering: basic lifecycle, reasoning ↔ text transitions, tool calls,
multi-step flows, source events, edge cases (double finish, abort, write after finish),
and StateStore integration.

---

## All event types

| Class | `type` field | Description |
|---|---|---|
| `StartEvent` | `start` | Message begins |
| `StartStepEvent` | `start-step` | Step begins |
| `ReasoningStartEvent` | `reasoning-start` | Reasoning part opens |
| `ReasoningDeltaEvent` | `reasoning-delta` | Reasoning chunk |
| `ReasoningEndEvent` | `reasoning-end` | Reasoning part closes |
| `TextStartEvent` | `text-start` | Text part opens |
| `TextDeltaEvent` | `text-delta` | Text chunk |
| `TextEndEvent` | `text-end` | Text part closes |
| `ToolInputStartEvent` | `tool-input-start` | Tool call begins |
| `ToolInputDeltaEvent` | `tool-input-delta` | Streaming tool input |
| `ToolInputAvailableEvent` | `tool-input-available` | Full tool input ready |
| `ToolOutputAvailableEvent` | `tool-output-available` | Tool result |
| `ToolOutputErrorEvent` | `tool-output-error` | Tool failure |
| `SourceUrlEvent` | `source-url` | Citation / source |
| `FinishStepEvent` | `finish-step` | Step closes |
| `FinishEvent` | `finish` | Message ends |
