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

With `StreamContext` and `ctx.run()`:

```python
# After — typed, lifecycle-safe, state-sharing
ctx = StreamContext()
await ctx.run(lambda ctx: my_work(ctx))
return StreamingResponse(ctx.stream(), media_type="text/event-stream", headers=ctx.response_headers)

async def my_work(ctx):
    await ctx.write_text("Hello world!")   # auto-emits start/start-step/text-start
    # ctx.run() auto-calls finish() and handles errors — no try/finally needed
```

### Key features

| Feature | Detail |
|---|---|
| **Typed events** | All 16 v6 protocol events as Pydantic models |
| **Lifecycle auto-management** | `start`, `start-step`, `text-start` etc. are emitted automatically |
| **Shared state** | `ctx.store.get/set()` — dot-path key-value store shared across modules |
| **Custom information** | `ctx.info` — typed, read-only Pydantic model for request-scoped metadata |
| **Pass as parameter** | `ctx` flows through your services like a logger or DB session |
| **Stream collection** | `collect=True` records all emitted content into `ctx.record` for DB persistence |
| **Low-level escape hatch** | `ctx.write(event)` / `ctx.write_event_to_stream(ev)` for raw control |
| **Abort support** | `ctx.abort()` terminates the stream safely on errors |

### How is this different from Vercel's `ai-python`?

[`vercel-labs/ai-python`](https://github.com/vercel-labs/ai-python) ([ai-python.dev](https://ai-python.dev)) is Vercel's own Python AI SDK, currently in public beta. It is an **agent framework** (models, tools, agent loop) that can output the UIMessageStream protocol from its own agent events.

`ai-sdk-stream-python` is a **framework-agnostic protocol writer**. Use whichever LLM library you like (raw OpenAI, the OpenAI Agents SDK, LlamaIndex, your own code) and push events from any service layer with `ctx.write_*()`.

|  | `ai-sdk-stream-python` | `ai-python` |
|---|---|---|
| Focus | Emit UIMessageStream from any backend | Build agents; UI stream is an output adapter |
| Input | Imperative `ctx.write_*()` calls | Its own `AgentEvent` stream |
| Shared state / typed `ctx.info` | Yes | No |
| Persistence (`collect`, `on_finish`) | Yes | No |
| OpenAI Agents SDK adapter | Yes | No |
| Python | 3.10+ | 3.12+ |

If you are starting a new project and want an all-in-one agent framework, `ai-python` is worth a look. If you already have an LLM stack and just need a correct stream for `useChat`, this library is the lighter fit.

---

## Installation

```bash
uv add ai-sdk-stream-python
# or
pip install ai-sdk-stream-python

# with the OpenAI Agents SDK adapter
pip install "ai-sdk-stream-python[agents]"
```

---

## Quick start (FastAPI)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ai_sdk_stream_python import StreamContext

app = FastAPI()

@app.post("/chat")
async def chat():
    ctx = StreamContext()

    async def _work(ctx: StreamContext) -> None:
        await ctx.write_text("Hello ")
        await ctx.write_text("world!")

    await ctx.run(_work)
    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
```

`ctx.run()` wraps your work function with automatic error handling and stream finalization — no `try/finally` or manual `asyncio.create_task()` needed. See [`ctx.run()`](#ctxrun--safe-task-runner) below.

---

## `StreamContext` API

### State store

```python
await ctx.store.set("user.name", "Alice")        # dot-path write
name = await ctx.store.get("user.name")          # → "Alice"
plan = await ctx.store.get("user.plan", default="free")  # with default
```

The store uses an `asyncio.Lock` — safe to use across concurrent coroutines.

### Custom information (`ctx.info`)

Pass a Pydantic model at construction time to carry static, read-only request-scoped data (e.g. `user_id`, `rate_limit`, `tenant_id`) through every service layer without threading extra arguments:

```python
from pydantic import BaseModel
from ai_sdk_stream_python import StreamContext

class RequestInfo(BaseModel):
    user_id: str
    rate_limit: int

# Typed constructor — IDE infers ctx.info as RequestInfo
ctx: StreamContext[RequestInfo] = StreamContext(
    custom_information=RequestInfo(user_id="u_42", rate_limit=100)
)

# In any service layer that receives ctx:
if ctx.info is not None:
    print(ctx.info.user_id)    # "u_42"
    print(ctx.info.rate_limit) # 100
```

- `ctx.info` is **read-only** (no setter). For mutable runtime state use `ctx.store`.
- Defaults to `None` when `custom_information` is not passed.
- `StreamContext` is generic — annotate as `StreamContext[YourModel]` for full IDE support.

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

### Stream collection

Pass `collect=True` to record all emitted content into a `StreamRecord` for database persistence or audit logging:

```python
ctx = StreamContext(collect=True)

async def _work(ctx: StreamContext) -> None:
    await ctx.write_reasoning("Let me think…")
    handle = await ctx.begin_tool_call("search", {"q": "hello"})
    await ctx.complete_tool_call(handle.toolCallId, {"results": [...]})
    await ctx.write_text("Here is the answer.")
    await ctx.write_source("s1", "https://example.com", "My Doc")

await ctx.run(_work)
response = StreamingResponse(
    ctx.stream(), media_type="text/event-stream", headers=ctx.response_headers
)

# After finish(), ctx.record is fully populated:
record = ctx.record
# record.text        → "Here is the answer."
# record.reasoning   → "Let me think…"
# record.tool_calls  → [ToolCallRecord(tool_name="search", input={...}, output={...})]
# record.sources     → [SourceRecord(source_id="s1", url="https://example.com", title="My Doc")]
# record.finish_reason → "stop"
# record.step_count  → 1

# Serialize to a plain dict for DB storage:
await db.insert(record.to_dict())
```

`ctx.record` is `None` when `collect=False` (the default). The record is built incrementally as events are emitted, and is fully available after `finish()` completes.

### `ctx.run()` — safe task runner

`ctx.run()` is the **recommended way** to launch your streaming work function. It provides three safety guarantees that eliminate an entire class of bugs:

| Guarantee | What it does |
|---|---|
| **Auto-finish** | Calls `ctx.finish()` in a `finally` block — the stream is always closed, even if your function returns early |
| **Auto-error** | Catches unhandled exceptions and emits `ctx.error()` — the frontend gets a proper error event instead of a silent hang |
| **Task GC prevention** | Stores the background task on the context — Python's garbage collector cannot silently discard it |

```python
@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    ctx = StreamContext()
    await ctx.run(lambda ctx: my_service.chat(request, ctx=ctx))
    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )
```

Without `ctx.run()`, you'd need to manually manage `try/finally`, `asyncio.create_task()`, and a task reference set — all of which are easy to get wrong and produce hard-to-debug hung streams.

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
ctx.info                 # custom_information model, or None
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

    async def _work(ctx: StreamContext) -> None:
        # db_service writes reasoning + stores user data in ctx.store
        user = await db_service.load_user(req.user_id, ctx=ctx)

        # search_service emits tool call events via ctx
        await ctx.new_step()
        docs = await search_service.search(req.query, ctx=ctx)

        # llm_service reads ctx.store + streams text via ctx
        await ctx.new_step()
        await llm_service.generate(req.query, docs, ctx=ctx)

    await ctx.run(_work)  # auto-handles finish, errors, and task GC
    return StreamingResponse(
        ctx.stream(),
        media_type="text/event-stream",
        headers=ctx.response_headers,
    )

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

## OpenAI Agents SDK

`contrib.openai_agents` streams an [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
run (`Runner.run_streamed`) through `StreamContext`. It covers text, reasoning,
parallel tool calls (matched by `call_id`), handoffs, nested agents-as-tools and usage.

```python
from agents import Agent, Runner
from ai_sdk_stream_python import StreamContext
from ai_sdk_stream_python.contrib.openai_agents import (
    agent_tool_stream_handler,
    consume_agent_stream,
)


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    ctx = StreamContext()
    research = researcher.as_tool(
        tool_name="research",
        tool_description="Research a topic",
        on_stream=agent_tool_stream_handler(ctx),  # nested run → reasoning notes
    )
    agent = Agent(name="Assistant", tools=[research])

    async def work(ctx: StreamContext) -> None:
        result = Runner.run_streamed(agent, user_input)
        await consume_agent_stream(result, ctx)

    await ctx.run(work)  # owns start / finish / error
    return StreamingResponse(
        ctx.stream(), media_type="text/event-stream", headers=ctx.response_headers
    )
```

| Agents SDK event | Written as |
|---|---|
| `response.output_text.delta` | `write_text` |
| `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` | `write_reasoning` |
| `tool_called` / `tool_output` | `begin_tool_call` / `complete_tool_call` or `fail_tool_call` |
| `handoff_occured` | reasoning note "Handing off to X" (`handoff_notes=False` to disable) |
| `agent_updated_stream_event` | `new_step()` |
| `mcp_list_tools`, `mcp_approval_*` | ignored unless `on_mcp_event=` is passed |
| end of stream | `set_usage` from `result.context_wrapper.usage` |

Notes:

- **Tool errors.** The SDK reports a raising tool as an ordinary string output
  (from the tool's `failure_error_function`). By default, exception objects and the
  SDK's default failure message become `tool-output-error`. If you use a custom
  `failure_error_function`, pass `detect_tool_error=lambda item: ...`.
- **Nested agents.** `agent_tool_stream_handler(ctx)` keeps sub-agent output out of
  the answer text. It writes whole-item reasoning notes (`[Researcher] Calling search`)
  so parallel sub-agents can't interleave inside one part. Pass
  `nested_tool_calls=True` to render inner tool calls as their own tool parts instead.
- **Cancellation.** If the task is cancelled or `ctx` is finished elsewhere (e.g.
  `ctx.abort()`), the run is stopped with `result.cancel()`.
- Exceptions from the run (`MaxTurnsExceeded`, guardrail tripwires) propagate, so
  `ctx.run()` turns them into an `error` event.

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

56 tests covering: basic lifecycle, reasoning ↔ text transitions, tool calls,
multi-step flows, source events, edge cases (double finish, abort, write after finish),
StateStore integration, stream collection (`collect=True`), and custom information (`ctx.info`).

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
