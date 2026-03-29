# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-23 · **Commit:** 43b0ad2 · **Branch:** alpha

## WHAT THIS LIBRARY SOLVES

The Vercel AI SDK v6 frontend (`useChat`) expects a **UIMessageStream** — a strictly ordered sequence of typed SSE events (`start → start-step → text-start → text-delta* → text-end → finish-step → finish → [DONE]`). Getting this ordering wrong silently breaks the frontend.

**This library eliminates that problem.** `StreamContext` is a lifecycle state machine that auto-emits required prefix/suffix events so Python backends never send events out of order. Inspired by [llama-index-workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) — a `Context` object that holds shared state and writes typed events to a stream.

### Core pattern

```python
ctx = StreamContext()
await ctx.run(lambda ctx: my_work(ctx))   # auto-finish, auto-error, GC-safe
return StreamingResponse(ctx.stream(), media_type="text/event-stream", headers=ctx.response_headers)
```

`ctx` flows as a parameter through service layers (like a DB session). Services call `ctx.write_text()`, `ctx.begin_tool_call()`, `ctx.store.set()` — never raw SSE strings.

## STRUCTURE

```
./
├── src/ai_sdk_stream_python/  # The library (see nested AGENTS.md for internals)
│   ├── context.py             # StreamContext — lifecycle state machine (729 LOC)
│   ├── events.py              # 20 Pydantic event models + SSE encoding
│   ├── state.py               # StateStore — async dot-path key/value store
│   ├── collect.py             # StreamRecord — optional event collection for DB persistence
│   ├── types.py               # Incoming request models (ChatRequest, UIMessage, MessagePart)
│   └── contrib/openai.py      # OpenAI stream consumer + UIMessage→OpenAI converter
├── tests/                     # 56+ tests: lifecycle, tool calls, collection, edge cases
├── example/
│   ├── backend/               # FastAPI demo: ctx passed through 3 service modules
│   └── frontend/              # Next.js + AI SDK v6 useChat (separate AGENTS.md)
```

## HOW IT WORKS

### The state machine

StreamContext tracks 5 flags: `_started`, `_step_open`, `_text_id`, `_reasoning_id`, `_finished`. High-level helpers auto-emit missing predecessors:

```
write_text("Hi") on fresh context auto-emits:
  StartEvent → StartStepEvent → TextStartEvent → TextDeltaEvent("Hi")

write_reasoning("think") while text is open auto-emits:
  TextEndEvent → ReasoningStartEvent → ReasoningDeltaEvent("think")

finish() auto-emits:
  ReasoningEndEvent → FinishStepEvent → FinishEvent → [DONE] sentinel
```

### Event flow (producer-consumer)

```
ctx.write_*() → Pydantic event → asyncio.Queue (put_nowait)
                                        ↓
ctx.stream() → awaits queue.get() → ev.encode() → "data: {json}\n\n"
                                        ↓
FastAPI StreamingResponse → SSE bytes to client
```

- `None` sentinel in queue → yields `"data: [DONE]\n\n"` → generator returns.
- Response header: `x-vercel-ai-ui-message-stream: v1`.

### Key capabilities

| Feature | How |
|---|---|
| **Auto-ordering** | `_ensure_started()`, `_ensure_step_open()` emit missing lifecycle events |
| **Part switching** | `write_text` auto-closes reasoning; `write_reasoning` auto-closes text |
| **Tool calls** | `begin_tool_call()` → `ToolCallHandle` → `complete_tool_call()` / `fail_tool_call()` |
| **Streaming tool input** | `start_tool_input()` → `stream_tool_input_delta()` → `finish_tool_input()` |
| **Shared state** | `ctx.store.get/set()` — asyncio.Lock-protected dot-path store |
| **Typed metadata** | `ctx.info` — read-only Pydantic model for request-scoped data |
| **Collection** | `collect=True` records all events into `ctx.record` (StreamRecord) |
| **Safe runner** | `ctx.run(coro)` — auto-finish in finally, auto-error on exception, prevents task GC |
| **Abort/Error** | `ctx.abort()` ends without finish; `ctx.error()` emits error event; both idempotent |

### Wire protocol (20 event types)

```
start → start-step → [reasoning-start → reasoning-delta* → reasoning-end]
                    → [tool-input-start → tool-input-available → tool-output-available]
                    → [text-start → text-delta* → text-end]
                    → [source-url*] [file*]
      → finish-step → finish → [DONE]
```

Special: `DataEvent` has dynamic type (`"data-{name}"`) and ensures `"data"` field is present even when null.

### OpenAI integration (`contrib/openai`)

Two functions — no hard `openai` dependency (duck-typed):
- `convert_to_openai_messages(messages)` — UIMessage parts → OpenAI `ChatCompletionMessageParam` list
- `consume_openai_stream(stream, ctx)` — maps OpenAI chunks → `ctx.write_text/write_reasoning/begin_tool_call`; returns `ConsumeResult` (content, tool_calls, finish_reason, usage). Does NOT call `ctx.finish()`.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Change lifecycle ordering | `context.py` `_ensure_*` helpers | State machine transitions |
| Add a new event type | `events.py` + add to `UIMessageStreamEvent` union | Follow BaseEvent pattern |
| Change SSE encoding | `events.py` `BaseEvent.encode()` | DataEvent overrides encode |
| Add collection fields | `collect.py` StreamRecord | Update context.py write helpers too |
| Change request parsing | `types.py` | Pydantic models with `extra="allow"` |
| OpenAI chunk mapping | `contrib/openai.py` `consume_openai_stream` | Supports reasoning + tool streaming |
| Fix streaming bugs | `context.py` or `events.py` | Never patch individual call sites |

## CONVENTIONS

- **Python 3.10** target. Ruff: 88 cols, double quotes. Pyright standard mode.
- **Conventional Commits** required — lefthook commit-msg hook enforces regex.
- **No `print()`** in library code — ruff T20 rule enforced.
- Tests: `pytest-asyncio` auto mode — plain `async def` tests, no decorators.
- Prefer `ctx.run(coro)` over manual `asyncio.create_task` + try/finally.
- Fix streaming at `StreamContext`/event layer, not individual call sites.

## ANTI-PATTERNS (THIS PROJECT)

- ❌ Hand-building SSE strings (`"data: " + json.dumps(...)`) — use StreamContext helpers
- ❌ Writing events after `ctx.finish()` — raises RuntimeError
- ❌ Manual `asyncio.create_task` without storing the reference — use `ctx.run()`
- ❌ Global state for request-scoped data — use `ctx.store` or `ctx.info`
- ❌ `as any` / `@ts-ignore` in frontend — separate AGENTS.md applies there
- ❌ Per-call `collect=True` when context was created with `collect=False` — raises RuntimeError

## COMMANDS

```bash
uv sync --all-groups          # install deps
uv run pytest tests/ -v       # test
uv run ruff format src tests  # format
uv run ruff check --fix src tests  # lint
uv run pyright src             # type check
uv build                       # build dist
```

Git hooks: `lefthook install` → pre-commit (format+lint+pyright), commit-msg (conventional), pre-push (pytest).

## NOTES

- `ctx.record` is `None` unless `collect=True` or `on_finish` callback provided.
- `on_finish` callback exceptions are swallowed — they never block stream termination.
- `StateStore.snapshot()` is NOT lock-protected — use `get()` for consistent reads under contention.
- `DataEvent` excluded from `UIMessageStreamEvent` discriminated union (dynamic type field).
- `example/frontend/` is a separate Next.js project — follow its own `AGENTS.md` and `.github/copilot-instructions.md` when working there.
