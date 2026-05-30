# Library Internals — `ai_sdk_stream_python`

## OVERVIEW

Lifecycle state machine + typed event models that guarantee correct Vercel AI SDK v6 UIMessageStream SSE ordering from Python backends.

## MODULE MAP

| Module | Role | Key exports |
|--------|------|-------------|
| `context.py` (729 LOC) | State machine + public API | `StreamContext`, `ToolCallHandle`, `OnFinishCallback` |
| `events.py` (235 LOC) | 20 Pydantic event models + SSE encode | `BaseEvent`, `UIMessageStreamEvent`, all `*Event` classes |
| `state.py` | Async dot-path key/value store | `StateStore` |
| `collect.py` | Optional event recording for persistence | `StreamRecord`, `ToolCallRecord`, `SourceRecord`, `FileRecord`, `DataPartRecord` |
| `types.py` (245 LOC) | Incoming request deserialization | `ChatRequest`, `UIMessage`, `MessagePart`, all `*UIPart` classes |
| `contrib/openai.py` (433 LOC) | OpenAI adapter (duck-typed, no hard dep) | `consume_openai_stream`, `convert_to_openai_messages`, `ConsumeResult` |

## STATE MACHINE (`context.py`)

### Internal flags

```python
_started: bool         # StartEvent emitted?
_step_open: bool       # StartStepEvent emitted, FinishStepEvent not yet?
_text_id: str | None   # Currently-open text part ID
_reasoning_id: str | None  # Currently-open reasoning part ID
_finished: bool        # finish()/abort()/error() called?
```

### Transition helpers (private)

| Helper | Trigger | Events emitted |
|--------|---------|----------------|
| `_ensure_started()` | Any write when `_started=False` | `StartEvent` |
| `_ensure_step_open()` | Any content write when `_step_open=False` | `StartStepEvent` (calls `_ensure_started` first) |
| `_ensure_text_closed()` | Switching to reasoning/tool/step-close | `TextEndEvent` |
| `_ensure_reasoning_closed()` | Switching to text/tool/step-close | `ReasoningEndEvent` |
| `_ensure_step_closed()` | `new_step()`/`finish()` | Closes text+reasoning, then `FinishStepEvent` |

### Public methods → events

| Method | Auto-emits predecessors | Primary event(s) |
|--------|------------------------|-------------------|
| `write_text(delta)` | start, start-step, close reasoning | `TextStartEvent` (if new), `TextDeltaEvent` |
| `write_reasoning(delta)` | start, start-step, close text | `ReasoningStartEvent` (if new), `ReasoningDeltaEvent` |
| `begin_tool_call(name, input)` | start, start-step, close text+reasoning | `ToolInputStartEvent` + `ToolInputAvailableEvent` |
| `start_tool_input(name)` | start, start-step, close text+reasoning | `ToolInputStartEvent` |
| `stream_tool_input_delta(id, delta)` | — | `ToolInputDeltaEvent` |
| `finish_tool_input(id, name, input)` | — | `ToolInputAvailableEvent` |
| `complete_tool_call(id, output)` | — | `ToolOutputAvailableEvent` |
| `fail_tool_call(id, error)` | — | `ToolOutputErrorEvent` |
| `write_source(id, url, title)` | start | `SourceUrlEvent` |
| `write_file(url, media_type)` | start, start-step | `FileEvent` |
| `write_data(name, data)` | start | `DataEvent` (type=`"data-{name}"`) |
| `new_step()` | — | close step → `FinishStepEvent` + `StartStepEvent` |
| `finish(reason)` | start, close step | `FinishEvent` → `None` sentinel |
| `abort(reason)` | — | `AbortEvent` → `None` sentinel (no FinishEvent) |
| `error(text)` | start | `ErrorEvent` → `None` sentinel |
| `write(event)` | start only | Enqueues raw event (caller manages ordering) |
| `write_event_to_stream(ev)` | nothing | Lowest-level sync push (no auto-emit) |

### `ctx.run(coro)` guarantees

1. **Auto-finish**: `finally` block calls `ctx.finish()` if not finished
2. **Auto-error**: catches exceptions → `ctx.error(str(exc))`
3. **GC-safe**: stores `asyncio.Task` reference on `self._task`

## EVENT ENCODING (`events.py`)

- `BaseEvent.encode()` → `"data: " + model_dump_json(exclude_none=True) + "\n\n"`
- `DataEvent.encode()` overrides to force `"data"` field present even when `None`
- `None` sentinel in queue → `"data: [DONE]\n\n"` (stream termination)

## COLLECTION (`collect.py`)

When `StreamContext(collect=True)` or `on_finish` callback provided:

| Field | Updated by |
|-------|-----------|
| `text` | `write_text` (concatenated) |
| `reasoning` | `write_reasoning` (concatenated) |
| `tool_calls: list[ToolCallRecord]` | `begin_tool_call` / `complete_tool_call` / `fail_tool_call` |
| `sources: list[SourceRecord]` | `write_source` |
| `files: list[FileRecord]` | `write_file` |
| `data_parts: list[DataPartRecord]` | `write_data` (non-transient only) |
| `answer_tokens / reasoning_tokens` | Incremented by count_func per delta |
| `finish_reason` | Set by `finish()` |
| `finished_at / duration_ms` | Set by `finish()` |

Per-call `collect=False` streams event but skips recording. Per-call `collect=True` on non-collecting context **raises**.

## REQUEST TYPES (`types.py`)

`ChatRequest` ← `useChat` POST body. Contains `messages: list[UIMessage]`.
`UIMessage.parts` is a discriminated union (`_discriminate_part`) of 8 part types.
All models use `ConfigDict(extra="allow")` for forward-compatibility.

## OPENAI ADAPTER (`contrib/openai.py`)

- **Input**: `convert_to_openai_messages(messages)` — maps UIMessage parts to OpenAI `ChatCompletionMessageParam` dicts. Handles text, images, tool calls (with results), reasoning (opt-in).
- **Output**: `consume_openai_stream(stream, ctx)` — async iterator over OpenAI chunks → `ctx.write_*()` calls. Buffers tool calls by index; supports streaming deltas (`stream_tool_deltas=True`). Returns `ConsumeResult`. Does NOT call `ctx.finish()`.
- Duck-typed: works with any `openai`-compatible client. No import-time dependency.

## CONVENTIONS (THIS PACKAGE)

- All `_ensure_*` helpers are sync-safe (no await needed internally, they call `write_event_to_stream` which is sync `put_nowait`).
- Event IDs generated via `_make_id()` (UUID4 hex prefix).
- `on_finish` callbacks: sync or async accepted; exceptions swallowed (logged, never block termination).
- `StateStore` lock is `asyncio.Lock` — single-event-loop safe only. `snapshot()` is NOT lock-protected.
