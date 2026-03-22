# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-groups

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_context.py::test_name  # single test

# Lint & format
uv run ruff format src tests       # auto-format
uv run ruff check --fix src tests  # lint with auto-fix
uv run ruff format --check src tests  # check only
uv run ruff check src tests           # check only

# Type check
uv run pyright src

# Build
uv build
```

Git hooks (via lefthook) run format + lint + pyright on pre-commit and pytest on pre-push.

## Architecture

This library provides a lifecycle-managed, type-safe wrapper for emitting **Vercel AI SDK v6 UIMessageStream** SSE events from Python backends.

### Core Problem

The Vercel AI SDK v6 wire protocol requires strict event ordering (start → start-step → text-start → text-delta → text-end → finish-step → finish). This library manages that state machine automatically so callers never emit events out of order.

### Modules

- **`context.py`** (`StreamContext`) — Main interface. Tracks lifecycle state and auto-emits required prefix events before content. Buffers events in an `asyncio.Queue`; `ctx.stream()` drains the queue as an async generator yielding SSE strings for FastAPI `StreamingResponse`.
- **`events.py`** — 16 Pydantic models representing all wire protocol events, with a `UIMessageStreamEvent` discriminated union. Each event has an `encode()` method returning the SSE string.
- **`state.py`** (`StateStore`) — Async-safe key-value store with dot-path access (e.g., `"user.name"`, `"scores.0"`), backed by `asyncio.Lock`.

### Data Flow

```
Request handler creates StreamContext
  → background task calls ctx.write_text(), ctx.write_reasoning(), ctx.begin_tool_call(), etc.
  → StreamContext auto-emits lifecycle events as needed, queues all events
  → ctx.stream() yields SSE strings from the queue
  → FastAPI StreamingResponse sends SSE to client
```

### Key Patterns

- **Lifecycle state machine**: `_started`, `_step_open`, `_text_id`, `_reasoning_id`, `_finished` flags ensure protocol invariants. Any high-level write auto-emits all missing predecessors.
- **`None` sentinel** in the queue signals end-of-stream → yields `"data: [DONE]\n\n"`.
- **`ToolCallHandle`**: returned by `begin_tool_call()`, carries generated IDs that are passed back to `complete_tool_call()` / `fail_tool_call()`.
- **Escape hatches**: `write_event_to_stream()` is a sync low-level push with no auto-emit (for LlamaIndex compatibility).

### Integration Pattern

`ctx` is passed as a parameter through service layers. Services write events via `ctx.write_*()` and share state via `ctx.store.get/set()`. No global state.

## Standards

- **Python 3.10+** target.
- **Conventional Commits** enforced on commit messages (`feat:`, `fix:`, `chore:`, etc.). `feat!`/`fix!` trigger major version bumps.
- Ruff line length: 88, double quotes. Pyright in standard mode.
- `asyncio_mode = "auto"` in pytest — all test coroutines run automatically.
