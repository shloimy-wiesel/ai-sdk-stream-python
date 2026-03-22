# Project Guidelines

## Scope

- The root workspace is a Python library in `src/ai_sdk_stream_python/` with tests in `tests/`.
- `example/backend/` is a FastAPI integration example for the library.
- `example/frontend/` is a separate Next.js app with its own nested `AGENTS.md` and TypeScript-specific `.github/copilot-instructions.md`; follow those when working there.

## Build And Test

- Install dependencies with `uv sync --all-groups`.
- Run the library tests with `uv run pytest tests/ -v`.
- Format with `uv run ruff format src tests`.
- Lint with `uv run ruff check --fix src tests`.
- Type-check with `uv run pyright src`.
- Build distributions with `uv build`.
- Git hooks via `lefthook` run format, lint, and pyright on commit and pytest on push, so keep changes hook-clean.

## Architecture

- `src/ai_sdk_stream_python/context.py` contains `StreamContext`, the main API and lifecycle state machine for Vercel AI SDK v6 `UIMessageStream` event ordering.
- `src/ai_sdk_stream_python/events.py` defines the typed protocol events; prefer these models and `StreamContext` helpers over hand-built SSE payloads.
- `src/ai_sdk_stream_python/state.py` provides the async-safe `StateStore`; pass `ctx` through service layers instead of introducing global state.
- `src/ai_sdk_stream_python/collect.py` contains the record models populated when `StreamContext(collect=True)` is used.

## Conventions

- Fix streaming behavior at the `StreamContext` or event-model layer when possible instead of patching individual call sites.
- Prefer high-level helpers such as `write_text`, `write_reasoning`, `begin_tool_call`, `finish`, and `abort`; use raw event writes only for deliberate protocol-level control.
- Maintain Python 3.10 compatibility and the existing Ruff style: 88-column lines, double quotes, and standard pyright checking.
- Tests run with `pytest-asyncio` in auto mode, so async tests can be plain `async def` tests without extra decorators.
- Use Conventional Commits. See `CONTRIBUTING.md` for commit message rules and hook setup.

## References

- See `README.md` for the Quick Start, FastAPI integration pattern, and public API examples.
- See `CLAUDE.md` for the concise command reference, architecture summary, and protocol notes.
- See `CONTRIBUTING.md` for contributor setup, git hooks, and release conventions.
