# CHANGELOG


## v0.2.0-a.4 (2026-03-23)

### Features

- Add `on_finish` callback for post-stream persistence
  ([#24](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/24),
  [`fcbb2fd`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/fcbb2fd9ceca1a0cd1083f4001bf918c8ef174f1))

* Initial plan

* feat: add on_finish callback for post-stream persistence

Co-authored-by: shloimy-wiesel <144027408+shloimy-wiesel@users.noreply.github.com>

Agent-Logs-Url:
  https://github.com/shloimy-wiesel/ai-sdk-stream-python/sessions/5f2f939a-91ee-46f4-9d8e-5abc6ef8a02e

---------

Co-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>


## v0.2.0-a.3 (2026-03-22)

### Bug Fixes

- Abort() emits proper abort event per v6 spec
  ([`093862b`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/093862b6ce97c463a6248a31ac2a06760d0338d6))

Add AbortEvent model (type: "abort", reason: str | None) and update ctx.abort() to accept an
  optional reason and emit the event before the [DONE] sentinel. Backward-compatible: bare
  ctx.abort() still works and omits the reason field from the wire output.

Resolves: #13

### Features

- Add custom data parts and ctx.write_data() helper
  ([`036fc74`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/036fc7446dd63df7bc49096ea207175bbce8766b))

Add DataEvent model (dynamic data-{name} type), ctx.write_data() with name validation and transient
  flag, DataPartRecord for collection, and data_parts list on StreamRecord with to_dict() support.
  Transient parts are sent on the wire but not persisted to ctx.record.

Resolves: #10

- Add error event type and ctx.error() helper
  ([`13836d4`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/13836d4ee7f0573a2c45069e8d191e20a8d9c5e0))

Add ErrorEvent Pydantic model (type: "error", errorText: str), ctx.error(error_text) which emits the
  event then terminates the stream, and exports ErrorEvent from the top-level package.

Resolves: #8

- Add file event type and ctx.write_file() helper
  ([`d11d6ec`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/d11d6ec169a071a84c17a74ccac6e9d9748e70e7))

Add FileEvent model (type: "file", url, mediaType), ctx.write_file() which auto-emits
  start/start-step before the event, FileRecord for collection, and files list on StreamRecord with
  to_dict() support.

Resolves: #9

- Add streaming tool input delta helpers
  ([`45baa02`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/45baa0280257464aa9df375938d069bc573a2510))

Add start_tool_input(), stream_tool_input_delta(), and finish_tool_input() to StreamContext.
  start_tool_input emits tool-input-start and returns a ToolCallHandle; each stream_tool_input_delta
  emits a tool-input-delta; finish_tool_input emits tool-input-available and updates the collected
  ToolCallRecord with the final input dict. begin_tool_call() is unchanged.

Resolves: #11


## v0.2.0-a.2 (2026-03-22)

### Features

- Add typed custom_information support to StreamContext
  ([#7](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/7),
  [`f6e8da9`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/f6e8da9a5ba6226e1f1e5255db9b017ecdf9683e))

* feat: add typed custom_information support to StreamContext

Closes #6

- StreamContext is now Generic[_InfoT] (bound to pydantic.BaseModel) - New `custom_information`
  constructor parameter stores a read-only Pydantic model accessible via ctx.info throughout the
  stream lifecycle - Useful for carrying request-scoped metadata (user_id, rate_limit, etc.) through
  service layers without extra function arguments - 5 new tests; pyright passes with 0 errors

* fix: correct docstring type and test organization for custom_information

- Fix `info : _InfoT` → `info : _InfoT | None` in class Attributes docstring - Group
  custom_information tests into TestCustomInformation class


## v0.2.0-a.1 (2026-03-22)

### Features

- Add collect=True option to StreamContext for recording stream data
  ([`a594acc`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/a594acc80a6529e7a54c1fe8b7d3fd78dfa9cee8))

Adds an opt-in collect: bool = False parameter to StreamContext that accumulates all emitted content
  (text, reasoning, tool calls, sources, step count, finish reason) into a StreamRecord accessible
  via ctx.record. Useful for persisting conversation turns to a database after streaming.


## v0.1.2-a.1 (2026-03-22)

### Bug Fixes

- Trigger new release
  ([`059f64f`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/059f64f50c17b1d52a97f40647d5b070440e3691))


## v0.1.1 (2026-03-22)

### Bug Fixes

- Use correct Python version specifier >=3.10,<3.13 for 3.10–3.12 support
  ([`ad4c603`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/ad4c603e4f4221f9556ef84517e74f5adb322e4c))

- Use correct Python version specifier >=3.10,<3.13 for 3.10–3.12 support
  ([`0ba2e9c`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/0ba2e9c26fa20311fd4ac1e2056de65adddfbfcd))


## v0.1.0 (2026-03-22)

### Bug Fixes

- **ci**: Disable PSR built-in build to avoid missing uv in Docker container
  ([`2d0a186`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/2d0a1868fc6d70da19b767d685f7628e5132e17a))

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

### Features

- Add chore func
  ([`5ada704`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/5ada704c696a1cc214e665824ce13d5fc1b13591))

- Add streaming protocol, context lifecycle, CI/CD, and example backend
  ([`6c21518`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/6c215180d24574413e2b8ddecfee5f9c3c57c384))

### Core Library - Add StreamContext (context.py) — lifecycle-managed wrapper that auto-emits
  required UIMessageStream SSE events in correct order (start → step → text → finish) - Add 16
  Pydantic event models (events.py) with UIMessageStreamEvent discriminated union - Add async-safe
  StateStore (state.py) with dot-path key access - Expose all public APIs via __init__.py

### Example App - Move Python backend into example/frontend/ for single Vercel deployment - Add
  FastAPI backend (api/) with /api/chat route supporting AI SDK UIMessage format (parts) - Add
  llm_service.py and db_service.py service layer - Fix: accept AI SDK UIMessage parts format in chat
  route - Fix: track lib/utils.ts excluded by root .gitignore

### CI/CD & Tooling - Add CI workflow (lint + tests on push/PR) - Add lint workflow (Ruff + Pyright)
  - Add release workflow with semantic versioning (Conventional Commits) - Replace
  python-publish.yml with new release pipeline - Add lefthook.yml for pre-commit
  (format/lint/typecheck) and pre-push (pytest) hooks - Update pyproject.toml to require Python
  3.10+, add all dev dependencies - Add uv.lock and example/frontend/uv.lock - Update
  CI/lint/release to use latest actions/checkout and astral-sh/setup-uv

### Docs & Config - Expand README.md with full usage, new_step() patterns, and text answer handling
  - Add CONTRIBUTING.md - Add CLAUDE.md with architecture overview and dev commands - Add AI SDK
  skill references under .agents/skills/ai-sdk/ - Add .claude/settings.json and skills-lock.json

### Tests - Add comprehensive test suite (tests/test_context.py, 387 lines) - Add Playwright test
  results to .gitignore

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **example**: Add FastAPI backend and improve chat UI
  ([`62d947e`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/62d947e89f5d850bb630e429af9616a9a59f5b2f))

Backend: - Add FastAPI backend with /chat streaming endpoint using StreamContext - Real LLM
  integration via OpenAI-compatible SDK (LLM_BASE_URL/API_KEY/MODEL env vars) - Tool calling
  support: LLM can call search_documents, results streamed via ctx - Stateless design: full
  conversation history sent on every request - python-dotenv for .env loading, openai package
  dependency - .env.example with LLM config template - README with setup instructions

Frontend: - Sidebar layout: persistent app title + New Chat button - Main header aligned with
  sidebar title height - Centered empty state with prompt input when no messages - Send full message
  history to backend (stateless BE) - Thin styled scrollbar (dark mode friendly) - Body locked to
  viewport height (overflow-hidden) so prompt input stays visible - Enter key submits, New Chat
  resets all state via key remount - Updated page metadata title/description

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

- **example**: Init frontend
  ([`bb36017`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/bb36017bb6077e4ed0b7363e0f5cac9d2ea6bdaf))


## v0.0.1 (2026-03-10)

### Features

- Init Python package with uv, fastapi, and pydantic
  ([`65be9aa`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/65be9aaca914fc68fe290ab0eec46fbb96c5a821))

Co-authored-by: shloimy-wiesel <144027408+shloimy-wiesel@users.noreply.github.com>
