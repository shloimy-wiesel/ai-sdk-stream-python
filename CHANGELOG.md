# CHANGELOG


## v0.3.0 (2026-05-25)

### Features

- Add stream_exclude and store_exclude parameters to StreamContext
  ([#47](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/47),
  [`a87b743`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/a87b743cf609018e64004303551346199fade325))

feat: add stream_exclude and store_exclude parameters to StreamContext

Agent-Logs-Url:
  https://github.com/shloimy-wiesel/ai-sdk-stream-python/sessions/ed622073-4b0a-4004-a3d8-eb53d3363f9d


## v0.2.2 (2026-05-05)

### Bug Fixes

- Respect context-level collect flag in _should_collect
  ([`b59668c`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/b59668cb6748f7bd8d2dd2a11713072b4c2dfe98))


## v0.2.1 (2026-05-05)

### Bug Fixes

- Update pydantic dependency version to 2.0.0
  ([`b545f9d`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/b545f9df424e6954a6e859d101d4fc5fc8b12a8f))


## v0.2.0 (2026-03-29)

### Features

- Complete v0.2.0 feature set — collection, typed info, events, OpenAI adapter, and request types
  ([`749cab2`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/749cab22b69af12cdf7f3025dec5fca7cca9079e))

Add the full StreamContext feature surface for AI SDK v6 compatibility:

Stream collection & persistence: - collect=True accumulates text, reasoning, tool calls, sources,
  files, data parts, token counts, and timing into StreamRecord (ctx.record) - on_finish callback
  for post-stream DB persistence - Per-call collect=False to skip recording ephemeral writes -
  RuntimeError when collect=True on non-collecting context

StreamContext enhancements: - Generic[_InfoT] typed custom_information (ctx.info) for request-scoped
  metadata - ctx.run() safe task runner: auto-finish, auto-error, GC-safe - ctx.error() emits
  ErrorEvent then terminates stream - ctx.abort() emits AbortEvent per v6 spec (with optional
  reason) - ctx.write_file() for FileEvent support - ctx.write_data() for dynamic data-{name} parts
  (any JSON-serializable value) - start_tool_input/stream_tool_input_delta/finish_tool_input for
  streaming tool input - start_metadata parameter for start event messageMetadata - Auto-count
  streaming tokens with configurable count_func + set_usage() override - Timing fields (created_at,
  finished_at, duration_ms) on StreamRecord

New event types: - ErrorEvent, AbortEvent, FileEvent, DataEvent (dynamic type) -
  ToolInputStartEvent, ToolInputDeltaEvent, ToolInputAvailableEvent

OpenAI contrib module (contrib/openai): - consume_openai_stream() maps OpenAI chunks to ctx.write_*
  calls - convert_to_openai_messages() converts UIMessage parts to ChatCompletionMessageParam -
  Duck-typed — no hard openai package dependency

Request body types (types.py): - ChatRequest, UIMessage, and all v6 part models (TextUIPart,
  ToolUIPart, etc.) - Discriminated union for MessagePart with dynamic prefix support -
  extra="allow" for forward compatibility

Bug fixes: - Tool name deduplication in consume_openai_stream - Tool call ID consistency using
  ToolCallHandle.toolCallId - write_data() accepts any JSON-serializable value, not just dict -
  DataEvent.encode() uses model_dump_json() for non-native types

Also adds chatbot-backend example (FastAPI + Redis + 5 tools), hierarchical AGENTS.md knowledge
  base, and 237 passing tests.


## v0.1.2 (2026-03-22)

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
