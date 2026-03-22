# CHANGELOG


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
