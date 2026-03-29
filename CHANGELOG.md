# CHANGELOG


## v0.2.0-a.8 (2026-03-29)

### Features

- Add Vercel Chatbot Python backend using ai-sdk-stream-python (example only)
  ([`9a02914`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/9a02914814c238170895d1eaed6019cbbbc0133f))

- Implemented 5 tools: getWeather, createDocument, updateDocument, editDocument, requestSuggestions
  - Fixed conversation history persistence: assistant messages now saved to Redis - Uses
  StreamContext for lifecycle-safe streaming - Redis for session persistence (chat history +
  documents) - Handles delta strategy per artifact kind (text=incremental, code/sheet=full) - All
  tools verified working via browser Playwright tests


## v0.1.2 (2026-03-22)

### Bug Fixes

- Prevent tool name duplication and ensure tool call ID consistency in consume_openai_stream
  ([`6351182`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/6351182445e7ed42a58a0080a0b3dadbbc73640f))

Agent-Logs-Url:
  https://github.com/shloimy-wiesel/ai-sdk-stream-python/sessions/f0d42dd6-0fe4-44ed-be7e-1e5e9abaaf43

Co-authored-by: shloimy-wiesel <144027408+shloimy-wiesel@users.noreply.github.com>


## v0.2.0-a.7 (2026-03-23)

### Features

- Add convert_to_openai_messages() to contrib.openai
  ([#42](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/42),
  [`153145b`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/153145b686b35b63d7259687e89fb59a7315022c))

Converts AI SDK v6 UIMessage parts-based format to list[ChatCompletionMessageParam] for the OpenAI
  API, completing the input → output pipeline in contrib.openai alongside consume_openai_stream().

Part mapping: - TextUIPart → content string or {"type":"text"} block - FileUIPart (image) →
  {"type":"image_url"} content block - ToolUIPart (output-available) → assistant tool_calls +
  role:"tool" message - ToolUIPart (output-error) → assistant tool_calls + tool error message -
  ToolUIPart (other states) → assistant tool_calls only - ReasoningUIPart → dropped by default;
  included in <reasoning> tags when include_reasoning=True

Closes #31.


## v0.2.0-a.6 (2026-03-23)

### Features

- Add Pydantic request body types for AI SDK v6 message format
  ([#39](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/39),
  [`12d3e93`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/12d3e9358b531ae662a4ff0be905e9de166bcc17))

* feat: add Pydantic request body types for AI SDK v6 message format

Closes #32.

Adds `types.py` with typed Pydantic models for deserialising incoming `useChat` request bodies:
  `ChatRequest`, `UIMessage`, and all part types (`TextUIPart`, `ReasoningUIPart`, `FileUIPart`,
  `SourceUrlUIPart`, `SourceDocumentUIPart`, `StepStartUIPart`, `DataUIPart`, `ToolUIPart`).

`MessagePart` is a discriminated union that handles both literal type fields and dynamic prefixes
  (`tool-*`, `data-*`, `dynamic-tool`). Tool states use the v6 names (`input-streaming`,
  `input-available`, `output-available`, `output-error`). `ChatRequest` uses `extra="allow"` so
  app-specific fields pass through transparently.

* fix: address code review issues in request types

- Add extra="allow" to all simple part models for forward-compat - Add model_validator to ToolUIPart
  requiring toolName on dynamic-tool parts - Add transient field to DataUIPart matching outbound
  DataEvent - Drop Literal|str no-op on trigger; use plain str with doc comment - Raise ValueError
  for unknown part types instead of silently falling back to TextUIPart - Fix
  test_uimessage_extra_fields_allowed to assert extra field is accessible - Reorganize test_types.py
  into class Test* blocks matching project convention


## v0.2.0-a.5 (2026-03-23)

### Bug Fixes

- Accept any JSON-serializable value in write_data(), not just dict
  ([`cf9625d`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/cf9625d45abd36982194b01076e7a722b493c7bb))

The AI SDK v6 wire protocol allows the `data` field in `data-{name}` events to be any JSON value
  (string, number, null, array, or object). The previous `dict[str, Any]` constraint made common
  patterns like streaming chat titles or signaling artifact completion impossible.

Closes #34

- Add collect param to write_data() and use model_dump_json in DataEvent.encode()
  ([`57b3d82`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/57b3d82bb23f80eda7f43e64af65d9eff033ab68))

- write_data() now accepts collect: bool | None = None and routes through _should_collect(),
  consistent with all other high-level write helpers - DataEvent.encode() uses model_dump_json()
  (Pydantic's serializer) instead of model_dump() + json.dumps(), fixing TypeError for
  non-JSON-native types (datetime, UUID, Enum, etc.) inside the data field

- Raise RuntimeError when collect=True is passed on non-collecting context
  ([`2bbda8d`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/2bbda8de2ef4d8df8c2ce6bb061f4343c2fdf0b5))

Change per-call collect parameter default from True to None (follow context setting). When a caller
  explicitly passes collect=True but the StreamContext was created with collect=False, raise
  RuntimeError instead of silently discarding the data. This prevents a class of silent bugs where
  developers think they are collecting but the context has no record.

Three-way semantics: - None (default): collect if context-level collection is enabled - True:
  require collection; raise if no record exists - False: skip collection even if context-level is
  enabled

### Features

- Add `start_metadata` parameter to `StreamContext` for `start` event `messageMetadata`
  ([#26](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/26),
  [`6d9ca16`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/6d9ca16304c0b6f0299ab73d8e9f4780a7bb936f))

feat: support start_metadata on StreamContext for start event messageMetadata

Co-authored-by: shloimy-wiesel <144027408+shloimy-wiesel@users.noreply.github.com>

Agent-Logs-Url:
  https://github.com/shloimy-wiesel/ai-sdk-stream-python/sessions/e1fc71ca-6c44-48e5-9f74-df0da0cb80f5

---------

Co-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>

- Add contrib.openai.consume_openai_stream() utility
  ([`c6bf3a1`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/c6bf3a1aea6d0bd98e5f7bd1165dde95bb81ceaf))

Adds a new optional contrib module that eliminates the ~40-line boilerplate every OpenAI +
  StreamContext backend must write.

- Maps delta.content → ctx.write_text() - Maps delta.reasoning / delta.reasoning_content →
  ctx.write_reasoning() - Buffers tool call chunks by index; emits tool-input-start +
  tool-input-available at end (or streams deltas when stream_tool_deltas=True) - Extracts
  finish_reason and usage (when stream_options include_usage=True) - No hard dependency on the
  openai package — uses duck typing throughout - Returns ConsumeResult with content, tool_calls,
  finish_reason, usage

Closes #23

- Add ctx.run() safe task runner to prevent stream hangs on crash
  ([`2f585d9`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/2f585d954a71206a58fd37dfbed90d306cbc25f5))

When a background task raises an unhandled exception the stream now terminates with an error event
  instead of blocking the client forever. run() also auto-calls finish() if the coroutine returns
  without it.

Resolves: #20

- Add per-call collect=False to skip recording individual writes
  ([`778ab1e`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/778ab1ebf2c79578a12c7c088c3c6bcf7617900f))

Add a `collect` keyword parameter (default True) to write_text, write_reasoning, write_source,
  write_file, begin_tool_call, and start_tool_input. When collect=False, the event is still emitted
  to the SSE stream but not recorded in ctx.record. This lets developers stream ephemeral content
  (status messages, internal tool calls) without polluting the persisted record.

Closes #30

- Add timing fields to StreamRecord
  ([#27](https://github.com/shloimy-wiesel/ai-sdk-stream-python/pull/27),
  [`c81e608`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/c81e6082a4751cc2497b1e3ad1fbd832bc6426a5))

feat: add timing (created_at/finished_at/duration_ms) to StreamRecord

Co-authored-by: shloimy-wiesel <144027408+shloimy-wiesel@users.noreply.github.com>

Agent-Logs-Url:
  https://github.com/shloimy-wiesel/ai-sdk-stream-python/sessions/781f63ee-0c28-4c82-bcc1-73f004039b4d

* fix: address review comments on StreamRecord timing fields

- Remove dead-code guard in to_dict(): created_at is non-optional so the `if self.created_at else
  None` branch was never reachable - Document created_at, finished_at, and duration_ms in
  StreamRecord docstring - Rename test_finished_at_none_before_finish →
  test_finished_at_none_initially to avoid confusion (the test called finish() internally)

---------

Co-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>

Co-authored-by: shloimy wiesel <w.63071@gmail.com>

- Auto-count streaming tokens in StreamRecord with optional tokenizer
  ([`db3a352`](https://github.com/shloimy-wiesel/ai-sdk-stream-python/commit/db3a352480db0f9b732fb6f3da12fb0e6251494e))

Closes #17

- Add `reasoning_tokens`, `answer_tokens`, `prompt_tokens` fields to `StreamRecord` - Add
  `total_output_tokens` and `total_tokens` computed properties - `write_text` and `write_reasoning`
  now auto-increment token counts via configurable `count_func` (defaults to `len`, i.e. character
  count) - Add `StreamContext(count_func=...)` parameter to swap in any `Callable[[str], int]` (e.g.
  tiktoken, word count) - Add `ctx.set_usage(prompt_tokens, reasoning_tokens, answer_tokens)` to
  override auto-counted values with exact LLM-reported numbers - Include all token fields in
  `StreamRecord.to_dict()`


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
