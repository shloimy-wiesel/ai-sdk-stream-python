/**
 * Helpers for building UIMessageStream SSE responses in tests.
 *
 * Event types match the exact wire format expected by AI SDK v6 useChat
 * (same as emitted by ai_sdk_stream_python on the Python side).
 *
 * Protocol: SSE lines `data: <json>\n\n`, terminated by `data: [DONE]\n\n`.
 * Header: `x-vercel-ai-ui-message-stream: v1`
 */

type SSEEvent = Record<string, unknown>;

const encodeEvent = (event: SSEEvent): string =>
  `data: ${JSON.stringify(event)}\n\n`;

const DONE_SENTINEL = "data: [DONE]\n\n";

/** Build a complete UIMessageStream SSE body as a single string. */
export const buildStream = (events: SSEEvent[]): string =>
  events.map(encodeEvent).join("") + DONE_SENTINEL;

/**
 * Lifecycle event constructors matching AI SDK v6 uiMessageChunkSchema
 * and ai_sdk_stream_python events.py.
 */
export const streamEvents = {
  // ── Message lifecycle ───────────────────────────────────────────────
  start: (messageId?: string): SSEEvent =>
    messageId ? { type: "start", messageId } : { type: "start" },

  startStep: (): SSEEvent => ({ type: "start-step" }),

  finishStep: (): SSEEvent => ({ type: "finish-step" }),

  finish: (finishReason = "stop"): SSEEvent => ({
    type: "finish",
    finishReason,
  }),

  // ── Text ────────────────────────────────────────────────────────────
  textStart: (id: string): SSEEvent => ({ type: "text-start", id }),

  textDelta: (id: string, delta: string): SSEEvent => ({
    type: "text-delta",
    id,
    delta,
  }),

  /** text-end (not text-finish) — matches AI SDK v6 uiMessageChunkSchema */
  textEnd: (id: string): SSEEvent => ({ type: "text-end", id }),

  // ── Reasoning ───────────────────────────────────────────────────────
  reasoningStart: (id: string): SSEEvent => ({
    type: "reasoning-start",
    id,
  }),

  reasoningDelta: (id: string, delta: string): SSEEvent => ({
    type: "reasoning-delta",
    id,
    delta,
  }),

  /** reasoning-end (not reasoning-finish) */
  reasoningEnd: (id: string): SSEEvent => ({ type: "reasoning-end", id }),

  // ── Tool calls ──────────────────────────────────────────────────────
  toolInputStart: (toolCallId: string, toolName: string): SSEEvent => ({
    type: "tool-input-start",
    toolCallId,
    toolName,
  }),

  /** inputTextDelta (not delta) — matches AI SDK v6 uiMessageChunkSchema */
  toolInputDelta: (toolCallId: string, inputTextDelta: string): SSEEvent => ({
    type: "tool-input-delta",
    toolCallId,
    inputTextDelta,
  }),

  /** tool-output-available (not tool-result) with output field (not result) */
  toolOutputAvailable: (toolCallId: string, output: unknown): SSEEvent => ({
    type: "tool-output-available",
    toolCallId,
    output,
  }),

  // ── Sources ─────────────────────────────────────────────────────────
  /** source-url (not source) — matches AI SDK v6 uiMessageChunkSchema */
  sourceUrl: (sourceId: string, url: string, title?: string): SSEEvent => ({
    type: "source-url",
    sourceId,
    url,
    ...(title ? { title } : {}),
  }),
};

/** Build a simple text-only response (single step). */
export const buildTextResponse = (
  text: string,
  messageId = "msg-1",
  textId = "txt-1"
): string => {
  const words = text.split(" ");
  return buildStream([
    streamEvents.start(messageId),
    streamEvents.startStep(),
    streamEvents.textStart(textId),
    ...words.map((word, i) =>
      streamEvents.textDelta(textId, i === 0 ? word : ` ${word}`)
    ),
    streamEvents.textEnd(textId),
    streamEvents.finishStep(),
    streamEvents.finish(),
  ]);
};

/** Build a response with reasoning + text (two steps). */
export const buildReasoningAndTextResponse = (
  reasoning: string,
  text: string,
  messageId = "msg-1"
): string =>
  buildStream([
    streamEvents.start(messageId),
    streamEvents.startStep(),
    streamEvents.reasoningStart("rsn-1"),
    streamEvents.reasoningDelta("rsn-1", reasoning),
    streamEvents.reasoningEnd("rsn-1"),
    streamEvents.finishStep(),
    streamEvents.startStep(),
    streamEvents.textStart("txt-1"),
    streamEvents.textDelta("txt-1", text),
    streamEvents.textEnd("txt-1"),
    streamEvents.finishStep(),
    streamEvents.finish(),
  ]);

/** Build a response with a tool call followed by text (two steps). */
export const buildToolAndTextResponse = (
  toolName: string,
  toolInput: string,
  toolOutput: unknown,
  text: string,
  messageId = "msg-1"
): string =>
  buildStream([
    streamEvents.start(messageId),
    streamEvents.startStep(),
    streamEvents.toolInputStart("tc-1", toolName),
    streamEvents.toolInputDelta("tc-1", toolInput),
    streamEvents.toolOutputAvailable("tc-1", toolOutput),
    streamEvents.finishStep(),
    streamEvents.startStep(),
    streamEvents.textStart("txt-1"),
    streamEvents.textDelta("txt-1", text),
    streamEvents.textEnd("txt-1"),
    streamEvents.finishStep(),
    streamEvents.finish(),
  ]);

/** UIMessageStream response headers (required for useChat to auto-detect format). */
export const STREAM_HEADERS: Record<string, string> = {
  "content-type": "text/event-stream",
  "cache-control": "no-cache",
  "x-vercel-ai-ui-message-stream": "v1",
};
