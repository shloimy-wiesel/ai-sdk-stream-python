/**
 * Proxy route: forwards useChat requests to the Python FastAPI backend.
 *
 * useChat (AI SDK v6) sends:  POST /api/chat  { messages: UIMessage[] }
 * We convert the full history to { role, content }[] and forward to:
 *   POST http://localhost:8000/chat  { messages: [{role, content}] }
 *
 * Sending the full history makes the backend stateless — it receives all
 * context it needs to produce a contextual reply on every request.
 *
 * The Python backend streams UIMessageStream SSE with the header
 * x-vercel-ai-ui-message-stream: v1, which useChat uses to select the
 * correct stream parser automatically.
 */

import type { UIMessage } from "ai";

export const maxDuration = 60;

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: Request): Promise<Response> {
  const { messages }: { messages: UIMessage[] } = await req.json();

  // Convert AI SDK UIMessage[] → simple {role, content}[] for the backend.
  // We extract only text parts and skip system/tool messages.
  const simpleMsgs = messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role,
      content: m.parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text")
        .map((p) => p.text)
        .join(""),
    }))
    .filter((m) => m.content.length > 0);

  const upstream = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: simpleMsgs }),
  });

  if (!(upstream.ok && upstream.body)) {
    return new Response("Backend error", { status: upstream.status });
  }

  return new Response(upstream.body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-vercel-ai-ui-message-stream":
        upstream.headers.get("x-vercel-ai-ui-message-stream") ?? "v1",
    },
  });
}
