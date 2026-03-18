/**
 * Proxy route: forwards useChat requests to the Python FastAPI backend.
 *
 * useChat (AI SDK v6) sends:  POST /api/chat  { messages: UIMessage[] }
 * We extract the last user message and forward to:
 *   POST http://localhost:8000/chat  { query, user_id }
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

  const lastUserMessage = [...messages]
    .reverse()
    .find((m) => m.role === "user");

  const query =
    lastUserMessage?.parts
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join("") ?? "";

  const upstream = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, user_id: "u1" }),
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
