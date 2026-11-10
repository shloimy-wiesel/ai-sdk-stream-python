// Feed nested_agent_failure.sse through the AI SDK v6 client reader and check
// that a failed and an interrupted sub-agent card both end as output-error,
// while their transcripts survive as data-subagent-transcript parts (the
// client drops a tool part's output on tool-output-error).
//
//   node tests/fixtures/verify_nested_failure_fixture.mjs [path/to/node_modules/ai]
//
// Without an argument, `ai` is resolved from the current directory
// (e.g. run it from example/frontend after `npm install`).
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const aiPath = process.argv[2]
  ? pathToFileURL(createRequire(`${process.cwd()}/`).resolve(process.argv[2]))
      .href
  : "ai";
const { readUIMessageStream } = await import(aiPath);

const sse = readFileSync(new URL("nested_agent_failure.sse", import.meta.url), "utf8");
const chunks = sse
  .split("\n\n")
  .map((l) => l.replace(/^data: /, "").trim())
  .filter((l) => l && l !== "[DONE]")
  .map((l) => JSON.parse(l));

const stream = new ReadableStream({
  start(c) {
    for (const chunk of chunks) c.enqueue(chunk);
    c.close();
  },
});

let last;
let streamError;
try {
  for await (const message of readUIMessageStream({
    stream,
    onError: (e) => {
      streamError = e;
    },
  }))
    last = message;
} catch (e) {
  streamError = e;
}

const fail = (msg) => {
  throw new Error(`${msg}\n${JSON.stringify(last.parts, null, 1)}`);
};
const expectCard = (id, errorText, text) => {
  const card = last.parts.find((p) => p.toolCallId === id);
  if (card?.state !== "output-error") fail(`${id}: not output-error`);
  if (card.errorText !== errorText) fail(`${id}: errorText ${card.errorText}`);
  const data = last.parts.find(
    (p) => p.type === "data-subagent-transcript" && p.id === `subagent_${id}`,
  );
  if (data?.data.toolCallId !== id) fail(`${id}: transcript data part missing`);
  const parts = data.data.message.parts;
  const got = parts.filter((p) => p.type === "text").map((p) => p.text).join("");
  if (got !== text) fail(`${id}: transcript text ${got}`);
  if (parts.some((p) => p.state === "streaming")) fail(`${id}: part still streaming`);
};
expectCard(
  "call_b",
  "An error occurred while running the tool. Please try again.",
  "B partial",
);
expectCard("call_a", "Sub-agent run did not complete", "A partial");
if ((streamError?.message ?? streamError) !== "model failed")
  throw new Error(`stream error: ${streamError}`);
console.log("ok");
