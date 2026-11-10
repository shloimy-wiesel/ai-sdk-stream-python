// Feed nested_agent_transcript.sse through the AI SDK v6 client reader and
// check that each sub-agent card is Running (preliminary) then Completed.
//
//   node tests/fixtures/verify_nested_fixture.mjs [path/to/node_modules/ai]
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

const sse = readFileSync(new URL("nested_agent_transcript.sse", import.meta.url), "utf8");
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

const seen = { call_a: [], call_b: [] };
let last;
for await (const message of readUIMessageStream({ stream })) {
  last = message;
  for (const part of message.parts) {
    if (!part.toolCallId || part.state !== "output-available") continue;
    const text = part.output.message.parts
      .filter((p) => p.type === "text")
      .map((p) => p.text)
      .join("");
    const entry = `${part.preliminary ? "running" : "done"}:${text}`;
    const list = seen[part.toolCallId];
    if (list.at(-1) !== entry) list.push(entry);
  }
}

const expect = (id, tag) => {
  const states = seen[id];
  const final = states.at(-1);
  if (final !== `done:${tag}-one ${tag}-two`) throw new Error(`${id}: ${states}`);
  if (!states.slice(0, -1).every((s) => s.startsWith("running:")))
    throw new Error(`${id}: ${states}`);
  if (states.length < 2) throw new Error(`${id}: never shown as running`);
};
expect("call_a", "A");
expect("call_b", "B");
const tools = last.parts.filter((p) => p.toolCallId);
if (tools.some((p) => p.preliminary || p.output.result === undefined))
  throw new Error("final snapshot lost");
console.log("ok", JSON.stringify(seen));
