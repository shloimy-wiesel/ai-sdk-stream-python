import { expect, test } from "@playwright/test";
import {
  buildReasoningAndTextResponse,
  buildTextResponse,
  buildToolAndTextResponse,
  STREAM_HEADERS,
} from "./helpers/sse-stream";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const sendMessage = async (
  page: import("@playwright/test").Page,
  text: string
) => {
  await page.getByTestId("chat-input").fill(text);
  await page.getByTestId("chat-submit").click();
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("UIMessageStream protocol — text events", () => {
  test("renders streamed text from text-delta events", async ({ page }) => {
    const responseText = "Hello from the AI assistant";
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse(responseText),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Hi");

    await expect(page.getByText(responseText)).toBeVisible({ timeout: 10_000 });
  });

  test("shows user message in conversation after sending", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse("OK"),
      });
    });

    await page.goto("/");
    await sendMessage(page, "What is 2 + 2?");

    await expect(page.getByText("What is 2 + 2?")).toBeVisible({
      timeout: 5000,
    });
  });

  test("clears input after message is sent", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse("OK"),
      });
    });

    await page.goto("/");
    const input = page.getByTestId("chat-input");
    await input.fill("test message");
    await page.getByTestId("chat-submit").click();

    // Input should be cleared after submit
    await expect(input).toHaveValue("", { timeout: 5000 });
  });

  test("handles multiple text-delta events accumulating into full text", async ({
    page,
  }) => {
    // Multi-word text that arrives as separate delta events
    const responseText = "The quick brown fox jumps over the lazy dog";
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse(responseText),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Say a sentence");

    await expect(page.getByText(responseText)).toBeVisible({
      timeout: 10_000,
    });
  });

  test("supports Enter key to submit message", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse("Received via Enter"),
      });
    });

    await page.goto("/");
    await page.getByTestId("chat-input").fill("Hello via enter");
    await page.getByTestId("chat-input").press("Enter");

    await expect(page.getByText("Received via Enter")).toBeVisible({
      timeout: 10_000,
    });
  });
});

test.describe("UIMessageStream protocol — reasoning events", () => {
  test("renders reasoning part from reasoning-delta events", async ({
    page,
  }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildReasoningAndTextResponse(
          "I need to think step by step.",
          "The answer is 42."
        ),
      });
    });

    await page.goto("/");
    await sendMessage(page, "What is the answer?");

    // Reasoning is shown in a <details> element
    await expect(page.getByText("Reasoning")).toBeVisible({ timeout: 10_000 });
    // Text answer is shown
    await expect(page.getByText("The answer is 42.")).toBeVisible({
      timeout: 10_000,
    });
  });

  test("reasoning section is collapsible via <details> element", async ({
    page,
  }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildReasoningAndTextResponse("My internal thoughts.", "Done."),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Think aloud");

    // The <details> summary should be visible
    const summary = page.getByText("Reasoning");
    await expect(summary).toBeVisible({ timeout: 10_000 });

    // Click to expand/collapse
    await summary.click();
    await expect(page.getByText("My internal thoughts.")).toBeVisible();
  });
});

test.describe("UIMessageStream protocol — tool events", () => {
  test("renders tool part after tool-input-start event", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildToolAndTextResponse(
          "search_documents",
          '{"query": "test"}',
          [{ title: "Result 1" }],
          "Here are the results."
        ),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Search for test");

    // Tool part rendered with data-testid
    await expect(page.getByTestId("tool-part-search_documents")).toBeVisible({
      timeout: 10_000,
    });
  });

  test("tool part shows tool name", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildToolAndTextResponse(
          "fetch_weather",
          '{"city": "NYC"}',
          { temp: 72, condition: "sunny" },
          "It is sunny in NYC."
        ),
      });
    });

    await page.goto("/");
    await sendMessage(page, "What is the weather?");

    const toolPart = page.getByTestId("tool-part-fetch_weather");
    await expect(toolPart).toBeVisible({ timeout: 10_000 });
    await expect(toolPart).toContainText("fetch_weather");
  });

  test("text appears after tool call completes", async ({ page }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildToolAndTextResponse(
          "lookup_user",
          '{"id": "u1"}',
          { name: "Alice" },
          "Hello Alice!"
        ),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Who am I?");

    await expect(page.getByTestId("tool-part-lookup_user")).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByText("Hello Alice!")).toBeVisible({
      timeout: 10_000,
    });
  });
});

test.describe("UIMessageStream protocol — conversation flow", () => {
  test("renders multiple turns of conversation", async ({ page }) => {
    let callCount = 0;
    await page.route("**/api/chat", (route) => {
      callCount++;
      const response =
        callCount === 1
          ? buildTextResponse("First response")
          : buildTextResponse("Second response");
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: response,
      });
    });

    await page.goto("/");

    // First turn
    await sendMessage(page, "Message 1");
    await expect(page.getByText("First response")).toBeVisible({
      timeout: 10_000,
    });

    // Second turn
    await sendMessage(page, "Message 2");
    await expect(page.getByText("Second response")).toBeVisible({
      timeout: 10_000,
    });

    // Both messages remain visible
    await expect(page.getByText("Message 1")).toBeVisible();
    await expect(page.getByText("Message 2")).toBeVisible();
  });

  test("shows empty state on initial load before any messages", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByText("Ask anything")).toBeVisible();
  });

  test("empty state disappears after first message is sent", async ({
    page,
  }) => {
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse("Hi there!"),
      });
    });

    await page.goto("/");
    await expect(page.getByText("Ask anything")).toBeVisible();

    await sendMessage(page, "Hello");
    await expect(page.getByText("Hi there!")).toBeVisible({ timeout: 10_000 });

    // Empty state gone once messages exist
    await expect(page.getByText("Ask anything")).not.toBeVisible();
  });
});

test.describe("UIMessageStream protocol — stream headers", () => {
  test("request to /api/chat includes required headers", async ({ page }) => {
    let capturedHeaders: Record<string, string> = {};

    await page.route("**/api/chat", async (route) => {
      capturedHeaders = Object.fromEntries(
        Object.entries(route.request().headers())
      );
      await route.fulfill({
        status: 200,
        headers: STREAM_HEADERS,
        body: buildTextResponse("OK"),
      });
    });

    await page.goto("/");
    await sendMessage(page, "test");
    await expect(page.getByText("OK")).toBeVisible({ timeout: 10_000 });

    expect(capturedHeaders["content-type"]).toContain("application/json");
  });

  test("response with x-vercel-ai-ui-message-stream header is parsed correctly", async ({
    page,
  }) => {
    // This verifies the proxy sets the correct header and useChat picks it up
    await page.route("**/api/chat", (route) => {
      route.fulfill({
        status: 200,
        headers: {
          ...STREAM_HEADERS,
          "x-vercel-ai-ui-message-stream": "v1",
        },
        body: buildTextResponse("Parsed correctly!"),
      });
    });

    await page.goto("/");
    await sendMessage(page, "Check parsing");

    await expect(page.getByText("Parsed correctly!")).toBeVisible({
      timeout: 10_000,
    });
  });
});
