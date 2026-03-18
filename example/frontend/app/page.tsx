"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart } from "ai";
import { PenSquareIcon } from "lucide-react";
import { useState } from "react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// ── Chat area — remounted on key change to fully reset useChat ─────────────

function ChatInterface() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  });

  const hasMessages = messages.length > 0;

  const promptInput = (
    <PromptInput
      onSubmit={({ text }) => {
        sendMessage({ text });
      }}
    >
      <PromptInputTextarea
        data-testid="chat-input"
        placeholder="Ask something..."
      />
      <PromptInputFooter>
        <PromptInputSubmit
          data-testid="chat-submit"
          onStop={stop}
          status={status}
        />
      </PromptInputFooter>
    </PromptInput>
  );

  if (!hasMessages) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-4 pb-16">
        <div className="w-full max-w-xl space-y-6 text-center">
          <p className="text-muted-foreground/70 text-sm">Ask anything</p>
          {promptInput}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-1 flex-col">
      <Conversation className="flex-1">
        <ConversationContent>
          {messages.map((message, messageIndex) => {
            const isLastMessage = messageIndex === messages.length - 1;
            return (
              <Message from={message.role} key={message.id}>
                <MessageContent>
                  {message.parts.map((part, i) => {
                    if (part.type === "text") {
                      return (
                        <MessageResponse
                          isAnimating={isLastMessage && status === "streaming"}
                          key={`${message.id}-text-${i}`}
                        >
                          {part.text}
                        </MessageResponse>
                      );
                    }
                    if (part.type === "reasoning") {
                      return (
                        <details
                          className="text-muted-foreground text-xs"
                          key={`${message.id}-reasoning-${i}`}
                        >
                          <summary className="cursor-pointer select-none py-1 font-medium">
                            Reasoning
                          </summary>
                          <p className="mt-1 whitespace-pre-wrap pl-2">
                            {part.text}
                          </p>
                        </details>
                      );
                    }
                    if (isToolUIPart(part)) {
                      const toolName = getToolName(part);
                      return (
                        <div
                          className="rounded border px-3 py-2 text-muted-foreground text-xs"
                          data-testid={`tool-part-${toolName}`}
                          key={`${message.id}-tool-${i}`}
                        >
                          <span className="font-medium">{toolName}</span>
                          {" · "}
                          <span>{part.state}</span>
                        </div>
                      );
                    }
                    return null;
                  })}
                </MessageContent>
              </Message>
            );
          })}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
      <div className="p-4">{promptInput}</div>
    </div>
  );
}

// ── Page shell — sidebar + chat area ──────────────────────────────────────

export default function Home() {
  const [chatKey, setChatKey] = useState(0);

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="flex w-52 flex-shrink-0 flex-col border-r">
        {/* Persistent title */}
        <div className="border-b px-4 py-4">
          <p className="font-semibold text-sm leading-tight">
            ai-sdk-stream-python
          </p>
          <p className="mt-0.5 text-muted-foreground text-xs">
            Python · AI SDK v6
          </p>
        </div>

        {/* New chat button */}
        <div className="p-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                className="w-full justify-start gap-2"
                onClick={() => setChatKey((k) => k + 1)}
                variant="ghost"
              >
                <PenSquareIcon className="size-4" />
                New chat
              </Button>
            </TooltipTrigger>
            <TooltipContent side="right">
              Start a new conversation
            </TooltipContent>
          </Tooltip>
        </div>
      </aside>

      {/* Main chat area */}
      <main className="flex flex-1 flex-col overflow-hidden">
        {/* Header — same height as sidebar title block */}
        <div className="flex flex-shrink-0 items-center border-b px-6 py-6">
          <span className="text-muted-foreground text-sm">
            New conversation
          </span>
        </div>
        <ChatInterface key={chatKey} />
      </main>
    </div>
  );
}
