"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart } from "ai";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
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

export default function Home() {
  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  });

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-1 flex-col">
      <Conversation className="flex-1">
        <ConversationContent>
          {messages.length === 0 && (
            <ConversationEmptyState
              description="Start a conversation with the AI assistant"
              title="Ask anything"
            />
          )}
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
      <div className="p-4">
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
      </div>
    </div>
  );
}
