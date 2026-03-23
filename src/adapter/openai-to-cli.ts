/**
 * Converts OpenAI chat request format to Claude CLI input
 */

import type { OpenAIChatRequest, OpenAIContentBlock } from "../types/openai.js";

export type ClaudeModel = "opus" | "sonnet" | "haiku";

export interface CliInput {
  prompt: string;
  model: ClaudeModel;
  sessionId?: string;
}

const MODEL_MAP: Record<string, ClaudeModel> = {
  // Direct model names (provider prefixes like `claude-code-cli/` and `claude-max/`
  // are stripped by extractModel before consulting this map)
  "claude-opus-4": "opus",
  "claude-opus-4-6": "opus",
  "claude-sonnet-4": "sonnet",
  "claude-sonnet-4-5": "sonnet",
  "claude-sonnet-4-6": "sonnet",
  "claude-haiku-4": "haiku",
  "claude-haiku-4-5": "haiku",
  // Bare aliases
  "opus": "opus",
  "sonnet": "sonnet",
  "haiku": "haiku",
  "opus-max": "opus",
  "sonnet-max": "sonnet",
};

/**
 * Extract Claude model alias from request model string
 */
export function extractModel(model: string): ClaudeModel {
  // Try direct lookup
  if (MODEL_MAP[model]) {
    return MODEL_MAP[model];
  }

  // Try stripping provider prefix
  const stripped = model.replace(/^(?:claude-code-cli|claude-max)\//, "");
  if (MODEL_MAP[stripped]) {
    return MODEL_MAP[stripped];
  }

  // Default to opus (Claude Max subscription)
  return "opus";
}

/**
 * Extract text from a content field that may be a string or array of content blocks.
 * OpenAI API allows content as either:
 *   - A plain string: "Hello"
 *   - An array of content blocks: [{"type": "text", "text": "Hello"}]
 */
function extractText(content: string | OpenAIContentBlock[]): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((block) => block.type === "text" || block.type === "input_text")
      .map((block) => block.text)
      .join("\n");
  }
  return String(content || "");
}

/**
 * Strip OpenClaw-specific tooling sections from system prompts.
 * These reference tools (exec, process, web_search, etc.) that don't exist
 * in the Claude Code CLI environment, causing the model to get confused.
 * We remove: ## Tooling, ## Tool Call Style, ## OpenClaw CLI Quick Reference,
 * ## OpenClaw Self-Update
 */
function stripOpenClawTooling(text: string): string {
  const sectionsToStrip = [
    "## Tooling",
    "## Tool Call Style",
    "## OpenClaw CLI Quick Reference",
    "## OpenClaw Self-Update",
  ];
  let result = text;
  for (const section of sectionsToStrip) {
    // Match from section header to the next ## header (or end of string)
    const pattern = new RegExp(
      section.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") +
        "\\n[\\s\\S]*?(?=\\n## |$)",
      "g"
    );
    result = result.replace(pattern, "");
  }
  // Clean up excessive blank lines left behind
  result = result.replace(/\n{3,}/g, "\n\n");
  return result.trim();
}

/**
 * Convert OpenAI messages array to a single prompt string for Claude CLI
 *
 * Claude Code CLI in --print mode expects a single prompt, not a conversation.
 * We format the messages into a readable format that preserves context.
 */
export function messagesToPrompt(
  messages: OpenAIChatRequest["messages"]
): string {
  const parts: string[] = [];

  for (const msg of messages) {
    const text = extractText(msg.content);
    switch (msg.role) {
      case "system":
        // System messages become context instructions
        // Strip OpenClaw tooling sections that conflict with Claude Code's native tools
        parts.push(`<system>\n${stripOpenClawTooling(text)}\n</system>\n`);
        break;

      case "user":
        // User messages are the main prompt
        parts.push(text);
        break;

      case "assistant":
        // Previous assistant responses for context
        parts.push(`<previous_response>\n${text}\n</previous_response>\n`);
        break;
    }
  }

  return parts.join("\n").trim();
}

/**
 * Detect if a message is "simple" (short conversational query) that can be
 * handled by a faster model (Sonnet) instead of Opus.
 *
 * Criteria:
 * - Last user message is under 200 chars
 * - No code blocks, no file paths, no complex instructions
 * - No explicit model override requesting opus
 */
function isSimpleQuery(messages: OpenAIChatRequest["messages"]): boolean {
  // Find last user message
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  if (!lastUser) return false;

  const text = extractText(lastUser.content);

  // Short messages are likely simple — bumped threshold for more Sonnet routing
  if (text.length > 300) return false;

  // Detect complex content patterns
  const complexPatterns = [
    /```/,                    // code blocks
    /\.(ts|js|py|md|pdf|pptx|json|yaml|sh)\b/i, // file extensions
    /\/[a-zA-Z]/,             // file paths (more precise than bare /)
    /(写|做|生成|创建|修改|分析|开发|实现|编写|设计|整合|优化|部署|配置)/,  // action verbs for complex tasks
    /(报告|方案|PPT|文档|手册|计划)/,  // document types
  ];

  for (const pattern of complexPatterns) {
    if (pattern.test(text)) return false;
  }

  return true;
}

/**
 * Convert OpenAI chat request to CLI input format.
 *
 * When AUTO_MODEL_ROUTING is enabled (default), simple conversational queries
 * are automatically routed to Sonnet for faster response times (~2-3x faster),
 * unless the request explicitly specifies an Opus model.
 */
export function openaiToCli(request: OpenAIChatRequest): CliInput {
  let model = extractModel(request.model);

  // Auto-route simple queries to Sonnet for speed (disabled by default for stability)
  const autoRoute = process.env.AUTO_MODEL_ROUTING === "true";
  const explicitOpus = /opus/i.test(request.model);
  if (autoRoute && model === "opus" && !explicitOpus && isSimpleQuery(request.messages)) {
    model = "sonnet";
  }

  return {
    prompt: messagesToPrompt(request.messages),
    model,
    sessionId: request.user, // Use OpenAI's user field for session mapping
  };
}
