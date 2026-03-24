/**
 * API Route Handlers
 *
 * Implements OpenAI-compatible endpoints for Clawdbot integration.
 * Uses a simple concurrency limiter to prevent resource exhaustion.
 */

import type { Request, Response } from "express";
import crypto from "crypto";
import { execFile, spawn } from "child_process";
import { v4 as uuidv4 } from "uuid";
import { ClaudeSubprocess } from "../subprocess/manager.js";
import { openaiToCli } from "../adapter/openai-to-cli.js";
import {
  cliResultToOpenai,
  createDoneChunk,
} from "../adapter/cli-to-openai.js";
import type { OpenAIChatRequest } from "../types/openai.js";
import type { ClaudeCliAssistant, ClaudeCliResult, ClaudeCliStreamEvent } from "../types/claude-cli.js";

// ---------------------------------------------------------------------------
// Result Cache — prevents wasted work when gateway reconnects
// ---------------------------------------------------------------------------

interface CachedResult {
  status: "running" | "complete" | "error";
  chunks: string[];          // accumulated SSE data lines
  finalResult?: ClaudeCliResult;
  errorMessage?: string;
  subprocess?: ClaudeSubprocess;
  createdAt: number;
  lastModel: string;
  requestId: string;
}

const resultCache = new Map<string, CachedResult>();
const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
const CACHE_MAX_SIZE = 50;

/** Derive a cache key from message content + model + stream mode */
function cacheKey(body: OpenAIChatRequest): string {
  const payload = JSON.stringify({ messages: body.messages, model: body.model, stream: !!body.stream });
  return crypto.createHash("sha256").update(payload).digest("hex").slice(0, 20);
}

/** Evict expired entries and enforce size limit */
function evictCache(): void {
  const now = Date.now();
  for (const [key, entry] of resultCache) {
    if (now - entry.createdAt > CACHE_TTL_MS) {
      resultCache.delete(key);
    }
  }
  // If still over size, remove oldest
  if (resultCache.size > CACHE_MAX_SIZE) {
    const sorted = [...resultCache.entries()].sort((a, b) => a[1].createdAt - b[1].createdAt);
    for (let i = 0; i < sorted.length - CACHE_MAX_SIZE; i++) {
      resultCache.delete(sorted[i][0]);
    }
  }
}

// Track all active subprocesses for graceful shutdown
export const activeSubprocesses = new Set<ClaudeSubprocess>();

// ---------------------------------------------------------------------------
// CLI Health Tracker — detects Claude Max rate limits and service issues
// ---------------------------------------------------------------------------

/** Keywords that indicate Claude Max usage/rate limiting */
const RATE_LIMIT_PATTERNS = [
  /rate.?limit/i,
  /usage.?limit/i,
  /too many requests/i,
  /overloaded/i,
  /capacity/i,
  /quota/i,
  /exceeded/i,
  /throttl/i,
  /429/,
  /limited/i,
  /try again later/i,
  /maximum.*usage/i,
];

interface CliOutcome {
  timestamp: number;
  success: boolean;
  isRateLimit: boolean;
  errorMessage?: string;
  exitCode?: number | null;
  durationMs?: number;
}

const CLI_HISTORY_SIZE = 20; // keep last 20 outcomes
const cliOutcomes: CliOutcome[] = [];
let lastRateLimitDetected: number | null = null;

function isRateLimitError(text: string): boolean {
  return RATE_LIMIT_PATTERNS.some((pat) => pat.test(text));
}

// ---------------------------------------------------------------------------
// Rate Limit Notification — sends Telegram alert when rate limit is detected
// ---------------------------------------------------------------------------

const NOTIFY_TARGET = process.env.NOTIFY_TELEGRAM_TARGET || "7857161753";
const NOTIFY_COOLDOWN_MS = 30 * 60 * 1000; // Don't spam — at most once per 30 min
let lastNotifySent = 0;

function notifyRateLimit(errorMessage?: string): void {
  const now = Date.now();
  if (now - lastNotifySent < NOTIFY_COOLDOWN_MS) return;
  lastNotifySent = now;

  const time = new Date().toLocaleString("zh-CN", { timeZone: "Asia/Shanghai" });
  const msg = [
    "⚠️ Claude Max 用量已达限额",
    "",
    `时间: ${time}`,
    errorMessage ? `错误: ${errorMessage.slice(0, 200)}` : "",
    "",
    "代理服务暂时无法处理新请求，需等待限额重置或切换备用账号。",
  ].filter(Boolean).join("\n");

  // Fire-and-forget — don't block request handling
  execFile(
    "openclaw",
    ["message", "send", "--channel", "telegram", "--target", NOTIFY_TARGET, "-m", msg],
    { timeout: 15000 },
    (err) => {
      if (err) {
        console.error("[Notify] Failed to send rate limit alert:", err.message);
      } else {
        console.log("[Notify] Rate limit alert sent to Telegram");
      }
    }
  );
}

// ---------------------------------------------------------------------------
// Proactive Quota Probe — periodically tests Claude CLI availability
// ---------------------------------------------------------------------------

const PROBE_INTERVAL_MS = parseInt(process.env.PROBE_INTERVAL_MS || "300000", 10); // 5 min
const PROBE_TIMEOUT_MS = 30_000; // 30s — if CLI can't answer "hi" in 30s, it's down
let lastProbeResult: { timestamp: number; ok: boolean; error?: string; durationMs: number } | null = null;
let probeTimer: NodeJS.Timeout | null = null;

/**
 * Send a minimal prompt to Claude CLI and check if it responds.
 * This catches quota exhaustion, auth issues, and CLI hangs that
 * passive monitoring misses (because no user request triggers detection).
 */
async function probeQuota(): Promise<void> {
  const start = Date.now();
  try {
    const result = await new Promise<{ ok: boolean; error?: string }>((resolve) => {
      const proc = spawn(process.env.CLAUDE_BIN || "claude", [
        "--print",
        "--dangerously-skip-permissions",
        "--output-format", "text",
        "--model", "sonnet",
        "--max-turns", "1",
        "-p", "说ok",
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: PROBE_TIMEOUT_MS,
        env: Object.fromEntries(
          Object.entries(process.env).filter(([k]) => k !== "CLAUDECODE")
        ),
      });

      let stdout = "";
      let stderr = "";
      proc.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
      proc.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

      const killTimer = setTimeout(() => {
        try { proc.kill("SIGKILL"); } catch { /* */ }
        resolve({ ok: false, error: "探测超时(30s)" });
      }, PROBE_TIMEOUT_MS);

      proc.on("close", (code) => {
        clearTimeout(killTimer);
        if (code === 0 && stdout.trim().length > 0) {
          resolve({ ok: true });
        } else {
          const errMsg = stderr.trim() || stdout.trim() || `exit code ${code}`;
          // Check if this is a rate limit
          if (isRateLimitError(errMsg)) {
            resolve({ ok: false, error: `限额: ${errMsg.slice(0, 200)}` });
          } else {
            resolve({ ok: false, error: errMsg.slice(0, 200) });
          }
        }
      });

      proc.on("error", (err) => {
        clearTimeout(killTimer);
        resolve({ ok: false, error: err.message });
      });
    });

    const durationMs = Date.now() - start;
    lastProbeResult = { timestamp: Date.now(), ok: result.ok, error: result.error, durationMs };

    if (!result.ok) {
      console.error(`[Probe] Claude CLI 不可用: ${result.error} (${durationMs}ms)`);
      // Record as failure for health tracking
      recordCliOutcome({
        timestamp: Date.now(),
        success: false,
        isRateLimit: result.error ? isRateLimitError(result.error) : false,
        errorMessage: `[probe] ${result.error}`,
      });
      // Notify BOSS — the notifyRateLimit function has its own cooldown
      notifyRateLimit(result.error || "Claude CLI 探测失败");
    } else {
      console.log(`[Probe] Claude CLI 正常 (${durationMs}ms)`);
    }
  } catch (err) {
    const durationMs = Date.now() - start;
    const msg = err instanceof Error ? err.message : "unknown error";
    lastProbeResult = { timestamp: Date.now(), ok: false, error: msg, durationMs };
    console.error(`[Probe] 探测异常: ${msg}`);
  }
}

/** Start the periodic quota probe */
export function startQuotaProbe(): void {
  if (probeTimer) return;
  // First probe after 60s (give server time to settle)
  setTimeout(() => {
    probeQuota();
    probeTimer = setInterval(probeQuota, PROBE_INTERVAL_MS);
    probeTimer.unref();
  }, 60_000);
  console.log(`[Probe] 限额探测已启动 (间隔=${PROBE_INTERVAL_MS / 1000}s)`);
}

/** Record a CLI subprocess outcome for health tracking */
function recordCliOutcome(outcome: CliOutcome): void {
  cliOutcomes.push(outcome);
  if (cliOutcomes.length > CLI_HISTORY_SIZE) {
    cliOutcomes.shift();
  }
  if (outcome.isRateLimit) {
    lastRateLimitDetected = outcome.timestamp;
    console.error(
      `[RateLimit] Claude Max rate limit detected! error="${outcome.errorMessage?.slice(0, 200)}"`
    );
    // Immediately notify BOSS via Telegram
    notifyRateLimit(outcome.errorMessage);
  }
}

/** Get CLI health summary for /health endpoint */
function getCliHealthSummary(): {
  status: "ok" | "degraded" | "rate_limited" | "failing";
  recent_success_rate: number;
  last_rate_limit: string | null;
  consecutive_failures: number;
  last_error: string | null;
} {
  if (cliOutcomes.length === 0) {
    return {
      status: "ok",
      recent_success_rate: 1,
      last_rate_limit: lastRateLimitDetected
        ? new Date(lastRateLimitDetected).toISOString()
        : null,
      consecutive_failures: 0,
      last_error: null,
    };
  }

  const successes = cliOutcomes.filter((o) => o.success).length;
  const rate = successes / cliOutcomes.length;

  // Count consecutive failures from the end
  let consecutiveFailures = 0;
  for (let i = cliOutcomes.length - 1; i >= 0; i--) {
    if (!cliOutcomes[i].success) consecutiveFailures++;
    else break;
  }

  // Check for recent rate limit (within last 30 minutes)
  const recentRateLimit =
    lastRateLimitDetected && Date.now() - lastRateLimitDetected < 30 * 60 * 1000;

  // Find last error message
  let lastError: string | null = null;
  for (let i = cliOutcomes.length - 1; i >= 0; i--) {
    if (cliOutcomes[i].errorMessage) {
      lastError = cliOutcomes[i].errorMessage!.slice(0, 300);
      break;
    }
  }

  let status: "ok" | "degraded" | "rate_limited" | "failing";
  if (recentRateLimit) {
    status = "rate_limited";
  } else if (consecutiveFailures >= 3) {
    status = "failing";
  } else if (rate < 0.7) {
    status = "degraded";
  } else {
    status = "ok";
  }

  return {
    status,
    recent_success_rate: Math.round(rate * 100) / 100,
    last_rate_limit: lastRateLimitDetected
      ? new Date(lastRateLimitDetected).toISOString()
      : null,
    consecutive_failures: consecutiveFailures,
    last_error: lastError,
  };
}

// ---------------------------------------------------------------------------
// Concurrency Control — single lane, simple and predictable
// ---------------------------------------------------------------------------

/**
 * Counting semaphore — limits concurrent access to a shared resource.
 * acquire() returns a promise that resolves when a slot is available.
 */
class Semaphore {
  private count: number;
  private readonly max: number;
  private readonly waiting: Array<() => void> = [];

  constructor(max: number) {
    this.max = max;
    this.count = max;
  }

  async acquire(): Promise<void> {
    if (this.count > 0) {
      this.count--;
      return;
    }
    return new Promise<void>((resolve) => this.waiting.push(resolve));
  }

  release(): void {
    if (this.waiting.length > 0) {
      this.waiting.shift()!();
    } else if (this.count < this.max) {
      this.count++;
    }
  }

  /** Number of waiters in queue */
  get queueLength(): number {
    return this.waiting.length;
  }

  /** Available slots right now */
  get available(): number {
    return this.count;
  }
}

// Single concurrency lane — 5 concurrent subprocesses max
const concurrency = new Semaphore(
  parseInt(process.env.MAX_CONCURRENCY || "3", 10)
);
// Max waiters before rejecting with 503
const MAX_QUEUE = parseInt(process.env.MAX_QUEUE || "20", 10);

// Output truncation — kills subprocess if output exceeds this limit.
// Prevents infinite loop output (e.g. model repeating "好了" thousands of times).
const MAX_OUTPUT_CHARS = parseInt(process.env.MAX_OUTPUT_CHARS || "80000", 10);

// Grace period before killing subprocess after client disconnects (ms).
// Allows short completions to finish (result goes to cache), but prevents
// long-running orphans from wasting tokens.
const DISCONNECT_GRACE_MS = parseInt(process.env.DISCONNECT_GRACE_MS || "10000", 10);

// Hard request timeout — kills subprocess if it exceeds this limit (ms).
// Prevents runaway requests from blocking concurrency slots indefinitely.
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || "600000", 10); // 10 min

// First-byte timeout — kills subprocess if no content arrives within this window (ms).
// Claude Opus sometimes takes 2-4 minutes to return the first byte, which causes
// upstream gateways to timeout. This ensures we fail fast instead of hanging.
const FIRST_BYTE_TIMEOUT_MS = parseInt(process.env.FIRST_BYTE_TIMEOUT_MS || "300000", 10); // 5 min

/**
 * Handle POST /v1/chat/completions
 *
 * Main endpoint for chat requests, supports both streaming and non-streaming.
 * Routes through quick or heavy bulkhead to prevent starvation.
 */
export async function handleChatCompletions(
  req: Request,
  res: Response
): Promise<void> {
  const requestId = uuidv4().replace(/-/g, "").slice(0, 24);
  const body = req.body as OpenAIChatRequest;
  const stream = body.stream === true;

  try {
    // Validate request
    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(400).json({
        error: {
          message: "messages is required and must be a non-empty array",
          type: "invalid_request_error",
          code: "invalid_messages",
        },
      });
      return;
    }

    // Reject if too many waiters already queued
    if (concurrency.queueLength >= MAX_QUEUE) {
      console.warn(`[Concurrency] Full (queue=${concurrency.queueLength}), rejecting ${requestId}`);
      res.status(503).json({
        error: {
          message: "Server busy, please retry",
          type: "server_error",
          code: "concurrency_full",
        },
      });
      return;
    }

    // --- Cache hit check ---
    const key = cacheKey(body);
    const cached = resultCache.get(key);

    if (cached) {
      if (cached.status === "complete") {
        console.log(`[Cache] HIT (complete) key=${key} for ${requestId}`);
        if (stream) {
          res.setHeader("Content-Type", "text/event-stream");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Connection", "keep-alive");
          res.flushHeaders();
          for (const chunk of cached.chunks) {
            res.write(chunk);
          }
          res.write("data: [DONE]\n\n");
          res.end();
        } else if (cached.finalResult) {
          res.json(cliResultToOpenai(cached.finalResult, requestId));
        }
        return;
      }
      if (cached.status === "running" && stream) {
        // Reattach to running subprocess — replay buffered chunks then stream live
        console.log(`[Cache] REATTACH (running) key=${key} for ${requestId}`);
        await reattachToRunning(res, cached, requestId);
        return;
      }
    }
    // --- End cache check ---

    if (concurrency.available === 0) {
      console.log(`[Concurrency] Queuing request ${requestId} (waiting=${concurrency.queueLength})`);
    }

    // Wait for a slot
    await concurrency.acquire();
    console.log(`[Concurrency] Slot acquired for ${requestId} (available=${concurrency.available})`);

    try {
      evictCache();
      // Convert to CLI input format
      const cliInput = openaiToCli(body);
      const startTime = Date.now();
      console.log(`[Perf] ${requestId} model=${cliInput.model} stream=${stream}`);
      const subprocess = new ClaudeSubprocess();
      activeSubprocesses.add(subprocess);

      // Initialize cache entry
      const cacheEntry: CachedResult = {
        status: "running",
        chunks: [],
        createdAt: Date.now(),
        lastModel: "claude-opus-4",
        requestId,
        subprocess,
      };
      resultCache.set(key, cacheEntry);

      try {
        if (stream) {
          await handleStreamingResponse(req, res, subprocess, cliInput, requestId, cacheEntry);
        } else {
          await handleNonStreamingResponse(res, subprocess, cliInput, requestId, cacheEntry);
        }
      } finally {
        activeSubprocesses.delete(subprocess);
      }
    } finally {
      concurrency.release();
      console.log(`[Concurrency] Slot released for ${requestId} (available=${concurrency.available})`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[handleChatCompletions] Error:", message);

    if (!res.headersSent) {
      res.status(500).json({
        error: {
          message,
          type: "server_error",
          code: null,
        },
      });
    }
  }
}

// ---------------------------------------------------------------------------
// English Narration Filter — prevents Claude's internal English text from
// leaking to the chat. Works per content block: buffers the first N chars,
// and if no CJK characters appear, suppresses the entire block.
// ---------------------------------------------------------------------------

/**
 * Lightweight English narration filter — only suppresses content blocks that
 * start with clearly identifiable English narration patterns (e.g. "I'll read
 * the file", "Let me check"). Once a block is identified as narration, it is
 * suppressed until CJK text reappears.
 *
 * IMPORTANT: This filter does NOT interrupt model reasoning. It only affects
 * the output stream — the model completes its full thinking process internally.
 * Non-narration English content (code, URLs, technical terms, data) passes
 * through unmodified.
 */

const CJK_RE = /[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]/;
const PROBE_LEN = 60;

/** Patterns that identify English narration (Claude's internal status updates) */
const EN_NARRATION = [
  /^I('ll| will| need| can| should| am going| see that| found| notice| want)/i,
  /^Let me /i,
  /^(Now|First|Next|Also|Finally),? (I('ll| will)|let me|let's)/i,
  /^(Reading|Writing|Editing|Running|Searching|Looking|Checking|Creating|Using|Analyzing|Processing|Updating|Starting|Building|Compiling|Installing|Fixing|Adding|Examining|Reviewing) /i,
  /^The (file|code|function|error|output|result|issue|problem|directory|module|package|config|user|request) /i,
  /^Here('s| is| are) (the|a|my|what)/i,
  /^(Based on|According to|It (looks|seems|appears)|This (is|looks|seems|will))/i,
  /^(Good|Great|Perfect|Done|OK|Alright)[,!.]? /i,
];

interface BlockFilter {
  phase: "probe" | "pass" | "suppress";
  buf: string;       // only used during probe
}

function createBlockFilter(): BlockFilter {
  return { phase: "probe", buf: "" };
}

function filterTextDelta(filter: BlockFilter, text: string): string {
  // ---- Suppress: drop everything unless CJK reappears ----
  if (filter.phase === "suppress") {
    if (CJK_RE.test(text)) {
      filter.phase = "pass";
      return text;
    }
    return "";
  }

  // ---- Pass: forward everything (no mid-block suppression) ----
  if (filter.phase === "pass") {
    return text;
  }

  // ---- Probe: buffer first PROBE_LEN chars to classify the block ----
  filter.buf += text;

  // CJK detected → this is user-facing content, pass immediately
  if (CJK_RE.test(filter.buf)) {
    const out = filter.buf;
    filter.buf = "";
    filter.phase = "pass";
    return out;
  }

  // Check for narration patterns once we have enough text
  if (filter.buf.length >= 15) {
    const trimmed = filter.buf.trimStart();
    for (const pat of EN_NARRATION) {
      if (pat.test(trimmed)) {
        filter.buf = "";
        filter.phase = "suppress";
        return "";
      }
    }
  }

  // Reached probe limit without matching narration → pass through
  // (could be code, data, or other legitimate English content)
  if (filter.buf.length >= PROBE_LEN) {
    const out = filter.buf;
    filter.buf = "";
    filter.phase = "pass";
    return out;
  }

  return ""; // still probing
}

/**
 * Reattach to a running subprocess — replay buffered chunks then stream live.
 * This handles the case where gateway dropped the SSE connection and reconnected.
 */
async function reattachToRunning(
  res: Response,
  cached: CachedResult,
  requestId: string
): Promise<void> {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);
  res.flushHeaders();
  res.write(":reconnected\n\n");

  // Replay buffered chunks
  for (const chunk of cached.chunks) {
    res.write(chunk);
  }

  // If already complete while we were replaying
  if (cached.status === "complete") {
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  // Stream live from the subprocess
  const sub = cached.subprocess;
  if (!sub) {
    res.write("data: [DONE]\n\n");
    res.end();
    return;
  }

  return new Promise<void>((resolve) => {
    let clientDisconnected = false;

    const safeWrite = (data: string): boolean => {
      if (clientDisconnected || res.writableEnded) return false;
      try { res.write(data); return true; } catch { clientDisconnected = true; return false; }
    };

    // SSE comment keepalive every 15s
    const heartbeat = setInterval(() => {
      if (!clientDisconnected && !res.writableEnded) {
        try { res.write(":heartbeat\n\n"); } catch { clientDisconnected = true; }
      }
    }, 15_000);

    res.on("close", () => {
      clearInterval(heartbeat);
      clientDisconnected = true;
      resolve();
    });

    const onDelta = (event: ClaudeCliStreamEvent) => {
      const text = event.event?.delta?.type === "text_delta" ? event.event.delta.text : "";
      if (text) {
        const chunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: cached.lastModel,
          choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
        };
        safeWrite(`data: ${JSON.stringify(chunk)}\n\n`);
      }
    };

    const onResult = () => {
      clearInterval(heartbeat);
      if (!clientDisconnected && !res.writableEnded) {
        safeWrite("data: [DONE]\n\n");
        res.end();
      }
      cleanup();
      resolve();
    };

    const onClose = () => {
      clearInterval(heartbeat);
      if (!clientDisconnected && !res.writableEnded) {
        safeWrite("data: [DONE]\n\n");
        res.end();
      }
      cleanup();
      resolve();
    };

    const cleanup = () => {
      sub.removeListener("content_delta", onDelta);
      sub.removeListener("result", onResult);
      sub.removeListener("close", onClose);
    };

    sub.on("content_delta", onDelta);
    sub.on("result", onResult);
    sub.on("close", onClose);
  });
}

/**
 * Handle streaming response (SSE)
 *
 * IMPORTANT: The Express req.on("close") event fires when the request body
 * is fully received, NOT when the client disconnects. For SSE connections,
 * we use res.on("close") to detect actual client disconnection.
 */
async function handleStreamingResponse(
  req: Request,
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  cacheEntry?: CachedResult
): Promise<void> {
  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Request-Id", requestId);

  // CRITICAL: Flush headers immediately to establish SSE connection
  // Without this, headers are buffered and client times out waiting
  res.flushHeaders();

  // Send initial comment to confirm connection is alive
  res.write(":ok\n\n");

  const streamStartTime = Date.now();

  return new Promise<void>((resolve, reject) => {
    let isFirst = true;
    let lastModel = "claude-opus-4";
    let isComplete = false;
    let hasEmittedText = false;
    let clientDisconnected = false;
    let blockFilter = createBlockFilter();
    let firstByteLogged = false;
    let totalOutputChars = 0;
    let outputTruncated = false;
    let disconnectTimer: NodeJS.Timeout | null = null;
    let requestTimer: NodeJS.Timeout | null = null;
    let firstByteTimer: NodeJS.Timeout | null = null;
    let thinkingTimer: NodeJS.Timeout | null = null;

    // Helper: safe write that checks both res state and client connection
    const safeWrite = (data: string): boolean => {
      if (clientDisconnected || res.writableEnded) return false;
      try {
        res.write(data);
        return true;
      } catch {
        clientDisconnected = true;
        return false;
      }
    };

    // SSE keepalive — sends comment lines every 15s to keep the TCP connection
    // alive through proxies/gateways. SSE comments (lines starting with `:`)
    // are ignored by SSE parsers but reset proxy read timeouts.
    const heartbeatInterval = setInterval(() => {
      if (clientDisconnected || res.writableEnded || isComplete) return;
      try {
        res.write(":heartbeat\n\n");
      } catch {
        clientDisconnected = true;
      }
    }, 15_000);

    const cleanup = () => {
      clearInterval(heartbeatInterval);
      if (requestTimer) { clearTimeout(requestTimer); requestTimer = null; }
      if (firstByteTimer) { clearTimeout(firstByteTimer); firstByteTimer = null; }
      if (thinkingTimer) { clearTimeout(thinkingTimer); thinkingTimer = null; }
    };

    // Hard request timeout — prevent runaway subprocesses from blocking slots
    requestTimer = setTimeout(() => {
      if (!isComplete && subprocess.isRunning()) {
        console.warn(
          `[Timeout] Request exceeded ${REQUEST_TIMEOUT_MS / 1000}s hard limit ` +
          `(pid=${subprocess.pid}, req=${requestId}, output=${totalOutputChars} chars). Killing.`
        );
        const notice = "\n\n[请求超时：已达到最大处理时间限制]";
        const noticeChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{ index: 0, delta: { content: notice }, finish_reason: null }],
        };
        safeWrite(`data: ${JSON.stringify(noticeChunk)}\n\n`);
        subprocess.kill();
      }
    }, REQUEST_TIMEOUT_MS);

    // Thinking indicator — sends a visible "thinking" message periodically
    // so the user knows the model is working, not stuck. First fires at 20s,
    // then repeats every 30s until first content byte arrives.
    const THINKING_DELAY_MS = parseInt(process.env.THINKING_DELAY_MS || "20000", 10);
    const THINKING_REPEAT_MS = parseInt(process.env.THINKING_REPEAT_MS || "30000", 10);
    let thinkingCount = 0;

    const sendThinkingIndicator = () => {
      if (firstByteLogged || isComplete || clientDisconnected || res.writableEnded) {
        return;
      }
      thinkingCount++;
      const elapsed = Math.round((Date.now() - streamStartTime) / 1000);
      const thinkingText = thinkingCount === 1
        ? "💭 思考中…"
        : `💭 仍在思考… (${elapsed}s)`;
      const thinkingChunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: { role: "assistant", content: thinkingText },
          finish_reason: null,
        }],
      };
      const sseData = `data: ${JSON.stringify(thinkingChunk)}\n\n`;
      safeWrite(sseData);
      if (cacheEntry) cacheEntry.chunks.push(sseData);
      isFirst = false;
      hasEmittedText = true;
      console.log(`[Thinking] Indicator #${thinkingCount} for ${requestId} (${elapsed}s elapsed)`);

      // Schedule next indicator
      thinkingTimer = setTimeout(sendThinkingIndicator, THINKING_REPEAT_MS);
    };

    thinkingTimer = setTimeout(sendThinkingIndicator, THINKING_DELAY_MS);

    // First-byte timeout — if Claude CLI doesn't return any content within
    // FIRST_BYTE_TIMEOUT_MS, kill it and return an error. This prevents the
    // common scenario where Opus sits in extended thinking for 2-4 minutes,
    // causing upstream gateways to timeout and the user to see "stuck".
    firstByteTimer = setTimeout(() => {
      if (!firstByteLogged && !isComplete && subprocess.isRunning()) {
        console.warn(
          `[FirstByte] No content received within ${FIRST_BYTE_TIMEOUT_MS / 1000}s ` +
          `(pid=${subprocess.pid}, req=${requestId}). Killing subprocess.`
        );
        const notice = "\n\n[响应超时：模型长时间未返回内容，请重试]";
        const noticeChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{ index: 0, delta: { content: notice }, finish_reason: null }],
        };
        safeWrite(`data: ${JSON.stringify(noticeChunk)}\n\n`);
        subprocess.kill();
      }
    }, FIRST_BYTE_TIMEOUT_MS);

    // Handle actual client disconnect (response stream closed).
    // IMPORTANT: Do NOT resolve() here — keep the concurrency slot occupied
    // until the subprocess actually finishes. Otherwise we leak slots and can
    // exceed MAX_CONCURRENCY with orphaned subprocesses.
    res.on("close", () => {
      cleanup();
      clientDisconnected = true;
      if (!isComplete) {
        console.warn(
          `[Streaming] Client disconnected while subprocess still running (pid=${subprocess.pid}). ` +
          `Will kill after ${DISCONNECT_GRACE_MS / 1000}s grace period.`
        );
        // Start grace period — if subprocess doesn't finish in time, kill it
        disconnectTimer = setTimeout(() => {
          if (!isComplete && subprocess.isRunning()) {
            console.warn(
              `[Streaming] Grace period expired, killing subprocess (pid=${subprocess.pid}, ` +
              `output=${totalOutputChars} chars)`
            );
            subprocess.kill();
          }
        }, DISCONNECT_GRACE_MS);
      } else {
        // Subprocess already finished, safe to resolve
        resolve();
      }
    });

    // Flush any text held in the probe buffer — called on block end
    // and subprocess completion to prevent silent data loss.
    const flushFilter = () => {
      if (blockFilter.phase === "probe" && blockFilter.buf.length > 0) {
        // Undecided probe buffer — check if it has CJK
        const text = blockFilter.buf;
        blockFilter.buf = "";
        if (CJK_RE.test(text)) {
          blockFilter.phase = "pass";
          const chunk = {
            id: `chatcmpl-${requestId}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: lastModel,
            choices: [{
              index: 0,
              delta: { content: text },
              finish_reason: null,
            }],
          };
          const sseData = `data: ${JSON.stringify(chunk)}\n\n`;
          safeWrite(sseData);
          if (cacheEntry) cacheEntry.chunks.push(sseData);
          hasEmittedText = true;
        }
        // No CJK → drop it (likely English narration)
      }
    };

    // When a new text content block starts, flush the previous filter and reset
    subprocess.on("text_block_start", () => {
      flushFilter();
      blockFilter = createBlockFilter();

      if (hasEmittedText) {
        const sepChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{
            index: 0,
            delta: {
              content: "\n\n",
            },
            finish_reason: null,
          }],
        };
        safeWrite(`data: ${JSON.stringify(sepChunk)}\n\n`);
      }
    });

    // Handle streaming content deltas — filtered to suppress English narration
    subprocess.on("content_delta", (event: ClaudeCliStreamEvent) => {
      if (outputTruncated) return; // already truncated, ignore further output

      const delta = event.event.delta;
      const rawText = (delta?.type === "text_delta" && delta.text) || "";
      if (!rawText) return;

      // Track total output for truncation protection
      totalOutputChars += rawText.length;
      if (totalOutputChars > MAX_OUTPUT_CHARS) {
        outputTruncated = true;
        console.warn(
          `[Truncation] Output exceeded ${MAX_OUTPUT_CHARS} chars (pid=${subprocess.pid}, ` +
          `req=${requestId}, total=${totalOutputChars}). Killing subprocess.`
        );
        // Send truncation notice to client
        const notice = "\n\n[输出已截断：内容超出长度限制]";
        const noticeChunk = {
          id: `chatcmpl-${requestId}`,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: lastModel,
          choices: [{ index: 0, delta: { content: notice }, finish_reason: null }],
        };
        safeWrite(`data: ${JSON.stringify(noticeChunk)}\n\n`);
        subprocess.kill();
        return;
      }

      // Run through English narration filter
      const text = filterTextDelta(blockFilter, rawText);
      if (!text) return; // suppressed or still probing

      const chunk = {
        id: `chatcmpl-${requestId}`,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: lastModel,
        choices: [{
          index: 0,
          delta: {
            role: isFirst ? "assistant" : undefined,
            content: text,
          },
          finish_reason: null,
        }],
      };
      const sseData = `data: ${JSON.stringify(chunk)}\n\n`;
      safeWrite(sseData);
      // Record in cache for reconnect replay
      if (cacheEntry) cacheEntry.chunks.push(sseData);
      if (!firstByteLogged) {
        firstByteLogged = true;
        // Cancel first-byte and thinking timers — we got content
        if (firstByteTimer) { clearTimeout(firstByteTimer); firstByteTimer = null; }
        if (thinkingTimer) { clearTimeout(thinkingTimer); thinkingTimer = null; }
        console.log(`[Perf] ${requestId} first-byte=${Date.now() - streamStartTime}ms`);
      }
      isFirst = false;
      hasEmittedText = true;
    });

    // NOTE: Tool call forwarding is disabled — Claude Code handles tools internally
    // via --print mode. OpenClaw would misinterpret internal tool_use as external
    // calls, causing an agentic loop.

    // Handle final assistant message (for model name)
    subprocess.on("assistant", (message: ClaudeCliAssistant) => {
      lastModel = message.message.model;
      if (cacheEntry) cacheEntry.lastModel = lastModel;
    });

    // Flush filter on content block stop to release any held short text
    subprocess.on("content_block_stop", () => {
      flushFilter();
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      isComplete = true;
      flushFilter();
      cleanup();

      // Track CLI outcome for health monitoring
      const errorText = result.is_error ? result.result : "";
      const rateLimited = result.is_error && isRateLimitError(errorText);
      recordCliOutcome({
        timestamp: Date.now(),
        success: !result.is_error,
        isRateLimit: rateLimited,
        errorMessage: result.is_error ? errorText : undefined,
        durationMs: result.duration_ms,
      });

      // Update cache
      if (cacheEntry) {
        cacheEntry.status = "complete";
        cacheEntry.finalResult = result;
        cacheEntry.subprocess = undefined; // release ref
      }
      if (!clientDisconnected && !res.writableEnded) {
        // Send final done chunk with finish_reason and usage data
        const doneChunk = createDoneChunk(requestId, lastModel);
        if (result.usage) {
          doneChunk.usage = {
            prompt_tokens: result.usage.input_tokens || 0,
            completion_tokens: result.usage.output_tokens || 0,
            total_tokens:
              (result.usage.input_tokens || 0) + (result.usage.output_tokens || 0),
          };
        }
        const doneData = `data: ${JSON.stringify(doneChunk)}\n\n`;
        safeWrite(doneData);
        if (cacheEntry) cacheEntry.chunks.push(doneData);
        safeWrite("data: [DONE]\n\n");
        res.end();
      }
      // Don't resolve here — wait for "close" event which always fires after
      // "result" and represents actual subprocess termination.
    });

    subprocess.on("error", (error: Error) => {
      cleanup();
      console.error("[Streaming] Error:", error.message);

      // Track CLI outcome for health monitoring
      recordCliOutcome({
        timestamp: Date.now(),
        success: false,
        isRateLimit: isRateLimitError(error.message),
        errorMessage: error.message,
      });

      if (cacheEntry) {
        cacheEntry.status = "error";
        cacheEntry.errorMessage = error.message;
        cacheEntry.subprocess = undefined;
      }
      if (!clientDisconnected && !res.writableEnded) {
        safeWrite(
          `data: ${JSON.stringify({
            error: { message: error.message, type: "server_error", code: null },
          })}\n\n`
        );
        res.end();
      }
      // Don't resolve here — wait for "close"
    });

    // Subprocess actually terminated — this is the ONLY place we resolve the
    // promise, which means the concurrency slot stays occupied for the entire
    // lifetime of the subprocess, regardless of whether the client is still
    // connected or not. This prevents resource exhaustion from orphaned procs.
    subprocess.on("close", (code: number | null) => {
      cleanup();
      // Clear disconnect grace timer if set
      if (disconnectTimer) {
        clearTimeout(disconnectTimer);
        disconnectTimer = null;
      }
      if (!clientDisconnected && !res.writableEnded) {
        if (code !== 0 && !isComplete) {
          safeWrite(`data: ${JSON.stringify({
            error: { message: `Process exited with code ${code}`, type: "server_error", code: null },
          })}\n\n`);
        }
        safeWrite("data: [DONE]\n\n");
        res.end();
      }
      // Track non-zero exit without result as failure
      if (code !== 0 && !isComplete) {
        recordCliOutcome({
          timestamp: Date.now(),
          success: false,
          isRateLimit: false,
          errorMessage: `Process exited with code ${code}`,
          exitCode: code,
        });
      }

      console.log(
        `[Streaming] Subprocess done (pid=${subprocess.pid}, req=${requestId}, ` +
        `output=${totalOutputChars} chars, truncated=${outputTruncated})`
      );
      // NOW release the concurrency slot (via finally block in handleChatCompletions)
      resolve();
    });

    // Monitor stderr for rate limit signals that don't surface as JSON errors
    subprocess.on("stderr", (text: string) => {
      if (isRateLimitError(text)) {
        recordCliOutcome({
          timestamp: Date.now(),
          success: false,
          isRateLimit: true,
          errorMessage: `[stderr] ${text.slice(0, 300)}`,
        });
      }
    });

    // Start the subprocess
    subprocess.start(cliInput.prompt, {
      model: cliInput.model,
      sessionId: cliInput.sessionId,
    }).catch((err) => {
      console.error("[Streaming] Subprocess start error:", err);
      reject(err);
    });
  });
}

/**
 * Handle non-streaming response
 */
async function handleNonStreamingResponse(
  res: Response,
  subprocess: ClaudeSubprocess,
  cliInput: ReturnType<typeof openaiToCli>,
  requestId: string,
  cacheEntry?: CachedResult
): Promise<void> {
  return new Promise((resolve) => {
    let finalResult: ClaudeCliResult | null = null;
    let clientDisconnected = false;
    let disconnectTimer: NodeJS.Timeout | null = null;

    // Hard request timeout for non-streaming
    const requestTimer = setTimeout(() => {
      if (!finalResult && subprocess.isRunning()) {
        console.warn(
          `[Timeout] Non-streaming request exceeded ${REQUEST_TIMEOUT_MS / 1000}s hard limit ` +
          `(pid=${subprocess.pid}, req=${requestId}). Killing.`
        );
        subprocess.kill();
      }
    }, REQUEST_TIMEOUT_MS);

    // Track client disconnect — start grace period then kill subprocess
    res.on("close", () => {
      if (!finalResult) {
        clientDisconnected = true;
        console.warn(
          `[NonStreaming] Client disconnected while subprocess still running (pid=${subprocess.pid}, req=${requestId}). ` +
          `Will kill after ${DISCONNECT_GRACE_MS / 1000}s grace period.`
        );
        disconnectTimer = setTimeout(() => {
          if (!finalResult && subprocess.isRunning()) {
            console.warn(
              `[NonStreaming] Grace period expired, killing subprocess (pid=${subprocess.pid})`
            );
            subprocess.kill();
          }
        }, DISCONNECT_GRACE_MS);
      }
    });

    subprocess.on("result", (result: ClaudeCliResult) => {
      finalResult = result;

      // Track CLI outcome for health monitoring
      const errorText = result.is_error ? result.result : "";
      const rateLimited = result.is_error && isRateLimitError(errorText);
      recordCliOutcome({
        timestamp: Date.now(),
        success: !result.is_error,
        isRateLimit: rateLimited,
        errorMessage: result.is_error ? errorText : undefined,
        durationMs: result.duration_ms,
      });

      if (cacheEntry) {
        cacheEntry.status = "complete";
        cacheEntry.finalResult = result;
        cacheEntry.subprocess = undefined;
      }
    });

    subprocess.on("error", (error: Error) => {
      console.error(`[NonStreaming] Error (pid=${subprocess.pid}, req=${requestId}):`, error.message);

      // Track CLI outcome for health monitoring
      recordCliOutcome({
        timestamp: Date.now(),
        success: false,
        isRateLimit: isRateLimitError(error.message),
        errorMessage: error.message,
      });

      if (cacheEntry) {
        cacheEntry.status = "error";
        cacheEntry.errorMessage = error.message;
        cacheEntry.subprocess = undefined;
      }
      if (!clientDisconnected && !res.headersSent) {
        res.status(500).json({
          error: {
            message: error.message,
            type: "server_error",
            code: null,
          },
        });
      }
      // Don't resolve here — wait for "close"
    });

    subprocess.on("close", (code: number | null) => {
      clearTimeout(requestTimer);
      if (disconnectTimer) {
        clearTimeout(disconnectTimer);
        disconnectTimer = null;
      }
      if (clientDisconnected) {
        console.log(
          `[NonStreaming] Subprocess finished after client disconnect (pid=${subprocess.pid}, ` +
          `code=${code}, result=${finalResult ? "yes" : "no"})`
        );
      } else if (finalResult) {
        res.json(cliResultToOpenai(finalResult, requestId));
      } else if (!res.headersSent) {
        res.status(500).json({
          error: {
            message: `Claude CLI exited with code ${code} without response`,
            type: "server_error",
            code: null,
          },
        });
      }
      resolve();
    });

    // Monitor stderr for rate limit signals
    subprocess.on("stderr", (text: string) => {
      if (isRateLimitError(text)) {
        recordCliOutcome({
          timestamp: Date.now(),
          success: false,
          isRateLimit: true,
          errorMessage: `[stderr] ${text.slice(0, 300)}`,
        });
      }
    });

    // Start the subprocess
    subprocess
      .start(cliInput.prompt, {
        model: cliInput.model,
        sessionId: cliInput.sessionId,
      })
      .catch((error) => {
        if (!clientDisconnected && !res.headersSent) {
          res.status(500).json({
            error: {
              message: error.message,
              type: "server_error",
              code: null,
            },
          });
        }
        resolve();
      });
  });
}

/**
 * Handle GET /v1/models
 *
 * Returns available models
 */
export function handleModels(_req: Request, res: Response): void {
  const now = Math.floor(Date.now() / 1000);
  const modelIds = [
    "claude-opus-4",
    "claude-opus-4-6",
    "claude-sonnet-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-4",
    "claude-haiku-4-5",
  ];
  res.json({
    object: "list",
    data: modelIds.map((id) => ({
      id,
      object: "model",
      owned_by: "anthropic",
      created: now,
    })),
  });
}

/**
 * Handle GET /health
 *
 * Health check endpoint
 */
const serverStartTime = Date.now();

export function handleHealth(_req: Request, res: Response): void {
  const mem = process.memoryUsage();
  const cliHealth = getCliHealthSummary();

  // Top-level status reflects CLI health — if CLI is rate limited or failing,
  // the whole service is effectively down even if the proxy process is fine.
  const overallStatus = cliHealth.status === "ok" ? "ok" : cliHealth.status;

  res.json({
    status: overallStatus,
    provider: "claude-code-cli",
    timestamp: new Date().toISOString(),
    uptime_seconds: Math.floor((Date.now() - serverStartTime) / 1000),
    cli: cliHealth,
    concurrency: {
      available: concurrency.available,
      queued: concurrency.queueLength,
      active: activeSubprocesses.size,
    },
    memory: {
      heap_used_mb: Math.round(mem.heapUsed / 1024 / 1024),
      rss_mb: Math.round(mem.rss / 1024 / 1024),
    },
    cache: {
      entries: resultCache.size,
    },
    limits: {
      max_output_chars: MAX_OUTPUT_CHARS,
      request_timeout_ms: REQUEST_TIMEOUT_MS,
      disconnect_grace_ms: DISCONNECT_GRACE_MS,
      first_byte_timeout_ms: FIRST_BYTE_TIMEOUT_MS,
      thinking_delay_ms: parseInt(process.env.THINKING_DELAY_MS || "20000", 10),
    },
    probe: lastProbeResult
      ? {
          last_check: new Date(lastProbeResult.timestamp).toISOString(),
          ok: lastProbeResult.ok,
          duration_ms: lastProbeResult.durationMs,
          error: lastProbeResult.error || null,
        }
      : null,
  });
}
