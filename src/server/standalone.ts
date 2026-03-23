#!/usr/bin/env node
/**
 * Standalone server for testing without Clawdbot
 *
 * Usage:
 *   npm run start
 *   # or
 *   node dist/server/standalone.js [port]
 */

import { startServer, stopServer } from "./index.js";
import { verifyClaude, verifyAuth } from "../subprocess/manager.js";
import { activeSubprocesses } from "./routes.js";

const DEFAULT_PORT = 3456;

async function main(): Promise<void> {
  console.log("Claude Code CLI Provider - Standalone Server");
  console.log("============================================\n");

  // Parse port from command line
  const port = parseInt(process.argv[2] || String(DEFAULT_PORT), 10);
  if (isNaN(port) || port < 1 || port > 65535) {
    console.error(`Invalid port: ${process.argv[2]}`);
    process.exit(1);
  }

  // Verify Claude CLI
  console.log("Checking Claude CLI...");
  const cliCheck = await verifyClaude();
  if (!cliCheck.ok) {
    console.error(`Error: ${cliCheck.error}`);
    process.exit(1);
  }
  console.log(`  Claude CLI: ${cliCheck.version || "OK"}`);

  // Verify authentication
  console.log("Checking authentication...");
  const authCheck = await verifyAuth();
  if (!authCheck.ok) {
    console.error(`Error: ${authCheck.error}`);
    console.error("Please run: claude auth login");
    process.exit(1);
  }
  console.log("  Authentication: OK\n");

  // Start server
  try {
    await startServer({ port });
    console.log("\nServer ready. Test with:");
    console.log(`  curl -X POST http://localhost:${port}/v1/chat/completions \\`);
    console.log(`    -H "Content-Type: application/json" \\`);
    console.log(`    -d '{"model": "claude-sonnet-4", "messages": [{"role": "user", "content": "Hello!"}]}'`);
    console.log("\nPress Ctrl+C to stop.\n");
  } catch (err) {
    console.error("Failed to start server:", err);
    process.exit(1);
  }

  // Memory monitoring — log heap usage every 60s, warn if over threshold
  const MEMORY_LOG_INTERVAL = 60_000;
  const MEMORY_WARN_MB = parseInt(process.env.MEMORY_WARN_MB || "512", 10);
  setInterval(() => {
    const mem = process.memoryUsage();
    const heapMB = Math.round(mem.heapUsed / 1024 / 1024);
    const rssMB = Math.round(mem.rss / 1024 / 1024);
    if (heapMB > MEMORY_WARN_MB) {
      console.warn(`[Memory] WARNING: heap=${heapMB}MB rss=${rssMB}MB (threshold=${MEMORY_WARN_MB}MB) active=${activeSubprocesses.size}`);
    } else if (process.env.DEBUG) {
      console.log(`[Memory] heap=${heapMB}MB rss=${rssMB}MB active=${activeSubprocesses.size}`);
    }
  }, MEMORY_LOG_INTERVAL).unref(); // unref() so it doesn't prevent clean shutdown

  // Handle graceful shutdown — wait for in-flight subprocesses
  let shutdownInProgress = false;
  const shutdown = async () => {
    if (shutdownInProgress) return;
    shutdownInProgress = true;
    const active = activeSubprocesses.size;
    if (active > 0) {
      console.log(`\nGraceful shutdown: waiting for ${active} subprocess(es) to finish (max 60s)...`);
      const start = Date.now();
      while (activeSubprocesses.size > 0 && Date.now() - start < 60_000) {
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (activeSubprocesses.size > 0) {
        console.warn(`[Shutdown] ${activeSubprocesses.size} subprocess(es) still running after 60s, killing...`);
        for (const sub of activeSubprocesses) {
          sub.kill("SIGKILL");
        }
      }
    }
    console.log("\nShutting down...");
    await stopServer();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((err) => {
  console.error("Unexpected error:", err);
  process.exit(1);
});
