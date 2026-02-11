/**
 * overnight.mjs — Loop `npm run bench:ci` up to MAX_ITERS times.
 *
 * Stops on:
 *   - exit code 0  → target met (GPU < 20 ms + PASS)
 *   - exit code 99 → automation failure (don't keep looping)
 *   - MAX_ITERS reached
 *
 * Appends a row per iteration to .cursor/scratchpad.md and benchmarks/bench-log.txt.
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const MAX_ITERS = Number(process.env.MAX_ITERS) || 200;
const SCRATCHPAD = path.join(".cursor", "scratchpad.md");

const EXIT_LABELS = {
  0:  "✅ PASS <20ms — TARGET MET",
  1:  "⏳ PASS ≥20ms",
  10: "❌ Validation FAIL",
  11: "❌ GPU ms parse error",
  99: "🛑 Automation failure",
};

function exitLabel(code) {
  return EXIT_LABELS[code] ?? `⚠️ Unknown exit ${code}`;
}

function appendScratchpad(line) {
  fs.mkdirSync(path.dirname(SCRATCHPAD), { recursive: true });
  fs.appendFileSync(SCRATCHPAD, line + "\n");
}

// ── Ensure scratchpad header exists ──
if (!fs.existsSync(SCRATCHPAD) || !fs.readFileSync(SCRATCHPAD, "utf8").includes("Overnight Benchmark Log")) {
  appendScratchpad("# Overnight Benchmark Log\n");
  appendScratchpad("| # | Timestamp | Exit | Status |");
  appendScratchpad("|---|-----------|------|--------|");
}

console.log(`\n╔══════════════════════════════════════╗`);
console.log(`║   OVERNIGHT BENCHMARK LOOP           ║`);
console.log(`║   Max iterations: ${String(MAX_ITERS).padStart(4)}              ║`);
console.log(`╚══════════════════════════════════════╝\n`);

let lastGpuMs = null;

for (let i = 1; i <= MAX_ITERS; i++) {
  const ts = new Date().toISOString();
  console.log(`\n═══ Iteration ${i}/${MAX_ITERS} at ${ts} ═══\n`);

  const result = spawnSync("npm", ["run", "bench:ci"], {
    stdio: "inherit",
    shell: true,
    timeout: 600_000, // 10 min hard timeout per iteration
    env: { ...process.env },
  });

  const code = result.status ?? 99;
  const label = exitLabel(code);
  const row = `| ${i} | ${ts} | ${code} | ${label} |`;

  appendScratchpad(row);
  console.log(`\n>>> Iteration ${i} result: exit=${code}  ${label}`);

  if (code === 0) {
    const msg = `\n🎉 **TARGET MET at iteration ${i}!** GPU < 20 ms with valid PASS.`;
    appendScratchpad(msg);
    console.log(msg);
    process.exit(0);
  }

  if (code === 99) {
    const msg = `\n🛑 **STOPPED at iteration ${i}**: automation failure (exit 99). Not safe to continue.`;
    appendScratchpad(msg);
    console.error(msg);
    process.exit(99);
  }

  // Brief pause between iterations to let GPU cool / context reset
  console.log("Pausing 5s before next iteration...");
  spawnSync("node", ["-e", "setTimeout(()=>{},5000)"], { stdio: "ignore" });
}

const msg = `\n⏹ Reached ${MAX_ITERS} iterations without hitting target. Best so far logged in benchmarks/bench-log.txt.`;
appendScratchpad(msg);
console.log(msg);
process.exit(1);

