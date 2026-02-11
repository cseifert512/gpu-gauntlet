/**
 * bench-ci.mjs — Run the /benchmark page via puppeteer-core + local Chrome.
 *
 * Exit codes:
 *   0  = PASS and GPU < 20 ms
 *   1  = PASS but GPU >= 20 ms  (expected for now)
 *  10  = validation failed
 *  11  = could not parse GPU ms
 *  99  = automation failure (crash, timeout, no output, etc.)
 *
 * Retries once automatically on failure.
 */

import { spawn, execSync } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import waitOn from "wait-on";
import puppeteer from "puppeteer-core";

/* ─── configuration ─── */

const PORT = process.env.PORT || "3456";
const URL  = `http://localhost:${PORT}/benchmark?auto=1`;

const BAD_ADAPTER_SUBSTRINGS = [
  "Basic Render Driver",
  "Microsoft Basic",
  "llvmpipe",
  "SwiftShader",
];

/* ─── find Chrome ─── */

function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;

  const candidates = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    path.join(process.env.LOCALAPPDATA || "", "Google\\Chrome\\Application\\chrome.exe"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return candidates[0]; // will fail later with a clear message
}

const CHROME_PATH = findChrome();

/* ─── helpers ─── */

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Kill every process listening on `port` (Windows-safe). */
function killPort(port) {
  try {
    if (process.platform === "win32") {
      const out = execSync(
        `netstat -ano | findstr :${port} | findstr LISTENING`,
        { encoding: "utf8", stdio: ["pipe", "pipe", "ignore"] },
      );
      const pids = [
        ...new Set(
          out
            .trim()
            .split(/\r?\n/)
            .map((l) => l.trim().split(/\s+/).pop())
            .filter(Boolean),
        ),
      ];
      for (const pid of pids) {
        try {
          execSync(`taskkill /PID ${pid} /T /F`, { stdio: "ignore" });
        } catch { /* already dead */ }
      }
    } else {
      execSync(`lsof -ti:${port} | xargs kill -9 2>/dev/null`, {
        stdio: "ignore",
      });
    }
  } catch { /* nothing listening — fine */ }
}

/** Kill a child-process tree (Windows: taskkill /T). */
function killProc(proc) {
  if (!proc || !proc.pid) return;
  try {
    if (process.platform === "win32") {
      execSync(`taskkill /PID ${proc.pid} /T /F`, { stdio: "ignore" });
    } else {
      process.kill(-proc.pid, "SIGTERM"); // negative PID = process group
    }
  } catch { /* already dead */ }
}

/* ─── single benchmark attempt ─── */

async function attemptOnce() {
  /* 0. Pre-flight: Chrome exists? */
  if (!fs.existsSync(CHROME_PATH)) {
    throw new Error(
      `Chrome not found at: ${CHROME_PATH}\n` +
        "Set CHROME_PATH env var to your chrome.exe location.",
    );
  }

  /* 1. Kill anything already on our port */
  killPort(PORT);
  await sleep(1500);

  /* 2. Start Next dev server */
  const dev = spawn("npx", ["next", "dev", "-p", PORT], {
    shell: true,
    stdio: "pipe",
    windowsHide: true,
    env: { ...process.env, BROWSER: "none", NODE_ENV: "development" },
  });

  // Pipe server output so we can see compilation messages
  dev.stdout?.on("data", (d) => process.stdout.write(d));
  dev.stderr?.on("data", (d) => process.stderr.write(d));

  let browser = null;
  try {
    /* 3. Wait for server */
    await waitOn({
      resources: [`http-get://127.0.0.1:${PORT}`],
      timeout: 120_000,
      interval: 1000,
      validateStatus: (s) => s >= 200 && s < 400,
    });
    console.log(`\n[bench-ci] Dev server ready on port ${PORT}`);

    /* 4. Launch Chrome — use a temp profile to avoid lock-file conflicts */
    const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), "bench-chrome-"));
    browser = await puppeteer.launch({
      headless: false,
      executablePath: CHROME_PATH,
      userDataDir,
      args: [
        "--window-size=1280,800",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--enable-unsafe-webgpu",
        "--disable-dawn-features=disallow_unsafe_apis",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
      ],
      defaultViewport: { width: 1280, height: 800 },
    });

    const page = await browser.newPage();

    /* 5. Collect console output */
    let adapterLine = null;
    const benchLines = [];
    let benchmarkComplete = false;

    page.on("console", (msg) => {
      const text = msg.text();

      if (text.startsWith("ADAPTER:")) {
        adapterLine = text;
        console.log(`[page] ${text}`);
      }
      if (
        text.includes("DOF:") &&
        text.includes("GPU:") &&
        text.includes("Valid:")
      ) {
        benchLines.push(text);
        console.log(`[page] ${text}`);
      }
      if (text === "BENCHMARK_COMPLETE") {
        benchmarkComplete = true;
        console.log("[bench-ci] Benchmark run finished.");
      }
    });

    page.on("pageerror", (err) => {
      console.error("[page error]", err.message);
    });

    /* 6. Navigate */
    await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

    /* 7. Wait for completion or timeout */
    const TIMEOUT_MS = 300_000; // 5 min
    const start = Date.now();

    while (Date.now() - start < TIMEOUT_MS) {
      if (benchmarkComplete) break;
      await sleep(1000);
    }

    /* 8. Parse results */
    // Prefer the largest-DOF line (should be last)
    const target =
      benchLines.find((l) => l.includes("61,009") || l.includes("61009")) ||
      benchLines[benchLines.length - 1];

    // Log to file
    fs.mkdirSync("benchmarks", { recursive: true });
    fs.appendFileSync(
      path.join("benchmarks", "bench-log.txt"),
      `[${new Date().toISOString()}] ${adapterLine ?? "ADAPTER: (missing)"} | ${target ?? "(no target line)"}\n`,
    );

    /* 9. Validate adapter */
    if (!adapterLine) {
      const e = new Error(
        "Missing ADAPTER line. The /benchmark page must console.log('ADAPTER: ...').",
      );
      e.code = 99;
      throw e;
    }
    if (BAD_ADAPTER_SUBSTRINGS.some((s) => adapterLine.includes(s))) {
      const e = new Error(`Bad adapter detected: ${adapterLine}`);
      e.code = 99;
      throw e;
    }

    /* 10. Validate benchmark output */
    if (!target) {
      const e = new Error("No benchmark output line captured.");
      e.code = 99;
      throw e;
    }

    const validPass = /Valid:\s*PASS/i.test(target);
    const gpuMatch = target.match(/GPU:\s*([0-9.]+)\s*ms/i);
    const gpuMs = gpuMatch ? Number(gpuMatch[1]) : Number.NaN;

    if (!validPass) {
      const e = new Error(`Benchmark did not PASS validation. Line: ${target}`);
      e.code = 10;
      throw e;
    }
    if (!Number.isFinite(gpuMs)) {
      const e = new Error(
        `Could not parse GPU ms from benchmark line: ${target}`,
      );
      e.code = 11;
      throw e;
    }

    console.log(
      `[bench-ci] Result — GPU: ${gpuMs.toFixed(1)} ms | Valid: PASS | ${gpuMs < 20 ? "TARGET MET ✓" : "target not met yet"}`,
    );

    return gpuMs < 20 ? 0 : 1;
  } finally {
    /* Cleanup: always close browser + kill dev server */
    if (browser) {
      try { await browser.close(); } catch { /* ignore */ }
    }
    killProc(dev);
    await sleep(500);
    killPort(PORT); // belt-and-suspenders

    // Clean up temp Chrome profile (best-effort)
    try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
}

/* ─── main: one retry on failure ─── */

(async () => {
  try {
    const code = await attemptOnce();
    process.exit(code);
  } catch (e1) {
    const c1 = e1?.code;
    if (c1 === 10 || c1 === 11) {
      // Deterministic failures — no point retrying
      console.error(`[bench-ci] ${e1.message}`);
      process.exit(c1);
    }

    console.error(
      `[bench-ci] Failed, retrying once...\n  ${e1?.message ?? e1}`,
    );
    await sleep(3000);

    try {
      const code = await attemptOnce();
      process.exit(code);
    } catch (e2) {
      console.error(
        `[bench-ci] Failed twice.\n  ${e2?.message ?? e2}`,
      );
      process.exit(e2?.code ?? 99);
    }
  }
})();
