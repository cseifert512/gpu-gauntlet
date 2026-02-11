# Overnight Benchmark Rules

## Commands
- **Single run:** `npm run bench:ci`
- **Overnight loop:** `npm run overnight`
- Never run the benchmark manually in the browser for CI purposes; always use the scripts above.

## Do NOT
- Touch the CPU reference solver unless strictly needed for logging or validation.
- Modify the exit-code contract in `scripts/bench-ci.mjs` (0/1/10/11/99).
- Change the port (3456) without updating both `bench-ci.mjs` and any running sessions.

## DO
- Log every iteration result — scripts already append to `benchmarks/bench-log.txt` and `.cursor/scratchpad.md`.
- Commit progress on the `overnight-grind` branch after meaningful improvements.
- Check `benchmarks/bench-log.txt` for historical GPU times.
- If `bench:ci` exits with 99 twice in a row, investigate before re-launching.

## Exit codes (bench:ci)
| Code | Meaning |
|------|---------|
| 0 | PASS and GPU < 20 ms — target met! |
| 1 | PASS but GPU ≥ 20 ms (expected until optimised) |
| 10 | Validation failed |
| 11 | Could not parse GPU ms |
| 99 | Automation failure (page crash, no output, etc.) |

