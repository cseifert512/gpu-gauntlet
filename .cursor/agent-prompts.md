# Agent Session Prompts
Copy-paste one at a time into new Cursor Agent chats (Opus 4.6).
Each session reads .cursor/scratchpad.md so it picks up where the last left off.

---

## Session 1: Diagnose — Why Does Validation Fail?

You are optimizing a WebGPU FEM plate solver. Read these files first:
- benchmarks/bench-log.txt (latest results)
- .cursor/scratchpad.md (progress log)
- src/lib/plate/gpu/solver.ts (GPU PCG solver)
- src/lib/plate/gpu/shaders/index.ts (all shader source)
- src/lib/plate/pcg.ts (CPU reference solver)

Current status: `npm run bench:ci` exits with code 10 (validation FAIL).
The GPU solver returns wrong results — relative error vs CPU exceeds 0.01%.
DOF is only ~4107 for the target config (expected ~61,009 — mesh may need refinement).

YOUR TASK (this session):
1. Run `npm run bench:ci` once to see current output.
2. Compare the GPU solver (src/lib/plate/gpu/solver.ts) to the CPU solver (src/lib/plate/pcg.ts) step by step. Identify where they diverge.
3. Check: Are boundary conditions applied identically? Is the preconditioner the same? Are the shader operations (dot product, axpy, etc.) mathematically equivalent?
4. Write your diagnosis to .cursor/scratchpad.md under a "## Session 1: Diagnosis" heading.
5. If the fix is obvious, implement it and run `npm run bench:ci` to verify.

Do NOT change the CPU solver. It is the ground truth reference.
Do NOT change exit codes or the bench-ci harness.
Commit any changes with: git add -A && git commit -m "fix: <what you changed>"

---

## Session 2: Fix Validation — Make GPU Match CPU

You are optimizing a WebGPU FEM plate solver. Read these files first:
- .cursor/scratchpad.md (contains diagnosis from previous session)
- benchmarks/bench-log.txt (result history)
- src/lib/plate/gpu/solver.ts
- src/lib/plate/gpu/shaders/index.ts

Current status: GPU solver validation FAILS (exit code 10).
The previous session diagnosed the root cause — check .cursor/scratchpad.md.

YOUR TASK:
1. Fix the GPU solver so it produces results matching the CPU solver within 0.01% relative error.
2. Run `npm run bench:ci` after each change to test.
3. Common issues to check:
   - Boundary condition zeroing (constrained DOFs must be zero in solution AND in residual)
   - Preconditioner inversion (block diagonal — is it computed the same way?)
   - Convergence tolerance (must be 1e-6 same as CPU)
   - Floating point: GPU shaders use f32, CPU might use f64 — tolerance may need slight loosening
4. Once Valid: PASS appears in the bench output, log the result to .cursor/scratchpad.md.
5. If you cannot get PASS, document what you tried and what the remaining error is.

Commit after each meaningful change: git add -A && git commit -m "fix: <description>"

---

## Session 3: Hit 61K DOF

Read first:
- .cursor/scratchpad.md (progress so far)
- benchmarks/bench-log.txt
- src/app/benchmark/page.tsx (TEST_CONFIGS and mesh generation)
- src/lib/plate/mesher.ts and src/lib/plate/mesher-unstructured.ts

Current status: Validation should be PASS now (or close). But DOF is only ~4107
for the "Target" config. The goal is ~61,009 DOF.

The TEST_CONFIGS use meshSize=0.28 for a 10x10m plate, which only produces ~4K DOF.
To get ~61K DOF with 3 DOF/node, we need ~20,336 nodes -> meshSize around 0.07.

YOUR TASK:
1. Update TEST_CONFIGS in src/app/benchmark/page.tsx so the Target config
   produces approximately 61,009 DOF. Adjust meshSize accordingly.
   Keep the smaller configs for quick smoke tests.
2. Run `npm run bench:ci` to verify the DOF count is close to 61,009.
3. Ensure validation still passes at the larger mesh size.
4. Log the results to .cursor/scratchpad.md.

Note: GPU time will be very slow at 61K DOF. That's expected.
We just need Valid: PASS at this DOF count.

Commit: git add -A && git commit -m "feat: adjust mesh to hit 61K DOF target"

---

## Session 4: First Optimization Pass

Read first:
- .cursor/scratchpad.md
- benchmarks/bench-log.txt
- src/lib/plate/gpu/solver.ts
- src/lib/plate/gpu/pipelines.ts
- src/lib/plate/gpu/buffers.ts
- src/lib/plate/gpu/shaders/ (all .wgsl files)

Current status: Validation PASS at ~61K DOF, but GPU time is very slow (seconds).
Target: GPU < 20ms.

YOUR TASK — find and fix the biggest performance bottlenecks:
1. Run `npm run bench:ci` to get a baseline time.
2. Check for these common WebGPU performance killers:
   - Excessive GPU<->CPU readbacks (readBuffer/mapAsync) inside the PCG loop
   - Creating new buffers or pipelines every iteration instead of reusing
   - Missing workgroup size tuning (should be 256 for NVIDIA)
   - Unnecessary pipeline barriers or redundant dispatches
   - Convergence check reading back a value every iteration (batch it: check every 10-50 iters)
3. Fix the top 2-3 issues. Run bench:ci after each fix.
4. Log before/after times to .cursor/scratchpad.md.

The single biggest win is usually eliminating per-iteration GPU->CPU readbacks for the
convergence check. Consider running a fixed number of iterations, or checking convergence
every N iterations, or using a GPU-side convergence flag.

Commit after each improvement: git add -A && git commit -m "perf: <description>"

---

## Session 5+: Deep Optimization (reuse this prompt)

Read first:
- .cursor/scratchpad.md (full history of changes and timings)
- benchmarks/bench-log.txt (all benchmark runs)
- src/lib/plate/gpu/solver.ts
- src/lib/plate/gpu/shaders/ (all .wgsl files)

Current status: Check .cursor/scratchpad.md for latest GPU time.
Target: 61K DOF, Valid PASS, GPU < 20ms.

YOUR TASK — continue optimizing. Pick the highest-impact item:
- [ ] Fuse multiple shader dispatches into fewer passes
- [ ] Use shared/workgroup memory in reduction shaders (dot product, norm)
- [ ] Overlap compute and memory transfers with double-buffering
- [ ] Tune workgroup sizes (try 64, 128, 256) — profile which is fastest
- [ ] Reduce element stiffness matrix assembly (precompute on CPU, upload once)
- [ ] Use f16 where precision allows (storage buffers, intermediate values)
- [ ] Minimize the number of dispatch calls per PCG iteration
- [ ] Consider matrix-free approach if assembly is the bottleneck

After each change:
1. Run `npm run bench:ci`
2. If GPU time improved, commit: git add -A && git commit -m "perf: <description>"
3. If GPU time got worse or validation broke, revert: git checkout -- .
4. Log result to .cursor/scratchpad.md

Keep going until you've exhausted ideas or hit the target.

