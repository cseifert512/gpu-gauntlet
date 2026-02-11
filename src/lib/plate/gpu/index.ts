/**
 * WebGPU-accelerated plate solver.
 *
 * Public API for GPU acceleration with automatic CPU fallback.
 * Achieves 100k DOF plate bending solve in ~13ms on consumer GPUs.
 *
 * Usage (fast path — prepare once, solve many times):
 * ```typescript
 * import { prepareGPUSolver, solveGPU, destroyGPUSolverContext } from './gpu';
 *
 * // Setup (once per geometry, ~50ms)
 * const ctx = await prepareGPUSolver(mesh, material, coloring, constrainedDOFs, blockDiagInv);
 *
 * // Solve (per load case, ~13ms for 100k DOF)
 * const result = await solveGPU(mesh, material, coloring, F, constrainedDOFs, {
 *   maxIterations: 25,
 *   preparedContext: ctx,
 *   precomputedBlockDiagInv: blockDiagInv,
 * });
 *
 * console.log(`${result.gpuTimeMs.toFixed(1)}ms, GPU: ${result.usedGPU}`);
 *
 * // Cleanup
 * destroyGPUSolverContext(ctx);
 * ```
 *
 * Usage (simple path — one-shot solve, includes setup overhead):
 * ```typescript
 * const result = await solveGPU(mesh, material, coloring, F, constrainedDOFs, {
 *   maxIterations: 25,
 * });
 * ```
 *
 * See ARCHITECTURE.md for the full technical documentation and integration guide.
 */

// Context management
export { isWebGPUAvailable, initGPU, destroyGPU, resetGPUCache } from './context';
export type { GPUContext } from './context';

// Buffer management
export {
  createPlateBuffers,
  destroyPlateBuffers,
  uploadToBuffer,
  readFromBuffer,
} from './buffers';
export type { PlateGPUBuffers } from './buffers';

// Pipeline management
export { createPipelines, GPUDispatcher, createParamsBuffer } from './pipelines';
export type { PlatePipelines } from './pipelines';

// Main GPU solver
export { solveGPU, clearPipelineCache, prepareGPUSolver, destroyGPUSolverContext } from './solver';
export type { GPUSolverOptions, GPUSolveResult, GPUSolverContext } from './solver';

// CPU fallback (for direct use or testing)
export { solveCPU, compareSolutions } from './fallback';
export type { CPUSolverOptions, CPUSolveResult } from './fallback';

// Single-color K·x test utilities
export { applySingleColorKx, applySingleColorKxCPU, compareResults } from './single_color_test';
