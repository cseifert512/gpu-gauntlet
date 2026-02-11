/**
 * WebGPU-accelerated plate solver.
 *
 * Public API for GPU acceleration with automatic CPU fallback.
 *
 * Usage:
 * ```typescript
 * import { solveGPU, isWebGPUAvailable } from './gpu';
 *
 * // Check availability
 * if (isWebGPUAvailable()) {
 *   console.log('GPU acceleration available');
 * }
 *
 * // Solve (automatically falls back to CPU if needed)
 * const result = await solveGPU(mesh, material, coloring, F, constrainedDOFs, {
 *   tolerance: 1e-8,
 *   maxIterations: 1000,
 * });
 *
 * console.log(`Solved in ${result.gpuTimeMs}ms, GPU: ${result.usedGPU}`);
 * ```
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
