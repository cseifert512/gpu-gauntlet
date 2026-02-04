/**
 * GPU-accelerated PCG solver for plate bending.
 *
 * Implements matrix-free K·x using element coloring for conflict-free parallelism.
 * Multi-color dispatch: each color batch processes elements that don't share nodes,
 * enabling simultaneous writes without race conditions.
 *
 * PCG Algorithm (on GPU):
 * 1. r = b - K·x (initial x=0, so r=b)
 * 2. z = M⁻¹·r (Jacobi preconditioner)
 * 3. p = z
 * 4. Iterate:
 *    - Ap = K·p (multi-color dispatch)
 *    - α = (r·z) / (p·Ap)
 *    - x = x + α·p
 *    - r = r - α·Ap
 *    - Check convergence
 *    - z = M⁻¹·r
 *    - β = (r_new·z_new) / (r·z)
 *    - p = z + β·p
 */

import type { GPUContext } from './context';
import type { PlateGPUBuffers } from './buffers';
import type { PlatePipelines } from './pipelines';
import type { PlateMesh, PlateMaterial, ElementColoring } from '../types';
import { initGPU, isWebGPUAvailable } from './context';
import { createPlateBuffers, destroyPlateBuffers, readFromBuffer, uploadToBuffer } from './buffers';
import { createPipelines, createParamsBuffer } from './pipelines';
import { computeDiagonal, applyBCsToDiagonal, computeBlockDiagonal, invertBlockDiagonal } from '../solver';

/**
 * GPU solver options.
 */
export interface GPUSolverOptions {
  /** Convergence tolerance (default 1e-8) */
  tolerance?: number;
  /** Max iterations (default 1000) */
  maxIterations?: number;
  /** Force CPU fallback even if GPU available */
  forceCPU?: boolean;
}

/**
 * GPU solve result.
 */
export interface GPUSolveResult {
  /** Solution vector [w0,θx0,θy0, w1,θx1,θy1, ...] */
  solution: Float32Array;
  /** Number of iterations */
  iterations: number;
  /** Final residual norm */
  finalResidual: number;
  /** Whether solution converged */
  converged: boolean;
  /** GPU solve time in milliseconds */
  gpuTimeMs: number;
  /** Whether GPU was used (vs CPU fallback) */
  usedGPU: boolean;
}

/**
 * Cached pipelines (avoid recompilation)
 */
let cachedPipelines: PlatePipelines | null = null;

/**
 * Main GPU solver function.
 *
 * Solves K·x = F using GPU-accelerated PCG with automatic CPU fallback.
 *
 * @param mesh - Plate mesh
 * @param material - Material properties
 * @param coloring - Element coloring for conflict-free assembly
 * @param F - Load vector (right-hand side)
 * @param constrainedDOFs - Set of constrained DOF indices
 * @param options - Solver options
 * @returns Solution and statistics
 */
export async function solveGPU(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions = {}
): Promise<GPUSolveResult> {
  // Check for forced CPU or unavailable GPU
  if (options.forceCPU || !isWebGPUAvailable()) {
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  }

  const ctx = await initGPU();
  if (!ctx) {
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  }

  try {
    return await solveGPUInternal(ctx, mesh, material, coloring, F, constrainedDOFs, options);
  } catch (e) {
    console.error('GPU solve failed, falling back to CPU:', e);
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  }
}

/**
 * Internal GPU solve implementation.
 */
async function solveGPUInternal(
  ctx: GPUContext,
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions
): Promise<GPUSolveResult> {
  const { device } = ctx;
  const startTime = performance.now();

  const tolerance = options.tolerance ?? 1e-6;
  const maxIterations = options.maxIterations ?? 1000;
  const dofCount = mesh.nodeCount * 3;

  // Compute scalar diagonal preconditioner (fallback)
  const diagonal = computeDiagonal(mesh, material);
  applyBCsToDiagonal(diagonal, constrainedDOFs);

  // Compute block diagonal preconditioner (much more effective!)
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  // Create GPU buffers
  const buffers = createPlateBuffers(ctx, mesh, coloring, material, diagonal, blockDiag, constrainedDOFs);

  // Create/get pipelines
  if (!cachedPipelines) {
    cachedPipelines = await createPipelines(ctx);
  }
  const pipelines = cachedPipelines;

  // Prepare color data for multi-color dispatch
  const colorOffsets: number[] = [];
  const colorCounts: number[] = [];
  let offset = 0;
  for (let c = 0; c < coloring.colorCount; c++) {
    colorOffsets.push(offset);
    colorCounts.push(coloring.colors[c].length);
    offset += coloring.colors[c].length;
  }

  // Upload F to r buffer (initial residual = F when x=0)
  device.queue.writeBuffer(buffers.r, 0, F.buffer, F.byteOffset, F.byteLength);

  // Run GPU PCG
  const result = await runGPUPCG(
    ctx,
    buffers,
    pipelines,
    mesh,
    coloring,
    colorOffsets,
    colorCounts,
    dofCount,
    tolerance,
    maxIterations
  );

  // Read solution back
  const solution = await readFromBuffer(device, buffers.x, buffers.stagingX, dofCount);

  // Cleanup
  destroyPlateBuffers(buffers);

  const gpuTimeMs = performance.now() - startTime;

  return {
    solution,
    iterations: result.iterations,
    finalResidual: result.finalResidual,
    converged: result.converged,
    gpuTimeMs,
    usedGPU: true,
  };
}

/**
 * GPU PCG iteration loop.
 */
async function runGPUPCG(
  ctx: GPUContext,
  buffers: PlateGPUBuffers,
  pipelines: PlatePipelines,
  mesh: PlateMesh,
  coloring: ElementColoring,
  colorOffsets: number[],
  colorCounts: number[],
  dofCount: number,
  tolerance: number,
  maxIterations: number
): Promise<{ iterations: number; finalResidual: number; converged: boolean }> {
  const { device } = ctx;
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const isTriangleMesh = nodesPerElem === 3;

  // Create params buffer for shader dispatch
  const paramsBuffer = createParamsBuffer(device, 16);

  // Helper: compute dot product on GPU
  async function gpuDot(aBuffer: GPUBuffer, bBuffer: GPUBuffer): Promise<number> {
    const workgroups = Math.ceil(dofCount / 256);

    // Phase 1: Compute partial sums
    {
      const params = new Uint32Array([dofCount, 0, 0, 0]);
      device.queue.writeBuffer(paramsBuffer, 0, params);

      const bindGroup = device.createBindGroup({
        layout: pipelines.dotProduct.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: buffers.dotPartial } },
          { binding: 3, resource: { buffer: paramsBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipelines.dotProduct);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    // Phase 2: Final reduction
    if (workgroups > 1) {
      const params = new Uint32Array([workgroups, 0, 0, 0]);
      device.queue.writeBuffer(paramsBuffer, 0, params);

      const bindGroup = device.createBindGroup({
        layout: pipelines.reduceSum.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.dotPartial } },
          { binding: 1, resource: { buffer: buffers.dotResult } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipelines.reduceSum);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();

      encoder.copyBufferToBuffer(buffers.dotResult, 0, buffers.stagingDot, 0, 4);
      device.queue.submit([encoder.finish()]);
    } else {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffers.dotPartial, 0, buffers.stagingDot, 0, 4);
      device.queue.submit([encoder.finish()]);
    }

    // Readback
    await buffers.stagingDot.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(buffers.stagingDot.getMappedRange().slice(0, 4))[0];
    buffers.stagingDot.unmap();

    return result;
  }

  // Helper: apply block preconditioner z = M^-1 * r (3x3 blocks per node)
  function gpuPrecondition(encoder: GPUCommandEncoder): void {
    const nodeCount = mesh.nodeCount;
    const params = new Uint32Array([nodeCount, 0, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: pipelines.blockPreconditioner.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.r } },
        { binding: 1, resource: { buffer: buffers.blockDiagInv } },
        { binding: 2, resource: { buffer: buffers.z } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.blockPreconditioner);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(nodeCount / 256));
    pass.end();
  }

  // Helper: copy buffer dst = src
  function gpuCopy(encoder: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    const params = new Uint32Array([dofCount, 0, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: pipelines.copy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.copy);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dofCount / 256));
    pass.end();
  }

  // Helper: axpy y = alpha * x + y
  function gpuAxpy(encoder: GPUCommandEncoder, xBuf: GPUBuffer, yBuf: GPUBuffer, alpha: number): void {
    const combined = new ArrayBuffer(16);
    new Float32Array(combined, 0, 1)[0] = alpha;
    new Uint32Array(combined, 4, 3).set([dofCount, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, combined);

    const bindGroup = device.createBindGroup({
      layout: pipelines.axpy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: yBuf } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.axpy);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dofCount / 256));
    pass.end();
  }

  // Helper: update p = z + beta * p
  function gpuUpdateP(encoder: GPUCommandEncoder, beta: number): void {
    const combined = new ArrayBuffer(16);
    new Float32Array(combined, 0, 1)[0] = beta;
    new Uint32Array(combined, 4, 3).set([dofCount, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, combined);

    const bindGroup = device.createBindGroup({
      layout: pipelines.updateP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.z } },
        { binding: 1, resource: { buffer: buffers.p } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.updateP);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dofCount / 256));
    pass.end();
  }

  // Helper: zero buffer
  function gpuZero(encoder: GPUCommandEncoder, buf: GPUBuffer): void {
    const params = new Uint32Array([dofCount, 0, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: pipelines.zeroBuffer.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.zeroBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dofCount / 256));
    pass.end();
  }

  // Helper: apply boundary conditions Ap[constrained] = p[constrained]
  function gpuApplyBC(encoder: GPUCommandEncoder): void {
    const params = new Uint32Array([dofCount, 0, 0, 0]);
    device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = device.createBindGroup({
      layout: pipelines.applyBC.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.p } },
        { binding: 1, resource: { buffer: buffers.Ap } },
        { binding: 2, resource: { buffer: buffers.constrainedMask } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines.applyBC);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dofCount / 256));
    pass.end();
  }

  /**
   * Multi-color K·p dispatch - THE KEY OPERATION
   *
   * For each color, dispatch all elements of that color in parallel.
   * Elements of the same color don't share nodes, so no race conditions.
   * The Ap buffer accumulates contributions from all colors.
   */
  function gpuApplyK(encoder: GPUCommandEncoder): void {
    // First, zero the Ap buffer
    gpuZero(encoder, buffers.Ap);

    // Select pipeline based on element type
    const pipeline = isTriangleMesh ? pipelines.applyKDKT : pipelines.applyKQ4;
    const workgroupSize = 64;

    // Dispatch for each color
    for (let c = 0; c < coloring.colorCount; c++) {
      const colorOffset = colorOffsets[c];
      const colorCount = colorCounts[c];

      if (colorCount === 0) continue;

      // Create per-color params buffer
      const colorParams = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `apply_k_params_color_${c}`,
      });
      device.queue.writeBuffer(colorParams, 0, new Uint32Array([colorOffset, colorCount, 0, 0]));

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.nodes } },
          { binding: 1, resource: { buffer: buffers.elements } },
          { binding: 2, resource: { buffer: buffers.elementsByColor } },
          { binding: 3, resource: { buffer: buffers.material } },
          { binding: 4, resource: { buffer: buffers.p } },  // Input: p
          { binding: 5, resource: { buffer: buffers.Ap } }, // Output: Ap (accumulate)
          { binding: 6, resource: { buffer: buffers.constrainedMask } },
          { binding: 7, resource: { buffer: colorParams } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(colorCount / workgroupSize));
      pass.end();

      // Note: colorParams will be garbage collected, but we could pool these for optimization
    }

    // Apply boundary conditions: Ap[constrained] = p[constrained]
    gpuApplyBC(encoder);
  }

  // ============================================================================
  // PCG Main Loop
  // ============================================================================

  // Initial: z = M⁻¹·r
  {
    const encoder = device.createCommandEncoder();
    gpuPrecondition(encoder);
    gpuCopy(encoder, buffers.z, buffers.p); // p = z
    device.queue.submit([encoder.finish()]);
  }

  // rz = r · z
  let rz = await gpuDot(buffers.r, buffers.z);

  // Compute initial residual norm
  const r0_norm = Math.sqrt(await gpuDot(buffers.r, buffers.r));
  if (r0_norm < 1e-30) {
    paramsBuffer.destroy();
    return { iterations: 0, finalResidual: 0, converged: true };
  }

  for (let iter = 0; iter < maxIterations; iter++) {
    // Ap = K·p (MULTI-COLOR DISPATCH)
    {
      const encoder = device.createCommandEncoder();
      gpuApplyK(encoder);
      device.queue.submit([encoder.finish()]);
    }

    // pAp = p · Ap
    const pAp = await gpuDot(buffers.p, buffers.Ap);

    if (Math.abs(pAp) < 1e-30) {
      paramsBuffer.destroy();
      return { iterations: iter + 1, finalResidual: Math.sqrt(rz), converged: false };
    }

    // α = rz / pAp
    const alpha = rz / pAp;

    // x = x + α·p, r = r - α·Ap
    {
      const encoder = device.createCommandEncoder();
      gpuAxpy(encoder, buffers.p, buffers.x, alpha);     // x += α·p
      gpuAxpy(encoder, buffers.Ap, buffers.r, -alpha);   // r -= α·Ap
      device.queue.submit([encoder.finish()]);
    }

    // Check convergence
    const rr = await gpuDot(buffers.r, buffers.r);
    const residualNorm = Math.sqrt(rr);

    if (residualNorm < tolerance || residualNorm / r0_norm < tolerance) {
      paramsBuffer.destroy();
      return { iterations: iter + 1, finalResidual: residualNorm, converged: true };
    }

    // z = M⁻¹·r
    {
      const encoder = device.createCommandEncoder();
      gpuPrecondition(encoder);
      device.queue.submit([encoder.finish()]);
    }

    // rz_new = r · z
    const rz_new = await gpuDot(buffers.r, buffers.z);

    // β = rz_new / rz
    const beta = rz_new / rz;

    // p = z + β·p
    {
      const encoder = device.createCommandEncoder();
      gpuUpdateP(encoder, beta);
      device.queue.submit([encoder.finish()]);
    }

    rz = rz_new;
  }

  // Max iterations reached
  const finalRR = await gpuDot(buffers.r, buffers.r);
  paramsBuffer.destroy();

  return {
    iterations: maxIterations,
    finalResidual: Math.sqrt(finalRR),
    converged: false,
  };
}

/**
 * CPU fallback solver using existing PCG implementation.
 */
function solveCPUFallback(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions
): GPUSolveResult {
  const startTime = performance.now();

  // Import CPU solver components
  const { solvePCG } = require('../pcg');
  const { applyGlobalK, computeBlockDiagonal, invertBlockDiagonal } = require('../solver');

  // Build block preconditioner (much more effective than scalar diagonal)
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  // Define K·x operation
  const applyK = (x: Float32Array, y: Float32Array) => {
    applyGlobalK(mesh, material, coloring, x, y, constrainedDOFs);
  };

  // Solve with block preconditioner
  const result = solvePCG(applyK, F, {
    tolerance: options.tolerance ?? 1e-6,
    maxIterations: options.maxIterations ?? 1000,
    blockPreconditioner: blockDiag,
  });

  return {
    solution: result.solution,
    iterations: result.iterations,
    finalResidual: result.finalResidual,
    converged: result.converged,
    gpuTimeMs: performance.now() - startTime,
    usedGPU: false,
  };
}

/**
 * Clear cached pipelines (for testing).
 */
export function clearPipelineCache(): void {
  cachedPipelines = null;
}

