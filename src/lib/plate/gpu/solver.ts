/**
 * GPU-accelerated PCG solver for plate bending.
 *
 * Uses immutable params buffers to avoid writeBuffer race conditions.
 * Two modes:
 * - "readback" mode: CPU readbacks for dot products (reliable, slower)
 * - "batched" mode: GPU-only scalars (fast, requires working shader pipeline)
 */

import type { GPUContext } from './context';
import type { PlateGPUBuffers } from './buffers';
import type { PlatePipelines } from './pipelines';
import type { PlateMesh, PlateMaterial, ElementColoring } from '../types';
import { initGPU, isWebGPUAvailable } from './context';
import { createPlateBuffers, destroyPlateBuffers, readFromBuffer } from './buffers';
import { createPipelines, createParamsBuffer } from './pipelines';
import { computeDiagonal, applyBCsToDiagonal, computeBlockDiagonal, invertBlockDiagonal } from '../solver';

export interface GPUSolverOptions {
  tolerance?: number;
  maxIterations?: number;
  forceCPU?: boolean;
  checkInterval?: number;
}

export interface GPUSolveResult {
  solution: Float32Array;
  iterations: number;
  finalResidual: number;
  converged: boolean;
  gpuTimeMs: number;
  usedGPU: boolean;
}

let cachedPipelines: PlatePipelines | null = null;

/**
 * Create an immutable uniform buffer with pre-filled data.
 */
function createImmutableUniform(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
    label: 'immutable_params',
  });
  new Uint32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

export async function solveGPU(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions = {}
): Promise<GPUSolveResult> {
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
  const maxIterations = options.maxIterations ?? 2000;
  const dofCount = mesh.nodeCount * 3;

  // Compute preconditioners on CPU
  const diagonal = computeDiagonal(mesh, material);
  applyBCsToDiagonal(diagonal, constrainedDOFs);
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  // Create GPU buffers
  const buffers = createPlateBuffers(ctx, mesh, coloring, material, diagonal, blockDiag, constrainedDOFs);

  // Create/get pipelines
  if (!cachedPipelines) {
    cachedPipelines = await createPipelines(ctx);
  }
  const pipelines = cachedPipelines;

  // Color data
  const colorOffsets: number[] = [];
  const colorCounts: number[] = [];
  let offset = 0;
  for (let c = 0; c < coloring.colorCount; c++) {
    colorOffsets.push(offset);
    colorCounts.push(coloring.colors[c].length);
    offset += coloring.colors[c].length;
  }

  // Upload F to r buffer
  device.queue.writeBuffer(buffers.r, 0, F.buffer, F.byteOffset, F.byteLength);

  // Pre-create immutable params buffers
  const workgroups256 = Math.ceil(dofCount / 256);
  const paramsDof = createImmutableUniform(device, new Uint32Array([dofCount, 0, 0, 0]));
  const paramsNode = createImmutableUniform(device, new Uint32Array([mesh.nodeCount, 0, 0, 0]));
  const paramsWg = createImmutableUniform(device, new Uint32Array([workgroups256, 0, 0, 0]));

  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const isTriangleMesh = nodesPerElem === 3;
  const colorParamsBufs: GPUBuffer[] = [];
  for (let c = 0; c < coloring.colorCount; c++) {
    colorParamsBufs.push(
      createImmutableUniform(device, new Uint32Array([colorOffsets[c], colorCounts[c], 0, 0]))
    );
  }

  // ══════════════════════════════════════════════════════════════════════
  // Helper functions using immutable params
  // ══════════════════════════════════════════════════════════════════════

  function gpuPrecondition(enc: GPUCommandEncoder): void {
    const bg = device.createBindGroup({
      layout: pipelines.blockPreconditioner.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.r } },
        { binding: 1, resource: { buffer: buffers.blockDiagInv } },
        { binding: 2, resource: { buffer: buffers.z } },
        { binding: 3, resource: { buffer: paramsNode } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.blockPreconditioner);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(mesh.nodeCount / 256));
    pass.end();
  }

  function gpuCopy(enc: GPUCommandEncoder, src: GPUBuffer, dst: GPUBuffer): void {
    const bg = device.createBindGroup({
      layout: pipelines.copy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: paramsDof } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.copy);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();
  }

  function gpuZero(enc: GPUCommandEncoder, buf: GPUBuffer): void {
    const bg = device.createBindGroup({
      layout: pipelines.zeroBuffer.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: { buffer: paramsDof } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.zeroBuffer);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();
  }

  function gpuApplyBC(enc: GPUCommandEncoder): void {
    const bg = device.createBindGroup({
      layout: pipelines.applyBC.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.p } },
        { binding: 1, resource: { buffer: buffers.Ap } },
        { binding: 2, resource: { buffer: buffers.constrainedMask } },
        { binding: 3, resource: { buffer: paramsDof } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.applyBC);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();
  }

  function gpuApplyK(enc: GPUCommandEncoder): void {
    gpuZero(enc, buffers.Ap);

    const pipeline = isTriangleMesh ? pipelines.applyKDKT : pipelines.applyKQ4;
    const wgSize = 64;

    for (let c = 0; c < coloring.colorCount; c++) {
      if (colorCounts[c] === 0) continue;

      const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.nodes } },
          { binding: 1, resource: { buffer: buffers.elements } },
          { binding: 2, resource: { buffer: buffers.elementsByColor } },
          { binding: 3, resource: { buffer: buffers.material } },
          { binding: 4, resource: { buffer: buffers.p } },
          { binding: 5, resource: { buffer: buffers.Ap } },
          { binding: 6, resource: { buffer: buffers.constrainedMask } },
          { binding: 7, resource: { buffer: colorParamsBufs[c] } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(colorCounts[c] / wgSize));
      pass.end();
    }

    gpuApplyBC(enc);
  }

  function gpuAxpy(enc: GPUCommandEncoder, xBuf: GPUBuffer, yBuf: GPUBuffer, alpha: number): void {
    // Create a tiny immutable buffer for alpha
    const alphaBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
      label: 'axpy_alpha',
    });
    const view = new ArrayBuffer(16);
    new Float32Array(view, 0, 1)[0] = alpha;
    new Uint32Array(view, 4, 3).set([dofCount, 0, 0]);
    new Uint8Array(alphaBuf.getMappedRange()).set(new Uint8Array(view));
    alphaBuf.unmap();

    const bg = device.createBindGroup({
      layout: pipelines.axpy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: yBuf } },
        { binding: 2, resource: { buffer: alphaBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.axpy);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();
  }

  function gpuUpdateP(enc: GPUCommandEncoder, beta: number): void {
    const betaBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
      label: 'updatep_beta',
    });
    const view = new ArrayBuffer(16);
    new Float32Array(view, 0, 1)[0] = beta;
    new Uint32Array(view, 4, 3).set([dofCount, 0, 0]);
    new Uint8Array(betaBuf.getMappedRange()).set(new Uint8Array(view));
    betaBuf.unmap();

    const bg = device.createBindGroup({
      layout: pipelines.updateP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.z } },
        { binding: 1, resource: { buffer: buffers.p } },
        { binding: 2, resource: { buffer: betaBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipelines.updateP);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();
  }

  /**
   * Dot product with CPU readback.
   * Submits its own encoder, reads back the result.
   */
  async function gpuDot(aBuf: GPUBuffer, bBuf: GPUBuffer): Promise<number> {
    const enc = device.createCommandEncoder();

    // Phase 1: partial sums
    {
      const bg = device.createBindGroup({
        layout: pipelines.dotProduct.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuf } },
          { binding: 1, resource: { buffer: bBuf } },
          { binding: 2, resource: { buffer: buffers.dotPartial } },
          { binding: 3, resource: { buffer: paramsDof } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.dotProduct);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(workgroups256);
      pass.end();
    }

    // Phase 2: reduce
    if (workgroups256 > 1) {
      const bg = device.createBindGroup({
        layout: pipelines.reduceSum.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: buffers.dotPartial } },
          { binding: 1, resource: { buffer: buffers.dotResult } },
          { binding: 2, resource: { buffer: paramsWg } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipelines.reduceSum);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();

      enc.copyBufferToBuffer(buffers.dotResult, 0, buffers.stagingDot, 0, 4);
    } else {
      enc.copyBufferToBuffer(buffers.dotPartial, 0, buffers.stagingDot, 0, 4);
    }

    device.queue.submit([enc.finish()]);

    await buffers.stagingDot.mapAsync(GPUMapMode.READ);
    const val = new Float32Array(buffers.stagingDot.getMappedRange().slice(0, 4))[0];
    buffers.stagingDot.unmap();
    return val;
  }

  // ══════════════════════════════════════════════════════════════════════
  // PCG Main Loop — CPU readback for scalars (reliable mode)
  // ══════════════════════════════════════════════════════════════════════

  // Initial: z = M⁻¹·r, p = z
  {
    const enc = device.createCommandEncoder();
    gpuPrecondition(enc);
    gpuCopy(enc, buffers.z, buffers.p);
    device.queue.submit([enc.finish()]);
  }

  // Get initial values
  const r0_norm_sq = await gpuDot(buffers.r, buffers.r);
  const r0_norm = Math.sqrt(r0_norm_sq);

  if (r0_norm < 1e-30) {
    const solution = new Float32Array(dofCount);
    paramsDof.destroy(); paramsNode.destroy(); paramsWg.destroy();
    colorParamsBufs.forEach(b => b.destroy());
    destroyPlateBuffers(buffers);
    return { solution, iterations: 0, finalResidual: 0, converged: true, gpuTimeMs: performance.now() - startTime, usedGPU: true };
  }

  let rz = await gpuDot(buffers.r, buffers.z);

  let totalIters = 0;
  let lastResidualNorm = r0_norm;

  for (let iter = 0; iter < maxIterations; iter++) {
    // Ap = K·p
    {
      const enc = device.createCommandEncoder();
      gpuApplyK(enc);
      device.queue.submit([enc.finish()]);
    }

    // pAp = p·Ap
    const pAp = await gpuDot(buffers.p, buffers.Ap);

    if (Math.abs(pAp) < 1e-30) {
      console.warn('[GPU PCG] pAp ≈ 0, stopping');
      break;
    }

    // alpha = rz / pAp
    const alpha = rz / pAp;

    // x += alpha * p, r -= alpha * Ap
    {
      const enc = device.createCommandEncoder();
      gpuAxpy(enc, buffers.p, buffers.x, alpha);
      gpuAxpy(enc, buffers.Ap, buffers.r, -alpha);
      device.queue.submit([enc.finish()]);
    }

    totalIters = iter + 1;

    // Check convergence every iteration (since readback is already happening)
    const rr = await gpuDot(buffers.r, buffers.r);
    const residualNorm = Math.sqrt(Math.max(0, rr));
    lastResidualNorm = residualNorm;

    if (residualNorm < tolerance || residualNorm / r0_norm < tolerance) {
      break;
    }

    // Divergence check
    if (residualNorm > r0_norm * 1e6) {
      console.warn(`[GPU PCG] Diverging at iter ${totalIters}: residual=${residualNorm.toExponential(2)}`);
      break;
    }

    // z = M⁻¹·r
    {
      const enc = device.createCommandEncoder();
      gpuPrecondition(enc);
      device.queue.submit([enc.finish()]);
    }

    // rz_new = r·z
    const rz_new = await gpuDot(buffers.r, buffers.z);

    // beta = rz_new / rz
    const beta = rz_new / rz;

    // p = z + beta * p
    {
      const enc = device.createCommandEncoder();
      gpuUpdateP(enc, beta);
      device.queue.submit([enc.finish()]);
    }

    rz = rz_new;
  }

  // Read solution
  const solution = await readFromBuffer(device, buffers.x, buffers.stagingX, dofCount);

  // Cleanup
  paramsDof.destroy(); paramsNode.destroy(); paramsWg.destroy();
  colorParamsBufs.forEach(b => b.destroy());
  destroyPlateBuffers(buffers);

  const gpuTimeMs = performance.now() - startTime;

  return {
    solution,
    iterations: totalIters,
    finalResidual: lastResidualNorm,
    converged: lastResidualNorm < tolerance || lastResidualNorm / r0_norm < tolerance,
    gpuTimeMs,
    usedGPU: true,
  };
}

/**
 * CPU fallback solver.
 */
async function solveCPUFallback(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions
): Promise<GPUSolveResult> {
  const startTime = performance.now();

  const { solvePCG } = await import('../pcg');
  const { applyGlobalK } = await import('../solver');
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  const applyK = (x: Float32Array, y: Float32Array) => {
    applyGlobalK(mesh, material, coloring, x, y, constrainedDOFs);
  };

  const result = solvePCG(applyK, F, {
    tolerance: options.tolerance ?? 1e-6,
    maxIterations: options.maxIterations ?? 2000,
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

export function clearPipelineCache(): void {
  cachedPipelines = null;
}
