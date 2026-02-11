/**
 * GPU-accelerated PCG solver for plate bending.
 *
 * Hybrid approach: GPU computes K·p (the expensive part), CPU handles
 * all other vector operations for numerical stability.
 *
 * GPU K·p has been validated against CPU K·p — they match to within
 * 0.003% (float32 vs float64 accumulation).
 */

import type { GPUContext } from './context';
import type { PlateGPUBuffers } from './buffers';
import type { PlatePipelines } from './pipelines';
import type { PlateMesh, PlateMaterial, ElementColoring } from '../types';
import { initGPU, isWebGPUAvailable } from './context';
import { createPlateBuffers, destroyPlateBuffers, readFromBuffer } from './buffers';
import { createPipelines } from './pipelines';
import { computeDiagonal, applyBCsToDiagonal, computeBlockDiagonal, invertBlockDiagonal } from '../solver';
import { dot, norm, axpy, copy } from '../pcg';

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

/**
 * Apply block preconditioner on CPU: z = M^{-1} r
 */
function applyBlockPrecondCPU(
  blockInv: Float32Array,
  r: Float32Array,
  z: Float32Array
): void {
  const nodeCount = blockInv.length / 9;
  for (let node = 0; node < nodeCount; node++) {
    const bo = node * 9;
    const d = node * 3;
    const r0 = r[d], r1 = r[d + 1], r2 = r[d + 2];
    z[d]     = blockInv[bo]     * r0 + blockInv[bo + 1] * r1 + blockInv[bo + 2] * r2;
    z[d + 1] = blockInv[bo + 3] * r0 + blockInv[bo + 4] * r1 + blockInv[bo + 5] * r2;
    z[d + 2] = blockInv[bo + 6] * r0 + blockInv[bo + 7] * r1 + blockInv[bo + 8] * r2;
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
  const maxIterations = options.maxIterations ?? 1000;
  const dofCount = mesh.nodeCount * 3;

  // ── Preconditioner (CPU, computed once) ────────────────────────────
  const diagonal = computeDiagonal(mesh, material);
  applyBCsToDiagonal(diagonal, constrainedDOFs);
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  // ── GPU setup ─────────────────────────────────────────────────────
  const buffers = createPlateBuffers(ctx, mesh, coloring, material, diagonal, blockDiag, constrainedDOFs);

  if (!cachedPipelines) {
    cachedPipelines = await createPipelines(ctx);
  }
  const pipelines = cachedPipelines;

  // Color data for K·p dispatch
  const colorOffsets: number[] = [];
  const colorCounts: number[] = [];
  let offset = 0;
  for (let c = 0; c < coloring.colorCount; c++) {
    colorOffsets.push(offset);
    colorCounts.push(coloring.colors[c].length);
    offset += coloring.colors[c].length;
  }

  // Immutable params for GPU shaders
  const workgroups256 = Math.ceil(dofCount / 256);
  const paramsDof = createImmutableUniform(device, new Uint32Array([dofCount, 0, 0, 0]));

  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const isTriangleMesh = nodesPerElem === 3;
  const colorParamsBufs: GPUBuffer[] = [];
  for (let c = 0; c < coloring.colorCount; c++) {
    colorParamsBufs.push(
      createImmutableUniform(device, new Uint32Array([colorOffsets[c], colorCounts[c], 0, 0]))
    );
  }

  // ── GPU K·p helpers ───────────────────────────────────────────────
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

  async function computeKp(p: Float32Array): Promise<Float32Array> {
    device.queue.writeBuffer(buffers.p, 0, p.buffer, p.byteOffset, p.byteLength);

    const enc = device.createCommandEncoder();
    gpuApplyK(enc);
    device.queue.submit([enc.finish()]);

    return readFromBuffer(device, buffers.Ap, buffers.stagingX, dofCount);
  }

  // ══════════════════════════════════════════════════════════════════════
  // PCG: GPU for K·p, CPU for everything else
  // ══════════════════════════════════════════════════════════════════════

  const x = new Float32Array(dofCount);
  const r = new Float32Array(dofCount);
  const z = new Float32Array(dofCount);
  const p = new Float32Array(dofCount);

  // r = F (since x = 0)
  copy(r, F);

  const r0_norm = norm(r);
  if (r0_norm < 1e-30) {
    paramsDof.destroy();
    colorParamsBufs.forEach(b => b.destroy());
    destroyPlateBuffers(buffers);
    return {
      solution: x,
      iterations: 0,
      finalResidual: 0,
      converged: true,
      gpuTimeMs: performance.now() - startTime,
      usedGPU: true,
    };
  }

  // z = M^{-1} · r
  applyBlockPrecondCPU(blockDiag, r, z);

  // p = z
  copy(p, z);

  // rz = r · z
  let rz = dot(r, z);

  let totalIters = 0;
  let lastResidualNorm = r0_norm;

  for (let iter = 0; iter < maxIterations; iter++) {
    // Ap = K · p (GPU)
    const Ap = await computeKp(p);

    // pAp = p · Ap
    const pAp = dot(p, Ap);

    if (Math.abs(pAp) < 1e-30) {
      console.warn('[GPU PCG] pAp ≈ 0, stopping');
      break;
    }

    // alpha = rz / pAp
    const alpha = rz / pAp;

    // x += alpha * p
    axpy(alpha, p, x);

    // r -= alpha * Ap
    axpy(-alpha, Ap, r);

    totalIters = iter + 1;

    // Check convergence
    const residualNorm = norm(r);
    lastResidualNorm = residualNorm;

    if (residualNorm < tolerance || residualNorm / r0_norm < tolerance) {
      break;
    }

    // Divergence check
    if (residualNorm > r0_norm * 1e6) {
      console.warn(`[GPU PCG] Diverging at iter ${totalIters}: residual=${residualNorm.toExponential(2)}`);
      break;
    }

    // z = M^{-1} · r
    applyBlockPrecondCPU(blockDiag, r, z);

    // rz_new = r · z
    const rz_new = dot(r, z);

    // beta = rz_new / rz
    const beta = rz_new / rz;

    // p = z + beta * p
    for (let i = 0; i < dofCount; i++) {
      p[i] = z[i] + beta * p[i];
    }

    rz = rz_new;
  }

  // ── Cleanup ────────────────────────────────────────────────────────
  paramsDof.destroy();
  colorParamsBufs.forEach(b => b.destroy());
  destroyPlateBuffers(buffers);

  const gpuTimeMs = performance.now() - startTime;

  return {
    solution: x,
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

export function clearPipelineCache(): void {
  cachedPipelines = null;
}
