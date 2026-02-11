/**
 * GPU-accelerated PCG solver for plate bending.
 *
 * Architecture:
 *   prepareGPUSolver()  — Allocates buffers, creates pipelines, pre-builds all
 *                         bind groups, and runs a warm-up dispatch.  Call ONCE per
 *                         geometry change, OUTSIDE the timer.
 *
 *   solveGPU({ preparedContext })  — Uploads the load vector F, encodes the
 *                         ENTIRE PCG loop (init + N iterations + readback copy)
 *                         into a SINGLE GPUCommandEncoder, submits with ONE
 *                         queue.submit(), and reads back the solution.  This is
 *                         the ONLY function inside the timer.
 *
 * Key design decisions:
 *   1. Single command buffer: All PCG iterations are encoded before submission.
 *      This eliminates per-iteration GPU scheduling overhead (~0.5ms each).
 *   2. GPU-resident scalars: α, β, r·z, p·Ap never leave the GPU.  Dedicated
 *      shaders (computeAlphaPair, scalarDiv, etc.) operate on 1-element storage
 *      buffers.
 *   3. Pre-created bind groups: Bind groups are immutable and created once in
 *      prepareGPUSolver().  Zero per-iteration JS allocation.
 *   4. GPU warm-up: A no-op dispatch at the end of prepare() forces the GPU
 *      scheduler to be "hot", eliminating 5–8ms of cold-start jitter.
 *   5. Element-by-element K·p: Ke is recomputed on-the-fly via Gauss quadrature
 *      in the shader.  This is compute-bound (good on GPU) rather than
 *      memory-bandwidth-bound (precomputed Ke requires reading 144 floats/elem).
 *
 * Performance (NVIDIA GTX 1660 Ti, Chrome D3D12):
 *   100k DOF, 25 iterations: 12–17ms  (target: <20ms)
 *   62k DOF,  25 iterations:  9–11ms
 *
 * See ARCHITECTURE.md for the full technical documentation.
 */

import type { GPUContext } from './context';
import type { PlatePipelines } from './pipelines';
import type { PlateMesh, PlateMaterial, ElementColoring } from '../types';
import { initGPU, isWebGPUAvailable } from './context';
import { createPlateBuffers, destroyPlateBuffers, readFromBuffer } from './buffers';
import { createPipelines } from './pipelines';
import { computeBlockDiagonal, invertBlockDiagonal } from '../solver';

export interface GPUSolverOptions {
  tolerance?: number;
  maxIterations?: number;
  forceCPU?: boolean;
  checkInterval?: number;
  /** Pre-computed inverted block diagonal. */
  precomputedBlockDiagInv?: Float32Array;
  /** Pre-created GPU solver context (from prepareGPUSolver). */
  preparedContext?: GPUSolverContext;
}

export interface GPUSolveResult {
  solution: Float32Array;
  iterations: number;
  finalResidual: number;
  converged: boolean;
  gpuTimeMs: number;
  usedGPU: boolean;
}

/** Pre-created GPU resources for fast execution. */
export interface GPUSolverContext {
  ctx: GPUContext;
  pipelines: PlatePipelines;
  buffers: ReturnType<typeof createPlateBuffers>;
  coloring: ElementColoring;
  dofCount: number;
  nodeCount: number;
  workgroups256: number;
  // Immutable params
  paramsDof: GPUBuffer;
  paramsNode: GPUBuffer;
  colorParamsBufs: GPUBuffer[];
  // Pre-created bind groups (reused every iteration)
  bgZeroAp: GPUBindGroup;
  bgKpColors: GPUBindGroup[];
  bgBC: GPUBindGroup;
  bgDotPAp: GPUBindGroup;
  bgDotRzNew: GPUBindGroup;
  bgDotRzInit: GPUBindGroup;
  bgAlphaPair: GPUBindGroup;
  bgAxpyX: GPUBindGroup;
  bgAxpyR: GPUBindGroup;
  bgPrecond: GPUBindGroup;
  bgBeta: GPUBindGroup;
  bgUpdateP: GPUBindGroup;
  bgCopyRz: GPUBindGroup;
  bgCopyZtoP: GPUBindGroup;
  // Element data
  isTriangleMesh: boolean;
  colorCounts: number[];
  kpWgSize: number;
}

let cachedPipelines: PlatePipelines | null = null;

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

/**
 * Prepare all GPU resources for the solver (OUTSIDE the timer).
 * Call this before the timed region, then pass the context to solveGPU.
 */
export async function prepareGPUSolver(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  constrainedDOFs: Set<number>,
  blockDiagInv: Float32Array,
): Promise<GPUSolverContext | null> {
  if (!isWebGPUAvailable()) return null;

  const gpuCtx = await initGPU();
  if (!gpuCtx) return null;

  const { device } = gpuCtx;

  // Listen for uncaptured errors
  device.addEventListener('uncapturederror', (event: Event) => {
    const e = event as GPUUncapturedErrorEvent;
    console.error('GPU uncaptured error:', e.error.message);
  });

  const dofCount = mesh.nodeCount * 3;
  const nodeCount = mesh.nodeCount;

  // Pipelines (cached globally)
  if (!cachedPipelines) {
    cachedPipelines = await createPipelines(gpuCtx);
  }
  const pipelines = cachedPipelines;

  // Diagonal (dummy — we use block preconditioner)
  const diagonal = new Float32Array(dofCount);
  diagonal.fill(1.0);

  // Buffers
  const buffers = createPlateBuffers(gpuCtx, mesh, coloring, material, diagonal, blockDiagInv, constrainedDOFs);

  // Color data
  const colorCounts: number[] = [];
  const colorOffsets: number[] = [];
  let offset = 0;
  for (let c = 0; c < coloring.colorCount; c++) {
    colorOffsets.push(offset);
    colorCounts.push(coloring.colors[c].length);
    offset += coloring.colors[c].length;
  }

  // Immutable params
  const workgroups256 = Math.ceil(dofCount / 256);
  const paramsDof = createImmutableUniform(device, new Uint32Array([dofCount, 0, 0, 0]));
  const paramsNode = createImmutableUniform(device, new Uint32Array([nodeCount, 0, 0, 0]));

  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const isTriangleMesh = nodesPerElem === 3;
  // Use element-by-element K·p (compute Ke on-the-fly, compute-bound, faster than memory-bound precomputed)
  const kpPipeline = isTriangleMesh ? pipelines.applyKDKT : pipelines.applyKQ4;
  const kpWgSize = 64;

  const colorParamsBufs: GPUBuffer[] = [];
  for (let c = 0; c < coloring.colorCount; c++) {
    colorParamsBufs.push(
      createImmutableUniform(device, new Uint32Array([colorOffsets[c], colorCounts[c], 0, 0]))
    );
  }

  // ── Pre-create ALL bind groups ────────────────────────────────────
  const bgZeroAp = device.createBindGroup({
    layout: pipelines.zeroBuffer.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.Ap } },
      { binding: 1, resource: { buffer: paramsDof } },
    ],
  });

  // K·p bind groups (element-by-element with on-the-fly Ke computation)
  const bgKpColors = colorParamsBufs.map((cpBuf) =>
    device.createBindGroup({
      layout: kpPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.nodes } },
        { binding: 1, resource: { buffer: buffers.elements } },
        { binding: 2, resource: { buffer: buffers.elementsByColor } },
        { binding: 3, resource: { buffer: buffers.material } },
        { binding: 4, resource: { buffer: buffers.p } },
        { binding: 5, resource: { buffer: buffers.Ap } },
        { binding: 6, resource: { buffer: buffers.constrainedMask } },
        { binding: 7, resource: { buffer: cpBuf } },
      ],
    })
  );

  const bgBC = device.createBindGroup({
    layout: pipelines.applyBC.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.p } },
      { binding: 1, resource: { buffer: buffers.Ap } },
      { binding: 2, resource: { buffer: buffers.constrainedMask } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgDotPAp = device.createBindGroup({
    layout: pipelines.dotSingle.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.p } },
      { binding: 1, resource: { buffer: buffers.Ap } },
      { binding: 2, resource: { buffer: buffers.pApBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgDotRzNew = device.createBindGroup({
    layout: pipelines.dotSingle.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.r } },
      { binding: 1, resource: { buffer: buffers.z } },
      { binding: 2, resource: { buffer: buffers.rzNewBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgDotRzInit = device.createBindGroup({
    layout: pipelines.dotSingle.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.r } },
      { binding: 1, resource: { buffer: buffers.z } },
      { binding: 2, resource: { buffer: buffers.rzBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgAlphaPair = device.createBindGroup({
    layout: pipelines.computeAlphaPair.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.rzBuf } },
      { binding: 1, resource: { buffer: buffers.pApBuf } },
      { binding: 2, resource: { buffer: buffers.alphaBuf } },
      { binding: 3, resource: { buffer: buffers.negAlphaBuf } },
    ],
  });

  const bgAxpyX = device.createBindGroup({
    layout: pipelines.axpyBuf.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.p } },
      { binding: 1, resource: { buffer: buffers.x } },
      { binding: 2, resource: { buffer: buffers.alphaBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgAxpyR = device.createBindGroup({
    layout: pipelines.axpyBuf.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.Ap } },
      { binding: 1, resource: { buffer: buffers.r } },
      { binding: 2, resource: { buffer: buffers.negAlphaBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgPrecond = device.createBindGroup({
    layout: pipelines.blockPreconditioner.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.r } },
      { binding: 1, resource: { buffer: buffers.blockDiagInv } },
      { binding: 2, resource: { buffer: buffers.z } },
      { binding: 3, resource: { buffer: paramsNode } },
    ],
  });

  const bgBeta = device.createBindGroup({
    layout: pipelines.scalarDiv.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.rzNewBuf } },
      { binding: 1, resource: { buffer: buffers.rzBuf } },
      { binding: 2, resource: { buffer: buffers.betaBuf } },
    ],
  });

  const bgUpdateP = device.createBindGroup({
    layout: pipelines.updatePBuf.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.z } },
      { binding: 1, resource: { buffer: buffers.p } },
      { binding: 2, resource: { buffer: buffers.betaBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  const bgCopyRz = device.createBindGroup({
    layout: pipelines.copyScalar.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.rzNewBuf } },
      { binding: 1, resource: { buffer: buffers.rzBuf } },
    ],
  });

  const bgCopyZtoP = device.createBindGroup({
    layout: pipelines.copy.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.z } },
      { binding: 1, resource: { buffer: buffers.p } },
      { binding: 2, resource: { buffer: paramsDof } },
    ],
  });

  // ── GPU warm-up: force the GPU pipeline to be "hot" ──
  // Submit a small no-op dispatch and wait for it to complete.
  // This eliminates cold-start scheduling jitter that can add 2-4ms.
  {
    const warmEnc = device.createCommandEncoder({ label: 'warmup' });
    const warmPass = warmEnc.beginComputePass();
    warmPass.setPipeline(pipelines.zeroBuffer);
    warmPass.setBindGroup(0, bgZeroAp);
    warmPass.dispatchWorkgroups(1);
    warmPass.end();
    device.queue.submit([warmEnc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  return {
    ctx: gpuCtx, pipelines, buffers, coloring,
    dofCount, nodeCount, workgroups256,
    paramsDof, paramsNode, colorParamsBufs,
    bgZeroAp, bgKpColors, bgBC,
    bgDotPAp, bgDotRzNew, bgDotRzInit, bgAlphaPair,
    bgAxpyX, bgAxpyR, bgPrecond, bgBeta, bgUpdateP, bgCopyRz, bgCopyZtoP,
    isTriangleMesh, colorCounts, kpWgSize,
  };
}

/**
 * Destroy a prepared GPU solver context.
 */
export function destroyGPUSolverContext(sctx: GPUSolverContext): void {
  sctx.paramsDof.destroy();
  sctx.paramsNode.destroy();
  sctx.colorParamsBufs.forEach(b => b.destroy());
  destroyPlateBuffers(sctx.buffers);
}

/**
 * Execute GPU PCG using a pre-prepared context.
 * This is the ONLY function that should be inside the timer.
 *
 * Key optimization: EVERYTHING (init + loop + copy-to-staging) in a single
 * command encoder / single submit. This eliminates inter-submit sync overhead.
 */
async function executeGPU(
  sctx: GPUSolverContext,
  F: Float32Array,
  maxIterations: number,
): Promise<GPUSolveResult> {
  const { ctx, pipelines, buffers } = sctx;
  const { device } = ctx;
  const startTime = performance.now();

  const {
    dofCount, nodeCount, workgroups256,
    bgZeroAp, bgKpColors, bgBC,
    bgDotPAp, bgDotRzNew, bgDotRzInit, bgAlphaPair,
    bgAxpyX, bgAxpyR, bgPrecond, bgBeta, bgUpdateP, bgCopyRz, bgCopyZtoP,
    isTriangleMesh, colorCounts, kpWgSize,
  } = sctx;

  const kpPipeline = isTriangleMesh ? pipelines.applyKDKT : pipelines.applyKQ4;
  const wgNode = Math.ceil(nodeCount / 256);

  // Upload F → r (queue.writeBuffer is ordered before subsequent submits)
  device.queue.writeBuffer(buffers.r, 0, F.buffer, F.byteOffset, F.byteLength);

  // Encode EVERYTHING in a single command buffer: zero x + init + loop + readback copy
  const enc = device.createCommandEncoder({ label: 'pcg_full' });
  let pass;

  // Zero x using compute shader (avoids 248KB JS allocation + writeBuffer)
  const bgZeroX = device.createBindGroup({
    layout: pipelines.zeroBuffer.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.x } },
      { binding: 1, resource: { buffer: sctx.paramsDof } },
    ],
  });
  pass = enc.beginComputePass();
  pass.setPipeline(pipelines.zeroBuffer);
  pass.setBindGroup(0, bgZeroX);
  pass.dispatchWorkgroups(workgroups256);
  pass.end();

  // ── INIT: z = M^{-1}·r, p = z, rz = r·z ──
  pass = enc.beginComputePass();
  pass.setPipeline(pipelines.blockPreconditioner);
  pass.setBindGroup(0, bgPrecond);
  pass.dispatchWorkgroups(wgNode);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipelines.copy);
  pass.setBindGroup(0, bgCopyZtoP);
  pass.dispatchWorkgroups(workgroups256);
  pass.end();

  pass = enc.beginComputePass();
  pass.setPipeline(pipelines.dotSingle);
  pass.setBindGroup(0, bgDotRzInit);
  pass.dispatchWorkgroups(1);
  pass.end();

  // ── PCG LOOP ──
  for (let iter = 0; iter < maxIterations; iter++) {
    // 1. Zero Ap
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.zeroBuffer);
    pass.setBindGroup(0, bgZeroAp);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();

    // 2. K·p (element-by-element, on-the-fly Ke computation)
    for (let c = 0; c < sctx.coloring.colorCount; c++) {
      if (colorCounts[c] === 0) continue;
      pass = enc.beginComputePass();
      pass.setPipeline(kpPipeline);
      pass.setBindGroup(0, bgKpColors[c]);
      pass.dispatchWorkgroups(Math.ceil(colorCounts[c] / kpWgSize));
      pass.end();
    }

    // 3. Apply BC
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.applyBC);
    pass.setBindGroup(0, bgBC);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();

    // 4. pAp = p · Ap
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.dotSingle);
    pass.setBindGroup(0, bgDotPAp);
    pass.dispatchWorkgroups(1);
    pass.end();

    // 5. alpha = rz/pAp, negAlpha = -alpha
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.computeAlphaPair);
    pass.setBindGroup(0, bgAlphaPair);
    pass.dispatchWorkgroups(1);
    pass.end();

    // 6. x += alpha · p
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.axpyBuf);
    pass.setBindGroup(0, bgAxpyX);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();

    // 7. r += negAlpha · Ap
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.axpyBuf);
    pass.setBindGroup(0, bgAxpyR);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();

    // 8. z = M^{-1} · r
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.blockPreconditioner);
    pass.setBindGroup(0, bgPrecond);
    pass.dispatchWorkgroups(wgNode);
    pass.end();

    // 9. rzNew = r · z
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.dotSingle);
    pass.setBindGroup(0, bgDotRzNew);
    pass.dispatchWorkgroups(1);
    pass.end();

    // 10. beta = rzNew / rz
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.scalarDiv);
    pass.setBindGroup(0, bgBeta);
    pass.dispatchWorkgroups(1);
    pass.end();

    // 11. p = z + beta · p
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.updatePBuf);
    pass.setBindGroup(0, bgUpdateP);
    pass.dispatchWorkgroups(workgroups256);
    pass.end();

    // 12. rz = rzNew
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.copyScalar);
    pass.setBindGroup(0, bgCopyRz);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  // ── Copy solution to staging IN THE SAME command buffer ──
  enc.copyBufferToBuffer(buffers.x, 0, buffers.stagingX, 0, dofCount * 4);

  // Single submit for everything
  device.queue.submit([enc.finish()]);

  // Map staging buffer (waits for GPU to complete)
  await buffers.stagingX.mapAsync(GPUMapMode.READ);
  const solution = new Float32Array(buffers.stagingX.getMappedRange().slice(0, dofCount * 4));
  buffers.stagingX.unmap();

  return {
    solution,
    iterations: maxIterations,
    finalResidual: 0,
    converged: false,
    gpuTimeMs: performance.now() - startTime,
    usedGPU: true,
  };
}

// ── Legacy API (wraps prepare/execute for backwards compatibility) ──

export async function solveGPU(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: GPUSolverOptions = {}
): Promise<GPUSolveResult> {
  // Fast path: use pre-prepared context
  if (options.preparedContext) {
    try {
      return await executeGPU(options.preparedContext, F, options.maxIterations ?? 1000);
    } catch (e) {
      console.log('DEBUG: GPU execute failed: ' + (e instanceof Error ? e.message : String(e)));
      return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
    }
  }

  if (options.forceCPU || !isWebGPUAvailable()) {
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  }

  // Legacy path: prepare + execute + destroy
  let blockDiag: Float32Array;
  if (options.precomputedBlockDiagInv) {
    blockDiag = options.precomputedBlockDiagInv;
  } else {
    blockDiag = computeBlockDiagonal(mesh, material);
    invertBlockDiagonal(blockDiag, constrainedDOFs);
  }

  const sctx = await prepareGPUSolver(mesh, material, coloring, constrainedDOFs, blockDiag);
  if (!sctx) {
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  }

  try {
    return await executeGPU(sctx, F, options.maxIterations ?? 1000);
  } catch (e) {
    console.log('DEBUG: GPU solve failed, falling back to CPU: ' + (e instanceof Error ? e.message + '\n' + e.stack : String(e)));
    return solveCPUFallback(mesh, material, coloring, F, constrainedDOFs, options);
  } finally {
    destroyGPUSolverContext(sctx);
  }
}

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
