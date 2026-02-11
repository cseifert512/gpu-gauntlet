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
  /** Compute moments (Mx, My, Mxy) on GPU in the same command buffer (default: false). */
  computeMoments?: boolean;
}

export interface GPUSolveResult {
  solution: Float32Array;
  iterations: number;
  finalResidual: number;
  converged: boolean;
  gpuTimeMs: number;
  usedGPU: boolean;
  /** GPU-computed moments (if computeMoments option was true) */
  Mx?: Float32Array;
  My?: Float32Array;
  Mxy?: Float32Array;
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
  // Moment computation bind groups (GPU post-processing)
  bgZeroMoments: GPUBindGroup[];   // Zero Mx,My,Mxy,count buffers
  bgMomentColors: GPUBindGroup[];  // Per-color moment accumulation
  bgAvgMoments: GPUBindGroup;      // Final averaging pass
  // Residual readback
  bgDotRR: GPUBindGroup;           // r·r for convergence info
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

  // ── Residual norm bind group (r·r for convergence info) ──
  const bgDotRR = device.createBindGroup({
    layout: pipelines.dotSingle.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.r } },
      { binding: 1, resource: { buffer: buffers.r } },
      { binding: 2, resource: { buffer: buffers.rrBuf } },
      { binding: 3, resource: { buffer: paramsDof } },
    ],
  });

  // ── Moment computation bind groups ──
  const momentPipeline = isTriangleMesh ? pipelines.computeMomentsDKT : pipelines.computeMomentsQ4;

  // Zero the 4 moment accumulators (Mx, My, Mxy, count)
  const bgZeroMoments = [buffers.momentMx, buffers.momentMy, buffers.momentMxy, buffers.momentCount].map(
    (buf, idx) => device.createBindGroup({
      layout: pipelines.zeroBuffer.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: { buffer: paramsNode } },
      ],
    })
  );

  // Per-color moment accumulation
  const bgMomentColors = colorParamsBufs.map((cpBuf) =>
    device.createBindGroup({
      layout: momentPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.nodes } },
        { binding: 1, resource: { buffer: buffers.elements } },
        { binding: 2, resource: { buffer: buffers.elementsByColor } },
        { binding: 3, resource: { buffer: buffers.material } },
        { binding: 4, resource: { buffer: buffers.x } },          // Solved displacements
        { binding: 5, resource: { buffer: buffers.momentMx } },
        { binding: 6, resource: { buffer: buffers.momentMy } },
        { binding: 7, resource: { buffer: buffers.momentMxy } },
        { binding: 8, resource: { buffer: buffers.momentCount } },
        { binding: 9, resource: { buffer: cpBuf } },
      ],
    })
  );

  // Averaging pass
  const bgAvgMoments = device.createBindGroup({
    layout: pipelines.averageMoments.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.momentMx } },
      { binding: 1, resource: { buffer: buffers.momentMy } },
      { binding: 2, resource: { buffer: buffers.momentMxy } },
      { binding: 3, resource: { buffer: buffers.momentCount } },
      { binding: 4, resource: { buffer: paramsNode } },
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
    bgZeroMoments, bgMomentColors, bgAvgMoments,
    bgDotRR,
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
  computeMomentsFlag: boolean = false,
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

  // ── Compute final residual norm r·r (always — near zero cost) ──
  pass = enc.beginComputePass();
  pass.setPipeline(pipelines.dotSingle);
  pass.setBindGroup(0, sctx.bgDotRR);
  pass.dispatchWorkgroups(1);
  pass.end();

  // ── Optional: GPU moment computation (same command buffer!) ──
  if (computeMomentsFlag) {
    const {
      bgZeroMoments, bgMomentColors, bgAvgMoments,
    } = sctx;
    const momentPipeline = isTriangleMesh ? pipelines.computeMomentsDKT : pipelines.computeMomentsQ4;

    // Zero moment accumulators
    for (const bg of bgZeroMoments) {
      pass = enc.beginComputePass();
      pass.setPipeline(pipelines.zeroBuffer);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgNode);
      pass.end();
    }

    // Accumulate moments per color (coloring prevents write conflicts)
    for (let c = 0; c < sctx.coloring.colorCount; c++) {
      if (colorCounts[c] === 0) continue;
      pass = enc.beginComputePass();
      pass.setPipeline(momentPipeline);
      pass.setBindGroup(0, bgMomentColors[c]);
      pass.dispatchWorkgroups(Math.ceil(colorCounts[c] / kpWgSize));
      pass.end();
    }

    // Average moments
    pass = enc.beginComputePass();
    pass.setPipeline(pipelines.averageMoments);
    pass.setBindGroup(0, bgAvgMoments);
    pass.dispatchWorkgroups(wgNode);
    pass.end();
  }

  // ── Copy solution and residual to staging IN THE SAME command buffer ──
  enc.copyBufferToBuffer(buffers.x, 0, buffers.stagingX, 0, dofCount * 4);
  enc.copyBufferToBuffer(buffers.rrBuf, 0, buffers.stagingDot, 0, 4);

  // Also copy moments to staging if computed
  if (computeMomentsFlag) {
    const momentBytes = nodeCount * 4;
    enc.copyBufferToBuffer(buffers.momentMx, 0, buffers.stagingMoments, 0, momentBytes);
    enc.copyBufferToBuffer(buffers.momentMy, 0, buffers.stagingMoments, momentBytes, momentBytes);
    enc.copyBufferToBuffer(buffers.momentMxy, 0, buffers.stagingMoments, momentBytes * 2, momentBytes);
  }

  // Single submit for everything
  device.queue.submit([enc.finish()]);

  // Map staging buffers (waits for GPU to complete)
  // Map solution and residual in parallel
  const mapPromises: Promise<void>[] = [
    buffers.stagingX.mapAsync(GPUMapMode.READ),
    buffers.stagingDot.mapAsync(GPUMapMode.READ),
  ];
  if (computeMomentsFlag) {
    mapPromises.push(buffers.stagingMoments.mapAsync(GPUMapMode.READ));
  }
  await Promise.all(mapPromises);

  const solution = new Float32Array(buffers.stagingX.getMappedRange().slice(0, dofCount * 4));
  buffers.stagingX.unmap();

  // Read residual norm
  const rrData = new Float32Array(buffers.stagingDot.getMappedRange().slice(0, 4));
  const finalResidual = Math.sqrt(Math.abs(rrData[0]));
  buffers.stagingDot.unmap();

  // Read moments if computed
  let Mx: Float32Array | undefined;
  let My: Float32Array | undefined;
  let Mxy: Float32Array | undefined;
  if (computeMomentsFlag) {
    const momData = new Float32Array(buffers.stagingMoments.getMappedRange().slice(0, nodeCount * 3 * 4));
    buffers.stagingMoments.unmap();
    Mx = new Float32Array(momData.buffer, 0, nodeCount);
    My = new Float32Array(momData.buffer, nodeCount * 4, nodeCount);
    Mxy = new Float32Array(momData.buffer, nodeCount * 8, nodeCount);
  }

  return {
    solution,
    iterations: maxIterations,
    finalResidual,
    converged: finalResidual < 1e-6,
    gpuTimeMs: performance.now() - startTime,
    usedGPU: true,
    Mx,
    My,
    Mxy,
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
      return await executeGPU(options.preparedContext, F, options.maxIterations ?? 1000, options.computeMoments ?? false);
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
    return await executeGPU(sctx, F, options.maxIterations ?? 1000, options.computeMoments ?? false);
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
