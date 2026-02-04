/**
 * Single-color K·x test utilities.
 *
 * For validating GPU shader correctness against CPU reference.
 * This module allows testing individual color dispatches.
 */

import type { GPUContext } from './context';
import type { PlateGPUBuffers } from './buffers';
import type { PlatePipelines } from './pipelines';
import type { PlateMesh, PlateMaterial, ElementColoring } from '../types';
import { initGPU } from './context';
import { createPlateBuffers, destroyPlateBuffers, readFromBuffer } from './buffers';
import { createPipelines, createParamsBuffer } from './pipelines';
import { computeDiagonal, applyBCsToDiagonal, computeBlockDiagonal, invertBlockDiagonal } from '../solver';

/**
 * Result of single-color K·x operation.
 */
export interface SingleColorKxResult {
  /** Output vector y = K_color · x */
  y: Float32Array;
  /** GPU execution time in ms */
  gpuTimeMs: number;
  /** Whether GPU was used (vs CPU fallback) */
  usedGPU: boolean;
}

/**
 * Apply K·x for a single color batch on GPU.
 *
 * This is the key function for validating GPU shader correctness.
 * It computes y = K_color · x where K_color is the stiffness contribution
 * from all elements of a single color.
 *
 * @param mesh - Plate mesh
 * @param material - Material properties
 * @param coloring - Element coloring
 * @param x - Input vector (dofCount floats)
 * @param colorIndex - Which color to process (0 to colorCount-1)
 * @param constrainedDOFs - Constrained DOF indices (for BC handling)
 * @returns y vector and timing info
 */
export async function applySingleColorKx(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  x: Float32Array,
  colorIndex: number,
  constrainedDOFs: Set<number> = new Set()
): Promise<SingleColorKxResult> {
  const startTime = performance.now();

  // Initialize GPU
  const ctx = await initGPU();
  if (!ctx) {
    throw new Error('WebGPU not available');
  }

  const { device } = ctx;
  const dofCount = mesh.nodeCount * 3;
  const nodesPerElem = mesh.nodesPerElement ?? 4;

  // Compute preconditioners (needed for buffer creation)
  const diagonal = computeDiagonal(mesh, material);
  applyBCsToDiagonal(diagonal, constrainedDOFs);
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);

  // Create buffers
  const buffers = createPlateBuffers(
    ctx,
    mesh,
    coloring,
    material,
    diagonal,
    blockDiag,
    constrainedDOFs
  );

  // Upload input vector x
  device.queue.writeBuffer(buffers.p, 0, x.buffer, x.byteOffset, x.byteLength);

  // Create pipelines
  const pipelines = await createPipelines(ctx);

  // Get color offset and count
  let colorOffset = 0;
  for (let c = 0; c < colorIndex; c++) {
    colorOffset += coloring.colors[c].length;
  }
  const colorCount = coloring.colors[colorIndex].length;

  // Create params buffer
  const paramsBuffer = createParamsBuffer(device);
  const params = new Uint32Array([colorOffset, colorCount, 0, 0]);
  device.queue.writeBuffer(paramsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);

  // Select pipeline based on element type
  const pipeline = nodesPerElem === 3 ? pipelines.applyKDKT : pipelines.applyKQ4;

  // Create bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.nodes } },
      { binding: 1, resource: { buffer: buffers.elements } },
      { binding: 2, resource: { buffer: buffers.elementsByColor } },
      { binding: 3, resource: { buffer: buffers.material } },
      { binding: 4, resource: { buffer: buffers.p } },
      { binding: 5, resource: { buffer: buffers.Ap } },
      { binding: 6, resource: { buffer: buffers.constrainedMask } },
      { binding: 7, resource: { buffer: paramsBuffer } },
    ],
  });

  // Zero output buffer
  const zeroParams = new Uint32Array([dofCount, 0, 0, 0]);
  const zeroParamsBuffer = createParamsBuffer(device);
  device.queue.writeBuffer(zeroParamsBuffer, 0, zeroParams.buffer, zeroParams.byteOffset, zeroParams.byteLength);

  const zeroBindGroup = device.createBindGroup({
    layout: pipelines.zeroBuffer.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.Ap } },
      { binding: 1, resource: { buffer: zeroParamsBuffer } },
    ],
  });

  // Execute
  const encoder = device.createCommandEncoder();

  // Zero y
  const zeroPass = encoder.beginComputePass();
  zeroPass.setPipeline(pipelines.zeroBuffer);
  zeroPass.setBindGroup(0, zeroBindGroup);
  zeroPass.dispatchWorkgroups(Math.ceil(dofCount / 256));
  zeroPass.end();

  // Apply K·x for single color
  const workgroups = Math.ceil(colorCount / 64);
  const applyPass = encoder.beginComputePass();
  applyPass.setPipeline(pipeline);
  applyPass.setBindGroup(0, bindGroup);
  applyPass.dispatchWorkgroups(workgroups);
  applyPass.end();

  device.queue.submit([encoder.finish()]);

  // Read back result
  const y = await readFromBuffer(device, buffers.Ap, buffers.stagingX, dofCount);

  const gpuTimeMs = performance.now() - startTime;

  // Cleanup
  paramsBuffer.destroy();
  zeroParamsBuffer.destroy();
  destroyPlateBuffers(buffers);

  return {
    y,
    gpuTimeMs,
    usedGPU: true,
  };
}

/**
 * CPU reference implementation of single-color K·x.
 *
 * Used for validation against GPU results.
 */
import { computeElementStiffness, computeDKTStiffness } from '../element';
import { getElementCoords, getElementNodeIndices } from '../mesher';

export function applySingleColorKxCPU(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  x: Float32Array,
  colorIndex: number,
  constrainedDOFs: Set<number> = new Set()
): Float32Array {

  const dofCount = mesh.nodeCount * 3;
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofsPerElem = nodesPerElem * 3;

  const y = new Float32Array(dofCount);

  // Get elements for this color
  const elementIndices = coloring.colors[colorIndex];

  for (let i = 0; i < elementIndices.length; i++) {
    const elemIdx = elementIndices[i];
    const nodeIndices = getElementNodeIndices(mesh, elemIdx);
    const coords = getElementCoords(mesh, elemIdx);

    // Compute element stiffness
    const Ke =
      nodesPerElem === 3
        ? computeDKTStiffness(coords, material)
        : computeElementStiffness(coords, material);

    // Build DOF index array
    const dofs: number[] = [];
    for (let n = 0; n < nodesPerElem; n++) {
      const baseGlobalDOF = nodeIndices[n] * 3;
      dofs.push(baseGlobalDOF, baseGlobalDOF + 1, baseGlobalDOF + 2);
    }

    // Gather local x
    const xLocal = new Float32Array(dofsPerElem);
    for (let j = 0; j < dofsPerElem; j++) {
      xLocal[j] = x[dofs[j]];
    }

    // Compute yLocal = Ke * xLocal
    const yLocal = new Float32Array(dofsPerElem);
    for (let row = 0; row < dofsPerElem; row++) {
      let sum = 0;
      for (let col = 0; col < dofsPerElem; col++) {
        sum += Ke[row * dofsPerElem + col] * xLocal[col];
      }
      yLocal[row] = sum;
    }

    // Scatter to global y (skip constrained DOFs)
    for (let j = 0; j < dofsPerElem; j++) {
      const globalDOF = dofs[j];
      if (!constrainedDOFs.has(globalDOF)) {
        y[globalDOF] += yLocal[j];
      }
    }
  }

  return y;
}

/**
 * Compare GPU and CPU results.
 *
 * @param gpuResult - GPU result vector
 * @param cpuResult - CPU reference vector
 * @param tolerance - Maximum relative error allowed
 * @returns Object with comparison statistics
 */
export function compareResults(
  gpuResult: Float32Array,
  cpuResult: Float32Array,
  tolerance: number = 1e-6
): {
  maxAbsError: number;
  maxRelError: number;
  indexOfMaxError: number;
  passed: boolean;
  mismatches: number;
} {
  let maxAbsError = 0;
  let maxRelError = 0;
  let indexOfMaxError = 0;
  let mismatches = 0;

  for (let i = 0; i < gpuResult.length; i++) {
    const gpu = gpuResult[i];
    const cpu = cpuResult[i];
    const absError = Math.abs(gpu - cpu);
    const relError = Math.abs(cpu) > 1e-15 ? absError / Math.abs(cpu) : absError;

    if (absError > maxAbsError) {
      maxAbsError = absError;
      indexOfMaxError = i;
    }
    if (relError > maxRelError) {
      maxRelError = relError;
    }
    if (relError > tolerance && absError > tolerance) {
      mismatches++;
    }
  }

  return {
    maxAbsError,
    maxRelError,
    indexOfMaxError,
    passed: maxRelError < tolerance || maxAbsError < tolerance,
    mismatches,
  };
}

