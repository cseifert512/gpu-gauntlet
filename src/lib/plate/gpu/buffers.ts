/**
 * GPU buffer management for plate solver.
 *
 * All buffers use Float32Array/Uint32Array for WebGPU compatibility.
 * Buffers are pre-allocated and reused between solves.
 */

import type { GPUContext } from './context';
import type { PlateMesh, ElementColoring, PlateMaterial } from '../types';
import { computeFlexuralRigidity } from '../types';

/**
 * GPU buffers for plate solver.
 */
export interface PlateGPUBuffers {
  // Mesh data (read-only)
  nodes: GPUBuffer; // [x0,y0, x1,y1, ...] (2 × nodeCount floats)
  elements: GPUBuffer; // [n0,n1,n2,n3, ...] or [n0,n1,n2, ...] (connectivity)

  // Coloring data
  colorOffsets: GPUBuffer; // Start index for each color
  colorCounts: GPUBuffer; // Element count for each color
  elementsByColor: GPUBuffer; // Element indices sorted by color

  // Material uniform
  material: GPUBuffer; // [E, nu, t, D, kappa*G*t] (uniform buffer)

  // Solver vectors (read-write)
  x: GPUBuffer; // Current solution (dofCount floats)
  r: GPUBuffer; // Residual
  z: GPUBuffer; // Preconditioned residual
  p: GPUBuffer; // Search direction
  Ap: GPUBuffer; // K·p result
  diagonal: GPUBuffer; // Preconditioner diagonal (scalar)
  blockDiagInv: GPUBuffer; // Block diagonal inverse (9 floats per node)

  // Constrained DOFs (for BC handling)
  constrainedMask: GPUBuffer; // 1 if constrained, 0 otherwise

  // Reduction buffers
  dotPartial: GPUBuffer; // Partial sums from workgroups
  dotResult: GPUBuffer; // Final dot product result

  // Staging buffers (for readback)
  stagingX: GPUBuffer; // For reading solution back to CPU
  stagingDot: GPUBuffer; // For reading dot products

  // Scalar buffers (for GPU-only PCG — avoid CPU readbacks)
  rzBuf: GPUBuffer;      // r·z (current)
  rzNewBuf: GPUBuffer;   // r·z (next iteration)
  pApBuf: GPUBuffer;     // p·Ap
  alphaBuf: GPUBuffer;   // alpha = rz / pAp
  negAlphaBuf: GPUBuffer; // -alpha = -(rz / pAp)
  betaBuf: GPUBuffer;    // beta = rz_new / rz
  rrBuf: GPUBuffer;      // r·r (for convergence check)

  // Metadata (not GPU buffers)
  nodeCount: number;
  elementCount: number;
  dofCount: number; // 3 × nodeCount
  nodesPerElement: number;
  colorCount: number;
}

/**
 * Create a GPU buffer with initial data.
 */
function createBufferWithData(
  device: GPUDevice,
  data: ArrayBuffer | ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string
): GPUBuffer {
  const byteLength =
    data instanceof ArrayBuffer ? data.byteLength : data.byteLength;
  const alignedSize = Math.ceil(byteLength / 4) * 4; // Align to 4 bytes

  const buffer = device.createBuffer({
    size: Math.max(alignedSize, 4), // Minimum 4 bytes
    usage,
    mappedAtCreation: true,
    label,
  });

  const src =
    data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  new Uint8Array(buffer.getMappedRange()).set(src);
  buffer.unmap();

  return buffer;
}

/**
 * Create an empty GPU buffer of specified size.
 */
function createEmptyBuffer(
  device: GPUDevice,
  byteSize: number,
  usage: GPUBufferUsageFlags,
  label?: string
): GPUBuffer {
  return device.createBuffer({
    size: Math.max(Math.ceil(byteSize / 4) * 4, 4), // Align to 4 bytes, min 4
    usage,
    label,
  });
}

/**
 * Create GPU buffers for plate solve.
 *
 * @param ctx - GPU context
 * @param mesh - Plate mesh
 * @param coloring - Element coloring
 * @param material - Material properties
 * @param diagonal - Preconditioner diagonal (for fallback)
 * @param blockDiagInv - Block diagonal inverse (9 floats per node)
 * @param constrainedDOFs - Set of constrained DOF indices
 */
export function createPlateBuffers(
  ctx: GPUContext,
  mesh: PlateMesh,
  coloring: ElementColoring,
  material: PlateMaterial,
  diagonal: Float32Array,
  blockDiagInv: Float32Array,
  constrainedDOFs: Set<number>
): PlateGPUBuffers {
  const { device } = ctx;
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofCount = mesh.nodeCount * 3;

  // Mesh buffers (read-only storage)
  const nodes = createBufferWithData(
    device,
    mesh.nodes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'nodes'
  );

  const elements = createBufferWithData(
    device,
    mesh.elements,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    'elements'
  );

  // Coloring buffers
  // Flatten colors into single array sorted by color
  const totalElements = coloring.colors.reduce((sum, c) => sum + c.length, 0);
  const elementsByColorData = new Uint32Array(totalElements);
  const colorOffsetsData = new Uint32Array(coloring.colorCount);
  const colorCountsData = new Uint32Array(coloring.colorCount);

  let offset = 0;
  for (let c = 0; c < coloring.colorCount; c++) {
    colorOffsetsData[c] = offset;
    colorCountsData[c] = coloring.colors[c].length;
    elementsByColorData.set(coloring.colors[c], offset);
    offset += coloring.colors[c].length;
  }

  const colorOffsets = createBufferWithData(
    device,
    colorOffsetsData,
    GPUBufferUsage.STORAGE,
    'colorOffsets'
  );

  const colorCounts = createBufferWithData(
    device,
    colorCountsData,
    GPUBufferUsage.STORAGE,
    'colorCounts'
  );

  const elementsByColor = createBufferWithData(
    device,
    elementsByColorData,
    GPUBufferUsage.STORAGE,
    'elementsByColor'
  );

  // Material uniform buffer
  // [E, nu, t, D, kappa*G*t]
  const D = computeFlexuralRigidity(material);
  const kappa = 5.0 / 6.0;
  const G = material.E / (2 * (1 + material.nu));
  const kappaGt = kappa * G * material.t;
  const materialData = new Float32Array([
    material.E,
    material.nu,
    material.t,
    D,
    kappaGt,
    0, // padding for 16-byte alignment
    0,
    0,
  ]);

  const materialBuffer = createBufferWithData(
    device,
    materialData,
    GPUBufferUsage.UNIFORM,
    'material'
  );

  // Solver vectors
  const vectorUsage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const vectorSize = dofCount * 4; // Float32

  const x = createEmptyBuffer(device, vectorSize, vectorUsage, 'x');
  const r = createEmptyBuffer(device, vectorSize, vectorUsage, 'r');
  const z = createEmptyBuffer(device, vectorSize, vectorUsage, 'z');
  const p = createEmptyBuffer(device, vectorSize, vectorUsage, 'p');
  const Ap = createEmptyBuffer(device, vectorSize, vectorUsage, 'Ap');

  const diagonalBuffer = createBufferWithData(
    device,
    diagonal,
    GPUBufferUsage.STORAGE,
    'diagonal'
  );

  // Block diagonal inverse (9 floats per node, already inverted)
  const blockDiagInvBuffer = createBufferWithData(
    device,
    blockDiagInv,
    GPUBufferUsage.STORAGE,
    'blockDiagInv'
  );

  // Constrained DOFs mask
  const constrainedMaskData = new Uint32Array(dofCount);
  for (const dof of constrainedDOFs) {
    constrainedMaskData[dof] = 1;
  }
  const constrainedMask = createBufferWithData(
    device,
    constrainedMaskData,
    GPUBufferUsage.STORAGE,
    'constrainedMask'
  );

  // Reduction buffers
  // Workgroup size 256, max workgroups = ceil(dofCount / 256)
  const maxWorkgroups = Math.ceil(dofCount / 256);
  const dotPartial = createEmptyBuffer(
    device,
    maxWorkgroups * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    'dotPartial'
  );

  const dotResult = createEmptyBuffer(
    device,
    4, // Single float
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    'dotResult'
  );

  // Staging buffers for CPU readback
  const stagingX = createEmptyBuffer(
    device,
    vectorSize,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    'stagingX'
  );

  const stagingDot = createEmptyBuffer(
    device,
    4,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    'stagingDot'
  );

  // Scalar GPU buffers for batched PCG (no CPU readback needed)
  const scalarUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const rzBuf = createEmptyBuffer(device, 4, scalarUsage, 'rzBuf');
  const rzNewBuf = createEmptyBuffer(device, 4, scalarUsage, 'rzNewBuf');
  const pApBuf = createEmptyBuffer(device, 4, scalarUsage, 'pApBuf');
  const alphaBuf = createEmptyBuffer(device, 4, scalarUsage, 'alphaBuf');
  const negAlphaBuf = createEmptyBuffer(device, 4, scalarUsage, 'negAlphaBuf');
  const betaBuf = createEmptyBuffer(device, 4, scalarUsage, 'betaBuf');
  const rrBuf = createEmptyBuffer(device, 4, scalarUsage, 'rrBuf');

  return {
    nodes,
    elements,
    colorOffsets,
    colorCounts,
    elementsByColor,
    material: materialBuffer,
    x,
    r,
    z,
    p,
    Ap,
    diagonal: diagonalBuffer,
    blockDiagInv: blockDiagInvBuffer,
    constrainedMask,
    dotPartial,
    dotResult,
    stagingX,
    stagingDot,
    rzBuf,
    rzNewBuf,
    pApBuf,
    alphaBuf,
    negAlphaBuf,
    betaBuf,
    rrBuf,
    nodeCount: mesh.nodeCount,
    elementCount: mesh.elementCount,
    dofCount,
    nodesPerElement: nodesPerElem,
    colorCount: coloring.colorCount,
  };
}

/**
 * Upload data to a GPU buffer.
 */
export function uploadToBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: Float32Array | Uint32Array
): void {
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Read data from a GPU buffer via staging buffer.
 *
 * @param device - GPU device
 * @param srcBuffer - Source buffer to read from
 * @param stagingBuffer - Staging buffer for readback
 * @param count - Number of elements to read
 */
export async function readFromBuffer(
  device: GPUDevice,
  srcBuffer: GPUBuffer,
  stagingBuffer: GPUBuffer,
  count: number
): Promise<Float32Array> {
  // Copy to staging buffer
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, stagingBuffer, 0, count * 4);
  device.queue.submit([encoder.finish()]);

  // Map and read
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(stagingBuffer.getMappedRange().slice(0, count * 4));
  stagingBuffer.unmap();

  return data;
}

/**
 * Destroy all GPU buffers.
 */
export function destroyPlateBuffers(buffers: PlateGPUBuffers): void {
  const bufferKeys: (keyof PlateGPUBuffers)[] = [
    'nodes',
    'elements',
    'colorOffsets',
    'colorCounts',
    'elementsByColor',
    'material',
    'x',
    'r',
    'z',
    'p',
    'Ap',
    'diagonal',
    'blockDiagInv',
    'constrainedMask',
    'dotPartial',
    'dotResult',
    'stagingX',
    'stagingDot',
    'rzBuf',
    'rzNewBuf',
    'pApBuf',
    'alphaBuf',
    'negAlphaBuf',
    'betaBuf',
    'rrBuf',
  ];

  for (const key of bufferKeys) {
    const value = buffers[key];
    if (value instanceof GPUBuffer) {
      value.destroy();
    }
  }
}

