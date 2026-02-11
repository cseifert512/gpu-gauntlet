/**
 * WebGPU compute pipeline management for plate solver.
 *
 * Creates and manages all compute pipelines and bind groups.
 */

import type { GPUContext } from './context';
import type { PlateGPUBuffers } from './buffers';

// Import shader sources from TypeScript modules
import {
  dotProductSource,
  reduceSumSource,
  axpySource,
  scaleSource,
  copySource,
  preconditionerSource,
  blockPreconditionerSource,
  updatePSource,
  applyKQ4Source,
  applyKDKTSource,
  applyBCSource,
  zeroBufferSource,
  scalarDivSource,
  scalarNegDivSource,
  axpyBufSource,
  updatePBufSource,
  copyScalarSource,
  dotSingleSource,
  computeAlphaPairSource,
  spmvCSRSource,
  applyKPrecomputedQ4Source,
  applyKPrecomputedDKTSource,
  computeMomentsQ4Source,
  computeMomentsDKTSource,
  averageMomentsSource,
} from './shaders';

/**
 * All compute pipelines needed for the plate solver.
 */
export interface PlatePipelines {
  dotProduct: GPUComputePipeline;
  reduceSum: GPUComputePipeline;
  axpy: GPUComputePipeline;
  scale: GPUComputePipeline;
  copy: GPUComputePipeline;
  preconditioner: GPUComputePipeline;
  blockPreconditioner: GPUComputePipeline;
  updateP: GPUComputePipeline;
  applyKQ4: GPUComputePipeline;
  applyKDKT: GPUComputePipeline;
  applyBC: GPUComputePipeline;
  zeroBuffer: GPUComputePipeline;
  // Scalar ops (keep alpha/beta/rz on GPU — eliminates CPU-GPU round-trips)
  scalarDiv: GPUComputePipeline;
  scalarNegDiv: GPUComputePipeline;
  axpyBuf: GPUComputePipeline;
  updatePBuf: GPUComputePipeline;
  copyScalar: GPUComputePipeline;
  // Optimized single-dispatch ops (no multi-pass reduction)
  dotSingle: GPUComputePipeline;
  computeAlphaPair: GPUComputePipeline;
  // CSR SpMV (replaces element-by-element K·p)
  spmvCSR: GPUComputePipeline;
  // Pre-computed Ke K·p (avoids Gauss quadrature in hot loop)
  applyKPrecomputedQ4: GPUComputePipeline;
  applyKPrecomputedDKT: GPUComputePipeline;
  // GPU moment post-processing
  computeMomentsQ4: GPUComputePipeline;
  computeMomentsDKT: GPUComputePipeline;
  averageMoments: GPUComputePipeline;
}

/**
 * Create all compute pipelines.
 */
export async function createPipelines(ctx: GPUContext): Promise<PlatePipelines> {
  const { device } = ctx;

  const createPipeline = async (
    source: string,
    label: string
  ): Promise<GPUComputePipeline> => {
    const module = device.createShaderModule({
      code: source,
      label: `${label}_module`,
    });

    return device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module,
        entryPoint: 'main',
      },
      label: `${label}_pipeline`,
    });
  };

  // Create all pipelines in parallel
  const [
    dotProduct,
    reduceSum,
    axpy,
    scale,
    copy,
    preconditioner,
    blockPreconditioner,
    updateP,
    applyKQ4,
    applyKDKT,
    applyBC,
    zeroBuffer,
    scalarDiv,
    scalarNegDiv,
    axpyBuf,
    updatePBuf,
    copyScalar,
    dotSingle,
    computeAlphaPair,
    spmvCSR,
    applyKPrecomputedQ4,
    applyKPrecomputedDKT,
    computeMomentsQ4,
    computeMomentsDKT,
    averageMoments,
  ] = await Promise.all([
    createPipeline(dotProductSource, 'dot_product'),
    createPipeline(reduceSumSource, 'reduce_sum'),
    createPipeline(axpySource, 'axpy'),
    createPipeline(scaleSource, 'scale'),
    createPipeline(copySource, 'copy'),
    createPipeline(preconditionerSource, 'preconditioner'),
    createPipeline(blockPreconditionerSource, 'block_preconditioner'),
    createPipeline(updatePSource, 'update_p'),
    createPipeline(applyKQ4Source, 'apply_k_q4'),
    createPipeline(applyKDKTSource, 'apply_k_dkt'),
    createPipeline(applyBCSource, 'apply_bc'),
    createPipeline(zeroBufferSource, 'zero_buffer'),
    createPipeline(scalarDivSource, 'scalar_div'),
    createPipeline(scalarNegDivSource, 'scalar_neg_div'),
    createPipeline(axpyBufSource, 'axpy_buf'),
    createPipeline(updatePBufSource, 'update_p_buf'),
    createPipeline(copyScalarSource, 'copy_scalar'),
    createPipeline(dotSingleSource, 'dot_single'),
    createPipeline(computeAlphaPairSource, 'compute_alpha_pair'),
    createPipeline(spmvCSRSource, 'spmv_csr'),
    createPipeline(applyKPrecomputedQ4Source, 'apply_k_precomputed_q4'),
    createPipeline(applyKPrecomputedDKTSource, 'apply_k_precomputed_dkt'),
    createPipeline(computeMomentsQ4Source, 'compute_moments_q4'),
    createPipeline(computeMomentsDKTSource, 'compute_moments_dkt'),
    createPipeline(averageMomentsSource, 'average_moments'),
  ]);

  return {
    dotProduct,
    reduceSum,
    axpy,
    scale,
    copy,
    preconditioner,
    blockPreconditioner,
    updateP,
    applyKQ4,
    applyKDKT,
    applyBC,
    zeroBuffer,
    scalarDiv,
    scalarNegDiv,
    axpyBuf,
    updatePBuf,
    copyScalar,
    dotSingle,
    computeAlphaPair,
    spmvCSR,
    applyKPrecomputedQ4,
    applyKPrecomputedDKT,
    computeMomentsQ4,
    computeMomentsDKT,
    averageMoments,
  };
}

/**
 * Uniform buffer for single u32 params.
 */
export function createParamsBuffer(
  device: GPUDevice,
  size: number = 16
): GPUBuffer {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'params_uniform',
  });
}

/**
 * GPU dispatch helpers.
 */
export class GPUDispatcher {
  private device: GPUDevice;
  private pipelines: PlatePipelines;
  private buffers: PlateGPUBuffers;
  private paramsBuffer: GPUBuffer;

  constructor(
    ctx: GPUContext,
    pipelines: PlatePipelines,
    buffers: PlateGPUBuffers
  ) {
    this.device = ctx.device;
    this.pipelines = pipelines;
    this.buffers = buffers;
    this.paramsBuffer = createParamsBuffer(ctx.device, 16);
  }

  /**
   * Compute workgroup count for given problem size.
   */
  private workgroups(n: number, workgroupSize: number = 256): number {
    return Math.ceil(n / workgroupSize);
  }

  /**
   * Zero a buffer.
   */
  zeroBuffer(encoder: GPUCommandEncoder, buffer: GPUBuffer, count: number): void {
    const params = new Uint32Array([count, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.zeroBuffer.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.zeroBuffer);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(count));
    pass.end();
  }

  /**
   * Copy buffer: dst = src.
   */
  copyBuffer(
    encoder: GPUCommandEncoder,
    src: GPUBuffer,
    dst: GPUBuffer,
    count: number
  ): void {
    const params = new Uint32Array([count, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.copy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.copy);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(count));
    pass.end();
  }

  /**
   * AXPY: y = alpha * x + y.
   */
  axpy(
    encoder: GPUCommandEncoder,
    xBuffer: GPUBuffer,
    yBuffer: GPUBuffer,
    alpha: number,
    count: number
  ): void {
    const params = new Float32Array([alpha]);
    const paramsU32 = new Uint32Array([count, 0, 0]);
    const combined = new ArrayBuffer(16);
    new Float32Array(combined, 0, 1).set(params);
    new Uint32Array(combined, 4, 3).set(paramsU32);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, combined);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.axpy.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuffer } },
        { binding: 1, resource: { buffer: yBuffer } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.axpy);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(count));
    pass.end();
  }

  /**
   * Scale: x = alpha * x.
   */
  scale(
    encoder: GPUCommandEncoder,
    xBuffer: GPUBuffer,
    alpha: number,
    count: number
  ): void {
    const params = new Float32Array([alpha]);
    const paramsU32 = new Uint32Array([count, 0, 0]);
    const combined = new ArrayBuffer(16);
    new Float32Array(combined, 0, 1).set(params);
    new Uint32Array(combined, 4, 3).set(paramsU32);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, combined);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.scale.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.scale);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(count));
    pass.end();
  }

  /**
   * Apply Jacobi preconditioner: z = r / diag.
   */
  applyPreconditioner(encoder: GPUCommandEncoder): void {
    const n = this.buffers.dofCount;
    const params = new Uint32Array([n, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.preconditioner.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.r } },
        { binding: 1, resource: { buffer: this.buffers.diagonal } },
        { binding: 2, resource: { buffer: this.buffers.z } },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.preconditioner);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(n));
    pass.end();
  }

  /**
   * Apply block Jacobi preconditioner: z = M^-1 * r using 3x3 blocks.
   * Much more effective than scalar diagonal for plate bending.
   */
  applyBlockPreconditioner(encoder: GPUCommandEncoder): void {
    const nodeCount = this.buffers.nodeCount;
    const params = new Uint32Array([nodeCount, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.blockPreconditioner.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.r } },
        { binding: 1, resource: { buffer: this.buffers.blockDiagInv } },
        { binding: 2, resource: { buffer: this.buffers.z } },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.blockPreconditioner);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(nodeCount));
    pass.end();
  }

  /**
   * Update p: p = z + beta * p.
   */
  updateP(encoder: GPUCommandEncoder, beta: number): void {
    const n = this.buffers.dofCount;
    const params = new Float32Array([beta]);
    const paramsU32 = new Uint32Array([n, 0, 0]);
    const combined = new ArrayBuffer(16);
    new Float32Array(combined, 0, 1).set(params);
    new Uint32Array(combined, 4, 3).set(paramsU32);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, combined);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.updateP.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.z } },
        { binding: 1, resource: { buffer: this.buffers.p } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.updateP);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(n));
    pass.end();
  }

  /**
   * Compute dot product: result = a · b.
   * Returns the dot product value (requires GPU readback).
   */
  async computeDotProduct(
    aBuffer: GPUBuffer,
    bBuffer: GPUBuffer,
    count: number
  ): Promise<number> {
    const workgroups = this.workgroups(count);

    // Phase 1: Compute partial sums
    {
      const params = new Uint32Array([count, 0, 0, 0]);
      this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.dotProduct.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: this.buffers.dotPartial } },
          { binding: 3, resource: { buffer: this.paramsBuffer } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.dotProduct);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    // Phase 2: Final reduction (if needed)
    if (workgroups > 1) {
      const params = new Uint32Array([workgroups, 0, 0, 0]);
      this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

      const bindGroup = this.device.createBindGroup({
        layout: this.pipelines.reduceSum.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers.dotPartial } },
          { binding: 1, resource: { buffer: this.buffers.dotResult } },
          { binding: 2, resource: { buffer: this.paramsBuffer } },
        ],
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.reduceSum);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();

      // Copy result to staging
      encoder.copyBufferToBuffer(
        this.buffers.dotResult,
        0,
        this.buffers.stagingDot,
        0,
        4
      );
      this.device.queue.submit([encoder.finish()]);
    } else {
      // Single workgroup - copy directly from partial sums
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(
        this.buffers.dotPartial,
        0,
        this.buffers.stagingDot,
        0,
        4
      );
      this.device.queue.submit([encoder.finish()]);
    }

    // Readback
    await this.buffers.stagingDot.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(
      this.buffers.stagingDot.getMappedRange().slice(0, 4)
    )[0];
    this.buffers.stagingDot.unmap();

    return result;
  }

  /**
   * Apply K·x for a single color (Q4 elements).
   * y buffer should be zeroed before first color, then accumulated.
   */
  applyKQ4SingleColor(
    encoder: GPUCommandEncoder,
    colorIndex: number
  ): void {
    // Get color offset and count from coloring data
    const colorOffset = colorIndex === 0 ? 0 : this.getColorOffset(colorIndex);
    const colorCount = this.getColorCount(colorIndex);

    if (colorCount === 0) return;

    // Create params buffer for this dispatch
    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `apply_k_params_color_${colorIndex}`,
    });
    const params = new Uint32Array([colorOffset, colorCount, 0, 0]);
    this.device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.applyKQ4.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.nodes } },
        { binding: 1, resource: { buffer: this.buffers.elements } },
        { binding: 2, resource: { buffer: this.buffers.elementsByColor } },
        { binding: 3, resource: { buffer: this.buffers.material } },
        { binding: 4, resource: { buffer: this.buffers.p } }, // Input x (p for K·p)
        { binding: 5, resource: { buffer: this.buffers.Ap } }, // Output y (Ap)
        { binding: 6, resource: { buffer: this.buffers.constrainedMask } },
        { binding: 7, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.applyKQ4);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(colorCount, 64));
    pass.end();
  }

  /**
   * Apply K·x for a single color (DKT elements).
   */
  applyKDKTSingleColor(
    encoder: GPUCommandEncoder,
    colorIndex: number
  ): void {
    const colorOffset = colorIndex === 0 ? 0 : this.getColorOffset(colorIndex);
    const colorCount = this.getColorCount(colorIndex);

    if (colorCount === 0) return;

    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `apply_k_params_color_${colorIndex}`,
    });
    const params = new Uint32Array([colorOffset, colorCount, 0, 0]);
    this.device.queue.writeBuffer(paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.applyKDKT.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.nodes } },
        { binding: 1, resource: { buffer: this.buffers.elements } },
        { binding: 2, resource: { buffer: this.buffers.elementsByColor } },
        { binding: 3, resource: { buffer: this.buffers.material } },
        { binding: 4, resource: { buffer: this.buffers.p } },
        { binding: 5, resource: { buffer: this.buffers.Ap } },
        { binding: 6, resource: { buffer: this.buffers.constrainedMask } },
        { binding: 7, resource: { buffer: paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.applyKDKT);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(colorCount, 64));
    pass.end();
  }

  /**
   * Apply boundary conditions: y[constrained] = x[constrained].
   */
  applyBC(encoder: GPUCommandEncoder): void {
    const n = this.buffers.dofCount;
    const params = new Uint32Array([n, 0, 0, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipelines.applyBC.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.p } },
        { binding: 1, resource: { buffer: this.buffers.Ap } },
        { binding: 2, resource: { buffer: this.buffers.constrainedMask } },
        { binding: 3, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.applyBC);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.workgroups(n));
    pass.end();
  }

  /**
   * Get color offset (helper - needs color data from buffers).
   * In practice, this would read from the color offsets buffer,
   * but for simplicity we compute it incrementally.
   */
  private colorOffsets: number[] = [];
  private colorCounts: number[] = [];

  setColoringData(offsets: number[], counts: number[]): void {
    this.colorOffsets = offsets;
    this.colorCounts = counts;
  }

  private getColorOffset(colorIndex: number): number {
    return this.colorOffsets[colorIndex] ?? 0;
  }

  private getColorCount(colorIndex: number): number {
    return this.colorCounts[colorIndex] ?? 0;
  }

  /**
   * Cleanup.
   */
  destroy(): void {
    this.paramsBuffer.destroy();
  }
}

