/**
 * PlateAnalyzer — High-level integration class for the plate solver engine.
 *
 * Manages the complete lifecycle of mesh generation, GPU resource preparation,
 * solving, post-processing, and result extraction. Designed for real-time
 * interactive applications where:
 *
 *   - Geometry/supports change infrequently (triggers full re-preparation)
 *   - Loads change frequently (triggers only GPU solve, ~13ms)
 *
 * Usage:
 * ```typescript
 * const analyzer = new PlateAnalyzer();
 *
 * // Setup (once per geometry change, ~50ms)
 * await analyzer.setup(geometry, material, supports, lineSupports, { meshSize: 0.5 });
 *
 * // Solve (per load case, ~13ms for 100k DOF)
 * const result = await analyzer.solve(loads);
 *
 * // Access results
 * console.log(result.maxDeflection, result.Mx, result.My);
 *
 * // Generate isocurves for visualization
 * const contours = analyzer.getIsocurves(result.w, { levels: 15 });
 *
 * // Cleanup
 * analyzer.destroy();
 * ```
 */

import type {
  PlateGeometry,
  PlateMaterial,
  PlateSupport,
  PlateLineSupport,
  PlateLoad,
  PlateMesh,
  ElementColoring,
} from './types';
import { DOFS_PER_NODE } from './types';
import { generateMesh } from './mesher';
import { computeElementColoring, computeElementColoringGreedy } from './coloring';
import {
  identifyConstrainedDOFs,
  computeBlockDiagonal,
  invertBlockDiagonal,
  buildLoadVector,
  applyBCsToRHS,
} from './solver';
import { resolveLineSupports } from './line-supports';
import { computeMoments, extractVerticalDisplacements, findMaxDisplacement, findMaxMoments } from './postprocess';
import { generateIsocurves, computeIsoValues } from './isocurves';
import type { ContourLevel, IsocurveOptions } from './isocurves';
import {
  isWebGPUAvailable,
  prepareGPUSolver,
  solveGPU,
  destroyGPUSolverContext,
} from './gpu';
import type { GPUSolverContext, GPUSolveResult } from './gpu';

/** Configuration for the analyzer setup. */
export interface AnalyzerSetupOptions {
  /** Target element size in meters (default: 0.5) */
  meshSize?: number;
  /** PCG iteration count for GPU solver (default: 25) */
  maxIterations?: number;
  /** Compute moments on GPU (default: true) */
  gpuMoments?: boolean;
  /** Snap tolerance for line supports (default: meshSize/4) */
  lineSupportTolerance?: number;
}

/** Full result from a plate analysis. */
export interface AnalysisResult {
  /** Full displacement vector [w0,θx0,θy0, w1,θx1,θy1, ...] */
  displacements: Float32Array;
  /** Vertical displacements only (one per node) */
  w: Float32Array;
  /** Bending moment Mx (one per node) */
  Mx: Float32Array;
  /** Bending moment My (one per node) */
  My: Float32Array;
  /** Twisting moment Mxy (one per node) */
  Mxy: Float32Array;
  /** Maximum absolute deflection value */
  maxDeflection: number;
  /** Node index of maximum deflection */
  maxDeflectionNode: number;
  /** Maximum moment values */
  maxMx: number;
  maxMy: number;
  maxMxy: number;
  /** Solve timing */
  solveTimeMs: number;
  /** Residual norm (from GPU) */
  residualNorm: number;
  /** Whether GPU was used */
  usedGPU: boolean;
  /** PCG iteration count */
  iterations: number;
}

/**
 * High-level plate analysis engine.
 *
 * Manages GPU resources, mesh, supports, and provides a clean API
 * for interactive real-time structural analysis.
 */
export class PlateAnalyzer {
  private gpuCtx: GPUSolverContext | null = null;
  private mesh: PlateMesh | null = null;
  private material: PlateMaterial | null = null;
  private coloring: ElementColoring | null = null;
  private constrainedDOFs: Set<number> | null = null;
  private blockDiag: Float32Array | null = null;
  private maxIterations: number = 25;
  private gpuMoments: boolean = true;
  private _isReady: boolean = false;

  /** Whether the analyzer is set up and ready for solving. */
  get isReady(): boolean {
    return this._isReady;
  }

  /** The current mesh (null if not set up). */
  get currentMesh(): PlateMesh | null {
    return this.mesh;
  }

  /** Number of DOFs in the current mesh. */
  get dofCount(): number {
    return this.mesh ? this.mesh.nodeCount * DOFS_PER_NODE : 0;
  }

  /**
   * Set up the analyzer for a new geometry.
   *
   * This generates the mesh, computes element coloring, identifies constrained
   * DOFs, builds the preconditioner, and prepares GPU resources. Should be
   * called once per geometry change.
   *
   * @param geometry - Plate boundary and holes
   * @param material - Material properties
   * @param pointSupports - Point support definitions
   * @param lineSupports - Line support definitions (resolved to mesh nodes)
   * @param options - Setup configuration
   */
  async setup(
    geometry: PlateGeometry,
    material: PlateMaterial,
    pointSupports: PlateSupport[] = [],
    lineSupports: PlateLineSupport[] = [],
    options: AnalyzerSetupOptions = {}
  ): Promise<void> {
    // Cleanup previous resources
    this.destroy();

    const meshSize = options.meshSize ?? 0.5;
    this.maxIterations = options.maxIterations ?? 25;
    this.gpuMoments = options.gpuMoments !== false;
    this.material = material;

    // 1. Generate mesh
    this.mesh = generateMesh(geometry, meshSize);

    // 2. Resolve line supports to point supports
    const resolvedSupports = resolveLineSupports(
      this.mesh,
      lineSupports,
      options.lineSupportTolerance ?? meshSize / 4
    );
    const allSupports = [...pointSupports, ...resolvedSupports];

    // 3. Compute element coloring
    const nodesPerElem = this.mesh.nodesPerElement ?? 4;
    this.coloring =
      nodesPerElem === 3 || this.mesh.gridNx === 0
        ? computeElementColoringGreedy(this.mesh)
        : computeElementColoring(this.mesh);

    // 4. Identify constrained DOFs
    this.constrainedDOFs = identifyConstrainedDOFs(this.mesh, allSupports);

    // 5. Build and invert block diagonal preconditioner
    this.blockDiag = computeBlockDiagonal(this.mesh, material);
    invertBlockDiagonal(this.blockDiag, this.constrainedDOFs);

    // 6. Prepare GPU resources
    if (isWebGPUAvailable()) {
      this.gpuCtx = await prepareGPUSolver(
        this.mesh,
        material,
        this.coloring,
        this.constrainedDOFs,
        this.blockDiag
      );
    }

    this._isReady = true;
  }

  /**
   * Solve for a given set of loads.
   *
   * This is the fast path (~13ms for 100k DOF). Call after setup().
   * The geometry, mesh, and GPU resources are reused.
   *
   * @param loads - Point loads to apply
   * @returns Full analysis result with deflections, moments, and metadata
   */
  async solve(loads: PlateLoad[]): Promise<AnalysisResult> {
    if (!this._isReady || !this.mesh || !this.material || !this.coloring || !this.constrainedDOFs || !this.blockDiag) {
      throw new Error('PlateAnalyzer: call setup() before solve()');
    }

    // Build load vector
    const F = buildLoadVector(this.mesh, loads);
    applyBCsToRHS(F, this.constrainedDOFs);

    // Solve
    const result = await solveGPU(
      this.mesh,
      this.material,
      this.coloring,
      F,
      this.constrainedDOFs,
      {
        maxIterations: this.maxIterations,
        preparedContext: this.gpuCtx ?? undefined,
        precomputedBlockDiagInv: this.blockDiag,
        computeMoments: this.gpuMoments,
      }
    );

    // Extract results
    const w = extractVerticalDisplacements(result.solution, this.mesh.nodeCount);

    // Moments: prefer GPU-computed, fall back to CPU
    let Mx: Float32Array;
    let My: Float32Array;
    let Mxy: Float32Array;
    if (result.Mx && result.My && result.Mxy) {
      Mx = result.Mx;
      My = result.My;
      Mxy = result.Mxy;
    } else {
      const moments = computeMoments(this.mesh, result.solution, this.material);
      Mx = moments.Mx;
      My = moments.My;
      Mxy = moments.Mxy;
    }

    const maxW = findMaxDisplacement(w, this.mesh.nodes);
    const maxMoments = findMaxMoments(Mx, My, Mxy);

    return {
      displacements: result.solution,
      w,
      Mx,
      My,
      Mxy,
      maxDeflection: maxW.maxW,
      maxDeflectionNode: maxW.nodeIndex,
      maxMx: maxMoments.maxMx,
      maxMy: maxMoments.maxMy,
      maxMxy: maxMoments.maxMxy,
      solveTimeMs: result.gpuTimeMs,
      residualNorm: result.finalResidual,
      usedGPU: result.usedGPU,
      iterations: result.iterations,
    };
  }

  /**
   * Generate isocurves for a scalar field over the current mesh.
   *
   * Can be called for deflection (w), moments (Mx, My, Mxy), or any
   * per-node scalar field.
   *
   * @param field - Scalar field values (length = nodeCount)
   * @param options - Isocurve generation options
   * @returns Array of contour levels with polylines
   */
  getIsocurves(field: Float32Array, options?: IsocurveOptions): ContourLevel[] {
    if (!this.mesh) throw new Error('PlateAnalyzer: call setup() first');
    return generateIsocurves(this.mesh, field, options);
  }

  /**
   * Compute evenly-spaced iso-values for a field.
   */
  getIsoValues(field: Float32Array, levels: number = 20): number[] {
    return computeIsoValues(field, levels);
  }

  /**
   * Release all GPU resources and reset state.
   */
  destroy(): void {
    if (this.gpuCtx) {
      destroyGPUSolverContext(this.gpuCtx);
      this.gpuCtx = null;
    }
    this.mesh = null;
    this.material = null;
    this.coloring = null;
    this.constrainedDOFs = null;
    this.blockDiag = null;
    this._isReady = false;
  }
}

