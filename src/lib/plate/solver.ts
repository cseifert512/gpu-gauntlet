/**
 * ============================================================================
 * CPU REFERENCE IMPLEMENTATION - DO NOT MODIFY
 * ============================================================================
 * 
 * This file is the CPU reference solver used for validation.
 * Your GPU optimizations in gpu/solver.ts must produce results that match
 * this implementation within the specified tolerance.
 * 
 * If you modify this file, your submission will be disqualified.
 * ============================================================================
 *
 * Main plate solver orchestration.
 *
 * Workflow:
 * 1. Generate mesh
 * 2. Compute element coloring
 * 3. Identify constrained DOFs
 * 4. Build diagonal preconditioner
 * 5. Build load vector
 * 6. Solve via PCG (matrix-free)
 * 7. Post-process for moments
 */

import type {
  PlateGeometry,
  PlateMaterial,
  PlateSupport,
  PlateLoad,
  PlateMesh,
  ElementColoring,
  PlateResult,
  SolvePlateOptions,
} from './types';
import { DOFS_PER_NODE, DOFS_PER_ELEMENT, DOFS_PER_TRIANGLE } from './types';
import {
  generateMesh,
  getElementCoords,
  getElementNodes,
  getElementNodeIndices,
} from './mesher';
import { computeElementColoring, computeElementColoringGreedy } from './coloring';
import { computeElementStiffness, computeDKTStiffness } from './element';
import { solvePCG } from './pcg';
import { computeMoments, extractVerticalDisplacements } from './postprocess';

// === ELEMENT STIFFNESS CACHE ===
// Critical fix: Cache Ke matrices to avoid recomputation in PCG loop
let cachedKe: Map<number, Float32Array> | null = null;
let cachedMesh: PlateMesh | null = null;
let cachedMaterial: PlateMaterial | null = null;

function getOrComputeKe(
  mesh: PlateMesh,
  material: PlateMaterial,
  elemIdx: number
): Float32Array {
  // Invalidate cache if mesh/material changed
  if (cachedMesh !== mesh || cachedMaterial !== material) {
    cachedKe = new Map();
    cachedMesh = mesh;
    cachedMaterial = material;
  }

  if (!cachedKe!.has(elemIdx)) {
    const coords = getElementCoords(mesh, elemIdx);
    const nodesPerElem = mesh.nodesPerElement ?? 4;
    const Ke =
      nodesPerElem === 3
        ? computeDKTStiffness(coords, material)
        : computeElementStiffness(coords, material);
    cachedKe!.set(elemIdx, Ke);
  }

  return cachedKe!.get(elemIdx)!;
}

/** Clear element stiffness cache (call after solve completes) */
export function clearKeCache(): void {
  cachedKe = null;
  cachedMesh = null;
  cachedMaterial = null;
}

/**
 * Identify constrained DOFs based on support conditions.
 *
 * @param mesh - Plate mesh
 * @param supports - Support definitions
 * @returns Set of constrained DOF indices
 */
export function identifyConstrainedDOFs(
  mesh: PlateMesh,
  supports: PlateSupport[]
): Set<number> {
  const constrained = new Set<number>();

  for (const support of supports) {
    let nodeIndices: number[];

    if (support.location === 'all_edges') {
      // All boundary nodes
      nodeIndices = Array.from(mesh.boundaryNodes);
    } else {
      // Specific node
      nodeIndices = [support.location];
    }

    for (const nodeIdx of nodeIndices) {
      const baseDOF = nodeIdx * DOFS_PER_NODE;

      switch (support.type) {
        case 'pinned':
          // Constrain w only (vertical displacement)
          constrained.add(baseDOF); // w
          break;

        case 'fixed':
          // Constrain all DOFs (w, θx, θy)
          constrained.add(baseDOF); // w
          constrained.add(baseDOF + 1); // θx
          constrained.add(baseDOF + 2); // θy
          break;

        case 'roller':
          // Constrain w only (same as pinned for plates)
          constrained.add(baseDOF); // w
          break;
      }
    }
  }

  return constrained;
}

/**
 * Compute diagonal of global stiffness matrix (for preconditioner).
 *
 * Supports both Q4 (quad) and DKT (triangle) elements.
 *
 * @param mesh - Plate mesh
 * @param material - Material properties
 * @returns Diagonal values
 */
export function computeDiagonal(
  mesh: PlateMesh,
  material: PlateMaterial
): Float32Array {
  const nDOF = mesh.nodeCount * DOFS_PER_NODE;
  const diagonal = new Float32Array(nDOF);
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofsPerElem = nodesPerElem * DOFS_PER_NODE;

  for (let elemIdx = 0; elemIdx < mesh.elementCount; elemIdx++) {
    const coords = getElementCoords(mesh, elemIdx);

    // Use appropriate element formulation
    const Ke =
      nodesPerElem === 3
        ? computeDKTStiffness(coords, material)
        : computeElementStiffness(coords, material);

    const nodeIndices = getElementNodeIndices(mesh, elemIdx);

    // Accumulate diagonal contributions
    for (let i = 0; i < nodesPerElem; i++) {
      const globalNode = nodeIndices[i];
      for (let d = 0; d < DOFS_PER_NODE; d++) {
        const localDOF = i * DOFS_PER_NODE + d;
        const globalDOF = globalNode * DOFS_PER_NODE + d;
        diagonal[globalDOF] += Ke[localDOF * dofsPerElem + localDOF];
      }
    }
  }

  return diagonal;
}

/**
 * Compute block diagonal (3x3 per node) of global stiffness matrix.
 * 
 * Block Jacobi preconditioner is much more effective than scalar diagonal
 * for plate bending since the 3 DOFs per node (w, θx, θy) are coupled.
 *
 * Returns: Float32Array of length nodeCount * 9 (row-major 3x3 blocks)
 * Each block is stored as [a00, a01, a02, a10, a11, a12, a20, a21, a22]
 */
export function computeBlockDiagonal(
  mesh: PlateMesh,
  material: PlateMaterial
): Float32Array {
  const nodeCount = mesh.nodeCount;
  const blockDiag = new Float32Array(nodeCount * 9); // 9 entries per 3x3 block
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofsPerElem = nodesPerElem * DOFS_PER_NODE;

  for (let elemIdx = 0; elemIdx < mesh.elementCount; elemIdx++) {
    const coords = getElementCoords(mesh, elemIdx);

    const Ke =
      nodesPerElem === 3
        ? computeDKTStiffness(coords, material)
        : computeElementStiffness(coords, material);

    const nodeIndices = getElementNodeIndices(mesh, elemIdx);

    // Accumulate 3x3 block contributions for each node
    for (let i = 0; i < nodesPerElem; i++) {
      const globalNode = nodeIndices[i];
      const blockOffset = globalNode * 9;
      const localBase = i * DOFS_PER_NODE;

      // Extract 3x3 sub-block from Ke and add to global block diagonal
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const localRow = localBase + r;
          const localCol = localBase + c;
          blockDiag[blockOffset + r * 3 + c] += Ke[localRow * dofsPerElem + localCol];
        }
      }
    }
  }

  return blockDiag;
}

/**
 * Invert 3x3 block diagonal in place.
 * After calling this, blockDiag contains the INVERSE of each 3x3 block.
 */
export function invertBlockDiagonal(
  blockDiag: Float32Array,
  constrainedDOFs: Set<number>
): void {
  const nodeCount = blockDiag.length / 9;

  for (let node = 0; node < nodeCount; node++) {
    const offset = node * 9;
    const baseDOF = node * 3;

    // Check if any DOF in this node is constrained
    const w_constrained = constrainedDOFs.has(baseDOF);
    const tx_constrained = constrainedDOFs.has(baseDOF + 1);
    const ty_constrained = constrainedDOFs.has(baseDOF + 2);

    // If all constrained, use identity
    if (w_constrained && tx_constrained && ty_constrained) {
      blockDiag[offset + 0] = 1; blockDiag[offset + 1] = 0; blockDiag[offset + 2] = 0;
      blockDiag[offset + 3] = 0; blockDiag[offset + 4] = 1; blockDiag[offset + 5] = 0;
      blockDiag[offset + 6] = 0; blockDiag[offset + 7] = 0; blockDiag[offset + 8] = 1;
      continue;
    }

    // For partially constrained nodes, zero out off-diagonal terms
    // involving constrained DOFs and set diagonal to 1
    if (w_constrained) {
      blockDiag[offset + 0] = 1; // M[0,0] = 1
      blockDiag[offset + 1] = 0; blockDiag[offset + 2] = 0; // Row 0
      blockDiag[offset + 3] = 0; blockDiag[offset + 6] = 0; // Col 0
    }
    if (tx_constrained) {
      blockDiag[offset + 4] = 1; // M[1,1] = 1
      blockDiag[offset + 1] = 0; blockDiag[offset + 7] = 0; // Col 1
      blockDiag[offset + 3] = 0; blockDiag[offset + 5] = 0; // Row 1
    }
    if (ty_constrained) {
      blockDiag[offset + 8] = 1; // M[2,2] = 1
      blockDiag[offset + 2] = 0; blockDiag[offset + 5] = 0; // Row 2
      blockDiag[offset + 6] = 0; blockDiag[offset + 7] = 0; // Col 2
    }

    // Extract block
    const a00 = blockDiag[offset + 0], a01 = blockDiag[offset + 1], a02 = blockDiag[offset + 2];
    const a10 = blockDiag[offset + 3], a11 = blockDiag[offset + 4], a12 = blockDiag[offset + 5];
    const a20 = blockDiag[offset + 6], a21 = blockDiag[offset + 7], a22 = blockDiag[offset + 8];

    // Compute determinant
    const det = a00 * (a11 * a22 - a12 * a21)
              - a01 * (a10 * a22 - a12 * a20)
              + a02 * (a10 * a21 - a11 * a20);

    // Guard against singular block
    if (Math.abs(det) < 1e-30) {
      // Fall back to identity for singular blocks
      blockDiag[offset + 0] = 1; blockDiag[offset + 1] = 0; blockDiag[offset + 2] = 0;
      blockDiag[offset + 3] = 0; blockDiag[offset + 4] = 1; blockDiag[offset + 5] = 0;
      blockDiag[offset + 6] = 0; blockDiag[offset + 7] = 0; blockDiag[offset + 8] = 1;
      continue;
    }

    const invDet = 1.0 / det;

    // Compute inverse (cofactor matrix transposed, divided by det)
    blockDiag[offset + 0] = invDet * (a11 * a22 - a12 * a21);
    blockDiag[offset + 1] = invDet * (a02 * a21 - a01 * a22);
    blockDiag[offset + 2] = invDet * (a01 * a12 - a02 * a11);
    blockDiag[offset + 3] = invDet * (a12 * a20 - a10 * a22);
    blockDiag[offset + 4] = invDet * (a00 * a22 - a02 * a20);
    blockDiag[offset + 5] = invDet * (a02 * a10 - a00 * a12);
    blockDiag[offset + 6] = invDet * (a10 * a21 - a11 * a20);
    blockDiag[offset + 7] = invDet * (a01 * a20 - a00 * a21);
    blockDiag[offset + 8] = invDet * (a00 * a11 - a01 * a10);
  }
}

/**
 * Apply block diagonal preconditioner: z = M^-1 * r
 * 
 * @param blockDiagInv - Inverted block diagonal (from invertBlockDiagonal)
 * @param r - Input residual vector
 * @param z - Output preconditioned vector
 */
export function applyBlockPreconditioner(
  blockDiagInv: Float32Array,
  r: Float32Array,
  z: Float32Array
): void {
  const nodeCount = blockDiagInv.length / 9;

  for (let node = 0; node < nodeCount; node++) {
    const blockOffset = node * 9;
    const dofOffset = node * 3;

    const r0 = r[dofOffset];
    const r1 = r[dofOffset + 1];
    const r2 = r[dofOffset + 2];

    // z = M^-1 * r (matrix-vector multiply for 3x3 block)
    z[dofOffset]     = blockDiagInv[blockOffset + 0] * r0 
                     + blockDiagInv[blockOffset + 1] * r1 
                     + blockDiagInv[blockOffset + 2] * r2;
    z[dofOffset + 1] = blockDiagInv[blockOffset + 3] * r0 
                     + blockDiagInv[blockOffset + 4] * r1 
                     + blockDiagInv[blockOffset + 5] * r2;
    z[dofOffset + 2] = blockDiagInv[blockOffset + 6] * r0 
                     + blockDiagInv[blockOffset + 7] * r1 
                     + blockDiagInv[blockOffset + 8] * r2;
  }
}

/**
 * Apply boundary conditions to diagonal (for preconditioner).
 *
 * @param diagonal - Diagonal values
 * @param constrainedDOFs - Set of constrained DOF indices
 */
export function applyBCsToDiagonal(
  diagonal: Float32Array,
  constrainedDOFs: Set<number>
): void {
  for (const dof of constrainedDOFs) {
    diagonal[dof] = 1.0; // Unit diagonal for constrained DOF
  }
}

/**
 * Build load vector from point loads.
 *
 * @param mesh - Plate mesh
 * @param loads - Applied loads
 * @returns Load vector
 */
export function buildLoadVector(
  mesh: PlateMesh,
  loads: PlateLoad[]
): Float32Array {
  const nDOF = mesh.nodeCount * DOFS_PER_NODE;
  const F = new Float32Array(nDOF);

  for (const load of loads) {
    // Find nearest node to load position
    const [px, py] = load.position;
    let nearestNode = 0;
    let minDist = Infinity;

    for (let i = 0; i < mesh.nodeCount; i++) {
      const nx = mesh.nodes[i * 2];
      const ny = mesh.nodes[i * 2 + 1];
      const dist = Math.sqrt((nx - px) ** 2 + (ny - py) ** 2);
      if (dist < minDist) {
        minDist = dist;
        nearestNode = i;
      }
    }

    // Apply load to w DOF of nearest node
    // Positive magnitude = downward = positive w
    F[nearestNode * DOFS_PER_NODE] += load.magnitude;
  }

  return F;
}

/**
 * Apply boundary conditions to RHS vector.
 *
 * @param F - Load vector
 * @param constrainedDOFs - Set of constrained DOF indices
 */
export function applyBCsToRHS(
  F: Float32Array,
  constrainedDOFs: Set<number>
): void {
  for (const dof of constrainedDOFs) {
    F[dof] = 0.0; // Zero RHS for constrained DOF
  }
}

/**
 * Apply global stiffness matrix in matrix-free fashion: y = K·x
 *
 * Uses element coloring for conflict-free assembly (GPU-ready).
 * Supports both Q4 (quad) and DKT (triangle) elements.
 *
 * @param mesh - Plate mesh
 * @param material - Material properties
 * @param coloring - Element coloring
 * @param x - Input vector
 * @param y - Output vector (will be overwritten)
 * @param constrainedDOFs - Constrained DOF indices
 */
export function applyGlobalK(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  x: Float32Array,
  y: Float32Array,
  constrainedDOFs: Set<number>
): void {
  // Zero output
  y.fill(0);

  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofsPerElem = nodesPerElem * DOFS_PER_NODE;

  // Temporary arrays for element operations
  const xLocal = new Float32Array(dofsPerElem);
  const yLocal = new Float32Array(dofsPerElem);

  // Process elements by color (can be parallelized)
  for (const elementIndices of coloring.colors) {
    for (let i = 0; i < elementIndices.length; i++) {
      const elemIdx = elementIndices[i];

      // Get element nodes
      const nodeIndices = getElementNodeIndices(mesh, elemIdx);

      // Build DOF index array
      const dofs: number[] = [];
      for (let n = 0; n < nodesPerElem; n++) {
        const baseGlobalDOF = nodeIndices[n] * DOFS_PER_NODE;
        dofs.push(baseGlobalDOF, baseGlobalDOF + 1, baseGlobalDOF + 2);
      }

      // Gather local x values
      for (let j = 0; j < dofsPerElem; j++) {
        xLocal[j] = x[dofs[j]];
      }

      // Get cached element stiffness (computed once, reused every iteration)
      const Ke = getOrComputeKe(mesh, material, elemIdx);

      yLocal.fill(0);
      for (let row = 0; row < dofsPerElem; row++) {
        let sum = 0;
        for (let col = 0; col < dofsPerElem; col++) {
          sum += Ke[row * dofsPerElem + col] * xLocal[col];
        }
        yLocal[row] = sum;
      }

      // Scatter to global (no conflicts within same color!)
      for (let j = 0; j < dofsPerElem; j++) {
        y[dofs[j]] += yLocal[j];
      }
    }
  }

  // Apply BC: for constrained DOFs, y[i] = x[i]
  for (const dof of constrainedDOFs) {
    y[dof] = x[dof];
  }
}

/**
 * Solve plate problem.
 *
 * Automatically handles both quad and triangle meshes.
 *
 * @param geometry - Plate boundary and holes
 * @param material - Material properties
 * @param supports - Boundary conditions
 * @param loads - Applied loads
 * @param options - Solver options
 * @returns Complete solution
 */
export function solvePlate(
  geometry: PlateGeometry,
  material: PlateMaterial,
  supports: PlateSupport[],
  loads: PlateLoad[],
  options: SolvePlateOptions = {}
): PlateResult {
  const startTotal = performance.now();

  // 1. Generate mesh
  const startMesh = performance.now();
  const mesh = generateMesh(geometry, options.meshSize ?? 0.5);
  const meshTimeMs = performance.now() - startMesh;

  // 2. Compute element coloring (use greedy for triangles, structured for quads)
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const coloring =
    nodesPerElem === 3 || mesh.gridNx === 0
      ? computeElementColoringGreedy(mesh)
      : computeElementColoring(mesh);

  // 3. Identify constrained DOFs
  const constrainedDOFs = identifyConstrainedDOFs(mesh, supports);

  // 4. Build block diagonal preconditioner (much more effective than scalar diagonal)
  const startAssembly = performance.now();
  const blockDiag = computeBlockDiagonal(mesh, material);
  invertBlockDiagonal(blockDiag, constrainedDOFs);
  const assemblyTimeMs = performance.now() - startAssembly;

  // 5. Build load vector
  const F = buildLoadVector(mesh, loads);
  applyBCsToRHS(F, constrainedDOFs);

  // 6. Define K·x operation
  const applyK = (x: Float32Array, y: Float32Array) => {
    applyGlobalK(mesh, material, coloring, x, y, constrainedDOFs);
  };

  // 7. Solve with block preconditioner
  const startSolve = performance.now();
  const pcgResult = solvePCG(applyK, F, {
    tolerance: options.tolerance ?? 1e-6, // Slightly looser tolerance for speed
    maxIterations: options.maxIterations ?? 1000,
    blockPreconditioner: blockDiag,
  });
  const solveTimeMs = performance.now() - startSolve;

  // 8. Post-process
  const startPost = performance.now();
  const { Mx, My, Mxy } = computeMoments(mesh, pcgResult.solution, material);
  const postprocessTimeMs = performance.now() - startPost;

  // Extract w from full displacement vector
  const w = extractVerticalDisplacements(pcgResult.solution, mesh.nodeCount);

  // Clear element stiffness cache to free memory
  clearKeCache();

  return {
    mesh,
    displacements: pcgResult.solution,
    w,
    Mx,
    My,
    Mxy,
    solverInfo: {
      iterations: pcgResult.iterations,
      finalResidual: pcgResult.finalResidual,
      converged: pcgResult.converged,
      meshTimeMs,
      assemblyTimeMs,
      solveTimeMs,
      postprocessTimeMs,
      totalTimeMs: performance.now() - startTotal,
    },
  };
}

