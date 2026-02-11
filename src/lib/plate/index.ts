/**
 * Plate Solver — Main exports.
 *
 * A finite element plate bending solver for 2D polygonal plates with:
 *   - Q4 Mindlin (quad) and DKT Kirchhoff (triangle) element formulations
 *   - Matrix-free Preconditioned Conjugate Gradient (PCG) solver
 *   - Block Jacobi preconditioner (3×3 per node)
 *   - GPU acceleration via WebGPU (see ./gpu/ for the fast path)
 *   - Structured and unstructured mesh generation
 *   - Element coloring for parallel GPU assembly
 *   - Post-processing: moments (Mx, My, Mxy) and displacements
 *
 * For GPU-accelerated solving, see @/lib/plate/gpu — it achieves
 * 100k DOF in ~13ms on consumer GPUs via single-submit PCG.
 *
 * See ARCHITECTURE.md for the full technical documentation.
 */

// Types
export type {
  PlateMaterial,
  PlateGeometry,
  PlateSupport,
  SupportType,
  PlateLoad,
  PlateUDL,
  PlateMesh,
  ElementColoring,
  SolverInfo,
  PlateResult,
  SolvePlateOptions,
  PCGOptions,
  PCGResult,
} from './types';

export {
  computeFlexuralRigidity,
  DOFS_PER_NODE,
  NODES_PER_ELEMENT,
  DOFS_PER_ELEMENT,
  NODES_PER_TRIANGLE,
  DOFS_PER_TRIANGLE,
} from './types';

// Meshing
export {
  generateMesh,
  generateRectangularMesh,
  generateTriangularMesh,
  isSimpleRectangle,
  getElementCoords,
  getElementNodes,
  getElementNodeIndices,
} from './mesher';

export {
  generateUnstructuredMesh,
  validateMeshQuality,
} from './mesher-unstructured';

export {
  computeBoundingBox,
  generateCirclePolygon,
  isInsidePolygon,
  ensureWindingOrder,
} from './mesher-utils';

// Element formulations
export {
  computeElementStiffness,
  computeDKTStiffness,
  computeShapeFunctions,
  computeShapeFunctionDerivatives,
  computeJacobian,
  computePhysicalDerivatives,
} from './element';

// Coloring
export {
  computeElementColoring,
  computeElementColoringGreedy,
  verifyColoring,
  getColoringStats,
} from './coloring';

// CPU Solver (reference implementation - DO NOT MODIFY)
export {
  solvePlate,
  identifyConstrainedDOFs,
  computeDiagonal,
  computeBlockDiagonal,
  invertBlockDiagonal,
  applyBlockPreconditioner,
  buildLoadVector,
  applyBCsToRHS,
  applyGlobalK,
} from './solver';

// PCG algorithm
export {
  solvePCG,
  dot,
  norm,
  copy,
  axpy,
} from './pcg';

// Post-processing
export {
  computeMoments,
  extractVerticalDisplacements,
  findMaxDisplacement,
  findMaxMoments,
} from './postprocess';

// GPU Solver - OPTIMIZATION TARGET
export {
  solveGPU,
  isWebGPUAvailable,
} from './gpu';

