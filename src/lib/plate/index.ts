/**
 * Plate Solver - Main exports
 * 
 * This is a finite element plate bending solver that supports:
 * - Q4 (quad) and DKT (triangle) elements
 * - Matrix-free PCG solver
 * - GPU acceleration via WebGPU
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

