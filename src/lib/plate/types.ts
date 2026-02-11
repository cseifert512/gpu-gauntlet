/**
 * Plate solver type definitions.
 *
 * Design for GPU compatibility:
 * - Use Float32Array for numerical data
 * - Use flat arrays instead of nested structures
 * - Pre-allocate buffers where possible
 */

// =============================================================================
// Material Properties
// =============================================================================

export interface PlateMaterial {
  /** Young's modulus (Pa) */
  E: number;
  /** Poisson's ratio */
  nu: number;
  /** Thickness (m) */
  t: number;
}

/**
 * Compute flexural rigidity D = E*t³ / (12*(1-ν²))
 */
export function computeFlexuralRigidity(mat: PlateMaterial): number {
  return (mat.E * mat.t ** 3) / (12 * (1 - mat.nu ** 2));
}

// =============================================================================
// Geometry
// =============================================================================

export interface PlateGeometry {
  /** Outer boundary vertices (CCW), flat: [x0,y0, x1,y1, ...] */
  boundary: Float32Array;
  /** Hole boundaries (CW), each as flat array */
  holes: Float32Array[];
}

// =============================================================================
// Boundary Conditions
// =============================================================================

export type SupportType = 'pinned' | 'fixed' | 'roller';

export interface PlateSupport {
  type: SupportType;
  /** Specific node index, or 'all_edges' for boundary nodes */
  location: number | 'all_edges';
}

/**
 * Line support — constrains all mesh nodes along a polyline.
 *
 * Used for walls, beams, or any linear support condition. The solver finds
 * all nodes within `tolerance` of the polyline segments and applies the
 * specified constraint type (pinned / fixed / roller) to each.
 *
 * Points are [x, y] pairs describing the polyline vertices.
 */
export interface PlateLineSupport {
  type: SupportType;
  /** Polyline vertices: [[x0,y0], [x1,y1], ...] (at least 2 points) */
  points: [number, number][];
  /** Distance tolerance for snapping nodes to the line (default: meshSize/4) */
  tolerance?: number;
}

// =============================================================================
// Loading
// =============================================================================

export interface PlateLoad {
  /** Position [x, y] in meters */
  position: [number, number];
  /** Magnitude in N (positive = downward) */
  magnitude: number;
}

export interface PlateUDL {
  /** Uniform pressure in Pa (N/m²), positive = downward */
  pressure: number;
}

// =============================================================================
// Mesh Data Structures
// =============================================================================

export interface PlateMesh {
  /** Node coordinates, flat: [x0,y0, x1,y1, ...] */
  nodes: Float32Array;
  /** Element connectivity, flat: [n0,n1,n2,n3, ...] for quads or [n0,n1,n2, ...] for triangles */
  elements: Uint32Array;
  /** Boundary node indices */
  boundaryNodes: Uint32Array;
  /** Number of nodes */
  nodeCount: number;
  /** Number of elements */
  elementCount: number;
  /** Nodes per element: 4 for quads, 3 for triangles */
  nodesPerElement?: number;
  /** Grid dimensions (for structured mesh, 0 for unstructured) */
  gridNx: number;
  gridNy: number;
}

// =============================================================================
// Element Coloring (for GPU parallelism)
// =============================================================================

export interface ElementColoring {
  /** For each color, array of element indices */
  colors: Uint32Array[];
  /** Color assigned to each element */
  elementColors: Uint8Array;
  /** Number of colors (should be ≤4 for structured quad grid) */
  colorCount: number;
}

// =============================================================================
// Solver Results
// =============================================================================

export interface SolverInfo {
  iterations: number;
  finalResidual: number;
  converged: boolean;
  meshTimeMs: number;
  assemblyTimeMs: number;
  solveTimeMs: number;
  postprocessTimeMs: number;
  totalTimeMs: number;
}

export interface PlateResult {
  /** Mesh used for solve */
  mesh: PlateMesh;
  /** Displacement field [w0,θx0,θy0, w1,θx1,θy1, ...] */
  displacements: Float32Array;
  /** Vertical displacement only (n_nodes) */
  w: Float32Array;
  /** Bending moment Mx (n_nodes) */
  Mx: Float32Array;
  /** Bending moment My (n_nodes) */
  My: Float32Array;
  /** Twisting moment Mxy (n_nodes) */
  Mxy: Float32Array;
  /** Solver statistics */
  solverInfo: SolverInfo;
}

// =============================================================================
// Solver Options
// =============================================================================

export interface SolvePlateOptions {
  /** Target element size (m), default 0.5 */
  meshSize?: number;
  /** PCG tolerance, default 1e-8 */
  tolerance?: number;
  /** PCG max iterations, default 1000 */
  maxIterations?: number;
}

// =============================================================================
// PCG Types
// =============================================================================

export interface PCGOptions {
  /** Convergence tolerance (default 1e-8) */
  tolerance: number;
  /** Max iterations (default 1000) */
  maxIterations: number;
  /** Diagonal preconditioner (simple Jacobi) */
  preconditioner?: Float32Array;
  /** Block diagonal preconditioner (3x3 inverted blocks, 9 floats per node) */
  blockPreconditioner?: Float32Array;
}

export interface PCGResult {
  solution: Float32Array;
  iterations: number;
  finalResidual: number;
  converged: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/** DOFs per node: w (vertical), θx (rotation about x), θy (rotation about y) */
export const DOFS_PER_NODE = 3;

/** Nodes per Q4 element (for backwards compatibility) */
export const NODES_PER_ELEMENT = 4;

/** DOFs per Q4 element (4 nodes × 3 DOFs) */
export const DOFS_PER_ELEMENT = 12;

/** Nodes per DKT triangle element */
export const NODES_PER_TRIANGLE = 3;

/** DOFs per DKT element (3 nodes × 3 DOFs) */
export const DOFS_PER_TRIANGLE = 9;

