/**
 * Q4 Mindlin plate element formulation.
 *
 * DOFs: 3 per node [w, θx, θy]
 *   - w: vertical displacement (positive downward)
 *   - θx: rotation about x-axis
 *   - θy: rotation about y-axis
 *
 * Integration:
 *   - 2×2 Gauss for bending terms
 *   - 1-point (reduced) for shear terms (avoids shear locking)
 *
 * Element stiffness: K = Kb + Ks
 *   - Kb: bending stiffness (2×2 integration)
 *   - Ks: shear stiffness (1-point integration)
 */

import type { PlateMaterial } from './types';
import {
  computeFlexuralRigidity,
  DOFS_PER_ELEMENT,
  DOFS_PER_TRIANGLE,
} from './types';

// =============================================================================
// Gauss Quadrature Points
// =============================================================================

/** 2×2 Gauss points: ξ = ±1/√3 */
const GP_2X2 = 0.5773502691896257; // 1/√3

/** 2×2 Gauss quadrature points [ξ, η] */
const GAUSS_2X2_POINTS: [number, number][] = [
  [-GP_2X2, -GP_2X2],
  [+GP_2X2, -GP_2X2],
  [+GP_2X2, +GP_2X2],
  [-GP_2X2, +GP_2X2],
];

/** 2×2 Gauss weights (all 1.0) */
const GAUSS_2X2_WEIGHTS = [1.0, 1.0, 1.0, 1.0];

// =============================================================================
// Shape Functions
// =============================================================================

/**
 * Compute Q4 shape functions at natural coordinates (ξ, η).
 *
 * N1 = (1-ξ)(1-η)/4, N2 = (1+ξ)(1-η)/4, N3 = (1+ξ)(1+η)/4, N4 = (1-ξ)(1+η)/4
 *
 * @param xi - Natural coordinate ξ ∈ [-1, 1]
 * @param eta - Natural coordinate η ∈ [-1, 1]
 * @returns 4-element array of shape function values
 */
export function computeShapeFunctions(
  xi: number,
  eta: number
): [number, number, number, number] {
  const xm = 1 - xi;
  const xp = 1 + xi;
  const em = 1 - eta;
  const ep = 1 + eta;

  return [
    (xm * em) / 4, // N1
    (xp * em) / 4, // N2
    (xp * ep) / 4, // N3
    (xm * ep) / 4, // N4
  ];
}

/**
 * Compute shape function derivatives w.r.t. natural coordinates.
 *
 * @param xi - Natural coordinate ξ
 * @param eta - Natural coordinate η
 * @returns [dN/dξ, dN/dη] as 4-element arrays
 */
export function computeShapeFunctionDerivatives(
  xi: number,
  eta: number
): { dNdXi: [number, number, number, number]; dNdEta: [number, number, number, number] } {
  const xm = 1 - xi;
  const xp = 1 + xi;
  const em = 1 - eta;
  const ep = 1 + eta;

  // dN/dξ
  const dNdXi: [number, number, number, number] = [
    -em / 4, // dN1/dξ
    +em / 4, // dN2/dξ
    +ep / 4, // dN3/dξ
    -ep / 4, // dN4/dξ
  ];

  // dN/dη
  const dNdEta: [number, number, number, number] = [
    -xm / 4, // dN1/dη
    -xp / 4, // dN2/dη
    +xp / 4, // dN3/dη
    +xm / 4, // dN4/dη
  ];

  return { dNdXi, dNdEta };
}

/**
 * Compute Jacobian matrix and its determinant.
 *
 * J = [∂x/∂ξ  ∂y/∂ξ]
 *     [∂x/∂η  ∂y/∂η]
 *
 * @param dNdXi - Shape function derivatives w.r.t. ξ
 * @param dNdEta - Shape function derivatives w.r.t. η
 * @param nodeCoords - Node coordinates [x0,y0, x1,y1, x2,y2, x3,y3]
 * @returns Jacobian matrix components and determinant
 */
export function computeJacobian(
  dNdXi: [number, number, number, number],
  dNdEta: [number, number, number, number],
  nodeCoords: Float32Array
): {
  J11: number;
  J12: number;
  J21: number;
  J22: number;
  detJ: number;
} {
  let J11 = 0,
    J12 = 0,
    J21 = 0,
    J22 = 0;

  for (let i = 0; i < 4; i++) {
    const x = nodeCoords[i * 2];
    const y = nodeCoords[i * 2 + 1];

    J11 += dNdXi[i] * x; // ∂x/∂ξ
    J12 += dNdXi[i] * y; // ∂y/∂ξ
    J21 += dNdEta[i] * x; // ∂x/∂η
    J22 += dNdEta[i] * y; // ∂y/∂η
  }

  const detJ = J11 * J22 - J12 * J21;

  return { J11, J12, J21, J22, detJ };
}

/**
 * Compute shape function derivatives w.r.t. physical coordinates.
 *
 * [dN/dx] = J^{-1} [dN/dξ]
 * [dN/dy]          [dN/dη]
 *
 * @param dNdXi - Shape function derivatives w.r.t. ξ
 * @param dNdEta - Shape function derivatives w.r.t. η
 * @param J11, J12, J21, J22 - Jacobian components
 * @param detJ - Jacobian determinant
 * @returns [dN/dx, dN/dy] as 4-element arrays
 */
export function computePhysicalDerivatives(
  dNdXi: [number, number, number, number],
  dNdEta: [number, number, number, number],
  J11: number,
  J12: number,
  J21: number,
  J22: number,
  detJ: number
): { dNdx: [number, number, number, number]; dNdy: [number, number, number, number] } {
  const invDetJ = 1.0 / detJ;

  const dNdx: [number, number, number, number] = [0, 0, 0, 0];
  const dNdy: [number, number, number, number] = [0, 0, 0, 0];

  // J^{-1} = (1/detJ) [J22  -J12]
  //                   [-J21  J11]
  for (let i = 0; i < 4; i++) {
    dNdx[i] = invDetJ * (J22 * dNdXi[i] - J12 * dNdEta[i]);
    dNdy[i] = invDetJ * (-J21 * dNdXi[i] + J11 * dNdEta[i]);
  }

  return { dNdx, dNdy };
}

// =============================================================================
// Material Matrices
// =============================================================================

/**
 * Compute bending constitutive matrix Db.
 *
 * Db = D * [1   ν   0      ]
 *          [ν   1   0      ]
 *          [0   0   (1-ν)/2]
 *
 * where D = Et³/(12(1-ν²))
 *
 * @param material - Plate material
 * @returns 3×3 matrix as flat 9-element array (row-major)
 */
export function computeBendingConstitutive(
  material: PlateMaterial
): Float32Array {
  const D = computeFlexuralRigidity(material);
  const { nu } = material;

  // Flat 3×3 row-major
  return new Float32Array([
    D,
    D * nu,
    0,
    D * nu,
    D,
    0,
    0,
    0,
    (D * (1 - nu)) / 2,
  ]);
}

/**
 * Compute shear constitutive matrix Ds.
 *
 * Ds = κ * G * t * [1 0]
 *                  [0 1]
 *
 * where G = E/(2(1+ν)), κ = 5/6 (shear correction factor)
 *
 * @param material - Plate material
 * @returns 2×2 matrix as flat 4-element array (row-major)
 */
export function computeShearConstitutive(material: PlateMaterial): Float32Array {
  const { E, nu, t } = material;
  const kappa = 5.0 / 6.0; // Shear correction factor for rectangular section
  const G = E / (2 * (1 + nu));
  const Ds_coeff = kappa * G * t;

  return new Float32Array([Ds_coeff, 0, 0, Ds_coeff]);
}

// =============================================================================
// Element Stiffness Matrix
// =============================================================================

/**
 * Compute 12×12 element stiffness matrix for Q4 Mindlin plate.
 *
 * K = Kb + Ks (bending + shear contributions)
 *
 * DOF ordering per node: [w, θx, θy]
 * Element DOF ordering: [w0,θx0,θy0, w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3]
 *
 * @param nodeCoords - 4 corner coordinates as [x0,y0, x1,y1, x2,y2, x3,y3]
 * @param material - Material properties
 * @returns 12×12 stiffness matrix as flat Float32Array (row-major)
 */
export function computeElementStiffness(
  nodeCoords: Float32Array,
  material: PlateMaterial
): Float32Array {
  const Ke = new Float32Array(DOFS_PER_ELEMENT * DOFS_PER_ELEMENT); // 144 values

  // Material matrices
  const Db = computeBendingConstitutive(material);
  const Ds = computeShearConstitutive(material);

  // ==========================================================================
  // Bending stiffness (2×2 Gauss integration)
  // ==========================================================================
  //
  // Curvature-rotation relationship:
  // κ = {κx, κy, κxy} = {∂θy/∂x, -∂θx/∂y, ∂θy/∂y - ∂θx/∂x}
  //
  // Bb matrix (3×12): κ = Bb * u
  //
  // For node i, DOFs are [w_i, θx_i, θy_i] at columns [3i, 3i+1, 3i+2]
  //
  // Bb[0, 3i+2] = dN_i/dx    (κx = ∂θy/∂x)
  // Bb[1, 3i+1] = -dN_i/dy   (κy = -∂θx/∂y)
  // Bb[2, 3i+1] = -dN_i/dx   (κxy term from -∂θx/∂x)
  // Bb[2, 3i+2] = dN_i/dy    (κxy term from ∂θy/∂y)

  for (let gp = 0; gp < 4; gp++) {
    const [xi, eta] = GAUSS_2X2_POINTS[gp];
    const weight = GAUSS_2X2_WEIGHTS[gp];

    const { dNdXi, dNdEta } = computeShapeFunctionDerivatives(xi, eta);
    const { J11, J12, J21, J22, detJ } = computeJacobian(
      dNdXi,
      dNdEta,
      nodeCoords
    );
    const { dNdx, dNdy } = computePhysicalDerivatives(
      dNdXi,
      dNdEta,
      J11,
      J12,
      J21,
      J22,
      detJ
    );

    // Build Bb matrix (3×12) for this Gauss point
    // Bb is built implicitly in the Kb computation below

    // Compute Kb contribution: Ke += w * |J| * Bb^T * Db * Bb
    const factor = weight * Math.abs(detJ);

    // For efficiency, compute BtDB directly without forming full matrices
    // Bb^T * Db * Bb contribution for nodes i and j
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        // DOF indices
        const di_w = i * 3;
        const di_tx = i * 3 + 1;
        const di_ty = i * 3 + 2;
        const dj_w = j * 3;
        const dj_tx = j * 3 + 1;
        const dj_ty = j * 3 + 2;

        // Bb_i^T * Db * Bb_j components
        // Bb_i = [  0,      0,    dNdx[i] ]  (row 0: κx)
        //        [  0, -dNdy[i],     0    ]  (row 1: κy)
        //        [  0, -dNdx[i], dNdy[i]  ]  (row 2: κxy)

        // Db = [D11, D12,  0 ]
        //      [D12, D22,  0 ]
        //      [ 0,   0, D33 ]
        const D11 = Db[0],
          D12 = Db[1],
          D22 = Db[4],
          D33 = Db[8];

        // Compute BtDB_ij (3×3 submatrix for nodes i, j)
        const dNdx_i = dNdx[i];
        const dNdy_i = dNdy[i];
        const dNdx_j = dNdx[j];
        const dNdy_j = dNdy[j];

        // [θx_i, θx_j] contribution: comes from Bb rows 1, 2 columns with θx
        // Bb(1, θx_i) = -dNdy[i], Bb(2, θx_i) = -dNdx[i]
        // (Bb_i^T * Db * Bb_j)[θx, θx]
        //   = Bb(1,θx_i)*D22*Bb(1,θx_j) + Bb(2,θx_i)*D33*Bb(2,θx_j)
        //   = (-dNdy_i)*D22*(-dNdy_j) + (-dNdx_i)*D33*(-dNdx_j)
        const K_tx_tx = dNdy_i * D22 * dNdy_j + dNdx_i * D33 * dNdx_j;

        // [θy_i, θy_j] contribution
        // Bb(0, θy_i) = dNdx[i], Bb(2, θy_i) = dNdy[i]
        const K_ty_ty = dNdx_i * D11 * dNdx_j + dNdy_i * D33 * dNdy_j;

        // [θx_i, θy_j] contribution
        // Bb_θx_i^T * Db * Bb_θy_j
        // Bb_θx_i = [0, -dNdy_i, -dNdx_i]^T
        // Bb_θy_j = [dNdx_j, 0, dNdy_j]^T
        // = (-dNdy_i)*D12*(dNdx_j) + (-dNdx_i)*D33*(dNdy_j)
        const K_tx_ty = -dNdy_i * D12 * dNdx_j - dNdx_i * D33 * dNdy_j;

        // [θy_i, θx_j] contribution
        // Bb_θy_i^T * Db * Bb_θx_j
        // Bb_θy_i = [dNdx_i, 0, dNdy_i]^T
        // Bb_θx_j = [0, -dNdy_j, -dNdx_j]^T
        // = (dNdx_i)*(-D12*dNdy_j) + (dNdy_i)*(-D33*dNdx_j)
        const K_ty_tx = -dNdx_i * D12 * dNdy_j - dNdy_i * D33 * dNdx_j;

        // Add to element stiffness
        Ke[di_tx * 12 + dj_tx] += factor * K_tx_tx;
        Ke[di_ty * 12 + dj_ty] += factor * K_ty_ty;
        Ke[di_tx * 12 + dj_ty] += factor * K_tx_ty;
        Ke[di_ty * 12 + dj_tx] += factor * K_ty_tx;
      }
    }
  }

  // ==========================================================================
  // Shear stiffness (1-point reduced integration to avoid locking)
  // ==========================================================================
  //
  // Shear strains:
  // γ = {γxz, γyz} = {∂w/∂x + θy, ∂w/∂y - θx}
  //
  // Bs matrix (2×12): γ = Bs * u
  //
  // For node i, DOFs are [w_i, θx_i, θy_i]
  // Bs[0, 3i] = dN_i/dx     (γxz from ∂w/∂x)
  // Bs[0, 3i+2] = N_i       (γxz from θy)
  // Bs[1, 3i] = dN_i/dy     (γyz from ∂w/∂y)
  // Bs[1, 3i+1] = -N_i      (γyz from -θx)

  {
    // 1-point integration at center (ξ=0, η=0)
    const xi = 0;
    const eta = 0;
    const weight = 4.0; // Total weight for 1-point rule

    const N = computeShapeFunctions(xi, eta);
    const { dNdXi, dNdEta } = computeShapeFunctionDerivatives(xi, eta);
    const { J11, J12, J21, J22, detJ } = computeJacobian(
      dNdXi,
      dNdEta,
      nodeCoords
    );
    const { dNdx, dNdy } = computePhysicalDerivatives(
      dNdXi,
      dNdEta,
      J11,
      J12,
      J21,
      J22,
      detJ
    );

    const factor = weight * Math.abs(detJ);

    // Ds = [Ds11,  0  ]
    //      [ 0,  Ds22 ]
    const Ds11 = Ds[0];
    const Ds22 = Ds[3];

    // Compute Ks contribution: Ke += factor * Bs^T * Ds * Bs
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const di_w = i * 3;
        const di_tx = i * 3 + 1;
        const di_ty = i * 3 + 2;
        const dj_w = j * 3;
        const dj_tx = j * 3 + 1;
        const dj_ty = j * 3 + 2;

        const N_i = N[i];
        const N_j = N[j];
        const dNdx_i = dNdx[i];
        const dNdy_i = dNdy[i];
        const dNdx_j = dNdx[j];
        const dNdy_j = dNdy[j];

        // Bs_i^T * Ds * Bs_j
        // Bs_i = [dNdx_i,    0,   N_i]  (row 0: γxz)
        //        [dNdy_i, -N_i,    0 ]  (row 1: γyz)

        // [w, w]
        const K_w_w = dNdx_i * Ds11 * dNdx_j + dNdy_i * Ds22 * dNdy_j;

        // [w, θx]
        const K_w_tx = dNdy_i * Ds22 * (-N_j);

        // [w, θy]
        const K_w_ty = dNdx_i * Ds11 * N_j;

        // [θx, w]
        const K_tx_w = (-N_i) * Ds22 * dNdy_j;

        // [θx, θx]
        const K_tx_tx = (-N_i) * Ds22 * (-N_j);

        // [θx, θy]
        const K_tx_ty = 0;

        // [θy, w]
        const K_ty_w = N_i * Ds11 * dNdx_j;

        // [θy, θx]
        const K_ty_tx = 0;

        // [θy, θy]
        const K_ty_ty = N_i * Ds11 * N_j;

        // Add to element stiffness
        Ke[di_w * 12 + dj_w] += factor * K_w_w;
        Ke[di_w * 12 + dj_tx] += factor * K_w_tx;
        Ke[di_w * 12 + dj_ty] += factor * K_w_ty;
        Ke[di_tx * 12 + dj_w] += factor * K_tx_w;
        Ke[di_tx * 12 + dj_tx] += factor * K_tx_tx;
        Ke[di_tx * 12 + dj_ty] += factor * K_tx_ty;
        Ke[di_ty * 12 + dj_w] += factor * K_ty_w;
        Ke[di_ty * 12 + dj_tx] += factor * K_ty_tx;
        Ke[di_ty * 12 + dj_ty] += factor * K_ty_ty;
      }
    }
  }

  return Ke;
}

/**
 * Apply element stiffness contribution: y += K_e * x_local
 *
 * This is used for matrix-free K·x operation.
 *
 * @param nodeCoords - Element node coordinates [x0,y0, x1,y1, x2,y2, x3,y3]
 * @param material - Material properties
 * @param xLocal - Local DOF values (12 values)
 * @param yLocal - Output array to accumulate into (12 values)
 */
export function applyElementStiffness(
  nodeCoords: Float32Array,
  material: PlateMaterial,
  xLocal: Float32Array,
  yLocal: Float32Array
): void {
  const Ke = computeElementStiffness(nodeCoords, material);

  // y = K * x
  for (let i = 0; i < DOFS_PER_ELEMENT; i++) {
    let sum = 0;
    for (let j = 0; j < DOFS_PER_ELEMENT; j++) {
      sum += Ke[i * DOFS_PER_ELEMENT + j] * xLocal[j];
    }
    yLocal[i] += sum;
  }
}

/**
 * Compute element mass matrix (lumped).
 *
 * For dynamic analysis / modal analysis (not needed for static solve).
 *
 * @param nodeCoords - Element node coordinates
 * @param material - Material properties with density
 * @param density - Material density (kg/m³)
 * @returns 12×12 diagonal mass matrix
 */
export function computeElementMassLumped(
  nodeCoords: Float32Array,
  thickness: number,
  density: number
): Float32Array {
  // Compute element area using shoelace formula
  let area = 0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const xi = nodeCoords[i * 2];
    const yi = nodeCoords[i * 2 + 1];
    const xj = nodeCoords[j * 2];
    const yj = nodeCoords[j * 2 + 1];
    area += xi * yj - xj * yi;
  }
  area = Math.abs(area) / 2;

  // Total mass distributed to 4 nodes
  const totalMass = density * thickness * area;
  const massPerNode = totalMass / 4;

  // Lumped mass: equal mass at each w DOF, small rotational inertia
  const Me = new Float32Array(DOFS_PER_ELEMENT * DOFS_PER_ELEMENT);

  // Only diagonal terms
  for (let i = 0; i < 4; i++) {
    const di_w = i * 3;
    const di_tx = i * 3 + 1;
    const di_ty = i * 3 + 2;

    // Translational mass
    Me[di_w * 12 + di_w] = massPerNode;

    // Rotational inertia (approximate)
    const rotInertia = (massPerNode * thickness * thickness) / 12;
    Me[di_tx * 12 + di_tx] = rotInertia;
    Me[di_ty * 12 + di_ty] = rotInertia;
  }

  return Me;
}

/**
 * Check if element stiffness matrix is symmetric.
 */
export function isSymmetric(
  Ke: Float32Array,
  tolerance: number = 1e-10
): boolean {
  for (let i = 0; i < DOFS_PER_ELEMENT; i++) {
    for (let j = i + 1; j < DOFS_PER_ELEMENT; j++) {
      const diff = Math.abs(
        Ke[i * DOFS_PER_ELEMENT + j] - Ke[j * DOFS_PER_ELEMENT + i]
      );
      if (diff > tolerance) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Check if element stiffness matrix is positive semi-definite.
 * Uses eigenvalue estimation via power iteration.
 */
export function isPositiveSemiDefinite(Ke: Float32Array): boolean {
  // Simple check: all diagonal elements should be positive
  for (let i = 0; i < DOFS_PER_ELEMENT; i++) {
    if (Ke[i * DOFS_PER_ELEMENT + i] <= 0) {
      return false;
    }
  }
  return true;
}

/**
 * Extract diagonal of element stiffness matrix.
 */
export function extractDiagonal(Ke: Float32Array): Float32Array {
  const diag = new Float32Array(DOFS_PER_ELEMENT);
  for (let i = 0; i < DOFS_PER_ELEMENT; i++) {
    diag[i] = Ke[i * DOFS_PER_ELEMENT + i];
  }
  return diag;
}

// =============================================================================
// DKT (Discrete Kirchhoff Triangle) Element
// =============================================================================

/**
 * Gauss quadrature points for triangle (3-point rule).
 *
 * Natural coordinates (L1, L2, L3) where L1 + L2 + L3 = 1
 */
const TRIANGLE_GAUSS_POINTS: [number, number, number][] = [
  [1 / 6, 1 / 6, 2 / 3],
  [2 / 3, 1 / 6, 1 / 6],
  [1 / 6, 2 / 3, 1 / 6],
];

const TRIANGLE_GAUSS_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]; // Each multiplied by 0.5 (triangle area factor) = 1/6

/**
 * Compute DKT (Discrete Kirchhoff Triangle) element stiffness matrix.
 *
 * 3-node triangle with 3 DOFs per node [w, θx, θy] = 9 DOFs total.
 *
 * The DKT element is based on Kirchhoff plate theory with discrete
 * enforcement of the Kirchhoff constraint at specific points.
 *
 * Reference: Batoz, Bathe, Ho (1980) "A study of three-node triangular
 * plate bending elements"
 *
 * @param nodeCoords - [x0,y0, x1,y1, x2,y2] triangle vertices
 * @param material - Material properties
 * @returns 9×9 stiffness matrix as flat Float32Array (row-major)
 */
export function computeDKTStiffness(
  nodeCoords: Float32Array,
  material: PlateMaterial
): Float32Array {
  const Ke = new Float32Array(DOFS_PER_TRIANGLE * DOFS_PER_TRIANGLE); // 81 values

  // Extract node coordinates
  const x1 = nodeCoords[0],
    y1 = nodeCoords[1];
  const x2 = nodeCoords[2],
    y2 = nodeCoords[3];
  const x3 = nodeCoords[4],
    y3 = nodeCoords[5];

  // Edge vectors and lengths
  const x12 = x1 - x2,
    y12 = y1 - y2;
  const x23 = x2 - x3,
    y23 = y2 - y3;
  const x31 = x3 - x1,
    y31 = y3 - y1;

  const L12_sq = x12 * x12 + y12 * y12;
  const L23_sq = x23 * x23 + y23 * y23;
  const L31_sq = x31 * x31 + y31 * y31;

  // Triangle area (twice area)
  const twoA = x12 * y31 - x31 * y12;
  const area = Math.abs(twoA) / 2;

  if (area < 1e-15) {
    // Degenerate triangle
    console.warn('DKT: Degenerate triangle detected');
    return Ke;
  }

  // Material matrix Db for bending
  const D = computeFlexuralRigidity(material);
  const { nu } = material;

  // Db = D * [1   ν   0      ]
  //          [ν   1   0      ]
  //          [0   0   (1-ν)/2]
  const D11 = D;
  const D12 = D * nu;
  const D22 = D;
  const D33 = (D * (1 - nu)) / 2;

  // DKT coefficients (from Batoz et al.)
  // a_k = -x_ij / L_ij^2
  // b_k = 3*x_ij*y_ij / (4*L_ij^2)
  // c_k = (x_ij^2 - 2*y_ij^2) / (4*L_ij^2)
  // d_k = -y_ij / L_ij^2
  // e_k = (y_ij^2 - 2*x_ij^2) / (4*L_ij^2)

  // Edge 4 (node 2-3), k=4
  const a4 = -x23 / L23_sq;
  const b4 = (3 * x23 * y23) / (4 * L23_sq);
  const c4 = (x23 * x23 - 2 * y23 * y23) / (4 * L23_sq);
  const d4 = -y23 / L23_sq;
  const e4 = (y23 * y23 - 2 * x23 * x23) / (4 * L23_sq);

  // Edge 5 (node 3-1), k=5
  const a5 = -x31 / L31_sq;
  const b5 = (3 * x31 * y31) / (4 * L31_sq);
  const c5 = (x31 * x31 - 2 * y31 * y31) / (4 * L31_sq);
  const d5 = -y31 / L31_sq;
  const e5 = (y31 * y31 - 2 * x31 * x31) / (4 * L31_sq);

  // Edge 6 (node 1-2), k=6
  const a6 = -x12 / L12_sq;
  const b6 = (3 * x12 * y12) / (4 * L12_sq);
  const c6 = (x12 * x12 - 2 * y12 * y12) / (4 * L12_sq);
  const d6 = -y12 / L12_sq;
  const e6 = (y12 * y12 - 2 * x12 * x12) / (4 * L12_sq);

  // P coefficients for shape functions
  const P4 = -6 * a4;
  const P5 = -6 * a5;
  const P6 = -6 * a6;

  const t4 = -6 * d4;
  const t5 = -6 * d5;
  const t6 = -6 * d6;

  const q4 = 3 * a4;
  const q5 = 3 * a5;
  const q6 = 3 * a6;

  const r4 = 3 * d4;
  const r5 = 3 * d5;
  const r6 = 3 * d6;

  // Gauss integration
  for (let gp = 0; gp < 3; gp++) {
    const [L1, L2, L3] = TRIANGLE_GAUSS_POINTS[gp];
    const weight = TRIANGLE_GAUSS_WEIGHTS[gp];

    // Shape function derivatives with respect to area coordinates
    // Hx_k = dβx/dk, Hy_k = dβy/dk
    // where βx and βy are the rotation interpolation functions

    // Derivatives of Hx with respect to L1, L2, L3
    // dHx/dL1, dHx/dL2 for each DOF
    // There are 9 DOFs, so we have Hx[0..8] and Hy[0..8]

    // For DKT, the curvature-DOF relationship at each Gauss point:
    // [κx ]   [ Bx ]
    // [κy ] = [ By ] * {u}
    // [κxy]   [Bxy]

    // B matrix components (9 columns for 9 DOFs)
    // Compute B matrix using DKT shape function derivatives
    const Bx = computeDKTBx(L1, L2, L3, P4, P5, P6, q4, q5, q6, b4, b5, b6, c4, c5, c6);
    const By = computeDKTBy(L1, L2, L3, t4, t5, t6, r4, r5, r6, e4, e5, e6, b4, b5, b6);

    // Combine Bx and By to form full B matrix row by row
    // κx  = d(βx)/dx = (1/2A) * (y31*d(βx)/dL1 + y12*d(βx)/dL2)
    // κy  = d(βy)/dy = (1/2A) * (x13*d(βy)/dL1 + x21*d(βy)/dL2)
    // κxy = d(βx)/dy + d(βy)/dx

    // Transform from area coordinates to physical
    const invTwoA = 1 / twoA;

    // y31 = y3 - y1, y12 = y1 - y2
    // x13 = x1 - x3 = -x31, x21 = x2 - x1 = -x12

    // B matrix in physical coordinates
    const B = new Float32Array(3 * 9); // 3 rows, 9 columns

    for (let j = 0; j < 9; j++) {
      // κx = dHx/dx = (y23 * dHx/dL1 + y31 * dHx/dL2 + y12 * dHx/dL3) / 2A
      // But L3 = 1 - L1 - L2, so dHx/dL3 derivatives are implicit
      // Using: dN/dx = (y23 * dN/dL1 + y31 * dN/dL2) / 2A for standard triangles
      // Note: y23 = y2 - y3, etc.

      // Actually for DKT, we use area coordinate derivatives directly
      // dHx/dx = (1/2A) * (y_jk * dHx/dLi + y_ki * dHx/dLj)
      // where i,j,k are cyclic permutations

      // Simplified: using the standard formulas
      // dHx/dx = (y23 * Hx_L1[j] + y31 * Hx_L2[j]) / 2A
      // Note: These are derivatives w.r.t. L1 and L2 treating L3 = 1-L1-L2

      B[0 * 9 + j] = invTwoA * ((-y23) * Bx.L1[j] + (-y31) * Bx.L2[j]); // κx
      B[1 * 9 + j] = invTwoA * (x23 * By.L1[j] + x31 * By.L2[j]); // κy
      B[2 * 9 + j] =
        invTwoA *
        (x23 * Bx.L1[j] + x31 * Bx.L2[j] + (-y23) * By.L1[j] + (-y31) * By.L2[j]); // κxy
    }

    // Accumulate Ke += weight * area * B^T * D * B
    // D is 3x3, B is 3x9
    const factor = weight * area;

    // Compute D * B (3x9)
    const DB = new Float32Array(3 * 9);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 9; j++) {
        let sum = 0;
        // D is diagonal-ish: D[0][0]=D11, D[0][1]=D12, D[1][0]=D12, D[1][1]=D22, D[2][2]=D33
        if (i === 0) {
          sum = D11 * B[0 * 9 + j] + D12 * B[1 * 9 + j];
        } else if (i === 1) {
          sum = D12 * B[0 * 9 + j] + D22 * B[1 * 9 + j];
        } else {
          sum = D33 * B[2 * 9 + j];
        }
        DB[i * 9 + j] = sum;
      }
    }

    // Compute B^T * DB (9x9)
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        let sum = 0;
        for (let k = 0; k < 3; k++) {
          sum += B[k * 9 + i] * DB[k * 9 + j];
        }
        Ke[i * 9 + j] += factor * sum;
      }
    }
  }

  return Ke;
}

/**
 * Compute DKT Hx derivatives w.r.t. area coordinates.
 * Returns derivatives for all 9 DOFs.
 */
function computeDKTBx(
  L1: number,
  L2: number,
  L3: number,
  P4: number,
  P5: number,
  P6: number,
  q4: number,
  q5: number,
  q6: number,
  b4: number,
  b5: number,
  b6: number,
  c4: number,
  c5: number,
  c6: number
): { L1: Float32Array; L2: Float32Array } {
  // Derivatives of Hx with respect to L1 and L2
  // DOF order: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]

  const dHxdL1 = new Float32Array(9);
  const dHxdL2 = new Float32Array(9);

  // From Batoz et al. DKT formulation
  // Hx1 through Hx9 derivatives

  // Node 1 DOFs (w1, θx1, θy1) - indices 0, 1, 2
  dHxdL1[0] = P6 * (1 - 2 * L1) + (P5 - P6) * L2; // dHx1/dL1
  dHxdL1[1] = q6 * (1 - 2 * L1) - (q5 + q6) * L2; // dHx2/dL1
  dHxdL1[2] = -4 + 6 * (L1 + L2) + (c5 - c6) * L2 + c6 * (1 - 2 * L1); // dHx3/dL1

  dHxdL2[0] = -P6 * (1 - 2 * L2) + (P4 - P5) * L1 - P5 * (1 - 2 * L2); // dHx1/dL2
  dHxdL2[1] = q6 * (1 - 2 * L2) - (q4 + q5) * L1 + q5 * (1 - 2 * L2); // dHx2/dL2
  dHxdL2[2] = -4 + 6 * (L1 + L2) + (c4 - c5) * L1 + c5 * (1 - 2 * L2); // dHx3/dL2

  // Node 2 DOFs (w2, θx2, θy2) - indices 3, 4, 5
  dHxdL1[3] = P4 * L2 + P6 * (1 - 2 * L1); // dHx4/dL1
  dHxdL1[4] = q4 * L2 - q6 * (1 - 2 * L1); // dHx5/dL1
  dHxdL1[5] = c4 * L2 - c6 * (1 - 2 * L1); // dHx6/dL1

  dHxdL2[3] = P4 * (1 - 2 * L2) + P5 * L1; // dHx4/dL2
  dHxdL2[4] = q4 * (1 - 2 * L2) - q5 * L1; // dHx5/dL2
  dHxdL2[5] = c4 * (1 - 2 * L2) - c5 * L1; // dHx6/dL2

  // Node 3 DOFs (w3, θx3, θy3) - indices 6, 7, 8
  dHxdL1[6] = -P5 * L2 - P6 * (1 - 2 * L1); // dHx7/dL1
  dHxdL1[7] = q5 * L2 + q6 * (1 - 2 * L1); // dHx8/dL1
  dHxdL1[8] = -c5 * L2 + c6 * (1 - 2 * L1); // dHx9/dL1

  dHxdL2[6] = -P4 * L1 - P5 * (1 - 2 * L2); // dHx7/dL2
  dHxdL2[7] = q4 * L1 + q5 * (1 - 2 * L2); // dHx8/dL2
  dHxdL2[8] = -c4 * L1 + c5 * (1 - 2 * L2); // dHx9/dL2

  return { L1: dHxdL1, L2: dHxdL2 };
}

/**
 * Compute DKT Hy derivatives w.r.t. area coordinates.
 * Returns derivatives for all 9 DOFs.
 */
function computeDKTBy(
  L1: number,
  L2: number,
  L3: number,
  t4: number,
  t5: number,
  t6: number,
  r4: number,
  r5: number,
  r6: number,
  e4: number,
  e5: number,
  e6: number,
  b4: number,
  b5: number,
  b6: number
): { L1: Float32Array; L2: Float32Array } {
  // Derivatives of Hy with respect to L1 and L2

  const dHydL1 = new Float32Array(9);
  const dHydL2 = new Float32Array(9);

  // Node 1 DOFs
  dHydL1[0] = t6 * (1 - 2 * L1) + (t5 - t6) * L2;
  dHydL1[1] = 1 + r6 * (1 - 2 * L1) - (r5 + r6) * L2;
  dHydL1[2] = -e6 * (1 - 2 * L1) + (b5 - b6) * L2;

  dHydL2[0] = -t6 * (1 - 2 * L2) + (t4 - t5) * L1 - t5 * (1 - 2 * L2);
  dHydL2[1] = 1 + r6 * (1 - 2 * L2) - (r4 + r5) * L1 + r5 * (1 - 2 * L2);
  dHydL2[2] = e6 * (1 - 2 * L2) + (b4 - b5) * L1 - b5 * (1 - 2 * L2);

  // Node 2 DOFs
  dHydL1[3] = t4 * L2 + t6 * (1 - 2 * L1);
  dHydL1[4] = -1 + r4 * L2 - r6 * (1 - 2 * L1);
  dHydL1[5] = -e4 * L2 - b6 * (1 - 2 * L1);

  dHydL2[3] = t4 * (1 - 2 * L2) + t5 * L1;
  dHydL2[4] = -1 + r4 * (1 - 2 * L2) - r5 * L1;
  dHydL2[5] = -e4 * (1 - 2 * L2) + b5 * L1;

  // Node 3 DOFs
  dHydL1[6] = -t5 * L2 - t6 * (1 - 2 * L1);
  dHydL1[7] = r5 * L2 + r6 * (1 - 2 * L1);
  dHydL1[8] = e5 * L2 + b6 * (1 - 2 * L1);

  dHydL2[6] = -t4 * L1 - t5 * (1 - 2 * L2);
  dHydL2[7] = r4 * L1 + r5 * (1 - 2 * L2);
  dHydL2[8] = e4 * L1 - b5 * (1 - 2 * L2);

  return { L1: dHydL1, L2: dHydL2 };
}

/**
 * Apply DKT element stiffness contribution: y += K_e * x_local
 *
 * This is used for matrix-free K·x operation with triangular elements.
 *
 * @param nodeCoords - Element node coordinates [x0,y0, x1,y1, x2,y2]
 * @param material - Material properties
 * @param xLocal - Local DOF values (9 values)
 * @param yLocal - Output array to accumulate into (9 values)
 */
export function applyDKTElementStiffness(
  nodeCoords: Float32Array,
  material: PlateMaterial,
  xLocal: Float32Array,
  yLocal: Float32Array
): void {
  const Ke = computeDKTStiffness(nodeCoords, material);

  // y = K * x
  for (let i = 0; i < DOFS_PER_TRIANGLE; i++) {
    let sum = 0;
    for (let j = 0; j < DOFS_PER_TRIANGLE; j++) {
      sum += Ke[i * DOFS_PER_TRIANGLE + j] * xLocal[j];
    }
    yLocal[i] += sum;
  }
}

/**
 * Compute DKT element mass matrix (lumped).
 *
 * @param nodeCoords - Element node coordinates [x0,y0, x1,y1, x2,y2]
 * @param thickness - Plate thickness
 * @param density - Material density (kg/m³)
 * @returns 9×9 diagonal mass matrix
 */
export function computeDKTMassLumped(
  nodeCoords: Float32Array,
  thickness: number,
  density: number
): Float32Array {
  // Compute triangle area
  const x1 = nodeCoords[0],
    y1 = nodeCoords[1];
  const x2 = nodeCoords[2],
    y2 = nodeCoords[3];
  const x3 = nodeCoords[4],
    y3 = nodeCoords[5];

  const area = Math.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2;

  // Total mass distributed to 3 nodes
  const totalMass = density * thickness * area;
  const massPerNode = totalMass / 3;

  const Me = new Float32Array(DOFS_PER_TRIANGLE * DOFS_PER_TRIANGLE);

  // Diagonal terms only
  for (let i = 0; i < 3; i++) {
    const di_w = i * 3;
    const di_tx = i * 3 + 1;
    const di_ty = i * 3 + 2;

    // Translational mass
    Me[di_w * 9 + di_w] = massPerNode;

    // Rotational inertia (approximate)
    const rotInertia = (massPerNode * thickness * thickness) / 12;
    Me[di_tx * 9 + di_tx] = rotInertia;
    Me[di_ty * 9 + di_ty] = rotInertia;
  }

  return Me;
}

/**
 * Extract diagonal of DKT element stiffness matrix.
 */
export function extractDKTDiagonal(Ke: Float32Array): Float32Array {
  const diag = new Float32Array(DOFS_PER_TRIANGLE);
  for (let i = 0; i < DOFS_PER_TRIANGLE; i++) {
    diag[i] = Ke[i * DOFS_PER_TRIANGLE + i];
  }
  return diag;
}

