/**
 * Post-processing for plate solver results.
 *
 * Computes bending moments Mx, My, Mxy from nodal displacements.
 * Supports both Q4 (quad) and DKT (triangle) elements.
 *
 * Moment-curvature relationship:
 *   Mx  = D * (κx + ν·κy)
 *   My  = D * (κy + ν·κx)
 *   Mxy = D * (1-ν)/2 * κxy
 *
 * where curvatures are:
 *   κx  = ∂θy/∂x
 *   κy  = -∂θx/∂y
 *   κxy = ∂θy/∂y - ∂θx/∂x
 */

import type { PlateMesh, PlateMaterial } from './types';
import { computeFlexuralRigidity, DOFS_PER_NODE } from './types';
import {
  computeShapeFunctionDerivatives,
  computeJacobian,
  computePhysicalDerivatives,
} from './element';

/**
 * Compute bending moments at nodes from displacement field.
 *
 * Strategy:
 * 1. For each element, compute moments at centroid
 * 2. Average contributions from adjacent elements
 *
 * Supports both Q4 (quad) and DKT (triangle) elements.
 *
 * @param mesh - Plate mesh
 * @param displacements - Full DOF vector [w0,θx0,θy0, w1,θx1,θy1, ...]
 * @param material - Material properties
 * @returns Nodal moment arrays
 */
export function computeMoments(
  mesh: PlateMesh,
  displacements: Float32Array,
  material: PlateMaterial
): { Mx: Float32Array; My: Float32Array; Mxy: Float32Array } {
  const nodesPerElem = mesh.nodesPerElement ?? 4;

  if (nodesPerElem === 3) {
    return computeMomentsTriangle(mesh, displacements, material);
  } else {
    return computeMomentsQuad(mesh, displacements, material);
  }
}

/**
 * Compute moments for Q4 (quad) elements.
 */
function computeMomentsQuad(
  mesh: PlateMesh,
  displacements: Float32Array,
  material: PlateMaterial
): { Mx: Float32Array; My: Float32Array; Mxy: Float32Array } {
  const { nodeCount, elementCount, elements, nodes } = mesh;
  const D = computeFlexuralRigidity(material);
  const { nu } = material;

  // Accumulators for nodal averaging
  const Mx_sum = new Float32Array(nodeCount);
  const My_sum = new Float32Array(nodeCount);
  const Mxy_sum = new Float32Array(nodeCount);
  const count = new Float32Array(nodeCount);

  // Process each element
  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    // Get element nodes
    const n0 = elements[elemIdx * 4];
    const n1 = elements[elemIdx * 4 + 1];
    const n2 = elements[elemIdx * 4 + 2];
    const n3 = elements[elemIdx * 4 + 3];
    const nodeIndices = [n0, n1, n2, n3];

    // Get node coordinates
    const coords = new Float32Array(8);
    for (let i = 0; i < 4; i++) {
      coords[i * 2] = nodes[nodeIndices[i] * 2];
      coords[i * 2 + 1] = nodes[nodeIndices[i] * 2 + 1];
    }

    // Get element DOFs (θx and θy for each node)
    const theta_x = new Float32Array(4);
    const theta_y = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
      const nodeIdx = nodeIndices[i];
      theta_x[i] = displacements[nodeIdx * DOFS_PER_NODE + 1]; // θx
      theta_y[i] = displacements[nodeIdx * DOFS_PER_NODE + 2]; // θy
    }

    // Compute moments at element center (ξ=0, η=0) and extrapolate to nodes
    const xi = 0;
    const eta = 0;

    const { dNdXi, dNdEta } = computeShapeFunctionDerivatives(xi, eta);
    const { J11, J12, J21, J22, detJ } = computeJacobian(
      dNdXi,
      dNdEta,
      coords
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

    // Compute curvatures
    let kappa_x = 0;
    let kappa_y = 0;
    let kappa_xy = 0;

    for (let i = 0; i < 4; i++) {
      kappa_x += dNdx[i] * theta_y[i];
      kappa_y += -dNdy[i] * theta_x[i];
      kappa_xy += dNdy[i] * theta_y[i] - dNdx[i] * theta_x[i];
    }

    // Compute moments
    const Mx_elem = D * (kappa_x + nu * kappa_y);
    const My_elem = D * (kappa_y + nu * kappa_x);
    const Mxy_elem = (D * (1 - nu) * kappa_xy) / 2;

    // Accumulate to nodes
    for (let i = 0; i < 4; i++) {
      const nodeIdx = nodeIndices[i];
      Mx_sum[nodeIdx] += Mx_elem;
      My_sum[nodeIdx] += My_elem;
      Mxy_sum[nodeIdx] += Mxy_elem;
      count[nodeIdx] += 1;
    }
  }

  // Average
  const Mx = new Float32Array(nodeCount);
  const My = new Float32Array(nodeCount);
  const Mxy = new Float32Array(nodeCount);

  for (let i = 0; i < nodeCount; i++) {
    if (count[i] > 0) {
      Mx[i] = Mx_sum[i] / count[i];
      My[i] = My_sum[i] / count[i];
      Mxy[i] = Mxy_sum[i] / count[i];
    }
  }

  return { Mx, My, Mxy };
}

/**
 * Compute moments for DKT (triangle) elements.
 *
 * Uses constant strain approach: curvatures computed at centroid.
 */
function computeMomentsTriangle(
  mesh: PlateMesh,
  displacements: Float32Array,
  material: PlateMaterial
): { Mx: Float32Array; My: Float32Array; Mxy: Float32Array } {
  const { nodeCount, elementCount, elements, nodes } = mesh;
  const D = computeFlexuralRigidity(material);
  const { nu } = material;

  // Accumulators for nodal averaging
  const Mx_sum = new Float32Array(nodeCount);
  const My_sum = new Float32Array(nodeCount);
  const Mxy_sum = new Float32Array(nodeCount);
  const count = new Float32Array(nodeCount);

  // Process each triangle element
  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    // Get element nodes
    const n0 = elements[elemIdx * 3];
    const n1 = elements[elemIdx * 3 + 1];
    const n2 = elements[elemIdx * 3 + 2];
    const nodeIndices = [n0, n1, n2];

    // Get node coordinates
    const x1 = nodes[n0 * 2],
      y1 = nodes[n0 * 2 + 1];
    const x2 = nodes[n1 * 2],
      y2 = nodes[n1 * 2 + 1];
    const x3 = nodes[n2 * 2],
      y3 = nodes[n2 * 2 + 1];

    // Triangle area (2A)
    const twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    if (Math.abs(twoA) < 1e-15) continue; // Skip degenerate triangles

    const invTwoA = 1 / twoA;

    // Get element DOFs (θx and θy for each node)
    const theta_x = new Float32Array(3);
    const theta_y = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      const nodeIdx = nodeIndices[i];
      theta_x[i] = displacements[nodeIdx * DOFS_PER_NODE + 1]; // θx
      theta_y[i] = displacements[nodeIdx * DOFS_PER_NODE + 2]; // θy
    }

    // For triangles, use linear shape functions for moment computation
    // dN/dx = (y_jk) / (2A), dN/dy = (x_kj) / (2A)
    // where i,j,k are cyclic

    // Shape function derivatives (constant for linear triangles)
    const dNdx = new Float32Array(3);
    const dNdy = new Float32Array(3);

    // Node 1 (i=0): j=1, k=2
    dNdx[0] = (y2 - y3) * invTwoA;
    dNdy[0] = (x3 - x2) * invTwoA;

    // Node 2 (i=1): j=2, k=0
    dNdx[1] = (y3 - y1) * invTwoA;
    dNdy[1] = (x1 - x3) * invTwoA;

    // Node 3 (i=2): j=0, k=1
    dNdx[2] = (y1 - y2) * invTwoA;
    dNdy[2] = (x2 - x1) * invTwoA;

    // Compute curvatures (constant within element)
    let kappa_x = 0;
    let kappa_y = 0;
    let kappa_xy = 0;

    for (let i = 0; i < 3; i++) {
      kappa_x += dNdx[i] * theta_y[i];
      kappa_y += -dNdy[i] * theta_x[i];
      kappa_xy += dNdy[i] * theta_y[i] - dNdx[i] * theta_x[i];
    }

    // Compute moments
    const Mx_elem = D * (kappa_x + nu * kappa_y);
    const My_elem = D * (kappa_y + nu * kappa_x);
    const Mxy_elem = (D * (1 - nu) * kappa_xy) / 2;

    // Accumulate to nodes
    for (let i = 0; i < 3; i++) {
      const nodeIdx = nodeIndices[i];
      Mx_sum[nodeIdx] += Mx_elem;
      My_sum[nodeIdx] += My_elem;
      Mxy_sum[nodeIdx] += Mxy_elem;
      count[nodeIdx] += 1;
    }
  }

  // Average
  const Mx = new Float32Array(nodeCount);
  const My = new Float32Array(nodeCount);
  const Mxy = new Float32Array(nodeCount);

  for (let i = 0; i < nodeCount; i++) {
    if (count[i] > 0) {
      Mx[i] = Mx_sum[i] / count[i];
      My[i] = My_sum[i] / count[i];
      Mxy[i] = Mxy_sum[i] / count[i];
    }
  }

  return { Mx, My, Mxy };
}

/**
 * Extract vertical displacements from full DOF vector.
 *
 * @param displacements - Full DOF vector [w0,θx0,θy0, w1,θx1,θy1, ...]
 * @param nodeCount - Number of nodes
 * @returns Array of w values
 */
export function extractVerticalDisplacements(
  displacements: Float32Array,
  nodeCount: number
): Float32Array {
  const w = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    w[i] = displacements[i * DOFS_PER_NODE]; // w is first DOF
  }
  return w;
}

/**
 * Find maximum displacement and its location.
 *
 * @param w - Vertical displacements
 * @param nodes - Node coordinates
 * @returns Maximum value and location
 */
export function findMaxDisplacement(
  w: Float32Array,
  nodes: Float32Array
): { maxW: number; nodeIndex: number; x: number; y: number } {
  let maxW = -Infinity;
  let maxIdx = 0;

  for (let i = 0; i < w.length; i++) {
    // Take absolute value since displacements can be positive or negative
    const absW = Math.abs(w[i]);
    if (absW > maxW) {
      maxW = absW;
      maxIdx = i;
    }
  }

  return {
    maxW: w[maxIdx], // Return actual value (with sign)
    nodeIndex: maxIdx,
    x: nodes[maxIdx * 2],
    y: nodes[maxIdx * 2 + 1],
  };
}

/**
 * Find maximum moment values.
 */
export function findMaxMoments(
  Mx: Float32Array,
  My: Float32Array,
  Mxy: Float32Array
): { maxMx: number; maxMy: number; maxMxy: number } {
  let maxMx = -Infinity;
  let maxMy = -Infinity;
  let maxMxy = -Infinity;

  for (let i = 0; i < Mx.length; i++) {
    if (Math.abs(Mx[i]) > Math.abs(maxMx)) maxMx = Mx[i];
    if (Math.abs(My[i]) > Math.abs(maxMy)) maxMy = My[i];
    if (Math.abs(Mxy[i]) > Math.abs(maxMxy)) maxMxy = Mxy[i];
  }

  return { maxMx, maxMy, maxMxy };
}

