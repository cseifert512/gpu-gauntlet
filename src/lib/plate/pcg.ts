/**
 * Preconditioned Conjugate Gradient (PCG) solver.
 *
 * Solves K·x = b where K is SPD (symmetric positive definite).
 * Uses Jacobi preconditioner (diagonal of K).
 *
 * Algorithm:
 * 1. r = b - K·x (residual)
 * 2. z = M⁻¹·r (preconditioned residual)
 * 3. p = z (search direction)
 * 4. Iterate:
 *    - α = (r·z) / (p·K·p)
 *    - x = x + α·p
 *    - r = r - α·K·p
 *    - Check convergence
 *    - z = M⁻¹·r
 *    - β = (r_new·z_new) / (r·z)
 *    - p = z + β·p
 */

import type { PCGOptions, PCGResult } from './types';

/**
 * Compute dot product of two vectors.
 */
export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Compute L2 norm of a vector.
 */
export function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

/**
 * Copy vector: dst = src
 */
export function copy(dst: Float32Array, src: Float32Array): void {
  dst.set(src);
}

/**
 * Scale vector in place: a = α·a
 */
export function scale(a: Float32Array, alpha: number): void {
  for (let i = 0; i < a.length; i++) {
    a[i] *= alpha;
  }
}

/**
 * AXPY operation: y = α·x + y
 */
export function axpy(
  alpha: number,
  x: Float32Array,
  y: Float32Array
): void {
  for (let i = 0; i < x.length; i++) {
    y[i] += alpha * x[i];
  }
}

/**
 * Apply block preconditioner: z = M^-1 * r
 * Block diagonal has 9 floats per node (3x3 inverted block)
 */
function applyBlockPrecond(
  blockInv: Float32Array,
  r: Float32Array,
  z: Float32Array
): void {
  const nodeCount = blockInv.length / 9;
  for (let node = 0; node < nodeCount; node++) {
    const bo = node * 9;
    const do_ = node * 3;
    const r0 = r[do_], r1 = r[do_ + 1], r2 = r[do_ + 2];
    z[do_]     = blockInv[bo + 0] * r0 + blockInv[bo + 1] * r1 + blockInv[bo + 2] * r2;
    z[do_ + 1] = blockInv[bo + 3] * r0 + blockInv[bo + 4] * r1 + blockInv[bo + 5] * r2;
    z[do_ + 2] = blockInv[bo + 6] * r0 + blockInv[bo + 7] * r1 + blockInv[bo + 8] * r2;
  }
}

/**
 * Solve K·x = b using Preconditioned Conjugate Gradient.
 *
 * @param applyK - Function that computes y = K·x (matrix-free)
 * @param b - Right-hand side vector
 * @param options - Solver options
 * @returns Solution and statistics
 */
export function solvePCG(
  applyK: (x: Float32Array, y: Float32Array) => void,
  b: Float32Array,
  options: PCGOptions
): PCGResult {
  const n = b.length;
  const { tolerance, maxIterations, preconditioner: M, blockPreconditioner: blockM } = options;

  // Allocate working vectors
  const x = new Float32Array(n); // Solution (initial guess = 0)
  const r = new Float32Array(n); // Residual
  const z = new Float32Array(n); // Preconditioned residual
  const p = new Float32Array(n); // Search direction
  const Ap = new Float32Array(n); // K·p

  // r = b - K·x (x=0, so r=b)
  copy(r, b);

  // Compute initial residual norm for relative convergence
  const b_norm = norm(b);
  if (b_norm < 1e-30) {
    // Zero RHS → zero solution
    return {
      solution: x,
      iterations: 0,
      finalResidual: 0,
      converged: true,
    };
  }

  // Apply preconditioner: z = M⁻¹·r
  // Prefer block preconditioner if available
  if (blockM) {
    applyBlockPrecond(blockM, r, z);
  } else if (M) {
    for (let i = 0; i < n; i++) {
      z[i] = M[i] > 1e-30 ? r[i] / M[i] : r[i];
    }
  } else {
    copy(z, r);
  }

  // p = z
  copy(p, z);

  // rz = r·z
  let rz = dot(r, z);

  for (let iter = 0; iter < maxIterations; iter++) {
    // Ap = K·p
    applyK(p, Ap);

    // pAp = p·Ap
    const pAp = dot(p, Ap);

    // Guard against division by zero
    if (Math.abs(pAp) < 1e-30) {
      return {
        solution: x,
        iterations: iter + 1,
        finalResidual: norm(r),
        converged: false,
      };
    }

    // α = rz / pAp
    const alpha = rz / pAp;

    // x = x + α·p
    axpy(alpha, p, x);

    // r = r - α·Ap
    axpy(-alpha, Ap, r);

    // Check convergence
    const residualNorm = norm(r);
    if (residualNorm < tolerance || residualNorm / b_norm < tolerance) {
      return {
        solution: x,
        iterations: iter + 1,
        finalResidual: residualNorm,
        converged: true,
      };
    }

    // z = M⁻¹·r
    if (blockM) {
      applyBlockPrecond(blockM, r, z);
    } else if (M) {
      for (let i = 0; i < n; i++) {
        z[i] = M[i] > 1e-30 ? r[i] / M[i] : r[i];
      }
    } else {
      copy(z, r);
    }

    // rz_new = r·z
    const rz_new = dot(r, z);

    // β = rz_new / rz
    const beta = rz_new / rz;

    // p = z + β·p
    for (let i = 0; i < n; i++) {
      p[i] = z[i] + beta * p[i];
    }

    rz = rz_new;
  }

  // Did not converge
  return {
    solution: x,
    iterations: maxIterations,
    finalResidual: norm(r),
    converged: false,
  };
}

/**
 * Solve a simple test system for validation.
 * Solves A·x = b where A is a simple tridiagonal SPD matrix.
 */
export function solveTestSystem(): PCGResult {
  const n = 10;

  // Create a simple tridiagonal SPD matrix: A[i,i] = 2, A[i,i±1] = -1
  const applyA = (x: Float32Array, y: Float32Array) => {
    y.fill(0);
    for (let i = 0; i < n; i++) {
      y[i] = 2 * x[i];
      if (i > 0) y[i] -= x[i - 1];
      if (i < n - 1) y[i] -= x[i + 1];
    }
  };

  // RHS: b = [1, 0, 0, ..., 0, 1]
  const b = new Float32Array(n);
  b[0] = 1;
  b[n - 1] = 1;

  // Diagonal preconditioner
  const M = new Float32Array(n).fill(2);

  return solvePCG(applyA, b, {
    tolerance: 1e-10,
    maxIterations: 100,
    preconditioner: M,
  });
}

