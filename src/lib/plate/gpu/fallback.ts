/**
 * CPU fallback solver for when WebGPU is unavailable.
 *
 * This module provides a clean interface that matches the GPU solver API,
 * using the existing CPU PCG implementation under the hood.
 */

import type { PlateMesh, PlateMaterial, ElementColoring, PCGResult } from '../types';
import { solvePCG } from '../pcg';
import { applyGlobalK, computeDiagonal, applyBCsToDiagonal } from '../solver';

/**
 * CPU fallback solver options.
 */
export interface CPUSolverOptions {
  /** Convergence tolerance (default 1e-8) */
  tolerance?: number;
  /** Max iterations (default 1000) */
  maxIterations?: number;
}

/**
 * CPU solve result (compatible with GPU result interface).
 */
export interface CPUSolveResult {
  /** Solution vector [w0,θx0,θy0, w1,θx1,θy1, ...] */
  solution: Float32Array;
  /** Number of iterations */
  iterations: number;
  /** Final residual norm */
  finalResidual: number;
  /** Whether solution converged */
  converged: boolean;
  /** Solve time in milliseconds */
  solveTimeMs: number;
}

/**
 * Solve plate problem using CPU (for fallback or comparison).
 *
 * @param mesh - Plate mesh
 * @param material - Material properties
 * @param coloring - Element coloring
 * @param F - Load vector
 * @param constrainedDOFs - Constrained DOF indices
 * @param options - Solver options
 * @returns Solution and statistics
 */
export function solveCPU(
  mesh: PlateMesh,
  material: PlateMaterial,
  coloring: ElementColoring,
  F: Float32Array,
  constrainedDOFs: Set<number>,
  options: CPUSolverOptions = {}
): CPUSolveResult {
  const startTime = performance.now();

  const tolerance = options.tolerance ?? 1e-8;
  const maxIterations = options.maxIterations ?? 1000;

  // Build preconditioner
  const diagonal = computeDiagonal(mesh, material);
  applyBCsToDiagonal(diagonal, constrainedDOFs);

  // Define K·x operation
  const applyK = (x: Float32Array, y: Float32Array) => {
    applyGlobalK(mesh, material, coloring, x, y, constrainedDOFs);
  };

  // Solve using PCG
  const result = solvePCG(applyK, F, {
    tolerance,
    maxIterations,
    preconditioner: diagonal,
  });

  const solveTimeMs = performance.now() - startTime;

  return {
    solution: result.solution,
    iterations: result.iterations,
    finalResidual: result.finalResidual,
    converged: result.converged,
    solveTimeMs,
  };
}

/**
 * Compare two solution vectors.
 *
 * @param a - First solution
 * @param b - Second solution
 * @param tolerance - Maximum relative error
 * @returns Comparison statistics
 */
export function compareSolutions(
  a: Float32Array,
  b: Float32Array,
  tolerance: number = 1e-4
): {
  maxAbsError: number;
  maxRelError: number;
  indexOfMaxError: number;
  passed: boolean;
  rmsError: number;
} {
  if (a.length !== b.length) {
    throw new Error(`Solution length mismatch: ${a.length} vs ${b.length}`);
  }

  let maxAbsError = 0;
  let maxRelError = 0;
  let indexOfMaxError = 0;
  let sumSquaredError = 0;

  for (let i = 0; i < a.length; i++) {
    const absError = Math.abs(a[i] - b[i]);
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]), 1e-15);
    const relError = absError / denom;

    sumSquaredError += absError * absError;

    if (relError > maxRelError) {
      maxRelError = relError;
      maxAbsError = absError;
      indexOfMaxError = i;
    }
  }

  const rmsError = Math.sqrt(sumSquaredError / a.length);
  const passed = maxRelError < tolerance;

  return {
    maxAbsError,
    maxRelError,
    indexOfMaxError,
    passed,
    rmsError,
  };
}

