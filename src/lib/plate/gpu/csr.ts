/**
 * CSR (Compressed Sparse Row) matrix assembly for the global stiffness matrix.
 *
 * Pre-assembles K into CSR format on CPU, then uploads to GPU.
 * GPU SpMV (sparse matrix-vector product) replaces element-by-element K·p,
 * reducing from ~6 color dispatches to a single SpMV dispatch per iteration.
 */

import type { PlateMesh, PlateMaterial } from '../types';
import { getElementCoords, getElementNodeIndices } from '../mesher';
import { computeElementStiffness, computeDKTStiffness } from '../element';

export interface CSRMatrix {
  rowPtr: Uint32Array;   // length: nRows + 1
  colIdx: Uint32Array;   // length: nnz
  values: Float32Array;  // length: nnz
  nRows: number;
  nnz: number;
}

/**
 * Assemble global stiffness matrix K in CSR format.
 * Boundary conditions are applied: constrained rows become identity rows.
 */
export function assembleCSR(
  mesh: PlateMesh,
  material: PlateMaterial,
  constrainedDOFs: Set<number>
): CSRMatrix {
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const dofsPerElem = nodesPerElem * 3;
  const dofCount = mesh.nodeCount * 3;

  // Phase 1: Determine sparsity pattern (which columns each row touches)
  // Use arrays of Sets for the pattern
  const rowCols: Set<number>[] = new Array(dofCount);
  for (let i = 0; i < dofCount; i++) {
    rowCols[i] = new Set<number>();
  }

  for (let elemIdx = 0; elemIdx < mesh.elementCount; elemIdx++) {
    const nodeIndices = getElementNodeIndices(mesh, elemIdx);
    const dofs: number[] = new Array(dofsPerElem);
    for (let n = 0; n < nodesPerElem; n++) {
      const base = nodeIndices[n] * 3;
      dofs[n * 3] = base;
      dofs[n * 3 + 1] = base + 1;
      dofs[n * 3 + 2] = base + 2;
    }

    for (let i = 0; i < dofsPerElem; i++) {
      const row = dofs[i];
      for (let j = 0; j < dofsPerElem; j++) {
        rowCols[row].add(dofs[j]);
      }
    }
  }

  // Ensure constrained DOFs have at least a diagonal entry
  for (const dof of constrainedDOFs) {
    rowCols[dof].add(dof);
  }

  // Phase 2: Build row pointers and sorted column indices
  const rowPtr = new Uint32Array(dofCount + 1);
  let totalNnz = 0;
  for (let i = 0; i < dofCount; i++) {
    rowPtr[i] = totalNnz;
    totalNnz += rowCols[i].size;
  }
  rowPtr[dofCount] = totalNnz;

  const colIdx = new Uint32Array(totalNnz);

  // Sort column indices for each row (for binary search during assembly)
  for (let i = 0; i < dofCount; i++) {
    const cols = Array.from(rowCols[i]).sort((a, b) => a - b);
    const start = rowPtr[i];
    for (let j = 0; j < cols.length; j++) {
      colIdx[start + j] = cols[j];
    }
  }

  // Phase 3: Assemble values
  const values = new Float32Array(totalNnz); // initialized to 0

  for (let elemIdx = 0; elemIdx < mesh.elementCount; elemIdx++) {
    const coords = getElementCoords(mesh, elemIdx);
    const Ke = nodesPerElem === 3
      ? computeDKTStiffness(coords, material)
      : computeElementStiffness(coords, material);

    const nodeIndices = getElementNodeIndices(mesh, elemIdx);
    const dofs: number[] = new Array(dofsPerElem);
    for (let n = 0; n < nodesPerElem; n++) {
      const base = nodeIndices[n] * 3;
      dofs[n * 3] = base;
      dofs[n * 3 + 1] = base + 1;
      dofs[n * 3 + 2] = base + 2;
    }

    for (let i = 0; i < dofsPerElem; i++) {
      const row = dofs[i];
      const rowStart = rowPtr[row];
      const rowEnd = rowPtr[row + 1];

      for (let j = 0; j < dofsPerElem; j++) {
        const col = dofs[j];
        const val = Ke[i * dofsPerElem + j];

        // Binary search for col in colIdx[rowStart..rowEnd)
        let lo = rowStart;
        let hi = rowEnd;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if (colIdx[mid] < col) lo = mid + 1;
          else hi = mid;
        }
        values[lo] += val;
      }
    }
  }

  // Phase 4: Apply boundary conditions (constrained rows → identity)
  for (const dof of constrainedDOFs) {
    const rowStart = rowPtr[dof];
    const rowEnd = rowPtr[dof + 1];
    for (let j = rowStart; j < rowEnd; j++) {
      values[j] = (colIdx[j] === dof) ? 1.0 : 0.0;
    }
  }

  return { rowPtr, colIdx, values, nRows: dofCount, nnz: totalNnz };
}

