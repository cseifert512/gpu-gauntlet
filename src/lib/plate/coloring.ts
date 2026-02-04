/**
 * Element coloring for conflict-free parallel assembly.
 *
 * For a structured quad grid, elements can be colored like a checkerboard
 * pattern with 4 colors such that no two adjacent elements share the same
 * color. Adjacent = sharing a node.
 *
 * This enables GPU parallelism: all elements of one color can be processed
 * simultaneously without atomic operations or race conditions.
 *
 * Coloring scheme for structured grid:
 *   Color = (i % 2) * 2 + (j % 2)
 *
 *   Where (i, j) is the element's position in the grid.
 *
 *   Example 4×3 grid:
 *     +---+---+---+---+
 *     | 2 | 3 | 2 | 3 |  j=2
 *     +---+---+---+---+
 *     | 0 | 1 | 0 | 1 |  j=1
 *     +---+---+---+---+
 *     | 2 | 3 | 2 | 3 |  j=0
 *     +---+---+---+---+
 *       i=0 i=1 i=2 i=3
 */

import type { PlateMesh, ElementColoring } from './types';

/**
 * Compute element coloring for structured quad mesh.
 *
 * For structured grid: simple 2×2 pattern gives 4 colors.
 * This ensures no two elements sharing a node have the same color.
 *
 * @param mesh - Plate mesh (must be structured quad grid)
 * @returns Coloring data structure
 */
export function computeElementColoring(mesh: PlateMesh): ElementColoring {
  const { elementCount, gridNx } = mesh;

  // Allocate color assignment for each element
  const elementColors = new Uint8Array(elementCount);

  // Count elements per color
  const colorCounts = [0, 0, 0, 0];

  // Assign colors based on grid position
  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    // For structured grid, element at flat index elemIdx has grid position:
    // i = elemIdx % gridNx
    // j = elemIdx / gridNx (integer division)
    const i = elemIdx % gridNx;
    const j = Math.floor(elemIdx / gridNx);

    // Color assignment: checkerboard pattern
    const color = (i % 2) * 2 + (j % 2);
    elementColors[elemIdx] = color;
    colorCounts[color]++;
  }

  // Build arrays of element indices per color
  const colors: Uint32Array[] = [];
  const colorIndices = [0, 0, 0, 0]; // Current write position for each color

  // Allocate arrays
  for (let c = 0; c < 4; c++) {
    colors.push(new Uint32Array(colorCounts[c]));
  }

  // Fill arrays
  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    const color = elementColors[elemIdx];
    colors[color][colorIndices[color]++] = elemIdx;
  }

  return {
    colors,
    elementColors,
    colorCount: 4,
  };
}

/**
 * Compute element coloring using greedy graph coloring.
 *
 * This is used for unstructured meshes (triangles) where the simple
 * checkerboard pattern doesn't apply.
 *
 * @param mesh - Plate mesh
 * @returns Coloring data structure
 */
export function computeElementColoringGreedy(mesh: PlateMesh): ElementColoring {
  const { elementCount, elements } = mesh;
  const nodesPerElem = mesh.nodesPerElement ?? 4;

  // Build adjacency: elements sharing a node are adjacent
  // nodeToElements: for each node, list of elements containing it
  const nodeToElements: number[][] = [];

  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    for (let i = 0; i < nodesPerElem; i++) {
      const nodeIdx = elements[elemIdx * nodesPerElem + i];
      if (!nodeToElements[nodeIdx]) {
        nodeToElements[nodeIdx] = [];
      }
      nodeToElements[nodeIdx].push(elemIdx);
    }
  }

  // Build element adjacency
  const elementAdjacency: Set<number>[] = [];
  for (let i = 0; i < elementCount; i++) {
    elementAdjacency.push(new Set());
  }

  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    for (let i = 0; i < nodesPerElem; i++) {
      const nodeIdx = elements[elemIdx * nodesPerElem + i];
      for (const neighborElem of nodeToElements[nodeIdx]) {
        if (neighborElem !== elemIdx) {
          elementAdjacency[elemIdx].add(neighborElem);
        }
      }
    }
  }

  // Greedy coloring
  const elementColors = new Uint8Array(elementCount);
  elementColors.fill(255); // Uncolored

  let maxColor = 0;

  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    // Find colors used by neighbors
    const usedColors = new Set<number>();
    for (const neighbor of elementAdjacency[elemIdx]) {
      if (elementColors[neighbor] !== 255) {
        usedColors.add(elementColors[neighbor]);
      }
    }

    // Find smallest available color
    let color = 0;
    while (usedColors.has(color)) {
      color++;
    }

    elementColors[elemIdx] = color;
    maxColor = Math.max(maxColor, color);
  }

  const colorCount = maxColor + 1;

  // Build color arrays
  const colorCounts = new Array(colorCount).fill(0);
  for (let i = 0; i < elementCount; i++) {
    colorCounts[elementColors[i]]++;
  }

  const colors: Uint32Array[] = [];
  const colorIndices = new Array(colorCount).fill(0);

  for (let c = 0; c < colorCount; c++) {
    colors.push(new Uint32Array(colorCounts[c]));
  }

  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    const color = elementColors[elemIdx];
    colors[color][colorIndices[color]++] = elemIdx;
  }

  return {
    colors,
    elementColors,
    colorCount,
  };
}

/**
 * Verify coloring is valid (no conflicts).
 *
 * Two elements conflict if they share a node and have the same color.
 *
 * @param mesh - Plate mesh
 * @param coloring - Computed coloring
 * @returns True if valid (no conflicts)
 */
export function verifyColoring(
  mesh: PlateMesh,
  coloring: ElementColoring
): boolean {
  const { elementCount, elements } = mesh;
  const { elementColors } = coloring;
  const nodesPerElem = mesh.nodesPerElement ?? 4;

  // Build node-to-elements mapping
  const nodeToElements: number[][] = [];

  for (let elemIdx = 0; elemIdx < elementCount; elemIdx++) {
    for (let i = 0; i < nodesPerElem; i++) {
      const nodeIdx = elements[elemIdx * nodesPerElem + i];
      if (!nodeToElements[nodeIdx]) {
        nodeToElements[nodeIdx] = [];
      }
      nodeToElements[nodeIdx].push(elemIdx);
    }
  }

  // Check for conflicts
  for (const elementsAtNode of nodeToElements) {
    if (!elementsAtNode) continue;

    // All elements at this node should have different colors
    const colorsUsed = new Set<number>();
    for (const elemIdx of elementsAtNode) {
      const color = elementColors[elemIdx];
      if (colorsUsed.has(color)) {
        return false; // Conflict found
      }
      colorsUsed.add(color);
    }
  }

  return true;
}

/**
 * Get statistics about the coloring.
 *
 * @param coloring - Element coloring
 * @returns Statistics object
 */
export function getColoringStats(coloring: ElementColoring): {
  colorCount: number;
  elementsPerColor: number[];
  minElementsPerColor: number;
  maxElementsPerColor: number;
  balance: number; // max/min ratio
} {
  const elementsPerColor = coloring.colors.map((c) => c.length);
  const minElements = Math.min(...elementsPerColor);
  const maxElements = Math.max(...elementsPerColor);

  return {
    colorCount: coloring.colorCount,
    elementsPerColor,
    minElementsPerColor: minElements,
    maxElementsPerColor: maxElements,
    balance: minElements > 0 ? maxElements / minElements : Infinity,
  };
}

/**
 * Iterate over elements by color (useful for parallel processing simulation).
 *
 * @param coloring - Element coloring
 * @param callback - Function to call for each element
 */
export function iterateByColor(
  coloring: ElementColoring,
  callback: (elementIndex: number, color: number) => void
): void {
  for (let color = 0; color < coloring.colorCount; color++) {
    const elements = coloring.colors[color];
    for (let i = 0; i < elements.length; i++) {
      callback(elements[i], color);
    }
  }
}

