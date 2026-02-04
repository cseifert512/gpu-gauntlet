/**
 * Mesher for plate geometry.
 *
 * Supports two strategies:
 * 1. Structured quad grid: For simple rectangular domains (faster)
 * 2. Unstructured triangles (CDT): For complex polygons, circles, holes
 *
 * The generateMesh function auto-selects the appropriate strategy.
 */

import type { PlateGeometry, PlateMesh } from './types';
import { generateUnstructuredMesh } from './mesher-unstructured';

/**
 * Check if a point is inside a polygon using ray casting algorithm.
 *
 * @param px - Point X coordinate
 * @param py - Point Y coordinate
 * @param polygon - Flat polygon array [x0,y0, x1,y1, ...]
 * @returns True if point is inside the polygon
 */
export function pointInPolygon(
  px: number,
  py: number,
  polygon: Float32Array
): boolean {
  const n = polygon.length / 2;
  let inside = false;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i * 2];
    const yi = polygon[i * 2 + 1];
    const xj = polygon[j * 2];
    const yj = polygon[j * 2 + 1];

    // Ray casting: count intersections with horizontal ray from (px, py) to right
    const intersect =
      yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;

    if (intersect) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * Check if a point is on the boundary of a polygon (within tolerance).
 *
 * @param px - Point X coordinate
 * @param py - Point Y coordinate
 * @param polygon - Flat polygon array [x0,y0, x1,y1, ...]
 * @param tolerance - Distance tolerance
 * @returns True if point is on polygon boundary
 */
export function pointOnPolygonBoundary(
  px: number,
  py: number,
  polygon: Float32Array,
  tolerance: number = 1e-6
): boolean {
  const n = polygon.length / 2;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const x1 = polygon[i * 2];
    const y1 = polygon[i * 2 + 1];
    const x2 = polygon[j * 2];
    const y2 = polygon[j * 2 + 1];

    // Check distance from point to line segment
    const dist = pointToSegmentDistance(px, py, x1, y1, x2, y2);
    if (dist < tolerance) {
      return true;
    }
  }

  return false;
}

/**
 * Compute distance from point to line segment.
 */
function pointToSegmentDistance(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSq = dx * dx + dy * dy;

  if (lengthSq === 0) {
    // Segment is a point
    return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
  }

  // Project point onto line, clamping to segment
  let t = ((px - x1) * dx + (py - y1) * dy) / lengthSq;
  t = Math.max(0, Math.min(1, t));

  const closestX = x1 + t * dx;
  const closestY = y1 + t * dy;

  return Math.sqrt((px - closestX) ** 2 + (py - closestY) ** 2);
}

/**
 * Compute bounding box of a polygon.
 */
function computeBoundingBox(polygon: Float32Array): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const n = polygon.length / 2;
  for (let i = 0; i < n; i++) {
    const x = polygon[i * 2];
    const y = polygon[i * 2 + 1];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return { minX, minY, maxX, maxY };
}

/**
 * Generate mesh for plate geometry.
 *
 * Automatically selects between:
 * - Structured quad grid: For simple rectangular domains (faster)
 * - Unstructured triangles: For complex polygons, circles, holes
 *
 * @param geometry - Plate geometry
 * @param targetSize - Target element edge length
 * @returns Mesh (quads or triangles depending on geometry)
 */
export function generateMesh(
  geometry: PlateGeometry,
  targetSize: number = 0.5
): PlateMesh {
  // Detect if geometry is simple rectangle
  if (isSimpleRectangle(geometry)) {
    return generateStructuredMesh(geometry, targetSize);
  }

  // Use unstructured for complex geometry
  return generateUnstructuredMesh(geometry, { targetEdgeLength: targetSize });
}

/**
 * Check if geometry is a simple axis-aligned rectangle with no holes.
 */
export function isSimpleRectangle(geometry: PlateGeometry): boolean {
  // Must have no holes
  if (geometry.holes.length > 0) return false;

  // Must have exactly 4 vertices
  if (geometry.boundary.length !== 8) return false;

  // Check if axis-aligned rectangle
  const xs = [
    geometry.boundary[0],
    geometry.boundary[2],
    geometry.boundary[4],
    geometry.boundary[6],
  ];
  const ys = [
    geometry.boundary[1],
    geometry.boundary[3],
    geometry.boundary[5],
    geometry.boundary[7],
  ];

  const uniqueX = [...new Set(xs)];
  const uniqueY = [...new Set(ys)];

  // Axis-aligned rectangle has exactly 2 unique X and 2 unique Y values
  return uniqueX.length === 2 && uniqueY.length === 2;
}

/**
 * Force unstructured mesh generation.
 * Use this when you specifically need triangular elements.
 */
export function generateTriangularMesh(
  geometry: PlateGeometry,
  targetSize: number = 0.5
): PlateMesh {
  return generateUnstructuredMesh(geometry, { targetEdgeLength: targetSize });
}

/**
 * Generate structured quad mesh for plate geometry.
 * Internal function - use generateMesh for auto-selection.
 *
 * @param geometry - Plate boundary and holes
 * @param targetSize - Target element edge length (m)
 * @returns Mesh data structure with quad elements
 */
function generateStructuredMesh(
  geometry: PlateGeometry,
  targetSize: number = 0.5
): PlateMesh {
  // 1. Compute bounding box
  const { minX, minY, maxX, maxY } = computeBoundingBox(geometry.boundary);
  const width = maxX - minX;
  const height = maxY - minY;

  // 2. Compute grid dimensions
  const nx = Math.max(2, Math.ceil(width / targetSize));
  const ny = Math.max(2, Math.ceil(height / targetSize));

  // Actual element sizes
  const dx = width / nx;
  const dy = height / ny;

  // 3. Generate grid nodes
  const gridNodesX = nx + 1;
  const gridNodesY = ny + 1;
  const totalGridNodes = gridNodesX * gridNodesY;

  // Temporary arrays for node classification
  const gridNodeCoords = new Float32Array(totalGridNodes * 2);
  const nodeActive = new Uint8Array(totalGridNodes); // 1 = inside/boundary, 0 = outside
  const nodeBoundary = new Uint8Array(totalGridNodes); // 1 = on boundary

  // Tolerance for boundary detection
  const boundaryTol = Math.min(dx, dy) * 0.01;

  // Generate node coordinates and classify
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) {
      const idx = j * gridNodesX + i;
      const x = minX + i * dx;
      const y = minY + j * dy;

      gridNodeCoords[idx * 2] = x;
      gridNodeCoords[idx * 2 + 1] = y;

      // Check if node is inside or on boundary of outer polygon
      const onBoundary = pointOnPolygonBoundary(
        x,
        y,
        geometry.boundary,
        boundaryTol
      );
      const inside = pointInPolygon(x, y, geometry.boundary);

      if (onBoundary || inside) {
        // Check if inside any hole
        let inHole = false;
        for (const hole of geometry.holes) {
          if (pointInPolygon(x, y, hole)) {
            inHole = true;
            break;
          }
        }

        if (!inHole) {
          nodeActive[idx] = 1;
          if (onBoundary) {
            nodeBoundary[idx] = 1;
          }
        }
      }
    }
  }

  // 4. Build compacted node list and element connectivity
  // Map from grid index to compacted index
  const gridToCompact = new Int32Array(totalGridNodes).fill(-1);
  let compactCount = 0;

  // First pass: count active nodes and build mapping
  for (let idx = 0; idx < totalGridNodes; idx++) {
    if (nodeActive[idx]) {
      gridToCompact[idx] = compactCount++;
    }
  }

  // Build compacted node array
  const nodes = new Float32Array(compactCount * 2);
  const boundaryNodesList: number[] = [];

  for (let idx = 0; idx < totalGridNodes; idx++) {
    if (nodeActive[idx]) {
      const compactIdx = gridToCompact[idx];
      nodes[compactIdx * 2] = gridNodeCoords[idx * 2];
      nodes[compactIdx * 2 + 1] = gridNodeCoords[idx * 2 + 1];

      if (nodeBoundary[idx]) {
        boundaryNodesList.push(compactIdx);
      }
    }
  }

  // 5. Generate elements
  // For Q4 elements: node ordering is counter-clockwise
  //   3 --- 2
  //   |     |
  //   0 --- 1
  const elementList: number[] = [];

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      // Grid indices of the 4 corners
      const n0_grid = j * gridNodesX + i; // bottom-left
      const n1_grid = j * gridNodesX + (i + 1); // bottom-right
      const n2_grid = (j + 1) * gridNodesX + (i + 1); // top-right
      const n3_grid = (j + 1) * gridNodesX + i; // top-left

      // Check if all 4 corners are active
      if (
        nodeActive[n0_grid] &&
        nodeActive[n1_grid] &&
        nodeActive[n2_grid] &&
        nodeActive[n3_grid]
      ) {
        // Add element with compacted node indices
        elementList.push(
          gridToCompact[n0_grid],
          gridToCompact[n1_grid],
          gridToCompact[n2_grid],
          gridToCompact[n3_grid]
        );
      }
    }
  }

  return {
    nodes,
    elements: new Uint32Array(elementList),
    boundaryNodes: new Uint32Array(boundaryNodesList),
    nodeCount: compactCount,
    elementCount: elementList.length / 4,
    nodesPerElement: 4, // Quads
    gridNx: nx,
    gridNy: ny,
  };
}

/**
 * Generate a simple rectangular mesh (no holes, full interior).
 * Optimized path for simple rectangular plates.
 *
 * @param width - Plate width (m)
 * @param height - Plate height (m)
 * @param targetSize - Target element edge length (m)
 * @returns Mesh data structure
 */
export function generateRectangularMesh(
  width: number,
  height: number,
  targetSize: number = 0.5
): PlateMesh {
  // Grid dimensions
  const nx = Math.max(2, Math.ceil(width / targetSize));
  const ny = Math.max(2, Math.ceil(height / targetSize));

  const dx = width / nx;
  const dy = height / ny;

  const nodesX = nx + 1;
  const nodesY = ny + 1;
  const nodeCount = nodesX * nodesY;
  const elementCount = nx * ny;

  // Generate nodes
  const nodes = new Float32Array(nodeCount * 2);
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) {
      const idx = j * nodesX + i;
      nodes[idx * 2] = i * dx;
      nodes[idx * 2 + 1] = j * dy;
    }
  }

  // Generate elements
  const elements = new Uint32Array(elementCount * 4);
  let elemIdx = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const n0 = j * nodesX + i;
      const n1 = j * nodesX + (i + 1);
      const n2 = (j + 1) * nodesX + (i + 1);
      const n3 = (j + 1) * nodesX + i;

      elements[elemIdx * 4] = n0;
      elements[elemIdx * 4 + 1] = n1;
      elements[elemIdx * 4 + 2] = n2;
      elements[elemIdx * 4 + 3] = n3;
      elemIdx++;
    }
  }

  // Boundary nodes: edges of rectangle
  const boundaryNodesList: number[] = [];

  // Bottom edge (j=0)
  for (let i = 0; i <= nx; i++) {
    boundaryNodesList.push(i);
  }
  // Right edge (i=nx), excluding corners already added
  for (let j = 1; j <= ny; j++) {
    boundaryNodesList.push(j * nodesX + nx);
  }
  // Top edge (j=ny), excluding corners already added
  for (let i = nx - 1; i >= 0; i--) {
    boundaryNodesList.push(ny * nodesX + i);
  }
  // Left edge (i=0), excluding corners already added
  for (let j = ny - 1; j >= 1; j--) {
    boundaryNodesList.push(j * nodesX);
  }

  return {
    nodes,
    elements,
    boundaryNodes: new Uint32Array(boundaryNodesList),
    nodeCount,
    elementCount,
    nodesPerElement: 4, // Quads
    gridNx: nx,
    gridNy: ny,
  };
}

/**
 * Get node coordinates for an element.
 *
 * @param mesh - Plate mesh
 * @param elementIndex - Element index
 * @returns Float32Array of coordinates (6 values for triangles, 8 for quads)
 */
export function getElementCoords(
  mesh: PlateMesh,
  elementIndex: number
): Float32Array {
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const coords = new Float32Array(nodesPerElem * 2);
  const base = elementIndex * nodesPerElem;

  for (let i = 0; i < nodesPerElem; i++) {
    const nodeIdx = mesh.elements[base + i];
    coords[i * 2] = mesh.nodes[nodeIdx * 2];
    coords[i * 2 + 1] = mesh.nodes[nodeIdx * 2 + 1];
  }

  return coords;
}

/**
 * Get element node indices for a Q4 (quad) element.
 *
 * @param mesh - Plate mesh
 * @param elementIndex - Element index
 * @returns 4-element array of node indices
 */
export function getElementNodes(
  mesh: PlateMesh,
  elementIndex: number
): [number, number, number, number] {
  const base = elementIndex * 4;
  return [
    mesh.elements[base],
    mesh.elements[base + 1],
    mesh.elements[base + 2],
    mesh.elements[base + 3],
  ];
}

/**
 * Get element node indices (generic version for triangles or quads).
 *
 * @param mesh - Plate mesh
 * @param elementIndex - Element index
 * @returns Array of node indices (3 for triangles, 4 for quads)
 */
export function getElementNodeIndices(
  mesh: PlateMesh,
  elementIndex: number
): number[] {
  const nodesPerElem = mesh.nodesPerElement ?? 4;
  const base = elementIndex * nodesPerElem;
  const nodes: number[] = [];
  
  for (let i = 0; i < nodesPerElem; i++) {
    nodes.push(mesh.elements[base + i]);
  }
  
  return nodes;
}

