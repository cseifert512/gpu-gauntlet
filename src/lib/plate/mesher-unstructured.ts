/**
 * Unstructured triangular mesh generation using Constrained Delaunay Triangulation.
 *
 * Uses poly2tri library for CDT. Handles:
 * - Arbitrary polygonal boundaries
 * - Circular boundaries
 * - Polygons with holes
 * - Mesh refinement via Steiner points
 */

import * as poly2tri from 'poly2tri';
import type { PlateGeometry, PlateMesh } from './types';
import {
  ensureWindingOrder,
  computeBoundingBox,
  isInsidePolygon,
  isPointOnSegment,
} from './mesher-utils';

export interface UnstructuredMeshOptions {
  /** Target edge length for refinement (optional) */
  targetEdgeLength?: number;
  /** Maximum area per triangle (optional, for refinement) */
  maxTriangleArea?: number;
  /** Minimum angle for quality triangles in degrees (default: 20) */
  minAngle?: number;
}

/**
 * Generate unstructured triangular mesh using CDT.
 *
 * @param geometry - Plate geometry with boundary and holes
 * @param options - Mesh generation options
 * @returns Triangular mesh (3 nodes per element instead of 4)
 */
export function generateUnstructuredMesh(
  geometry: PlateGeometry,
  options: UnstructuredMeshOptions = {}
): PlateMesh {
  // 1. Ensure correct winding order
  //    - Outer boundary: CCW
  //    - Holes: CW
  const outerBoundary = ensureWindingOrder(geometry.boundary, false); // CCW
  const holes = geometry.holes.map((h) => ensureWindingOrder(h, true)); // CW

  // 2. Convert to poly2tri format
  const contour = flatToPoints(outerBoundary);

  // 3. Create sweep context
  const swctx = new poly2tri.SweepContext(contour);

  // 4. Add holes
  for (const hole of holes) {
    const holePoints = flatToPoints(hole);
    swctx.addHole(holePoints);
  }

  // 5. Optional: Add Steiner points for refinement
  if (options.targetEdgeLength) {
    const steinerPoints = generateSteinerPoints(
      outerBoundary,
      holes,
      options.targetEdgeLength
    );
    for (const p of steinerPoints) {
      swctx.addPoint(p);
    }
  }

  // 6. Triangulate
  swctx.triangulate();
  const triangles = swctx.getTriangles();

  // 7. Build mesh data structures
  return buildMeshFromTriangles(triangles, outerBoundary, holes);
}

/**
 * Convert flat array to poly2tri Point array.
 */
function flatToPoints(vertices: Float32Array): poly2tri.Point[] {
  const points: poly2tri.Point[] = [];
  for (let i = 0; i < vertices.length; i += 2) {
    points.push(new poly2tri.Point(vertices[i], vertices[i + 1]));
  }
  return points;
}

/**
 * Generate internal Steiner points for mesh refinement.
 *
 * Creates a grid of points inside the boundary but outside holes.
 */
function generateSteinerPoints(
  boundary: Float32Array,
  holes: Float32Array[],
  targetEdgeLength: number
): poly2tri.Point[] {
  const points: poly2tri.Point[] = [];
  const bbox = computeBoundingBox(boundary);

  // Add small margin to avoid placing points exactly on boundary
  const margin = targetEdgeLength * 0.1;

  // Create grid of candidate points
  const nx = Math.ceil((bbox.maxX - bbox.minX) / targetEdgeLength);
  const ny = Math.ceil((bbox.maxY - bbox.minY) / targetEdgeLength);

  // Use offset grid pattern for better triangle quality
  for (let j = 1; j < ny; j++) {
    const yOffset = j % 2 === 0 ? 0 : targetEdgeLength * 0.5;
    for (let i = 1; i < nx; i++) {
      const x = bbox.minX + i * targetEdgeLength + yOffset;
      const y = bbox.minY + j * targetEdgeLength;

      // Skip if point is outside the adjusted domain
      if (
        x < bbox.minX + margin ||
        x > bbox.maxX - margin ||
        y < bbox.minY + margin ||
        y > bbox.maxY - margin
      ) {
        continue;
      }

      // Only add if inside boundary and outside all holes
      if (
        isInsidePolygon(x, y, boundary) &&
        !holes.some((h) => isInsidePolygon(x, y, h))
      ) {
        // Also check we're not too close to boundary edges
        if (!isNearBoundary(x, y, boundary, holes, margin)) {
          points.push(new poly2tri.Point(x, y));
        }
      }
    }
  }

  return points;
}

/**
 * Check if point is near any boundary or hole edge.
 */
function isNearBoundary(
  x: number,
  y: number,
  boundary: Float32Array,
  holes: Float32Array[],
  tolerance: number
): boolean {
  // Check boundary edges
  const n = boundary.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    if (
      isPointOnSegment(
        x,
        y,
        boundary[i * 2],
        boundary[i * 2 + 1],
        boundary[j * 2],
        boundary[j * 2 + 1],
        tolerance
      )
    ) {
      return true;
    }
  }

  // Check hole edges
  for (const hole of holes) {
    const nh = hole.length / 2;
    for (let i = 0; i < nh; i++) {
      const j = (i + 1) % nh;
      if (
        isPointOnSegment(
          x,
          y,
          hole[i * 2],
          hole[i * 2 + 1],
          hole[j * 2],
          hole[j * 2 + 1],
          tolerance
        )
      ) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Build mesh data structures from poly2tri triangles.
 */
function buildMeshFromTriangles(
  triangles: poly2tri.Triangle[],
  boundary: Float32Array,
  holes: Float32Array[]
): PlateMesh {
  // Build unique node list
  const nodeMap = new Map<string, number>();
  const nodeCoords: number[] = [];

  const getNodeIndex = (p: { x: number; y: number }): number => {
    // Use sufficient precision for uniqueness
    const key = `${p.x.toFixed(10)},${p.y.toFixed(10)}`;
    if (nodeMap.has(key)) {
      return nodeMap.get(key)!;
    }
    const idx = nodeCoords.length / 2;
    nodeCoords.push(p.x, p.y);
    nodeMap.set(key, idx);
    return idx;
  };

  // Build element connectivity (triangles have 3 nodes)
  const elements: number[] = [];
  for (const tri of triangles) {
    const p0 = tri.getPoint(0);
    const p1 = tri.getPoint(1);
    const p2 = tri.getPoint(2);

    elements.push(getNodeIndex(p0), getNodeIndex(p1), getNodeIndex(p2));
  }

  // Find boundary nodes (on outer boundary or hole boundaries)
  const boundaryNodes = findBoundaryNodes(
    new Float32Array(nodeCoords),
    boundary,
    holes
  );

  return {
    nodes: new Float32Array(nodeCoords),
    elements: new Uint32Array(elements),
    boundaryNodes: new Uint32Array(boundaryNodes),
    nodeCount: nodeCoords.length / 2,
    elementCount: triangles.length,
    nodesPerElement: 3, // Triangles!
    gridNx: 0, // Not applicable for unstructured mesh
    gridNy: 0,
  };
}

/**
 * Find nodes that lie on any boundary (outer or holes).
 */
function findBoundaryNodes(
  nodes: Float32Array,
  boundary: Float32Array,
  holes: Float32Array[],
  tolerance: number = 1e-6
): number[] {
  const boundaryNodes: number[] = [];
  const nodeCount = nodes.length / 2;

  for (let i = 0; i < nodeCount; i++) {
    const x = nodes[i * 2];
    const y = nodes[i * 2 + 1];

    // Check if on outer boundary
    if (isOnPolygonBoundary(x, y, boundary, tolerance)) {
      boundaryNodes.push(i);
      continue;
    }

    // Check if on any hole boundary
    for (const hole of holes) {
      if (isOnPolygonBoundary(x, y, hole, tolerance)) {
        boundaryNodes.push(i);
        break;
      }
    }
  }

  return boundaryNodes;
}

/**
 * Check if point lies on polygon boundary.
 */
function isOnPolygonBoundary(
  x: number,
  y: number,
  polygon: Float32Array,
  tolerance: number
): boolean {
  const n = polygon.length / 2;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const x1 = polygon[i * 2];
    const y1 = polygon[i * 2 + 1];
    const x2 = polygon[j * 2];
    const y2 = polygon[j * 2 + 1];

    if (isPointOnSegment(x, y, x1, y1, x2, y2, tolerance)) {
      return true;
    }
  }

  return false;
}

/**
 * Validate mesh quality.
 *
 * Checks for:
 * - Degenerate triangles (zero area)
 * - Very thin triangles (bad aspect ratio)
 *
 * @param mesh - Generated mesh
 * @returns Quality statistics
 */
export function validateMeshQuality(mesh: PlateMesh): {
  valid: boolean;
  minArea: number;
  maxArea: number;
  minAngle: number;
  degenerateCount: number;
} {
  let minArea = Infinity;
  let maxArea = 0;
  let minAngle = Infinity;
  let degenerateCount = 0;

  const nodesPerElem = mesh.nodesPerElement ?? 3;

  for (let e = 0; e < mesh.elementCount; e++) {
    const base = e * nodesPerElem;

    if (nodesPerElem === 3) {
      const n0 = mesh.elements[base];
      const n1 = mesh.elements[base + 1];
      const n2 = mesh.elements[base + 2];

      const x0 = mesh.nodes[n0 * 2];
      const y0 = mesh.nodes[n0 * 2 + 1];
      const x1 = mesh.nodes[n1 * 2];
      const y1 = mesh.nodes[n1 * 2 + 1];
      const x2 = mesh.nodes[n2 * 2];
      const y2 = mesh.nodes[n2 * 2 + 1];

      // Compute area
      const area = Math.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2;

      if (area < 1e-12) {
        degenerateCount++;
      }

      minArea = Math.min(minArea, area);
      maxArea = Math.max(maxArea, area);

      // Compute minimum angle
      const angles = computeTriangleAngles(x0, y0, x1, y1, x2, y2);
      minAngle = Math.min(minAngle, ...angles);
    }
  }

  return {
    valid: degenerateCount === 0,
    minArea,
    maxArea,
    minAngle: (minAngle * 180) / Math.PI, // Convert to degrees
    degenerateCount,
  };
}

/**
 * Compute all three angles of a triangle.
 */
function computeTriangleAngles(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): [number, number, number] {
  // Edge vectors
  const v01x = x1 - x0;
  const v01y = y1 - y0;
  const v02x = x2 - x0;
  const v02y = y2 - y0;
  const v12x = x2 - x1;
  const v12y = y2 - y1;

  // Edge lengths
  const l01 = Math.sqrt(v01x * v01x + v01y * v01y);
  const l02 = Math.sqrt(v02x * v02x + v02y * v02y);
  const l12 = Math.sqrt(v12x * v12x + v12y * v12y);

  // Angles using dot product
  // angle at v0: between edges v01 and v02
  const dot0 = v01x * v02x + v01y * v02y;
  const angle0 = Math.acos(Math.max(-1, Math.min(1, dot0 / (l01 * l02))));

  // angle at v1: between edges v10 and v12
  const v10x = -v01x;
  const v10y = -v01y;
  const dot1 = v10x * v12x + v10y * v12y;
  const angle1 = Math.acos(Math.max(-1, Math.min(1, dot1 / (l01 * l12))));

  // angle at v2: remaining
  const angle2 = Math.PI - angle0 - angle1;

  return [angle0, angle1, angle2];
}

