/**
 * Polygon processing utilities for mesh generation.
 *
 * Provides functions for:
 * - Winding order detection and correction
 * - Bounding box computation
 * - Circle polygon generation
 * - Point-in-polygon testing
 */

/**
 * Ensure polygon vertices are in correct winding order.
 *
 * Uses the shoelace formula to compute signed area:
 * - Positive area = Counter-clockwise (CCW)
 * - Negative area = Clockwise (CW)
 *
 * @param vertices - Flat array [x0,y0, x1,y1, ...]
 * @param clockwise - If true, ensure CW; if false, ensure CCW
 * @returns Vertices in correct order (may be reversed)
 */
export function ensureWindingOrder(
  vertices: Float32Array,
  clockwise: boolean
): Float32Array {
  // Compute signed area using shoelace formula
  let area = 0;
  const n = vertices.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += vertices[i * 2] * vertices[j * 2 + 1];
    area -= vertices[j * 2] * vertices[i * 2 + 1];
  }

  const isCCW = area > 0;
  const needsReverse = clockwise ? isCCW : !isCCW;

  if (needsReverse) {
    const reversed = new Float32Array(vertices.length);
    for (let i = 0; i < n; i++) {
      reversed[i * 2] = vertices[(n - 1 - i) * 2];
      reversed[i * 2 + 1] = vertices[(n - 1 - i) * 2 + 1];
    }
    return reversed;
  }

  return vertices;
}

/**
 * Check if polygon vertices are in counter-clockwise order.
 *
 * @param vertices - Flat array [x0,y0, x1,y1, ...]
 * @returns True if CCW, false if CW
 */
export function isCounterClockwise(vertices: Float32Array): boolean {
  let area = 0;
  const n = vertices.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += vertices[i * 2] * vertices[j * 2 + 1];
    area -= vertices[j * 2] * vertices[i * 2 + 1];
  }
  return area > 0;
}

/**
 * Compute polygon centroid.
 *
 * @param vertices - Flat array [x0,y0, x1,y1, ...]
 * @returns Centroid [x, y]
 */
export function computeCentroid(vertices: Float32Array): [number, number] {
  let cx = 0;
  let cy = 0;
  const n = vertices.length / 2;
  for (let i = 0; i < n; i++) {
    cx += vertices[i * 2];
    cy += vertices[i * 2 + 1];
  }
  return [cx / n, cy / n];
}

/**
 * Compute polygon bounding box.
 *
 * @param vertices - Flat array [x0,y0, x1,y1, ...]
 * @returns Bounding box { minX, minY, maxX, maxY }
 */
export function computeBoundingBox(vertices: Float32Array): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < vertices.length; i += 2) {
    minX = Math.min(minX, vertices[i]);
    minY = Math.min(minY, vertices[i + 1]);
    maxX = Math.max(maxX, vertices[i]);
    maxY = Math.max(maxY, vertices[i + 1]);
  }

  return { minX, minY, maxX, maxY };
}

/**
 * Generate circular polygon approximation.
 *
 * Generates vertices in counter-clockwise order.
 *
 * @param cx - Center X
 * @param cy - Center Y
 * @param radius - Circle radius
 * @param segments - Number of segments (default: 64)
 * @returns Flat vertex array [x0,y0, x1,y1, ...]
 */
export function generateCirclePolygon(
  cx: number,
  cy: number,
  radius: number,
  segments: number = 64
): Float32Array {
  const vertices = new Float32Array(segments * 2);
  for (let i = 0; i < segments; i++) {
    const theta = (2 * Math.PI * i) / segments;
    vertices[i * 2] = cx + radius * Math.cos(theta);
    vertices[i * 2 + 1] = cy + radius * Math.sin(theta);
  }
  return vertices;
}

/**
 * Point-in-polygon test using ray casting algorithm.
 *
 * @param x - Point X coordinate
 * @param y - Point Y coordinate
 * @param polygon - Flat polygon array [x0,y0, x1,y1, ...]
 * @returns True if point is inside the polygon
 */
export function isInsidePolygon(
  x: number,
  y: number,
  polygon: Float32Array
): boolean {
  let inside = false;
  const n = polygon.length / 2;

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i * 2];
    const yi = polygon[i * 2 + 1];
    const xj = polygon[j * 2];
    const yj = polygon[j * 2 + 1];

    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * Check if point lies on a line segment (within tolerance).
 *
 * @param px - Point X
 * @param py - Point Y
 * @param x1 - Segment start X
 * @param y1 - Segment start Y
 * @param x2 - Segment end X
 * @param y2 - Segment end Y
 * @param tolerance - Distance tolerance
 * @returns True if point is on segment
 */
export function isPointOnSegment(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  tolerance: number
): boolean {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);

  if (len < tolerance) return false;

  // Project point onto line
  const t = ((px - x1) * dx + (py - y1) * dy) / (len * len);

  if (t < -tolerance / len || t > 1 + tolerance / len) return false;

  // Distance from point to line
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  const dist = Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);

  return dist < tolerance;
}

/**
 * Compute polygon area using shoelace formula.
 *
 * @param vertices - Flat array [x0,y0, x1,y1, ...]
 * @returns Absolute area
 */
export function computePolygonArea(vertices: Float32Array): number {
  let area = 0;
  const n = vertices.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += vertices[i * 2] * vertices[j * 2 + 1];
    area -= vertices[j * 2] * vertices[i * 2 + 1];
  }
  return Math.abs(area) / 2;
}

/**
 * Compute triangle area from three points.
 *
 * @param x1 - First point X
 * @param y1 - First point Y
 * @param x2 - Second point X
 * @param y2 - Second point Y
 * @param x3 - Third point X
 * @param y3 - Third point Y
 * @returns Absolute area
 */
export function computeTriangleArea(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  x3: number,
  y3: number
): number {
  return Math.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2;
}

