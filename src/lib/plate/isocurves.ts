/**
 * Isocurve (contour line) generation for FEM result fields.
 *
 * Generates contour polylines at specified iso-values for any nodal scalar
 * field (deflection w, moments Mx/My/Mxy, etc.) over the mesh. Works with
 * both structured quad and unstructured triangle meshes.
 *
 * Algorithm:
 *   For each element, check which edges are crossed by the iso-value
 *   (one vertex above, one below). Linearly interpolate the crossing point
 *   along each crossed edge. Connect crossing points within each element
 *   to form contour segments. Optionally chain segments into polylines.
 *
 * Performance target: <2ms for 100k-element mesh with 20 contour levels.
 */

import type { PlateMesh } from './types';

/** A single contour line segment (two endpoints). */
export interface ContourSegment {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** A contour level with its value and all segments/polylines. */
export interface ContourLevel {
  value: number;
  segments: ContourSegment[];
  /** Chained polylines: each is a flat array [x0,y0, x1,y1, ...] */
  polylines: Float32Array[];
}

/** Options for isocurve generation. */
export interface IsocurveOptions {
  /** Number of evenly-spaced contour levels (default: 20) */
  levels?: number;
  /** Explicit contour values (overrides levels) */
  values?: number[];
  /** Minimum field value (auto-computed if not given) */
  min?: number;
  /** Maximum field value (auto-computed if not given) */
  max?: number;
  /** Chain segments into polylines (default: true) */
  chain?: boolean;
}

/**
 * Generate isocurves for a scalar nodal field over the mesh.
 *
 * @param mesh - Plate mesh (quads or triangles)
 * @param field - Scalar values at each node (length = mesh.nodeCount)
 * @param options - Generation options
 * @returns Array of contour levels with segments and polylines
 */
export function generateIsocurves(
  mesh: PlateMesh,
  field: Float32Array,
  options: IsocurveOptions = {}
): ContourLevel[] {
  const nodesPerElem = mesh.nodesPerElement ?? 4;

  // Determine contour values
  let values: number[];
  if (options.values) {
    values = options.values;
  } else {
    let min = options.min ?? Infinity;
    let max = options.max ?? -Infinity;
    if (min === Infinity || max === -Infinity) {
      for (let i = 0; i < field.length; i++) {
        if (field[i] < min) min = field[i];
        if (field[i] > max) max = field[i];
      }
    }
    const numLevels = options.levels ?? 20;
    values = [];
    for (let i = 1; i < numLevels + 1; i++) {
      values.push(min + (max - min) * i / (numLevels + 1));
    }
  }

  const results: ContourLevel[] = [];
  const shouldChain = options.chain !== false;

  for (const value of values) {
    const segments: ContourSegment[] = [];

    if (nodesPerElem === 3) {
      generateContourSegmentsTriangle(mesh, field, value, segments);
    } else {
      generateContourSegmentsQuad(mesh, field, value, segments);
    }

    const polylines = shouldChain ? chainSegments(segments) : [];

    results.push({ value, segments, polylines });
  }

  return results;
}

/**
 * Generate contour segments for triangle elements.
 *
 * For each triangle, the contour line crosses edges where the field value
 * is on opposite sides of the iso-value. A triangle can have 0 or 2 crossings.
 */
function generateContourSegmentsTriangle(
  mesh: PlateMesh,
  field: Float32Array,
  isoValue: number,
  segments: ContourSegment[]
): void {
  const { elementCount, elements, nodes } = mesh;

  for (let e = 0; e < elementCount; e++) {
    const base = e * 3;
    const n0 = elements[base];
    const n1 = elements[base + 1];
    const n2 = elements[base + 2];

    const v0 = field[n0] - isoValue;
    const v1 = field[n1] - isoValue;
    const v2 = field[n2] - isoValue;

    // Find edge crossings (where sign changes)
    const crossings: { x: number; y: number }[] = [];

    // Edge 0→1
    if ((v0 > 0) !== (v1 > 0) && v0 !== 0 && v1 !== 0) {
      const t = v0 / (v0 - v1);
      crossings.push({
        x: nodes[n0 * 2] + t * (nodes[n1 * 2] - nodes[n0 * 2]),
        y: nodes[n0 * 2 + 1] + t * (nodes[n1 * 2 + 1] - nodes[n0 * 2 + 1]),
      });
    }

    // Edge 1→2
    if ((v1 > 0) !== (v2 > 0) && v1 !== 0 && v2 !== 0) {
      const t = v1 / (v1 - v2);
      crossings.push({
        x: nodes[n1 * 2] + t * (nodes[n2 * 2] - nodes[n1 * 2]),
        y: nodes[n1 * 2 + 1] + t * (nodes[n2 * 2 + 1] - nodes[n1 * 2 + 1]),
      });
    }

    // Edge 2→0
    if ((v2 > 0) !== (v0 > 0) && v2 !== 0 && v0 !== 0) {
      const t = v2 / (v2 - v0);
      crossings.push({
        x: nodes[n2 * 2] + t * (nodes[n0 * 2] - nodes[n2 * 2]),
        y: nodes[n2 * 2 + 1] + t * (nodes[n0 * 2 + 1] - nodes[n2 * 2 + 1]),
      });
    }

    if (crossings.length >= 2) {
      segments.push({
        x0: crossings[0].x, y0: crossings[0].y,
        x1: crossings[1].x, y1: crossings[1].y,
      });
    }
  }
}

/**
 * Generate contour segments for quad elements using marching squares.
 *
 * A quad is split into 4 sub-triangles via the centroid for robust handling
 * of saddle points. This avoids the ambiguity cases of standard marching squares.
 */
function generateContourSegmentsQuad(
  mesh: PlateMesh,
  field: Float32Array,
  isoValue: number,
  segments: ContourSegment[]
): void {
  const { elementCount, elements, nodes } = mesh;

  for (let e = 0; e < elementCount; e++) {
    const base = e * 4;
    const n0 = elements[base];
    const n1 = elements[base + 1];
    const n2 = elements[base + 2];
    const n3 = elements[base + 3];

    const x0 = nodes[n0 * 2], y0 = nodes[n0 * 2 + 1];
    const x1 = nodes[n1 * 2], y1 = nodes[n1 * 2 + 1];
    const x2 = nodes[n2 * 2], y2 = nodes[n2 * 2 + 1];
    const x3 = nodes[n3 * 2], y3 = nodes[n3 * 2 + 1];

    const v0 = field[n0] - isoValue;
    const v1 = field[n1] - isoValue;
    const v2 = field[n2] - isoValue;
    const v3 = field[n3] - isoValue;

    // Centroid for sub-triangle decomposition
    const cx = (x0 + x1 + x2 + x3) * 0.25;
    const cy = (y0 + y1 + y2 + y3) * 0.25;
    const vc = (v0 + v1 + v2 + v3) * 0.25;

    // Process 4 sub-triangles: (n0,n1,c), (n1,n2,c), (n2,n3,c), (n3,n0,c)
    processSubTriangle(x0, y0, v0, x1, y1, v1, cx, cy, vc, segments);
    processSubTriangle(x1, y1, v1, x2, y2, v2, cx, cy, vc, segments);
    processSubTriangle(x2, y2, v2, x3, y3, v3, cx, cy, vc, segments);
    processSubTriangle(x3, y3, v3, x0, y0, v0, cx, cy, vc, segments);
  }
}

/** Process a single sub-triangle for contour crossings. */
function processSubTriangle(
  x0: number, y0: number, v0: number,
  x1: number, y1: number, v1: number,
  x2: number, y2: number, v2: number,
  segments: ContourSegment[]
): void {
  const crossings: { x: number; y: number }[] = [];

  if ((v0 > 0) !== (v1 > 0) && v0 !== 0 && v1 !== 0) {
    const t = v0 / (v0 - v1);
    crossings.push({ x: x0 + t * (x1 - x0), y: y0 + t * (y1 - y0) });
  }
  if ((v1 > 0) !== (v2 > 0) && v1 !== 0 && v2 !== 0) {
    const t = v1 / (v1 - v2);
    crossings.push({ x: x1 + t * (x2 - x1), y: y1 + t * (y2 - y1) });
  }
  if ((v2 > 0) !== (v0 > 0) && v2 !== 0 && v0 !== 0) {
    const t = v2 / (v2 - v0);
    crossings.push({ x: x2 + t * (x0 - x2), y: y2 + t * (y0 - y2) });
  }

  if (crossings.length >= 2) {
    segments.push({
      x0: crossings[0].x, y0: crossings[0].y,
      x1: crossings[1].x, y1: crossings[1].y,
    });
  }
}

/**
 * Chain disconnected contour segments into continuous polylines.
 *
 * Uses a spatial hash to efficiently find segment endpoints that connect.
 * Produces polylines as flat Float32Arrays [x0,y0, x1,y1, ...].
 */
function chainSegments(segments: ContourSegment[]): Float32Array[] {
  if (segments.length === 0) return [];

  // Spatial hash for endpoint matching
  const TOLERANCE = 1e-8;
  const used = new Uint8Array(segments.length);
  const polylines: Float32Array[] = [];

  // Build endpoint index: key → list of { segIdx, endIdx (0 or 1) }
  const endpointMap = new Map<string, { segIdx: number; endIdx: number }[]>();

  const makeKey = (x: number, y: number): string => {
    // Round to tolerance grid to handle floating point
    const rx = Math.round(x / TOLERANCE) * TOLERANCE;
    const ry = Math.round(y / TOLERANCE) * TOLERANCE;
    return `${rx.toFixed(10)},${ry.toFixed(10)}`;
  };

  for (let i = 0; i < segments.length; i++) {
    const s = segments[i];
    const k0 = makeKey(s.x0, s.y0);
    const k1 = makeKey(s.x1, s.y1);

    if (!endpointMap.has(k0)) endpointMap.set(k0, []);
    endpointMap.get(k0)!.push({ segIdx: i, endIdx: 0 });

    if (!endpointMap.has(k1)) endpointMap.set(k1, []);
    endpointMap.get(k1)!.push({ segIdx: i, endIdx: 1 });
  }

  // Build polylines by following chains
  for (let startIdx = 0; startIdx < segments.length; startIdx++) {
    if (used[startIdx]) continue;

    // Start a new polyline from this segment
    used[startIdx] = 1;
    const chain: number[] = [];
    const s = segments[startIdx];
    chain.push(s.x0, s.y0, s.x1, s.y1);

    // Grow forward from end (x1, y1)
    let currentKey = makeKey(s.x1, s.y1);
    let growing = true;
    while (growing) {
      growing = false;
      const candidates = endpointMap.get(currentKey);
      if (!candidates) break;

      for (const cand of candidates) {
        if (used[cand.segIdx]) continue;
        used[cand.segIdx] = 1;
        const cs = segments[cand.segIdx];

        if (cand.endIdx === 0) {
          // This segment's start matches our chain end → append its end
          chain.push(cs.x1, cs.y1);
          currentKey = makeKey(cs.x1, cs.y1);
        } else {
          // This segment's end matches our chain end → append its start
          chain.push(cs.x0, cs.y0);
          currentKey = makeKey(cs.x0, cs.y0);
        }
        growing = true;
        break;
      }
    }

    // Grow backward from start (x0, y0)
    currentKey = makeKey(s.x0, s.y0);
    growing = true;
    while (growing) {
      growing = false;
      const candidates = endpointMap.get(currentKey);
      if (!candidates) break;

      for (const cand of candidates) {
        if (used[cand.segIdx]) continue;
        used[cand.segIdx] = 1;
        const cs = segments[cand.segIdx];

        if (cand.endIdx === 0) {
          // Prepend: this segment start matches → prepend its end
          chain.unshift(cs.x1, cs.y1);
          currentKey = makeKey(cs.x1, cs.y1);
        } else {
          // Prepend: this segment end matches → prepend its start
          chain.unshift(cs.x0, cs.y0);
          currentKey = makeKey(cs.x0, cs.y0);
        }
        growing = true;
        break;
      }
    }

    polylines.push(new Float32Array(chain));
  }

  return polylines;
}

/**
 * Generate evenly-spaced iso-values for a scalar field.
 *
 * @param field - Scalar field values
 * @param levels - Number of levels
 * @returns Array of iso-values
 */
export function computeIsoValues(
  field: Float32Array,
  levels: number = 20
): number[] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < field.length; i++) {
    if (field[i] < min) min = field[i];
    if (field[i] > max) max = field[i];
  }

  const values: number[] = [];
  for (let i = 1; i <= levels; i++) {
    values.push(min + (max - min) * i / (levels + 1));
  }
  return values;
}

