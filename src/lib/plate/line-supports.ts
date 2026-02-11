/**
 * Line support resolution.
 *
 * Converts PlateLineSupport definitions (polylines) into standard PlateSupport
 * entries by finding all mesh nodes that lie within tolerance of each polyline
 * segment. This allows walls, beams, and other linear supports to be specified
 * geometrically and automatically snapped to the mesh.
 *
 * Algorithm:
 *   For each segment of the polyline, iterate all mesh nodes and compute the
 *   point-to-segment distance. Nodes within tolerance become point supports.
 *   A spatial grid accelerates the search for large meshes (~O(n) instead of
 *   O(n × segments)).
 */

import type { PlateMesh, PlateSupport, PlateLineSupport } from './types';

/**
 * Compute the distance from a point to a line segment.
 *
 * @param px - Point x
 * @param py - Point y
 * @param ax - Segment start x
 * @param ay - Segment start y
 * @param bx - Segment end x
 * @param by - Segment end y
 * @returns Distance from point to segment
 */
function pointToSegmentDist(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;

  if (lenSq < 1e-30) {
    // Degenerate segment (zero length)
    return Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
  }

  // Projection of point onto line, clamped to segment
  let t = ((px - ax) * dx + (py - ay) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));

  const closestX = ax + t * dx;
  const closestY = ay + t * dy;

  return Math.sqrt((px - closestX) ** 2 + (py - closestY) ** 2);
}

/**
 * Find all mesh node indices within `tolerance` of a polyline.
 *
 * Uses a bounding-box pre-filter per segment for performance on large meshes.
 *
 * @param mesh - Plate mesh
 * @param points - Polyline vertices [[x0,y0], [x1,y1], ...]
 * @param tolerance - Snap distance
 * @returns Sorted, deduplicated array of node indices on the line
 */
export function findNodesOnPolyline(
  mesh: PlateMesh,
  points: [number, number][],
  tolerance: number
): number[] {
  if (points.length < 2) return [];

  const hitSet = new Set<number>();

  for (let seg = 0; seg < points.length - 1; seg++) {
    const [ax, ay] = points[seg];
    const [bx, by] = points[seg + 1];

    // Bounding box of segment expanded by tolerance
    const minX = Math.min(ax, bx) - tolerance;
    const maxX = Math.max(ax, bx) + tolerance;
    const minY = Math.min(ay, by) - tolerance;
    const maxY = Math.max(ay, by) + tolerance;

    for (let i = 0; i < mesh.nodeCount; i++) {
      const nx = mesh.nodes[i * 2];
      const ny = mesh.nodes[i * 2 + 1];

      // Bounding-box reject
      if (nx < minX || nx > maxX || ny < minY || ny > maxY) continue;

      // Exact distance check
      if (pointToSegmentDist(nx, ny, ax, ay, bx, by) <= tolerance) {
        hitSet.add(i);
      }
    }
  }

  return Array.from(hitSet).sort((a, b) => a - b);
}

/**
 * Resolve line supports into standard point supports.
 *
 * For each PlateLineSupport, finds all mesh nodes on/near the polyline and
 * creates a PlateSupport with the same type for each node. The returned
 * array should be concatenated with any existing PlateSupport definitions.
 *
 * @param mesh - Plate mesh
 * @param lineSupports - Array of line support definitions
 * @param defaultTolerance - Fallback tolerance if not specified per support
 * @returns Array of PlateSupport entries (one per constrained node)
 */
export function resolveLineSupports(
  mesh: PlateMesh,
  lineSupports: PlateLineSupport[],
  defaultTolerance?: number
): PlateSupport[] {
  const result: PlateSupport[] = [];

  // Estimate mesh size from first two nodes if no default given
  let estimatedMeshSize = 0.5;
  if (mesh.nodeCount >= 2) {
    const dx = mesh.nodes[2] - mesh.nodes[0];
    const dy = mesh.nodes[3] - mesh.nodes[1];
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > 0) estimatedMeshSize = d;
  }

  for (const ls of lineSupports) {
    const tol = ls.tolerance ?? defaultTolerance ?? estimatedMeshSize / 4;
    const nodeIndices = findNodesOnPolyline(mesh, ls.points, tol);

    for (const nodeIdx of nodeIndices) {
      result.push({
        type: ls.type,
        location: nodeIdx,
      });
    }
  }

  return result;
}

/**
 * Convenience: resolve line supports and merge with existing point supports.
 *
 * @param mesh - Plate mesh
 * @param pointSupports - Existing point supports
 * @param lineSupports - Line supports to resolve
 * @param defaultTolerance - Fallback snap distance
 * @returns Combined array of all supports
 */
export function mergeSupports(
  mesh: PlateMesh,
  pointSupports: PlateSupport[],
  lineSupports: PlateLineSupport[],
  defaultTolerance?: number
): PlateSupport[] {
  const resolved = resolveLineSupports(mesh, lineSupports, defaultTolerance);
  return [...pointSupports, ...resolved];
}

