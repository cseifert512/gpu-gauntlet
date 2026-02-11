/**
 * GPU compute shader for DKT (triangle) element moment computation.
 *
 * Uses linear shape function derivatives (constant per element) to compute
 * curvatures and moments at element centroid, then scatter to nodes.
 */

export const computeMomentsDKTSource = /* wgsl */ `
// compute_moments_dkt.wgsl
// Compute Mx, My, Mxy for DKT triangle elements.

struct Material {
    E: f32,
    nu: f32,
    t: f32,
    D: f32,
    kappaGt: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Params {
    color_offset: u32,
    color_count: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<f32>;
@group(0) @binding(1) var<storage, read> elements: array<u32>;
@group(0) @binding(2) var<storage, read> element_indices: array<u32>;
@group(0) @binding(3) var<uniform> material: Material;
@group(0) @binding(4) var<storage, read> displacements: array<f32>;
@group(0) @binding(5) var<storage, read_write> Mx_sum: array<f32>;
@group(0) @binding(6) var<storage, read_write> My_sum: array<f32>;
@group(0) @binding(7) var<storage, read_write> Mxy_sum: array<f32>;
@group(0) @binding(8) var<storage, read_write> count: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }

    let elem_idx = element_indices[params.color_offset + thread_idx];

    // 3 nodes per DKT triangle
    let ebase = elem_idx * 3u;
    let n0 = elements[ebase + 0u];
    let n1 = elements[ebase + 1u];
    let n2 = elements[ebase + 2u];

    // Node coordinates
    let x1 = nodes[n0 * 2u]; let y1 = nodes[n0 * 2u + 1u];
    let x2 = nodes[n1 * 2u]; let y2 = nodes[n1 * 2u + 1u];
    let x3 = nodes[n2 * 2u]; let y3 = nodes[n2 * 2u + 1u];

    // Triangle area (2A)
    let twoA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    if (abs(twoA) < 1e-15) {
        return; // Degenerate triangle
    }
    let invTwoA = 1.0 / twoA;

    // Rotations
    let tx0 = displacements[n0 * 3u + 1u]; let ty0 = displacements[n0 * 3u + 2u];
    let tx1 = displacements[n1 * 3u + 1u]; let ty1 = displacements[n1 * 3u + 2u];
    let tx2 = displacements[n2 * 3u + 1u]; let ty2 = displacements[n2 * 3u + 2u];

    // Linear shape function derivatives (constant for triangle)
    let dNdx0 = (y2 - y3) * invTwoA;
    let dNdx1 = (y3 - y1) * invTwoA;
    let dNdx2 = (y1 - y2) * invTwoA;

    let dNdy0 = (x3 - x2) * invTwoA;
    let dNdy1 = (x1 - x3) * invTwoA;
    let dNdy2 = (x2 - x1) * invTwoA;

    // Curvatures
    let kappa_x  = dNdx0 * ty0 + dNdx1 * ty1 + dNdx2 * ty2;
    let kappa_y  = -(dNdy0 * tx0 + dNdy1 * tx1 + dNdy2 * tx2);
    let kappa_xy = (dNdy0 * ty0 + dNdy1 * ty1 + dNdy2 * ty2)
                 - (dNdx0 * tx0 + dNdx1 * tx1 + dNdx2 * tx2);

    let D = material.D;
    let nu = material.nu;
    let Mx_e  = D * (kappa_x + nu * kappa_y);
    let My_e  = D * (kappa_y + nu * kappa_x);
    let Mxy_e = D * (1.0 - nu) * kappa_xy / 2.0;

    // Scatter to nodes
    Mx_sum[n0]  += Mx_e;  Mx_sum[n1]  += Mx_e;  Mx_sum[n2]  += Mx_e;
    My_sum[n0]  += My_e;  My_sum[n1]  += My_e;  My_sum[n2]  += My_e;
    Mxy_sum[n0] += Mxy_e; Mxy_sum[n1] += Mxy_e; Mxy_sum[n2] += Mxy_e;
    count[n0]   += 1.0;   count[n1]   += 1.0;   count[n2]   += 1.0;
}
`;

