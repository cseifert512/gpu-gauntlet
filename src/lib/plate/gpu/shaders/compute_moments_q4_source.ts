/**
 * GPU compute shader for Q4 element moment computation.
 *
 * Computes Mx, My, Mxy at element centroids from the displacement solution,
 * then scatter-adds contributions to per-node accumulators.
 *
 * Two-pass approach:
 *   Pass 1 (this shader): Accumulate weighted moment contributions per node
 *   Pass 2 (average shader): Divide sums by contribution count
 *
 * This runs INSIDE the same command encoder as the PCG solve, so moments
 * are available in the same GPU→CPU readback (zero extra latency).
 */

export const computeMomentsQ4Source = /* wgsl */ `
// compute_moments_q4.wgsl
// Compute Mx, My, Mxy for Q4 elements at centroid, scatter to nodes.

struct Material {
    E: f32,
    nu: f32,
    t: f32,
    D: f32,       // Flexural rigidity E*t^3/(12*(1-nu^2))
    kappaGt: f32, // Shear correction
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
@group(0) @binding(4) var<storage, read> displacements: array<f32>;  // Solution x
@group(0) @binding(5) var<storage, read_write> Mx_sum: array<f32>;   // Moment accumulators
@group(0) @binding(6) var<storage, read_write> My_sum: array<f32>;
@group(0) @binding(7) var<storage, read_write> Mxy_sum: array<f32>;
@group(0) @binding(8) var<storage, read_write> count: array<f32>;     // Contribution count per node
@group(0) @binding(9) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }

    let elem_idx = element_indices[params.color_offset + thread_idx];

    // Read 4 node indices
    let ebase = elem_idx * 4u;
    let n0 = elements[ebase + 0u];
    let n1 = elements[ebase + 1u];
    let n2 = elements[ebase + 2u];
    let n3 = elements[ebase + 3u];

    // Read node coordinates
    let x0 = nodes[n0 * 2u]; let y0 = nodes[n0 * 2u + 1u];
    let x1 = nodes[n1 * 2u]; let y1 = nodes[n1 * 2u + 1u];
    let x2 = nodes[n2 * 2u]; let y2 = nodes[n2 * 2u + 1u];
    let x3 = nodes[n3 * 2u]; let y3 = nodes[n3 * 2u + 1u];

    // Read rotations (theta_x, theta_y) for each node from displacement vector
    // DOF layout: [w, theta_x, theta_y] per node
    let tx0 = displacements[n0 * 3u + 1u]; let ty0 = displacements[n0 * 3u + 2u];
    let tx1 = displacements[n1 * 3u + 1u]; let ty1 = displacements[n1 * 3u + 2u];
    let tx2 = displacements[n2 * 3u + 1u]; let ty2 = displacements[n2 * 3u + 2u];
    let tx3 = displacements[n3 * 3u + 1u]; let ty3 = displacements[n3 * 3u + 2u];

    // Shape function derivatives at centroid (xi=0, eta=0)
    // N = 0.25 * [(1-xi)(1-eta), (1+xi)(1-eta), (1+xi)(1+eta), (1-xi)(1+eta)]
    // dN/dxi at (0,0):  [-0.25, 0.25, 0.25, -0.25]
    // dN/deta at (0,0): [-0.25, -0.25, 0.25, 0.25]
    let dNdXi0 = -0.25; let dNdXi1 = 0.25; let dNdXi2 = 0.25; let dNdXi3 = -0.25;
    let dNdEta0 = -0.25; let dNdEta1 = -0.25; let dNdEta2 = 0.25; let dNdEta3 = 0.25;

    // Jacobian
    let J11 = dNdXi0 * x0 + dNdXi1 * x1 + dNdXi2 * x2 + dNdXi3 * x3;
    let J12 = dNdXi0 * y0 + dNdXi1 * y1 + dNdXi2 * y2 + dNdXi3 * y3;
    let J21 = dNdEta0 * x0 + dNdEta1 * x1 + dNdEta2 * x2 + dNdEta3 * x3;
    let J22 = dNdEta0 * y0 + dNdEta1 * y1 + dNdEta2 * y2 + dNdEta3 * y3;

    let detJ = J11 * J22 - J12 * J21;
    if (abs(detJ) < 1e-15) {
        return; // Degenerate element
    }
    let invDetJ = 1.0 / detJ;

    // Physical derivatives: dN/dx = invJ * dN/dxi
    let dNdx0 = invDetJ * (J22 * dNdXi0 - J12 * dNdEta0);
    let dNdx1 = invDetJ * (J22 * dNdXi1 - J12 * dNdEta1);
    let dNdx2 = invDetJ * (J22 * dNdXi2 - J12 * dNdEta2);
    let dNdx3 = invDetJ * (J22 * dNdXi3 - J12 * dNdEta3);

    let dNdy0 = invDetJ * (-J21 * dNdXi0 + J11 * dNdEta0);
    let dNdy1 = invDetJ * (-J21 * dNdXi1 + J11 * dNdEta1);
    let dNdy2 = invDetJ * (-J21 * dNdXi2 + J11 * dNdEta2);
    let dNdy3 = invDetJ * (-J21 * dNdXi3 + J11 * dNdEta3);

    // Curvatures
    let kappa_x  = dNdx0 * ty0 + dNdx1 * ty1 + dNdx2 * ty2 + dNdx3 * ty3;
    let kappa_y  = -(dNdy0 * tx0 + dNdy1 * tx1 + dNdy2 * tx2 + dNdy3 * tx3);
    let kappa_xy = (dNdy0 * ty0 + dNdy1 * ty1 + dNdy2 * ty2 + dNdy3 * ty3)
                 - (dNdx0 * tx0 + dNdx1 * tx1 + dNdx2 * tx2 + dNdx3 * tx3);

    // Bending moments
    let D = material.D;
    let nu = material.nu;
    let Mx_e  = D * (kappa_x + nu * kappa_y);
    let My_e  = D * (kappa_y + nu * kappa_x);
    let Mxy_e = D * (1.0 - nu) * kappa_xy / 2.0;

    // Scatter to nodes (accumulate — coloring ensures no race)
    Mx_sum[n0]  += Mx_e;  Mx_sum[n1]  += Mx_e;  Mx_sum[n2]  += Mx_e;  Mx_sum[n3]  += Mx_e;
    My_sum[n0]  += My_e;  My_sum[n1]  += My_e;  My_sum[n2]  += My_e;  My_sum[n3]  += My_e;
    Mxy_sum[n0] += Mxy_e; Mxy_sum[n1] += Mxy_e; Mxy_sum[n2] += Mxy_e; Mxy_sum[n3] += Mxy_e;
    count[n0]   += 1.0;   count[n1]   += 1.0;   count[n2]   += 1.0;   count[n3]   += 1.0;
}
`;

