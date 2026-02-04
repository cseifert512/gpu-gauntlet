// apply_k_dkt.wgsl
// Matrix-free K·x operation for DKT (Discrete Kirchhoff Triangle) elements.
//
// Each thread processes one element from a single color batch.
// Elements of the same color don't share nodes, so no write conflicts.
//
// DOFs per node: [w, θx, θy] (3 DOFs)
// DOFs per DKT element: 9 (3 nodes × 3 DOFs)
//
// Reference: Batoz, Bathe, Ho (1980) "A study of three-node triangular
// plate bending elements"

// Material uniform buffer
struct Material {
    E: f32,          // Young's modulus (Pa)
    nu: f32,         // Poisson's ratio
    t: f32,          // Thickness (m)
    D: f32,          // Flexural rigidity: E*t³/(12*(1-ν²))
    kappa_G_t: f32,  // Shear (not used for DKT - pure bending)
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

// Dispatch parameters
struct Params {
    color_offset: u32,    // Offset into elementsByColor for this color
    color_count: u32,     // Number of elements in this color
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<f32>;           // [x,y] per node
@group(0) @binding(1) var<storage, read> elements: array<u32>;        // [n0,n1,n2] per element
@group(0) @binding(2) var<storage, read> element_indices: array<u32>; // Elements for this color
@group(0) @binding(3) var<uniform> material: Material;
@group(0) @binding(4) var<storage, read> x: array<f32>;               // Input vector
@group(0) @binding(5) var<storage, read_write> y: array<f32>;         // Output vector (accumulate)
@group(0) @binding(6) var<storage, read> constrained: array<u32>;     // BC mask
@group(0) @binding(7) var<uniform> params: Params;

// 3-point Gauss quadrature for triangles (area coordinates)
const TRIANGLE_GAUSS: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
    vec3<f32>(1.0/6.0, 1.0/6.0, 2.0/3.0),  // L1, L2, L3
    vec3<f32>(2.0/3.0, 1.0/6.0, 1.0/6.0),
    vec3<f32>(1.0/6.0, 2.0/3.0, 1.0/6.0)
);

const TRIANGLE_WEIGHTS: array<f32, 3> = array<f32, 3>(
    1.0/3.0, 1.0/3.0, 1.0/3.0
);

// DKT coefficient structure (computed from edge geometry)
struct DKTCoeffs {
    // Edge 4 (node 2-3)
    a4: f32, b4: f32, c4: f32, d4: f32, e4: f32,
    // Edge 5 (node 3-1)
    a5: f32, b5: f32, c5: f32, d5: f32, e5: f32,
    // Edge 6 (node 1-2)
    a6: f32, b6: f32, c6: f32, d6: f32, e6: f32,
    // P, t, q, r coefficients
    P4: f32, P5: f32, P6: f32,
    t4: f32, t5: f32, t6: f32,
    q4: f32, q5: f32, q6: f32,
    r4: f32, r5: f32, r6: f32,
}

fn compute_dkt_coeffs(
    x12: f32, y12: f32, x23: f32, y23: f32, x31: f32, y31: f32,
    L12_sq: f32, L23_sq: f32, L31_sq: f32
) -> DKTCoeffs {
    var c: DKTCoeffs;
    
    // Edge 4 (node 2-3), k=4
    c.a4 = -x23 / L23_sq;
    c.b4 = (3.0 * x23 * y23) / (4.0 * L23_sq);
    c.c4 = (x23 * x23 - 2.0 * y23 * y23) / (4.0 * L23_sq);
    c.d4 = -y23 / L23_sq;
    c.e4 = (y23 * y23 - 2.0 * x23 * x23) / (4.0 * L23_sq);
    
    // Edge 5 (node 3-1), k=5
    c.a5 = -x31 / L31_sq;
    c.b5 = (3.0 * x31 * y31) / (4.0 * L31_sq);
    c.c5 = (x31 * x31 - 2.0 * y31 * y31) / (4.0 * L31_sq);
    c.d5 = -y31 / L31_sq;
    c.e5 = (y31 * y31 - 2.0 * x31 * x31) / (4.0 * L31_sq);
    
    // Edge 6 (node 1-2), k=6
    c.a6 = -x12 / L12_sq;
    c.b6 = (3.0 * x12 * y12) / (4.0 * L12_sq);
    c.c6 = (x12 * x12 - 2.0 * y12 * y12) / (4.0 * L12_sq);
    c.d6 = -y12 / L12_sq;
    c.e6 = (y12 * y12 - 2.0 * x12 * x12) / (4.0 * L12_sq);
    
    // P, t, q, r coefficients
    c.P4 = -6.0 * c.a4;
    c.P5 = -6.0 * c.a5;
    c.P6 = -6.0 * c.a6;
    
    c.t4 = -6.0 * c.d4;
    c.t5 = -6.0 * c.d5;
    c.t6 = -6.0 * c.d6;
    
    c.q4 = 3.0 * c.a4;
    c.q5 = 3.0 * c.a5;
    c.q6 = 3.0 * c.a6;
    
    c.r4 = 3.0 * c.d4;
    c.r5 = 3.0 * c.d5;
    c.r6 = 3.0 * c.d6;
    
    return c;
}

// Compute DKT Hx derivatives w.r.t. area coordinates L1, L2
fn compute_dkt_Bx(
    L1: f32, L2: f32, L3: f32,
    c: DKTCoeffs,
    dHxdL1: ptr<function, array<f32, 9>>,
    dHxdL2: ptr<function, array<f32, 9>>
) {
    // Node 1 DOFs (w1, θx1, θy1) - indices 0, 1, 2
    (*dHxdL1)[0] = c.P6 * (1.0 - 2.0 * L1) + (c.P5 - c.P6) * L2;
    (*dHxdL1)[1] = c.q6 * (1.0 - 2.0 * L1) - (c.q5 + c.q6) * L2;
    (*dHxdL1)[2] = -4.0 + 6.0 * (L1 + L2) + (c.c5 - c.c6) * L2 + c.c6 * (1.0 - 2.0 * L1);
    
    (*dHxdL2)[0] = -c.P6 * (1.0 - 2.0 * L2) + (c.P4 - c.P5) * L1 - c.P5 * (1.0 - 2.0 * L2);
    (*dHxdL2)[1] = c.q6 * (1.0 - 2.0 * L2) - (c.q4 + c.q5) * L1 + c.q5 * (1.0 - 2.0 * L2);
    (*dHxdL2)[2] = -4.0 + 6.0 * (L1 + L2) + (c.c4 - c.c5) * L1 + c.c5 * (1.0 - 2.0 * L2);
    
    // Node 2 DOFs (w2, θx2, θy2) - indices 3, 4, 5
    (*dHxdL1)[3] = c.P4 * L2 + c.P6 * (1.0 - 2.0 * L1);
    (*dHxdL1)[4] = c.q4 * L2 - c.q6 * (1.0 - 2.0 * L1);
    (*dHxdL1)[5] = c.c4 * L2 - c.c6 * (1.0 - 2.0 * L1);
    
    (*dHxdL2)[3] = c.P4 * (1.0 - 2.0 * L2) + c.P5 * L1;
    (*dHxdL2)[4] = c.q4 * (1.0 - 2.0 * L2) - c.q5 * L1;
    (*dHxdL2)[5] = c.c4 * (1.0 - 2.0 * L2) - c.c5 * L1;
    
    // Node 3 DOFs (w3, θx3, θy3) - indices 6, 7, 8
    (*dHxdL1)[6] = -c.P5 * L2 - c.P6 * (1.0 - 2.0 * L1);
    (*dHxdL1)[7] = c.q5 * L2 + c.q6 * (1.0 - 2.0 * L1);
    (*dHxdL1)[8] = -c.c5 * L2 + c.c6 * (1.0 - 2.0 * L1);
    
    (*dHxdL2)[6] = -c.P4 * L1 - c.P5 * (1.0 - 2.0 * L2);
    (*dHxdL2)[7] = c.q4 * L1 + c.q5 * (1.0 - 2.0 * L2);
    (*dHxdL2)[8] = -c.c4 * L1 + c.c5 * (1.0 - 2.0 * L2);
}

// Compute DKT Hy derivatives w.r.t. area coordinates L1, L2
fn compute_dkt_By(
    L1: f32, L2: f32, L3: f32,
    c: DKTCoeffs,
    dHydL1: ptr<function, array<f32, 9>>,
    dHydL2: ptr<function, array<f32, 9>>
) {
    // Node 1 DOFs
    (*dHydL1)[0] = c.t6 * (1.0 - 2.0 * L1) + (c.t5 - c.t6) * L2;
    (*dHydL1)[1] = 1.0 + c.r6 * (1.0 - 2.0 * L1) - (c.r5 + c.r6) * L2;
    (*dHydL1)[2] = -c.e6 * (1.0 - 2.0 * L1) + (c.b5 - c.b6) * L2;
    
    (*dHydL2)[0] = -c.t6 * (1.0 - 2.0 * L2) + (c.t4 - c.t5) * L1 - c.t5 * (1.0 - 2.0 * L2);
    (*dHydL2)[1] = 1.0 + c.r6 * (1.0 - 2.0 * L2) - (c.r4 + c.r5) * L1 + c.r5 * (1.0 - 2.0 * L2);
    (*dHydL2)[2] = c.e6 * (1.0 - 2.0 * L2) + (c.b4 - c.b5) * L1 - c.b5 * (1.0 - 2.0 * L2);
    
    // Node 2 DOFs
    (*dHydL1)[3] = c.t4 * L2 + c.t6 * (1.0 - 2.0 * L1);
    (*dHydL1)[4] = -1.0 + c.r4 * L2 - c.r6 * (1.0 - 2.0 * L1);
    (*dHydL1)[5] = -c.e4 * L2 - c.b6 * (1.0 - 2.0 * L1);
    
    (*dHydL2)[3] = c.t4 * (1.0 - 2.0 * L2) + c.t5 * L1;
    (*dHydL2)[4] = -1.0 + c.r4 * (1.0 - 2.0 * L2) - c.r5 * L1;
    (*dHydL2)[5] = -c.e4 * (1.0 - 2.0 * L2) + c.b5 * L1;
    
    // Node 3 DOFs
    (*dHydL1)[6] = -c.t5 * L2 - c.t6 * (1.0 - 2.0 * L1);
    (*dHydL1)[7] = c.r5 * L2 + c.r6 * (1.0 - 2.0 * L1);
    (*dHydL1)[8] = c.e5 * L2 + c.b6 * (1.0 - 2.0 * L1);
    
    (*dHydL2)[6] = -c.t4 * L1 - c.t5 * (1.0 - 2.0 * L2);
    (*dHydL2)[7] = c.r4 * L1 + c.r5 * (1.0 - 2.0 * L2);
    (*dHydL2)[8] = c.e4 * L1 - c.b5 * (1.0 - 2.0 * L2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }
    
    // Get element index for this thread
    let elem_idx = element_indices[params.color_offset + thread_idx];
    
    // Get node indices for this element (3 nodes for triangle)
    let base = elem_idx * 3u;
    let n0 = elements[base + 0u];
    let n1 = elements[base + 1u];
    let n2 = elements[base + 2u];
    
    // Get node coordinates
    let x1 = nodes[n0 * 2u];
    let y1 = nodes[n0 * 2u + 1u];
    let x2 = nodes[n1 * 2u];
    let y2 = nodes[n1 * 2u + 1u];
    let x3 = nodes[n2 * 2u];
    let y3 = nodes[n2 * 2u + 1u];
    
    // Edge vectors and lengths squared
    let x12 = x1 - x2;
    let y12 = y1 - y2;
    let x23 = x2 - x3;
    let y23 = y2 - y3;
    let x31 = x3 - x1;
    let y31 = y3 - y1;
    
    let L12_sq = x12 * x12 + y12 * y12;
    let L23_sq = x23 * x23 + y23 * y23;
    let L31_sq = x31 * x31 + y31 * y31;
    
    // Triangle area (twice area)
    let twoA = x12 * y31 - x31 * y12;
    let area = abs(twoA) * 0.5;
    
    // Skip degenerate triangles
    if (area < 1e-15) {
        return;
    }
    
    // Gather local DOF values (w, θx, θy per node = 9 total)
    var x_local: array<f32, 9>;
    var node_dofs: array<u32, 3>;
    node_dofs[0] = n0 * 3u;
    node_dofs[1] = n1 * 3u;
    node_dofs[2] = n2 * 3u;
    
    for (var i = 0u; i < 3u; i++) {
        let dof_base = node_dofs[i];
        x_local[i * 3u + 0u] = x[dof_base + 0u];  // w
        x_local[i * 3u + 1u] = x[dof_base + 1u];  // θx
        x_local[i * 3u + 2u] = x[dof_base + 2u];  // θy
    }
    
    // Initialize y_local = 0
    var y_local: array<f32, 9>;
    for (var j = 0u; j < 9u; j++) {
        y_local[j] = 0.0;
    }
    
    // Compute DKT coefficients
    let coeffs = compute_dkt_coeffs(x12, y12, x23, y23, x31, y31, L12_sq, L23_sq, L31_sq);
    
    // Material D matrix coefficients
    let D = material.D;
    let nu = material.nu;
    let D11 = D;
    let D12 = D * nu;
    let D22 = D;
    let D33 = D * (1.0 - nu) * 0.5;
    
    let invTwoA = 1.0 / twoA;
    
    // 3-point Gauss integration
    for (var gp = 0u; gp < 3u; gp++) {
        let L1 = TRIANGLE_GAUSS[gp].x;
        let L2 = TRIANGLE_GAUSS[gp].y;
        let L3 = TRIANGLE_GAUSS[gp].z;
        let weight = TRIANGLE_WEIGHTS[gp];
        
        // Compute DKT shape function derivatives
        var dHxdL1: array<f32, 9>;
        var dHxdL2: array<f32, 9>;
        var dHydL1: array<f32, 9>;
        var dHydL2: array<f32, 9>;
        
        compute_dkt_Bx(L1, L2, L3, coeffs, &dHxdL1, &dHxdL2);
        compute_dkt_By(L1, L2, L3, coeffs, &dHydL1, &dHydL2);
        
        // Transform to physical coordinates and build B matrix contributions
        // κx = dβx/dx = (1/2A) * (y23 * dHx/dL1 + y31 * dHx/dL2) [with sign corrections]
        // κy = dβy/dy = (1/2A) * (x32 * dHy/dL1 + x13 * dHy/dL2) [with sign corrections]
        // κxy = dβx/dy + dβy/dx
        
        // Note: y23 = y2 - y3, x23 = x2 - x3, etc.
        // For the area coordinate transformation:
        // d/dx = (y_jk * d/dL_i + ...) / 2A
        
        // Compute curvatures from current DOF values
        var kappa_x = 0.0;
        var kappa_y = 0.0;
        var kappa_xy = 0.0;
        
        for (var j = 0u; j < 9u; j++) {
            // B matrix rows (3) for column j
            let Bx_j = invTwoA * ((-y23) * dHxdL1[j] + (-y31) * dHxdL2[j]);
            let By_j = invTwoA * (x23 * dHydL1[j] + x31 * dHydL2[j]);
            let Bxy_j = invTwoA * (
                x23 * dHxdL1[j] + x31 * dHxdL2[j] +
                (-y23) * dHydL1[j] + (-y31) * dHydL2[j]
            );
            
            kappa_x += Bx_j * x_local[j];
            kappa_y += By_j * x_local[j];
            kappa_xy += Bxy_j * x_local[j];
        }
        
        // Moment resultants: M = D * κ
        let Mx = D11 * kappa_x + D12 * kappa_y;
        let My = D12 * kappa_x + D22 * kappa_y;
        let Mxy = D33 * kappa_xy;
        
        let factor = weight * area;
        
        // Accumulate y_local = B^T * M
        for (var i = 0u; i < 9u; i++) {
            let Bx_i = invTwoA * ((-y23) * dHxdL1[i] + (-y31) * dHxdL2[i]);
            let By_i = invTwoA * (x23 * dHydL1[i] + x31 * dHydL2[i]);
            let Bxy_i = invTwoA * (
                x23 * dHxdL1[i] + x31 * dHxdL2[i] +
                (-y23) * dHydL1[i] + (-y31) * dHydL2[i]
            );
            
            y_local[i] += factor * (Bx_i * Mx + By_i * My + Bxy_i * Mxy);
        }
    }
    
    // Scatter to global y (no conflicts within same color!)
    for (var i = 0u; i < 3u; i++) {
        let dof_base = node_dofs[i];
        
        for (var d = 0u; d < 3u; d++) {
            let global_dof = dof_base + d;
            let local_dof = i * 3u + d;
            
            // Only accumulate if not constrained
            if (constrained[global_dof] == 0u) {
                y[global_dof] += y_local[local_dof];
            }
        }
    }
}

