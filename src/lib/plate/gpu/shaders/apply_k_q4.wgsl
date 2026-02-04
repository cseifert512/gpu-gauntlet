// apply_k_q4.wgsl
// Matrix-free K·x operation for Q4 Mindlin plate elements.
//
// Each thread processes one element from a single color batch.
// Elements of the same color don't share nodes, so no write conflicts.
//
// DOFs per node: [w, θx, θy] (3 DOFs)
// DOFs per Q4 element: 12 (4 nodes × 3 DOFs)
//
// Element stiffness: K = Kb + Ks (bending + shear)
// - Kb: 2×2 Gauss integration for bending
// - Ks: 1-point reduced integration for shear (avoids locking)

// Material uniform buffer
struct Material {
    E: f32,          // Young's modulus (Pa)
    nu: f32,         // Poisson's ratio
    t: f32,          // Thickness (m)
    D: f32,          // Flexural rigidity: E*t³/(12*(1-ν²))
    kappa_G_t: f32,  // Shear: κ*G*t where κ=5/6, G=E/(2(1+ν))
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
@group(0) @binding(1) var<storage, read> elements: array<u32>;        // [n0,n1,n2,n3] per element
@group(0) @binding(2) var<storage, read> element_indices: array<u32>; // Elements for this color
@group(0) @binding(3) var<uniform> material: Material;
@group(0) @binding(4) var<storage, read> x: array<f32>;               // Input vector
@group(0) @binding(5) var<storage, read_write> y: array<f32>;         // Output vector (accumulate)
@group(0) @binding(6) var<storage, read> constrained: array<u32>;     // BC mask
@group(0) @binding(7) var<uniform> params: Params;

// 2×2 Gauss quadrature points: ±1/√3
const GP: f32 = 0.5773502691896257;

// Gauss points for 2×2 integration
const GAUSS_2X2: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-GP, -GP),
    vec2<f32>( GP, -GP),
    vec2<f32>( GP,  GP),
    vec2<f32>(-GP,  GP)
);

// Compute Q4 shape functions at (xi, eta)
// N_i = (1 ± ξ)(1 ± η)/4
fn shape_functions(xi: f32, eta: f32) -> vec4<f32> {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    
    return vec4<f32>(
        xm * em * 0.25,  // N0
        xp * em * 0.25,  // N1
        xp * ep * 0.25,  // N2
        xm * ep * 0.25   // N3
    );
}

// Compute shape function derivatives w.r.t. natural coordinates
// Returns: (dN/dξ, dN/dη) packed as dNdXi in .xyzw, dNdEta in second vec4
fn shape_derivatives(xi: f32, eta: f32, dNdXi: ptr<function, vec4<f32>>, dNdEta: ptr<function, vec4<f32>>) {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    
    // dN/dξ
    (*dNdXi)[0] = -em * 0.25;  // dN0/dξ
    (*dNdXi)[1] =  em * 0.25;  // dN1/dξ
    (*dNdXi)[2] =  ep * 0.25;  // dN2/dξ
    (*dNdXi)[3] = -ep * 0.25;  // dN3/dξ
    
    // dN/dη
    (*dNdEta)[0] = -xm * 0.25;  // dN0/dη
    (*dNdEta)[1] = -xp * 0.25;  // dN1/dη
    (*dNdEta)[2] =  xp * 0.25;  // dN2/dη
    (*dNdEta)[3] =  xm * 0.25;  // dN3/dη
}

// Compute Jacobian and its inverse
// J = [∂x/∂ξ  ∂y/∂ξ]
//     [∂x/∂η  ∂y/∂η]
fn compute_jacobian(
    coords: array<vec2<f32>, 4>,
    dNdXi: vec4<f32>,
    dNdEta: vec4<f32>
) -> mat2x2<f32> {
    var J = mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    
    for (var i = 0u; i < 4u; i++) {
        let dxi = dNdXi[i];
        let deta = dNdEta[i];
        
        J[0][0] += dxi * coords[i].x;   // ∂x/∂ξ
        J[0][1] += dxi * coords[i].y;   // ∂y/∂ξ
        J[1][0] += deta * coords[i].x;  // ∂x/∂η
        J[1][1] += deta * coords[i].y;  // ∂y/∂η
    }
    
    return J;
}

// Compute physical derivatives from natural derivatives
fn compute_physical_derivatives(
    dNdXi: vec4<f32>,
    dNdEta: vec4<f32>,
    J11: f32, J12: f32, J21: f32, J22: f32,
    detJ: f32,
    dNdx: ptr<function, vec4<f32>>,
    dNdy: ptr<function, vec4<f32>>
) {
    let invDetJ = 1.0 / detJ;
    
    // J^{-1} = (1/detJ) [J22  -J12]
    //                   [-J21  J11]
    for (var i = 0u; i < 4u; i++) {
        (*dNdx)[i] = invDetJ * (J22 * dNdXi[i] - J12 * dNdEta[i]);
        (*dNdy)[i] = invDetJ * (-J21 * dNdXi[i] + J11 * dNdEta[i]);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }
    
    // Get element index for this thread
    let elem_idx = element_indices[params.color_offset + thread_idx];
    
    // Get node indices for this element
    let base = elem_idx * 4u;
    let n0 = elements[base + 0u];
    let n1 = elements[base + 1u];
    let n2 = elements[base + 2u];
    let n3 = elements[base + 3u];
    
    // Gather node coordinates
    var coords: array<vec2<f32>, 4>;
    coords[0] = vec2<f32>(nodes[n0 * 2u], nodes[n0 * 2u + 1u]);
    coords[1] = vec2<f32>(nodes[n1 * 2u], nodes[n1 * 2u + 1u]);
    coords[2] = vec2<f32>(nodes[n2 * 2u], nodes[n2 * 2u + 1u]);
    coords[3] = vec2<f32>(nodes[n3 * 2u], nodes[n3 * 2u + 1u]);
    
    // Gather local DOF values (w, θx, θy per node = 12 total)
    var x_local: array<f32, 12>;
    var node_dofs: array<u32, 4>;
    node_dofs[0] = n0 * 3u;
    node_dofs[1] = n1 * 3u;
    node_dofs[2] = n2 * 3u;
    node_dofs[3] = n3 * 3u;
    
    for (var i = 0u; i < 4u; i++) {
        let dof_base = node_dofs[i];
        x_local[i * 3u + 0u] = x[dof_base + 0u];  // w
        x_local[i * 3u + 1u] = x[dof_base + 1u];  // θx
        x_local[i * 3u + 2u] = x[dof_base + 2u];  // θy
    }
    
    // Initialize y_local = 0
    var y_local: array<f32, 12>;
    for (var j = 0u; j < 12u; j++) {
        y_local[j] = 0.0;
    }
    
    // Material properties
    let D_b = material.D;
    let nu = material.nu;
    
    // D matrix coefficients for bending
    // Db = D * [1   ν   0      ]
    //          [ν   1   0      ]
    //          [0   0   (1-ν)/2]
    let D11 = D_b;
    let D12 = D_b * nu;
    let D22 = D_b;
    let D33 = D_b * (1.0 - nu) * 0.5;
    
    // =========================================================================
    // Bending stiffness (2×2 Gauss integration)
    // =========================================================================
    // Curvature-rotation relationship:
    // κ = {κx, κy, κxy} = {∂θy/∂x, -∂θx/∂y, ∂θy/∂y - ∂θx/∂x}
    //
    // Bb matrix entries for node i:
    // Bb[0, θy_i] = dN_i/dx    (κx)
    // Bb[1, θx_i] = -dN_i/dy   (κy)
    // Bb[2, θx_i] = -dN_i/dx   (κxy)
    // Bb[2, θy_i] = dN_i/dy    (κxy)
    
    for (var gp = 0u; gp < 4u; gp++) {
        let xi = GAUSS_2X2[gp].x;
        let eta = GAUSS_2X2[gp].y;
        
        var dNdXi: vec4<f32>;
        var dNdEta: vec4<f32>;
        shape_derivatives(xi, eta, &dNdXi, &dNdEta);
        
        let J = compute_jacobian(coords, dNdXi, dNdEta);
        let detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        
        var dNdx: vec4<f32>;
        var dNdy: vec4<f32>;
        compute_physical_derivatives(dNdXi, dNdEta, J[0][0], J[0][1], J[1][0], J[1][1], detJ, &dNdx, &dNdy);
        
        let weight = 1.0;  // Gauss weight for 2×2
        let factor = weight * abs(detJ);
        
        // Compute y_local contribution from bending: y += B^T * D * B * x
        // Instead of forming full matrices, compute directly
        
        for (var i = 0u; i < 4u; i++) {
            let di_w = i * 3u;
            let di_tx = i * 3u + 1u;
            let di_ty = i * 3u + 2u;
            
            let dNdx_i = dNdx[i];
            let dNdy_i = dNdy[i];
            
            // Compute stress resultants from all nodes
            // κx = sum_j dNdx_j * θy_j
            // κy = sum_j -dNdy_j * θx_j
            // κxy = sum_j (dNdy_j * θy_j - dNdx_j * θx_j)
            var kappa_x = 0.0;
            var kappa_y = 0.0;
            var kappa_xy = 0.0;
            
            for (var j = 0u; j < 4u; j++) {
                let dNdx_j = dNdx[j];
                let dNdy_j = dNdy[j];
                let theta_x_j = x_local[j * 3u + 1u];
                let theta_y_j = x_local[j * 3u + 2u];
                
                kappa_x += dNdx_j * theta_y_j;
                kappa_y -= dNdy_j * theta_x_j;
                kappa_xy += dNdy_j * theta_y_j - dNdx_j * theta_x_j;
            }
            
            // Moment resultants: M = D * κ
            let Mx = D11 * kappa_x + D12 * kappa_y;
            let My = D12 * kappa_x + D22 * kappa_y;
            let Mxy = D33 * kappa_xy;
            
            // Accumulate to y_local: y_i = B_i^T * M
            // y[θx_i] += -dNdy_i * My - dNdx_i * Mxy
            // y[θy_i] += dNdx_i * Mx + dNdy_i * Mxy
            
            y_local[di_tx] += factor * (-dNdy_i * My - dNdx_i * Mxy);
            y_local[di_ty] += factor * (dNdx_i * Mx + dNdy_i * Mxy);
        }
    }
    
    // =========================================================================
    // Shear stiffness (1-point reduced integration to avoid locking)
    // =========================================================================
    // Shear strains: γ = {γxz, γyz} = {∂w/∂x + θy, ∂w/∂y - θx}
    //
    // Bs matrix entries for node i:
    // Bs[0, w_i] = dN_i/dx     (γxz)
    // Bs[0, θy_i] = N_i        (γxz)
    // Bs[1, w_i] = dN_i/dy     (γyz)
    // Bs[1, θx_i] = -N_i       (γyz)
    {
        let xi = 0.0;
        let eta = 0.0;
        let weight = 4.0;  // Total weight for 1-point rule
        
        let N = shape_functions(xi, eta);
        
        var dNdXi: vec4<f32>;
        var dNdEta: vec4<f32>;
        shape_derivatives(xi, eta, &dNdXi, &dNdEta);
        
        let J = compute_jacobian(coords, dNdXi, dNdEta);
        let detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        
        var dNdx: vec4<f32>;
        var dNdy: vec4<f32>;
        compute_physical_derivatives(dNdXi, dNdEta, J[0][0], J[0][1], J[1][0], J[1][1], detJ, &dNdx, &dNdy);
        
        let factor = weight * abs(detJ);
        
        // Shear constitutive: Ds = κ*G*t * I
        let Ds = material.kappa_G_t;
        
        // Compute shear strains
        var gamma_xz = 0.0;
        var gamma_yz = 0.0;
        
        for (var j = 0u; j < 4u; j++) {
            let dNdx_j = dNdx[j];
            let dNdy_j = dNdy[j];
            let N_j = N[j];
            let w_j = x_local[j * 3u + 0u];
            let theta_x_j = x_local[j * 3u + 1u];
            let theta_y_j = x_local[j * 3u + 2u];
            
            gamma_xz += dNdx_j * w_j + N_j * theta_y_j;
            gamma_yz += dNdy_j * w_j - N_j * theta_x_j;
        }
        
        // Shear forces: Q = Ds * γ
        let Qx = Ds * gamma_xz;
        let Qy = Ds * gamma_yz;
        
        // Accumulate to y_local: y_i = Bs_i^T * Q
        for (var i = 0u; i < 4u; i++) {
            let di_w = i * 3u;
            let di_tx = i * 3u + 1u;
            let di_ty = i * 3u + 2u;
            
            let dNdx_i = dNdx[i];
            let dNdy_i = dNdy[i];
            let N_i = N[i];
            
            y_local[di_w] += factor * (dNdx_i * Qx + dNdy_i * Qy);
            y_local[di_tx] += factor * (-N_i * Qy);
            y_local[di_ty] += factor * (N_i * Qx);
        }
    }
    
    // =========================================================================
    // Scatter to global y (no conflicts within same color!)
    // =========================================================================
    for (var i = 0u; i < 4u; i++) {
        let dof_base = node_dofs[i];
        
        for (var d = 0u; d < 3u; d++) {
            let global_dof = dof_base + d;
            let local_dof = i * 3u + d;
            
            // Only accumulate if not constrained (BCs handled separately)
            if (constrained[global_dof] == 0u) {
                y[global_dof] += y_local[local_dof];
            }
        }
    }
}

