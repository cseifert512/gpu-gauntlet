/**
 * Q4 Mindlin plate element K·x shader source.
 * Exported as string for cross-platform compatibility.
 */

export const applyKQ4Source = /* wgsl */ `
// apply_k_q4.wgsl
// Matrix-free K·x operation for Q4 Mindlin plate elements.

struct Material {
    E: f32,
    nu: f32,
    t: f32,
    D: f32,
    kappa_G_t: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
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
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> constrained: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

const GP: f32 = 0.5773502691896257;

const GAUSS_2X2: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-GP, -GP),
    vec2<f32>( GP, -GP),
    vec2<f32>( GP,  GP),
    vec2<f32>(-GP,  GP)
);

fn shape_functions(xi: f32, eta: f32) -> vec4<f32> {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    
    return vec4<f32>(
        xm * em * 0.25,
        xp * em * 0.25,
        xp * ep * 0.25,
        xm * ep * 0.25
    );
}

fn shape_derivatives(xi: f32, eta: f32, dNdXi: ptr<function, vec4<f32>>, dNdEta: ptr<function, vec4<f32>>) {
    let xm = 1.0 - xi;
    let xp = 1.0 + xi;
    let em = 1.0 - eta;
    let ep = 1.0 + eta;
    
    (*dNdXi)[0] = -em * 0.25;
    (*dNdXi)[1] =  em * 0.25;
    (*dNdXi)[2] =  ep * 0.25;
    (*dNdXi)[3] = -ep * 0.25;
    
    (*dNdEta)[0] = -xm * 0.25;
    (*dNdEta)[1] = -xp * 0.25;
    (*dNdEta)[2] =  xp * 0.25;
    (*dNdEta)[3] =  xm * 0.25;
}

fn compute_jacobian(
    coords: array<vec2<f32>, 4>,
    dNdXi: vec4<f32>,
    dNdEta: vec4<f32>
) -> mat2x2<f32> {
    var J = mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    
    for (var i = 0u; i < 4u; i++) {
        let dxi = dNdXi[i];
        let deta = dNdEta[i];
        
        J[0][0] += dxi * coords[i].x;
        J[0][1] += dxi * coords[i].y;
        J[1][0] += deta * coords[i].x;
        J[1][1] += deta * coords[i].y;
    }
    
    return J;
}

fn compute_physical_derivatives(
    dNdXi: vec4<f32>,
    dNdEta: vec4<f32>,
    J11: f32, J12: f32, J21: f32, J22: f32,
    detJ: f32,
    dNdx: ptr<function, vec4<f32>>,
    dNdy: ptr<function, vec4<f32>>
) {
    let invDetJ = 1.0 / detJ;
    
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
    
    let elem_idx = element_indices[params.color_offset + thread_idx];
    
    let base = elem_idx * 4u;
    let n0 = elements[base + 0u];
    let n1 = elements[base + 1u];
    let n2 = elements[base + 2u];
    let n3 = elements[base + 3u];
    
    var coords: array<vec2<f32>, 4>;
    coords[0] = vec2<f32>(nodes[n0 * 2u], nodes[n0 * 2u + 1u]);
    coords[1] = vec2<f32>(nodes[n1 * 2u], nodes[n1 * 2u + 1u]);
    coords[2] = vec2<f32>(nodes[n2 * 2u], nodes[n2 * 2u + 1u]);
    coords[3] = vec2<f32>(nodes[n3 * 2u], nodes[n3 * 2u + 1u]);
    
    var x_local: array<f32, 12>;
    var node_dofs: array<u32, 4>;
    node_dofs[0] = n0 * 3u;
    node_dofs[1] = n1 * 3u;
    node_dofs[2] = n2 * 3u;
    node_dofs[3] = n3 * 3u;
    
    for (var i = 0u; i < 4u; i++) {
        let dof_base = node_dofs[i];
        x_local[i * 3u + 0u] = x[dof_base + 0u];
        x_local[i * 3u + 1u] = x[dof_base + 1u];
        x_local[i * 3u + 2u] = x[dof_base + 2u];
    }
    
    var y_local: array<f32, 12>;
    for (var j = 0u; j < 12u; j++) {
        y_local[j] = 0.0;
    }
    
    let D_b = material.D;
    let nu = material.nu;
    
    let D11 = D_b;
    let D12 = D_b * nu;
    let D22 = D_b;
    let D33 = D_b * (1.0 - nu) * 0.5;
    
    // Bending stiffness (2×2 Gauss integration)
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
        
        let weight = 1.0;
        let factor = weight * abs(detJ);
        
        for (var i = 0u; i < 4u; i++) {
            let di_tx = i * 3u + 1u;
            let di_ty = i * 3u + 2u;
            
            let dNdx_i = dNdx[i];
            let dNdy_i = dNdy[i];
            
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
            
            let Mx = D11 * kappa_x + D12 * kappa_y;
            let My = D12 * kappa_x + D22 * kappa_y;
            let Mxy = D33 * kappa_xy;
            
            y_local[di_tx] += factor * (-dNdy_i * My - dNdx_i * Mxy);
            y_local[di_ty] += factor * (dNdx_i * Mx + dNdy_i * Mxy);
        }
    }
    
    // Shear stiffness (1-point reduced integration)
    {
        let xi = 0.0;
        let eta = 0.0;
        let weight = 4.0;
        
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
        let Ds = material.kappa_G_t;
        
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
        
        let Qx = Ds * gamma_xz;
        let Qy = Ds * gamma_yz;
        
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
    
    // Scatter to global y
    for (var i = 0u; i < 4u; i++) {
        let dof_base = node_dofs[i];
        
        for (var d = 0u; d < 3u; d++) {
            let global_dof = dof_base + d;
            let local_dof = i * 3u + d;
            
            if (constrained[global_dof] == 0u) {
                y[global_dof] += y_local[local_dof];
            }
        }
    }
}
`;

