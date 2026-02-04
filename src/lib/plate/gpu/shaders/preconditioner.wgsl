// preconditioner.wgsl
// Apply Jacobi preconditioner: z = r / diag
//
// Element-wise division with safeguard against zero diagonal.

struct Params {
    n: u32,            // Vector length (number of DOFs)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> r: array<f32>;
@group(0) @binding(1) var<storage, read> diag: array<f32>;
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        // Avoid division by zero with safeguard
        let d = max(abs(diag[i]), 1e-30);
        z[i] = r[i] / d;
    }
}

