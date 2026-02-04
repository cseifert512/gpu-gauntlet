// update_p.wgsl
// PCG update: p = z + beta * p
//
// Combined operation for search direction update.

struct Params {
    beta: f32,         // Scalar multiplier for old p
    n: u32,            // Vector length
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        p[i] = z[i] + params.beta * p[i];
    }
}

