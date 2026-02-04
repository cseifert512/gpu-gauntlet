// apply_bc.wgsl
// Apply boundary conditions to K·x result.
// For constrained DOFs: y[i] = x[i] (identity)

struct Params {
    n: u32,            // Vector length (number of DOFs)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> constrained: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        // For constrained DOFs, set y = x (identity behavior)
        if (constrained[i] != 0u) {
            y[i] = x[i];
        }
    }
}

