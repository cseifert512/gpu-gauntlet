// zero_buffer.wgsl
// Zero out a buffer.

struct Params {
    n: u32,            // Buffer length
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> buffer: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        buffer[i] = 0.0;
    }
}

