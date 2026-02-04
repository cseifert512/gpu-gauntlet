// reduce_sum.wgsl
// Final reduction pass to sum partial results from dot_product kernel.
//
// Takes the partial sums from each workgroup and reduces to a single value.

struct Params {
    n: u32,            // Number of partial sums
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_id = lid.x;
    
    // Load value (or 0 if out of bounds)
    var val = 0.0;
    if (local_id < params.n) {
        val = input[local_id];
    }
    sharedData[local_id] = val;
    
    workgroupBarrier();
    
    // Tree reduction
    if (local_id < 128u) {
        sharedData[local_id] += sharedData[local_id + 128u];
    }
    workgroupBarrier();
    
    if (local_id < 64u) {
        sharedData[local_id] += sharedData[local_id + 64u];
    }
    workgroupBarrier();
    
    if (local_id < 32u) {
        sharedData[local_id] += sharedData[local_id + 32u];
    }
    workgroupBarrier();
    
    if (local_id < 16u) {
        sharedData[local_id] += sharedData[local_id + 16u];
    }
    workgroupBarrier();
    
    if (local_id < 8u) {
        sharedData[local_id] += sharedData[local_id + 8u];
    }
    workgroupBarrier();
    
    if (local_id < 4u) {
        sharedData[local_id] += sharedData[local_id + 4u];
    }
    workgroupBarrier();
    
    if (local_id < 2u) {
        sharedData[local_id] += sharedData[local_id + 2u];
    }
    workgroupBarrier();
    
    // First thread writes final result
    if (local_id == 0u) {
        output[0] = sharedData[0] + sharedData[1];
    }
}

