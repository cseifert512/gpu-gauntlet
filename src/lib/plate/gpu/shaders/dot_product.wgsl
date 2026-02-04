// dot_product.wgsl
// Parallel dot product with workgroup reduction
//
// Computes sum(a[i] * b[i]) using tree reduction within workgroups.
// Output is partial sums per workgroup, requiring a final reduction pass.

struct Params {
    n: u32,            // Vector length
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = gid.x;
    let local_id = lid.x;
    
    // Each thread computes one product (or 0 if out of bounds)
    var val = 0.0;
    if (i < params.n) {
        val = a[i] * b[i];
    }
    sharedData[local_id] = val;
    
    workgroupBarrier();
    
    // Tree reduction within workgroup
    // Unrolled for efficiency
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
    
    // First thread writes workgroup result
    if (local_id == 0u) {
        partial_sums[wid.x] = sharedData[0] + sharedData[1];
    }
}

