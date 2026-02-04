/**
 * Shader source loading.
 *
 * This module provides shader sources as strings.
 * Works in both browser (bundled) and Node.js (Jest) environments.
 */

// In Node.js/Jest environment, we load shaders from files
// In browser with bundler, these would be imported with ?raw
// For now, we inline them as template strings for cross-platform compatibility

export const dotProductSource = /* wgsl */ `
// dot_product.wgsl
// Parallel dot product with workgroup reduction

struct Params {
    n: u32,
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
    
    var val = 0.0;
    if (i < params.n) {
        val = a[i] * b[i];
    }
    sharedData[local_id] = val;
    
    workgroupBarrier();
    
    if (local_id < 128u) { sharedData[local_id] += sharedData[local_id + 128u]; }
    workgroupBarrier();
    if (local_id < 64u) { sharedData[local_id] += sharedData[local_id + 64u]; }
    workgroupBarrier();
    if (local_id < 32u) { sharedData[local_id] += sharedData[local_id + 32u]; }
    workgroupBarrier();
    if (local_id < 16u) { sharedData[local_id] += sharedData[local_id + 16u]; }
    workgroupBarrier();
    if (local_id < 8u) { sharedData[local_id] += sharedData[local_id + 8u]; }
    workgroupBarrier();
    if (local_id < 4u) { sharedData[local_id] += sharedData[local_id + 4u]; }
    workgroupBarrier();
    if (local_id < 2u) { sharedData[local_id] += sharedData[local_id + 2u]; }
    workgroupBarrier();
    
    if (local_id == 0u) {
        partial_sums[wid.x] = sharedData[0] + sharedData[1];
    }
}
`;

export const reduceSumSource = /* wgsl */ `
// reduce_sum.wgsl
// Final reduction pass

struct Params {
    n: u32,
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
    
    var val = 0.0;
    if (local_id < params.n) {
        val = input[local_id];
    }
    sharedData[local_id] = val;
    
    workgroupBarrier();
    
    if (local_id < 128u) { sharedData[local_id] += sharedData[local_id + 128u]; }
    workgroupBarrier();
    if (local_id < 64u) { sharedData[local_id] += sharedData[local_id + 64u]; }
    workgroupBarrier();
    if (local_id < 32u) { sharedData[local_id] += sharedData[local_id + 32u]; }
    workgroupBarrier();
    if (local_id < 16u) { sharedData[local_id] += sharedData[local_id + 16u]; }
    workgroupBarrier();
    if (local_id < 8u) { sharedData[local_id] += sharedData[local_id + 8u]; }
    workgroupBarrier();
    if (local_id < 4u) { sharedData[local_id] += sharedData[local_id + 4u]; }
    workgroupBarrier();
    if (local_id < 2u) { sharedData[local_id] += sharedData[local_id + 2u]; }
    workgroupBarrier();
    
    if (local_id == 0u) {
        output[0] = sharedData[0] + sharedData[1];
    }
}
`;

export const axpySource = /* wgsl */ `
// axpy.wgsl - y = alpha * x + y

struct Params {
    alpha: f32,
    n: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        y[i] = params.alpha * x[i] + y[i];
    }
}
`;

export const scaleSource = /* wgsl */ `
// scale.wgsl - x = alpha * x

struct Params {
    alpha: f32,
    n: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        x[i] = params.alpha * x[i];
    }
}
`;

export const copySource = /* wgsl */ `
// copy.wgsl - dst = src

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        dst[i] = src[i];
    }
}
`;

export const preconditionerSource = /* wgsl */ `
// preconditioner.wgsl - z = r / diag (scalar diagonal preconditioner)

struct Params {
    n: u32,
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
        let d = max(abs(diag[i]), 1e-30);
        z[i] = r[i] / d;
    }
}
`;

export const blockPreconditionerSource = /* wgsl */ `
// block_preconditioner.wgsl - z = M^-1 * r using 3x3 block inverses
// Each node has 3 DOFs and a 3x3 inverted block (9 floats per node)
// One thread per NODE (not per DOF)

struct Params {
    nodeCount: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> r: array<f32>;
@group(0) @binding(1) var<storage, read> blockInv: array<f32>;  // 9 floats per node
@group(0) @binding(2) var<storage, read_write> z: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if (node >= params.nodeCount) {
        return;
    }
    
    let dof_offset = node * 3u;
    let block_offset = node * 9u;
    
    // Load residual for this node's 3 DOFs
    let r0 = r[dof_offset];
    let r1 = r[dof_offset + 1u];
    let r2 = r[dof_offset + 2u];
    
    // Load 3x3 inverse block (row-major)
    let m00 = blockInv[block_offset + 0u];
    let m01 = blockInv[block_offset + 1u];
    let m02 = blockInv[block_offset + 2u];
    let m10 = blockInv[block_offset + 3u];
    let m11 = blockInv[block_offset + 4u];
    let m12 = blockInv[block_offset + 5u];
    let m20 = blockInv[block_offset + 6u];
    let m21 = blockInv[block_offset + 7u];
    let m22 = blockInv[block_offset + 8u];
    
    // Compute z = M^-1 * r (3x3 matrix-vector multiply)
    z[dof_offset]      = m00 * r0 + m01 * r1 + m02 * r2;
    z[dof_offset + 1u] = m10 * r0 + m11 * r1 + m12 * r2;
    z[dof_offset + 2u] = m20 * r0 + m21 * r1 + m22 * r2;
}
`;

export const updatePSource = /* wgsl */ `
// update_p.wgsl - p = z + beta * p

struct Params {
    beta: f32,
    n: u32,
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
`;

export const zeroBufferSource = /* wgsl */ `
// zero_buffer.wgsl

struct Params {
    n: u32,
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
`;

export const applyBCSource = /* wgsl */ `
// apply_bc.wgsl - y[constrained] = x[constrained]

struct Params {
    n: u32,
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
        if (constrained[i] != 0u) {
            y[i] = x[i];
        }
    }
}
`;

// Import the larger apply_k shaders from separate files (they're too large to inline comfortably)
import { applyKQ4Source } from './apply_k_q4_source';
import { applyKDKTSource } from './apply_k_dkt_source';

export { applyKQ4Source, applyKDKTSource };

