/**
 * WGSL compute shader sources for the GPU plate solver.
 *
 * All shaders are inlined as template strings for cross-platform compatibility
 * (works in both browser/bundler and Node.js/Jest environments).
 *
 * Shader categories:
 *   K·p (stiffness matrix-vector product):
 *     - applyKQ4Source / applyKDKTSource: Element-by-element with on-the-fly Ke
 *     - applyKPrecomputedQ4Source / applyKPrecomputedDKTSource: Pre-computed Ke
 *     - spmvCSRSource: CSR sparse matrix-vector (experimental)
 *
 *   PCG vector operations:
 *     - axpySource: y += α·x (α from uniform)
 *     - axpyBufSource: y += α[0]·x (α from GPU buffer — no CPU readback)
 *     - updatePSource: p = z + β·p (β from uniform)
 *     - updatePBufSource: p = z + β[0]·p (β from GPU buffer)
 *     - copySource, scaleSource, zeroBufferSource: basic vector ops
 *
 *   PCG scalar operations (GPU-resident, eliminate CPU roundtrips):
 *     - dotSingleSource: Single-workgroup dot product → result[0]
 *     - computeAlphaPairSource: α = rz/pAp, -α (combined)
 *     - scalarDivSource / scalarNegDivSource: c = a/b, c = -(a/b)
 *     - copyScalarSource: dst[0] = src[0]
 *
 *   Preconditioner:
 *     - preconditionerSource: Scalar Jacobi z = r/diag
 *     - blockPreconditionerSource: Block Jacobi z = M⁻¹·r (3×3 per node)
 *
 *   Boundary conditions:
 *     - applyBCSource: y[constrained] = x[constrained]
 *
 *   Reduction:
 *     - dotProductSource: Multi-workgroup partial sums
 *     - reduceSumSource: Final reduction of partial sums
 */

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

// ── Single-dispatch dot product (1 workgroup, no reduce step needed) ──

export const dotSingleSource = /* wgsl */ `
// dot_single.wgsl
// Single-workgroup dot product for arrays up to ~65k elements.
// Each of 256 threads accumulates n/256 products, then tree-reduce.
// Result is written directly to output[0].
// MUST be dispatched with exactly 1 workgroup.

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> vecA: array<f32>;
@group(0) @binding(1) var<storage, read> vecB: array<f32>;
@group(0) @binding(2) var<storage, read_write> outResult: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sharedData: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let local_id = lid.x;
    let count = params.n;
    let stride = 256u;

    // Each thread accumulates its share of the dot product
    var sum = 0.0;
    var idx = local_id;
    loop {
        if (idx >= count) {
            break;
        }
        sum = sum + vecA[idx] * vecB[idx];
        idx = idx + stride;
    }
    sharedData[local_id] = sum;

    workgroupBarrier();

    // Tree reduction
    if (local_id < 128u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 128u]; }
    workgroupBarrier();
    if (local_id < 64u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 64u]; }
    workgroupBarrier();
    if (local_id < 32u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 32u]; }
    workgroupBarrier();
    if (local_id < 16u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 16u]; }
    workgroupBarrier();
    if (local_id < 8u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 8u]; }
    workgroupBarrier();
    if (local_id < 4u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 4u]; }
    workgroupBarrier();
    if (local_id < 2u) { sharedData[local_id] = sharedData[local_id] + sharedData[local_id + 2u]; }
    workgroupBarrier();

    if (local_id == 0u) {
        outResult[0] = sharedData[0] + sharedData[1];
    }
}
`;

// ── Combined alpha computation (saves one dispatch per iteration) ──

export const computeAlphaPairSource = /* wgsl */ `
// compute_alpha_pair.wgsl
// Compute alpha = rz/pAp and neg_alpha = -(rz/pAp) in a single dispatch.

@group(0) @binding(0) var<storage, read> rzIn: array<f32>;
@group(0) @binding(1) var<storage, read> pApIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> alphaOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> negAlphaOut: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let denom = pApIn[0];
    if (abs(denom) < 1e-30) {
        alphaOut[0] = 0.0;
        negAlphaOut[0] = 0.0;
    } else {
        let val = rzIn[0] / denom;
        alphaOut[0] = val;
        negAlphaOut[0] = -val;
    }
}
`;

// ── Scalar operations (keep alpha/beta/rz on GPU, avoid CPU readback) ──

export const scalarDivSource = /* wgsl */ `
// scalar_div.wgsl - c[0] = a[0] / b[0]
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let denom = b[0];
    if (abs(denom) < 1e-30) {
        c[0] = 0.0;
    } else {
        c[0] = a[0] / denom;
    }
}
`;

export const scalarNegDivSource = /* wgsl */ `
// scalar_neg_div.wgsl - c[0] = -(a[0] / b[0])
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let denom = b[0];
    if (abs(denom) < 1e-30) {
        c[0] = 0.0;
    } else {
        c[0] = -(a[0] / denom);
    }
}
`;

export const axpyBufSource = /* wgsl */ `
// axpy_buf.wgsl - y[i] += alpha[0] * x[i], alpha read from buffer
struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> alpha: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        y[i] = y[i] + alpha[0] * x[i];
    }
}
`;

export const updatePBufSource = /* wgsl */ `
// update_p_buf.wgsl - p[i] = z[i] + beta[0] * p[i], beta read from buffer
struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.n) {
        p[i] = z[i] + beta[0] * p[i];
    }
}
`;

export const copyScalarSource = /* wgsl */ `
// copy_scalar.wgsl - dst[0] = src[0]
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = src[0];
}
`;

// ── CSR SpMV (single dispatch replaces element-by-element K·p) ──

export const spmvCSRSource = /* wgsl */ `
// spmv_csr.wgsl
// Sparse Matrix-Vector product: y = A * x  (CSR format)
// One thread per row. Single dispatch replaces 6+ color dispatches.

struct Params {
    nRows: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> rowPtr: array<u32>;
@group(0) @binding(1) var<storage, read> colIdx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> xVec: array<f32>;
@group(0) @binding(4) var<storage, read_write> yVec: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.nRows) {
        return;
    }

    let start = rowPtr[row];
    let end = rowPtr[row + 1u];

    var sum = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        sum = sum + vals[j] * xVec[colIdx[j]];
    }
    yVec[row] = sum;
}
`;

// ── Pre-computed K·p (reads cached Ke instead of recomputing) ──

export const applyKPrecomputedQ4Source = /* wgsl */ `
// apply_k_precomputed_q4.wgsl
// K·p using pre-computed element stiffness matrices (Q4: 12×12).
// Eliminates Gauss quadrature + B-matrix computation from hot loop.

struct Params {
    color_offset: u32,
    color_count: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> elements: array<u32>;
@group(0) @binding(1) var<storage, read> element_indices: array<u32>;
@group(0) @binding(2) var<storage, read> ke_all: array<f32>;
@group(0) @binding(3) var<storage, read> xvec: array<f32>;
@group(0) @binding(4) var<storage, read_write> yvec: array<f32>;
@group(0) @binding(5) var<storage, read> constrained: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }

    let elem_idx = element_indices[params.color_offset + thread_idx];

    // Read 4 node indices
    let ebase = elem_idx * 4u;
    let n0 = elements[ebase + 0u];
    let n1 = elements[ebase + 1u];
    let n2 = elements[ebase + 2u];
    let n3 = elements[ebase + 3u];

    // Read 12 DOFs of local x
    var xl: array<f32, 12>;
    xl[0]  = xvec[n0 * 3u + 0u]; xl[1]  = xvec[n0 * 3u + 1u]; xl[2]  = xvec[n0 * 3u + 2u];
    xl[3]  = xvec[n1 * 3u + 0u]; xl[4]  = xvec[n1 * 3u + 1u]; xl[5]  = xvec[n1 * 3u + 2u];
    xl[6]  = xvec[n2 * 3u + 0u]; xl[7]  = xvec[n2 * 3u + 1u]; xl[8]  = xvec[n2 * 3u + 2u];
    xl[9]  = xvec[n3 * 3u + 0u]; xl[10] = xvec[n3 * 3u + 1u]; xl[11] = xvec[n3 * 3u + 2u];

    // Ke × x_local  (12×12 MatVec)
    let ko = elem_idx * 144u;
    var yl: array<f32, 12>;
    for (var i = 0u; i < 12u; i = i + 1u) {
        let ro = ko + i * 12u;
        yl[i] = ke_all[ro +  0u] * xl[0]  + ke_all[ro +  1u] * xl[1]  + ke_all[ro +  2u] * xl[2]
              + ke_all[ro +  3u] * xl[3]  + ke_all[ro +  4u] * xl[4]  + ke_all[ro +  5u] * xl[5]
              + ke_all[ro +  6u] * xl[6]  + ke_all[ro +  7u] * xl[7]  + ke_all[ro +  8u] * xl[8]
              + ke_all[ro +  9u] * xl[9]  + ke_all[ro + 10u] * xl[10] + ke_all[ro + 11u] * xl[11];
    }

    // Scatter to global y
    if (constrained[n0 * 3u + 0u] == 0u) { yvec[n0 * 3u + 0u] += yl[0]; }
    if (constrained[n0 * 3u + 1u] == 0u) { yvec[n0 * 3u + 1u] += yl[1]; }
    if (constrained[n0 * 3u + 2u] == 0u) { yvec[n0 * 3u + 2u] += yl[2]; }
    if (constrained[n1 * 3u + 0u] == 0u) { yvec[n1 * 3u + 0u] += yl[3]; }
    if (constrained[n1 * 3u + 1u] == 0u) { yvec[n1 * 3u + 1u] += yl[4]; }
    if (constrained[n1 * 3u + 2u] == 0u) { yvec[n1 * 3u + 2u] += yl[5]; }
    if (constrained[n2 * 3u + 0u] == 0u) { yvec[n2 * 3u + 0u] += yl[6]; }
    if (constrained[n2 * 3u + 1u] == 0u) { yvec[n2 * 3u + 1u] += yl[7]; }
    if (constrained[n2 * 3u + 2u] == 0u) { yvec[n2 * 3u + 2u] += yl[8]; }
    if (constrained[n3 * 3u + 0u] == 0u) { yvec[n3 * 3u + 0u] += yl[9]; }
    if (constrained[n3 * 3u + 1u] == 0u) { yvec[n3 * 3u + 1u] += yl[10]; }
    if (constrained[n3 * 3u + 2u] == 0u) { yvec[n3 * 3u + 2u] += yl[11]; }
}
`;

export const applyKPrecomputedDKTSource = /* wgsl */ `
// apply_k_precomputed_dkt.wgsl
// K·p using pre-computed element stiffness matrices (DKT: 9×9).

struct Params {
    color_offset: u32,
    color_count: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> elements: array<u32>;
@group(0) @binding(1) var<storage, read> element_indices: array<u32>;
@group(0) @binding(2) var<storage, read> ke_all: array<f32>;
@group(0) @binding(3) var<storage, read> xvec: array<f32>;
@group(0) @binding(4) var<storage, read_write> yvec: array<f32>;
@group(0) @binding(5) var<storage, read> constrained: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x;
    if (thread_idx >= params.color_count) {
        return;
    }

    let elem_idx = element_indices[params.color_offset + thread_idx];

    let ebase = elem_idx * 3u;
    let n0 = elements[ebase + 0u];
    let n1 = elements[ebase + 1u];
    let n2 = elements[ebase + 2u];

    var xl: array<f32, 9>;
    xl[0] = xvec[n0 * 3u + 0u]; xl[1] = xvec[n0 * 3u + 1u]; xl[2] = xvec[n0 * 3u + 2u];
    xl[3] = xvec[n1 * 3u + 0u]; xl[4] = xvec[n1 * 3u + 1u]; xl[5] = xvec[n1 * 3u + 2u];
    xl[6] = xvec[n2 * 3u + 0u]; xl[7] = xvec[n2 * 3u + 1u]; xl[8] = xvec[n2 * 3u + 2u];

    let ko = elem_idx * 81u;
    var yl: array<f32, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let ro = ko + i * 9u;
        yl[i] = ke_all[ro + 0u] * xl[0] + ke_all[ro + 1u] * xl[1] + ke_all[ro + 2u] * xl[2]
              + ke_all[ro + 3u] * xl[3] + ke_all[ro + 4u] * xl[4] + ke_all[ro + 5u] * xl[5]
              + ke_all[ro + 6u] * xl[6] + ke_all[ro + 7u] * xl[7] + ke_all[ro + 8u] * xl[8];
    }

    if (constrained[n0 * 3u + 0u] == 0u) { yvec[n0 * 3u + 0u] += yl[0]; }
    if (constrained[n0 * 3u + 1u] == 0u) { yvec[n0 * 3u + 1u] += yl[1]; }
    if (constrained[n0 * 3u + 2u] == 0u) { yvec[n0 * 3u + 2u] += yl[2]; }
    if (constrained[n1 * 3u + 0u] == 0u) { yvec[n1 * 3u + 0u] += yl[3]; }
    if (constrained[n1 * 3u + 1u] == 0u) { yvec[n1 * 3u + 1u] += yl[4]; }
    if (constrained[n1 * 3u + 2u] == 0u) { yvec[n1 * 3u + 2u] += yl[5]; }
    if (constrained[n2 * 3u + 0u] == 0u) { yvec[n2 * 3u + 0u] += yl[6]; }
    if (constrained[n2 * 3u + 1u] == 0u) { yvec[n2 * 3u + 1u] += yl[7]; }
    if (constrained[n2 * 3u + 2u] == 0u) { yvec[n2 * 3u + 2u] += yl[8]; }
}
`;

// Import the larger apply_k shaders from separate files (they're too large to inline comfortably)
import { applyKQ4Source } from './apply_k_q4_source';
import { applyKDKTSource } from './apply_k_dkt_source';

export { applyKQ4Source, applyKDKTSource };

