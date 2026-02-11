/**
 * GPU compute shader for averaging accumulated moment contributions.
 *
 * After the per-color moment accumulation passes, this shader divides
 * each node's moment sum by the contribution count to produce the
 * final nodal-averaged moments.
 */

export const averageMomentsSource = /* wgsl */ `
// average_moments.wgsl
// Mx[i] /= count[i], My[i] /= count[i], Mxy[i] /= count[i]

struct Params {
    nodeCount: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> Mx: array<f32>;
@group(0) @binding(1) var<storage, read_write> My: array<f32>;
@group(0) @binding(2) var<storage, read_write> Mxy: array<f32>;
@group(0) @binding(3) var<storage, read> count: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.nodeCount) {
        return;
    }

    let c = count[i];
    if (c > 0.0) {
        let invC = 1.0 / c;
        Mx[i]  = Mx[i] * invC;
        My[i]  = My[i] * invC;
        Mxy[i] = Mxy[i] * invC;
    }
}
`;

