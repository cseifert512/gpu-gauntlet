# Architecture Deep Dive

This document provides a technical deep-dive into the plate solver for competitors.

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Finite Element Formulation](#finite-element-formulation)
3. [PCG Solver](#pcg-solver)
4. [GPU Implementation](#gpu-implementation)
5. [Known Bottlenecks](#known-bottlenecks)
6. [Optimization Opportunities](#optimization-opportunities)

## Problem Overview

We're solving the plate bending equation using the Finite Element Method (FEM). Given:
- A plate geometry (2D domain)
- Material properties (E, ν, thickness)
- Boundary conditions (supports)
- Applied loads

We solve for the displacement field: vertical deflection `w` and rotations `θx`, `θy` at each node.

### DOF Structure

Each node has 3 degrees of freedom (DOF):
- `w`: vertical displacement
- `θx`: rotation about x-axis  
- `θy`: rotation about y-axis

A 60,000 DOF problem has 20,000 nodes.

## Finite Element Formulation

### Element Types

**Q4 (Quad)** - 4-node quadrilateral
- 4 nodes × 3 DOF = 12 DOF per element
- Uses Mindlin plate theory (includes shear deformation)
- 2×2 Gauss integration for bending, 1-point for shear

**DKT (Triangle)** - 3-node triangle
- 3 nodes × 3 DOF = 9 DOF per element
- Discrete Kirchhoff Triangle formulation
- Better for unstructured meshes

### Element Stiffness Matrix

For each element, we compute a local stiffness matrix `Ke` (12×12 for Q4, 9×9 for DKT).

```typescript
// From element.ts
Ke = computeElementStiffness(nodeCoords, material);
```

The global system `K·u = F` is assembled from all elements, but we **never form K explicitly**. Instead, we use matrix-free operations.

### Matrix-Free K·x

To compute `y = K·x`:

```
for each element e:
  xLocal = gather(x, element_nodes[e])  // 12 local DOFs
  yLocal = Ke · xLocal                   // 12×12 × 12 = 12
  scatter_add(y, element_nodes[e], yLocal)
```

The scatter-add has conflicts if two elements share nodes. We use **element coloring** to process non-conflicting elements in parallel.

### Element Coloring

For structured quad meshes, a 4-color checkerboard pattern ensures no two adjacent elements (sharing a node) have the same color:

```
+---+---+---+---+
| 2 | 3 | 2 | 3 |
+---+---+---+---+
| 0 | 1 | 0 | 1 |
+---+---+---+---+
| 2 | 3 | 2 | 3 |
+---+---+---+---+
```

All elements of one color can be processed in parallel without race conditions.

## PCG Solver

### Algorithm

Preconditioned Conjugate Gradient (PCG) solves `K·u = F`:

```
r = F - K·u          // residual (u starts at 0, so r = F)
z = M⁻¹·r            // apply preconditioner
p = z                // search direction
ρ = r·z              // dot product

for iter = 1 to maxIter:
  q = K·p            // matrix-vector multiply
  α = ρ / (p·q)      // step size
  u += α·p           // update solution
  r -= α·q           // update residual
  
  if ||r|| < tol:    // convergence check
    return u
  
  z = M⁻¹·r          // apply preconditioner
  ρ_new = r·z
  β = ρ_new / ρ
  p = z + β·p        // update search direction
  ρ = ρ_new
```

### Preconditioner

The preconditioner `M` approximates `K⁻¹` cheaply. Better preconditioners = fewer iterations.

**Current: Block Jacobi**
- Extract 3×3 diagonal blocks (one per node)
- Invert each 3×3 block
- Apply as: `z[node] = M_node⁻¹ · r[node]`

This is better than scalar Jacobi (diagonal only) but still requires ~1000 iterations.

**Better options:**
- Incomplete Cholesky (ILU)
- Algebraic Multigrid (AMG)
- Deflation for low-frequency modes

## GPU Implementation

### File Structure

```
gpu/
├── context.ts    # WebGPU device initialization
├── buffers.ts    # GPU buffer allocation
├── pipelines.ts  # Compute pipeline creation
├── solver.ts     # Main solve orchestration
└── shaders/      # WGSL compute kernels
```

### Buffer Layout

```typescript
interface PlateGPUBuffers {
  // Mesh data (read-only)
  nodes: GPUBuffer;        // [x,y] per node
  elements: GPUBuffer;     // node indices per element
  colorOffsets: GPUBuffer; // element indices by color
  
  // Material data (read-only)
  material: GPUBuffer;     // [E, nu, t]
  diagonal: GPUBuffer;     // preconditioner diagonal
  blockDiagInv: GPUBuffer; // inverted 3x3 blocks
  
  // Working vectors
  x: GPUBuffer;   // solution
  r: GPUBuffer;   // residual
  z: GPUBuffer;   // preconditioned residual
  p: GPUBuffer;   // search direction
  Ap: GPUBuffer;  // K·p result
  
  // Reduction results
  dots: GPUBuffer; // dot product partial results
}
```

### Compute Pipelines

Each PCG operation has a dedicated compute shader:

| Operation | Shader | Workgroup Size | Notes |
|-----------|--------|----------------|-------|
| K·p | apply_k_q4.wgsl | 64 | One workgroup per element color batch |
| α·x + y | axpy.wgsl | 256 | Simple vector update |
| x·y | dot_product.wgsl | 256 | Parallel reduction |
| M⁻¹·r | preconditioner.wgsl | 256 | Block diagonal multiply |
| Copy | copy.wgsl | 256 | Buffer copy |
| Zero | zero_buffer.wgsl | 256 | Clear buffer |

### Current Flow

```typescript
async function solveGPU(mesh, material, loads, options) {
  // 1. Initialize GPU
  const ctx = await initGPU();
  const pipelines = await createPipelines(ctx);
  const buffers = createPlateBuffers(ctx, mesh, ...);
  
  // 2. Upload data to GPU
  uploadMesh(buffers, mesh);
  uploadMaterial(buffers, material);
  uploadLoads(buffers, loads);
  
  // 3. PCG loop
  for (let iter = 0; iter < maxIter; iter++) {
    // Ap = K·p (GPU)
    dispatchApplyK(encoder, pipelines, buffers);
    
    // pAp = p·Ap (GPU + CPU readback)
    const pAp = await computeDotProduct(buffers.p, buffers.Ap);  // ← SYNC
    
    // α = rz / pAp
    const alpha = rz / pAp;
    
    // x += α·p, r -= α·Ap (GPU)
    dispatchAxpy(encoder, alpha, buffers.p, buffers.x);
    dispatchAxpy(encoder, -alpha, buffers.Ap, buffers.r);
    
    // ||r|| (GPU + CPU readback)
    const residual = await computeNorm(buffers.r);  // ← SYNC
    
    if (residual < tol) break;
    
    // z = M⁻¹·r (GPU)
    dispatchPreconditioner(encoder, buffers);
    
    // rz_new = r·z (GPU + CPU readback)
    const rz_new = await computeDotProduct(buffers.r, buffers.z);  // ← SYNC
    
    // β = rz_new / rz
    const beta = rz_new / rz;
    
    // p = z + β·p (GPU)
    dispatchUpdateP(encoder, beta, buffers.z, buffers.p);
    
    rz = rz_new;
  }
  
  // 4. Download results
  return await downloadSolution(buffers.x);
}
```

## Known Bottlenecks

### 1. GPU-CPU Synchronization (Critical)

Each `await computeDotProduct()` call:
1. Submits GPU command buffer
2. Waits for GPU completion
3. Maps GPU buffer to CPU
4. Reads result
5. Unmaps buffer

This takes **~5-15ms per call** due to:
- Command buffer submission overhead
- GPU-CPU synchronization latency
- Buffer mapping overhead

With 3 syncs per iteration × 1000 iterations = **3000 syncs** = **15-45 seconds** just in sync overhead!

### 2. PCG Convergence

The block Jacobi preconditioner is weak for plate bending. The condition number is high, requiring ~1000 iterations.

Better preconditioners could reduce this to 50-100 iterations, which would:
- Reduce sync count by 10-20×
- Reduce total computation

### 3. Element Stiffness Computation

Currently, element stiffness matrices are:
- Cached on CPU (good)
- Recomputed implicitly on GPU (wasteful)

Precomputing and uploading Ke matrices could save GPU cycles.

## Optimization Opportunities

### Level 1: Reduce Sync Frequency

**Batched Iterations**
Run multiple PCG iterations on GPU before checking convergence:
```
for (let batch = 0; batch < maxIter / BATCH_SIZE; batch++) {
  // Run BATCH_SIZE iterations entirely on GPU
  for (let i = 0; i < BATCH_SIZE; i++) {
    // All operations on GPU, no readback
  }
  // Single readback to check convergence
  const residual = await readResidual();
  if (residual < tol) break;
}
```

**Atomic Reductions**
Use atomic operations for dot products:
```wgsl
@group(0) @binding(0) var<storage, read_write> result: atomic<f32>;

@compute @workgroup_size(256)
fn dot_atomic(@builtin(global_invocation_id) gid: vec3<u32>) {
  let val = a[gid.x] * b[gid.x];
  atomicAdd(&result, val);
}
```

### Level 2: Better Preconditioner

**Incomplete Cholesky**
Factor K ≈ LLᵀ approximately, use L⁻¹L⁻ᵀ as preconditioner.

**Multigrid**
Coarse grid correction dramatically improves convergence for elliptic problems.

**Deflation**
Remove low-frequency error modes explicitly.

### Level 3: Algorithm Changes

**Pipelined PCG**
Overlap computation and communication by restructuring the algorithm.

**s-step CG**
Compute multiple iterations' worth of basis vectors before orthogonalizing.

**Mixed Precision**
Use fp16 for most operations, fp32 only where needed.

### Level 4: Hardware-Specific

**Persistent Kernels**
Keep the GPU busy with a single long-running kernel that loops.

**Shared Memory**
Use workgroup shared memory for reductions within a workgroup.

**Occupancy Optimization**
Tune workgroup sizes for the target GPU architecture.

## Theoretical Limits

**Memory Bandwidth Bound**
60k DOF × 4 bytes × 8 vectors ≈ 2 MB per iteration
At 500 GB/s bandwidth: 0.004 ms per iteration
1000 iterations: 4 ms

**Compute Bound**
Matrix-vector multiply: 60k × 12 × 12 operations ≈ 8.6M FLOPs per iteration
At 10 TFLOPS: 0.0009 ms per iteration

So theoretically, **sub-20ms is definitely achievable** if we can eliminate the sync overhead!

## Reference Material

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [PCG Algorithm](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [Multigrid Methods](https://en.wikipedia.org/wiki/Multigrid_method)
- [Batoz DKT Element](https://doi.org/10.1002/nme.1620150513)

Good luck! 🚀

