# 🏆 WebGPU Plate Solver Challenge

> **First to achieve 60k DOF solve in under 20ms wins $1,000!**

## The Challenge

We have a finite element plate solver that works, but it's **slow**. The current GPU implementation takes ~18,000ms for a 60k DOF problem. We need it to run in **under 20ms**.

That's a **900x speedup**. Think you can do it?

## Current Performance

| Metric | Current | Target |
|--------|---------|--------|
| 60k DOF Solve Time | ~18,000ms | **<20ms** |
| PCG Iterations | ~1000 | <100 (with better preconditioner) |
| GPU Utilization | Low (sync bottleneck) | High |

## The Problem

The bottleneck is **GPU-CPU synchronization**. The current implementation does 3 GPU readbacks per PCG iteration:
- Dot product (r·z)
- Dot product (p·Ap) 
- Convergence check (||r||)

With 1000 iterations, that's **3000 round trips** to the GPU. Each readback costs ~15ms.

## Prize Structure

- **$1,000** - First person to achieve <20ms for 60k DOF
- **$3,000** - First person to achieve <10ms for 60k DOF
- Results must be reproducible and validated against CPU reference

## Browser Requirements

WebGPU is required. Supported browsers:
- ✅ **Chrome 113+** (recommended)
- ✅ **Edge 113+**
- ⚠️ **Firefox Nightly** (enable `dom.webgpu.enabled` in about:config)
- ❌ Safari (not yet supported)

## Quick Start

```bash
# Clone this repo
git clone https://github.com/cseifert512/gpu-gauntlet

# Install dependencies
cd gpu-gauntlet
npm install

# Run development server
npm run dev

# Open http://localhost:3000/benchmark in Chrome
```

## Sample Output

When you run the benchmark, you'll see something like:

```
DOF: 1,089   | CPU: 45.2ms   | GPU: 312.5ms   | Valid: ✓ PASS (0.0001%)
DOF: 10,201  | CPU: 892.1ms  | GPU: 2,847.3ms | Valid: ✓ PASS (0.0003%)
DOF: 30,625  | CPU: 4,521ms  | GPU: 8,932.1ms | Valid: ✓ PASS (0.0002%)
DOF: 61,009  | CPU: 9,847ms  | GPU: 17,892ms  | Valid: ✓ PASS (0.0004%)  ← TARGET
```

**Your goal**: Get that last GPU time under 20ms! 🎯

## Troubleshooting

**"WebGPU not supported"**
- Use Chrome 113+ or Edge 113+
- Check `chrome://gpu` to verify WebGPU is enabled
- Update your GPU drivers

**"GPU FAILED" in benchmark**
- Open browser DevTools (F12) → Console for error details
- Common issue: WGSL shader compilation errors
- Try refreshing the page

**Build errors after modifying shaders**
- Clear Next.js cache: `rm -rf .next && npm run dev`
- Check WGSL syntax - `shared` is a reserved keyword!

## Rules

1. **You CAN modify**: Everything in `src/lib/plate/gpu/`
2. **You CANNOT modify**: `src/lib/plate/solver.ts` (CPU reference)
3. **Validation**: GPU results must match CPU within 0.01% relative error
4. **Hardware**: Must work on standard consumer GPUs (no exotic requirements)

See [RULES.md](./RULES.md) for complete competition rules.

## Architecture

The solver uses Preconditioned Conjugate Gradient (PCG) with matrix-free operations:

```
For each iteration:
  1. Ap = K·p        (matrix-vector multiply - GPU parallel)
  2. α = (r·z)/(p·Ap)  (two dot products - reduction)
  3. x += α·p        (vector update - GPU parallel)
  4. r -= α·Ap       (vector update - GPU parallel)
  5. Check ||r||     (norm - reduction)
  6. z = M⁻¹·r       (preconditioner - GPU parallel)
  7. β = (r·z)/(r_old·z_old)
  8. p = z + β·p     (vector update - GPU parallel)
```

The GPU operations (1, 3, 4, 6, 8) are fast. The reductions (2, 5, 7) require sync.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed technical documentation.

## Optimization Ideas

Here are some approaches that might help:

### Reduce Sync Overhead
- Batch multiple iterations before reading back results
- Use persistent kernels that loop on GPU
- Implement dot product via atomics + single readback

### Improve Convergence  
- Better preconditioner (Incomplete Cholesky, AMG)
- Multigrid methods
- Deflation for low-frequency modes

### Reduce Work Per Iteration
- Cache and reuse intermediate computations
- Use lower precision (fp16) where acceptable
- Exploit matrix sparsity structure

## File Structure

```
src/lib/plate/
├── types.ts          # Type definitions
├── element.ts        # Element stiffness matrices
├── mesher.ts         # Mesh generation
├── coloring.ts       # Element coloring for parallel assembly
├── pcg.ts            # PCG algorithm
├── solver.ts         # CPU REFERENCE (DO NOT MODIFY)
├── postprocess.ts    # Post-processing
│
└── gpu/              # ← OPTIMIZATION TARGET
    ├── context.ts    # WebGPU initialization
    ├── buffers.ts    # GPU buffer management
    ├── pipelines.ts  # Compute pipelines
    ├── solver.ts     # Main GPU solver
    └── shaders/      # WGSL compute shaders
        ├── apply_k_q4.wgsl
        ├── dot_product.wgsl
        ├── axpy.wgsl
        └── ...
```

## Submitting Your Solution

1. Fork this repository
2. Implement your optimization
3. Run the benchmark and verify validation passes
4. Create a Pull Request with:
   - Your benchmark results (screenshot or copy-paste)
   - Brief description of your approach
   - Hardware specs (GPU model)

## Leaderboard

| Rank | Contributor | Time (60k DOF) | Date | Approach |
|------|-------------|----------------|------|----------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |

*Be the first to claim a spot!*

## Questions?

Open an issue in this repository. Good luck! 🚀

## License

MIT

