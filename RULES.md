# Competition Rules

## Eligibility

- Anyone can participate
- No geographic restrictions
- Teams allowed (prize shared among team members)

## Target

**Goal**: Solve a 60,000 DOF plate bending problem in under 20 milliseconds using WebGPU.

**Validation**: Results must match the CPU reference implementation within 0.01% relative error for the maximum displacement.

## What You Can Modify

### ✅ Allowed Modifications

Everything in `src/lib/plate/gpu/`:
- `context.ts` - WebGPU initialization
- `buffers.ts` - Buffer management
- `pipelines.ts` - Compute pipelines
- `solver.ts` - Main GPU solver logic
- `shaders/*.wgsl` - All WGSL compute shaders
- `shaders/index.ts` - Shader exports

You may also:
- Add new files to the `gpu/` directory
- Add new shaders
- Add new npm dependencies (but consider bundle size)
- Modify the benchmark page for testing purposes

### ❌ Not Allowed

**DO NOT MODIFY** these files:
- `src/lib/plate/solver.ts` - CPU reference implementation
- `src/lib/plate/pcg.ts` - PCG algorithm (CPU)
- `src/lib/plate/element.ts` - Element formulations
- `src/lib/plate/mesher.ts` - Mesh generation
- `src/lib/plate/types.ts` - Type definitions

Modifying these files will disqualify your submission.

## Validation Requirements

1. **Accuracy**: Maximum displacement must match CPU within 0.01% relative error
   ```
   |gpu_result - cpu_result| / |cpu_result| < 0.0001
   ```

2. **Convergence**: Solver must converge (residual below tolerance)

3. **Reproducibility**: Results must be consistent across runs (within floating-point tolerance)

## Hardware Requirements

Your solution must work on:
- Standard consumer GPUs (NVIDIA GTX/RTX, AMD RX, Intel Arc)
- WebGPU-capable browsers (Chrome 113+, Edge 113+, Firefox Nightly)
- No requirements for exotic hardware features

## Submission Process

1. **Fork** this repository
2. **Implement** your optimization in the `gpu/` folder
3. **Test** using the benchmark page at `/benchmark`
4. **Verify** validation passes (GPU matches CPU)
5. **Create Pull Request** with:
   - Screenshot or copy-paste of benchmark results
   - Brief description of your approach (can be detailed, we're curious!)
   - Your hardware specs (GPU model, browser version)

## Judging Criteria

1. **Performance**: Time to solve 60k DOF problem
2. **Correctness**: Validation must pass
3. **Reproducibility**: Must work on reviewer's hardware
4. **Code Quality**: Clean, readable code preferred (not required for prize)

## Prize Details

### First Place: $500

- First valid submission achieving <20ms for 60k DOF
- Must pass validation
- Must be reproducible on at least two different GPUs

### Claiming the Prize

1. Your PR is reviewed and merged
2. We verify the results on our test hardware
3. Payment via PayPal, Venmo, or GitHub Sponsors

## Timeline

- **Start**: Now
- **End**: When someone wins (or we decide to close the challenge)
- **Updates**: Check this repo for rule clarifications

## Questions & Disputes

- Open an issue for questions
- Maintainers' decisions are final
- Rules may be clarified (not changed to invalidate existing work)

## Code of Conduct

- Be respectful in issues and PRs
- Don't disparage other participants
- Help others learn (optional but appreciated)

## Disclaimer

- Prize is offered in good faith
- We reserve the right to modify rules if loopholes are found
- This is a personal challenge, not backed by any company

---

Good luck! May the fastest solver win! 🏆

