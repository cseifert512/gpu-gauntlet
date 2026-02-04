'use client';

import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* Hero Section */}
      <header className="text-center mb-16">
        <div className="inline-block mb-4 px-4 py-1 rounded-full border border-[var(--color-primary)] text-[var(--color-primary)] text-sm">
          🏆 $500 Prize Pool
        </div>
        
        <h1 className="text-5xl md:text-7xl font-bold mb-6">
          <span className="glow-green text-[var(--color-primary)]">WebGPU</span>
          <br />
          <span className="text-white">Solver Challenge</span>
        </h1>
        
        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
          Can you solve 60,000 DOF in under 20ms?
          <br />
          <span className="text-[var(--color-primary)]">900× speedup needed.</span>
        </p>
        
        <div className="flex gap-4 justify-center">
          <Link href="/benchmark" className="btn-primary text-lg px-8 py-3">
            Run Benchmark →
          </Link>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn-secondary text-lg px-8 py-3"
          >
            View Source
          </a>
        </div>
      </header>

      {/* Stats Section */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
        <div className="card text-center">
          <div className="text-4xl font-bold text-[var(--color-error)] mb-2">~18,000ms</div>
          <div className="text-gray-400">Current Performance</div>
        </div>
        <div className="card text-center box-glow-green">
          <div className="text-4xl font-bold text-[var(--color-primary)] mb-2">&lt;20ms</div>
          <div className="text-gray-400">Target Performance</div>
        </div>
        <div className="card text-center">
          <div className="text-4xl font-bold text-[var(--color-secondary)] mb-2">900×</div>
          <div className="text-gray-400">Speedup Required</div>
        </div>
      </section>

      {/* The Problem */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-6 text-white">The Problem</h2>
        <div className="card">
          <p className="text-gray-300 mb-4">
            We have a finite element plate solver that uses WebGPU for acceleration.
            The math is correct, the results validate, but it's <span className="text-[var(--color-error)]">painfully slow</span>.
          </p>
          <p className="text-gray-300 mb-4">
            The bottleneck is <span className="text-[var(--color-primary)]">GPU-CPU synchronization</span>.
            The PCG solver needs dot products for convergence checks, and each readback
            costs ~15ms. With 1000 iterations, that's 45 seconds of just waiting.
          </p>
          <pre className="text-sm">
{`// Current flow (3 syncs per iteration!)
for (let iter = 0; iter < 1000; iter++) {
  dispatchGPU(K·p);           // Fast
  await readDotProduct(p, Ap); // SLOW - 15ms sync
  dispatchGPU(axpy);          // Fast  
  await readNorm(r);          // SLOW - 15ms sync
  dispatchGPU(precondition);  // Fast
  await readDotProduct(r, z);  // SLOW - 15ms sync
}`}
          </pre>
        </div>
      </section>

      {/* What You Can Do */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-6 text-white">Optimization Ideas</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card card-hover">
            <h3 className="text-xl font-bold text-[var(--color-primary)] mb-3">🔄 Batch Iterations</h3>
            <p className="text-gray-400">
              Run multiple PCG iterations on GPU before checking convergence.
              Trade a few extra iterations for massively reduced sync overhead.
            </p>
          </div>
          <div className="card card-hover">
            <h3 className="text-xl font-bold text-[var(--color-secondary)] mb-3">⚡ Better Preconditioner</h3>
            <p className="text-gray-400">
              Current block Jacobi needs ~1000 iterations. Multigrid or ILU
              could get this down to ~100, reducing total syncs by 10×.
            </p>
          </div>
          <div className="card card-hover">
            <h3 className="text-xl font-bold text-[var(--color-warning)] mb-3">🧮 Atomic Reductions</h3>
            <p className="text-gray-400">
              Use atomics for dot products to avoid multi-pass reductions.
              Single kernel, single readback.
            </p>
          </div>
          <div className="card card-hover">
            <h3 className="text-xl font-bold text-purple-400 mb-3">🔁 Persistent Kernels</h3>
            <p className="text-gray-400">
              Keep the GPU busy with a single long-running kernel that
              loops internally. Zero sync until final result.
            </p>
          </div>
        </div>
      </section>

      {/* Rules Summary */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-6 text-white">Rules</h2>
        <div className="card">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-bold text-[var(--color-primary)] mb-3">✅ You CAN Modify</h3>
              <ul className="space-y-2 text-gray-300">
                <li><code>src/lib/plate/gpu/*</code></li>
                <li>All WGSL shaders</li>
                <li>GPU buffer management</li>
                <li>Solver orchestration</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-bold text-[var(--color-error)] mb-3">❌ Cannot Modify</h3>
              <ul className="space-y-2 text-gray-300">
                <li><code>src/lib/plate/solver.ts</code> (CPU reference)</li>
                <li>Element formulations</li>
                <li>Mesh generation</li>
                <li>Type definitions</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Leaderboard */}
      <section className="mb-16">
        <h2 className="text-3xl font-bold mb-6 text-white">Leaderboard</h2>
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Contributor</th>
                <th>Time (60k DOF)</th>
                <th>Approach</th>
              </tr>
            </thead>
            <tbody>
              <tr className="text-gray-500">
                <td>1</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
              </tr>
              <tr className="text-gray-500">
                <td>2</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
              </tr>
              <tr className="text-gray-500">
                <td>3</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <p className="text-center text-gray-500 mt-6 text-sm">
            Be the first to claim a spot!
          </p>
        </div>
      </section>

      {/* CTA */}
      <section className="text-center">
        <div className="card box-glow-green inline-block px-12 py-8">
          <h2 className="text-2xl font-bold mb-4 text-white">Ready to Compete?</h2>
          <p className="text-gray-400 mb-6">
            Fork the repo, run the benchmark, beat 20ms.
          </p>
          <Link href="/benchmark" className="btn-primary text-lg px-8 py-3">
            Start Benchmarking →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-16 pt-8 border-t border-[var(--color-border)] text-center text-gray-500 text-sm">
        <p>
          WebGPU Solver Challenge •{' '}
          <a href="/RULES.md" className="text-[var(--color-primary)] hover:underline">Rules</a>
          {' • '}
          <a href="/ARCHITECTURE.md" className="text-[var(--color-primary)] hover:underline">Architecture</a>
          {' • '}
          <a href="https://github.com" className="text-[var(--color-primary)] hover:underline">GitHub</a>
        </p>
      </footer>
    </div>
  );
}

