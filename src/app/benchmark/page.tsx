'use client';

import { useState, useCallback, useEffect } from 'react';
import Link from 'next/link';

interface BenchmarkResult {
  dofCount: number;
  cpuTimeMs: number;
  gpuTimeMs: number;
  cpuIterations: number;
  gpuIterations: number;
  cpuMaxW: number;
  gpuMaxW: number;
  relativeError: number;
  validationPassed: boolean;
  targetMet: boolean;
}

// Test configurations
const TEST_CONFIGS = [
  { name: 'Small (1k DOF)', meshSize: 2.0, expectedDOF: 1000 },
  { name: 'Medium (10k DOF)', meshSize: 0.7, expectedDOF: 10000 },
  { name: 'Large (30k DOF)', meshSize: 0.4, expectedDOF: 30000 },
  { name: 'Target (60k DOF)', meshSize: 0.28, expectedDOF: 60000 },
];

export default function BenchmarkPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState<string | null>(null);
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [gpuSupported, setGpuSupported] = useState<boolean | null>(null);
  const [gpuInfo, setGpuInfo] = useState<string>('Checking...');

  // Check GPU support on mount
  useEffect(() => {
    checkGPUSupport();
  }, []);

  const checkGPUSupport = async () => {
    if (!navigator.gpu) {
      setGpuSupported(false);
      setGpuInfo('WebGPU not supported in this browser');
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        setGpuSupported(false);
        setGpuInfo('No GPU adapter found');
        return;
      }

      const device = await adapter.requestDevice();
      setGpuSupported(true);
      
      // GPU detected
      setGpuInfo('WebGPU GPU detected');
      
      device.destroy();
    } catch (err) {
      setGpuSupported(false);
      setGpuInfo(`GPU init failed: ${err}`);
    }
  };

  const runBenchmark = useCallback(async () => {
    if (!gpuSupported) {
      setError('WebGPU is not supported on this device');
      return;
    }

    setIsRunning(true);
    setResults([]);
    setError(null);

    try {
      // Dynamically import the solver to avoid SSR issues
      const { 
        solvePlate,
        generateRectangularMesh,
        computeElementColoring,
        identifyConstrainedDOFs,
        computeBlockDiagonal,
        invertBlockDiagonal,
        buildLoadVector,
        applyBCsToRHS,
        DOFS_PER_NODE,
      } = await import('@/lib/plate');
      
      const { solveGPU } = await import('@/lib/plate/gpu');

      // Plate configuration
      const width = 10; // meters
      const height = 10; // meters
      const material = { E: 210e9, nu: 0.3, t: 0.01 }; // Steel, 10mm thick
      const supports = [{ type: 'pinned' as const, location: 'all_edges' as const }];
      const loads = [{ position: [5, 5] as [number, number], magnitude: 1000 }]; // 1kN center load

      for (const config of TEST_CONFIGS) {
        setCurrentTest(config.name);
        
        // Generate geometry
        const geometry = {
          boundary: new Float32Array([0, 0, width, 0, width, height, 0, height]),
          holes: [] as Float32Array[],
        };

        // --- CPU Solve ---
        const cpuStart = performance.now();
        const cpuResult = solvePlate(geometry, material, supports, loads, {
          meshSize: config.meshSize,
          tolerance: 1e-6,
          maxIterations: 1000,
        });
        const cpuTimeMs = performance.now() - cpuStart;

        const actualDOF = cpuResult.mesh.nodeCount * DOFS_PER_NODE;

        // Find max displacement (CPU)
        let cpuMaxW = 0;
        for (let i = 0; i < cpuResult.w.length; i++) {
          if (Math.abs(cpuResult.w[i]) > Math.abs(cpuMaxW)) {
            cpuMaxW = cpuResult.w[i];
          }
        }

        // --- GPU Solve ---
        // Note: GPU solver uses the same mesh and setup as CPU for fair comparison
        let gpuTimeMs = 0;
        let gpuMaxW = 0;
        let gpuIterations = 0;

        try {
          // Reuse mesh from CPU solve
          const mesh = cpuResult.mesh;
          const coloring = computeElementColoring(mesh);
          const constrainedDOFs = identifyConstrainedDOFs(mesh, supports);
          const F = buildLoadVector(mesh, loads);
          applyBCsToRHS(F, constrainedDOFs);

          const gpuStart = performance.now();
          const gpuResult = await solveGPU(mesh, material, coloring, F, constrainedDOFs, {
            tolerance: 1e-6,
            maxIterations: 1000,
          });
          gpuTimeMs = performance.now() - gpuStart;
          
          // Find max displacement (GPU) - extract w from full solution
          const gpuSolution = gpuResult.solution;
          for (let i = 0; i < mesh.nodeCount; i++) {
            const w = gpuSolution[i * 3]; // w is first DOF per node
            if (Math.abs(w) > Math.abs(gpuMaxW)) {
              gpuMaxW = w;
            }
          }
          gpuIterations = gpuResult.iterations;
        } catch (gpuErr) {
          console.error('GPU solve failed:', gpuErr);
          gpuTimeMs = -1; // Indicate failure
        }

        // Calculate validation
        const relativeError = Math.abs(cpuMaxW) > 1e-15 
          ? Math.abs(gpuMaxW - cpuMaxW) / Math.abs(cpuMaxW)
          : 0;
        const validationPassed = relativeError < 0.0001; // 0.01%
        const targetMet = gpuTimeMs > 0 && gpuTimeMs < 20 && actualDOF >= 60000;

        const result: BenchmarkResult = {
          dofCount: actualDOF,
          cpuTimeMs,
          gpuTimeMs,
          cpuIterations: cpuResult.solverInfo.iterations,
          gpuIterations,
          cpuMaxW,
          gpuMaxW,
          relativeError,
          validationPassed,
          targetMet,
        };

        setResults(prev => [...prev, result]);
      }
    } catch (err) {
      setError(`Benchmark failed: ${err}`);
    } finally {
      setIsRunning(false);
      setCurrentTest(null);
    }
  }, [gpuSupported]);

  const formatTime = (ms: number) => {
    if (ms < 0) return 'FAILED';
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatNumber = (n: number) => {
    if (Math.abs(n) < 0.001) return n.toExponential(3);
    return n.toFixed(6);
  };

  const target60kResult = results.find(r => r.dofCount >= 60000);
  const challengeWon = target60kResult?.targetMet === true;

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* Header */}
      <header className="mb-8">
        <Link href="/" className="text-[var(--color-primary)] hover:underline mb-4 inline-block">
          ← Back to Challenge
        </Link>
        <h1 className="text-4xl font-bold text-white">Benchmark Runner</h1>
        <p className="text-gray-400 mt-2">
          Test your optimization against the target: 60k DOF in &lt;20ms
        </p>
      </header>

      {/* GPU Status */}
      <section className="card mb-8">
        <h2 className="text-xl font-bold mb-4 text-white">GPU Status</h2>
        <div className="flex items-center gap-4">
          <div className={`w-3 h-3 rounded-full ${
            gpuSupported === null ? 'bg-yellow-500 animate-pulse' :
            gpuSupported ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span className="text-gray-300">{gpuInfo}</span>
        </div>
      </section>

      {/* Run Button */}
      <section className="mb-8">
        <button
          onClick={runBenchmark}
          disabled={isRunning || !gpuSupported}
          className="btn-primary text-lg px-8 py-4 w-full md:w-auto"
        >
          {isRunning ? (
            <span className="flex items-center gap-2">
              <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Running {currentTest}...
            </span>
          ) : (
            '🚀 Run Full Benchmark'
          )}
        </button>
      </section>

      {/* Error Display */}
      {error && (
        <div className="card mb-8 border-[var(--color-error)]">
          <p className="text-[var(--color-error)]">{error}</p>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4 text-white">Results</h2>
          
          {/* Summary Card */}
          {target60kResult && (
            <div className={`card mb-6 ${challengeWon ? 'box-glow-green border-[var(--color-primary)]' : ''}`}>
              <div className="text-center">
                {challengeWon ? (
                  <>
                    <div className="text-6xl mb-4">🏆</div>
                    <h3 className="text-3xl font-bold text-[var(--color-primary)] mb-2">
                      CHALLENGE COMPLETE!
                    </h3>
                    <p className="text-xl text-gray-300">
                      {formatTime(target60kResult.gpuTimeMs)} for {target60kResult.dofCount.toLocaleString()} DOF
                    </p>
                    <p className="text-gray-400 mt-4">
                      Submit a PR to claim your prize!
                    </p>
                  </>
                ) : (
                  <>
                    <h3 className="text-2xl font-bold text-white mb-2">
                      Target: 60k DOF in &lt;20ms
                    </h3>
                    <p className="text-xl">
                      <span className="text-gray-400">Current: </span>
                      <span className={target60kResult.gpuTimeMs < 0 ? 'text-red-500' : 'text-[var(--color-warning)]'}>
                        {formatTime(target60kResult.gpuTimeMs)}
                      </span>
                      <span className="text-gray-500 ml-4">
                        ({target60kResult.gpuTimeMs > 0 ? Math.ceil(target60kResult.gpuTimeMs / 20) : '∞'}× slower)
                      </span>
                    </p>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Results Table */}
          <div className="card overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>DOF Count</th>
                  <th>CPU Time</th>
                  <th>GPU Time</th>
                  <th>Speedup</th>
                  <th>GPU Iterations</th>
                  <th>Validation</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, idx) => (
                  <tr key={idx}>
                    <td className="font-mono">{result.dofCount.toLocaleString()}</td>
                    <td className="font-mono">{formatTime(result.cpuTimeMs)}</td>
                    <td className={`font-mono ${
                      result.gpuTimeMs < 0 ? 'text-red-500' :
                      result.gpuTimeMs < 20 && result.dofCount >= 60000 ? 'text-[var(--color-primary)]' :
                      'text-white'
                    }`}>
                      {formatTime(result.gpuTimeMs)}
                    </td>
                    <td className={`font-mono ${
                      result.gpuTimeMs > 0 && result.gpuTimeMs < result.cpuTimeMs 
                        ? 'text-[var(--color-primary)]' 
                        : 'text-[var(--color-error)]'
                    }`}>
                      {result.gpuTimeMs > 0 
                        ? `${(result.cpuTimeMs / result.gpuTimeMs).toFixed(1)}×`
                        : 'N/A'
                      }
                    </td>
                    <td className="font-mono">{result.gpuIterations || 'N/A'}</td>
                    <td>
                      {result.gpuTimeMs < 0 ? (
                        <span className="status-fail">GPU FAILED</span>
                      ) : result.validationPassed ? (
                        <span className="status-pass">✓ PASS ({(result.relativeError * 100).toFixed(4)}%)</span>
                      ) : (
                        <span className="status-fail">✗ FAIL ({(result.relativeError * 100).toFixed(2)}%)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detailed Results */}
          <div className="card mt-6">
            <h3 className="text-lg font-bold mb-4 text-white">Detailed Results (for submission)</h3>
            <pre className="text-xs overflow-x-auto">
{results.map(r => `DOF: ${r.dofCount} | CPU: ${formatTime(r.cpuTimeMs)} | GPU: ${formatTime(r.gpuTimeMs)} | Valid: ${r.validationPassed} | maxW_cpu: ${formatNumber(r.cpuMaxW)} | maxW_gpu: ${formatNumber(r.gpuMaxW)}`).join('\n')}
            </pre>
            <button
              onClick={() => {
                const text = results.map(r => 
                  `DOF: ${r.dofCount} | CPU: ${formatTime(r.cpuTimeMs)} | GPU: ${formatTime(r.gpuTimeMs)} | Valid: ${r.validationPassed}`
                ).join('\n');
                navigator.clipboard.writeText(text);
              }}
              className="btn-secondary mt-4 text-sm"
            >
              Copy Results
            </button>
          </div>
        </section>
      )}

      {/* Instructions */}
      <section className="card">
        <h2 className="text-xl font-bold mb-4 text-white">How to Submit</h2>
        <ol className="space-y-2 text-gray-300 list-decimal list-inside">
          <li>Run the benchmark above and verify validation passes</li>
          <li>If GPU time for 60k DOF is under 20ms, you've won!</li>
          <li>Take a screenshot or copy the results</li>
          <li>Create a Pull Request with your optimizations</li>
          <li>Include your benchmark results and hardware specs</li>
        </ol>
      </section>
    </div>
  );
}

