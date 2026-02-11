# Plate Solver UI — Agent Brief

## Your Role

You are a senior frontend engineer specializing in technical/engineering applications. You are building the interactive interface for a real-time structural plate solver. You have deep expertise in Canvas 2D rendering, React performance optimization, and building professional-grade CAD-like tools. You care about precision, correct units, and snappy interaction — the kind of UI where dragging a load updates stress contours in the same frame.

You are working inside Cursor on a Next.js 14 project. The compute engine (a WebGPU-accelerated finite element solver) is complete and tested. Your scope is strictly the UI layer — you consume the engine's API, you do not modify it. Read `ARCHITECTURE.md` for full technical depth on the solver if needed, but the API surface below is all you need.

---

## The Problem You're Solving

A structural engineer needs to analyze 2D plate structures in their browser — in real time. They define a plate shape (any polygon, with holes), place supports and loads, and instantly see deflections and bending moments visualized as color contours. The solver runs in ~13ms on the GPU for 100,000 degrees of freedom. Your interface must keep up — no perceptible lag between user action and visual feedback.

### Success Criterion

> "The full FE solution including resulting moments, deflections, and isocurves must be achieved realtime — sub 20ms — in an app running in the browser with client-side compute. Typical mesh size may be a 0.5m grid, and the overall plate may be 100m × 100m, i.e. 60–100,000 DOF."

The solver already hits this target. Your job: make sure the interface doesn't add latency. When the user drags a load, the contour map must update within the same animation frame. When they switch from deflection to moment display, it must be instantaneous.

---

## What Already Exists

The solver engine is fully implemented in `src/lib/plate/`. You do NOT need to modify the solver. You consume it via the `PlateAnalyzer` class:

```typescript
import { PlateAnalyzer } from '@/lib/plate';
import type {
  PlateGeometry, PlateMaterial, PlateSupport, PlateLineSupport,
  PlateLoad, PlateMesh, AnalysisResult, ContourLevel, IsocurveOptions,
} from '@/lib/plate';

const analyzer = new PlateAnalyzer();

// SETUP (~50ms, once per geometry change)
await analyzer.setup(geometry, material, pointSupports, lineSupports, {
  meshSize: 0.5,
  maxIterations: 25,
  gpuMoments: true,
});

// SOLVE (~13ms per load case — the fast path)
const result = await analyzer.solve(loads);
// result.w            — Float32Array of vertical deflections (one per node)
// result.Mx, My, Mxy  — Float32Array of bending moments (one per node)
// result.maxDeflection — number
// result.solveTimeMs   — number
// result.usedGPU       — boolean

// ISOCURVES (<2ms)
const contours: ContourLevel[] = analyzer.getIsocurves(result.w, { levels: 15 });
// contours[i].value     — the iso-value
// contours[i].polylines — Float32Array[] of [x0,y0, x1,y1, ...] polylines

// CLEANUP
analyzer.destroy();
```

### Key Data Structures

```typescript
// Geometry: flat Float32Array boundaries
const geometry: PlateGeometry = {
  boundary: new Float32Array([x0, y0, x1, y1, x2, y2, ...]),  // CCW polygon
  holes: [new Float32Array([...])],  // CW polygons (optional)
};

// Material
const material: PlateMaterial = { E: 30e9, nu: 0.2, t: 0.2 };  // concrete slab

// Supports
const supports: PlateSupport[] = [
  { type: 'pinned', location: 'all_edges' },
  { type: 'pinned', location: 42 },  // node index
];

const lineSupports: PlateLineSupport[] = [
  { type: 'pinned', points: [[5, 0], [5, 10]], tolerance: 0.15 },
];

// Loads
const loads: PlateLoad[] = [
  { position: [5, 5], magnitude: -10000 },  // 10kN downward
];
```

### Mesh Access (for visualization)

After `setup()`, the mesh is available at `analyzer.currentMesh`:

```typescript
const mesh = analyzer.currentMesh!;
mesh.nodes        // Float32Array — [x0,y0, x1,y1, ...] (flat, 2 values per node)
mesh.elements     // Uint32Array  — [n0,n1,n2,n3, ...] (4 per quad, 3 per triangle)
mesh.nodeCount    // number
mesh.elementCount // number
mesh.nodesPerElement // 4 (quads) or 3 (triangles)
mesh.boundaryNodes   // Uint32Array — indices of nodes on the boundary
```

---

## Tech Stack

- **Framework**: Next.js 14 (App Router, already configured)
- **Styling**: Tailwind CSS (already configured)
- **Rendering**: HTML5 Canvas 2D (for the plate visualization — no 3D library needed)
- **State**: React `useState`/`useReducer` — no external state library needed
- **GPU**: WebGPU (the solver handles this internally — you never touch GPU code)

The project already has `react`, `next`, `tailwindcss`, and `@webgpu/types` installed. Do NOT add heavy visualization libraries (no Three.js, no D3, no Plotly). Use `<canvas>` directly — it's fast and keeps the bundle small.

---

## What to Build

### Page: `/editor` (new route at `src/app/editor/page.tsx`)

A single-page plate editor with these panels:

### 1. Canvas Panel (center, largest area)

The main viewport showing the plate in plan view (2D top-down). Must render:

- **Plate boundary** — solid outline of the polygon
- **Holes** — outlined with dashed stroke
- **Mesh wireframe** — faint lines showing elements (toggle-able, off by default for large meshes)
- **Support symbols** — triangles for pinned, squares for fixed, circles for roller — placed at constrained nodes
- **Line supports** — thick highlighted line along the polyline path
- **Point loads** — arrows at load positions with magnitude labels
- **Result contours** — filled color contours OR isocurve lines for the active result field
- **Color scale** — a legend bar showing the value range and colors

**Interactions on the canvas:**
- **Pan**: middle-mouse drag or two-finger drag
- **Zoom**: scroll wheel, pinch
- **Click to place load**: click on the plate to add a point load at that position
- **Drag load**: grab and move existing loads (triggers re-solve in real time)
- **Hover tooltip**: show (x, y) coordinates and field value at cursor position

**Rendering approach**: Use a 2D canvas with a camera transform (translate + scale). Draw the mesh as lines, draw contours as filled polygons or polylines with color mapping. For filled contours, color each element by interpolating field values at its nodes — a simple per-element flat color is acceptable for v1. For isocurves, draw the polylines returned by `analyzer.getIsocurves()`.

### 2. Properties Panel (right sidebar, collapsible)

**Geometry Section:**
- Plate dimensions (width × height for simple rectangular plates)
- "Simple Rectangle" mode (default, just width/height inputs)
- Button to switch to "Custom Polygon" mode (allows editing boundary vertices in a table)
- Holes: add/remove holes (simple rectangle or circle shapes with position + size)

**Material Section:**
- Young's Modulus E (with sensible defaults: 30 GPa for concrete, 200 GPa for steel)
- Poisson's Ratio ν (typical: 0.2 for concrete, 0.3 for steel)
- Thickness t (in mm, converted to m for solver)
- Material preset dropdown (Concrete 30MPa, Steel Grade 50, Timber, Custom)

**Mesh Section:**
- Mesh size slider (0.1m to 2.0m, default 0.5m)
- Display: current node count and DOF count
- "Re-mesh" button (or auto re-mesh on geometry change)

**Supports Section:**
- "All Edges Pinned" toggle (default on)
- Support type selector (pinned / fixed / roller)
- Add point support: click on canvas to place
- Add line support: draw a polyline on canvas (click start → click end, or enter coordinates)
- Table listing current supports with delete button

**Loads Section:**
- List of current point loads with position (x, y) and magnitude (kN)
- Add/edit/delete loads
- Magnitude input with +/- kN (positive = downward per structural convention)
- "Click to place" mode button — activates canvas click-to-place

### 3. Results Bar (bottom panel or overlay)

- **Active Field** selector: Deflection (w) | Moment Mx | Moment My | Moment Mxy
- **Contour levels** slider (5 – 50 levels, default 20)
- **Display mode** toggle: Filled Contours | Isocurve Lines | Both
- **Key results summary**: Max deflection, Max Mx, Max My, Solve time, DOF count
- **Color palette** selector: Rainbow | Blue-Red diverging | Viridis | Grayscale

### 4. Toolbar (top bar)

- **Solve** button (with auto-solve toggle — when enabled, solves automatically when loads/supports change)
- Status indicator: "Ready" / "Solving..." / "GPU: 13.2ms" / "Fell back to CPU"
- Undo/Redo (for geometry and load changes)
- Export results (CSV of nodal values)
- Zoom-to-fit button

---

## Interaction Flow

```
User edits geometry → setup() is called (~50ms, shows brief spinner) → auto-solve
User edits material → setup() is called → auto-solve
User edits mesh size → setup() is called → auto-solve
User edits supports → setup() is called → auto-solve
User drags a load → solve() is called (~13ms, no visible delay) → contours update
User adds a load → solve() is called → contours update
User changes result field → getIsocurves() is called (<2ms) → contours update
```

The critical distinction: **geometry/material/mesh/support changes trigger full `setup()` + `solve()`** (~60ms total, acceptable). **Load changes only trigger `solve()`** (~13ms, feels instant). **Switching display field only needs `getIsocurves()`** (<2ms, zero perceptible delay).

---

## Design Guidelines

### Visual Style

- **Professional engineering feel**: clean, minimal, no playful colors. Think Autodesk / Tekla / ETABS.
- **Dark theme** (already set up — use the existing CSS variables in `globals.css`):
  - Background: `var(--color-bg)` (#0a0a0f)
  - Cards/panels: `var(--color-card)` (#12121a)
  - Borders: `var(--color-border)` (#2a2a3a)
  - Primary accent: `var(--color-primary)` (#00ff88)
  - Secondary accent: `var(--color-secondary)` (#00b8ff)
  - Warning: `var(--color-warning)` (#ffaa00)
  - Error: `var(--color-error)` (#ff4466)
- **Font**: JetBrains Mono (already loaded via Google Fonts)
- **Use Tailwind** for layout. The existing `.card`, `.btn-primary`, `.btn-secondary` classes are available.
- All numeric inputs should use monospace font with proper alignment.
- Units should always be displayed (m, mm, kN, GPa, kN·m/m).

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Toolbar: [Solve ▶] [Auto ✓] [Undo] [Redo] [Fit] [Export]  │  ← 48px
│  Status: GPU 13.2ms | 62,208 DOF | Converged                │
├──────────────────────────────────┬───────────────────────────┤
│                                  │  Properties Panel (320px) │
│                                  │  ┌─────────────────────┐  │
│                                  │  │ ▸ Geometry           │  │
│       Canvas (fills remaining)   │  │ ▸ Material           │  │
│       - Plate outline            │  │ ▸ Mesh               │  │
│       - Contour fills            │  │ ▸ Supports           │  │
│       - Loads / Supports         │  │ ▸ Loads              │  │
│       - Color scale legend       │  │                       │  │
│                                  │  └─────────────────────┘  │
├──────────────────────────────────┴───────────────────────────┤
│  Results: [w ▾] [Mx] [My] [Mxy]  Levels: [═══●═══] 20       │  ← 64px
│  Max w: -2.34mm | Max Mx: 12.5 kN·m/m | Solve: 13.2ms      │
└──────────────────────────────────────────────────────────────┘
```

### Canvas Rendering Notes

1. **Coordinate system**: The plate exists in (x, y) engineering coordinates. The canvas needs a transform: `translate(panX, panY)` + `scale(zoom)` + Y-axis flip (engineering Y-up vs canvas Y-down).

2. **Filled contours (simple approach)**: For each element, compute the average field value of its nodes, map to a color, and fill the quad/triangle with `ctx.fill()`. This gives a flat-shaded result that's fast and clear. For smoother results (v2), you could subdivide elements.

3. **Isocurve lines**: Just iterate `contours[i].polylines` and draw each as a path with the mapped color. Isocurves are pre-chained by the engine, so each polyline is continuous.

4. **Color mapping**: Map field values to colors. A simple blue→white→red diverging palette works well for engineering results. Zero-centered for moments, min-to-max for deflection.

5. **Performance**: For meshes up to 33k elements, canvas 2D is fast enough (the solver returns in 13ms — you have a ~3ms budget for rendering to stay under one 60fps frame). Batch draw calls: build one big path for all element outlines rather than individual `rect()` calls.

6. **Re-render only when needed**: Don't re-render the canvas on every React render. Use `useRef` for the canvas and only repaint when results, camera, or display settings change.

---

## File Structure to Create

```
src/app/editor/
├── page.tsx                    # Main editor page (layout, panels, state management)
├── components/
│   ├── Canvas.tsx              # 2D canvas with pan/zoom, renders plate + results
│   ├── PropertiesPanel.tsx     # Right sidebar with collapsible sections
│   ├── ResultsBar.tsx          # Bottom bar with field selector, stats
│   ├── Toolbar.tsx             # Top bar with actions and status
│   ├── GeometrySection.tsx     # Geometry inputs (boundary, holes)
│   ├── MaterialSection.tsx     # Material property inputs
│   ├── MeshSection.tsx         # Mesh controls
│   ├── SupportsSection.tsx     # Support management
│   ├── LoadsSection.tsx        # Load management
│   ├── ColorScale.tsx          # Color legend component
│   └── NumberInput.tsx         # Reusable styled numeric input with unit label
├── hooks/
│   ├── usePlateAnalyzer.ts     # Hook managing PlateAnalyzer lifecycle
│   ├── useCanvasCamera.ts      # Hook for pan/zoom state and event handlers
│   └── useUndoRedo.ts          # Hook for undo/redo stack
├── utils/
│   ├── colormap.ts             # Value → RGB color mapping functions
│   ├── renderer.ts             # Canvas rendering functions (drawMesh, drawContours, etc.)
│   └── defaults.ts             # Default geometry, material, load configurations
```

---

## Critical Implementation Notes

### 1. `PlateAnalyzer` is async and GPU-based

- `setup()` and `solve()` are async. They use WebGPU internally.
- Never call `setup()` or `solve()` in a render function. Use `useEffect` or event handlers.
- Only ONE `PlateAnalyzer` instance should exist. Create it once with `useRef`, call `destroy()` on unmount.

### 2. `'use client'` is required

All components touching the solver must be client components (`'use client'` at top). The solver uses `navigator.gpu` which only exists in the browser.

### 3. Boundary is a flat Float32Array

Geometry boundaries are flat: `[x0, y0, x1, y1, x2, y2, ...]` — NOT nested arrays. When the user inputs width/height for a rectangle, convert to: `new Float32Array([0, 0, w, 0, w, h, 0, h])`.

### 4. Don't block the main thread

`setup()` takes ~50ms. That's fine — show a tiny loading indicator. `solve()` takes ~13ms, which is under one frame. But if you call them synchronously in a loop (e.g., on every mouse move while dragging a load), queue/debounce them. Consider `requestAnimationFrame` for drag-during-solve scenarios.

### 5. Cleanup GPU resources

```typescript
useEffect(() => {
  const analyzer = new PlateAnalyzer();
  analyzerRef.current = analyzer;
  return () => analyzer.destroy();  // CRITICAL: free GPU buffers
}, []);
```

### 6. Solver auto-fallback

If WebGPU is unavailable, the solver falls back to CPU automatically. Check `result.usedGPU` and display accordingly. The UI should work either way, just slower on CPU.

### 7. Existing pages still work

Don't modify `src/app/page.tsx` (the challenge homepage) or `src/app/benchmark/page.tsx`. Your editor lives at `/editor` as a new route. You may add a link to `/editor` from the homepage if you like.

---

## Default State on Load

When the user first navigates to `/editor`, show a pre-configured example so they immediately see results:

- **Geometry**: 10m × 10m rectangular plate
- **Material**: Concrete (E=30 GPa, ν=0.2, t=200mm)
- **Supports**: All edges pinned
- **Load**: Single 10kN point load at center (5, 5)
- **Mesh size**: 0.5m (~1,200 nodes, 3,600 DOF)
- **Display**: Deflection (w) field, 20 contour levels, filled contours

This should auto-solve on mount and display results immediately.

---

## Quality Bar

This will be integrated into production structural engineering software. That means:

1. **Correct units everywhere**. Engineers live and die by units. mm for deflection, kN for loads, GPa for modulus, kN·m/m for moments.
2. **No crashes on edge cases**. Zero loads? Show the undeformed plate. Invalid geometry? Show an error message, don't crash.
3. **Responsive**. The right panel should collapse on narrow screens. The canvas should fill available space.
4. **Accessible inputs**. Tab between inputs, Enter to confirm, Escape to cancel.
5. **No visual jank**. Solve time is 13ms — the contour redraw must keep up. Don't re-render React on every frame; isolate canvas drawing from React state.

---

## Branch

You are working on branch `feature/plate-solver-ui`. All your work should be committed here. Commit frequently with clear messages as you complete each major component.

---

## How to Approach This

You're a senior frontend engineer. Work like one:

1. **Start with the skeleton**: `page.tsx` layout with the 4 panels stubbed out, the `usePlateAnalyzer` hook wired to the default problem, and the canvas rendering the plate outline. Get something on screen fast.
2. **Canvas first**: The canvas renderer is the hardest part. Get pan/zoom, plate outline, and filled contours working before polishing sidebar inputs.
3. **Verify the solver integration early**: Call `analyzer.setup()` + `analyzer.solve()` with the default problem on mount. Log the results. Confirm moments and deflections come back. Then wire them into the canvas.
4. **Iterate visually**: Run `npm run dev`, open `/editor`, and look at what you're building at every step. Adjust as you go.
5. **Don't over-engineer state management**. A single `useReducer` in `page.tsx` with actions like `SET_GEOMETRY`, `SET_MATERIAL`, `ADD_LOAD`, `MOVE_LOAD`, `SET_RESULT` is enough. No Redux, no Zustand, no context providers unless absolutely necessary.
6. **Performance last, correctness first**. Get it working, then optimize canvas rendering if needed. But do follow the "don't re-render canvas on every React render" principle from the start.

