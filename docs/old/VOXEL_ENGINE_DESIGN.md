# 4D Voxel Engine Design (Working Draft)

This document proposes a new voxel-specific renderer for the `game` world.
It is intended as a high-performance engine path alongside the existing tetra
rasterizer and tetra path tracer.

The goal is to preserve the established 4D camera semantics while replacing
the tetra-heavy frame cost with voxel-native traversal and aggressive culling.


## 1. Problem Statement

Current game rendering feeds voxel surfaces through tesseract -> tetrahedron
decomposition. This is correct but not scalable for large 4D worlds:

- World volume grows with `O(n^4)`.
- Surface complexity grows with `O(n^3)`.
- Tetra raster cost still scales with clipped tetra overlap per pixel.

Measured frame profiles already show:

- `tet_clip`: small
- `tet_bin`: small
- `tet_raster`: dominant by a wide margin

So the next step is not another local kernel tweak; it is a representation and
pipeline change.


## 2. Two-Stage 4D Camera Semantics

To make the renderer more "4D genuine", define camera semantics in two stages.

### 2.1 Stage A: 4D -> 3D Hyper-Image

Use a 4D orthonormal camera basis:

- `R`: right axis
- `U`: up axis
- `A`: ana axis (the third sensor axis, orthogonal to `R,U,F`)
- `F`: forward axis

For sensor coordinates `(u, v, s)`, construct ray direction:

```text
rayDir = normalize(u * R + v * U + s * A + f * F)
```

This is the direct 4D pinhole generalization. Stage A output is a 3D
hyper-image `H(u, v, s)` (sampled stochastically or binned in `s`).

### 2.2 Stage B: 3D -> 2D Display Operator

The final 2D image is produced by applying an operator over `s`:

- `slice`: evaluate near a single `s = s0`.
- `thick_slice`: evaluate over band `[s0-w, s0+w]`.
- `integral`: average/integrate over full `s` range (current behavior analog).
- `max`/other debug operators: optional analysis views.

This separation makes the hidden dimension explicit and easier to reason about.

Comment:

- `integral` is the primary gameplay/exploration mode.
- `slice` and `thick_slice` are diagnostic/intuition tools, not the main output.

### 2.3 Compatibility with Current Semantics

Current ZW-angle sampling can be treated as one particular Stage B operator
(`integral`) with a specific Stage A parameterization:

```text
viewAngle = (pi/2) / focal_zw
s_range   = tan(viewAngle / 2)
s         = s_norm * s_range, s_norm in [-1, 1]
```

So we preserve existing behavior while gaining explicit slice/thick-slice modes.


## 3. High-Level Architecture

### 3.1 Proposed Engine Name

`Voxel Traversal Engine (VTE)`

### 3.2 Core Idea

Render directly from voxel chunks using ray traversal in 4D:

- No tetra decomposition in the game path.
- No per-pixel loop over thousands of clipped tetrahedra.
- First-hit traversal handles occlusion naturally (front-most solid wins).

### 3.3 Pipeline Overview

1. CPU updates chunk data and culling metadata.
2. GPU receives compact visible chunk set and voxel data.
3. Compute shader generates Stage A rays from `(u, v, s)` sensor coordinates.
4. Ray traverses chunk grid (coarse) then voxel cells (fine) via 4D DDA.
5. First surface hit provides:
   - material id
   - normal
   - world hit position
   - texture coordinates
6. Shade and write contribution to `H(u, v, s)` or directly apply Stage B.
7. Stage B display operator collapses `s` to final 2D image.


## 4. Data Model (GPU-Oriented)

Use world/chunk data directly from voxel structures (`CHUNK_SIZE = 8`).

### 4.1 Chunk Payload

Per chunk:

- `occupancy bitset`: 4096 bits (512 bytes) for solid/air.
- `material array`: 4096 bytes (`u8` ids) or palette-indexed form.
- Optional:
  - macro occupancy mask for empty-space skipping inside chunk.
  - dirty/version id for incremental updates.

### 4.2 Suggested GPU Structures

```c
struct GpuChunkHeader {
    int4 chunk_coord;      // chunk-space coordinate
    uint occupancy_offset; // word offset into global occupancy buffer
    uint material_offset;  // byte offset into global material buffer
    uint macro_offset;     // optional macro mask offset
    uint flags;            // bitfield: empty/full/has_macro/etc
};
```

```c
struct VisibleChunkRef {
    uint chunk_index;      // index into GpuChunkHeader[]
};
```

### 4.3 Indirection Strategy

Use compact arrays, not sparse random access hash lookups in shader hot paths:

- CPU compacts visible chunk refs each frame.
- GPU traversal operates on compact visible/indexed buffers.
- Keep mapping from `chunk_coord -> chunk_index` in a GPU hash only if needed
  for chunk-DDA random lookup.


## 5. Traversal Algorithm

### 5.1 Ray Generation

Use Stage A semantics from Section 2.1.

For each pixel:

1. Generate `K` `s` samples (random, stratified, blue-noise, or deterministic bins).
2. Build view-space ray for each `(u, v, s)` sample.
3. Transform to world-space with `viewMatrixInverse`.

`s` sample range comes from camera configuration (legacy-compatible with
`focal_zw`, or explicit `s_min/s_max` in future camera API).

### 5.2 Two-Stage DDA

Stage A: chunk-level DDA (step in chunk units)

- Traverse world chunk grid in 4D.
- Skip empty or non-visible chunks quickly.
- Descend only when a potentially solid chunk is reached.

Stage B: voxel-level DDA (within chunk)

- Traverse the `8x8x8x8` local grid.
- On first solid voxel, compute hit details and terminate that sample.

### 5.3 Hit Information

For each hit sample:

- `hit_pos_world`: 4D world-space hit point.
- `normal4`: axis-aligned voxel face normal in 4D (`±X/±Y/±Z/±W`).
- `material_id`: voxel material.
- `t`: hit distance (useful for optional temporal occlusion tests).


## 6. Texture Mapping on Voxel Faces

Merging or implicit traversal does not prevent texture mapping.

Each voxel boundary is a 3D hyperface in 4D. If the hit face normal is aligned
to axis `A`, use the remaining 3 axes as texture coordinates.

Example axis mapping:

- hit on `±X` face -> texture coords from `(Y, Z, W)`
- hit on `±Y` face -> `(X, Z, W)`
- hit on `±Z` face -> `(X, Y, W)`
- hit on `±W` face -> `(X, Y, Z)`

### 6.1 Tiled Mapping (Minecraft-like)

```text
uvw = frac((p_axis_triplet * tile_scale) + tile_offset)
```

This repeats textures uniformly regardless of merged patch size.

### 6.2 Stretched Mapping (Patch-local)

If we later add merged patch primitives:

```text
uvw = (p_axis_triplet - patch_origin) / patch_extent
```

### 6.3 Orientation Consistency

Define a fixed per-face tangent basis lookup table so negative faces do not
randomly mirror textures. The basis is data-driven and can be tuned once.


## 7. Aggressive Culling Strategy

Culling is a first-class design objective.

Use layered culling so each stage reduces work for the next:

### 7.1 L0: World Streaming Culling (CPU)

- Radius/window culling in chunk space around camera.
- Avoid uploading distant chunks altogether.

### 7.2 L1: Stage A Frustum Culling (CPU or GPU)

For each candidate chunk AABB (16 corners in 4D), transform corners to camera
basis coordinates `(r, u, a, fwd)`:

- Reject if all corners are behind near depth (`fwd <= near`).
- Reject if projected `u/fwd` range misses screen X.
- Reject if projected `v/fwd` range misses screen Y.

Use conservative bounds; false positives are acceptable, false negatives are not.

### 7.3 L1b: Stage B `s`-Interval Culling

Compute conservative projected `s` interval for each chunk:

```text
s_proj ~ (a / fwd) * f_s
s_interval = [min_corner(s_proj), max_corner(s_proj)]
```

Operator-specific rejection:

- `slice(s0)`: reject if `s_interval` does not overlap `[s0-eps, s0+eps]`.
- `thick_slice(s0, w)`: reject if no overlap with `[s0-w, s0+w]`.
- `integral`: do not apply this rejection (full range contributes).

This is a key culling win not available in the old monolithic formulation.

### 7.4 L2: Chunk Occupancy Culling

- Skip chunks flagged empty.
- Fast-path full chunks with specialized boundary hit logic if useful.

### 7.5 L3: In-Chunk Hierarchy Culling

Within non-empty chunks, use macro occupancy (e.g. `2x2x2x2` macro cells):

- DDA can skip empty macro regions before voxel-level stepping.

### 7.6 L4: Traversal Occlusion Culling (Natural First-Hit)

Ray traversal inherently culls obscured voxels:

- Once first opaque boundary hit is found for sample, terminate.
- No overdraw pass or back geometry processing for that sample.

### 7.7 L5: Temporal Occlusion Hinting (Optional, Advanced)

Maintain low-resolution previous-frame nearest-hit depth by tile/sample-bin:

- If chunk AABB entry `t_min` is well beyond known near depth for tile/bin,
  skip chunk traversal for that sample.
- Keep conservative safety margin to avoid popping.


## 8. Integral-Mode Optimization Strategy (Primary)

This section defines how VTE should be optimized when `integral` mode is active.

Comment:

- Integral mode is the target performance benchmark.
- Slice/thick-slice optimizations are secondary.

### 8.1 Integral Cost Drivers

In integral mode, Stage B cannot reject by `s` interval. Cost is dominated by:

- number of `s` samples per pixel
- chunk-level traversal steps per sample
- in-chunk voxel steps until first hit
- divergence across rays in a workgroup

### 8.2 Integral-Focused Levers

1. Aggressive empty-space skipping:
- prioritize chunk occupancy flags + macro occupancy hierarchy.
- reduce average traversal steps before first solid hit.

2. Low-discrepancy `s` sampling:
- use deterministic low-discrepancy sequences (per-pixel scrambled) instead of
  naive uniform RNG each frame.
- improves convergence at equal sample count.

3. Temporal accumulation in integral mode:
- accumulate integral samples across frames with camera-motion gating.
- reset or blend down accumulation under high motion/rotation.

4. Motion-adaptive sample count:
- low samples while moving quickly, higher samples when still.
- keep stable frame-time target while preserving integral semantics.

5. Ray coherence batching:
- dispatch/tile strategy that traces similar `s` values together where possible.
- reduces branch divergence and improves cache locality.

6. Opaque-first shading path:
- first-hit terminate for opaque materials (default).
- defer transparency/translucency to later milestones.

7. Shader hot-path minimization:
- precompute camera basis and reusable constants once per dispatch.
- keep material sampling branch-light and texture-coordinate math simple.

### 8.3 What Not to Rely On

- Do not count on `s`-interval culling for integral mode.
- Do not count on thin/thick slice mode for baseline game performance claims.

### 8.4 Future Acceleration Tracks (Post-Baseline)

These are explicitly future steps, not blockers for initial VTE implementation.

1. `s`-coherent ray bundles (neighbor reuse):
- trace sparse skeleton `s` samples with full DDA
- infer nearby `s` samples from local hit neighborhoods
- fall back to full DDA on misses (correctness-preserving)

2. Beam/shared chunk traversal:
- share chunk-level traversal across an `s`-sample bundle for a pixel
- fan out to voxel-level traversal only when needed

3. Voxel splatting to `H(u, v, s)` (alternate architecture):
- invert traversal into surface-driven projection
- evaluate if integral DDA path misses performance targets

4. 4D sparse tree acceleration:
- consider 16-tree hierarchy once chunk+macro occupancy path matures

Decision gate:

- Implement and profile integral baseline first.
- Add (1), then (2), before considering architecture pivot (3).


## 9. Shader/Pass Plan

Proposed new compute stages in `src/render.rs` orchestration:

1. `voxel_clear` (if needed)
2. `voxel_trace_stage_a` (main traversal kernel producing Stage A samples)
3. `voxel_display_stage_b` (slice/thick-slice/integral resolve)
4. optional `voxel_temporal_resolve`

Potential helper kernels:

- `build_visible_chunks_cs` (GPU-side compaction if CPU compaction is not enough)
- `compute_chunk_s_intervals_cs` (optional if `s`-interval culling is GPU-driven)
- `voxel_debug_stats_cs` (counters for traversal and culling efficiency)

### 9.1 Output Contract

Two practical options:

Option A (compatibility-first):

- Keep current pixel buffer layout and map Stage A samples into existing buffer.
- Present pass stays mostly unchanged.

Option B (cleaner):

- Use dedicated `RWStructuredBuffer<float4> voxelOutput2D`.
- Add a present shader branch for this engine.


## 10. Performance Model

Current tetra raster cost is roughly:

```text
O(pixels * tets_per_tile * depth_factor)
```

VTE target cost:

```text
O(pixels * s_samples * traversal_steps_until_hit)
```

This is a better fit for sparse or surface-dominant voxel worlds and gives
stronger scaling as scene complexity grows.

With Stage B operators:

- `slice` / `thick_slice`: effective samples and candidate chunks drop
  significantly via `s`-interval culling.
- `integral`: highest quality/highest cost mode, equivalent to current-style
  full hidden-dimension integration.


## 11. Integration Plan

### Milestone M0: Design + Interfaces

- Add `RenderBackend` enum for engine selection.
- Add VTE-specific debug counters.

### Milestone M1: Integral Baseline Path

- Upload chunk occupancy/material buffers.
- Implement Stage A ray -> chunk DDA -> voxel DDA first-hit.
- Implement Stage B `integral` only (initially no slice/thick-slice dependency).
- Flat/basic shading by material id.

### Milestone M2: Integral Visual Parity

- Implement face-local texture coordinates.
- Reuse material sampling and basic lighting (ambient + sun diffuse).
- Validate visual continuity against current integral behavior.

### Milestone M3: Integral Performance Stack

- L0/L1 candidate reduction (always-on).
- L2 occupancy flags and L3 macro skipping (always-on).
- Low-discrepancy `s` sampling + motion-adaptive sample count.
- Temporal accumulation/reprojection for integral convergence.
- Collect and expose per-stage rejection/perf metrics.

### Milestone M4: Secondary Display Operators

- Add `slice` and `thick_slice` as optional diagnostic modes.
- Add L1b `s`-interval rejection for non-integral operators.
- Keep `integral` as default exploration mode.


## 12. Validation and Metrics

Track these per frame:

- candidate chunks
- frustum-culled chunks
- `s`-interval-culled chunks (non-integral operators)
- empty chunks skipped
- macrocells skipped
- average chunk steps per ray
- average voxel steps per ray
- primary-hit rate
- `s` samples per pixel (current/adaptive)
- Stage B operator mode
- GPU ms per VTE phase

Success criteria:

- Visual semantics match existing 4D camera behavior.
- Texture mapping stable and orientation-consistent.
- Significant reduction vs current `tet_raster` frame cost in game scenes.


## 13. Open Questions

1. Should VTE become game-default immediately, or ship behind runtime toggle?
2. What integral sample budget targets should we enforce:
   moving camera vs stationary camera?
3. Is chunk lookup best done by:
   - GPU hash map, or
   - sorted visible chunk list + direct address table?
4. Should transparent voxel materials be supported in VTE v1?
   (Opaque-only greatly simplifies first-hit termination.)
5. Do we want merged surface patch generation as an optional future path, or keep
   pure implicit voxel traversal?


## 14. Recommendation

Implement VTE as the new game-focused backend and keep tetra-based engines as:

- correctness reference
- non-voxel/general-geometry path
- research path tracer path
- keep `integral` as the default game operator
- keep `slice`/`thick_slice` as optional diagnostic overlays/modes

This matches project goals: interactive exploration of large 4D voxel worlds
with performance that scales better than tetra rasterization.
