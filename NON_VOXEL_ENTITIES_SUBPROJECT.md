# Non-Voxel Entities in `polychora` (Pill Avatar Subproject)

## Goal

Render a small number of non-voxel entities (example: primitive pill-like player avatars) in the `crates/polychora` world while preserving current voxel rendering behavior.


## Current State (What Exists Today)

1. `polychora` already sends non-voxel `ModelInstance`s to the renderer.
- A held-item preview tesseract is built every frame in `crates/polychora/src/main.rs:597` and rendered in both backends at `crates/polychora/src/main.rs:1073`.

2. All tetra instances currently share one hardcoded model mesh (tesseract tetra decomposition).
- `OneTimeBuffers` always loads `generate_tesseract_tetrahedrons()` in `src/render.rs:1230`.
- The shader maps each output tetra as `modelIndex = outputIndex % modelCount` and `instanceIndex = outputIndex / modelCount` in `slang-shaders/src/rasterizer.slang:199`.

3. Instance format is limited.
- `ModelInstance` currently has only `model_transform` and `cell_material_ids[8]` in `common/src/lib.rs:78`.
- Shader-side layout matches this in `slang-shaders/src/types.slang:42`.

4. VTE + tetra composition is currently overlay-style, not true depth composition.
- VTE backend enables optional tetra overlays in `src/render.rs:4841`.
- Rasterizer writes only where tetra projects and leaves uncovered pixels untouched in VTE mode at `slang-shaders/src/rasterizer.slang:878`.
- Presented VTE image is layer-0 collapsed output in `slang-shaders/src/present.slang:99`.


## Constraints That Matter

1. Multiple model shapes are not first-class yet.
- Every instance implicitly uses the same `modelTetrahedrons` buffer.

2. `cell_material_ids` is fixed-width (8 entries), tied to current tesseract cell usage.

3. VTE currently owns world hit resolution; tetra in VTE mode behaves as screen overlay, not voxel-occluded world geometry.


## Viable Paths

## Path A: Fastest, Low-Risk (Recommended First Milestone)

Use existing tesseract model instances to build approximate capsule/pill avatars (stack/scale/rotate a few transformed tesseracts per avatar).

### Changes

1. Add game-level dynamic entity data model.
- New module in `crates/polychora/src/` (e.g. `entities.rs`) with entity id, position/orientation, material, and simple avatar primitive parameters.

2. Build avatar render instances each frame.
- Extend scene/app flow to append avatar instances to the same `Vec<ModelInstance>` path used today (`crates/polychora/src/main.rs:1084`).

3. Keep VTE behavior explicit for milestone 1.
- Option A1: Render avatars only on tetra backends.
- Option A2: In VTE mode, render as overlay (current behavior), accepting non-occlusion by voxels.

### Pros

1. Minimal renderer/shader changes.
2. Fast implementation.
3. Enough for "small number of primitive avatars" to validate gameplay/readability.

### Cons

1. Not true capsule geometry.
2. In VTE mode, overlays are not world-depth-correct.


## Path B: True Pill Geometry in Tetra Pipeline

Introduce a distinct pill/capsule tetra mesh and per-instance model selection.

### Required Renderer/Shader Work

1. Generalize model selection.
- Extend shared structs in `common/src/lib.rs` and `slang-shaders/src/types.slang`.
- Update tetra preprocess indexing logic in `slang-shaders/src/rasterizer.slang:199` (and raytracer analog) so instances can choose model ranges, not implicit modulo across one global model.

2. Add pill model generation/upload.
- Add mesh generation and buffer upload path near `src/render.rs:1216`.

3. Keep edge/raytrace paths consistent.
- Update any code that assumes one model edge/tet set.

### Pros

1. Proper geometry quality.
2. Cleaner long-term non-voxel entity support on tetra backends.

### Cons

1. Medium complexity and risk.
2. More perf sensitivity if implemented with naive per-instance filtering.


## Path C: Depth-Correct Avatars in VTE

Integrate avatar primitive intersection directly in VTE Stage A (or add explicit depth-composition buffers/passes).

### Required Work

1. Add avatar primitive buffers/meta in live GPU data (parallel to voxel meta path in `src/render.rs:4566`).
2. In `slang-shaders/src/voxel.slang:1353`, intersect ray with avatar primitives and resolve nearest hit vs voxel traversal hit.
3. Unify shading/material mapping for avatar hits.

### Pros

1. Correct world occlusion in VTE.
2. Best long-term if VTE is intended gameplay default.

### Cons

1. Highest complexity.
2. Requires careful perf budgeting and debugging.


## Recommended Plan

1. Milestone 1: Path A (fast validation).
- Add dynamic entity list.
- Render 2-16 avatars as composed tesseract instances.
- Ship with tetra backend support first; define explicit VTE behavior.

2. Milestone 2: Decide long-term backend target.
- If tetra remains primary: pursue Path B.
- If VTE is primary: pursue Path C.

3. Milestone 3: Add tests/validation.
- Unit tests for entity-to-instance generation.
- Golden screenshot checks for both backends and camera poses.
- Perf checks for baseline world + N avatars.


## Open Decisions

1. Must avatars be depth-correct in VTE for initial release?
2. Is approximate pill geometry acceptable for first playable milestone?
3. Are non-voxel entities expected to participate in gameplay collisions now, or render-only initially?
