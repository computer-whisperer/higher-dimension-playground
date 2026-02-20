# Next-Gen Render Pipeline Plan (Root BVH + Leaf Traversal)

Status: Active design, implementation not complete  
Updated: 2026-02-20

## 1) Goal

Rebuild the client render pipeline so we can scale visible range from ~80 blocks to hundreds/thousands of blocks, using:

1. A root acceleration tree (BVH) over world-space render regions.
2. Specialized leaf traversal/render paths (voxel + tetra) under the same root tree.

This is a hard architecture reset for rendering, not an incremental patch to the legacy split pipeline.

## 2) Non-Negotiable Boundaries

1. Server/client world streaming remains `RegionTreeCore` patches (`WorldSubtreePatch`).
2. The client world cache (`RegionChunkTree`) remains authoritative world state on client.
3. Render structures are derived client-side from world cache. They are not wire protocol types.
4. Reduced functionality is acceptable for MVP, but data structures and boundaries must match final architecture.
5. No protocol coupling to renderer internals.

## 3) Why We Are Rebuilding

Current rendering is split:

1. Voxel traversal path (`render_voxel_frame`) with chunk metadata/payload buffers.
2. Separate non-voxel instance path passed alongside VTE inputs.
3. A fallback tetra raster path (`render_tetra_frame`).

This worked for near-field visualization, but it does not provide a unified scalable scene representation for very large view distances.

## 4) Target Architecture

### 4.1 High-level flow

1. Server sends world patch (`RegionTreeCore`) for requested bounds.
2. Client splices/prunes authoritative world cache tree.
3. Client updates a derived render tree (root BVH + leaf payload refs).
4. Renderer consumes render tree-derived GPU buffers.
5. Shader traversal:
   1. Traverse root BVH.
   2. Dispatch to leaf traversal mode:
      1. Voxel leaf traversal (VTE-style).
      2. Tetra leaf traversal (linear/BVH depending on leaf payload).

### 4.2 Leaf categories (MVP-friendly)

1. `LeafEmpty` (or no leaf emission).
2. `LeafUniform` (material fill).
3. `LeafVoxelSegmentRef` (chunk-array-like voxel segment).
4. `LeafTetSegmentRef` (entity/structure tetra segment, optional in first MVP step).
5. `LeafProceduralRef` is allowed in world tree, but render tree should receive realized renderable leaves for currently requested/visible regions.

## 5) MVP Scope (Correct Architecture, Reduced Functionality)

### 5.1 Must-have

1. Keep protocol and world patch flow unchanged (`RegionTreeCore` only).
2. Add client-local render-tree module and caches.
3. Build deterministic world-cache -> render-tree transform.
4. Render voxel leaves through unified root BVH path.
5. Keep existing behavior functional for singleplayer + multiplayer world streaming.

### 5.2 Allowed temporary limitations

1. LOD beyond L0 remains disabled.
2. Tet leaves can be staged behind a feature flag if voxel-root path is stabilized first.
3. Uniform leaves may initially lower to voxel payloads, then be optimized to analytic/uniform shader handling.

## 6) Rip-Out-First Plan

Delete/disable architecture-breaking paths early, then rebuild:

1. Remove renderer assumptions that non-voxel scene data is a side-channel to VTE.
2. Remove temporary bridging that translates protocol payloads into renderer-specific wire types.
3. Keep one authoritative render input path for world-backed geometry.
4. Keep fallback debug paths only if they do not become runtime authority.

## 7) Implementation Plan (Ordered)

### Phase A: Data model and transform (client-side only)

1. Introduce `shared/render_tree/` (shared type definitions, no protocol dependency).
2. Add client-owned render-tree cache and segment stores.
3. On each world patch:
   1. Splice authoritative world tree.
   2. Compute affected render-tree bounds.
   3. Incrementally rebuild only affected render-tree regions.
4. Add unit tests:
   1. World patch -> render-tree patch correctness.
   2. Splice/prune correctness under repeated overlapping patches.
   3. No-op patch behavior.

### Phase B: Render backend integration

1. Add new render frame input modeled as root BVH + leaf payload tables.
2. Replace call-site split (`voxel_input` + `tetra_entity_instances`) with unified scene input.
3. Keep old renderer path behind explicit debug flag for one migration step, then remove.
4. Add diagnostics:
   1. BVH node count.
   2. Leaf counts by type.
   3. Per-frame rebuild cost.
   4. Traversal counters by leaf type.

### Phase C: Shader pipeline rebuild

1. Introduce root-BVH traversal in voxel traversal shader path (or shared traversal shader layer).
2. Implement leaf dispatch:
   1. Voxel leaf traversal path.
   2. Tet leaf traversal path (linear/BVH selectable by payload).
3. Ensure camera/frustum/clip behavior matches current voxel gold-standard path.
4. Validate correctness with screenshot-based and counter-based checks.

### Phase D: Non-voxel merge

1. Move non-voxel entities/preview HUD geometry into render-tree tet leaves.
2. Delete legacy non-voxel side-channel data flow from gameplay loop to renderer.
3. Keep only one world-space acceleration hierarchy in the final runtime.

## 8) Testing and Acceptance Gates

### 8.1 Correctness gates

1. Multiplayer patch apply remains stable with no world disappearance/flicker regressions.
2. Client world cache remains authoritative and debuggable independent of render tree.
3. Render tree can be rebuilt incrementally without full-scene rebuild each patch.
4. Non-voxel and voxel content render under same camera/clip manifold.

### 8.2 Performance gates

1. No-op world patches do near-zero work in render tree update path.
2. Patch cost scales with changed region, not entire visible window.
3. Frame-time remains stable while approaching large structures at high render distance.
4. Render distance can be raised substantially beyond current defaults without immediate collapse.

## 9) Immediate File Touchpoints

1. Client patch/apply path:
   - `crates/polychora/src/app_multiplayer.rs`
   - `crates/polychora/src/scene.rs`
2. Client frame build:
   - `crates/polychora/src/scene/voxel_runtime.rs`
   - `crates/polychora/src/app_gameplay_loop.rs`
3. Renderer API and internals:
   - `src/render.rs`
   - `src/render/vte.rs`
   - `slang-shaders/src/voxel.slang` (and related shader modules)
4. Shared data structures:
   - `crates/polychora/src/shared/render_tree/`

## 10) Out of Scope for This MVP

1. Changing world-streaming protocol payloads away from `RegionTreeCore`.
2. Server-side render tree generation.
3. Full L1/L2/Ln server-driven LOD system.
4. Save format changes.

## 11) Definition of Done (MVP)

1. Client receives world patches as `RegionTreeCore`, maintains authoritative world tree.
2. Client derives and maintains root-BVH render tree from world tree incrementally.
3. Renderer consumes unified render-tree-backed input path for world geometry.
4. Legacy split path (voxel + non-voxel side-channel) is removed or hard-disabled by default.
5. Documented diagnostics show per-patch and per-frame costs are bounded and interpretable.
