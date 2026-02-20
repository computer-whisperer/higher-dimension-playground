# Next-Gen Render Pipeline Plan (Root BVH + Leaf Traversal)

Status: Active implementation; architecture reset in progress  
Updated: 2026-02-20 (explicit GPU mutation contract)

## 1) Goal

Rebuild the client render pipeline so visible range scales from ~80 blocks to hundreds/thousands while preserving sparse 4D structure efficiency.

The target is:

1. A root BVH over renderable world regions.
2. Leaf-specialized traversal (uniform + voxel now, tetra next) under that root BVH.
3. Stable frame-time by moving heavy work out of per-frame camera updates.

This remains a hard architecture reset, not an incremental patch on legacy assumptions.

## 2) Current Snapshot and Problem Statement

The current system already has world-tree -> render-tree -> GPU traversal pieces, but profiling shows a critical issue:

1. `client-cpu` is dominated by `voxel-build` (~50 ms/frame in recent capture).
2. `update` and `render-submit` are low (<1 ms).
3. Multiplayer patch cost is not the active bottleneck in the sampled run.

Interpretation:

1. We are still coupling camera-frame updates to expensive scene rebuild/repack work.
2. The architecture needs explicit separation between:
   1. world/render scene mutation work,
   2. frame-time camera/render work.

## 3) Non-Negotiable Boundaries

1. Server/client world streaming remains `RegionTreeCore` patches (`WorldSubtreePatch`).
2. Client world cache (`RegionChunkTree`) remains authoritative world state on client.
3. Render structures are derived client-side from world cache (not wire protocol types).
4. Sparse, large `Uniform` regions must stay efficient end-to-end (no forced densification).
5. Renderer internals do not leak into network protocol contracts.
6. CPU render-tree is a planning/mirroring tool only; world patches must mutate the persistent on-GPU render scene directly.
7. Append-only BVH mutation is explicitly disallowed as a production path.
8. Camera movement must not trigger full scene/BVH rebuilds; rebuild is reserved for explicit fallback or async compaction.

## 4) Target Runtime Model

### 4.1 Three layers with clean ownership

1. World layer (authoritative): client `RegionChunkTree`.
2. CPU render-scene layer (derived): leaf set + CPU-side BVH/edit queue.
3. GPU render-scene layer (resident): persistent buffers + active GPU BVH.

### 4.2 Frame path invariant

Per-frame camera motion must only:

1. update uniforms/constants,
2. issue traversal/raster work,
3. avoid full scene repack/rebuild.

### 4.3 Mutation path invariant

World patch/edit events may do heavy work, but only through budgeted incremental updates and/or background rebuilds.

### 4.4 Patch execution contract (explicit)

1. Server patch ingress produces a CPU-side planned diff against the current render scene.
2. The output of planning is an explicit GPU mutation batch, not a rebuilt CPU snapshot.
3. GPU mutation batch updates resident arenas by handle/ID and updates root handle.
4. CPU mirror is then updated to match the applied mutation batch.
5. Rebuild path is fallback-only and must be observable in diagnostics.

## 5) Scene Representation

### 5.1 Leaf categories (MVP to final)

1. `LeafUniform(material)` (analytic fill; no dense payload required).
2. `LeafVoxelChunkArray(...)` (chunk-indexed dense payload regions).
3. `LeafTetSegmentRef(...)` (planned next phase).
4. Empty space is represented by BVH miss / absent leaves.

### 5.2 Sparse-first behavior

1. Large platforms should remain a handful of `LeafUniform` leaves.
2. Local edits split only where required, preserving large untouched uniform leaves.

## 6) GPU BVH Lifecycle (New Core Feature)

### 6.0 Required mutation model

The canonical mutation payload must be explicit arena edits:

1. `expected_root_id` (optional stale-check guard),
2. `new_root_id`,
3. `node_writes: Vec<(node_id, node_record)>`,
4. `leaf_writes: Vec<(leaf_id, leaf_record)>`,
5. `payload_writes: Vec<(payload_span_or_id, bytes)>`,
6. `freed_node_ids`,
7. `freed_leaf_ids`,
8. `freed_payload_spans`.

`appended_*_range` style deltas are not sufficient for long-running mutation and are not the target architecture.

### 6.1 Persistent active BVH with incremental edits

Keep the active BVH and payload buffers resident on GPU across frames.

On each dirty-region update:

1. Consume queued dirty bounds (budgeted).
2. Compute subtree/leaf edits in CPU planner.
3. Apply corresponding incremental edits to GPU pools:
   1. leaf records,
   2. node records,
   3. payload tables/ranges.
4. Refit affected BVH ancestors only.
5. Update CPU mirror to match the applied GPU scene.

No full-scene rebuild for ordinary movement/patch churn.

### 6.2 Asynchronous full rebuild/compaction

Run a background task that occasionally rebuilds an optimized BVH from a snapshot:

1. Snapshot CPU render-scene leaves at revision `R`.
2. Build compact/optimized BVH offline.
3. Upload into inactive GPU buffer set.
4. Swap active set atomically at frame boundary if revision is still compatible.
5. If scene diverged too far, discard and restart.

This allows temporary lopsided/stretch trees while preserving smooth frame-time, then compacts without spikes.

### 6.3 Swap model

Use double-buffered scene sets:

1. `ActiveSceneSet` (currently rendered).
2. `StagingSceneSet` (background-built).

Swap is pointer/root-handle update only, not a blocking rebuild on the render thread.

## 7) Scheduling and Budgeting

### 7.1 Per-frame mutation budgets

Introduce explicit budgets for mutation processing:

1. max CPU ms for scene edits,
2. max GPU upload bytes,
3. max edit ops/leaves per frame.

Overflow remains queued; rendering continues using the current active scene.

### 7.2 QoS priorities

1. Near-camera dirty edits first.
2. Coalesce overlapping dirty bounds before processing.
3. Defer low-priority far edits when over budget.

## 8) Frame-Time and Rebuild Triggers

### 8.1 Allowed frame-time triggers

1. Camera transform/uniform update.
2. Small bounded queue consumption (if budget allows).

### 8.2 Disallowed frame-time triggers

1. Full world-tree to GPU payload repack on look-direction change.
2. Full root BVH rebuild from camera-only movement.
3. Transitional async wrappers that still source frame-time updates from a full rebuild path.
4. Rebuilding CPU BVH snapshots per patch and re-uploading them as a substitute for explicit GPU mutation batches.
5. Append-only node/leaf growth without reclamation/free-list reuse.

## 9) Implementation Plan (Ordered)

### Phase A: Decouple frame build from scene rebuild

1. Split current `build_voxel_frame_data` into:
   1. `build_frame_uniforms(...)` (camera-only),
   2. `update_scene_if_needed(...)` (mutation-driven).
2. Remove camera-forward/voxel position from scene invalidation keys.
3. Keep scene revision keyed to world/tree mutations and explicit bounds changes only.

### Phase B: Incremental scene edit pipeline

1. Add dirty-bounds queue in scene/render subsystem.
2. Introduce stable GPU handles/IDs and free lists for nodes, leaves, and payload spans.
3. Implement bounded `apply_scene_edits_budgeted(...)` that emits explicit arena mutations.
4. Apply mutation batches directly to GPU-resident buffers by handle/index writes.
5. Add diagnostics for queue depth, bytes uploaded, edits applied per frame, and fallback rebuild count.

### Phase C: Async BVH compaction path

1. Add background BVH rebuild worker.
2. Add inactive/staging GPU scene set.
3. Implement safe swap at frame boundary.
4. Add guardrails for stale-snapshot cancellation.

### Phase D: Tetra leaf merge (follow-up)

1. Introduce tetra leaf payloads under the same root BVH.
2. Remove remaining non-voxel side-channel pipeline.

## 10) Diagnostics and Acceptance Gates

### 10.1 Required diagnostics

1. `scene_revision`, `active_bvh_revision`, `compaction_revision`.
2. mutation queue length and coalesced bounds count.
3. per-frame:
   1. scene edit CPU ms,
   2. GPU upload bytes,
   3. edits applied/deferred.
4. compaction:
   1. build ms,
   2. upload ms,
   3. swap success/fail reason.
5. mutation path correctness:
   1. explicit mutation op counts (writes/frees by arena),
   2. free-list occupancy,
   3. fallback rebuild trigger reason.

### 10.2 Correctness gates

1. No world disappearance/flicker regressions under multiplayer patch churn.
2. Uniform and chunk-array leaves render identically for same material semantics.
3. Incremental update path and full rebuild path converge to identical hit/material results.

### 10.3 Performance gates

1. Camera look-around does not trigger large `voxel-build` spikes.
2. Movement-induced interest expansion does not cause frame hitches from full rebuilds.
3. No-op patches do near-zero scene/GPU work.
4. High render-distance traversal remains bounded by GPU traversal cost, not CPU repack cost.

## 11) Immediate File Touchpoints

1. Client scene/runtime:
   - `crates/polychora/src/scene.rs`
   - `crates/polychora/src/scene/voxel_runtime.rs`
   - `crates/polychora/src/app_gameplay_loop.rs`
2. Client multiplayer patch ingress:
   - `crates/polychora/src/app_multiplayer.rs`
3. Renderer runtime/buffers:
   - `src/render.rs`
   - `src/render/vte.rs`
4. Shared render-tree structures:
   - `crates/polychora/src/shared/render_tree/`
5. Shader traversal path:
   - `slang-shaders/src/voxel.slang`

## 12) Out of Scope for This MVP

1. Protocol changes away from `RegionTreeCore`.
2. Server-side render tree generation.
3. Server-driven multi-LOD world format changes.
4. Save format changes.

## 13) Definition of Done (Current MVP Milestone)

1. World patches still arrive as `RegionTreeCore`; client world cache remains authoritative.
2. Active GPU BVH/payloads are persistent across frames and updated by explicit mutation batches (not append-only growth).
3. Camera-only movement no longer causes heavy scene rebuild/repack.
4. Background compaction path can rebuild and atomically swap optimized BVH sets.
5. Profile logs clearly separate frame-time rendering from mutation-time scene maintenance.
