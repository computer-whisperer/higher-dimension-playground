# Region Tree Migration Status

Status: Working document  
Updated: 2026-02-16

## Scope
Track migration from legacy chunk-first runtime to tree-native world/query/mutation/streaming.

## Completed in this pass (Step 1-2)
- Server runtime world authority is now `ServerWorldField` (owned base + procgen + blocked keepout + chunk tree state).
- `ServerState` now stores only `world: ServerWorldField` for world state; split base/procgen/overlay state was removed.
- Dedicated region-tree cache data structures now live in `server/region_tree_cache.rs`:
  - `RegionTreeWorkingSet`
  - `RegionTreeRefreshResult`
- Player near-stream cache now uses that dedicated data structure:
  - `PlayerState.working_set: RegionTreeWorkingSet`
- Stream sync now uses world-field query only:
  1. compute interest bounds
  2. `query_region_core(bounds, Exact)`
  3. refresh/graft through `RegionTreeWorkingSet::refresh_from_core`
  4. diff old/new payload maps via `RegionTreeRefreshResult` for load/unload
- Authoritative voxel edits now route through `ServerWorldField::apply_voxel_edit`.
- Region tree naming is now unified in runtime-facing code:
  - `RegionChunkTree` (removed `RegionOverrideTree` alias)
  - `chunk_tree`/`chunk_at`/`set_chunk`/`remove_chunk` naming in `ServerWorldField`
  - `RegionTreeWorkingSet` helper naming now references chunk semantics (not override semantics)

## System Status Matrix

### Finalized (for current phase)
- `ServerWorldField` as single runtime world facade in server internals.
- Working-tree based near-player stream cache and tree-based diffing.
- Dedicated reusable region-tree cache module (no ad-hoc tree helper block in `server/mod.rs`).
- World-query path for stream planning (`query_region_core`) without ad-hoc base/procgen synthesis in `server/mod.rs`.

### Temporary bridge layers (intentional)
- Save/snapshot bridge still converts through `VoxelWorld` (`ServerWorldField::to_legacy_world`).
- Wire protocol still uses:
  - `WorldChunkBatch`
  - `WorldChunkUnloadBatch`
  - `WorldVoxelSet`
- Region refresh now applies per-chunk diff between prior/new working-set views; subtree-native patch ops are still pending.

### Old / to-be-replaced
- Legacy chunk-list replication model (not region-subtree patch transport).
- World persistence contract based on `VoxelWorld` + region blobs, instead of semantic `RegionTreeCore` + region clocks.
- Server/client replication versioning based on `world_revision`, not region clock preconditions.

## Current Ownership Boundaries
- `ServerWorldField` owns:
  - base world kind and floor chunk realization
  - procgen seed + enabled flag
  - procgen keepout blocked cells
  - unified `RegionChunkTree`
  - realization cache
  - world dirty state
- Server runtime (`server/mod.rs`) owns:
  - entity simulation and gameplay orchestration
  - per-player working trees for streamed interest windows
  - transport encoding and batching (legacy wire for now)

## Next Migration Targets
1. Add explicit tree patch/diff operations (replace-by-bounds, subtree trim/diff) to avoid brute-force cell clears.
2. Replace chunk batch wire transport with `RegionSubtreePatch` + region clocks.
3. Move persistence from `VoxelWorld` bridge to canonical tree + region clocks.
4. Introduce handshake capabilities (`protocol_version`, `generator_manifest_hash`, `feature_bits`) for symbolic/refined tree transport policy.

## Acceptance checks for this stage
- No direct base+procgen composition logic remains in `server/mod.rs` stream planner.
- Player stream state is represented as tree + bounds, not chunk hash sets.
- Geometry entering stream cache is sourced from `WorldField` query responses.

## Current gaps vs region-tree-worldfield spec
- Replication still uses `world_revision` + chunk batch messages; region clocks and `RegionSubtreePatch` are not implemented yet.
- Persistence still roundtrips through `VoxelWorld` bridge; canonical semantic-tree persistence is pending.
- `query_region_core` currently materializes bounded `ChunkArray` responses directly from base/procgen/chunk-tree composition rather than returning long-lived symbolic `ProceduralRef`/branch topology from a persistent semantic tree.
- Runtime realization cache key is currently `(chunk_key, profile)` in `ServerWorldField`; full snapshot/node-handle/generator-version keyed sidecar cache is pending.
