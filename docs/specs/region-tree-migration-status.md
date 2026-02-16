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
- Stream refresh gating now uses per-player near-bounds `RegionClockMap` snapshots (instead of global `world_revision`) to avoid unnecessary out-of-area refresh work.
- Server now emits `WorldRegionClockUpdate` messages carrying updated region clocks on authoritative voxel edits/explosions.
- Stream sync now sends near-bounds region-clock snapshots in chunk-batch mode so clients can establish/refresh local clock baselines.
- Optional bridge mode: `R4D_STREAM_REGION_PATCH=1` emits `WorldRegionPatch` (`RegionSubtreePatch`) alongside chunk batches for live patch-path exercise.
- Optional bridge-only mode: `R4D_STREAM_REGION_PATCH_ONLY=1` emits/syncs via `WorldRegionPatch` without chunk load/unload batches.
- Bridge patch payloads now include region clock preconditions derived from each client's prior streamed region clock snapshot.
- Bridge patch `patch_seq` is now per-client monotonic (`PlayerState.next_stream_patch_seq`), decoupled from global `world_revision`.
- Client now applies `WorldRegionPatch` by validating preconditions, enforcing monotonic patch sequence, grafting bounded near-chunk payloads, and applying patch clock updates.
- Client now emits `WorldRegionResyncRequest` on patch sequence/precondition failures; server replies with bounded `WorldRegionPatch` built from requested regions.
- Bridge path now enforces `MAX_PATCH_BYTES` wire budget with recursive spatial split attempts; stream falls back to chunk batches only if splitting cannot satisfy budget, and resync replies return server error only when splitting still cannot satisfy budget.
- Authoritative voxel edits now route through `ServerWorldField::apply_voxel_edit`.
- Region tree naming is now unified in runtime-facing code:
  - `RegionChunkTree` (removed `RegionOverrideTree` alias)
  - `chunk_tree`/`chunk_at`/`set_chunk`/`remove_chunk` naming in `ServerWorldField`
  - `RegionTreeWorkingSet` helper naming now references chunk semantics (not override semantics)
- `RegionChunkTree` now exposes bounded chunk operations used by stream cache updates:
  - `collect_chunks_in_bounds`
  - `diff_chunks_in_bounds`
  - `apply_chunk_diff`
- Shared `RegionTreeCore` extraction helper now exists:
  - `collect_non_empty_chunks_from_core_in_bounds`
- Server now tracks internal `RegionClockMap` and bumps touched region clocks on authoritative voxel edits/explosions.
- Added reusable deterministic test tooling for tree semantics in `shared/worldfield_testkit.rs`:
  - deterministic RNG helpers
  - reference chunk-store model with canonicalization + bounded diff/apply behavior
- Added randomized integration-style coverage for final datastructures:
  - `RegionChunkTree` set/remove and bounded diff/apply invariants vs reference model
  - `ServerWorldField::query_region_core` coherence vs realized non-empty chunk materialization
- `RegionChunkTree` now supports direct core-driven bounded updates:
  - `diff_non_empty_core_in_bounds`
  - `apply_non_empty_core_in_bounds`
- `RegionTreeWorkingSet::refresh_from_core` now consumes tree-native core diff/apply output and carries direct `(ChunkPos, ChunkPayload)` load results (no intermediate desired-map rebuild).
- `ServerWorldField::query_region_core` now emits sparse branch topology by y-slice and trims each slice to realized non-empty x/z/w bounds (instead of always emitting full query-volume chunk arrays).

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
  - `WorldRegionClockUpdate`
  - `WorldRegionPatch` + `WorldRegionResyncRequest` bridge messages are now emitted/applied optionally, but chunk batches remain canonical.
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
1. Extend chunk-level diff ops into subtree-native patch/diff operations (replace-by-bounds, subtree trim/diff).
2. Replace chunk batch wire transport with `RegionSubtreePatch` + region clocks.
3. Move persistence from `VoxelWorld` bridge to canonical tree + region clocks.
4. Introduce handshake capabilities (`protocol_version`, `generator_manifest_hash`, `feature_bits`) for symbolic/refined tree transport policy.

## Acceptance checks for this stage
- No direct base+procgen composition logic remains in `server/mod.rs` stream planner.
- Player stream state is represented as tree + bounds, not chunk hash sets.
- Geometry entering stream cache is sourced from `WorldField` query responses.

## Current gaps vs region-tree-worldfield spec
- Replication still uses `world_revision` + chunk batch messages as canonical transport; `RegionSubtreePatch` is bridge-only.
- Bridge patch flow still lacks spec-complete behavior:
  - Client/server now use bounded resync requests with a client-side throttle interval, but full server-side coalesce/throttle policy from spec is still missing.
- Persistence still roundtrips through `VoxelWorld` bridge; canonical semantic-tree persistence is pending.
- `query_region_core` currently materializes bounded `ChunkArray` responses directly from base/procgen/chunk-tree composition rather than returning long-lived symbolic `ProceduralRef`/branch topology from a persistent semantic tree.
- Runtime realization cache key is currently `(chunk_key, profile)` in `ServerWorldField`; full snapshot/node-handle/generator-version keyed sidecar cache is pending.
