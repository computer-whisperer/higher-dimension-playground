# Region Tree Migration Status

Status: Active migration (hard-cut mode)
Updated: 2026-02-18

## Core mandate (non-negotiable)
1. Runtime world authority must be tree-native end-to-end.
2. No new runtime compatibility layers.
3. `VoxelWorld` is migration-only, not server runtime authority.
4. Trees are the simplifying primitive; duplicated legacy representations are a blocker.

## Current code-truth snapshot
This reflects current runtime code paths, not target intent.

### Already in place
1. Server runtime world state is held in `ServerWorldField` with `RegionChunkTree` (`crates/polychora/src/server/world_field/legacy_generator.rs`).
2. Runtime v4 persistence path is chunk-payload/tree-based (`LoadedState.world_chunk_payloads`, `save_state_from_chunk_payloads`) (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).
3. Server startup loads v4 metadata without eagerly materializing all chunks; persisted block regions hydrate on demand for local bounds (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).
4. Legacy per-player `RegionTreeWorkingSet` runtime path has been removed from the server (`crates/polychora/src/server/region_tree_cache.rs` deleted).
5. Client scene runtime world cache remains tree-authoritative (`Scene.world_tree` with derived chunk payload cache), not `RegionChunkWorld`-based (`crates/polychora/src/scene.rs`).
6. `WorldField` API is now query-only (`query_region_core`), and `WorldOverlay` is currently a passthrough read-only interposer boundary (`crates/polychora/src/server/world_field/mod.rs`).

### Still hybrid (must be removed)
1. Multiplayer world streaming is currently stubbed server-side:
   - `sync_streamed_chunks_for_client` only hydrates persisted regions and does not plan per-client deltas.
   - `force_sync_streamed_clients_for_changed_chunks` is a no-op.
   - per-client delta planning is not implemented yet.
2. Protocol now uses bounds-based request/response transport (`WorldSubtreeRequest`, `WorldSubtreePatch`), but it is not yet optimized into minimal deltas (`crates/polychora/src/shared/protocol.rs`, `crates/polychora/src/server/mod.rs`).
3. Client now requests subtree bounds directly and applies returned subtree patches, but still needs smarter demand planning and caching policy (`crates/polychora/src/app_multiplayer.rs`).
4. Runtime still eagerly materializes full persisted entity lists at startup, and region hydration currently performs direct file I/O under the server lock (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).
5. Migration logic remains split into `save_v4_migration` and `legacy_migration` (acceptable as tooling boundary, but still needs strict runtime isolation checks).

## What is explicitly disallowed going forward
1. Adding any new `to_legacy_*` / `from_legacy_*` runtime bridge for world state.
2. Reintroducing chunk-batch/snapshot transport for world streaming.
3. Converting tree state to `VoxelWorld` merely to satisfy save APIs.
4. Maintaining dual world authorities in server runtime.

## Migration plan (hard cut, ordered)

### Step 1: Rebuild multiplayer world streaming around tree primitives
1. Server must compute client-visible subtree deltas against server-side per-client tree state using efficient region-tree splice/trim operations.
2. Server must only send true deltas (`added/changed/removed` via bounded patch cores), not full-window retransmits.
3. Resync/fill flow must be explicit and bounded: request/response bounds fetch for bootstrap, delta stream for steady state.
4. Keep protocol versioning strict; no hidden compatibility path.

Acceptance gate:
1. `sync_streamed_chunks_for_client` is not a stub.
2. `force_sync_streamed_clients_for_changed_chunks` targets only impacted clients/regions.
3. Multiplayer movement/edit traffic no longer causes no-op patch churn.

### Step 2: Save/runtime storage contract cleanup
1. Keep runtime world authority exclusively in tree-native structures.
2. Keep `VoxelWorld` and legacy readers/writers migration-only.
3. Keep dirty tracking region-scoped and tree-native, with no duplicate world authority caches.

Acceptance gate:
1. No runtime world conversion into `VoxelWorld` during tick, stream, or autosave.

### Step 3: Client authority and cache cleanup
1. Make multiplayer world cache tree-authoritative on client.
2. Materialize render/sim chunks from tree cache as needed; do not treat chunk hash map as authoritative state.
3. Keep patch-apply cost proportional to changed tree region, not window size.

Acceptance gate:
1. Patch apply path mutates client tree authority first; render cache is derived.

### Step 4: Legacy boundary containment
1. Restrict legacy world-file readers/writers to migration commands only.
2. Runtime startup for multiplayer server accepts v4 only.
3. Keep explicit versioned migration commands for older formats.

Acceptance gate:
1. Runtime world startup path has no v1/v2/v3 world-file dependency.

## Immediate blockers to resolve next
1. Replace current streaming stubs with real tree-delta planner and sender.
2. Add server-side per-client world-tree state (or equivalent) to avoid full-window recomputation.
3. Add targeted integration tests for stream delta correctness under:
   - player movement,
   - voxel edits,
   - out-of-order patch/reconnect recovery.
4. Move region hydration I/O off lock path into bounded worker/prefetch scheduling.

## Verification checklist for “migration complete”
1. `rg "from_legacy_world|to_legacy_world|SaveRequest \\{\\s*world: &'a VoxelWorld"` returns no runtime-path hits.
2. Multiplayer server load/save world path compiles and runs without constructing `VoxelWorld`.
3. Protocol remains patch-only and all patch/resync tests pass with non-stub server handlers.
4. Single-player and multiplayer smoke tests pass with equivalent world behavior.
