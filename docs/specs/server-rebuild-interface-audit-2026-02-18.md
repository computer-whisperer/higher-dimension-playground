# Server Rebuild Interface Audit (2026-02-18)

## Scope
This audit reflects current code-truth after the first hard-delete/stub pass for server world streaming.

## 1) Client <-> Server Wire Contract (Current)

### Messages currently in use
1. Client -> Server:
   - `Hello`
   - `UpdatePlayer`
   - `SetVoxel`
   - `SpawnEntity`
   - `ConsoleCommand`
   - `Ping`
2. Server -> Client:
   - `Welcome`
   - `Error`
   - `Pong`
   - `EntitySpawned`
   - `EntityDestroyed`
   - `EntityTransforms`
   - `Explosion`
   - `PlayerMovementModifier`

### World stream messages (current)
1. `ClientMessage::WorldSubtreeRequest` requests chunk-AABB subtree data directly.
2. `ServerMessage::WorldSubtreePatch` returns a subtree for requested bounds.

### Current world-stream behavior
1. `sync_streamed_chunks_for_client` still only hydrates persisted save regions in server memory.
2. `force_sync_streamed_clients_for_changed_chunks` is a temporary no-op.
3. `send_world_subtree_patch_to_client` handles direct request->query->patch for requested bounds.

Implication:
1. Multiplayer world transport is now request/response subtree fetch, not region-clock delta sync.
2. Entity replication and simulation traffic remains active.

## 2) Runtime Interfaces To Keep
1. `WorldField::query_region_core(query, detail)` as the only generator-facing world query API (`crates/polychora/src/server/world_field/mod.rs`).
2. `RegionChunkTree` as shared mutable tree structure for server/client caches and tree operations (`crates/polychora/src/shared/region_tree/tree.rs`).
3. `WorldOverlay` as the server interposer boundary between runtime and generator queries (`crates/polychora/src/server/world_field/mod.rs`).
4. Current overlay phase is passthrough/read-only with no cache semantics; mutation and streaming cache behaviors are deferred.

## 3) Interfaces/Paths Removed in This Pass
1. Legacy per-player streamed cache module removed:
   - `crates/polychora/src/server/region_tree_cache.rs`
2. Server region-clock and bounded patch planner path removed from `server/mod.rs`:
   - Region clock preconditions/updates map logic.
   - Patch splitting/wire-size budget planner.
   - Working-set-driven refresh planner.

## 4) Region Tree Audit Snapshot
The current shared tree implementation already exposes the core operations needed for rebuilt streaming:
1. Slice:
   - `slice_core_in_bounds`
   - `slice_non_empty_core_in_bounds`
2. Extract:
   - `take_non_empty_core_in_bounds`
3. Splice:
   - `splice_non_empty_core_in_bounds`
4. Lazy eviction:
   - `lazy_drop_outside_bounds`

Tests:
1. `cargo test -p polychora --lib shared::region_tree -- --nocapture`
2. Result: 28 passed, 0 failed.

## 5) Immediate Rebuild Targets
1. Reintroduce server world streaming with true tree deltas only:
   - Use server-side per-client tree state.
   - Compute delta by tree splice/slice/evict operations.
2. Replace pure request/response `WorldSubtreePatch` with true minimal delta emission once per-client state exists:
   - sequence semantics are deterministic,
   - patch bounds are minimal,
   - no-op patch generation is eliminated.
3. Keep protocol bounds-native (no fixed region-id lattice in streaming path).
