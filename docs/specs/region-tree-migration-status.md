# Region Tree Migration Status

Status: Active migration (hard-cut mode)  
Updated: 2026-02-17

## Core mandate (non-negotiable)
1. Runtime world authority must be tree-native end-to-end.
2. No new runtime compatibility layers.
3. `VoxelWorld` is migration-only, not server runtime authority.
4. Trees are the simplifying primitive; duplicated legacy representations are a blocker.

## Current code-truth snapshot
This reflects current runtime code paths, not target intent.

### Already in place
1. Server runtime world state is held in `ServerWorldField` with `RegionChunkTree` (`crates/polychora/src/server/world_field.rs`).
2. Multiplayer world transport is patch-only (`WorldRegionPatch` + `WorldRegionResyncRequest`) (`crates/polychora/src/shared/protocol.rs`).
3. Per-player streamed cache is tree-based (`RegionTreeWorkingSet`) (`crates/polychora/src/server/region_tree_cache.rs`).
4. Server stream planner queries `WorldField` and builds bounded tree patches (no legacy snapshot/chunk-batch protocol path).
5. Client scene runtime world cache is tree-authoritative (`Scene.world_tree` + explicit chunk payload cache), not `RegionChunkWorld`-based (`crates/polychora/src/scene.rs`).
6. Runtime v4 persistence path is chunk-payload/tree-based (`LoadedState.world_chunk_payloads`, `save_state_from_chunk_payloads`) (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).
7. Server startup now loads v4 metadata (global/players/entities/index) without eagerly materializing all world chunks; persisted block regions are hydrated on demand for stream/resync/edit bounds (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).

### Still hybrid (must be removed)
1. Migration logic is split into `save_v4_migration` (`crates/polychora/src/save_v4_migration.rs`); runtime `save_v4` no longer owns migration entrypoints.
2. Client scene still stores both tree metadata and materialized explicit chunk payloads (`world_tree` + `world_chunks`) rather than a stricter single-structure cache contract (`crates/polychora/src/scene.rs`).
3. Runtime still eagerly materializes full persisted entity lists at startup, and region hydration currently performs direct file IO on the server lock path; this needs a dedicated streaming/cache scheduler (`crates/polychora/src/save_v4.rs`, `crates/polychora/src/server/mod.rs`).

## What is explicitly disallowed going forward
1. Adding any new `to_legacy_*` / `from_legacy_*` runtime bridge for world state.
2. Reintroducing chunk-batch/snapshot transport for world streaming.
3. Converting tree state to `VoxelWorld` merely to satisfy save APIs.
4. Maintaining dual world authorities in server runtime.

## Migration plan (hard cut, ordered)

### Step 1: Save API cutover (server-facing)
1. Add tree-native load/save surfaces in `save_v4`:
   - load returns base-world metadata + region tree payloads + entities/players.
   - save accepts base-world metadata + chunk/tree payload deltas + entities/players.
2. Keep `VoxelWorld`-based helpers only in explicit migration tooling paths (CLI/menu migration), not runtime server load/save.
3. Remove server runtime dependency on `LoadedState.world: VoxelWorld`.

Acceptance gate:
1. `server/mod.rs` does not call `ServerWorldField::from_legacy_world`.
2. Server load/save path uses only tree payload APIs.

### Step 2: Runtime world authority cleanup
1. Remove `ServerWorldField::{from_legacy_world,to_legacy_world}` from runtime flow.
2. Keep a single in-memory world authority: base/procgen + override tree in `ServerWorldField`.
3. Keep dirty tracking region-scoped and tree-native.

Acceptance gate:
1. No runtime world conversion into `VoxelWorld` during tick, stream, or autosave.

### Step 3: Client authority cleanup
1. Make multiplayer world cache tree-authoritative on client.
2. Materialize render/sim chunks from tree cache as needed; do not treat chunk hash map as authoritative state.
3. Keep protocol unchanged (already patch-only) while replacing internal representation.

Acceptance gate:
1. Patch apply path mutates client tree authority first; render cache is derived.

### Step 4: Legacy boundary containment
1. Restrict legacy world-file readers/writers to migration commands only.
2. Runtime startup for multiplayer server accepts v4 only.
3. Keep explicit versioned migration commands for older formats.

Acceptance gate:
1. Runtime world startup path has no v1/v2/v3 world-file dependency.

## Immediate blockers to resolve next
1. Move on-demand world region hydration off the server lock path and introduce bounded prefetch/worker scheduling.
2. Decide and enforce client-side cache contract (`world_tree` sole authority with derivable chunk cache semantics).
3. Add true streaming entity-region load/unload (runtime currently loads all persisted entities at startup).
4. Keep trimming remaining runtime/test helper overlap inside `save_v4` where it no longer serves runtime behavior.

## Verification checklist for “migration complete”
1. `rg "from_legacy_world|to_legacy_world|SaveRequest \\{\\s*world: &'a VoxelWorld"` returns no runtime-path hits.
2. Multiplayer server load/save world path compiles and runs without constructing `VoxelWorld`.
3. Protocol remains patch-only and all patch/resync tests pass.
4. Single-player and multiplayer smoke tests pass with equivalent world behavior.
