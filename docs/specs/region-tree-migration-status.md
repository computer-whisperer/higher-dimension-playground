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

### Still hybrid (must be removed)
1. `save_v4::load_state` materializes a `VoxelWorld` and returns it in `LoadedState.world` (`crates/polychora/src/save_v4.rs`).
2. Server bootstrap still bridges through `ServerWorldField::from_legacy_world(loaded.world, ...)` (`crates/polychora/src/server/mod.rs`).
3. `save_v4` public API still exposes `VoxelWorld` contracts (`SaveRequest { world: &VoxelWorld }`, `all_block_regions(world: &VoxelWorld, ...)`) (`crates/polychora/src/save_v4.rs`).
4. Client applies region patches by writing chunk payloads into `scene.world` chunk storage; tree is not yet the sole client-side authority (`crates/polychora/src/app_multiplayer.rs`).

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
1. Replace `save_v4::LoadedState.world: VoxelWorld` with tree-native loaded world representation.
2. Remove server bootstrap bridge from loaded `VoxelWorld` into `ServerWorldField`.
3. Move any remaining server autosave/world-region operations to tree-native persistence interfaces only.

## Verification checklist for “migration complete”
1. `rg "from_legacy_world|to_legacy_world|SaveRequest \\{\\s*world: &'a VoxelWorld"` returns no runtime-path hits.
2. Multiplayer server load/save world path compiles and runs without constructing `VoxelWorld`.
3. Protocol remains patch-only and all patch/resync tests pass.
4. Single-player and multiplayer smoke tests pass with equivalent world behavior.
