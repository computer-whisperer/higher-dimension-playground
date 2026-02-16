# Region Tree WorldField Spec (Draft v7)

Status: Draft  
Audience: server/runtime/procgen/networking  
Owner: world streaming and procgen stack  
Date: 2026-02-16

## 1. Goal

Replace flat-world chunk-first assumptions with a tree-native world model that scales for procedurally infinite 4D platform volumes.

Target shape:

- Infinite 4D space.
- Platforms hundreds of voxels in x/z/w and tens in y.
- Gravity on y axis.
- Structures spawning on platform surfaces.

## 2. Scope and Core Invariants

Phase 1 scope:

- Partition-BVH `RegionTree` as the shared spatial topology.
- Region-scoped replication/versioning.
- Local patch propagation for singular edits.
- Canonical persistence from semantic tree + region clocks.

Out of scope in Phase 1:

- Globally optimal partitioning.
- Full client execution of every generator variant.
- Secondary payload interning for identical non-uniform leaves.

Normative invariants:

- Tree topology is partition-BVH (disjoint children, no overlap).
- Runtime lopsided trees are valid; optimization is asynchronous.
- Protocol compatibility is exact-version at handshake; mismatch disconnects.
- Server/tree/wire authoritative state is single-resolution chunk space (no server-side LOD lattice).
- Patch applicability is region-scoped using region clocks.
- Every accepted patch atomically applies subtree graft + clock updates.
- Edits bump clocks for intersecting regions in the single-resolution lattice.
- Region clocks are the only authoritative version counters for replication and persistence contracts.
- Singular block edits must produce local region patches; full-tree resend is recovery-only.
- Network transport unit is `RegionSubtreePatch` (tree-native, not chunk-list batches).
- Canonical persisted state is semantic tree + region clocks only.

## 3. Data Model

### 3.1 Identifiers

- `ChunkKey { pos: [i32; 4] }`
- `RegionId` from fixed replication lattice in chunk space.
- `RegionClockMap: RegionId -> u64`.
- `NodeHandle` (opaque, snapshot-local runtime handle; never serialized).

Defaults:

- `REGION_TILE_EDGE_CHUNKS = 4`

Rules:

- Every chunk maps to exactly one `RegionId`.
- Region clocks are monotonic per `RegionId`.
- `RegionId` mapping is `floor_div(chunk_pos_axis, REGION_TILE_EDGE_CHUNKS)` for each axis.

### 3.2 Semantic Tree (`RegionTreeCore`)

Node kinds:

- `Empty`
- `Uniform(MaterialId)`
- `ProceduralRef(GeneratorRef)`
- `ChunkArray(ChunkArrayData)`
- `Branch(children)`

Notes:

- Repeated content is represented by specialized `ProceduralRef` generators; no separate `RepeatedChunk` node kind.
- `ChunkArray` is the bounded materialized form for finite, expensive procgen structures and concrete replication fallback.
- `MaterialId` is `u16` minimum.

Required fields:

- `bounds: Aabb4i` (chunk-aligned)
- `kind`
- `generator_version_hash: u64` (generator-dependent nodes)

### 3.2.1 `ChunkArray` Leaf (Bounded Materialized Region)

`ChunkArray` stores concrete chunk content over a finite chunk-aligned region without requiring one leaf node per chunk.

Fields:

- `bounds: Aabb4i` (chunk-aligned finite region)
- `chunk_palette: Vec<ChunkPayload>` (deduplicated concrete chunk payloads)
- `index_codec: ChunkArrayIndexCodec`
- `index_data: Vec<u8>` (codec-defined mapping from covered chunk cells to palette indices)
- `default_chunk_idx: Option<u16>` (implicit index for untouched/omitted cells)

`ChunkArrayIndexCodec`:

- `DenseU16` (runtime/internal codec for bounded, high-occupancy regions)
- `PagedSparseRle` (canonical wire/persist codec; runtime also supported)

Rules:

- Covered domain is exactly the chunk lattice inside `bounds`.
- `ChunkArray` must not force dense n^4 index storage when sparse codec is more efficient.
- Single-voxel edit inside a `ChunkArray` mutates one chunk payload via copy-on-write and updates only touched cell mapping.
- Palette growth or churn may trigger local split/repartition under normal tree optimization rules.
- `ChunkArray` content is logically inline in semantic tree nodes; external reference indirection is not part of the semantic model.
- Implementations may still use internal sharing (for example `Arc<ChunkArrayData>`) as an in-memory optimization.
- Canonical wire/persist serialization for `ChunkArray` MUST use `PagedSparseRle`.
- Canonical traversal order for chunk-cell indexing is x, y, z, w (minor to major).
- Canonical palette order is first appearance in canonical traversal; unused palette entries are forbidden.
- If a `ChunkArray` region is representable as `Empty` or `Uniform`, writer MUST collapse to that simpler node kind.
- Before replication, any `ChunkArray` exceeding `MAX_CHUNKARRAY_CELLS_PER_NODE` or `MAX_PATCH_BYTES` MUST be spatially split into smaller nodes; protocol-level continuation framing is out of scope.

Canonical `PagedSparseRle` layout requirements:

- Integer encoding is little-endian for all multi-byte numeric fields.
- Chunk-cell traversal order is x, y, z, w.
- Default page edge is `PAGED_RLE_PAGE_EDGE_CELLS = 4` (4^4 = 256 cells per page).
- Pages are emitted in canonical page-grid traversal order (x, y, z, w).
- Inside each page, runs are emitted in canonical cell traversal order.
- Runs are `(run_len, palette_idx)` pairs with `run_len >= 1`; adjacent runs with identical `palette_idx` are forbidden.
- Empty pages are represented by omission when `default_chunk_idx` is present; otherwise they are represented by a single run covering full page length.

### 3.3 Runtime Sidecar (`RegionRuntimeSidecar`)

Ephemeral state keyed by node identity (not serialized, not persisted):

- `pin_count_render: u32`
- `pin_count_sim: u32`
- `last_access_tick: u64`
- `realized_by_profile: [Option<RealizedEntry>; PROFILE_COUNT]`

`RealizedEntry`:

- `payload: ChunkPayload`
- `cache_generator_version_hash: u64`
- `cache_profile: RealizeProfile`
- `source_snapshot_epoch: u64`
- `last_access_tick: u64`

### 3.4 Realization Cache Key

Realization cache identity:

- `(source_snapshot_epoch, node_handle, realize_profile, generator_version_hash)`

Invalidation rule:

- Accepted patches invalidate overlapping cached entries by spatial intersection.
- Snapshot retirement invalidates all entries keyed by retired `source_snapshot_epoch`.

Phase 1 `RealizeProfile`:

- `Render`
- `Simulation`

Sidecar cache policy defaults:

- `Render` cache soft limit: 256 MiB, hard limit: 384 MiB.
- `Simulation` cache soft limit: 64 MiB, hard limit: 96 MiB.
- Eviction policy: weighted LRU within profile, never evict pinned entries, prefer evicting larger cold entries first.
- On hard-limit breach, insertion path must evict until under hard limit before accepting new entries.

### 3.5 Chunk Payload

Variable-size payload forms:

- `Empty`
- `Uniform(MaterialId)`
- `PalettePacked { palette: Vec<MaterialId>, bit_width: u8, packed_indices: Vec<u64> }`
- `Dense16 { materials: [u16; CHUNK_VOLUME] }`

### 3.6 Wire Tree (`RegionReplication`)

Wire nodes are semantic-only:

- `RegionWireNode = Empty | Uniform | ProceduralRef | ChunkArray | Branch`

Wire schema excludes runtime sidecar fields.

## 4. WorldField API

```rust
pub trait WorldField {
    fn query_region_core(
        &self,
        query: QueryVolume,
        detail: QueryDetail,
    ) -> Arc<RegionTreeCore>;

    fn query_content_bounds(
        &self,
        query: QueryVolume,
    ) -> Option<Aabb4i>;

    fn realize_chunk(
        &mut self,
        key: ChunkKey,
        profile: RealizeProfile,
    ) -> ChunkPayload;
}
```

Semantics:

- `query_region_core` is the planning/trim source of truth.
- `query_content_bounds` returns a conservative bounding box for non-empty content inside `query`; it may over-approximate but must never under-approximate.
- `query_content_bounds` should avoid forced realization where possible.
- If exact bounds exceed budget, implementation should return a conservative coarse bounds quickly (including full `query` if needed).
- `realize_chunk` uses sidecar cache key from section 3.4.

## 5. Topology and Mutation Rules

Mandatory partition invariants:

- Child bounds are disjoint and contained by parent bounds.
- Missing child coverage is implicitly `Empty`.
- Any chunk query descends through at most one child per level.

Default (recommended, non-normative) split/merge policy:

- Split on longest axis, ties x/y/z/w, midpoint plane.
- Collapse equivalent children (`Empty` all or same `Uniform`).
- Merge `ProceduralRef` only on exact generator id/params/version hash match.
- Normalize child order lexicographically by min corner.

Optimization passes may rebalance/refit/repartition snapshot-local subtrees, but correctness cannot depend on optimization timing.

## 6. Patch Contract (Normative)

### 6.1 Patch Schema

Primary message:

- `RegionSubtreePatch { patch_seq, bounds, preconditions, clock_updates, subtree }`

Transport defaults:

- `MAX_PATCH_BYTES = 1_048_576` (1 MiB serialized)
- `MAX_PATCHES_PER_TICK_PER_CLIENT = 16`
- `MAX_CHUNKARRAY_CELLS_PER_NODE = 256`

Preconditions:

- `RegionClockPrecondition { region_id, expected_clock }`

Clock updates:

- `RegionClockUpdate { region_id, new_clock }`

Resync request:

- `RegionResyncRequest { regions: Vec<RegionResyncEntry> }`
- `RegionResyncEntry { region_id, client_clock }`

Ordering rules:

- `patch_seq` is per-client and strictly monotonic (`+1` for each sent patch).
- Client applies patches in `patch_seq` order only.
- If client detects a gap or duplicate `patch_seq`, it rejects out-of-order patch and requests bounded resync for covered regions.
- Server must spatially split patches until each serialized patch is within `MAX_PATCH_BYTES`.
- When a subtree is split into multiple patches, emission order is canonical by patch bounds min corner (x, y, z, w).

### 6.2 Server Obligations per Edit

For `SetVoxelIntent { pos, material, client_edit_id }`, server MUST:

1. Deduplicate by `(client_id, client_edit_id)`.
2. Apply copy-on-write mutation in the minimal touched chunk-sized region.
3. Stop if effective value is unchanged.
4. Bump clocks for intersecting regions.
5. Invalidate overlapping sidecar realized entries for relevant profiles only.
6. Emit minimal per-client `RegionSubtreePatch` intersected with client interest.
7. Persist only dirty regions touched by payload changes.

### 6.3 Client Apply Contract

1. Validate all region clock preconditions.
2. If valid: atomically `graft_replace(bounds, subtree)` and apply `clock_updates`.
3. If invalid: reject and request bounded region resync (`RegionResyncRequest { regions }` with current `client_clock` values).
4. Coalesce and throttle repeated resync failures for same regions.

### 6.4 Procedural Edit Locality

When edits target space currently represented by `ProceduralRef`, server MUST:

1. Split to touched chunk coverage only.
2. Materialize touched chunks only.
3. Write overrides at touched leaves.
4. Keep untouched surrounding space symbolic `ProceduralRef`.
5. Emit region-scoped patches only for affected regions.

When edits target space represented by `ChunkArray`, server MUST:

1. Materialize/mutate touched chunk payload only.
2. Preserve unaffected chunk palette entries and index cells.
3. Emit region-scoped patches only for affected regions.

### 6.5 Symbolic `ProceduralRef` Transport Rule

Exactly one compatibility rule:

- Symbolic `ProceduralRef` leaves are allowed over wire only when peers have exact `generator_manifest_hash` match from handshake capabilities; otherwise sender must transmit concrete content (`ChunkArray`, `Uniform`, and/or `Empty` leaves) for the affected region.
- When transmitting concrete `ChunkArray` content, sender MUST canonicalize to `PagedSparseRle` and enforce pre-send spatial split limits from section 3.2.1.

### 6.6 Handshake Capability Contract

`WorldField` replication handshake MUST exchange:

- `protocol_version: u16` (exact match required; mismatch disconnects)
- `generator_manifest_hash: u64`
- `feature_bits: u64`

Rules:

- Version mismatch disconnects before world replication starts.
- Generator manifest mismatch does not require disconnect; it disables symbolic `ProceduralRef` transport and forces concrete fallback.
- `feature_bits` unknown-required mismatch disconnects; unknown-optional bits are ignored.

## 7. Streaming and Realization

Planner flow:

1. Query semantic tree for union of client interest volumes.
2. Traverse non-empty frontier.
3. Realize only chunks needed for visible delivery and simulation guarantees.
4. Keep deep enclosed solids symbolic.

Simulation requirements override visibility-only culling.

Client render LOD note:

- Client-side LOD generation and caching are intentionally out of scope for server/tree/wire format in this spec.
- Any client LOD system must be derived from replicated single-resolution semantic state and must not alter protocol contracts.

Policy for heavy structures:

- Server should prefer emitting bounded `ChunkArray` nodes for finite high-cost structures (for example maze-like structures) so clients do not rerun expensive procgen for identical content.

## 8. Concurrency and Persistence

Concurrency:

- Single-writer mutation over semantic tree.
- Readers use immutable snapshots.
- Sidecar entries are version-aware and epoch-scoped.
- Retired snapshot sidecar entries become GC-eligible when reader refcount reaches zero.

Persistence:

Persist only:

- Canonical `RegionTreeCore` semantic snapshot.
- `RegionClockMap`.
- Referenced generator descriptors/version hashes.

Do not persist:

- Runtime sidecar metadata.
- Realized chunk cache entries.

Optional WAL/journal is recovery-only and must fold into canonical snapshot before export.

Persistence cadence defaults:

- World snapshot save interval: 5 seconds when dirty.
- Player record refresh interval: 60 seconds when dirty.
- Force save on orderly shutdown.

Atomic commit procedure:

1. Write payload files to `*.tmp` paths, flush, and `fsync`.
2. Rename each payload temp file to final path.
3. Write manifest to `manifest.json.tmp`, flush, and `fsync`.
4. Rename manifest temp to `manifest.json` (commit point).
5. `fsync` save root directory.

Recovery rule:

- Loader treats manifest as source of truth; orphan temp files are ignored and may be GCed.

## 9. Migration Plan

### Phase A: Core/Sidecar Split

- Introduce `RegionTreeCore` + `RegionRuntimeSidecar`.
- Remove runtime metadata and semantic `local_revision` from nodes.

### Phase B: Region Clock Lattice

- Introduce fixed `RegionId` mapping and `RegionClockMap`.
- Replace root-scoped checks with region-scoped preconditions.

### Phase C: Tree-Native Replication

- Replace chunk-list protocol with `RegionSubtreePatch`.
- Implement trim/serialize/deserialize/graft and bounded region resync.
- Add `patch_seq`, patch size splitting, and handshake capability exchange (`protocol_version`, `generator_manifest_hash`, `feature_bits`).

### Phase D: Payload + Realization Profiles

- Route realization through profile-keyed sidecar cache.
- Use variable payload encodings for realized leaves internally, while canonicalizing `ChunkArray` wire/persist output.
- Implement canonical `PagedSparseRle` writer/reader and collapse-to-`Empty`/`Uniform` fast path.

### Phase E: Platform-Volume Procgen

- Implement macro-cell platform generator dominated by `Uniform` and `ProceduralRef` output.
- Spawn structures on platform top surfaces.
- Materialize finite heavy structures into bounded `ChunkArray` regions for replication efficiency.

## 10. Tuning Knobs and Risks

- Region tile edge default (`REGION_TILE_EDGE_CHUNKS = 4`) may require retuning for very dense edit workloads.
- Sustained local edits can fragment trees and require rebuild triggers.
- Over-culling can violate simulation correctness if simulation domain rules are weak.
- Resync storms require region-level coalescing/throttling.
- Sidecar cache defaults may need retuning under high-entity multiplayer load.
- Platform generator layering policy (`Uniform` first vs mixed symbolic) remains to tune.
- Patch/ChunkArray size defaults may need telemetry-driven adjustment.

## 11. Success Metrics

- No flat-floor assumptions in streaming or realization.
- Singular block edits produce bounded local patches.
- Empty-space memory scales with tree complexity, not scanned volume.
- Stable server tick under multi-client dense platform scenes.
- Deterministic outputs for fixed seed, generator versions, and edit history.
