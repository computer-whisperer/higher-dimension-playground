# Polychora Save Format v4 (Target Spec)

## Status
- This document defines the intended v4 on-disk format.
- Deployed v3 worlds are documented in `SAVE_FORMAT_V3_SPEC.md`.
- Historical note: this target was previously mislabeled as v3.

## v4 Ship Scope
1. Directory save root with generation-swap commits.
2. Single world index file per generation (`index/ix-main.g*.v4ix`).
3. Index fully loaded in memory at runtime.
4. BVH-style 4D override tree in that index.
5. Append-only immutable data blob files (`data/dt-*.v4dt`).
6. Sparse override storage only (virgin space is implicit).
7. Players persisted separately from world override index/data.

## Deferred (v4.1+)
1. Multi-page index trees and subtree sharding.
2. Coarse supercell directory for index partitioning.
3. Online compaction and advanced GC scheduling.
4. WAL/transaction journaling.

## Primary Goal
Support very large mostly-virgin worlds where persisted storage only captures sparse overrides, and where topology edits in the index do not rewrite unchanged dense chunk payload data.

## Version Contract
1. `manifest.version = 4` means this spec.
2. `manifest.version = 3` means deployed v3 (`SAVE_FORMAT_V3_SPEC.md`).
3. Loaders must fail on unknown versions with explicit upgrade errors.

## Core Invariants (Normative)
1. World blocks are represented as `overlay(procgen_base, persisted_overrides)`.
2. Persisted override storage must be sparse; virgin space must not be serialized as explicit chunk payload records.
3. Absence of override coverage for a chunk means "use procedural/base result".
4. Persisted world-node kinds mirror in-memory region node kinds, except `ProceduralRef` which is excluded from persisted v4 world index nodes.
5. Index topology and data payload storage are decoupled:
   - topology rewrites rewrite index/global/players generation files
   - topology rewrites must not rewrite unchanged chunk payload blobs
6. Data blob records are immutable after append.
7. Data files are append-only during normal operation.
8. Players are persisted outside world spatial override storage.
9. `custom_global_payload` is reserved and must not be primary world block storage.

## Save Root Layout

```text
<world>/
  manifest.json
  global.g0000000000000000.v4g
  players.g0000000000000000.v4p
  index/
    ix-main.g0000000000000000.v4ix
  data/
    dt-000001.v4dt
    dt-000002.v4dt
    ...
```

## Manifest (`manifest.json`)
Required fields:
1. `format: "polychora-save"`
2. `version: 4`
3. `current_generation: u64`
4. `created_ms: u64`
5. `last_modified_ms: u64`
6. `active_data_file_id: u32`
7. `data_file_count: u32`
8. `index_file: String`
9. `global_file: String`
10. `players_file: String`
11. `limits: SaveLimits`

`SaveLimits` initial defaults:
1. `data_file_max_bytes: u64` (`128 MiB`)
2. `index_soft_max_bytes: u64` (`8 MiB`)
3. `chunk_payload_target_bytes: u32` (`128 KiB`)
4. `chunk_payload_hard_max_bytes: u32` (`512 KiB`)
5. `entity_blob_target_bytes: u32` (`64 KiB`)
6. `entity_blob_hard_max_bytes: u32` (`256 KiB`)

Commit atomicity:
1. Append new data blob records.
2. Build next-generation `ix-main`, `global`, and `players` files.
3. `fsync` new files.
4. Atomically swap `manifest.json` to the next generation.
5. `fsync` save root directory.

## Global Payload (`global.g*.v4g`)
Payload fields:
1. `base_world_kind`
2. `world_seed`
3. `procgen_manifest_hash`
4. `next_entity_id`
5. `next_data_file_id`
6. `last_modified_ms`
7. `player_entity_hints`
8. `custom_global_payload` (opaque reserved bytes)

## Players Payload (`players.g*.v4p`)
Payload fields:
1. `players: Vec<PlayerRecord>`

Durability intent:
- Player progression/inventory durability is independent from world override churn.

## Index File (`index/ix-main.g*.v4ix`)

Header fields:
1. `magic: "V4IX"`
2. `format_version: u32`
3. `generation: u64`
4. `root_node_id: u32`
5. `node_count: u32`
6. `entity_root_node_id: Option<u32>`
7. `payload_crc32: u32`

Body fields:
1. `nodes: Vec<IndexNode>`

`IndexNode` fields:
1. `node_id: u32`
2. `bounds_min_chunk: [i32;4]`
3. `bounds_max_chunk: [i32;4]`
4. `kind: IndexNodeKind`

`IndexNodeKind`:
1. `Branch { child_node_ids: Vec<u32> }`
2. `LeafEmpty`
3. `LeafUniform { material: u16 }`
4. `LeafChunkArray { chunk_array_ref: BlobRef }`

Notes:
1. Persisted v4 world index intentionally excludes `LeafProceduralRef`.
2. `LeafEmpty` is an explicit empty override (distinct from virgin/absent coverage).

`BlobRef` fields:
1. `data_file_id: u32`
2. `record_offset: u64`
3. `record_len: u32`
4. `record_crc32: u32`
5. `blob_type: u8`
6. `blob_version: u16`

### 4D Geometric Semantics (Normative)
1. Each node defines a closed 4D axis-aligned chunk box in `Z^4`:
   `B = { p in Z^4 | min_i <= p_i <= max_i }`.
2. Branch child bounds must be fully contained in parent bounds.
3. Committed branch siblings must be pairwise chunk-disjoint:
   for siblings `A`, `B`, `A != B => B_A intersect B_B = empty`.
4. Equivalent statement: sibling 4D hypervolume intersection is zero.
5. Boundary contact is allowed only at codimension `>= 1` boundaries
   (3D cells, 2D faces, 1D edges, 0D corners).
6. Temporary overlap is allowed only in in-memory edit staging; committed file output must be normalized to disjoint siblings.
7. Writer must run commit-time validation and reject any overlap violation.

### Canonical Ordering
1. Branch child lists must be deterministic and stable.
2. Canonical order key: `(min_x, min_y, min_z, min_w, max_x, max_y, max_z, max_w, node_id)`.

## Normalization and Determinism (Normative)
1. Runtime edit application may create temporary overlapping nodes in memory.
2. Before commit, writer must normalize to a committed tree that satisfies 4D disjoint-sibling invariants.
3. Normalization precedence for overlap resolution is deterministic and stable: latest edit epoch wins; ties break by canonical order.
4. Writer should merge adjacent siblings when:
   - bounds are merge-compatible into one 4D axis-aligned box, and
   - node kind and payload identity are equal.
5. Writer must remove redundant overrides when realized leaf content equals procedural/base result for the same bounds (return-to-virgin).
6. Writer must validate disjointness and containment after normalization and reject commit on violation.
7. Node IDs are reassigned from canonical traversal order for each committed generation.
8. Equivalent logical trees must serialize to byte-identical `ix-main` payload bytes.

## Data File Container (`data/dt-*.v4dt`)
File header:
1. `magic: "V4DT"`
2. `format_version: u32`
3. `data_file_id: u32`

Each record:
1. `record_magic: "RECD"`
2. `blob_type: u8`
3. `blob_version: u16`
4. `payload_len: u32`
5. `payload_crc32: u32`
6. `payload bytes`

Blob types:
1. `blob_type=1`: `ChunkPayloadBlob`
2. `blob_type=2`: `ChunkArrayBlob`
3. `blob_type=3`: `EntityBlob`

`ChunkPayloadBlob`:
1. `encoding: ChunkEncoding` (`Uniform`, `PalettePacked`, `Dense16`)
2. `payload`

`ChunkArrayBlob`:
1. `volume_min_chunk: [i32;4]`
2. `volume_max_chunk: [i32;4]`
3. `payload_palette: Vec<BlobRef>` (refs to `ChunkPayloadBlob`)
4. `index_codec`
5. `index_data: Vec<u8>`
6. `default_palette_index: Option<u16>`

`EntityBlob`:
1. `volume_min_chunk: [i32;4]`
2. `volume_max_chunk: [i32;4]`
3. `entities: Vec<PersistedEntityRecord>`

## Query Semantics
For query bounds `Q`:
1. Evaluate procedural/base world for `Q` lazily.
2. Traverse in-memory index tree for intersecting override nodes.
3. Materialize referenced blobs only for intersecting override leaves.
4. Apply override overlay onto procedural/base result.
5. Return merged fragment.

If no override leaf intersects `Q`, result is pure procedural/base with no chunk payload blob reads.

## Edit and Save Semantics
For each edit transaction:
1. Compute affected bounds.
2. Compute baseline procedural/base result for those bounds.
3. Compare edited result vs baseline.
4. Remove override coverage where edited result equals baseline.
5. Insert/update override leaves only for true deltas.
6. Normalize tree to disjoint sibling invariants.
7. Rebalance locally (optional) without changing unchanged blob refs.
8. Append only new/changed data blobs.
9. Reuse existing blob refs when payload bytes match.
10. Write next-generation index/global/players and swap manifest.

## No-Thrash Rules (Normative)
1. Splitting/merging/reparenting nodes must not force rewrite of unchanged `ChunkPayloadBlob` or `ChunkArrayBlob` records.
2. Topology-only saves must append zero block-data blobs.
3. New block-data blobs are permitted only when payload bytes actually change.
4. Identical payload bytes should reuse existing blob refs when known.

## Reachability and Compaction
1. Live blobs are those reachable from the current generation index/entity roots.
2. Unreachable blobs may remain until compaction.
3. Compaction is offline/maintenance and must preserve logical world content.

## Startup Recovery
1. Read `manifest.json`.
2. Load generation-referenced `ix-main`, `global`, and `players`.
3. Validate checksums for referenced metadata/index files.
4. Validate blob checksums when dereferenced.
5. Ignore unreferenced newer files.
6. Fail fast on missing referenced files or checksum mismatch.

## Migration From v3
1. Input: deployed v3 world (`manifest.version = 3`).
2. Materialize canonical block overrides + entities + players + global opaque bytes.
3. Build v4 single-file BVH index and append referenced blobs.
4. Write v4 output generation.
5. Verify equivalence before finalizing migration output.

Equivalence checks:
1. Block materialization equality over validated sample/coverage bounds.
2. Entity identity/payload equality.
3. Player record equality.
4. Global opaque payload byte equality.

## Testing Checklist
1. Empty-world v4 roundtrip.
2. Sparse far-apart edits without index/data explosion.
3. Topology rebalance with zero unchanged blob rewrites.
4. Partial 4D volume fetch overlays correctly against procedural baseline.
5. Returning edited regions to virgin removes redundant overrides.
6. Crash between blob append and manifest swap preserves prior committed generation.
7. Corrupt checksum detection for index/global/players/blob records.
