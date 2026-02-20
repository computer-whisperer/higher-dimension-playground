# Polychora Save Format v3 (Implemented/Deployed)

## Status
- This document is the normative description of the format currently written/read by `crates/polychora/src/save_v3.rs`.
- Existing worlds with `manifest.version = 3` are expected to match this file, not the v4 target design.
- The intended tree-index successor spec now lives in `SAVE_FORMAT_V4_SPEC.md`.

## Scope
- Directory-based save root with generation-swapped metadata files.
- Append-only data pack files for blob payload records.
- Flat region-leaf index (`Vec<LeafEntry>`), not a hierarchical index-node tree.

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
```

Notes:
- Generation file names are produced from `current_generation`.
- Data files are monotonic by `data_file_id` and append-only.

## Manifest (`manifest.json`)
Fields:
1. `format: "polychora-save"`
2. `version: 3`
3. `current_generation: u64`
4. `created_ms: u64`
5. `last_modified_ms: u64`
6. `active_data_file_id: u32`
7. `data_file_count: u32`
8. `index_file: String`
9. `global_file: String`
10. `players_file: String`
11. `limits: SaveLimits`

`SaveLimits` fields:
1. `region_chunk_edge: i32` (default `4`)
2. `data_file_max_bytes: u64` (default `128 MiB`)
3. `index_soft_max_bytes: u64` (default `4 MiB`)
4. `block_blob_target_bytes: u32` (default `128 KiB`)
5. `block_blob_hard_max_bytes: u32` (default `512 KiB`)
6. `entity_blob_target_bytes: u32` (default `64 KiB`)
7. `entity_blob_hard_max_bytes: u32` (default `256 KiB`)

## Payload File Container (`.v4g`, `.v4p`, `.v4ix`)
Each payload file uses a shared wrapper:
1. `magic: [u8; 4]`
2. `format_version: u32` (currently `1`)
3. `payload_len: u32`
4. `payload_crc32: u32`
5. `payload_bytes: postcard-encoded struct`

Magic values:
- Global: `"V4G0"`
- Players: `"V4P0"`
- Index: `"V4IX"`

## Global Payload (`global.g*.v4g`)
`GlobalPayload` fields:
1. `base_world_kind: PersistedBaseWorldKind`
2. `world_seed: u64`
3. `next_entity_id: u64`
4. `next_data_file_id: u32`
5. `last_modified_ms: u64`
6. `player_entity_hints: Vec<PlayerEntityHint>`
7. `custom_global_payload: Vec<u8>`

`custom_global_payload` is intentionally opaque to `save_v3`.

## Players Payload (`players.g*.v4p`)
`PlayersPayload`:
1. `players: Vec<PlayerRecord>`

Players are persisted outside spatial blobs.

## Index Payload (`index/ix-main.g*.v4ix`)
`IndexPayload` fields:
1. `generation: u64`
2. `region_chunk_edge: i32`
3. `leaves: Vec<LeafEntry>`

`LeafEntry` fields:
1. `region: [i32; 4]`
2. `block_blob: Option<BlobRef>`
3. `entity_blob: Option<BlobRef>`

`BlobRef` fields:
1. `data_file_id: u32`
2. `record_offset: u64`
3. `record_len: u32`
4. `record_crc32: u32`
5. `blob_type: u8` (`1=block`, `2=entity`)
6. `blob_version: u16`

Important:
- This is a flat region map, not a tree of index nodes.
- One `LeafEntry` per region; each region carries at most one block blob ref and one entity blob ref in a generation snapshot.

## Data File Container (`data/dt-*.v4dt`)
File header:
1. `magic: "V4DT"`
2. `format_version: u32` (currently `1`)
3. `data_file_id: u32`

Each appended record:
1. `record_magic: "RECD"`
2. `blob_type: u8`
3. `blob_version: u16`
4. `payload_len: u32`
5. `payload_crc32: u32`
6. `payload_bytes`

## Blob Schemas

### Block Blob (`blob_type = 1`)
`BlockBlob` fields:
1. `region: [i32; 4]`
2. `region_chunk_edge: i32`
3. `chunk_dictionary: Vec<ChunkPayload>`
4. `assignments: Vec<ChunkAssignment>`

`ChunkPayload`:
1. `voxels: Vec<u8>` (must be exactly `CHUNK_VOLUME` entries)

`ChunkAssignment`:
1. `local_chunk_coord: [u16; 4]`
2. `state: ChunkFillState` (`Virgin`, `EmptyOverride`, `DictRef`)
3. `dict_index: u16`

Writer behavior:
- Assignments are sparse and only generated for chunks present in `VoxelWorld.chunks` within the region.
- `EmptyOverride` is emitted for explicit empty override chunks.
- `DictRef` references dictionary chunks for non-empty chunk payloads.
- Assignments are sorted by `local_chunk_coord`.
- The writer currently does not emit `Virgin`, but loader supports it.

Loader behavior:
- `Virgin`: remove chunk override.
- `EmptyOverride`: insert explicit empty chunk override.
- `DictRef`: materialize from dictionary entry.

### Entity Blob (`blob_type = 2`)
`EntityBlob` fields:
1. `region: [i32; 4]`
2. `region_chunk_edge: i32`
3. `entities: Vec<PersistedEntityRecord>`

`PersistedEntityRecord` stores full entity persistence payload (`entity_id`, class/kind, transform, velocity, material, tags, opaque payload, timestamp).

## Regioning Rules
- Chunk-to-region mapping uses `div_euclid(region_chunk_edge)` per axis.
- `region_chunk_edge <= 0` is normalized to default limits on save.

## Save Semantics (Current Implementation)
1. Load current generation from manifest.
2. Build mutable `leaf_by_region` map from existing index leaves.
3. Block persistence:
   - If `disable_block_persistence == false`, update block blobs from dirty/full region set.
   - If `disable_block_persistence == true`, clear all `block_blob` refs in leaves.
4. Entity persistence updates entity blobs by dirty/full entity region set.
5. Drop leaves where both blob refs are `None`.
6. Write next-generation index/global/players payload files.
7. Atomically swap manifest to next generation.

## Load Semantics
- `load_state` reconstructs a `VoxelWorld` from block blobs + base world kind.
- Entities are assembled from all entity blobs and deduplicated by `entity_id`.

## Legacy Migration
`migrate_legacy_world_to_v3` supports:
1. Input legacy `.v4dw` world.
2. Optional legacy entity sidecar JSON.
3. Output v3 save root.

## Known Divergence From v4 Target
- No fixed-16 spatial node tree in index payload.
- No `IndexNode` graph, root bounds, or subtree index files.
- Spatial indexing is region-leaf keyed only.

## Deployed Bridge Extension (Server Runtime)
- Server autosave may use `custom_global_payload` to carry opaque world override data and set `disable_block_persistence = true`.
- This behavior is implementation-defined by server code and not a structural change to the v3 file container itself.
- Compatibility tooling must preserve `custom_global_payload` bytes even when it does not interpret them.
