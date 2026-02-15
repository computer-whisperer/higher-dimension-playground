# Polychora Save Format v3 (Runtime Spec)

## Status
- This document defines the stripped-down v3 runtime target.
- Migration from legacy formats is handled by a dedicated CLI tool.
- Runtime only reads and writes v3 directory saves.

## v3 Ship Slice
1. Directory-based saves.
2. Fixed 16-way spatial split in 4D chunk space.
3. Single index file for v3 initial runtime.
4. Append-only bounded data files.
5. Block blobs use chunk dictionary plus sparse fill assignments only.
6. Players persist outside region/index data.
7. Crash safety via copy-on-write commit and manifest generation swap.
8. No WAL in v3 initial runtime.

## Deferred To v3.1+
1. Index subtree file splitting.
2. Dense/RLE fill structure mode for block blobs.
3. WAL-based transaction log.
4. Background compaction.

## Goals
1. Keep small worlds tiny.
2. Scale to large sparse worlds without monolithic rewrites.
3. Avoid file explosion from scattered edits.
4. Preserve player durability independently of regional entity churn.
5. Keep write operations resilient to crashes.

## Save Root Layout

```text
<world>/
  manifest.json
  global.v4g
  players.v4p
  index/
    ix-main.v4ix
  data/
    dt-000001.v4dt
    dt-000002.v4dt
```

## Key Runtime Invariants
1. `manifest.json` is authoritative for which generation is live.
2. Data files are append-only.
3. Index/global/players are rewritten copy-on-write into next generation files.
4. Players are never written into spatial region blobs.
5. Non-player entities are spatially indexed and persisted.

## Spatial Model
1. Space is chunk coordinates.
2. Tree fanout is fixed at 16 children per internal node.
3. Child selection bit is computed per axis (`x,y,z,w`) around node midpoint.
4. Leaf extents are bounded by implementation thresholds.

## Default Thresholds (Initial)
1. Data file max size: `128 MiB`.
2. Index soft max serialized size: `4 MiB`.
3. Block blob target payload size: `128 KiB`.
4. Block blob hard max payload size: `512 KiB`.
5. Entity blob target payload size: `64 KiB`.
6. Entity blob hard max payload size: `256 KiB`.
7. Entity movement hysteresis margin: `0.20` chunk widths.
8. Entity movement checkpoint interval: `10s`.

## File Specifications

### `manifest.json`
Required fields:
1. `format`: `"polychora-save"`
2. `version`: `3`
3. `current_generation`: `u64`
4. `created_ms`: `u64`
5. `last_modified_ms`: `u64`
6. `active_data_file_id`: `u32`
7. `data_file_count`: `u32`
8. `index_file`: relative path
9. `global_file`: relative path
10. `players_file`: relative path
11. `limits`: threshold values

Atomicity rule:
1. Write `manifest.json.tmp`.
2. `fsync`.
3. Rename to `manifest.json`.

### `global.v4g`
Binary file with fixed header + postcard payload.

Header:
1. magic: `"V4G0"` (4 bytes)
2. format_version: `u32` (`1` for first revision)
3. payload_len: `u32`
4. payload_crc32: `u32`

Payload fields:
1. `base_world_kind` and params
2. `world_seed`
3. `procgen_params`
4. `next_entity_id`
5. `next_data_file_id`
6. `last_modified_ms`
7. `player_entity_hints` (`player_id -> entity_id/last_region_hint`)
8. `custom_global_payload: Vec<u8>`

### `players.v4p`
Binary file with fixed header + postcard payload.

Header:
1. magic: `"V4P0"` (4 bytes)
2. format_version: `u32`
3. payload_len: `u32`
4. payload_crc32: `u32`

Payload:
1. `players: Vec<PlayerRecord>`
2. `PlayerRecord` contains stable player id, auth key/id, position/orientation, inventory payload, and metadata.

Durability intent:
1. Player file writes are copy-on-write per save transaction.
2. Player file is validated by checksum on load.

### `index/ix-main.v4ix`
Single index file in v3 initial runtime.

Header:
1. magic: `"V4IX"` (4 bytes)
2. format_version: `u32` (`1`)
3. generation: `u64`
4. root_min_chunk: `[i32;4]`
5. root_max_chunk: `[i32;4]`
6. node_count: `u32`
7. leaf_count: `u32`
8. payload_crc32: `u32`

Body:
1. `nodes: Vec<IndexNode>`
2. `leaves: Vec<LeafEntry>`

`IndexNode`:
1. `node_id: u32`
2. `parent_id: u32` (or sentinel)
3. `is_leaf: u8`
4. `depth: u8`
5. `min_chunk: [i32;4]`
6. `max_chunk: [i32;4]`
7. `child_ids: [u32;16]` (sentinel for none)
8. `leaf_entry_id: u32` (sentinel for non-leaf)

`LeafEntry`:
1. `leaf_id: u32`
2. `block_blob_refs: Vec<BlobRef>`
3. `entity_blob_refs: Vec<BlobRef>`

`BlobRef`:
1. `data_file_id: u32`
2. `record_offset: u64`
3. `record_len: u32`
4. `record_crc32: u32`
5. `blob_type: u8`
6. `blob_version: u16`

### `data/dt-*.v4dt`
Append-only packfile.

File header:
1. magic: `"V4DT"` (4 bytes)
2. format_version: `u32` (`1`)
3. data_file_id: `u32`

Each record:
1. record_magic: `"RECD"` (4 bytes)
2. blob_type: `u8` (`1=block`, `2=entity`)
3. blob_version: `u16`
4. payload_len: `u32`
5. payload_crc32: `u32`
6. payload bytes

## Blob Payload Schemas

### Block Blob (`blob_type=1`)
Payload fields:
1. `volume_min_chunk: [i32;4]`
2. `volume_max_chunk: [i32;4]`
3. `chunk_dictionary: Vec<ChunkPayload>`
4. `assignments: Vec<ChunkAssignment>`

`ChunkAssignment`:
1. `local_chunk_coord: [u16;4]`
2. `state: u8` (`0=Virgin`, `1=EmptyOverride`, `2=DictRef`)
3. `dict_index: u16` (used when state is `DictRef`)

Notes:
1. v3 initial runtime uses sparse assignments only.
2. Dense fill encoding is deferred.

### Entity Blob (`blob_type=2`)
Payload fields:
1. `volume_min_chunk: [i32;4]`
2. `volume_max_chunk: [i32;4]`
3. `entities: Vec<PersistedEntityRecord>`

`PersistedEntityRecord`:
1. `entity_id: u64`
2. `class: u8`
3. `kind: u16`
4. `position: [f32;4]`
5. `orientation: [f32;4]`
6. `velocity: [f32;4]`
7. `scale: f32`
8. `material: u8`
9. `display_name: Option<String>`
10. `tags: Vec<String>`
11. `payload: Vec<u8>`
12. `last_saved_ms: u64`

## Entity Persistence Policy

### Players
1. Players are persisted only in `players.v4p`.
2. Players are excluded from spatial entity blobs.
3. Reason: minimize risk of inventory/progression loss due to regional churn.

### Non-player entities
1. All non-player entities persist.
2. Stored in entity blobs and indexed spatially.
3. Flexible fields `tags` and `payload` are reserved for future componentization.

## Entity Movement Handling
1. Each entity has stable `entity_id`.
2. Runtime tracks entity-to-leaf membership.
3. Position updates always update in-memory state.
4. Persistence update happens when either condition is met:
1. entity crosses leaf boundary with hysteresis
2. checkpoint timer expires
5. On boundary crossing:
1. mark old leaf entity set dirty
2. mark new leaf entity set dirty
3. include entity in next entity-blob flush

## Save Transaction Protocol (No WAL)
1. Gather dirty regions and dirty global/player state.
2. Append new block/entity blob records to active data file(s).
3. `fsync` modified data file descriptors.
4. Build next-generation `ix-main.v4ix`, `global.v4g`, `players.v4p` in temp files.
5. `fsync` temp files.
6. Atomically rename temp files to generation-target names.
7. Write and atomically swap `manifest.json` with incremented `current_generation`.
8. `fsync` save root directory.

Guarantee:
1. On crash, runtime loads the last fully committed manifest generation.
2. Orphaned newer files not referenced by manifest are ignored.

## Startup Recovery
1. Read `manifest.json`.
2. Validate referenced `global/index/players` headers + checksums.
3. Open referenced data files and lazily validate records by checksum when loaded.
4. Ignore unreferenced generation files.
5. If manifest is missing/corrupt, fail fast with explicit error and require CLI repair.

## Tiny vs Large World Behavior
1. Tiny worlds:
1. few blobs
2. shallow tree
3. minimal index/data footprint
2. Large sparse worlds:
1. many blobs packed into bounded data files
2. index grows, but writes remain bounded to rewritten index + appended blobs
3. no one-file-per-edit behavior

## Implementation Plan (Order)
1. Add save root detection and strict v3 loader path.
2. Implement manifest/global/players serializers and validators.
3. Implement data file append writer + blob reader + checksums.
4. Implement single-file fixed16 index serializer/loader.
5. Implement block blob sparse assignment encoding.
6. Implement entity blob encoding with `tags` and opaque payload.
7. Implement dirty tracking for block leaves and entity leaves.
8. Implement copy-on-write commit path and startup recovery.
9. Add tests for crash-like interruption points and checksum corruption.

## Runtime Implementation Details

### Module split
1. `server/save_v3/mod.rs`: orchestration API and transaction boundary.
2. `server/save_v3/manifest.rs`: manifest schema, load/store, atomic swap.
3. `server/save_v3/global.rs`: `v4g` serializer/deserializer.
4. `server/save_v3/players.rs`: `v4p` serializer/deserializer.
5. `server/save_v3/index.rs`: `v4ix` tree build/load/query.
6. `server/save_v3/data.rs`: `v4dt` append/read and checksum validation.
7. `server/save_v3/blob_blocks.rs`: block blob build/parse.
8. `server/save_v3/blob_entities.rs`: entity blob build/parse.

### Core API sketch
1. `load_world_state(root: &Path) -> io::Result<LoadedStateV3>`
2. `begin_save_tx(state: &ServerState, dirty: &DirtySets) -> SaveTxPlan`
3. `append_blob_records(plan: &mut SaveTxPlan) -> io::Result<AppendedRefs>`
4. `build_next_generation_files(plan: &SaveTxPlan, refs: &AppendedRefs) -> io::Result<GenFiles>`
5. `commit_generation(root: &Path, gen: GenFiles) -> io::Result<()>`

### Dirty tracking rules
1. On voxel edit:
1. Convert voxel position to chunk position.
2. Resolve leaf id from chunk position.
3. Mark leaf in `dirty_block_leaves`.
2. On non-player entity spawn/despawn:
1. Resolve current leaf id.
2. Mark leaf in `dirty_entity_leaves`.
3. On non-player entity movement:
1. Update in-memory transform every tick.
2. Recompute leaf id when checkpoint timer expires or boundary-cross candidate detected.
3. Apply hysteresis; only cross when safely inside target leaf by margin.
4. If leaf changed, mark old and new leaf dirty.

### Block blob construction (deterministic)
1. Gather effective overridden chunks for one leaf volume.
2. Build dictionary by chunk payload hash + byte equality.
3. Emit assignments sorted lexicographically by local chunk coord (`w,z,y,x` or `x,y,z,w`, pick one and keep fixed).
4. Use only sparse assignments in v3.
5. If leaf has no non-virgin assignments after coalescing, remove its block blob refs.

### Entity blob construction (deterministic)
1. Collect all persistent non-player entities whose persisted leaf equals target leaf.
2. Sort by `entity_id`.
3. Serialize full record including `tags` and opaque `payload`.
4. If leaf has zero entities, remove entity blob refs.

### Data file rollover
1. Active data file comes from manifest.
2. Before append, estimate record bytes.
3. If append would exceed max file size, create next `dt-*.v4dt` and update active id in next generation manifest/global.

### Recovery behavior details
1. Manifest points to current generation file names.
2. Any generation files newer than manifest are ignored.
3. Missing referenced generation file is fatal load error.
4. Corrupt checksum in referenced `v4g/v4p/v4ix` is fatal load error.
5. Corrupt record checksum in data file is fatal when record is dereferenced.

## Testing Checklist
1. Empty world roundtrip.
2. Sparse edits across far-apart regions do not create many tiny files.
3. Entity movement across leaf boundary updates correct leaves.
4. Player inventory survives repeated save/load cycles.
5. Corrupted blob checksum is detected and surfaced.
6. Crash between data append and manifest swap preserves last committed state.
