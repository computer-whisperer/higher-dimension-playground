use crate::shared::protocol::{EntityClass, EntityKind};
use crate::shared::voxel::{
    load_world, BaseWorldKind, Chunk, ChunkPos, VoxelType, VoxelWorld, CHUNK_SIZE, CHUNK_VOLUME,
};
use crate::shared::worldfield::{
    Aabb4i, ChunkArrayData, ChunkArrayIndexCodec, ChunkPayload as FieldChunkPayload,
};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MANIFEST_FILE: &str = "manifest.json";
const INDEX_DIR: &str = "index";
const DATA_DIR: &str = "data";
const GLOBAL_MAGIC: &[u8; 4] = b"V4G0";
const PLAYERS_MAGIC: &[u8; 4] = b"V4P0";
const INDEX_MAGIC: &[u8; 4] = b"V4IX";
const DATA_MAGIC: &[u8; 4] = b"V4DT";
const RECORD_MAGIC: &[u8; 4] = b"RECD";
const INDEX_FILE_VERSION: u32 = 1;
const INDEX_ENTITY_ROOT_NONE: u32 = u32::MAX;

const BLOB_KIND_CHUNK_PAYLOAD: u8 = 1;
const BLOB_KIND_CHUNK_ARRAY: u8 = 2;
const BLOB_KIND_ENTITY: u8 = 3;

pub const SAVE_FORMAT_TAG: &str = "polychora-save";
pub const SAVE_FORMAT_VERSION: u32 = 4;
pub const PAYLOAD_FILE_VERSION: u32 = 1;
pub const DATA_FILE_VERSION: u32 = 1;
pub const CHUNK_PAYLOAD_BLOB_VERSION: u16 = 1;
pub const CHUNK_ARRAY_BLOB_VERSION: u16 = 1;
pub const ENTITY_BLOB_VERSION: u16 = 1;
pub const DEFAULT_REGION_CHUNK_EDGE: i32 = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveLimits {
    pub data_file_max_bytes: u64,
    pub index_soft_max_bytes: u64,
    pub chunk_payload_target_bytes: u32,
    pub chunk_payload_hard_max_bytes: u32,
    pub entity_blob_target_bytes: u32,
    pub entity_blob_hard_max_bytes: u32,
}

impl Default for SaveLimits {
    fn default() -> Self {
        Self {
            data_file_max_bytes: 128 * 1024 * 1024,
            index_soft_max_bytes: 8 * 1024 * 1024,
            chunk_payload_target_bytes: 128 * 1024,
            chunk_payload_hard_max_bytes: 512 * 1024,
            entity_blob_target_bytes: 64 * 1024,
            entity_blob_hard_max_bytes: 256 * 1024,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub current_generation: u64,
    pub created_ms: u64,
    pub last_modified_ms: u64,
    pub active_data_file_id: u32,
    pub data_file_count: u32,
    pub index_file: String,
    pub global_file: String,
    pub players_file: String,
    pub limits: SaveLimits,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PersistedBaseWorldKind {
    Empty,
    FlatFloor { material: u8 },
}

impl PersistedBaseWorldKind {
    pub fn from_runtime(base: BaseWorldKind) -> Self {
        match base {
            BaseWorldKind::Empty => Self::Empty,
            BaseWorldKind::FlatFloor { material } => Self::FlatFloor {
                material: material.0,
            },
        }
    }

    pub fn to_runtime(self) -> BaseWorldKind {
        match self {
            PersistedBaseWorldKind::Empty => BaseWorldKind::Empty,
            PersistedBaseWorldKind::FlatFloor { material } => BaseWorldKind::FlatFloor {
                material: VoxelType(material),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerEntityHint {
    pub player_id: u64,
    pub entity_id: Option<u64>,
    pub last_region_hint: Option<[i32; 4]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalPayload {
    pub base_world_kind: PersistedBaseWorldKind,
    pub world_seed: u64,
    pub procgen_manifest_hash: u64,
    pub next_entity_id: u64,
    pub next_data_file_id: u32,
    pub last_modified_ms: u64,
    pub player_entity_hints: Vec<PlayerEntityHint>,
    pub custom_global_payload: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PlayersPayload {
    pub players: Vec<PlayerRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerRecord {
    pub player_id: u64,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub tags: Vec<String>,
    pub inventory_payload: Vec<u8>,
    pub last_saved_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobRef {
    pub data_file_id: u32,
    pub record_offset: u64,
    pub record_len: u32,
    pub record_crc32: u32,
    pub blob_type: u8,
    pub blob_version: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum IndexNodeKind {
    Branch { child_node_ids: Vec<u32> },
    LeafEmpty,
    LeafUniform { material: u16 },
    LeafChunkArray { chunk_array_ref: BlobRef },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexNode {
    pub node_id: u32,
    pub bounds_min_chunk: [i32; 4],
    pub bounds_max_chunk: [i32; 4],
    pub kind: IndexNodeKind,
}

#[derive(Clone, Debug, Default)]
pub struct IndexPayload {
    pub generation: u64,
    pub root_node_id: u32,
    pub entity_root_node_id: Option<u32>,
    pub nodes: Vec<IndexNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexBody {
    nodes: Vec<IndexNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkPayloadBlob {
    pub payload: FieldChunkPayload,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkArrayBlob {
    pub volume_min_chunk: [i32; 4],
    pub volume_max_chunk: [i32; 4],
    pub payload_palette: Vec<BlobRef>,
    pub index_codec: ChunkArrayIndexCodec,
    pub index_data: Vec<u8>,
    pub default_palette_index: Option<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedEntityRecord {
    pub entity_id: u64,
    pub class: EntityClass,
    pub kind: EntityKind,
    pub position: [f32; 4],
    pub orientation: [f32; 4],
    pub velocity: [f32; 4],
    pub scale: f32,
    pub material: u8,
    pub display_name: Option<String>,
    pub tags: Vec<String>,
    pub payload: Vec<u8>,
    pub last_saved_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityBlob {
    pub volume_min_chunk: [i32; 4],
    pub volume_max_chunk: [i32; 4],
    pub entities: Vec<PersistedEntityRecord>,
}

#[derive(Debug)]
pub struct LoadedState {
    pub manifest: Manifest,
    pub global: GlobalPayload,
    pub players: PlayersPayload,
    pub index: IndexPayload,
    pub world: VoxelWorld,
    pub entities: Vec<PersistedEntityRecord>,
}

pub struct SaveRequest<'a> {
    pub world: &'a VoxelWorld,
    pub entities: &'a [PersistedEntityRecord],
    pub players: &'a [PlayerRecord],
    pub world_seed: u64,
    pub next_entity_id: u64,
    pub dirty_block_regions: &'a HashSet<[i32; 4]>,
    pub dirty_entity_regions: &'a HashSet<[i32; 4]>,
    pub force_full_blocks: bool,
    pub force_full_entities: bool,
    pub player_entity_hints: Option<Vec<PlayerEntityHint>>,
    pub custom_global_payload: Option<Vec<u8>>,
    pub disable_block_persistence: bool,
    pub now_ms: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SaveResult {
    pub generation: u64,
    pub saved_block_regions: usize,
    pub saved_entity_regions: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacySidecarBlob {
    version: u32,
    entities: Vec<LegacySidecarEntity>,
}

#[derive(Clone, Debug, Deserialize)]
struct LegacySidecarEntity {
    class: EntityClass,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    mob: Option<serde_json::Value>,
}

#[derive(Clone, Debug)]
struct LeafDescriptor {
    min: [i32; 4],
    max: [i32; 4],
    kind: IndexNodeKind,
}

#[derive(Clone, Debug)]
enum TempNodeKind {
    Leaf(IndexNodeKind),
    Branch(Vec<TempNode>),
}

#[derive(Clone, Debug)]
struct TempNode {
    min: [i32; 4],
    max: [i32; 4],
    kind: TempNodeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexSubtreeKind {
    World,
    Entity,
}

pub fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis().min(u64::MAX as u128) as u64
}

pub fn is_v4_save_root(path: &Path) -> bool {
    path.is_dir() && path.join(MANIFEST_FILE).is_file()
}

pub fn chunk_from_world_position(position: [f32; 4]) -> [i32; 4] {
    let cs = CHUNK_SIZE as i32;
    [
        (position[0].floor() as i32).div_euclid(cs),
        (position[1].floor() as i32).div_euclid(cs),
        (position[2].floor() as i32).div_euclid(cs),
        (position[3].floor() as i32).div_euclid(cs),
    ]
}

pub fn region_from_chunk(chunk: [i32; 4], region_chunk_edge: i32) -> [i32; 4] {
    let edge = region_chunk_edge.max(1);
    [
        chunk[0].div_euclid(edge),
        chunk[1].div_euclid(edge),
        chunk[2].div_euclid(edge),
        chunk[3].div_euclid(edge),
    ]
}

pub fn region_from_chunk_pos(chunk: ChunkPos, region_chunk_edge: i32) -> [i32; 4] {
    region_from_chunk([chunk.x, chunk.y, chunk.z, chunk.w], region_chunk_edge)
}

pub fn all_block_regions(world: &VoxelWorld, region_chunk_edge: i32) -> HashSet<[i32; 4]> {
    world
        .chunks
        .keys()
        .copied()
        .map(|pos| region_from_chunk_pos(pos, region_chunk_edge))
        .collect()
}

pub fn all_entity_regions(
    entities: &[PersistedEntityRecord],
    region_chunk_edge: i32,
) -> HashSet<[i32; 4]> {
    entities
        .iter()
        .map(|entity| {
            region_from_chunk(
                chunk_from_world_position(entity.position),
                region_chunk_edge,
            )
        })
        .collect()
}

pub fn load_or_init_state(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedState> {
    if !is_v4_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    load_state(root)
}

pub fn load_state(root: &Path) -> io::Result<LoadedState> {
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let players: PlayersPayload =
        read_payload_file(root.join(&manifest.players_file), PLAYERS_MAGIC)?;
    let index = read_index_file(root.join(&manifest.index_file))?;

    let mut world = VoxelWorld::new_with_base(global.base_world_kind.to_runtime());
    materialize_world_from_index(root, &index, &mut world)?;

    let mut entities = materialize_entities_from_index(root, &index)?;
    entities.sort_unstable_by_key(|entity| entity.entity_id);

    world.clear_dirty();
    let _ = world.drain_pending_chunk_updates();

    Ok(LoadedState {
        manifest,
        global,
        players,
        index,
        world,
        entities,
    })
}

pub fn save_state(root: &Path, request: SaveRequest<'_>) -> io::Result<SaveResult> {
    let _ = request.dirty_block_regions;
    let _ = request.dirty_entity_regions;
    let _ = request.force_full_blocks;
    let _ = request.force_full_entities;

    let loaded = load_or_init_state(
        root,
        request.world.base_kind(),
        request.world_seed,
        request.now_ms,
    )?;
    let mut manifest = loaded.manifest;

    let mut world_leaves = Vec::<LeafDescriptor>::new();
    if !request.disable_block_persistence {
        let mut sorted_chunks: Vec<(ChunkPos, Chunk)> = request
            .world
            .chunks
            .iter()
            .map(|(&pos, chunk)| (pos, chunk.clone()))
            .collect();
        sorted_chunks.sort_unstable_by_key(|(pos, _)| [pos.x, pos.y, pos.z, pos.w]);

        for (chunk_pos, chunk) in sorted_chunks {
            let min = [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w];
            let max = min;
            let payload = FieldChunkPayload::from_chunk_compact(&chunk);
            match payload {
                FieldChunkPayload::Empty => {
                    world_leaves.push(LeafDescriptor {
                        min,
                        max,
                        kind: IndexNodeKind::LeafEmpty,
                    });
                }
                FieldChunkPayload::Uniform(material) => {
                    world_leaves.push(LeafDescriptor {
                        min,
                        max,
                        kind: IndexNodeKind::LeafUniform { material },
                    });
                }
                other => {
                    let payload_blob = ChunkPayloadBlob {
                        payload: other.clone(),
                    };
                    let payload_blob_bytes = postcard::to_stdvec(&payload_blob)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                    let payload_ref = append_blob_record(
                        root,
                        &mut manifest,
                        BLOB_KIND_CHUNK_PAYLOAD,
                        CHUNK_PAYLOAD_BLOB_VERSION,
                        &payload_blob_bytes,
                    )?;

                    let chunk_array_blob = ChunkArrayBlob {
                        volume_min_chunk: min,
                        volume_max_chunk: max,
                        payload_palette: vec![payload_ref],
                        index_codec: ChunkArrayIndexCodec::DenseU16,
                        index_data: vec![0, 0],
                        default_palette_index: None,
                    };
                    let chunk_array_blob_bytes = postcard::to_stdvec(&chunk_array_blob)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                    let chunk_array_ref = append_blob_record(
                        root,
                        &mut manifest,
                        BLOB_KIND_CHUNK_ARRAY,
                        CHUNK_ARRAY_BLOB_VERSION,
                        &chunk_array_blob_bytes,
                    )?;

                    world_leaves.push(LeafDescriptor {
                        min,
                        max,
                        kind: IndexNodeKind::LeafChunkArray { chunk_array_ref },
                    });
                }
            }
        }
    }

    let mut entities_by_chunk = HashMap::<[i32; 4], Vec<PersistedEntityRecord>>::new();
    for entity in request.entities {
        let chunk = chunk_from_world_position(entity.position);
        entities_by_chunk
            .entry(chunk)
            .or_default()
            .push(entity.clone());
    }
    for entities in entities_by_chunk.values_mut() {
        entities.sort_unstable_by_key(|entity| entity.entity_id);
    }

    let mut entity_chunks: Vec<[i32; 4]> = entities_by_chunk.keys().copied().collect();
    entity_chunks.sort_unstable();

    let mut entity_leaves = Vec::<LeafDescriptor>::new();
    for chunk in entity_chunks {
        let entities = entities_by_chunk.remove(&chunk).unwrap_or_default();
        if entities.is_empty() {
            continue;
        }

        let entity_blob = EntityBlob {
            volume_min_chunk: chunk,
            volume_max_chunk: chunk,
            entities,
        };
        let entity_blob_bytes = postcard::to_stdvec(&entity_blob)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let entity_ref = append_blob_record(
            root,
            &mut manifest,
            BLOB_KIND_ENTITY,
            ENTITY_BLOB_VERSION,
            &entity_blob_bytes,
        )?;

        entity_leaves.push(LeafDescriptor {
            min: chunk,
            max: chunk,
            kind: IndexNodeKind::LeafChunkArray {
                chunk_array_ref: entity_ref,
            },
        });
    }

    let mut world_temp = build_temp_tree_from_leaves(&world_leaves)
        .unwrap_or_else(|| make_empty_branch_root([0, 0, 0, 0], [0, 0, 0, 0]));
    canonicalize_temp_tree(&mut world_temp);

    let mut entity_temp = build_temp_tree_from_leaves(&entity_leaves);
    if let Some(node) = entity_temp.as_mut() {
        canonicalize_temp_tree(node);
    }

    let mut nodes = Vec::<IndexNode>::new();
    let root_node_id = flatten_temp_tree(&world_temp, &mut nodes);
    let entity_root_node_id = entity_temp
        .as_ref()
        .map(|node| flatten_temp_tree(node, &mut nodes));

    let next_generation = manifest.current_generation.saturating_add(1);
    let next_index_file = index_generation_path(next_generation);
    let next_global_file = global_generation_path(next_generation);
    let next_players_file = players_generation_path(next_generation);

    let next_index = IndexPayload {
        generation: next_generation,
        root_node_id,
        entity_root_node_id,
        nodes,
    };
    validate_index_payload(&next_index)?;
    write_index_file(root.join(&next_index_file), &next_index)?;

    let next_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(request.world.base_kind()),
        world_seed: request.world_seed,
        procgen_manifest_hash: loaded.global.procgen_manifest_hash,
        next_entity_id: request.next_entity_id,
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: request.now_ms,
        player_entity_hints: request
            .player_entity_hints
            .clone()
            .unwrap_or(loaded.global.player_entity_hints),
        custom_global_payload: request
            .custom_global_payload
            .clone()
            .unwrap_or(loaded.global.custom_global_payload),
    };
    write_payload_file(
        root.join(&next_global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_global,
    )?;

    let next_players = PlayersPayload {
        players: request.players.to_vec(),
    };
    write_payload_file(
        root.join(&next_players_file),
        PLAYERS_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_players,
    )?;

    manifest.current_generation = next_generation;
    manifest.last_modified_ms = request.now_ms;
    manifest.index_file = next_index_file;
    manifest.global_file = next_global_file;
    manifest.players_file = next_players_file;
    save_manifest_atomic(root, &manifest)?;

    Ok(SaveResult {
        generation: next_generation,
        saved_block_regions: world_leaves.len(),
        saved_entity_regions: entity_leaves.len(),
    })
}

pub fn load_legacy_sidecar_entities(
    path: &Path,
    now_ms: u64,
) -> io::Result<Vec<PersistedEntityRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed: LegacySidecarBlob = serde_json::from_reader(reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if parsed.version == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid legacy sidecar version 0",
        ));
    }

    let mut next_entity_id = 1u64;
    let mut out = Vec::with_capacity(parsed.entities.len());
    for entity in parsed.entities {
        let payload = if let Some(mob) = entity.mob {
            serde_json::to_vec(&mob)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        } else {
            Vec::new()
        };
        out.push(PersistedEntityRecord {
            entity_id: next_entity_id,
            class: entity.class,
            kind: entity.kind,
            position: entity.position,
            orientation: entity.orientation,
            velocity: [0.0, 0.0, 0.0, 0.0],
            scale: entity.scale,
            material: entity.material,
            display_name: entity.display_name,
            tags: Vec::new(),
            payload,
            last_saved_ms: now_ms,
        });
        next_entity_id = next_entity_id.saturating_add(1);
    }
    Ok(out)
}

pub fn migrate_legacy_world_to_v4(
    legacy_world: &Path,
    legacy_sidecar: Option<&Path>,
    output_root: &Path,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<SaveResult> {
    let file = File::open(legacy_world)?;
    let mut reader = BufReader::new(file);
    let world = load_world(&mut reader)?;

    let entities = if let Some(sidecar_path) = legacy_sidecar {
        load_legacy_sidecar_entities(sidecar_path, now_ms)?
    } else {
        Vec::new()
    };

    let block_regions = all_block_regions(&world, DEFAULT_REGION_CHUNK_EDGE);
    let entity_regions = all_entity_regions(&entities, DEFAULT_REGION_CHUNK_EDGE);
    let next_entity_id = entities
        .iter()
        .map(|entity| entity.entity_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let empty_players: Vec<PlayerRecord> = Vec::new();

    save_state(
        output_root,
        SaveRequest {
            world: &world,
            entities: &entities,
            players: &empty_players,
            world_seed,
            next_entity_id,
            dirty_block_regions: &block_regions,
            dirty_entity_regions: &entity_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
}

pub fn migrate_v3_save_to_v4(
    v3_root: &Path,
    output_root: &Path,
    overwrite: bool,
    now_ms: u64,
) -> io::Result<SaveResult> {
    if output_root.exists() {
        if !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "output path '{}' already exists (use overwrite to replace)",
                    output_root.display()
                ),
            ));
        }
        if output_root.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output path '{}' is a file", output_root.display()),
            ));
        }
        std::fs::remove_dir_all(output_root)?;
    }

    let loaded = crate::save_v3::load_state(v3_root)?;
    let crate::save_v3::LoadedState {
        global,
        players,
        world,
        entities,
        manifest,
        ..
    } = loaded;

    let entities: Vec<PersistedEntityRecord> = entities
        .into_iter()
        .map(|entity| PersistedEntityRecord {
            entity_id: entity.entity_id,
            class: entity.class,
            kind: entity.kind,
            position: entity.position,
            orientation: entity.orientation,
            velocity: entity.velocity,
            scale: entity.scale,
            material: entity.material,
            display_name: entity.display_name,
            tags: entity.tags,
            payload: entity.payload,
            last_saved_ms: entity.last_saved_ms,
        })
        .collect();
    let players: Vec<PlayerRecord> = players
        .players
        .into_iter()
        .map(|player| PlayerRecord {
            player_id: player.player_id,
            position: player.position,
            orientation: player.orientation,
            tags: player.tags,
            inventory_payload: player.inventory_payload,
            last_saved_ms: player.last_saved_ms,
        })
        .collect();
    let player_entity_hints: Vec<PlayerEntityHint> = global
        .player_entity_hints
        .into_iter()
        .map(|hint| PlayerEntityHint {
            player_id: hint.player_id,
            entity_id: hint.entity_id,
            last_region_hint: hint.last_region_hint,
        })
        .collect();

    let region_chunk_edge = manifest.limits.region_chunk_edge.max(1);
    let block_regions = all_block_regions(&world, region_chunk_edge);
    let entity_regions = all_entity_regions(&entities, region_chunk_edge);

    save_state(
        output_root,
        SaveRequest {
            world: &world,
            entities: &entities,
            players: &players,
            world_seed: global.world_seed,
            next_entity_id: global.next_entity_id,
            dirty_block_regions: &block_regions,
            dirty_entity_regions: &entity_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: Some(player_entity_hints),
            custom_global_payload: Some(global.custom_global_payload),
            disable_block_persistence: false,
            now_ms,
        },
    )
}

fn init_empty_save_root(
    root: &Path,
    base_world_kind: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<()> {
    std::fs::create_dir_all(root.join(INDEX_DIR))?;
    std::fs::create_dir_all(root.join(DATA_DIR))?;

    let limits = SaveLimits::default();
    let generation = 0u64;
    let index_file = index_generation_path(generation);
    let global_file = global_generation_path(generation);
    let players_file = players_generation_path(generation);

    let initial_index = IndexPayload {
        generation,
        root_node_id: 0,
        entity_root_node_id: None,
        nodes: vec![IndexNode {
            node_id: 0,
            bounds_min_chunk: [0, 0, 0, 0],
            bounds_max_chunk: [0, 0, 0, 0],
            kind: IndexNodeKind::Branch {
                child_node_ids: Vec::new(),
            },
        }],
    };
    write_index_file(root.join(&index_file), &initial_index)?;

    let initial_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(base_world_kind),
        world_seed,
        procgen_manifest_hash: 0,
        next_entity_id: 1,
        next_data_file_id: 2,
        last_modified_ms: now_ms,
        player_entity_hints: Vec::new(),
        custom_global_payload: Vec::new(),
    };
    write_payload_file(
        root.join(&global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &initial_global,
    )?;

    let initial_players = PlayersPayload::default();
    write_payload_file(
        root.join(&players_file),
        PLAYERS_MAGIC,
        PAYLOAD_FILE_VERSION,
        &initial_players,
    )?;

    ensure_data_file(root, 1)?;
    let manifest = Manifest {
        format: SAVE_FORMAT_TAG.to_string(),
        version: SAVE_FORMAT_VERSION,
        current_generation: generation,
        created_ms: now_ms,
        last_modified_ms: now_ms,
        active_data_file_id: 1,
        data_file_count: 1,
        index_file,
        global_file,
        players_file,
        limits,
    };
    save_manifest_atomic(root, &manifest)?;
    Ok(())
}

fn materialize_world_from_index(
    root: &Path,
    index: &IndexPayload,
    world: &mut VoxelWorld,
) -> io::Result<()> {
    let node_by_id = build_node_lookup(index);
    apply_world_node(root, &node_by_id, index.root_node_id, world)
}

fn materialize_entities_from_index(
    root: &Path,
    index: &IndexPayload,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let Some(entity_root) = index.entity_root_node_id else {
        return Ok(Vec::new());
    };

    let node_by_id = build_node_lookup(index);
    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();
    apply_entity_node(root, &node_by_id, entity_root, &mut entities_by_id)?;
    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);
    Ok(entities)
}

fn build_node_lookup(index: &IndexPayload) -> HashMap<u32, &IndexNode> {
    index
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect()
}

fn apply_world_node(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    world: &mut VoxelWorld,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                apply_world_node(root, node_by_id, *child_id, world)?;
            }
        }
        IndexNodeKind::LeafEmpty => {
            for_each_chunk_in_bounds(node.bounds_min_chunk, node.bounds_max_chunk, |chunk_pos| {
                world.insert_chunk(chunk_pos, Chunk::new());
            });
        }
        IndexNodeKind::LeafUniform { material } => {
            let chunk = build_uniform_chunk(*material);
            for_each_chunk_in_bounds(node.bounds_min_chunk, node.bounds_max_chunk, |chunk_pos| {
                world.insert_chunk(chunk_pos, chunk.clone());
            });
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            apply_chunk_array_ref_to_world(root, chunk_array_ref, world)?;
        }
    }
    Ok(())
}

fn apply_entity_node(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    entities_by_id: &mut HashMap<u64, PersistedEntityRecord>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing entity node id {node_id}"),
        ));
    };

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                apply_entity_node(root, node_by_id, *child_id, entities_by_id)?;
            }
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_ENTITY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "entity subtree leaf references non-entity blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob: EntityBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            for entity in blob.entities {
                entities_by_id.insert(entity.entity_id, entity);
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {}
    }

    Ok(())
}

fn apply_chunk_array_ref_to_world(
    root: &Path,
    chunk_array_ref: &BlobRef,
    world: &mut VoxelWorld,
) -> io::Result<()> {
    let payload = read_blob_payload(root, chunk_array_ref)?;
    match chunk_array_ref.blob_type {
        BLOB_KIND_CHUNK_ARRAY => {
            let blob: ChunkArrayBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            apply_chunk_array_blob_to_world(root, &blob, world)
        }
        BLOB_KIND_CHUNK_PAYLOAD => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "world index leaf must reference chunk-array blob, not chunk-payload blob",
        )),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported block blob type {other}"),
        )),
    }
}

fn apply_chunk_array_blob_to_world(
    root: &Path,
    blob: &ChunkArrayBlob,
    world: &mut VoxelWorld,
) -> io::Result<()> {
    let bounds = Aabb4i::new(blob.volume_min_chunk, blob.volume_max_chunk);
    if !bounds.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid chunk array bounds",
        ));
    }

    let palette_len = blob.payload_palette.len().max(1);
    let dummy_palette = vec![FieldChunkPayload::Empty; palette_len];
    let chunk_array_data = ChunkArrayData {
        bounds,
        chunk_palette: dummy_palette,
        index_codec: blob.index_codec,
        index_data: blob.index_data.clone(),
        default_chunk_idx: blob.default_palette_index,
    };
    let indices = chunk_array_data
        .decode_dense_indices()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;

    let mut palette_chunks = Vec::<Chunk>::new();
    for payload_ref in &blob.payload_palette {
        if payload_ref.blob_type != BLOB_KIND_CHUNK_PAYLOAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "chunk array palette contains non-chunk-payload blob type {}",
                    payload_ref.blob_type
                ),
            ));
        }
        let payload = read_blob_payload(root, payload_ref)?;
        let payload_blob: ChunkPayloadBlob = postcard::from_bytes(&payload)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        palette_chunks.push(payload_blob_to_chunk(&payload_blob)?);
    }

    let extents = bounds.chunk_extents().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid chunk array extents during decode",
        )
    })?;

    for w in bounds.min[3]..=bounds.max[3] {
        for z in bounds.min[2]..=bounds.max[2] {
            for y in bounds.min[1]..=bounds.max[1] {
                for x in bounds.min[0]..=bounds.max[0] {
                    let local = [
                        (x - bounds.min[0]) as usize,
                        (y - bounds.min[1]) as usize,
                        (z - bounds.min[2]) as usize,
                        (w - bounds.min[3]) as usize,
                    ];
                    let linear = linear_cell_index(local, extents);
                    let Some(&palette_idx) = indices.get(linear) else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "chunk array decoded index out of bounds",
                        ));
                    };
                    let Some(chunk) = palette_chunks.get(palette_idx as usize) else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "chunk array palette index out of bounds",
                        ));
                    };
                    world.insert_chunk(ChunkPos::new(x, y, z, w), chunk.clone());
                }
            }
        }
    }

    Ok(())
}

fn payload_blob_to_chunk(blob: &ChunkPayloadBlob) -> io::Result<Chunk> {
    blob.payload
        .to_voxel_chunk()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))
}

fn build_uniform_chunk(material: u16) -> Chunk {
    let voxel = VoxelType(u8::try_from(material).unwrap_or(u8::MAX));
    let mut chunk = Chunk::new();
    if voxel.is_air() {
        chunk.dirty = false;
        return chunk;
    }
    for entry in chunk.voxels.iter_mut() {
        *entry = voxel;
    }
    chunk.solid_count = CHUNK_VOLUME as u32;
    chunk.dirty = true;
    chunk
}

fn for_each_chunk_in_bounds<F>(min: [i32; 4], max: [i32; 4], mut f: F)
where
    F: FnMut(ChunkPos),
{
    for w in min[3]..=max[3] {
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    f(ChunkPos::new(x, y, z, w));
                }
            }
        }
    }
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn bounds_key(min: [i32; 4], max: [i32; 4]) -> [i32; 8] {
    [
        min[0], min[1], min[2], min[3], max[0], max[1], max[2], max[3],
    ]
}

fn make_empty_branch_root(min: [i32; 4], max: [i32; 4]) -> TempNode {
    TempNode {
        min,
        max,
        kind: TempNodeKind::Branch(Vec::new()),
    }
}

fn bounds_are_valid(min: [i32; 4], max: [i32; 4]) -> bool {
    (0..4).all(|axis| min[axis] <= max[axis])
}

fn bounds_contains(
    parent_min: [i32; 4],
    parent_max: [i32; 4],
    child_min: [i32; 4],
    child_max: [i32; 4],
) -> bool {
    (0..4).all(|axis| parent_min[axis] <= child_min[axis] && child_max[axis] <= parent_max[axis])
}

fn bounds_intersect(a_min: [i32; 4], a_max: [i32; 4], b_min: [i32; 4], b_max: [i32; 4]) -> bool {
    (0..4).all(|axis| a_min[axis] <= b_max[axis] && a_max[axis] >= b_min[axis])
}

fn build_temp_tree_from_leaves(leaves: &[LeafDescriptor]) -> Option<TempNode> {
    if leaves.is_empty() {
        return None;
    }

    if leaves.len() == 1 {
        let leaf = &leaves[0];
        return Some(TempNode {
            min: leaf.min,
            max: leaf.max,
            kind: TempNodeKind::Leaf(leaf.kind.clone()),
        });
    }

    let mut min = [i32::MAX; 4];
    let mut max = [i32::MIN; 4];
    for leaf in leaves {
        for axis in 0..4 {
            min[axis] = min[axis].min(leaf.min[axis]);
            max[axis] = max[axis].max(leaf.max[axis]);
        }
    }

    let mut split_axis = 0usize;
    let mut split_span = i64::MIN;
    for axis in 0..4 {
        let span = i64::from(max[axis]) - i64::from(min[axis]);
        if span > split_span {
            split_span = span;
            split_axis = axis;
        }
    }

    let mut sorted = leaves.to_vec();
    sorted.sort_unstable_by_key(|leaf| {
        (
            i64::from(leaf.min[split_axis]) + i64::from(leaf.max[split_axis]),
            bounds_key(leaf.min, leaf.max),
        )
    });

    let mid = (sorted.len() / 2).max(1).min(sorted.len() - 1);
    let left = build_temp_tree_from_leaves(&sorted[..mid]).expect("left subtree");
    let right = build_temp_tree_from_leaves(&sorted[mid..]).expect("right subtree");
    let mut children = vec![left, right];
    children.sort_unstable_by_key(|child| bounds_key(child.min, child.max));

    Some(TempNode {
        min,
        max,
        kind: TempNodeKind::Branch(children),
    })
}

fn canonicalize_temp_tree(node: &mut TempNode) {
    let TempNodeKind::Branch(children) = &mut node.kind else {
        return;
    };
    for child in children.iter_mut() {
        canonicalize_temp_tree(child);
    }
    children.sort_unstable_by_key(temp_node_order_key);
}

fn temp_node_order_key(node: &TempNode) -> ([i32; 8], u8, u64, u64, u64, u64, u64) {
    let (kind_rank, v0, v1, v2, v3, v4) = temp_kind_order_key(&node.kind);
    (
        bounds_key(node.min, node.max),
        kind_rank,
        v0,
        v1,
        v2,
        v3,
        v4,
    )
}

fn temp_kind_order_key(kind: &TempNodeKind) -> (u8, u64, u64, u64, u64, u64) {
    match kind {
        TempNodeKind::Branch(_) => (0, 0, 0, 0, 0, 0),
        TempNodeKind::Leaf(index_kind) => index_kind_order_key(index_kind),
    }
}

fn index_kind_order_key(kind: &IndexNodeKind) -> (u8, u64, u64, u64, u64, u64) {
    match kind {
        IndexNodeKind::Branch { child_node_ids } => (0, child_node_ids.len() as u64, 0, 0, 0, 0),
        IndexNodeKind::LeafEmpty => (1, 0, 0, 0, 0, 0),
        IndexNodeKind::LeafUniform { material } => (2, u64::from(*material), 0, 0, 0, 0),
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => (
            3,
            u64::from(chunk_array_ref.data_file_id),
            chunk_array_ref.record_offset,
            u64::from(chunk_array_ref.record_len),
            u64::from(chunk_array_ref.record_crc32),
            u64::from(chunk_array_ref.blob_type) | (u64::from(chunk_array_ref.blob_version) << 32),
        ),
    }
}

fn flatten_temp_tree(node: &TempNode, out: &mut Vec<IndexNode>) -> u32 {
    let kind = match &node.kind {
        TempNodeKind::Leaf(kind) => kind.clone(),
        TempNodeKind::Branch(children) => {
            let mut child_ids = Vec::with_capacity(children.len());
            for child in children {
                child_ids.push(flatten_temp_tree(child, out));
            }
            IndexNodeKind::Branch {
                child_node_ids: child_ids,
            }
        }
    };

    let node_id = out.len() as u32;
    out.push(IndexNode {
        node_id,
        bounds_min_chunk: node.min,
        bounds_max_chunk: node.max,
        kind,
    });
    node_id
}

fn index_generation_path(generation: u64) -> String {
    format!("{INDEX_DIR}/ix-main.g{generation:016}.v4ix")
}

fn global_generation_path(generation: u64) -> String {
    format!("global.g{generation:016}.v4g")
}

fn players_generation_path(generation: u64) -> String {
    format!("players.g{generation:016}.v4p")
}

fn data_file_path(root: &Path, data_file_id: u32) -> PathBuf {
    root.join(DATA_DIR)
        .join(format!("dt-{data_file_id:06}.v4dt"))
}

fn ensure_data_file(root: &Path, data_file_id: u32) -> io::Result<()> {
    std::fs::create_dir_all(root.join(DATA_DIR))?;
    let path = data_file_path(root, data_file_id);
    if path.exists() {
        return Ok(());
    }
    let mut file = BufWriter::new(File::create(&path)?);
    file.write_all(DATA_MAGIC)?;
    file.write_all(&DATA_FILE_VERSION.to_le_bytes())?;
    file.write_all(&data_file_id.to_le_bytes())?;
    file.flush()?;
    let file = file
        .into_inner()
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
    file.sync_all()?;
    Ok(())
}

fn append_blob_record(
    root: &Path,
    manifest: &mut Manifest,
    blob_type: u8,
    blob_version: u16,
    payload: &[u8],
) -> io::Result<BlobRef> {
    let record_header_len = 4 + 1 + 2 + 4 + 4;
    let record_len = (record_header_len + payload.len()) as u64;
    let payload_crc = crc32(payload);

    loop {
        ensure_data_file(root, manifest.active_data_file_id)?;
        let path = data_file_path(root, manifest.active_data_file_id);
        let current_len = std::fs::metadata(&path)?.len();
        let can_roll = current_len > (4 + 4 + 4) as u64;
        if current_len.saturating_add(record_len) > manifest.limits.data_file_max_bytes && can_roll
        {
            manifest.active_data_file_id = manifest.active_data_file_id.saturating_add(1);
            manifest.data_file_count = manifest.data_file_count.max(manifest.active_data_file_id);
            continue;
        }

        let mut file = OpenOptions::new().append(true).read(true).open(&path)?;
        let offset = file.seek(SeekFrom::End(0))?;
        file.write_all(RECORD_MAGIC)?;
        file.write_all(&[blob_type])?;
        file.write_all(&blob_version.to_le_bytes())?;
        file.write_all(&(payload.len() as u32).to_le_bytes())?;
        file.write_all(&payload_crc.to_le_bytes())?;
        file.write_all(payload)?;
        file.flush()?;
        file.sync_data()?;
        return Ok(BlobRef {
            data_file_id: manifest.active_data_file_id,
            record_offset: offset,
            record_len: record_len as u32,
            record_crc32: payload_crc,
            blob_type,
            blob_version,
        });
    }
}

fn read_blob_payload(root: &Path, blob: &BlobRef) -> io::Result<Vec<u8>> {
    let path = data_file_path(root, blob.data_file_id);
    let mut file = BufReader::new(File::open(path)?);

    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[0..4] != DATA_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid data file magic",
        ));
    }

    file.seek(SeekFrom::Start(blob.record_offset))?;
    let mut record_header = [0u8; 15];
    file.read_exact(&mut record_header)?;
    if &record_header[0..4] != RECORD_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid record magic",
        ));
    }

    let blob_type = record_header[4];
    let blob_version = u16::from_le_bytes([record_header[5], record_header[6]]);
    let payload_len = u32::from_le_bytes([
        record_header[7],
        record_header[8],
        record_header[9],
        record_header[10],
    ]) as usize;
    let payload_crc = u32::from_le_bytes([
        record_header[11],
        record_header[12],
        record_header[13],
        record_header[14],
    ]);
    if blob_type != blob.blob_type || blob_version != blob.blob_version {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "blob type/version mismatch",
        ));
    }

    let mut payload = vec![0u8; payload_len];
    file.read_exact(&mut payload)?;
    if crc32(&payload) != payload_crc || payload_crc != blob.record_crc32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "blob payload checksum mismatch",
        ));
    }
    Ok(payload)
}

fn load_manifest(root: &Path) -> io::Result<Manifest> {
    let path = root.join(MANIFEST_FILE);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let manifest: Manifest = serde_json::from_reader(reader)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if manifest.format != SAVE_FORMAT_TAG || manifest.version != SAVE_FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported v4 save manifest format/version",
        ));
    }
    Ok(manifest)
}

fn save_manifest_atomic(root: &Path, manifest: &Manifest) -> io::Result<()> {
    std::fs::create_dir_all(root)?;
    let tmp_path = root.join(format!("{MANIFEST_FILE}.tmp"));
    let final_path = root.join(MANIFEST_FILE);
    {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, manifest)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        writer.flush()?;
        let file = writer
            .into_inner()
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp_path, &final_path)?;
    fsync_directory(root);
    Ok(())
}

fn write_index_file(path: PathBuf, index: &IndexPayload) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let body = IndexBody {
        nodes: index.nodes.clone(),
    };
    let payload = postcard::to_stdvec(&body)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let checksum = crc32(&payload);

    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(INDEX_MAGIC)?;
    writer.write_all(&INDEX_FILE_VERSION.to_le_bytes())?;
    writer.write_all(&index.generation.to_le_bytes())?;
    writer.write_all(&index.root_node_id.to_le_bytes())?;
    writer.write_all(&(index.nodes.len() as u32).to_le_bytes())?;
    writer.write_all(
        &index
            .entity_root_node_id
            .unwrap_or(INDEX_ENTITY_ROOT_NONE)
            .to_le_bytes(),
    )?;
    writer.write_all(&checksum.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
    file.sync_all()?;
    Ok(())
}

fn read_index_file(path: PathBuf) -> io::Result<IndexPayload> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut header = [0u8; 32];
    reader.read_exact(&mut header)?;
    if &header[0..4] != INDEX_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "index file magic mismatch",
        ));
    }

    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != INDEX_FILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported index file version {version}"),
        ));
    }

    let generation = u64::from_le_bytes([
        header[8], header[9], header[10], header[11], header[12], header[13], header[14],
        header[15],
    ]);
    let root_node_id = u32::from_le_bytes([header[16], header[17], header[18], header[19]]);
    let node_count = u32::from_le_bytes([header[20], header[21], header[22], header[23]]);
    let entity_root_raw = u32::from_le_bytes([header[24], header[25], header[26], header[27]]);
    let checksum = u32::from_le_bytes([header[28], header[29], header[30], header[31]]);

    let mut payload = Vec::new();
    reader.read_to_end(&mut payload)?;
    if crc32(&payload) != checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "index file payload checksum mismatch",
        ));
    }

    let body: IndexBody = postcard::from_bytes(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if body.nodes.len() != node_count as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "index node count mismatch: header={}, payload={}",
                node_count,
                body.nodes.len()
            ),
        ));
    }

    let entity_root_node_id =
        (entity_root_raw != INDEX_ENTITY_ROOT_NONE).then_some(entity_root_raw);

    let index = IndexPayload {
        generation,
        root_node_id,
        entity_root_node_id,
        nodes: body.nodes,
    };
    validate_index_payload(&index)?;
    Ok(index)
}

fn validate_index_payload(index: &IndexPayload) -> io::Result<()> {
    if index.nodes.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "index must contain at least one node",
        ));
    }

    let mut node_by_id = HashMap::<u32, &IndexNode>::new();
    for node in &index.nodes {
        if !bounds_are_valid(node.bounds_min_chunk, node.bounds_max_chunk) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "node {} has invalid bounds min={:?} max={:?}",
                    node.node_id, node.bounds_min_chunk, node.bounds_max_chunk
                ),
            ));
        }
        if node_by_id.insert(node.node_id, node).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate node id {}", node.node_id),
            ));
        }
    }

    if !node_by_id.contains_key(&index.root_node_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing root node id {}", index.root_node_id),
        ));
    }

    let mut seen = HashSet::<u32>::new();
    let mut stack = HashSet::<u32>::new();
    validate_index_subtree(
        index.root_node_id,
        None,
        IndexSubtreeKind::World,
        &node_by_id,
        &mut seen,
        &mut stack,
    )?;

    if let Some(entity_root) = index.entity_root_node_id {
        if !node_by_id.contains_key(&entity_root) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing entity root node id {}", entity_root),
            ));
        }
        validate_index_subtree(
            entity_root,
            None,
            IndexSubtreeKind::Entity,
            &node_by_id,
            &mut seen,
            &mut stack,
        )?;
    }

    if seen.len() != index.nodes.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "index contains {} unreachable nodes (reachable={} total={})",
                index.nodes.len().saturating_sub(seen.len()),
                seen.len(),
                index.nodes.len()
            ),
        ));
    }

    Ok(())
}

fn validate_index_subtree(
    node_id: u32,
    parent_bounds: Option<([i32; 4], [i32; 4])>,
    subtree_kind: IndexSubtreeKind,
    node_by_id: &HashMap<u32, &IndexNode>,
    seen: &mut HashSet<u32>,
    stack: &mut HashSet<u32>,
) -> io::Result<()> {
    if !stack.insert(node_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cycle detected in index at node {}", node_id),
        ));
    }
    if !seen.insert(node_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("node {} referenced multiple times", node_id),
        ));
    }

    let Some(node) = node_by_id.get(&node_id).copied() else {
        stack.remove(&node_id);
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing node id {}", node_id),
        ));
    };

    if let Some((parent_min, parent_max)) = parent_bounds {
        if !bounds_contains(
            parent_min,
            parent_max,
            node.bounds_min_chunk,
            node.bounds_max_chunk,
        ) {
            stack.remove(&node_id);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "node {} bounds {:?}..{:?} not contained in parent {:?}..{:?}",
                    node.node_id,
                    node.bounds_min_chunk,
                    node.bounds_max_chunk,
                    parent_min,
                    parent_max
                ),
            ));
        }
    }

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            let mut child_nodes = Vec::<&IndexNode>::with_capacity(child_node_ids.len());
            let mut unique_child_ids = HashSet::<u32>::new();
            for child_id in child_node_ids {
                if !unique_child_ids.insert(*child_id) {
                    stack.remove(&node_id);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "node {} contains duplicate child id {}",
                            node.node_id, child_id
                        ),
                    ));
                }
                let Some(child) = node_by_id.get(child_id).copied() else {
                    stack.remove(&node_id);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "node {} references missing child id {}",
                            node.node_id, child_id
                        ),
                    ));
                };
                child_nodes.push(child);
            }

            for window in child_nodes.windows(2) {
                let left = window[0];
                let right = window[1];
                let left_key = (
                    bounds_key(left.bounds_min_chunk, left.bounds_max_chunk),
                    index_kind_order_key(&left.kind),
                );
                let right_key = (
                    bounds_key(right.bounds_min_chunk, right.bounds_max_chunk),
                    index_kind_order_key(&right.kind),
                );
                if left_key > right_key {
                    stack.remove(&node_id);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "node {} has non-canonical child ordering between {} and {}",
                            node.node_id, left.node_id, right.node_id
                        ),
                    ));
                }
            }

            for i in 0..child_nodes.len() {
                for j in (i + 1)..child_nodes.len() {
                    let a = child_nodes[i];
                    let b = child_nodes[j];
                    if bounds_intersect(
                        a.bounds_min_chunk,
                        a.bounds_max_chunk,
                        b.bounds_min_chunk,
                        b.bounds_max_chunk,
                    ) {
                        stack.remove(&node_id);
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "node {} has overlapping children {} and {}",
                                node.node_id, a.node_id, b.node_id
                            ),
                        ));
                    }
                }
            }

            for child in child_nodes {
                validate_index_subtree(
                    child.node_id,
                    Some((node.bounds_min_chunk, node.bounds_max_chunk)),
                    subtree_kind,
                    node_by_id,
                    seen,
                    stack,
                )?;
            }
        }
        IndexNodeKind::LeafEmpty => {}
        IndexNodeKind::LeafUniform { .. } => {
            if subtree_kind == IndexSubtreeKind::Entity {
                stack.remove(&node_id);
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "entity subtree node {} cannot use LeafUniform",
                        node.node_id
                    ),
                ));
            }
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => match subtree_kind {
            IndexSubtreeKind::World => {
                if chunk_array_ref.blob_type != BLOB_KIND_CHUNK_ARRAY {
                    stack.remove(&node_id);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "world subtree node {} references blob type {}, expected {}",
                            node.node_id, chunk_array_ref.blob_type, BLOB_KIND_CHUNK_ARRAY
                        ),
                    ));
                }
            }
            IndexSubtreeKind::Entity => {
                if chunk_array_ref.blob_type != BLOB_KIND_ENTITY {
                    stack.remove(&node_id);
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "entity subtree node {} references blob type {}, expected {}",
                            node.node_id, chunk_array_ref.blob_type, BLOB_KIND_ENTITY
                        ),
                    ));
                }
            }
        },
    }

    stack.remove(&node_id);
    Ok(())
}

fn write_payload_file<T: Serialize>(
    path: PathBuf,
    magic: &[u8; 4],
    file_version: u32,
    payload: &T,
) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let encoded = postcard::to_stdvec(payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let checksum = crc32(&encoded);

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(magic)?;
    writer.write_all(&file_version.to_le_bytes())?;
    writer.write_all(&(encoded.len() as u32).to_le_bytes())?;
    writer.write_all(&checksum.to_le_bytes())?;
    writer.write_all(&encoded)?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
    file.sync_all()?;
    Ok(())
}

fn read_payload_file<T: for<'de> Deserialize<'de>>(
    path: PathBuf,
    expected_magic: &[u8; 4],
) -> io::Result<T> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut header = [0u8; 16];
    reader.read_exact(&mut header)?;
    if &header[0..4] != expected_magic {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload file magic mismatch",
        ));
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != PAYLOAD_FILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported payload version {version}"),
        ));
    }
    let payload_len = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let checksum = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;
    if crc32(&payload) != checksum {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload file checksum mismatch",
        ));
    }
    postcard::from_bytes(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn fsync_directory(path: &Path) {
    if let Ok(dir) = File::open(path) {
        let _ = dir.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_UNIQUIFIER: AtomicU64 = AtomicU64::new(0);

    fn test_root(name: &str) -> PathBuf {
        let serial = TEST_UNIQUIFIER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "polychora-save-v4-{name}-{}-{}",
            std::process::id(),
            serial
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test save root");
        path
    }

    #[test]
    fn save_and_load_roundtrip_preserves_world_entities_players() {
        let root = test_root("roundtrip");
        let now_ms = now_unix_ms();

        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(1, 1, 1, 1, VoxelType(3));
        world.set_voxel(40, 2, -5, 17, VoxelType(7));

        let entities = vec![PersistedEntityRecord {
            entity_id: 42,
            class: EntityClass::Accent,
            kind: EntityKind::TestRotor,
            position: [2.0, 3.0, 4.0, 5.0],
            orientation: [0.0, 0.0, 1.0, 0.0],
            velocity: [0.1, 0.2, 0.3, 0.4],
            scale: 0.75,
            material: 9,
            display_name: Some("spin".to_string()),
            tags: vec!["demo".to_string()],
            payload: vec![1, 2, 3, 4],
            last_saved_ms: now_ms,
        }];
        let players = vec![PlayerRecord {
            player_id: 7,
            position: [8.0, 9.0, 10.0, 11.0],
            orientation: [0.0, 0.0, 1.0, 0.0],
            tags: vec!["tester".to_string()],
            inventory_payload: vec![99],
            last_saved_ms: now_ms,
        }];
        let empty_regions = HashSet::new();

        let save_result = save_state(
            &root,
            SaveRequest {
                world: &world,
                entities: &entities,
                players: &players,
                world_seed: 4242,
                next_entity_id: 1000,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: true,
                force_full_entities: true,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("save state");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&root).expect("load state");
        assert_eq!(loaded.manifest.version, SAVE_FORMAT_VERSION);
        assert_eq!(loaded.global.world_seed, 4242);
        assert_eq!(loaded.global.next_entity_id, 1000);
        assert_eq!(loaded.players.players.len(), 1);
        assert_eq!(loaded.entities.len(), 1);
        assert_eq!(loaded.world.get_voxel(1, 1, 1, 1), VoxelType(3));
        assert_eq!(loaded.world.get_voxel(40, 2, -5, 17), VoxelType(7));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migrate_legacy_world_to_v4_roundtrip() {
        let root = test_root("migrate");
        let legacy_world = root.join("legacy.v4dw");
        let output_root = root.join("migrated-v4");

        let mut world = VoxelWorld::new();
        world.set_voxel(2, 2, 2, 2, VoxelType(9));
        {
            let mut writer = BufWriter::new(File::create(&legacy_world).expect("create legacy"));
            crate::shared::voxel::save_world(&world, &mut writer).expect("save legacy");
            writer.flush().expect("flush legacy");
        }

        let save_result =
            migrate_legacy_world_to_v4(&legacy_world, None, &output_root, 777, now_unix_ms())
                .expect("migrate legacy to v4");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&output_root).expect("load migrated");
        assert_eq!(loaded.global.world_seed, 777);
        assert_eq!(loaded.world.get_voxel(2, 2, 2, 2), VoxelType(9));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migrate_v3_save_to_v4_roundtrip() {
        let root = test_root("migrate-v3");
        let v3_root = root.join("input-v3");
        let output_root = root.join("output-v4");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(4, 1, 2, -3, VoxelType(8));

        let entities = vec![crate::save_v3::PersistedEntityRecord {
            entity_id: 99,
            class: EntityClass::Accent,
            kind: EntityKind::TestCube,
            position: [1.0, 2.0, 3.0, 4.0],
            orientation: [0.0, 0.0, 1.0, 0.0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            scale: 1.0,
            material: 7,
            display_name: Some("marker".to_string()),
            tags: vec!["persist".to_string()],
            payload: vec![9, 8, 7],
            last_saved_ms: now_ms,
        }];
        let players = vec![crate::save_v3::PlayerRecord {
            player_id: 123,
            position: [0.0, 1.0, 2.0, 3.0],
            orientation: [0.0, 0.0, 1.0, 0.0],
            tags: vec!["p".to_string()],
            inventory_payload: vec![5, 4, 3],
            last_saved_ms: now_ms,
        }];
        let custom_payload = vec![42, 24, 7];

        crate::save_v3::save_state(
            &v3_root,
            crate::save_v3::SaveRequest {
                world: &world,
                entities: &entities,
                players: &players,
                world_seed: 2026,
                next_entity_id: 1000,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: true,
                force_full_entities: true,
                custom_global_payload: Some(custom_payload.clone()),
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("create v3 input");

        let save_result =
            migrate_v3_save_to_v4(&v3_root, &output_root, false, now_ms.saturating_add(1))
                .expect("migrate v3 -> v4");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&output_root).expect("load v4 output");
        assert_eq!(loaded.manifest.version, SAVE_FORMAT_VERSION);
        assert_eq!(loaded.global.world_seed, 2026);
        assert_eq!(loaded.global.next_entity_id, 1000);
        assert_eq!(loaded.global.custom_global_payload, custom_payload);
        assert_eq!(loaded.world.get_voxel(4, 1, 2, -3), VoxelType(8));
        assert_eq!(loaded.entities.len(), 1);
        assert_eq!(loaded.players.players.len(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn validate_index_rejects_overlapping_siblings() {
        let index = IndexPayload {
            generation: 1,
            root_node_id: 0,
            entity_root_node_id: None,
            nodes: vec![
                IndexNode {
                    node_id: 0,
                    bounds_min_chunk: [0, 0, 0, 0],
                    bounds_max_chunk: [1, 0, 0, 0],
                    kind: IndexNodeKind::Branch {
                        child_node_ids: vec![1, 2],
                    },
                },
                IndexNode {
                    node_id: 1,
                    bounds_min_chunk: [0, 0, 0, 0],
                    bounds_max_chunk: [0, 0, 0, 0],
                    kind: IndexNodeKind::LeafEmpty,
                },
                IndexNode {
                    node_id: 2,
                    bounds_min_chunk: [0, 0, 0, 0],
                    bounds_max_chunk: [0, 0, 0, 0],
                    kind: IndexNodeKind::LeafEmpty,
                },
            ],
        };

        let err = validate_index_payload(&index).expect_err("overlap should fail");
        assert!(err.to_string().contains("overlapping children"));
    }

    #[test]
    fn validate_index_rejects_non_canonical_child_order() {
        let index = IndexPayload {
            generation: 1,
            root_node_id: 0,
            entity_root_node_id: None,
            nodes: vec![
                IndexNode {
                    node_id: 0,
                    bounds_min_chunk: [0, 0, 0, 0],
                    bounds_max_chunk: [1, 0, 0, 0],
                    kind: IndexNodeKind::Branch {
                        child_node_ids: vec![2, 1],
                    },
                },
                IndexNode {
                    node_id: 1,
                    bounds_min_chunk: [0, 0, 0, 0],
                    bounds_max_chunk: [0, 0, 0, 0],
                    kind: IndexNodeKind::LeafEmpty,
                },
                IndexNode {
                    node_id: 2,
                    bounds_min_chunk: [1, 0, 0, 0],
                    bounds_max_chunk: [1, 0, 0, 0],
                    kind: IndexNodeKind::LeafEmpty,
                },
            ],
        };

        let err = validate_index_payload(&index).expect_err("order should fail");
        assert!(err.to_string().contains("non-canonical child ordering"));
    }

    #[test]
    fn validate_index_rejects_wrong_world_leaf_blob_type() {
        let index = IndexPayload {
            generation: 1,
            root_node_id: 0,
            entity_root_node_id: None,
            nodes: vec![IndexNode {
                node_id: 0,
                bounds_min_chunk: [0, 0, 0, 0],
                bounds_max_chunk: [0, 0, 0, 0],
                kind: IndexNodeKind::LeafChunkArray {
                    chunk_array_ref: BlobRef {
                        data_file_id: 1,
                        record_offset: 12,
                        record_len: 64,
                        record_crc32: 0,
                        blob_type: BLOB_KIND_ENTITY,
                        blob_version: ENTITY_BLOB_VERSION,
                    },
                },
            }],
        };

        let err = validate_index_payload(&index).expect_err("wrong blob type should fail");
        assert!(err.to_string().contains("expected 2"));
    }

    #[test]
    fn save_state_writes_deterministic_index_for_equivalent_worlds() {
        let root_a = test_root("deterministic-a");
        let root_b = test_root("deterministic-b");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world_a = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world_a.set_voxel(1, 2, 3, 4, VoxelType(5));
        world_a.set_voxel(-8, 0, 1, -3, VoxelType(9));
        world_a.set_voxel(33, 7, -2, 12, VoxelType(4));

        let mut world_b = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world_b.set_voxel(33, 7, -2, 12, VoxelType(4));
        world_b.set_voxel(1, 2, 3, 4, VoxelType(5));
        world_b.set_voxel(-8, 0, 1, -3, VoxelType(9));

        save_state(
            &root_a,
            SaveRequest {
                world: &world_a,
                entities: &[],
                players: &[],
                world_seed: 99,
                next_entity_id: 1,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: true,
                force_full_entities: true,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("save root a");
        save_state(
            &root_b,
            SaveRequest {
                world: &world_b,
                entities: &[],
                players: &[],
                world_seed: 99,
                next_entity_id: 1,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: true,
                force_full_entities: true,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("save root b");

        let manifest_a = load_manifest(&root_a).expect("manifest a");
        let manifest_b = load_manifest(&root_b).expect("manifest b");
        let index_a = std::fs::read(root_a.join(&manifest_a.index_file)).expect("read index a");
        let index_b = std::fs::read(root_b.join(&manifest_b.index_file)).expect("read index b");
        assert_eq!(index_a, index_b);

        let _ = std::fs::remove_dir_all(root_a);
        let _ = std::fs::remove_dir_all(root_b);
    }

    #[test]
    fn disable_block_persistence_clears_block_overrides_and_keeps_custom_payload() {
        let root = test_root("disable-block-persistence");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(8, 8, 8, 8, VoxelType(3));

        save_state(
            &root,
            SaveRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 123,
                next_entity_id: 1,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: true,
                force_full_entities: true,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("initial save");

        let payload = vec![7, 6, 5, 4, 3, 2, 1];
        let empty_world = VoxelWorld::new_with_base(world.base_kind());
        save_state(
            &root,
            SaveRequest {
                world: &empty_world,
                entities: &[],
                players: &[],
                world_seed: 123,
                next_entity_id: 2,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: false,
                force_full_entities: false,
                player_entity_hints: None,
                custom_global_payload: Some(payload.clone()),
                disable_block_persistence: true,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("save without block persistence");

        let loaded = load_state(&root).expect("load state");
        assert_eq!(loaded.global.custom_global_payload, payload);
        assert_eq!(loaded.world.get_voxel(8, 8, 8, 8), VoxelType(0));

        let _ = std::fs::remove_dir_all(root);
    }
}
