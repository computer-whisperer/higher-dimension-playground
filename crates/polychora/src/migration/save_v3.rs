use crate::migration::legacy_voxel::{Chunk, RegionChunkWorld};
use crate::migration::legacy_world_io::load_world;
use crate::shared::protocol::{EntityClass, EntityKind};
use crate::shared::voxel::{BaseWorldKind, ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
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
const BLOB_KIND_BLOCK: u8 = 1;
const BLOB_KIND_ENTITY: u8 = 2;

pub const SAVE_FORMAT_TAG: &str = "polychora-save";
pub const SAVE_FORMAT_VERSION: u32 = 3;
pub const PAYLOAD_FILE_VERSION: u32 = 1;
pub const DATA_FILE_VERSION: u32 = 1;
pub const BLOCK_BLOB_VERSION: u16 = 1;
pub const ENTITY_BLOB_VERSION: u16 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveLimits {
    pub region_chunk_edge: i32,
    pub data_file_max_bytes: u64,
    pub index_soft_max_bytes: u64,
    pub block_blob_target_bytes: u32,
    pub block_blob_hard_max_bytes: u32,
    pub entity_blob_target_bytes: u32,
    pub entity_blob_hard_max_bytes: u32,
}

impl Default for SaveLimits {
    fn default() -> Self {
        Self {
            region_chunk_edge: 4,
            data_file_max_bytes: 128 * 1024 * 1024,
            index_soft_max_bytes: 4 * 1024 * 1024,
            block_blob_target_bytes: 128 * 1024,
            block_blob_hard_max_bytes: 512 * 1024,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeafEntry {
    pub region: [i32; 4],
    pub block_blob: Option<BlobRef>,
    pub entity_blob: Option<BlobRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IndexPayload {
    pub generation: u64,
    pub region_chunk_edge: i32,
    pub leaves: Vec<LeafEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkPayload {
    pub voxels: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkFillState {
    Virgin,
    EmptyOverride,
    DictRef,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkAssignment {
    pub local_chunk_coord: [u16; 4],
    pub state: ChunkFillState,
    pub dict_index: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockBlob {
    pub region: [i32; 4],
    pub region_chunk_edge: i32,
    pub chunk_dictionary: Vec<ChunkPayload>,
    pub assignments: Vec<ChunkAssignment>,
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
    pub region: [i32; 4],
    pub region_chunk_edge: i32,
    pub entities: Vec<PersistedEntityRecord>,
}

#[derive(Debug)]
pub struct LoadedState {
    pub manifest: Manifest,
    pub global: GlobalPayload,
    pub players: PlayersPayload,
    pub index: IndexPayload,
    pub world: RegionChunkWorld,
    pub entities: Vec<PersistedEntityRecord>,
}

pub struct SaveRequest<'a> {
    pub world: &'a RegionChunkWorld,
    pub entities: &'a [PersistedEntityRecord],
    pub players: &'a [PlayerRecord],
    pub world_seed: u64,
    pub next_entity_id: u64,
    pub dirty_block_regions: &'a HashSet<[i32; 4]>,
    pub dirty_entity_regions: &'a HashSet<[i32; 4]>,
    pub force_full_blocks: bool,
    pub force_full_entities: bool,
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

pub fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis().min(u64::MAX as u128) as u64
}

pub fn is_v3_save_root(path: &Path) -> bool {
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

pub fn all_block_regions(world: &RegionChunkWorld, region_chunk_edge: i32) -> HashSet<[i32; 4]> {
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
    if !is_v3_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    load_state(root)
}

pub fn load_state(root: &Path) -> io::Result<LoadedState> {
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let players: PlayersPayload =
        read_payload_file(root.join(&manifest.players_file), PLAYERS_MAGIC)?;
    let index: IndexPayload = read_payload_file(root.join(&manifest.index_file), INDEX_MAGIC)?;

    let mut world = RegionChunkWorld::new_with_base(global.base_world_kind.to_runtime());
    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();

    for leaf in &index.leaves {
        if let Some(block_ref) = &leaf.block_blob {
            let payload = read_blob_payload(root, block_ref)?;
            let blob: BlockBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            apply_block_blob_to_world(&blob, &mut world)?;
        }
        if let Some(entity_ref) = &leaf.entity_blob {
            let payload = read_blob_payload(root, entity_ref)?;
            let blob: EntityBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            for entity in blob.entities {
                entities_by_id.insert(entity.entity_id, entity);
            }
        }
    }

    world.clear_dirty();
    let _ = world.drain_pending_chunk_updates();

    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);

    Ok(LoadedState {
        manifest,
        global,
        players,
        index,
        world,
        entities,
    })
}

fn load_state_without_blocks(root: &Path) -> io::Result<LoadedState> {
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let players: PlayersPayload =
        read_payload_file(root.join(&manifest.players_file), PLAYERS_MAGIC)?;
    let index: IndexPayload = read_payload_file(root.join(&manifest.index_file), INDEX_MAGIC)?;

    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();
    for leaf in &index.leaves {
        if let Some(entity_ref) = &leaf.entity_blob {
            let payload = read_blob_payload(root, entity_ref)?;
            let blob: EntityBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            for entity in blob.entities {
                entities_by_id.insert(entity.entity_id, entity);
            }
        }
    }

    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);

    let mut world = RegionChunkWorld::new_with_base(global.base_world_kind.to_runtime());
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

fn load_or_init_state_without_blocks(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedState> {
    if !is_v3_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    load_state_without_blocks(root)
}

pub fn save_state(root: &Path, request: SaveRequest<'_>) -> io::Result<SaveResult> {
    let loaded = if request.disable_block_persistence {
        load_or_init_state_without_blocks(
            root,
            request.world.base_kind(),
            request.world_seed,
            request.now_ms,
        )?
    } else {
        load_or_init_state(
            root,
            request.world.base_kind(),
            request.world_seed,
            request.now_ms,
        )?
    };
    let mut manifest = loaded.manifest;
    let mut index = loaded.index;

    if manifest.limits.region_chunk_edge <= 0 {
        manifest.limits.region_chunk_edge = SaveLimits::default().region_chunk_edge;
    }
    index.region_chunk_edge = manifest.limits.region_chunk_edge;

    let mut leaf_by_region: HashMap<[i32; 4], LeafEntry> = index
        .leaves
        .into_iter()
        .map(|leaf| (leaf.region, leaf))
        .collect();

    let mut saved_block_regions = 0usize;
    if request.disable_block_persistence {
        for leaf in leaf_by_region.values_mut() {
            leaf.block_blob = None;
        }
    } else {
        let mut block_regions = HashSet::new();
        if request.force_full_blocks {
            block_regions.extend(all_block_regions(
                request.world,
                manifest.limits.region_chunk_edge,
            ));
            for leaf in leaf_by_region.values() {
                if leaf.block_blob.is_some() {
                    block_regions.insert(leaf.region);
                }
            }
        } else {
            block_regions.extend(request.dirty_block_regions.iter().copied());
        }

        let mut sorted_block_regions: Vec<[i32; 4]> = block_regions.into_iter().collect();
        sorted_block_regions.sort_unstable();

        for region in sorted_block_regions {
            let maybe_blob = build_block_blob_for_region(
                request.world,
                region,
                manifest.limits.region_chunk_edge,
            );
            match maybe_blob {
                Some(blob) => {
                    let payload = postcard::to_stdvec(&blob)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                    let blob_ref = append_blob_record(
                        root,
                        &mut manifest,
                        BLOB_KIND_BLOCK,
                        BLOCK_BLOB_VERSION,
                        &payload,
                    )?;
                    let leaf = leaf_by_region.entry(region).or_insert(LeafEntry {
                        region,
                        block_blob: None,
                        entity_blob: None,
                    });
                    leaf.block_blob = Some(blob_ref);
                    saved_block_regions += 1;
                }
                None => {
                    if let Some(leaf) = leaf_by_region.get_mut(&region) {
                        leaf.block_blob = None;
                    }
                }
            }
        }
    }

    let mut entities_by_region = HashMap::<[i32; 4], Vec<PersistedEntityRecord>>::new();
    for entity in request.entities {
        let region = region_from_chunk(
            chunk_from_world_position(entity.position),
            manifest.limits.region_chunk_edge,
        );
        entities_by_region
            .entry(region)
            .or_default()
            .push(entity.clone());
    }
    for entities in entities_by_region.values_mut() {
        entities.sort_unstable_by_key(|entity| entity.entity_id);
    }

    let mut entity_regions = HashSet::new();
    if request.force_full_entities {
        entity_regions.extend(entities_by_region.keys().copied());
        for leaf in leaf_by_region.values() {
            if leaf.entity_blob.is_some() {
                entity_regions.insert(leaf.region);
            }
        }
    } else {
        entity_regions.extend(request.dirty_entity_regions.iter().copied());
    }

    let mut sorted_entity_regions: Vec<[i32; 4]> = entity_regions.into_iter().collect();
    sorted_entity_regions.sort_unstable();
    let mut saved_entity_regions = 0usize;

    for region in sorted_entity_regions {
        let entities = entities_by_region.remove(&region).unwrap_or_default();
        if entities.is_empty() {
            if let Some(leaf) = leaf_by_region.get_mut(&region) {
                leaf.entity_blob = None;
            }
            continue;
        }

        let blob = EntityBlob {
            region,
            region_chunk_edge: manifest.limits.region_chunk_edge,
            entities,
        };
        let payload = postcard::to_stdvec(&blob)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        let blob_ref = append_blob_record(
            root,
            &mut manifest,
            BLOB_KIND_ENTITY,
            ENTITY_BLOB_VERSION,
            &payload,
        )?;
        let leaf = leaf_by_region.entry(region).or_insert(LeafEntry {
            region,
            block_blob: None,
            entity_blob: None,
        });
        leaf.entity_blob = Some(blob_ref);
        saved_entity_regions += 1;
    }

    leaf_by_region.retain(|_, leaf| leaf.block_blob.is_some() || leaf.entity_blob.is_some());
    let mut leaves: Vec<LeafEntry> = leaf_by_region.into_values().collect();
    leaves.sort_unstable_by_key(|leaf| leaf.region);

    let next_generation = manifest.current_generation.saturating_add(1);
    let next_index_file = index_generation_path(next_generation);
    let next_global_file = global_generation_path(next_generation);
    let next_players_file = players_generation_path(next_generation);

    let next_index = IndexPayload {
        generation: next_generation,
        region_chunk_edge: manifest.limits.region_chunk_edge,
        leaves,
    };
    write_payload_file(
        root.join(&next_index_file),
        INDEX_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_index,
    )?;

    let next_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(request.world.base_kind()),
        world_seed: request.world_seed,
        next_entity_id: request.next_entity_id,
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: request.now_ms,
        player_entity_hints: loaded.global.player_entity_hints,
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
        saved_block_regions,
        saved_entity_regions,
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

pub fn migrate_legacy_world_to_v3(
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

    let block_regions = all_block_regions(&world, SaveLimits::default().region_chunk_edge);
    let entity_regions = all_entity_regions(&entities, SaveLimits::default().region_chunk_edge);
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
            custom_global_payload: None,
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
        region_chunk_edge: limits.region_chunk_edge,
        leaves: Vec::new(),
    };
    write_payload_file(
        root.join(&index_file),
        INDEX_MAGIC,
        PAYLOAD_FILE_VERSION,
        &initial_index,
    )?;

    let initial_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(base_world_kind),
        world_seed,
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

fn apply_block_blob_to_world(blob: &BlockBlob, world: &mut RegionChunkWorld) -> io::Result<()> {
    let edge = blob.region_chunk_edge.max(1);
    let region_min = [
        blob.region[0].saturating_mul(edge),
        blob.region[1].saturating_mul(edge),
        blob.region[2].saturating_mul(edge),
        blob.region[3].saturating_mul(edge),
    ];

    for assignment in &blob.assignments {
        let chunk_pos = ChunkPos::new(
            region_min[0].saturating_add(assignment.local_chunk_coord[0] as i32),
            region_min[1].saturating_add(assignment.local_chunk_coord[1] as i32),
            region_min[2].saturating_add(assignment.local_chunk_coord[2] as i32),
            region_min[3].saturating_add(assignment.local_chunk_coord[3] as i32),
        );
        match assignment.state {
            ChunkFillState::Virgin => {
                let _ = world.remove_chunk_override(chunk_pos);
            }
            ChunkFillState::EmptyOverride => {
                world.insert_chunk(chunk_pos, Chunk::new());
            }
            ChunkFillState::DictRef => {
                let Some(payload) = blob.chunk_dictionary.get(assignment.dict_index as usize)
                else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "block blob dict index out of bounds",
                    ));
                };
                let chunk = chunk_from_payload(payload)?;
                world.insert_chunk(chunk_pos, chunk);
            }
        }
    }
    Ok(())
}

fn build_block_blob_for_region(
    world: &RegionChunkWorld,
    region: [i32; 4],
    region_chunk_edge: i32,
) -> Option<BlockBlob> {
    let mut assignments = Vec::<ChunkAssignment>::new();
    let mut dictionary = Vec::<ChunkPayload>::new();
    let mut dict_index_by_bytes = HashMap::<Vec<u8>, u16>::new();

    for (&pos, chunk) in &world.chunks {
        let chunk_region = region_from_chunk_pos(pos, region_chunk_edge);
        if chunk_region != region {
            continue;
        }
        let region_min = [
            region[0].saturating_mul(region_chunk_edge.max(1)),
            region[1].saturating_mul(region_chunk_edge.max(1)),
            region[2].saturating_mul(region_chunk_edge.max(1)),
            region[3].saturating_mul(region_chunk_edge.max(1)),
        ];
        let local = [
            pos.x.saturating_sub(region_min[0]) as u16,
            pos.y.saturating_sub(region_min[1]) as u16,
            pos.z.saturating_sub(region_min[2]) as u16,
            pos.w.saturating_sub(region_min[3]) as u16,
        ];
        if chunk.is_empty() {
            assignments.push(ChunkAssignment {
                local_chunk_coord: local,
                state: ChunkFillState::EmptyOverride,
                dict_index: 0,
            });
            continue;
        }
        let bytes: Vec<u8> = chunk.voxels.iter().map(|voxel| voxel.0).collect();
        let dict_index = if let Some(index) = dict_index_by_bytes.get(&bytes).copied() {
            index
        } else {
            let index = dictionary.len() as u16;
            dictionary.push(ChunkPayload {
                voxels: bytes.clone(),
            });
            dict_index_by_bytes.insert(bytes, index);
            index
        };
        assignments.push(ChunkAssignment {
            local_chunk_coord: local,
            state: ChunkFillState::DictRef,
            dict_index,
        });
    }

    if assignments.is_empty() {
        return None;
    }
    assignments.sort_unstable_by_key(|assignment| assignment.local_chunk_coord);
    Some(BlockBlob {
        region,
        region_chunk_edge: region_chunk_edge.max(1),
        chunk_dictionary: dictionary,
        assignments,
    })
}

fn chunk_from_payload(payload: &ChunkPayload) -> io::Result<Chunk> {
    if payload.voxels.len() != CHUNK_VOLUME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid chunk payload length {}, expected {}",
                payload.voxels.len(),
                CHUNK_VOLUME
            ),
        ));
    }
    let mut chunk = Chunk::new();
    chunk.solid_count = 0;
    for (idx, &value) in payload.voxels.iter().enumerate() {
        chunk.voxels[idx] = VoxelType(value);
        if value != VoxelType::AIR.0 {
            chunk.solid_count = chunk.solid_count.saturating_add(1);
        }
    }
    chunk.dirty = true;
    Ok(chunk)
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
            "unsupported v3 save manifest format/version",
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
            "polychora-save-v3-{name}-{}-{}",
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

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
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
    fn migrate_legacy_world_to_v3_roundtrip() {
        let root = test_root("migrate");
        let legacy_world = root.join("legacy.v4dw");
        let output_root = root.join("migrated-v3");

        let mut world = RegionChunkWorld::new();
        world.set_voxel(2, 2, 2, 2, VoxelType(9));
        {
            let mut writer = BufWriter::new(File::create(&legacy_world).expect("create legacy"));
            crate::migration::legacy_world_io::save_world(&world, &mut writer)
                .expect("save legacy");
            writer.flush().expect("flush legacy");
        }

        let save_result =
            migrate_legacy_world_to_v3(&legacy_world, None, &output_root, 777, now_unix_ms())
                .expect("migrate legacy to v3");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&output_root).expect("load migrated");
        assert_eq!(loaded.global.world_seed, 777);
        assert_eq!(loaded.world.get_voxel(2, 2, 2, 2), VoxelType(9));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn disable_block_persistence_clears_block_blobs_and_keeps_custom_payload() {
        let root = test_root("disable-block-persistence");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
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
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms,
            },
        )
        .expect("initial save");

        let payload = vec![7, 6, 5, 4, 3, 2, 1];
        let empty_world = RegionChunkWorld::new_with_base(world.base_kind());
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
                custom_global_payload: Some(payload.clone()),
                disable_block_persistence: true,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("save without block persistence");

        let loaded = load_state(&root).expect("load state");
        assert_eq!(loaded.global.custom_global_payload, payload);
        assert_eq!(loaded.world.get_voxel(8, 8, 8, 8), VoxelType(0));
        assert!(loaded
            .index
            .leaves
            .iter()
            .all(|leaf| leaf.block_blob.is_none()));

        let _ = std::fs::remove_dir_all(root);
    }
}
