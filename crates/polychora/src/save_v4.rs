#[cfg(test)]
use crate::migration::legacy_voxel::{Chunk as LegacyChunk, RegionChunkWorld};
use crate::shared::chunk_payload::{
    ChunkArrayData, ChunkArrayIndexCodec, ChunkPayload as FieldChunkPayload, ResolvedChunkPayload,
};
use crate::shared::protocol::{EntityClass, EntityKind};
use crate::shared::region_tree::{slice_region_core_in_bounds, RegionNodeKind, RegionTreeCore};
use crate::shared::spatial::Aabb4i;
#[cfg(test)]
use crate::shared::voxel::CHUNK_VOLUME;
use crate::shared::voxel::{BaseWorldKind, BlockData, ChunkPos, VoxelType, CHUNK_SIZE};
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
const RECORD_HEADER_LEN: u32 = 4 + 1 + 2 + 4 + 4;
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
pub const CHUNK_ARRAY_BLOB_VERSION: u16 = 2;
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
    MassivePlatforms { material: u8 },
}

impl PersistedBaseWorldKind {
    pub fn from_runtime(base: BaseWorldKind) -> Self {
        match base {
            BaseWorldKind::Empty => Self::Empty,
            BaseWorldKind::FlatFloor { material } => Self::FlatFloor {
                material: material.0,
            },
            BaseWorldKind::MassivePlatforms { material } => Self::MassivePlatforms {
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
            PersistedBaseWorldKind::MassivePlatforms { material } => {
                BaseWorldKind::MassivePlatforms {
                    material: VoxelType(material),
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    LeafUniform { block: BlockData },
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
    pub block_palette: Vec<BlockData>,
    pub index_codec: ChunkArrayIndexCodec,
    pub index_data: Vec<u8>,
    pub default_palette_index: Option<u16>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    pub world_chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
    pub entities: Vec<PersistedEntityRecord>,
}

#[derive(Debug)]
pub struct LoadedStateMetadata {
    pub manifest: Manifest,
    pub global: GlobalPayload,
    pub players: PlayersPayload,
    pub index: IndexPayload,
    pub entities: Vec<PersistedEntityRecord>,
}

#[derive(Debug)]
struct LoadedSaveMetadata {
    manifest: Manifest,
    global: GlobalPayload,
    index: IndexPayload,
}

#[cfg(test)]
struct SaveWorldRequest<'a> {
    pub world: &'a RegionChunkWorld,
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

pub struct SaveChunkPayloadRequest<'a> {
    pub base_world_kind: BaseWorldKind,
    pub chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
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

pub struct SaveChunkPayloadPatchRequest {
    pub base_world_kind: BaseWorldKind,
    pub dirty_chunk_payloads: Vec<([i32; 4], Option<ResolvedChunkPayload>)>,
    pub world_seed: u64,
    pub next_entity_id: u64,
    pub player_entity_hints: Option<Vec<PlayerEntityHint>>,
    pub custom_global_payload: Option<Vec<u8>>,
    pub now_ms: u64,
}

struct SaveStateCommon<'a> {
    entities: &'a [PersistedEntityRecord],
    players: &'a [PlayerRecord],
    world_seed: u64,
    next_entity_id: u64,
    dirty_block_regions: &'a HashSet<[i32; 4]>,
    dirty_entity_regions: &'a HashSet<[i32; 4]>,
    force_full_blocks: bool,
    force_full_entities: bool,
    player_entity_hints: Option<Vec<PlayerEntityHint>>,
    custom_global_payload: Option<Vec<u8>>,
    disable_block_persistence: bool,
    now_ms: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SaveResult {
    pub generation: u64,
    pub saved_block_regions: usize,
    pub saved_entity_regions: usize,
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BlobReuseKey {
    blob_type: u8,
    blob_version: u16,
    payload_len: u32,
    payload_crc32: u32,
}

#[derive(Clone, Debug, Default)]
struct BlobReuseIndex {
    refs_by_key: HashMap<BlobReuseKey, Vec<BlobRef>>,
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
    let metadata = load_or_init_state_metadata(root, default_base, world_seed, now_ms)?;
    let world_chunk_payloads = materialize_world_chunk_payloads_from_index_filtered(
        root,
        &metadata.index,
        None,
        true,
        None,
    )?;
    Ok(LoadedState {
        manifest: metadata.manifest,
        global: metadata.global,
        players: metadata.players,
        index: metadata.index,
        world_chunk_payloads,
        entities: metadata.entities,
    })
}

pub fn load_or_init_state_metadata(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedStateMetadata> {
    if !is_v4_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    load_state_metadata(root)
}

fn load_or_init_save_metadata(
    root: &Path,
    default_base: BaseWorldKind,
    world_seed: u64,
    now_ms: u64,
) -> io::Result<LoadedSaveMetadata> {
    if !is_v4_save_root(root) {
        init_empty_save_root(root, default_base, world_seed, now_ms)?;
    }
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    Ok(LoadedSaveMetadata {
        manifest,
        global,
        index,
    })
}

pub fn load_state(root: &Path) -> io::Result<LoadedState> {
    let metadata = load_state_metadata(root)?;
    let world_chunk_payloads = materialize_world_chunk_payloads_from_index_filtered(
        root,
        &metadata.index,
        None,
        true,
        None,
    )?;
    Ok(LoadedState {
        manifest: metadata.manifest,
        global: metadata.global,
        players: metadata.players,
        index: metadata.index,
        world_chunk_payloads,
        entities: metadata.entities,
    })
}

pub fn load_state_metadata(root: &Path) -> io::Result<LoadedStateMetadata> {
    let manifest = load_manifest(root)?;
    let global: GlobalPayload = read_payload_file(root.join(&manifest.global_file), GLOBAL_MAGIC)?;
    let players: PlayersPayload =
        read_payload_file(root.join(&manifest.players_file), PLAYERS_MAGIC)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    let mut entities = materialize_entities_from_index(root, &index)?;
    entities.sort_unstable_by_key(|entity| entity.entity_id);

    Ok(LoadedStateMetadata {
        manifest,
        global,
        players,
        index,
        entities,
    })
}

pub fn load_world_chunk_payloads_for_regions(
    root: &Path,
    regions: &HashSet<[i32; 4]>,
) -> io::Result<Vec<([i32; 4], ResolvedChunkPayload)>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    materialize_world_chunk_payloads_from_index_filtered(root, &index, Some(regions), true, None)
}

pub fn load_world_chunk_payloads_for_bounds(
    root: &Path,
    bounds: Aabb4i,
) -> io::Result<Vec<([i32; 4], ResolvedChunkPayload)>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    load_world_chunk_payloads_for_bounds_from_index(root, &index, bounds)
}

pub fn load_world_chunk_payloads_for_bounds_from_index(
    root: &Path,
    index: &IndexPayload,
    bounds: Aabb4i,
) -> io::Result<Vec<([i32; 4], ResolvedChunkPayload)>> {
    if !bounds.is_valid() {
        return Ok(Vec::new());
    }
    materialize_world_chunk_payloads_from_index_filtered(root, index, None, true, Some(bounds))
}

pub fn load_world_subtree_core_for_bounds_from_index(
    root: &Path,
    index: &IndexPayload,
    bounds: Aabb4i,
) -> io::Result<RegionTreeCore> {
    if !bounds.is_valid() {
        return Ok(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        });
    }

    let node_by_id = build_node_lookup(index);
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    let mut chunk_array_cache = HashMap::<(u32, u64, u32, u32, u8, u16), ChunkArrayData>::new();
    let root_core = load_world_subtree_core_from_index_node_filtered(
        root,
        &node_by_id,
        index.root_node_id,
        bounds,
        &mut payload_cache,
        &mut chunk_array_cache,
    )?
    .unwrap_or(RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    });
    Ok(slice_region_core_in_bounds(&root_core, bounds))
}

fn load_world_subtree_core_from_index_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    bounds_filter: Aabb4i,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    chunk_array_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), ChunkArrayData>,
) -> io::Result<Option<RegionTreeCore>> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };

    let node_bounds = Aabb4i::new(node.bounds_min_chunk, node.bounds_max_chunk);
    if !node_bounds.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index world node {node_id} has invalid bounds"),
        ));
    }
    if !node_bounds.intersects(&bounds_filter) {
        return Ok(None);
    }

    let kind = match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            let mut children = Vec::new();
            for child_id in child_node_ids {
                if let Some(child_core) = load_world_subtree_core_from_index_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    bounds_filter,
                    payload_cache,
                    chunk_array_cache,
                )? {
                    children.push(child_core);
                }
            }
            if children.is_empty() {
                RegionNodeKind::Empty
            } else {
                RegionNodeKind::Branch(children)
            }
        }
        IndexNodeKind::LeafEmpty => RegionNodeKind::Empty,
        IndexNodeKind::LeafUniform { block } => {
            RegionNodeKind::Uniform(block.clone())
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_CHUNK_ARRAY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "world subtree leaf references non-chunk-array blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let key = blob_ref_identity(chunk_array_ref);
            let chunk_array = if let Some(cached) = chunk_array_cache.get(&key) {
                cached.clone()
            } else {
                let payload = read_blob_payload(root, chunk_array_ref)?;
                let blob: ChunkArrayBlob = postcard::from_bytes(&payload)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                let chunk_array_bounds = Aabb4i::new(blob.volume_min_chunk, blob.volume_max_chunk);
                if !chunk_array_bounds.is_valid() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid chunk array bounds",
                    ));
                }
                let mut chunk_palette =
                    Vec::<FieldChunkPayload>::with_capacity(blob.payload_palette.len());
                for payload_ref in &blob.payload_palette {
                    chunk_palette.push(load_chunk_payload_from_ref(
                        root,
                        payload_ref,
                        payload_cache,
                    )?);
                }
                let decoded = ChunkArrayData {
                    bounds: chunk_array_bounds,
                    chunk_palette,
                    index_codec: blob.index_codec,
                    index_data: blob.index_data.clone(),
                    default_chunk_idx: blob.default_palette_index,
                    block_palette: blob.block_palette.clone(),
                };
                chunk_array_cache.insert(key, decoded.clone());
                decoded
            };
            RegionNodeKind::ChunkArray(chunk_array)
        }
    };

    Ok(Some(RegionTreeCore {
        bounds: node_bounds,
        kind,
        generator_version_hash: 0,
    }))
}

pub fn load_entities_for_regions(
    root: &Path,
    regions: &HashSet<[i32; 4]>,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let manifest = load_manifest(root)?;
    let index = read_index_file(root.join(&manifest.index_file))?;
    materialize_entities_from_index_filtered(root, &index, Some(regions), true)
}

#[cfg(test)]
fn field_chunk_payload_from_legacy_chunk(chunk: &LegacyChunk) -> FieldChunkPayload {
    let materials: Vec<u16> = chunk
        .voxels
        .iter()
        .map(|voxel| u16::from(voxel.0))
        .collect();
    FieldChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(FieldChunkPayload::Dense16 { materials })
}

#[cfg(test)]
fn legacy_chunk_from_field_chunk_payload(payload: &FieldChunkPayload) -> io::Result<LegacyChunk> {
    let materials = payload
        .dense_materials()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    if materials.len() != CHUNK_VOLUME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chunk payload decoded to {} materials; expected {}",
                materials.len(),
                CHUNK_VOLUME
            ),
        ));
    }

    let mut voxels = Box::new([VoxelType::AIR; CHUNK_VOLUME]);
    let mut solid_count = 0u32;
    for (idx, material) in materials.into_iter().enumerate() {
        let voxel = VoxelType(u8::try_from(material).unwrap_or(u8::MAX));
        if voxel.is_solid() {
            solid_count = solid_count.saturating_add(1);
        }
        voxels[idx] = voxel;
    }
    Ok(LegacyChunk {
        voxels,
        solid_count,
        dirty: false,
    })
}

#[cfg(test)]
fn save_state_from_world(root: &Path, request: SaveWorldRequest<'_>) -> io::Result<SaveResult> {
    let chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)> = request
        .world
        .chunks
        .iter()
        .map(|(&chunk_pos, chunk)| {
            (
                [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
                ResolvedChunkPayload::from_legacy_payload(
                    field_chunk_payload_from_legacy_chunk(chunk),
                ),
            )
        })
        .collect();

    save_state_internal(
        root,
        request.world.base_kind(),
        chunk_payloads,
        SaveStateCommon {
            entities: request.entities,
            players: request.players,
            world_seed: request.world_seed,
            next_entity_id: request.next_entity_id,
            dirty_block_regions: request.dirty_block_regions,
            dirty_entity_regions: request.dirty_entity_regions,
            force_full_blocks: request.force_full_blocks,
            force_full_entities: request.force_full_entities,
            player_entity_hints: request.player_entity_hints,
            custom_global_payload: request.custom_global_payload,
            disable_block_persistence: request.disable_block_persistence,
            now_ms: request.now_ms,
        },
    )
}

pub fn save_state_from_chunk_payloads(
    root: &Path,
    request: SaveChunkPayloadRequest<'_>,
) -> io::Result<SaveResult> {
    save_state_internal(
        root,
        request.base_world_kind,
        request.chunk_payloads,
        SaveStateCommon {
            entities: request.entities,
            players: request.players,
            world_seed: request.world_seed,
            next_entity_id: request.next_entity_id,
            dirty_block_regions: request.dirty_block_regions,
            dirty_entity_regions: request.dirty_entity_regions,
            force_full_blocks: request.force_full_blocks,
            force_full_entities: request.force_full_entities,
            player_entity_hints: request.player_entity_hints,
            custom_global_payload: request.custom_global_payload,
            disable_block_persistence: request.disable_block_persistence,
            now_ms: request.now_ms,
        },
    )
}

pub fn save_state_from_chunk_payload_patch(
    root: &Path,
    request: SaveChunkPayloadPatchRequest,
) -> io::Result<Option<SaveResult>> {
    let loaded = load_or_init_save_metadata(
        root,
        request.base_world_kind,
        request.world_seed,
        request.now_ms,
    )?;
    let LoadedSaveMetadata {
        mut manifest,
        global: loaded_global,
        index: loaded_index,
    } = loaded;
    let mut reuse_index = build_blob_reuse_index(root, &manifest)?;

    let mut dirty_chunk_payloads = HashMap::<[i32; 4], Option<ResolvedChunkPayload>>::new();
    for (chunk_pos, payload) in request.dirty_chunk_payloads {
        dirty_chunk_payloads.insert(chunk_pos, payload);
    }
    if dirty_chunk_payloads.is_empty() {
        return Ok(None);
    }

    let dirty_chunks: HashSet<[i32; 4]> = dirty_chunk_payloads.keys().copied().collect();
    let node_by_id = build_node_lookup(&loaded_index);
    let mut preserved_world_leaves = Vec::<LeafDescriptor>::new();
    let mut reencode_chunk_payloads = Vec::<([i32; 4], ResolvedChunkPayload)>::new();
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    collect_world_leaf_descriptors_with_chunk_patch(
        root,
        &node_by_id,
        loaded_index.root_node_id,
        &dirty_chunks,
        &mut payload_cache,
        &mut preserved_world_leaves,
        &mut reencode_chunk_payloads,
    )?;

    let mut dirty_chunk_positions: Vec<[i32; 4]> = dirty_chunk_payloads.keys().copied().collect();
    dirty_chunk_positions.sort_unstable();
    for chunk_pos in dirty_chunk_positions {
        if let Some(Some(payload)) = dirty_chunk_payloads.get(&chunk_pos) {
            reencode_chunk_payloads.push((chunk_pos, payload.clone()));
        }
    }

    let mut world_leaves = preserved_world_leaves;
    world_leaves.extend(build_world_leaf_descriptors_from_payloads(
        root,
        &mut manifest,
        &mut reuse_index,
        reencode_chunk_payloads,
    )?);

    let mut world_temp = build_temp_tree_from_leaves(&world_leaves)
        .unwrap_or_else(|| make_empty_branch_root([0, 0, 0, 0], [0, 0, 0, 0]));
    canonicalize_temp_tree(&mut world_temp);

    let mut entity_temp = match loaded_index.entity_root_node_id {
        Some(entity_root) => Some(build_temp_tree_from_index_subtree(
            &node_by_id,
            entity_root,
        )?),
        None => None,
    };
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

    let next_index = IndexPayload {
        generation: next_generation,
        root_node_id,
        entity_root_node_id,
        nodes,
    };
    validate_index_payload(&next_index)?;
    write_index_file(root.join(&next_index_file), &next_index)?;

    let next_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(request.base_world_kind),
        world_seed: request.world_seed,
        procgen_manifest_hash: loaded_global.procgen_manifest_hash,
        next_entity_id: request.next_entity_id.max(1),
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: request.now_ms,
        player_entity_hints: request
            .player_entity_hints
            .unwrap_or(loaded_global.player_entity_hints),
        custom_global_payload: request
            .custom_global_payload
            .unwrap_or(loaded_global.custom_global_payload),
    };
    write_payload_file(
        root.join(&next_global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_global,
    )?;

    manifest.current_generation = next_generation;
    manifest.last_modified_ms = request.now_ms;
    manifest.index_file = next_index_file;
    manifest.global_file = next_global_file;
    save_manifest_atomic(root, &manifest)?;

    Ok(Some(SaveResult {
        generation: next_generation,
        saved_block_regions: world_leaves.len(),
        saved_entity_regions: 0,
    }))
}

fn save_state_internal(
    root: &Path,
    base_world_kind: BaseWorldKind,
    chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
    common: SaveStateCommon<'_>,
) -> io::Result<SaveResult> {
    let loaded =
        load_or_init_save_metadata(root, base_world_kind, common.world_seed, common.now_ms)?;
    let LoadedSaveMetadata {
        mut manifest,
        global: loaded_global,
        index: loaded_index,
    } = loaded;
    let mut reuse_index = build_blob_reuse_index(root, &manifest)?;

    let effective_chunk_payloads = if common.disable_block_persistence {
        Vec::new()
    } else {
        let persisted_chunk_payloads = if common.force_full_blocks {
            Vec::new()
        } else if common.dirty_block_regions.is_empty() {
            materialize_world_chunk_payloads_from_index_filtered(
                root,
                &loaded_index,
                None,
                true,
                None,
            )?
        } else {
            materialize_world_chunk_payloads_from_index_filtered(
                root,
                &loaded_index,
                Some(common.dirty_block_regions),
                false,
                None,
            )?
        };
        resolve_world_chunk_payloads_for_save(
            persisted_chunk_payloads,
            chunk_payloads,
            common.dirty_block_regions,
            common.force_full_blocks,
        )
    };

    let world_leaves = if common.disable_block_persistence {
        Vec::new()
    } else {
        build_world_leaf_descriptors_from_payloads(
            root,
            &mut manifest,
            &mut reuse_index,
            effective_chunk_payloads,
        )?
    };

    let persisted_entities = if common.force_full_entities {
        Vec::new()
    } else if common.dirty_entity_regions.is_empty() {
        materialize_entities_from_index_filtered(root, &loaded_index, None, true)?
    } else {
        materialize_entities_from_index_filtered(
            root,
            &loaded_index,
            Some(common.dirty_entity_regions),
            false,
        )?
    };
    let effective_entities = resolve_entities_for_save(
        persisted_entities,
        common.entities,
        common.dirty_entity_regions,
        common.force_full_entities,
    );

    let mut entities_by_chunk = HashMap::<[i32; 4], Vec<PersistedEntityRecord>>::new();
    for entity in effective_entities {
        let chunk = chunk_from_world_position(entity.position);
        entities_by_chunk.entry(chunk).or_default().push(entity);
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
        let entity_ref = append_or_reuse_blob_record(
            root,
            &mut manifest,
            &mut reuse_index,
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
        base_world_kind: PersistedBaseWorldKind::from_runtime(base_world_kind),
        world_seed: common.world_seed,
        procgen_manifest_hash: loaded_global.procgen_manifest_hash,
        next_entity_id: common.next_entity_id,
        next_data_file_id: manifest.active_data_file_id.saturating_add(1),
        last_modified_ms: common.now_ms,
        player_entity_hints: common
            .player_entity_hints
            .unwrap_or(loaded_global.player_entity_hints),
        custom_global_payload: common
            .custom_global_payload
            .unwrap_or(loaded_global.custom_global_payload),
    };
    write_payload_file(
        root.join(&next_global_file),
        GLOBAL_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_global,
    )?;

    let next_players = PlayersPayload {
        players: common.players.to_vec(),
    };
    write_payload_file(
        root.join(&next_players_file),
        PLAYERS_MAGIC,
        PAYLOAD_FILE_VERSION,
        &next_players,
    )?;

    manifest.current_generation = next_generation;
    manifest.last_modified_ms = common.now_ms;
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

fn resolve_world_chunk_payloads_for_save(
    persisted_chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
    chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
    dirty_block_regions: &HashSet<[i32; 4]>,
    force_full_blocks: bool,
) -> Vec<([i32; 4], ResolvedChunkPayload)> {
    let normalized_current = normalize_chunk_payloads_latest_wins(chunk_payloads);
    if force_full_blocks {
        return normalized_current;
    }

    let mut merged = HashMap::<[i32; 4], ResolvedChunkPayload>::new();
    for (chunk_pos, payload) in persisted_chunk_payloads {
        merged.insert(chunk_pos, payload);
    }
    if dirty_block_regions.is_empty() {
        let mut out: Vec<([i32; 4], ResolvedChunkPayload)> = merged.into_iter().collect();
        out.sort_unstable_by_key(|(pos, _)| *pos);
        return out;
    }

    merged.retain(|chunk_pos, _| {
        !dirty_block_regions.contains(&region_from_chunk(*chunk_pos, DEFAULT_REGION_CHUNK_EDGE))
    });
    for (chunk_pos, payload) in normalized_current {
        if dirty_block_regions.contains(&region_from_chunk(chunk_pos, DEFAULT_REGION_CHUNK_EDGE)) {
            merged.insert(chunk_pos, payload);
        }
    }

    let mut out: Vec<([i32; 4], ResolvedChunkPayload)> = merged.into_iter().collect();
    out.sort_unstable_by_key(|(pos, _)| *pos);
    out
}

fn normalize_chunk_payloads_latest_wins(
    chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
) -> Vec<([i32; 4], ResolvedChunkPayload)> {
    let mut by_pos = HashMap::<[i32; 4], (u64, ResolvedChunkPayload)>::new();
    for (idx, (chunk_pos, payload)) in chunk_payloads.into_iter().enumerate() {
        let epoch = idx as u64;
        match by_pos.entry(chunk_pos) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((epoch, payload));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                // For ResolvedChunkPayload, last write wins (higher epoch).
                let (current_epoch, _) = entry.get();
                if epoch >= *current_epoch {
                    entry.insert((epoch, payload));
                }
            }
        }
    }
    let mut out: Vec<([i32; 4], ResolvedChunkPayload)> = by_pos
        .into_iter()
        .map(|(chunk_pos, (_, payload))| (chunk_pos, payload))
        .collect();
    out.sort_unstable_by_key(|(pos, _)| *pos);
    out
}

fn resolve_entities_for_save(
    loaded_entities: Vec<PersistedEntityRecord>,
    current_entities: &[PersistedEntityRecord],
    dirty_entity_regions: &HashSet<[i32; 4]>,
    force_full_entities: bool,
) -> Vec<PersistedEntityRecord> {
    if force_full_entities {
        let mut out = current_entities.to_vec();
        out.sort_unstable_by_key(|entity| entity.entity_id);
        return out;
    }

    if dirty_entity_regions.is_empty() {
        let mut out = loaded_entities;
        out.sort_unstable_by_key(|entity| entity.entity_id);
        return out;
    }

    let mut merged = HashMap::<u64, PersistedEntityRecord>::new();
    for entity in loaded_entities {
        let chunk = chunk_from_world_position(entity.position);
        let region = region_from_chunk(chunk, DEFAULT_REGION_CHUNK_EDGE);
        if !dirty_entity_regions.contains(&region) {
            merged.insert(entity.entity_id, entity);
        }
    }
    for entity in current_entities {
        let chunk = chunk_from_world_position(entity.position);
        let region = region_from_chunk(chunk, DEFAULT_REGION_CHUNK_EDGE);
        if dirty_entity_regions.contains(&region) {
            merged.insert(entity.entity_id, entity.clone());
        }
    }

    let mut out: Vec<PersistedEntityRecord> = merged.into_values().collect();
    out.sort_unstable_by_key(|entity| entity.entity_id);
    out
}

fn build_world_leaf_descriptors_from_payloads(
    root: &Path,
    manifest: &mut Manifest,
    reuse_index: &mut BlobReuseIndex,
    mut chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)>,
) -> io::Result<Vec<LeafDescriptor>> {
    let mut world_leaves = Vec::<LeafDescriptor>::new();
    chunk_payloads.sort_unstable_by_key(|(pos, _)| *pos);
    let mut non_uniform_rows =
        HashMap::<(i32, i32, i32), Vec<(i32, ResolvedChunkPayload)>>::new();
    for (chunk_pos, resolved) in chunk_payloads {
        let min = chunk_pos;
        let max = chunk_pos;
        if let Some(block) = resolved.uniform_block() {
            if block.is_air() {
                world_leaves.push(LeafDescriptor {
                    min,
                    max,
                    kind: IndexNodeKind::LeafEmpty,
                });
            } else {
                world_leaves.push(LeafDescriptor {
                    min,
                    max,
                    kind: IndexNodeKind::LeafUniform {
                        block: block.clone(),
                    },
                });
            }
        } else {
            non_uniform_rows
                .entry((chunk_pos[1], chunk_pos[2], chunk_pos[3]))
                .or_default()
                .push((chunk_pos[0], resolved));
        }
    }

    let mut row_keys: Vec<(i32, i32, i32)> = non_uniform_rows.keys().copied().collect();
    row_keys.sort_unstable();
    for (row_y, row_z, row_w) in row_keys {
        let Some(mut row_payloads) = non_uniform_rows.remove(&(row_y, row_z, row_w)) else {
            continue;
        };
        row_payloads.sort_unstable_by_key(|(x, _)| *x);

        let mut run_start = 0usize;
        while run_start < row_payloads.len() {
            let mut run_end = run_start + 1;
            while run_end < row_payloads.len()
                && row_payloads[run_end].0 == row_payloads[run_end - 1].0 + 1
            {
                run_end += 1;
            }
            let run = &row_payloads[run_start..run_end];
            let min = [run[0].0, row_y, row_z, row_w];
            let max = [run[run.len() - 1].0, row_y, row_z, row_w];

            // Build a unified block_palette across all payloads in this run,
            // remapping each payload's internal u16 indices.
            let mut unified_block_palette = vec![BlockData::AIR];
            let mut block_to_unified = HashMap::<BlockData, u16>::new();
            block_to_unified.insert(BlockData::AIR, 0);

            let mut remapped_payloads = Vec::<FieldChunkPayload>::with_capacity(run.len());
            for (_, resolved) in run {
                let Ok(dense_indices) = resolved.payload.dense_materials() else {
                    remapped_payloads.push(FieldChunkPayload::Empty);
                    continue;
                };
                let mut remapped = Vec::<u16>::with_capacity(dense_indices.len());
                for idx in &dense_indices {
                    let block = resolved
                        .block_palette
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or(BlockData::AIR);
                    let unified_idx = match block_to_unified.get(&block) {
                        Some(&idx) => idx,
                        None => {
                            let new_idx = unified_block_palette.len() as u16;
                            unified_block_palette.push(block.clone());
                            block_to_unified.insert(block, new_idx);
                            new_idx
                        }
                    };
                    remapped.push(unified_idx);
                }
                let payload = FieldChunkPayload::from_dense_materials_compact(&remapped)
                    .unwrap_or(FieldChunkPayload::Dense16 {
                        materials: remapped,
                    });
                remapped_payloads.push(payload);
            }

            let mut palette = Vec::<FieldChunkPayload>::new();
            let mut palette_lookup = HashMap::<FieldChunkPayload, u16>::new();
            let mut dense_indices = Vec::<u16>::with_capacity(remapped_payloads.len());
            for payload in &remapped_payloads {
                let palette_idx = if let Some(idx) = palette_lookup.get(payload).copied() {
                    idx
                } else {
                    let idx = u16::try_from(palette.len()).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "chunk array payload palette exceeds DenseU16 capacity",
                        )
                    })?;
                    palette.push(payload.clone());
                    palette_lookup.insert(payload.clone(), idx);
                    idx
                };
                dense_indices.push(palette_idx);
            }

            let mut payload_palette = Vec::<BlobRef>::with_capacity(palette.len());
            for payload in palette {
                let payload_blob = ChunkPayloadBlob { payload };
                let payload_blob_bytes = postcard::to_stdvec(&payload_blob)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                let payload_ref = append_or_reuse_blob_record(
                    root,
                    manifest,
                    reuse_index,
                    BLOB_KIND_CHUNK_PAYLOAD,
                    CHUNK_PAYLOAD_BLOB_VERSION,
                    &payload_blob_bytes,
                )?;
                payload_palette.push(payload_ref);
            }

            let mut index_data = Vec::with_capacity(dense_indices.len() * 2);
            for idx in dense_indices {
                index_data.extend_from_slice(&idx.to_le_bytes());
            }
            let chunk_array_blob = ChunkArrayBlob {
                volume_min_chunk: min,
                volume_max_chunk: max,
                payload_palette,
                block_palette: unified_block_palette,
                index_codec: ChunkArrayIndexCodec::DenseU16,
                index_data,
                default_palette_index: None,
            };
            let chunk_array_blob_bytes = postcard::to_stdvec(&chunk_array_blob)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let chunk_array_ref = append_or_reuse_blob_record(
                root,
                manifest,
                reuse_index,
                BLOB_KIND_CHUNK_ARRAY,
                CHUNK_ARRAY_BLOB_VERSION,
                &chunk_array_blob_bytes,
            )?;

            world_leaves.push(LeafDescriptor {
                min,
                max,
                kind: IndexNodeKind::LeafChunkArray { chunk_array_ref },
            });
            run_start = run_end;
        }
    }
    Ok(world_leaves)
}

#[cfg(test)]
fn chunk_payloads_to_voxel_world(
    base_world_kind: BaseWorldKind,
    chunk_payloads: &[([i32; 4], ResolvedChunkPayload)],
) -> io::Result<RegionChunkWorld> {
    use crate::materials::block_to_material_appearance;
    let mut world = RegionChunkWorld::new_with_base(base_world_kind);
    for (chunk_pos, resolved) in chunk_payloads {
        let legacy = resolved.to_legacy_payload(|block| {
            block_to_material_appearance(block.namespace, block.block_type)
        });
        let chunk = legacy_chunk_from_field_chunk_payload(&legacy)?;
        world.insert_chunk(
            ChunkPos::new(chunk_pos[0], chunk_pos[1], chunk_pos[2], chunk_pos[3]),
            chunk,
        );
    }
    world.clear_dirty();
    let _ = world.drain_pending_chunk_updates();
    Ok(world)
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

fn include_region_for_filter(
    region: [i32; 4],
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
) -> bool {
    match region_filter {
        Some(filter) => filter.contains(&region) == include_matches,
        None => true,
    }
}

fn include_chunk_for_filter(
    chunk_pos: [i32; 4],
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
) -> bool {
    include_region_for_filter(
        region_from_chunk(chunk_pos, DEFAULT_REGION_CHUNK_EDGE),
        region_filter,
        include_matches,
    )
}

fn load_chunk_payload_from_ref(
    root: &Path,
    blob_ref: &BlobRef,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
) -> io::Result<FieldChunkPayload> {
    if blob_ref.blob_type != BLOB_KIND_CHUNK_PAYLOAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chunk payload ref has invalid blob type {}, expected {}",
                blob_ref.blob_type, BLOB_KIND_CHUNK_PAYLOAD
            ),
        ));
    }

    let key = blob_ref_identity(blob_ref);
    if let Some(payload) = payload_cache.get(&key) {
        return Ok(payload.clone());
    }

    let payload = read_blob_payload(root, blob_ref)?;
    let payload_blob: ChunkPayloadBlob = postcard::from_bytes(&payload)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    payload_cache.insert(key, payload_blob.payload.clone());
    Ok(payload_blob.payload)
}

fn collect_world_chunk_payloads_from_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
    bounds_filter: Option<Aabb4i>,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    out: &mut Vec<([i32; 4], ResolvedChunkPayload)>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };

    if let Some(filter_bounds) = bounds_filter {
        if !Aabb4i::new(node.bounds_min_chunk, node.bounds_max_chunk).intersects(&filter_bounds) {
            return Ok(());
        }
    }

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_world_chunk_payloads_from_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    region_filter,
                    include_matches,
                    bounds_filter,
                    payload_cache,
                    out,
                )?;
            }
        }
        IndexNodeKind::LeafEmpty => {
            for_each_chunk_in_bounds(node.bounds_min_chunk, node.bounds_max_chunk, |chunk_pos| {
                let chunk = [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w];
                let in_bounds = bounds_filter
                    .map(|filter| filter.contains_chunk(chunk))
                    .unwrap_or(true);
                if in_bounds && include_chunk_for_filter(chunk, region_filter, include_matches) {
                    out.push((chunk, ResolvedChunkPayload::empty()));
                }
            });
        }
        IndexNodeKind::LeafUniform { block } => {
            for_each_chunk_in_bounds(node.bounds_min_chunk, node.bounds_max_chunk, |chunk_pos| {
                let chunk = [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w];
                let in_bounds = bounds_filter
                    .map(|filter| filter.contains_chunk(chunk))
                    .unwrap_or(true);
                if in_bounds && include_chunk_for_filter(chunk, region_filter, include_matches) {
                    out.push((chunk, ResolvedChunkPayload::uniform(block.clone())));
                }
            });
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_CHUNK_ARRAY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "world subtree leaf references non-chunk-array blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob: ChunkArrayBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let bounds = Aabb4i::new(blob.volume_min_chunk, blob.volume_max_chunk);
            if !bounds.is_valid() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array bounds",
                ));
            }

            let palette_len = blob.payload_palette.len().max(1);
            let dummy_palette = vec![FieldChunkPayload::Empty; palette_len];
            let block_palette = blob.block_palette.clone();
            let chunk_array_data = ChunkArrayData {
                bounds,
                chunk_palette: dummy_palette,
                index_codec: blob.index_codec,
                index_data: blob.index_data.clone(),
                default_chunk_idx: blob.default_palette_index,
                block_palette: block_palette.clone(),
            };
            let indices = chunk_array_data
                .decode_dense_indices()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let extents = bounds.chunk_extents().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array extents during decode",
                )
            })?;

            let mut palette = Vec::<FieldChunkPayload>::with_capacity(blob.payload_palette.len());
            for payload_ref in &blob.payload_palette {
                palette.push(load_chunk_payload_from_ref(
                    root,
                    payload_ref,
                    payload_cache,
                )?);
            }

            for w in bounds.min[3]..=bounds.max[3] {
                for z in bounds.min[2]..=bounds.max[2] {
                    for y in bounds.min[1]..=bounds.max[1] {
                        for x in bounds.min[0]..=bounds.max[0] {
                            let chunk = [x, y, z, w];
                            if bounds_filter
                                .map(|filter| !filter.contains_chunk(chunk))
                                .unwrap_or(false)
                            {
                                continue;
                            }
                            if !include_chunk_for_filter(chunk, region_filter, include_matches) {
                                continue;
                            }
                            let local = [
                                (x - bounds.min[0]) as usize,
                                (y - bounds.min[1]) as usize,
                                (z - bounds.min[2]) as usize,
                                (w - bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let palette_idx = indices.get(linear).copied().ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "chunk array decoded index out of bounds",
                                )
                            })?;
                            let payload =
                                palette.get(palette_idx as usize).cloned().ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "chunk array palette index out of bounds",
                                    )
                                })?;
                            out.push((
                                chunk,
                                ResolvedChunkPayload {
                                    payload,
                                    block_palette: block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn materialize_world_chunk_payloads_from_index_filtered(
    root: &Path,
    index: &IndexPayload,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
    bounds_filter: Option<Aabb4i>,
) -> io::Result<Vec<([i32; 4], ResolvedChunkPayload)>> {
    let node_by_id = build_node_lookup(index);
    let mut out = Vec::<([i32; 4], ResolvedChunkPayload)>::new();
    let mut payload_cache = HashMap::<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>::new();
    collect_world_chunk_payloads_from_node_filtered(
        root,
        &node_by_id,
        index.root_node_id,
        region_filter,
        include_matches,
        bounds_filter,
        &mut payload_cache,
        &mut out,
    )?;
    out.sort_unstable_by_key(|(pos, _)| *pos);
    Ok(out)
}

fn collect_entities_from_node_filtered(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
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
                collect_entities_from_node_filtered(
                    root,
                    node_by_id,
                    *child_id,
                    region_filter,
                    include_matches,
                    entities_by_id,
                )?;
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
                let region = region_from_chunk(
                    chunk_from_world_position(entity.position),
                    DEFAULT_REGION_CHUNK_EDGE,
                );
                if include_region_for_filter(region, region_filter, include_matches) {
                    entities_by_id.insert(entity.entity_id, entity);
                }
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {}
    }

    Ok(())
}

fn materialize_entities_from_index_filtered(
    root: &Path,
    index: &IndexPayload,
    region_filter: Option<&HashSet<[i32; 4]>>,
    include_matches: bool,
) -> io::Result<Vec<PersistedEntityRecord>> {
    let Some(entity_root) = index.entity_root_node_id else {
        return Ok(Vec::new());
    };
    let node_by_id = build_node_lookup(index);
    let mut entities_by_id = HashMap::<u64, PersistedEntityRecord>::new();
    collect_entities_from_node_filtered(
        root,
        &node_by_id,
        entity_root,
        region_filter,
        include_matches,
        &mut entities_by_id,
    )?;

    let mut entities: Vec<PersistedEntityRecord> = entities_by_id.into_values().collect();
    entities.sort_unstable_by_key(|entity| entity.entity_id);
    Ok(entities)
}

fn subtract_aabb(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    if !outer.intersects(&inner) {
        return vec![outer];
    }
    let intersection = Aabb4i::new(
        [
            outer.min[0].max(inner.min[0]),
            outer.min[1].max(inner.min[1]),
            outer.min[2].max(inner.min[2]),
            outer.min[3].max(inner.min[3]),
        ],
        [
            outer.max[0].min(inner.max[0]),
            outer.max[1].min(inner.max[1]),
            outer.max[2].min(inner.max[2]),
            outer.max[3].min(inner.max[3]),
        ],
    );
    if !intersection.is_valid() {
        return vec![outer];
    }
    if intersection == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;
    for axis in 0..4 {
        if core.min[axis] < intersection.min[axis] {
            let mut piece = core;
            piece.max[axis] = intersection.min[axis] - 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = intersection.min[axis];
        }
        if core.max[axis] > intersection.max[axis] {
            let mut piece = core;
            piece.min[axis] = intersection.max[axis] + 1;
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = intersection.max[axis];
        }
    }
    pieces
}

fn carve_bounds_excluding_dirty_chunks(
    bounds: Aabb4i,
    dirty_chunks: &HashSet<[i32; 4]>,
) -> Vec<Aabb4i> {
    let mut dirty_in_bounds: Vec<[i32; 4]> = dirty_chunks
        .iter()
        .copied()
        .filter(|chunk| bounds.contains_chunk(*chunk))
        .collect();
    if dirty_in_bounds.is_empty() {
        return vec![bounds];
    }
    dirty_in_bounds.sort_unstable();

    let mut pieces = vec![bounds];
    for chunk in dirty_in_bounds {
        let chunk_bounds = Aabb4i::new(chunk, chunk);
        let mut next = Vec::new();
        for piece in pieces {
            next.extend(subtract_aabb(piece, chunk_bounds));
        }
        pieces = next;
        if pieces.is_empty() {
            break;
        }
    }
    pieces
}

fn collect_world_leaf_descriptors_with_chunk_patch(
    root: &Path,
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
    dirty_chunks: &HashSet<[i32; 4]>,
    payload_cache: &mut HashMap<(u32, u64, u32, u32, u8, u16), FieldChunkPayload>,
    preserved_out: &mut Vec<LeafDescriptor>,
    reencode_payloads_out: &mut Vec<([i32; 4], ResolvedChunkPayload)>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing world node id {node_id}"),
        ));
    };
    let bounds = Aabb4i::new(node.bounds_min_chunk, node.bounds_max_chunk);
    if !bounds.is_valid() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid world node bounds while collecting patch leaves",
        ));
    }

    let has_dirty = dirty_chunks
        .iter()
        .any(|chunk| bounds.contains_chunk(*chunk));
    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_world_leaf_descriptors_with_chunk_patch(
                    root,
                    node_by_id,
                    *child_id,
                    dirty_chunks,
                    payload_cache,
                    preserved_out,
                    reencode_payloads_out,
                )?;
            }
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } if !has_dirty => {
            preserved_out.push(LeafDescriptor {
                min: node.bounds_min_chunk,
                max: node.bounds_max_chunk,
                kind: node.kind.clone(),
            });
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {
            let pieces = carve_bounds_excluding_dirty_chunks(bounds, dirty_chunks);
            for piece in pieces {
                preserved_out.push(LeafDescriptor {
                    min: piece.min,
                    max: piece.max,
                    kind: node.kind.clone(),
                });
            }
        }
        IndexNodeKind::LeafChunkArray { .. } if !has_dirty => {
            preserved_out.push(LeafDescriptor {
                min: node.bounds_min_chunk,
                max: node.bounds_max_chunk,
                kind: node.kind.clone(),
            });
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            if chunk_array_ref.blob_type != BLOB_KIND_CHUNK_ARRAY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "world subtree leaf references non-chunk-array blob type {}",
                        chunk_array_ref.blob_type
                    ),
                ));
            }
            let payload = read_blob_payload(root, chunk_array_ref)?;
            let blob: ChunkArrayBlob = postcard::from_bytes(&payload)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let chunk_array_bounds = Aabb4i::new(blob.volume_min_chunk, blob.volume_max_chunk);
            if !chunk_array_bounds.is_valid() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array bounds while collecting patch leaves",
                ));
            }

            let palette_len = blob.payload_palette.len().max(1);
            let block_palette = blob.block_palette.clone();
            let chunk_array_data = ChunkArrayData {
                bounds: chunk_array_bounds,
                chunk_palette: vec![FieldChunkPayload::Empty; palette_len],
                index_codec: blob.index_codec,
                index_data: blob.index_data.clone(),
                default_chunk_idx: blob.default_palette_index,
                block_palette: block_palette.clone(),
            };
            let indices = chunk_array_data
                .decode_dense_indices()
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
            let extents = chunk_array_bounds.chunk_extents().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid chunk array extents during patch decode",
                )
            })?;

            let mut palette = Vec::<FieldChunkPayload>::with_capacity(blob.payload_palette.len());
            for payload_ref in &blob.payload_palette {
                palette.push(load_chunk_payload_from_ref(
                    root,
                    payload_ref,
                    payload_cache,
                )?);
            }

            for w in chunk_array_bounds.min[3]..=chunk_array_bounds.max[3] {
                for z in chunk_array_bounds.min[2]..=chunk_array_bounds.max[2] {
                    for y in chunk_array_bounds.min[1]..=chunk_array_bounds.max[1] {
                        for x in chunk_array_bounds.min[0]..=chunk_array_bounds.max[0] {
                            let chunk = [x, y, z, w];
                            if dirty_chunks.contains(&chunk) {
                                continue;
                            }
                            let local = [
                                (x - chunk_array_bounds.min[0]) as usize,
                                (y - chunk_array_bounds.min[1]) as usize,
                                (z - chunk_array_bounds.min[2]) as usize,
                                (w - chunk_array_bounds.min[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let palette_idx = indices.get(linear).copied().ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "chunk array decoded index out of bounds during patch decode",
                                )
                            })?;
                            let payload =
                                palette.get(palette_idx as usize).cloned().ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "chunk array palette index out of bounds during patch decode",
                                    )
                                })?;
                            reencode_payloads_out.push((
                                chunk,
                                ResolvedChunkPayload {
                                    payload,
                                    block_palette: block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn build_node_lookup(index: &IndexPayload) -> HashMap<u32, &IndexNode> {
    index
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect()
}

fn build_temp_tree_from_index_subtree(
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
) -> io::Result<TempNode> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing node id {node_id}"),
        ));
    };
    let kind = match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            let mut children = Vec::with_capacity(child_node_ids.len());
            for child_id in child_node_ids {
                children.push(build_temp_tree_from_index_subtree(node_by_id, *child_id)?);
            }
            TempNodeKind::Branch(children)
        }
        leaf_kind => TempNodeKind::Leaf(leaf_kind.clone()),
    };
    Ok(TempNode {
        min: node.bounds_min_chunk,
        max: node.bounds_max_chunk,
        kind,
    })
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

    // Save-v4 index validation requires direct branch siblings to have non-overlapping
    // hypervolumes. A BVH split can create overlapping sibling bounds even when leaf
    // volumes themselves are disjoint, so emit direct leaf children here.
    let mut children = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        children.push(TempNode {
            min: leaf.min,
            max: leaf.max,
            kind: TempNodeKind::Leaf(leaf.kind.clone()),
        });
    }
    children.sort_unstable_by_key(temp_node_order_key);

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
        IndexNodeKind::LeafUniform { block } => {
            use std::hash::{Hash, Hasher as StdHasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            block.hash(&mut hasher);
            let h = hasher.finish();
            (2, h, 0, 0, 0, 0)
        }
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

fn blob_ref_identity(blob_ref: &BlobRef) -> (u32, u64, u32, u32, u8, u16) {
    (
        blob_ref.data_file_id,
        blob_ref.record_offset,
        blob_ref.record_len,
        blob_ref.record_crc32,
        blob_ref.blob_type,
        blob_ref.blob_version,
    )
}

fn blob_ref_to_reuse_key(blob_ref: &BlobRef) -> io::Result<BlobReuseKey> {
    if blob_ref.record_len < RECORD_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "blob record_len {} too small for header {}",
                blob_ref.record_len, RECORD_HEADER_LEN
            ),
        ));
    }
    Ok(BlobReuseKey {
        blob_type: blob_ref.blob_type,
        blob_version: blob_ref.blob_version,
        payload_len: blob_ref.record_len.saturating_sub(RECORD_HEADER_LEN),
        payload_crc32: blob_ref.record_crc32,
    })
}

fn payload_to_reuse_key(
    blob_type: u8,
    blob_version: u16,
    payload: &[u8],
) -> io::Result<BlobReuseKey> {
    let payload_len = u32::try_from(payload.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("payload too large for blob record: {} bytes", payload.len()),
        )
    })?;
    Ok(BlobReuseKey {
        blob_type,
        blob_version,
        payload_len,
        payload_crc32: crc32(payload),
    })
}

#[cfg(test)]
fn collect_leaf_chunk_array_refs_recursive(
    node_id: u32,
    node_by_id: &HashMap<u32, &IndexNode>,
    out: &mut Vec<BlobRef>,
) -> io::Result<()> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing node {} while collecting blob refs", node_id),
        ));
    };

    match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            for child_id in child_node_ids {
                collect_leaf_chunk_array_refs_recursive(*child_id, node_by_id, out)?;
            }
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => {
            out.push(chunk_array_ref.clone());
        }
        IndexNodeKind::LeafEmpty | IndexNodeKind::LeafUniform { .. } => {}
    }
    Ok(())
}

fn build_blob_reuse_index(root: &Path, manifest: &Manifest) -> io::Result<BlobReuseIndex> {
    let mut out = BlobReuseIndex::default();
    let max_data_file_id = manifest.data_file_count.max(manifest.active_data_file_id);
    if max_data_file_id == 0 {
        return Ok(out);
    }

    for data_file_id in 1..=max_data_file_id {
        let path = data_file_path(root, data_file_id);
        if !path.exists() {
            continue;
        }

        let mut reader = BufReader::new(File::open(path)?);
        let mut header = [0u8; 12];
        reader.read_exact(&mut header)?;
        if &header[0..4] != DATA_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid data file magic while building reuse index",
            ));
        }
        let data_version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        if data_version != DATA_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported data file version {data_version}"),
            ));
        }
        let header_file_id = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if header_file_id != data_file_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "data file id mismatch while building reuse index: header={} path={}",
                    header_file_id, data_file_id
                ),
            ));
        }

        loop {
            let record_offset = reader.stream_position()?;
            let mut record_magic = [0u8; 4];
            match reader.read_exact(&mut record_magic) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(error),
            }
            if &record_magic != RECORD_MAGIC {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid record magic at offset {}", record_offset),
                ));
            }

            let mut record_rest = [0u8; 11];
            reader.read_exact(&mut record_rest)?;
            let blob_type = record_rest[0];
            let blob_version = u16::from_le_bytes([record_rest[1], record_rest[2]]);
            let payload_len = u32::from_le_bytes([
                record_rest[3],
                record_rest[4],
                record_rest[5],
                record_rest[6],
            ]);
            let payload_crc32 = u32::from_le_bytes([
                record_rest[7],
                record_rest[8],
                record_rest[9],
                record_rest[10],
            ]);
            reader.seek(SeekFrom::Current(i64::from(payload_len)))?;

            let record_len = RECORD_HEADER_LEN.checked_add(payload_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "blob record length overflow")
            })?;
            let blob_ref = BlobRef {
                data_file_id,
                record_offset,
                record_len,
                record_crc32: payload_crc32,
                blob_type,
                blob_version,
            };
            let key = blob_ref_to_reuse_key(&blob_ref)?;
            out.refs_by_key.entry(key).or_default().push(blob_ref);
        }
    }

    Ok(out)
}

fn append_or_reuse_blob_record(
    root: &Path,
    manifest: &mut Manifest,
    reuse_index: &mut BlobReuseIndex,
    blob_type: u8,
    blob_version: u16,
    payload: &[u8],
) -> io::Result<BlobRef> {
    let key = payload_to_reuse_key(blob_type, blob_version, payload)?;
    if let Some(candidates) = reuse_index.refs_by_key.get(&key) {
        for candidate in candidates {
            if let Ok(existing_payload) = read_blob_payload(root, candidate) {
                if existing_payload == payload {
                    return Ok(candidate.clone());
                }
            }
        }
    }

    let blob_ref = append_blob_record(root, manifest, blob_type, blob_version, payload)?;
    reuse_index
        .refs_by_key
        .entry(key)
        .or_default()
        .push(blob_ref.clone());
    Ok(blob_ref)
}

fn append_blob_record(
    root: &Path,
    manifest: &mut Manifest,
    blob_type: u8,
    blob_version: u16,
    payload: &[u8],
) -> io::Result<BlobRef> {
    let record_len = (RECORD_HEADER_LEN as usize + payload.len()) as u64;
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
    let data_version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if data_version != DATA_FILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported data file version {data_version}"),
        ));
    }
    let header_file_id = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
    if header_file_id != blob.data_file_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "data file id mismatch: header={} blob_ref={}",
                header_file_id, blob.data_file_id
            ),
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
    let expected_record_len = u32::try_from(payload_len)
        .ok()
        .and_then(|len| RECORD_HEADER_LEN.checked_add(len))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "blob record length overflow"))?;
    if expected_record_len != blob.record_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "blob record length mismatch: expected {} got {}",
                blob.record_len, expected_record_len
            ),
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

    fn collect_world_chunk_array_refs(index: &IndexPayload) -> Vec<BlobRef> {
        let node_by_id = build_node_lookup(index);
        let mut refs = Vec::new();
        collect_leaf_chunk_array_refs_recursive(index.root_node_id, &node_by_id, &mut refs)
            .expect("collect world refs");
        refs.sort_unstable_by_key(blob_ref_identity);
        refs
    }

    fn total_data_bytes(root: &Path) -> u64 {
        let data_root = root.join(DATA_DIR);
        let Ok(entries) = std::fs::read_dir(&data_root) else {
            return 0;
        };
        entries
            .flatten()
            .filter_map(|entry| entry.metadata().ok())
            .filter(|meta| meta.is_file())
            .map(|meta| meta.len())
            .sum()
    }

    fn test_entity(entity_id: u64, position: [f32; 4], payload: Vec<u8>) -> PersistedEntityRecord {
        PersistedEntityRecord {
            entity_id,
            class: EntityClass::Accent,
            kind: EntityKind::TestRotor,
            position,
            orientation: [0.0, 0.0, 1.0, 0.0],
            velocity: [0.0, 0.0, 0.0, 0.0],
            scale: 1.0,
            material: 7,
            display_name: None,
            tags: vec!["persist".to_string()],
            payload,
            last_saved_ms: now_unix_ms(),
        }
    }

    fn materialize_loaded_world(loaded: &LoadedState) -> RegionChunkWorld {
        chunk_payloads_to_voxel_world(
            loaded.global.base_world_kind.to_runtime(),
            &loaded.world_chunk_payloads,
        )
        .expect("materialize loaded world")
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

        let save_result = save_state_from_world(
            &root,
            SaveWorldRequest {
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
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), VoxelType(3));
        assert_eq!(loaded_world.get_voxel(40, 2, -5, 17), VoxelType(7));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_respects_dirty_block_regions_when_not_forced() {
        let root = test_root("dirty-block-regions");
        let now_ms = now_unix_ms();
        let chunk_size = CHUNK_SIZE as i32;
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(2));
        world.set_voxel(chunk_size * 10 + 1, 1, 1, 1, VoxelType(3));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 1,
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

        world.set_voxel(1, 1, 1, 1, VoxelType(4));
        world.set_voxel(chunk_size * 10 + 1, 1, 1, 1, VoxelType(9));

        let mut dirty_blocks = HashSet::new();
        dirty_blocks.insert(region_from_chunk([0, 0, 0, 0], DEFAULT_REGION_CHUNK_EDGE));
        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 1,
                next_entity_id: 1,
                dirty_block_regions: &dirty_blocks,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: false,
                force_full_entities: false,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("incremental save");

        let loaded = load_state(&root).expect("load state");
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), VoxelType(4));
        assert_eq!(
            loaded_world.get_voxel(chunk_size * 10 + 1, 1, 1, 1),
            VoxelType(3)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_respects_dirty_entity_regions_when_not_forced() {
        let root = test_root("dirty-entity-regions");
        let now_ms = now_unix_ms();
        let chunk_size = CHUNK_SIZE as f32;
        let empty_regions = HashSet::new();
        let world = RegionChunkWorld::new();

        let entity_a_chunk = [0, 0, 0, 0];
        let entity_b_chunk = [20, 0, 0, 0];
        let entity_a_pos = [1.0, 1.0, 1.0, 1.0];
        let entity_b_pos = [chunk_size * 20.0 + 1.0, 1.0, 1.0, 1.0];
        let entities_initial = vec![
            test_entity(1, entity_a_pos, vec![1]),
            test_entity(2, entity_b_pos, vec![2]),
        ];

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &entities_initial,
                players: &[],
                world_seed: 1,
                next_entity_id: 3,
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

        let entities_next = vec![
            test_entity(1, entity_a_pos, vec![9]),
            test_entity(2, entity_b_pos, vec![8]),
        ];
        let mut dirty_entities = HashSet::new();
        dirty_entities.insert(region_from_chunk(entity_a_chunk, DEFAULT_REGION_CHUNK_EDGE));
        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &entities_next,
                players: &[],
                world_seed: 1,
                next_entity_id: 3,
                dirty_block_regions: &empty_regions,
                dirty_entity_regions: &dirty_entities,
                force_full_blocks: false,
                force_full_entities: false,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("incremental save");

        let loaded = load_state(&root).expect("load state");
        let mut loaded_entities = loaded.entities;
        loaded_entities.sort_unstable_by_key(|entity| entity.entity_id);
        assert_eq!(loaded_entities.len(), 2);
        assert_eq!(loaded_entities[0].entity_id, 1);
        assert_eq!(loaded_entities[0].payload, vec![9]);
        assert_eq!(loaded_entities[1].entity_id, 2);
        assert_eq!(loaded_entities[1].payload, vec![2]);
        assert_eq!(
            region_from_chunk(
                chunk_from_world_position(loaded_entities[1].position),
                DEFAULT_REGION_CHUNK_EDGE,
            ),
            region_from_chunk(entity_b_chunk, DEFAULT_REGION_CHUNK_EDGE)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_region_filtered_world_and_entities() {
        let root = test_root("region-filtered-load");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let chunk_size_i32 = CHUNK_SIZE as i32;
        let chunk_size_f32 = CHUNK_SIZE as f32;
        let chunk_a = [0, 0, 0, 0];
        let chunk_b = [DEFAULT_REGION_CHUNK_EDGE, 0, 0, 0];

        let mut world = RegionChunkWorld::new();
        world.set_voxel(chunk_a[0] * chunk_size_i32 + 1, 1, 1, 1, VoxelType(2));
        world.set_voxel(chunk_b[0] * chunk_size_i32 + 1, 1, 1, 1, VoxelType(5));

        let entities = vec![
            test_entity(1, [1.0, 2.0, 3.0, 4.0], vec![1]),
            test_entity(
                2,
                [chunk_b[0] as f32 * chunk_size_f32 + 1.0, 2.0, 3.0, 4.0],
                vec![2],
            ),
        ];

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &entities,
                players: &[],
                world_seed: 1,
                next_entity_id: 3,
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

        let mut region_filter = HashSet::new();
        region_filter.insert(region_from_chunk(chunk_a, DEFAULT_REGION_CHUNK_EDGE));

        let loaded_chunk_payloads =
            load_world_chunk_payloads_for_regions(&root, &region_filter).expect("load chunks");
        assert_eq!(loaded_chunk_payloads.len(), 1);
        assert_eq!(loaded_chunk_payloads[0].0, chunk_a);
        let legacy_payload = loaded_chunk_payloads[0].1.to_legacy_payload(|block| {
            crate::materials::block_to_material_appearance(block.namespace, block.block_type)
        });
        let chunk = legacy_chunk_from_field_chunk_payload(&legacy_payload)
            .expect("decode chunk payload");
        assert_eq!(chunk.get(1, 1, 1, 1), VoxelType(2));

        let loaded_entities = load_entities_for_regions(&root, &region_filter).expect("load ents");
        assert_eq!(loaded_entities.len(), 1);
        assert_eq!(loaded_entities[0].entity_id, 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_bounds_filtered_world_chunks() {
        let root = test_root("bounds-filtered-load");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let chunk_size_i32 = CHUNK_SIZE as i32;
        let chunk_a = [0, 0, 0, 0];
        let chunk_b = [3, 0, 0, 0];

        let mut world = RegionChunkWorld::new();
        world.set_voxel(chunk_a[0] * chunk_size_i32 + 1, 1, 1, 1, VoxelType(2));
        world.set_voxel(chunk_b[0] * chunk_size_i32 + 1, 1, 1, 1, VoxelType(5));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 1,
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
        .expect("save state");

        let only_chunk_a = Aabb4i::new(
            [chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]],
            [chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]],
        );
        let loaded_a =
            load_world_chunk_payloads_for_bounds(&root, only_chunk_a).expect("load chunk a");
        assert_eq!(loaded_a.len(), 1);
        assert_eq!(loaded_a[0].0, chunk_a);

        let both_bounds = Aabb4i::new(
            [chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3]],
            [chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3]],
        );
        let loaded_both =
            load_world_chunk_payloads_for_bounds(&root, both_bounds).expect("load both chunks");
        assert_eq!(loaded_both.len(), 2);
        assert_eq!(loaded_both[0].0, chunk_a);
        assert_eq!(loaded_both[1].0, chunk_b);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_world_subtree_core_for_bounds_keeps_uniform_leaf() {
        let root = test_root("bounds-core-uniform");
        let index = IndexPayload {
            generation: 1,
            root_node_id: 7,
            entity_root_node_id: None,
            nodes: vec![IndexNode {
                node_id: 7,
                bounds_min_chunk: [-128, -8, -128, -128],
                bounds_max_chunk: [127, 8, 127, 127],
                kind: IndexNodeKind::LeafUniform { block: BlockData::simple(0, 11) },
            }],
        };
        let bounds = Aabb4i::new([-4, -2, -4, -4], [4, 2, 4, 4]);
        let core = load_world_subtree_core_for_bounds_from_index(&root, &index, bounds)
            .expect("load subtree core");
        assert_eq!(core.bounds, bounds);
        assert!(matches!(
            core.kind,
            RegionNodeKind::Uniform(ref block) if block.block_type == 11
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_from_chunk_payloads_duplicate_chunk_latest_wins() {
        let root = test_root("chunk-payload-latest-wins");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let chunk_pos = [2, 0, 0, 0];
        let chunk_size = CHUNK_SIZE as i32;

        let save_result = save_state_from_chunk_payloads(
            &root,
            SaveChunkPayloadRequest {
                base_world_kind: BaseWorldKind::Empty,
                chunk_payloads: vec![
                    (chunk_pos, ResolvedChunkPayload::uniform(BlockData::simple(0, 3))),
                    (chunk_pos, ResolvedChunkPayload::uniform(BlockData::simple(0, 9))),
                ],
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
        .expect("save state");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&root).expect("load state");
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(
            loaded_world.get_voxel(chunk_pos[0] * chunk_size + 1, 1, 1, 1),
            VoxelType(9)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_from_chunk_payloads_merges_dirty_region_without_full_snapshot() {
        let root = test_root("chunk-payload-dirty-region-merge");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let chunk_a = [0, 0, 0, 0];
        let chunk_b = [DEFAULT_REGION_CHUNK_EDGE, 0, 0, 0];
        let mut dirty_regions = HashSet::new();
        dirty_regions.insert(region_from_chunk(chunk_a, DEFAULT_REGION_CHUNK_EDGE));
        let chunk_size = CHUNK_SIZE as i32;

        save_state_from_chunk_payloads(
            &root,
            SaveChunkPayloadRequest {
                base_world_kind: BaseWorldKind::Empty,
                chunk_payloads: vec![
                    (chunk_a, ResolvedChunkPayload::uniform(BlockData::simple(0, 3))),
                    (chunk_b, ResolvedChunkPayload::uniform(BlockData::simple(0, 7))),
                ],
                entities: &[],
                players: &[],
                world_seed: 1,
                next_entity_id: 2,
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
        .expect("initial full save");

        save_state_from_chunk_payloads(
            &root,
            SaveChunkPayloadRequest {
                base_world_kind: BaseWorldKind::Empty,
                chunk_payloads: vec![(chunk_a, ResolvedChunkPayload::uniform(BlockData::simple(0, 9)))],
                entities: &[],
                players: &[],
                world_seed: 1,
                next_entity_id: 2,
                dirty_block_regions: &dirty_regions,
                dirty_entity_regions: &empty_regions,
                force_full_blocks: false,
                force_full_entities: false,
                player_entity_hints: None,
                custom_global_payload: None,
                disable_block_persistence: false,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("dirty region save");

        let loaded = load_state(&root).expect("load state");
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(
            loaded_world.get_voxel(chunk_a[0] * chunk_size + 1, 1, 1, 1),
            VoxelType(9)
        );
        assert_eq!(
            loaded_world.get_voxel(chunk_b[0] * chunk_size + 1, 1, 1, 1),
            VoxelType(7)
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_from_chunk_payloads_roundtrip_preserves_world() {
        let root = test_root("roundtrip-chunk-payloads");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(1, 1, 1, 1, VoxelType(3));
        world.set_voxel(40, 2, -5, 17, VoxelType(7));

        let chunk_payloads: Vec<([i32; 4], ResolvedChunkPayload)> = world
            .chunks
            .iter()
            .map(|(&chunk_pos, chunk)| {
                (
                    [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
                    ResolvedChunkPayload::from_legacy_payload(
                        field_chunk_payload_from_legacy_chunk(chunk),
                    ),
                )
            })
            .collect();

        let save_result = save_state_from_chunk_payloads(
            &root,
            SaveChunkPayloadRequest {
                base_world_kind: world.base_kind(),
                chunk_payloads,
                entities: &[],
                players: &[],
                world_seed: 4242,
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
        .expect("save state from chunk payloads");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&root).expect("load state");
        assert_eq!(loaded.global.world_seed, 4242);
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), VoxelType(3));
        assert_eq!(loaded_world.get_voxel(40, 2, -5, 17), VoxelType(7));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_from_chunk_payload_patch_persists_without_overlapping_children() {
        let root = test_root("chunk-payload-patch-no-overlap");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut dense_blocks = vec![BlockData::simple(0, 1); CHUNK_VOLUME];
        dense_blocks[0] = BlockData::simple(0, 2);
        let dense_resolved = ResolvedChunkPayload::from_dense_blocks(&dense_blocks)
            .expect("dense resolved");

        let mut initial_payloads = Vec::new();
        for x in 0..=3 {
            initial_payloads.push(([x, 0, 0, 0], dense_resolved.clone()));
            initial_payloads.push(([x, 1, 0, 0], dense_resolved.clone()));
        }

        let initial = save_state_from_chunk_payloads(
            &root,
            SaveChunkPayloadRequest {
                base_world_kind: BaseWorldKind::Empty,
                chunk_payloads: initial_payloads,
                entities: &[],
                players: &[],
                world_seed: 7,
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
        assert_eq!(initial.generation, 1);

        let patched = save_state_from_chunk_payload_patch(
            &root,
            SaveChunkPayloadPatchRequest {
                base_world_kind: BaseWorldKind::Empty,
                dirty_chunk_payloads: vec![([1, 0, 0, 0], Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 9))))],
                world_seed: 7,
                next_entity_id: 1,
                player_entity_hints: None,
                custom_global_payload: None,
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("patch save")
        .expect("patch should write a generation");
        assert_eq!(patched.generation, 2);

        let loaded = load_state(&root).expect("load patched state");
        let loaded_world = materialize_loaded_world(&loaded);
        let chunk_size = CHUNK_SIZE as i32;
        assert_eq!(
            loaded_world.get_voxel(1 * chunk_size, 0, 0, 0),
            VoxelType(9)
        );
        assert_eq!(loaded_world.get_voxel(0, chunk_size, 0, 0), VoxelType(2));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migrate_legacy_world_to_v4_roundtrip() {
        let root = test_root("migrate");
        let legacy_world = root.join("legacy.v4dw");
        let output_root = root.join("migrated-v4");

        let mut world = RegionChunkWorld::new();
        world.set_voxel(2, 2, 2, 2, VoxelType(9));
        {
            let mut writer = BufWriter::new(File::create(&legacy_world).expect("create legacy"));
            crate::migration::legacy_world_io::save_world(&world, &mut writer)
                .expect("save legacy");
            writer.flush().expect("flush legacy");
        }

        let save_result = crate::save_v4_migration::migrate_legacy_world_to_v4(
            &legacy_world,
            None,
            &output_root,
            777,
            now_unix_ms(),
        )
        .expect("migrate legacy to v4");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&output_root).expect("load migrated");
        assert_eq!(loaded.global.world_seed, 777);
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(2, 2, 2, 2), VoxelType(9));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn migrate_v3_save_to_v4_roundtrip() {
        let root = test_root("migrate-v3");
        let v3_root = root.join("input-v3");
        let output_root = root.join("output-v4");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(4, 1, 2, -3, VoxelType(8));

        let entities = vec![crate::migration::save_v3::PersistedEntityRecord {
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
        let players = vec![crate::migration::save_v3::PlayerRecord {
            player_id: 123,
            position: [0.0, 1.0, 2.0, 3.0],
            orientation: [0.0, 0.0, 1.0, 0.0],
            tags: vec!["p".to_string()],
            inventory_payload: vec![5, 4, 3],
            last_saved_ms: now_ms,
        }];
        let custom_payload = vec![42, 24, 7];

        crate::migration::save_v3::save_state(
            &v3_root,
            crate::migration::save_v3::SaveRequest {
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

        let save_result = crate::save_v4_migration::migrate_v3_save_to_v4(
            &v3_root,
            &output_root,
            false,
            now_ms.saturating_add(1),
        )
        .expect("migrate v3 -> v4");
        assert_eq!(save_result.generation, 1);

        let loaded = load_state(&output_root).expect("load v4 output");
        assert_eq!(loaded.manifest.version, SAVE_FORMAT_VERSION);
        assert_eq!(loaded.global.world_seed, 2026);
        assert_eq!(loaded.global.next_entity_id, 1000);
        assert_eq!(loaded.global.custom_global_payload, custom_payload);
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(4, 1, 2, -3), VoxelType(8));
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
    fn build_temp_tree_from_leaves_emits_non_overlapping_branch_children() {
        // These leaves are disjoint, but BVH-style grouping can produce overlapping
        // sibling union bounds. The temp-tree builder must avoid that pattern.
        let leaves = vec![
            LeafDescriptor {
                min: [0, 0, 0, 0],
                max: [100, 0, 0, 0],
                kind: IndexNodeKind::LeafUniform { block: BlockData::simple(0, 1) },
            },
            LeafDescriptor {
                min: [0, 2, 0, 0],
                max: [100, 2, 0, 0],
                kind: IndexNodeKind::LeafUniform { block: BlockData::simple(0, 1) },
            },
            LeafDescriptor {
                min: [50, 1, 0, 0],
                max: [150, 1, 0, 0],
                kind: IndexNodeKind::LeafUniform { block: BlockData::simple(0, 1) },
            },
            LeafDescriptor {
                min: [50, 3, 0, 0],
                max: [150, 3, 0, 0],
                kind: IndexNodeKind::LeafUniform { block: BlockData::simple(0, 1) },
            },
        ];

        let mut tree = build_temp_tree_from_leaves(&leaves).expect("temp tree");
        canonicalize_temp_tree(&mut tree);

        let mut nodes = Vec::<IndexNode>::new();
        let root_node_id = flatten_temp_tree(&tree, &mut nodes);
        let index = IndexPayload {
            generation: 1,
            root_node_id,
            entity_root_node_id: None,
            nodes,
        };

        validate_index_payload(&index).expect("leaf-based temp tree should validate");
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

        let mut world_a = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world_a.set_voxel(1, 2, 3, 4, VoxelType(5));
        world_a.set_voxel(-8, 0, 1, -3, VoxelType(9));
        world_a.set_voxel(33, 7, -2, 12, VoxelType(4));

        let mut world_b = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world_b.set_voxel(33, 7, -2, 12, VoxelType(4));
        world_b.set_voxel(1, 2, 3, 4, VoxelType(5));
        world_b.set_voxel(-8, 0, 1, -3, VoxelType(9));

        save_state_from_world(
            &root_a,
            SaveWorldRequest {
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
        save_state_from_world(
            &root_b,
            SaveWorldRequest {
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
    fn save_state_reuses_identical_chunk_payload_within_single_save() {
        let root = test_root("reuse-within-save");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(7));
        world.set_voxel(17, 1, 1, 1, VoxelType(7));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 1,
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
        .expect("save state");

        let manifest = load_manifest(&root).expect("load manifest");
        let index = read_index_file(root.join(&manifest.index_file)).expect("load index");
        let chunk_array_refs = collect_world_chunk_array_refs(&index);
        assert_eq!(chunk_array_refs.len(), 2);

        let mut payload_refs = Vec::<BlobRef>::new();
        for chunk_array_ref in chunk_array_refs {
            let payload =
                read_blob_payload(&root, &chunk_array_ref).expect("read chunk array blob");
            let chunk_array_blob: ChunkArrayBlob =
                postcard::from_bytes(&payload).expect("decode chunk array blob");
            assert_eq!(chunk_array_blob.payload_palette.len(), 1);
            payload_refs.push(chunk_array_blob.payload_palette[0].clone());
        }

        assert_eq!(payload_refs[0], payload_refs[1]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_packs_contiguous_non_uniform_row_into_single_chunk_array() {
        let root = test_root("pack-contiguous-row");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let chunk_size = CHUNK_SIZE as i32;

        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(7));
        world.set_voxel(chunk_size + 1, 1, 1, 1, VoxelType(9));
        world.set_voxel(chunk_size * 2 + 1, 1, 1, 1, VoxelType(7));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 1,
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
        .expect("save state");

        let manifest = load_manifest(&root).expect("load manifest");
        let index = read_index_file(root.join(&manifest.index_file)).expect("load index");
        let chunk_array_refs = collect_world_chunk_array_refs(&index);
        assert_eq!(chunk_array_refs.len(), 1);

        let payload =
            read_blob_payload(&root, &chunk_array_refs[0]).expect("read chunk array blob");
        let chunk_array_blob: ChunkArrayBlob =
            postcard::from_bytes(&payload).expect("decode chunk array blob");
        assert_eq!(chunk_array_blob.volume_min_chunk, [0, 0, 0, 0]);
        assert_eq!(chunk_array_blob.volume_max_chunk, [2, 0, 0, 0]);
        assert_eq!(chunk_array_blob.payload_palette.len(), 2);
        assert_eq!(chunk_array_blob.index_codec, ChunkArrayIndexCodec::DenseU16);

        let chunk_array_data = ChunkArrayData {
            bounds: Aabb4i::new(
                chunk_array_blob.volume_min_chunk,
                chunk_array_blob.volume_max_chunk,
            ),
            chunk_palette: vec![FieldChunkPayload::Empty; chunk_array_blob.payload_palette.len()],
            index_codec: chunk_array_blob.index_codec,
            index_data: chunk_array_blob.index_data,
            default_chunk_idx: chunk_array_blob.default_palette_index,
            block_palette: chunk_array_blob.block_palette,
        };
        let dense = chunk_array_data
            .decode_dense_indices()
            .expect("decode dense indices");
        assert_eq!(dense, vec![0, 1, 0]);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn save_state_identical_resave_does_not_append_data_blobs() {
        let root = test_root("no-thrash-resave");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(3, 2, 1, 0, VoxelType(5));
        world.set_voxel(-13, 4, -2, 7, VoxelType(9));

        save_state_from_world(
            &root,
            SaveWorldRequest {
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
        let manifest_a = load_manifest(&root).expect("manifest a");
        let index_a = read_index_file(root.join(&manifest_a.index_file)).expect("index a");
        let refs_a = collect_world_chunk_array_refs(&index_a);
        let data_bytes_a = total_data_bytes(&root);

        save_state_from_world(
            &root,
            SaveWorldRequest {
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
                now_ms: now_ms.saturating_add(1),
            },
        )
        .expect("second save");
        let manifest_b = load_manifest(&root).expect("manifest b");
        let index_b = read_index_file(root.join(&manifest_b.index_file)).expect("index b");
        let refs_b = collect_world_chunk_array_refs(&index_b);
        let data_bytes_b = total_data_bytes(&root);

        assert_eq!(refs_a, refs_b);
        assert_eq!(data_bytes_a, data_bytes_b);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_state_rejects_corrupt_index_payload_checksum() {
        let root = test_root("corrupt-index-checksum");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(7));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 7,
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
        .expect("save state");

        let manifest = load_manifest(&root).expect("manifest");
        let index_path = root.join(&manifest.index_file);
        let mut bytes = std::fs::read(&index_path).expect("read index");
        assert!(
            bytes.len() > 32,
            "index payload should contain at least one node"
        );
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&index_path, bytes).expect("rewrite tampered index");

        let err = load_state(&root).expect_err("corrupt index should fail load");
        assert!(
            err.to_string()
                .contains("index file payload checksum mismatch"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_state_rejects_corrupt_blob_payload_checksum() {
        let root = test_root("corrupt-blob-checksum");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(7));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 7,
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
        .expect("save state");

        let manifest = load_manifest(&root).expect("manifest");
        let index = read_index_file(root.join(&manifest.index_file)).expect("index");
        let chunk_array_refs = collect_world_chunk_array_refs(&index);
        assert!(!chunk_array_refs.is_empty(), "expected chunk-array refs");
        let target = &chunk_array_refs[0];

        let data_path = data_file_path(&root, target.data_file_id);
        let mut bytes = std::fs::read(&data_path).expect("read data file");
        let payload_offset = target
            .record_offset
            .saturating_add(RECORD_HEADER_LEN as u64) as usize;
        assert!(
            payload_offset < bytes.len(),
            "blob payload offset should be in range"
        );
        bytes[payload_offset] ^= 0x01;
        std::fs::write(&data_path, bytes).expect("rewrite tampered data file");

        let err = load_state(&root).expect_err("corrupt blob should fail load");
        assert!(
            err.to_string().contains("blob payload checksum mismatch"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_state_rejects_corrupt_data_file_header() {
        let root = test_root("corrupt-data-header");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();
        let mut world = RegionChunkWorld::new();
        world.set_voxel(1, 1, 1, 1, VoxelType(7));

        save_state_from_world(
            &root,
            SaveWorldRequest {
                world: &world,
                entities: &[],
                players: &[],
                world_seed: 7,
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
        .expect("save state");

        let manifest = load_manifest(&root).expect("manifest");
        let index = read_index_file(root.join(&manifest.index_file)).expect("index");
        let chunk_array_refs = collect_world_chunk_array_refs(&index);
        assert!(!chunk_array_refs.is_empty(), "expected chunk-array refs");
        let target = &chunk_array_refs[0];

        let data_path = data_file_path(&root, target.data_file_id);
        let mut bytes = std::fs::read(&data_path).expect("read data file");
        assert!(
            bytes.len() >= 8,
            "data header should contain version bytes at offset 4..8"
        );
        bytes[4] ^= 0x01;
        std::fs::write(&data_path, bytes).expect("rewrite tampered data file header");

        let err = load_state(&root).expect_err("corrupt data header should fail load");
        assert!(
            err.to_string().contains("unsupported data file version"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn disable_block_persistence_clears_block_overrides_and_keeps_custom_payload() {
        let root = test_root("disable-block-persistence");
        let now_ms = now_unix_ms();
        let empty_regions = HashSet::new();

        let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        });
        world.set_voxel(8, 8, 8, 8, VoxelType(3));

        save_state_from_world(
            &root,
            SaveWorldRequest {
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
        let empty_world = RegionChunkWorld::new_with_base(world.base_kind());
        save_state_from_world(
            &root,
            SaveWorldRequest {
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
        let loaded_world = materialize_loaded_world(&loaded);
        assert_eq!(loaded_world.get_voxel(8, 8, 8, 8), VoxelType(0));

        let _ = std::fs::remove_dir_all(root);
    }
}
