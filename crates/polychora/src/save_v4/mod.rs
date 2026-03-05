//! Save format v4: chunked world persistence with index trees and blob storage.

mod entity_io;
mod index_tree;
mod load;
mod save;

#[cfg(test)]
mod tests;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use entity_io::load_entities_for_regions;
pub use load::{
    load_or_init_state, load_or_init_state_metadata, load_state, load_state_metadata,
    load_world_chunk_payloads_for_bounds, load_world_chunk_payloads_for_bounds_from_index,
    load_world_chunk_payloads_for_regions, load_world_subtree_core_for_bounds_from_index,
};
pub use save::{save_state_from_chunk_payload_patch, save_state_from_chunk_payloads};

// Re-export submodule items used by tests (via `use super::*` in tests.rs).
#[cfg(test)]
use index_tree::{
    build_temp_tree_from_leaves, canonicalize_temp_tree, flatten_temp_tree, LeafDescriptor,
};
#[cfg(test)]
use load::materialize_world_chunk_payloads_from_index_filtered;
#[cfg(test)]
use save::save_state_from_world;

// ── Shared imports ──────────────────────────────────────────────────────────

#[cfg(test)]
use crate::migration::legacy_voxel::{Chunk as LegacyChunk, LegacyVoxel, RegionChunkWorld};
use crate::shared::chunk_payload::{
    ChunkArrayData, ChunkArrayIndexCodec, ChunkPayload as FieldChunkPayload, ResolvedChunkPayload,
};
use crate::shared::protocol::Entity;
#[cfg(test)]
use crate::shared::protocol::EntityPose;
#[cfg(test)]
use crate::shared::region_tree::chunk_key_i32;
use crate::shared::region_tree::{
    slice_region_core_in_bounds, ChunkKey, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::{
    chunk_key_from_lattice, lattice_from_fixed, Aabb4, Aabb4i, ChunkCoord,
};
#[cfg(test)]
use crate::shared::voxel::CHUNK_VOLUME;
use crate::shared::voxel::{linear_cell_index, BaseWorldKind, BlockData, CHUNK_SIZE};
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// ── File format constants ───────────────────────────────────────────────────

const MANIFEST_FILE: &str = "manifest.json";
const INDEX_DIR: &str = "index";
const DATA_DIR: &str = "data";
const GLOBAL_MAGIC: &[u8; 4] = b"V4G0";
const PLAYERS_MAGIC: &[u8; 4] = b"V4P0";
const INDEX_MAGIC: &[u8; 4] = b"V4IX";
const DATA_MAGIC: &[u8; 4] = b"V4DT";
const RECORD_MAGIC: &[u8; 4] = b"RECD";
const RECORD_HEADER_LEN: u32 = 4 + 1 + 2 + 4 + 4;
const INDEX_FILE_VERSION: u32 = 3;
const INDEX_ENTITY_ROOT_NONE: u32 = u32::MAX;

const BLOB_KIND_CHUNK_PAYLOAD: u8 = 1;
const BLOB_KIND_CHUNK_ARRAY: u8 = 2;
const BLOB_KIND_ENTITY: u8 = 3;

pub const SAVE_FORMAT_TAG: &str = "polychora-save";
pub const SAVE_FORMAT_VERSION: u32 = 4;
pub const PAYLOAD_FILE_VERSION: u32 = 1;
pub const DATA_FILE_VERSION: u32 = 1;
pub const CHUNK_PAYLOAD_BLOB_VERSION: u16 = 1;
pub const CHUNK_ARRAY_BLOB_VERSION: u16 = 3;
pub const ENTITY_BLOB_VERSION: u16 = 2;
pub const DEFAULT_REGION_CHUNK_EDGE: i32 = 4;

// ── Shared types ────────────────────────────────────────────────────────────

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
    pub fn from_runtime(base: &BaseWorldKind) -> Self {
        match base {
            BaseWorldKind::Empty => Self::Empty,
            BaseWorldKind::FlatFloor { material } => Self::FlatFloor {
                material: crate::content_registry::material_token_from_block_data(material),
            },
            BaseWorldKind::MassivePlatforms { material } => Self::MassivePlatforms {
                material: crate::content_registry::material_token_from_block_data(material),
            },
        }
    }

    pub fn to_runtime(self) -> BaseWorldKind {
        match self {
            PersistedBaseWorldKind::Empty => BaseWorldKind::Empty,
            PersistedBaseWorldKind::FlatFloor { material } => BaseWorldKind::FlatFloor {
                material: crate::content_registry::block_data_from_material_token(material),
            },
            PersistedBaseWorldKind::MassivePlatforms { material } => {
                BaseWorldKind::MassivePlatforms {
                    material: crate::content_registry::block_data_from_material_token(material),
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
    pub bounds_min_fixed: [i64; 4],
    pub bounds_max_fixed: [i64; 4],
    /// Scale exponent for leaf nodes (0 for branches and entity leaves).
    pub scale_exp: i8,
    pub kind: IndexNodeKind,
}

impl IndexNode {
    pub fn bounds_aabb4(&self) -> Aabb4i {
        Aabb4i::from_fixed_bits(self.bounds_min_fixed, self.bounds_max_fixed)
    }
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

/// Legacy V1 index node format with [i32; 4] bounds (scale-0 only).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyIndexNodeV1 {
    node_id: u32,
    bounds_min_chunk: [i32; 4],
    bounds_max_chunk: [i32; 4],
    kind: IndexNodeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyIndexBodyV1 {
    nodes: Vec<LegacyIndexNodeV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkPayloadBlob {
    pub payload: FieldChunkPayload,
}

fn default_chunk_array_blob_scale_exp() -> i8 {
    0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkArrayBlob {
    pub volume_min_chunk: [i32; 4],
    pub volume_max_chunk: [i32; 4],
    #[serde(default = "default_chunk_array_blob_scale_exp")]
    pub scale_exp: i8,
    pub payload_palette: Vec<BlobRef>,
    pub block_palette: Vec<BlockData>,
    pub index_codec: ChunkArrayIndexCodec,
    pub index_data: Vec<u8>,
    pub default_palette_index: Option<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyChunkArrayBlobV2Compat {
    volume_min_chunk: [i32; 4],
    volume_max_chunk: [i32; 4],
    payload_palette: Vec<BlobRef>,
    block_palette: Vec<BlockData>,
    index_codec: ChunkArrayIndexCodec,
    index_data: Vec<u8>,
    default_palette_index: Option<u16>,
}

fn decode_chunk_array_blob(payload: &[u8]) -> io::Result<ChunkArrayBlob> {
    match postcard::from_bytes::<ChunkArrayBlob>(payload) {
        Ok(blob) => Ok(blob),
        Err(primary_error) => {
            let legacy: LegacyChunkArrayBlobV2Compat = postcard::from_bytes(payload).map_err(
                |legacy_error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "failed to decode chunk array blob (v3: {primary_error}, v2 fallback: {legacy_error})"
                        ),
                    )
                },
            )?;
            Ok(ChunkArrayBlob {
                volume_min_chunk: legacy.volume_min_chunk,
                volume_max_chunk: legacy.volume_max_chunk,
                scale_exp: 0,
                payload_palette: legacy.payload_palette,
                block_palette: legacy.block_palette,
                index_codec: legacy.index_codec,
                index_data: legacy.index_data,
                default_palette_index: legacy.default_palette_index,
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedEntityRecord {
    pub entity_id: u64,
    pub entity: Entity,
    pub display_name: Option<String>,
    pub tags: Vec<String>,
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
    pub world_chunk_payloads: Vec<(ChunkKey, ResolvedChunkPayload)>,
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
    pub chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
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
    pub dirty_chunk_payloads: Vec<(ChunkKey, i8, Option<ResolvedChunkPayload>)>,
    pub world_seed: u64,
    pub next_entity_id: u64,
    pub player_entity_hints: Option<Vec<PlayerEntityHint>>,
    pub custom_global_payload: Option<Vec<u8>>,
    pub players: Option<Vec<PlayerRecord>>,
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

// ── Public utility functions ────────────────────────────────────────────────

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

/// Convert a ChunkKey to a region key (integer lattice).
pub fn region_from_chunk_key(chunk_key: ChunkKey, region_chunk_edge: i32) -> [i32; 4] {
    region_from_chunk(chunk_key.map(|c| c.to_num::<i32>()), region_chunk_edge)
}

pub fn all_entity_regions(
    entities: &[PersistedEntityRecord],
    region_chunk_edge: i32,
) -> HashSet<[i32; 4]> {
    entities
        .iter()
        .map(|entity| {
            region_from_chunk(
                chunk_from_world_position(entity.entity.pose.position),
                region_chunk_edge,
            )
        })
        .collect()
}

// ── Internal shared helpers ─────────────────────────────────────────────────

fn strip_scale_exp(
    payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)>,
) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
    payloads.into_iter().map(|(k, _, p)| (k, p)).collect()
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

/// Find the coarsest scale at which `bounds` round-trips through
/// `to_chunk_lattice_bounds` -> `from_lattice_bounds` without loss.
fn coarsest_lattice_aligned_scale(bounds: &Aabb4i, max_scale: i8) -> i8 {
    bounds.coarsest_lattice_aligned_scale(max_scale)
}

fn for_each_chunk_in_bounds<F>(min: [i32; 4], max: [i32; 4], mut f: F)
where
    F: FnMut([i32; 4]),
{
    for w in min[3]..=max[3] {
        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    f([x, y, z, w]);
                }
            }
        }
    }
}

/// Convert an `[i32; 4]` to a scale-0 ChunkKey.
#[cfg(test)]
fn chunk_key_from_i32(pos: [i32; 4]) -> ChunkKey {
    pos.map(ChunkCoord::from_num)
}

/// Convert a ChunkKey to lattice [i32; 4] at the given scale_exp.
fn chunk_key_to_lattice(key: &ChunkKey, scale_exp: i8) -> [i32; 4] {
    [
        lattice_from_fixed(key[0], scale_exp),
        lattice_from_fixed(key[1], scale_exp),
        lattice_from_fixed(key[2], scale_exp),
        lattice_from_fixed(key[3], scale_exp),
    ]
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

fn build_node_lookup(index: &IndexPayload) -> HashMap<u32, &IndexNode> {
    index
        .nodes
        .iter()
        .map(|node| (node.node_id, node))
        .collect()
}

fn fixed_bounds_are_valid(min: [i64; 4], max: [i64; 4]) -> bool {
    (0..4).all(|axis| min[axis] < max[axis])
}

fn fixed_bounds_contains(
    parent_min: [i64; 4],
    parent_max: [i64; 4],
    child_min: [i64; 4],
    child_max: [i64; 4],
) -> bool {
    (0..4).all(|axis| parent_min[axis] <= child_min[axis] && child_max[axis] <= parent_max[axis])
}

fn fixed_bounds_intersect(
    a_min: [i64; 4],
    a_max: [i64; 4],
    b_min: [i64; 4],
    b_max: [i64; 4],
) -> bool {
    (0..4).all(|axis| a_min[axis] < b_max[axis] && a_max[axis] > b_min[axis])
}

// ── Test helpers (cfg(test) only) ───────────────────────────────────────────

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

    let mut voxels = Box::new([LegacyVoxel::AIR; CHUNK_VOLUME]);
    let mut solid_count = 0u32;
    for (idx, material) in materials.into_iter().enumerate() {
        let voxel = LegacyVoxel(u8::try_from(material).unwrap_or(u8::MAX));
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
fn test_content_registry() -> crate::content_registry::ContentRegistry {
    let (registry, _pending) = crate::plugin_loader::create_full_registry();
    registry
}

#[cfg(test)]
fn chunk_payloads_to_voxel_world(
    base_world_kind: BaseWorldKind,
    chunk_payloads: &[(ChunkKey, ResolvedChunkPayload)],
) -> io::Result<RegionChunkWorld> {
    let reg = test_content_registry();
    let mut world = RegionChunkWorld::new_with_base(base_world_kind);
    for (key, resolved) in chunk_payloads {
        let legacy = resolved.to_material_token_payload(|block| {
            reg.block_material_token(block.namespace, block.block_type) as u8
        });
        let chunk = legacy_chunk_from_field_chunk_payload(&legacy)?;
        let pos = key.map(|c| c.to_num::<i32>());
        world.insert_chunk(pos, chunk);
    }
    world.clear_dirty();
    let _ = world.drain_pending_chunk_updates();
    Ok(world)
}

// ── File I/O helpers ────────────────────────────────────────────────────────

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

    let empty_wb = Aabb4i::chunk_world_bounds([ChunkCoord::ZERO; 4], 0);
    let initial_index = IndexPayload {
        generation,
        root_node_id: 0,
        entity_root_node_id: None,
        nodes: vec![IndexNode {
            node_id: 0,
            bounds_min_fixed: empty_wb.min.map(|c| c.to_bits()),
            bounds_max_fixed: empty_wb.max.map(|c| c.to_bits()),
            scale_exp: 0,
            kind: IndexNodeKind::Branch {
                child_node_ids: Vec::new(),
            },
        }],
    };
    write_index_file(root.join(&index_file), &initial_index)?;

    let initial_global = GlobalPayload {
        base_world_kind: PersistedBaseWorldKind::from_runtime(&base_world_kind),
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

    save::ensure_data_file(root, 1)?;
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
        save::ensure_data_file(root, manifest.active_data_file_id)?;
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
        let file = writer.into_inner().map_err(io::Error::other)?;
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
    let file = writer.into_inner().map_err(io::Error::other)?;
    file.sync_all()?;
    Ok(())
}

/// Migrate V2 index nodes from key-space inclusive bounds to world-space half-open bounds.
///
/// V2 stored node bounds as `[min_key, max_key]` inclusive in fixed-point.
/// V3 stores bounds as `[min_world, max_world)` half-open in world-space.
///
/// For leaf nodes: convert using the node's `scale_exp`.
/// For branch nodes: recompute as the hull of children's converted bounds.
fn migrate_v2_index_nodes_to_v3(mut nodes: Vec<IndexNode>) -> Vec<IndexNode> {
    use crate::shared::spatial::step_for_scale;

    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);

    // Build child -> parent mapping and identify leaf vs branch.
    let node_id_to_idx: HashMap<u32, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.node_id, i))
        .collect();

    // Convert leaf nodes first (bottom-up).
    // For leaf: world_min = key_min * CS, world_max = (key_max * CS) + CS * step(scale_exp)
    for node in nodes.iter_mut() {
        if matches!(node.kind, IndexNodeKind::Branch { .. }) {
            continue;
        }
        let old_bounds = node.bounds_aabb4();
        let step = step_for_scale(node.scale_exp);
        let chunk_world_size = cs.saturating_mul(step);
        let new_min = old_bounds.min.map(|k| k.saturating_mul(cs));
        let new_max = old_bounds
            .max
            .map(|k| k.saturating_mul(cs).saturating_add(chunk_world_size));
        node.bounds_min_fixed = new_min.map(|v| v.to_bits());
        node.bounds_max_fixed = new_max.map(|v| v.to_bits());
    }

    // Recompute branch bounds as hull of children (may need multiple passes
    // for deeply nested branches, but typically only one level deep).
    let mut changed = true;
    while changed {
        changed = false;
        // Collect branch updates (can't borrow nodes mutably while iterating).
        let mut updates: Vec<(usize, [i64; 4], [i64; 4])> = Vec::new();
        for (idx, node) in nodes.iter().enumerate() {
            if let IndexNodeKind::Branch { child_node_ids } = &node.kind {
                let mut hull_min = [i64::MAX; 4];
                let mut hull_max = [i64::MIN; 4];
                for child_id in child_node_ids {
                    if let Some(&child_idx) = node_id_to_idx.get(child_id) {
                        let child = &nodes[child_idx];
                        for axis in 0..4 {
                            hull_min[axis] = hull_min[axis].min(child.bounds_min_fixed[axis]);
                            hull_max[axis] = hull_max[axis].max(child.bounds_max_fixed[axis]);
                        }
                    }
                }
                if hull_min[0] != i64::MAX
                    && (node.bounds_min_fixed != hull_min || node.bounds_max_fixed != hull_max)
                {
                    updates.push((idx, hull_min, hull_max));
                    changed = true;
                }
            }
        }
        for (idx, min, max) in updates {
            nodes[idx].bounds_min_fixed = min;
            nodes[idx].bounds_max_fixed = max;
        }
    }

    nodes
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
    if version != INDEX_FILE_VERSION && version != 2 && version != 1 {
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

    let nodes = if version == 1 {
        // Legacy V1: bounds stored as [i32; 4], convert to [i64; 4] fixed-point bits.
        let legacy_body: LegacyIndexBodyV1 = postcard::from_bytes(&payload)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if legacy_body.nodes.len() != node_count as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "index node count mismatch: header={}, payload={}",
                    node_count,
                    legacy_body.nodes.len()
                ),
            ));
        }
        legacy_body
            .nodes
            .into_iter()
            .map(|legacy| IndexNode {
                node_id: legacy.node_id,
                bounds_min_fixed: legacy
                    .bounds_min_chunk
                    .map(|v| ChunkCoord::from_num(v).to_bits()),
                bounds_max_fixed: legacy
                    .bounds_max_chunk
                    .map(|v| ChunkCoord::from_num(v).to_bits()),
                scale_exp: 0,
                kind: legacy.kind,
            })
            .collect()
    } else if version == 2 {
        // V2: bounds are key-space inclusive. Convert to world-space half-open.
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
        migrate_v2_index_nodes_to_v3(body.nodes)
    } else {
        // V3+: bounds are already world-space half-open.
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
        body.nodes
    };

    let entity_root_node_id =
        (entity_root_raw != INDEX_ENTITY_ROOT_NONE).then_some(entity_root_raw);

    let index = IndexPayload {
        generation,
        root_node_id,
        entity_root_node_id,
        nodes,
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
        if !fixed_bounds_are_valid(node.bounds_min_fixed, node.bounds_max_fixed) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "node {} has invalid bounds min={:?} max={:?}",
                    node.node_id, node.bounds_min_fixed, node.bounds_max_fixed
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
    parent_bounds: Option<([i64; 4], [i64; 4])>,
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
        if !fixed_bounds_contains(
            parent_min,
            parent_max,
            node.bounds_min_fixed,
            node.bounds_max_fixed,
        ) {
            stack.remove(&node_id);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "node {} bounds {:?}..{:?} not contained in parent {:?}..{:?}",
                    node.node_id,
                    node.bounds_min_fixed,
                    node.bounds_max_fixed,
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
                    index_tree::fixed_bounds_key(left.bounds_min_fixed, left.bounds_max_fixed),
                    index_tree::index_kind_order_key(&left.kind),
                );
                let right_key = (
                    index_tree::fixed_bounds_key(right.bounds_min_fixed, right.bounds_max_fixed),
                    index_tree::index_kind_order_key(&right.kind),
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
                    if fixed_bounds_intersect(
                        a.bounds_min_fixed,
                        a.bounds_max_fixed,
                        b.bounds_min_fixed,
                        b.bounds_max_fixed,
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
                    Some((node.bounds_min_fixed, node.bounds_max_fixed)),
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
    let file = writer.into_inner().map_err(io::Error::other)?;
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
