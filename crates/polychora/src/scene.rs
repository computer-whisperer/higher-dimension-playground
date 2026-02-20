use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::world::BaseWorldKind;
use crate::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkHeader, GpuVoxelYSliceBounds, VoxelFrameInput, VTE_MAX_CHUNKS,
};
use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use polychora::shared::region_tree::{
    chunk_key_from_chunk_pos, chunk_pos_from_chunk_key,
    collect_non_empty_chunks_from_core_in_bounds, slice_non_empty_region_core_in_bounds,
    RegionChunkTree, RegionNodeKind, RegionTreeCore,
};
use polychora::shared::render_tree::{self, RenderTreeCore};
use polychora::shared::spatial::Aabb4i;
use polychora::shared::voxel::world_to_chunk;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

mod voxel_runtime;

const RENDER_DISTANCE: f32 = 64.0;
const VOXEL_NEAR_ACTIVE_DISTANCE: f32 = 32.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 4; // packed 4x u8 per u32
const MACRO_CELLS_PER_AXIS: usize = CHUNK_SIZE / 2; // 2x2x2x2 macro cells
const MACRO_CELLS_PER_CHUNK: usize =
    MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS;
const MACRO_WORDS_PER_CHUNK: usize = MACRO_CELLS_PER_CHUNK / 32;
const Y_SLICE_LOOKUP_MAX_ENTRIES: usize = 131_072;
const Y_SLICE_LOOKUP_MAX_ENTRIES_PER_SLICE: usize = 65_536;
const EDIT_RAY_EPSILON: f32 = 1e-4;
const EDIT_RAY_MAX_STEPS: usize = 4096;
const COLLISION_PUSHUP_STEP: f32 = 0.05;
const COLLISION_MAX_PUSHUP_STEPS: usize = 80;
const COLLISION_BINARY_STEPS: usize = 14;
const HARD_WORLD_FLOOR_Y: f32 = -4.0;
const FLAT_FLOOR_CHUNK_Y: i32 = -1;
const GPU_PAYLOAD_SLOT_CAPACITY: usize = VTE_MAX_CHUNKS;
pub const VOXEL_LOD_LEVEL_NEAR: u8 = 0;
const FLAT_PRESET_FLOOR_MATERIAL: VoxelType = VoxelType(11);
const SHOWCASE_MATERIALS: [u8; 37] = [
    15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 45, 50, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
];
type DenseChunk = [VoxelType; CHUNK_VOLUME];

fn set_chunk_voxel_by_index(chunk: &mut DenseChunk, idx: usize, v: VoxelType) {
    chunk[idx] = v;
}

#[inline]
fn empty_dense_chunk() -> DenseChunk {
    [VoxelType::AIR; CHUNK_VOLUME]
}

#[inline]
fn dense_chunk_is_empty(chunk: &DenseChunk) -> bool {
    chunk.iter().all(|voxel| voxel.is_air())
}

#[inline]
fn dense_chunk_local_index(x: usize, y: usize, z: usize, w: usize) -> usize {
    w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

#[inline]
fn dense_chunk_set(
    chunk: &mut DenseChunk,
    x: usize,
    y: usize,
    z: usize,
    w: usize,
    voxel: VoxelType,
) {
    let idx = dense_chunk_local_index(x, y, z, w);
    chunk[idx] = voxel;
}

fn dense_chunk_to_payload_compact(chunk: &DenseChunk) -> ChunkPayload {
    let materials: Vec<u16> = chunk.iter().map(|voxel| u16::from(voxel.0)).collect();
    ChunkPayload::from_dense_materials_compact(&materials)
        .unwrap_or(ChunkPayload::Dense16 { materials })
}

fn dense_chunk_from_payload(payload: &ChunkPayload) -> Result<DenseChunk, String> {
    let materials = payload
        .dense_materials()
        .map_err(|error| error.to_string())?;
    if materials.len() != CHUNK_VOLUME {
        return Err(format!(
            "unexpected dense material length {}, expected {}",
            materials.len(),
            CHUNK_VOLUME
        ));
    }
    let mut chunk = empty_dense_chunk();
    for (idx, material) in materials.into_iter().enumerate() {
        chunk[idx] = VoxelType(u8::try_from(material).unwrap_or(u8::MAX));
    }
    Ok(chunk)
}

struct CachedChunkPayload {
    hash: u64,
    solid_count: u32,
    is_full: bool,
    ref_count: u32,
    gpu_slot: u32,
    occupancy_words: Box<[u32; OCCUPANCY_WORDS_PER_CHUNK]>,
    material_words: Box<[u32; MATERIAL_WORDS_PER_CHUNK]>,
    macro_words: Box<[u32; MACRO_WORDS_PER_CHUNK]>,
    solid_local_min: [i32; 4],
    solid_local_max: [i32; 4],
}

#[derive(Copy, Clone)]
struct ChunkPayloadCacheEntry {
    payload_id: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct RuntimeChunkKey {
    lod_level: u8,
    chunk_pos: ChunkPos,
}

#[derive(Copy, Clone, Debug)]
pub enum ScenePreset {
    Empty,
    Flat,
    DemoCubes,
}

impl ScenePreset {
    fn label(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Flat => "flat",
            Self::DemoCubes => "demo_cubes",
        }
    }
}

pub struct VoxelFrameData {
    pub metadata_generation: u64,
    pub chunk_headers: Vec<GpuVoxelChunkHeader>,
    pub payload_update_slots: Vec<u32>,
    pub occupancy_words: Vec<u32>,
    pub material_words: Vec<u32>,
    pub macro_words: Vec<u32>,
    pub visible_chunk_indices: Vec<u32>,
    pub y_slice_bounds: Vec<GpuVoxelYSliceBounds>,
    pub y_slice_lookup_entries: Vec<u32>,
}

impl VoxelFrameData {
    pub fn as_input(&self) -> VoxelFrameInput<'_> {
        VoxelFrameInput {
            metadata_generation: self.metadata_generation,
            chunk_headers: &self.chunk_headers,
            payload_update_slots: &self.payload_update_slots,
            occupancy_words: &self.occupancy_words,
            material_words: &self.material_words,
            macro_words: &self.macro_words,
            visible_chunk_indices: &self.visible_chunk_indices,
            y_slice_bounds: &self.y_slice_bounds,
            y_slice_lookup_entries: &self.y_slice_lookup_entries,
        }
    }
}

pub struct Scene {
    world_tree: RegionChunkTree,
    world_chunks: HashMap<ChunkPos, DenseChunk>,
    world_base_kind: BaseWorldKind,
    world_flat_floor_chunk: DenseChunk,
    world_dirty: bool,
    world_pending_chunk_updates: Vec<ChunkPos>,
    world_pending_chunk_update_set: HashSet<ChunkPos>,
    surface: SurfaceData,
    culled_instances: Vec<common::ModelInstance>,
    cull_log_counter: u64,
    voxel_chunk_payload_cache: HashMap<RuntimeChunkKey, ChunkPayloadCacheEntry>,
    voxel_chunk_payloads: Vec<Option<CachedChunkPayload>>,
    voxel_chunk_payload_free_ids: Vec<u32>,
    voxel_chunk_payload_hash_buckets: HashMap<u64, Vec<u32>>,
    voxel_payload_slot_to_payload: Vec<Option<u32>>,
    voxel_payload_free_slots: Vec<u32>,
    voxel_pending_payload_uploads: Vec<u32>,
    voxel_pending_payload_upload_set: HashSet<u32>,
    voxel_active_chunks: Vec<RuntimeChunkKey>,
    voxel_active_chunk_indices: HashMap<RuntimeChunkKey, usize>,
    voxel_world_revision: u64,
    voxel_visibility_generation: u64,
    voxel_cached_visibility_camera_chunk: Option<[i32; 12]>,
    voxel_cached_visibility_world_revision: u64,
    voxel_payload_slot_overflow_logged: bool,
    voxel_frame_data: VoxelFrameData,
}

#[derive(Copy, Clone)]
struct VoxelRayHit {
    solid_voxel: [i32; 4],
    last_empty_voxel: Option<[i32; 4]>,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockEditTargets {
    pub hit_voxel: Option<[i32; 4]>,
    pub place_voxel: Option<[i32; 4]>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NearRegionPatchStats {
    pub previous_non_empty: usize,
    pub desired_non_empty: usize,
    pub upserts: usize,
    pub removals: usize,
    pub changed_chunks: usize,
    pub previous_total_chunks: usize,
    pub desired_total_chunks: usize,
    pub invalidated_cached_chunks: usize,
    pub queued_updates: usize,
    pub collect_previous_ms: f64,
    pub splice_ms: f64,
    pub collect_desired_ms: f64,
    pub diff_ms: f64,
}

impl Scene {
    fn chunk_payload_material_at(payload: &ChunkPayload, idx: usize) -> Option<u16> {
        if idx >= CHUNK_VOLUME {
            return None;
        }
        match payload {
            ChunkPayload::Empty => Some(0),
            ChunkPayload::Uniform(material) => Some(*material),
            ChunkPayload::Dense16 { materials } => materials.get(idx).copied(),
            ChunkPayload::PalettePacked {
                palette,
                bit_width,
                packed_indices,
            } => {
                if palette.is_empty() {
                    return None;
                }
                let palette_idx = if *bit_width == 0 {
                    0usize
                } else {
                    let bit_width = usize::from(*bit_width);
                    let bit_index = idx.checked_mul(bit_width)?;
                    let word_index = bit_index / 64;
                    let bit_offset = bit_index % 64;
                    let first = *packed_indices.get(word_index).unwrap_or(&0);
                    let raw = if bit_offset + bit_width <= 64 {
                        first >> bit_offset
                    } else {
                        let second = *packed_indices.get(word_index + 1).unwrap_or(&0);
                        (first >> bit_offset) | (second << (64 - bit_offset))
                    };
                    let mask = if bit_width >= 64 {
                        u64::MAX
                    } else {
                        (1u64 << bit_width) - 1
                    };
                    (raw & mask) as usize
                };
                palette.get(palette_idx).copied()
            }
        }
    }

    fn chunk_payload_has_solid_material(payload: &ChunkPayload) -> bool {
        match payload {
            ChunkPayload::Empty => false,
            ChunkPayload::Uniform(material) => *material != 0,
            ChunkPayload::Dense16 { materials } => materials.iter().any(|material| *material != 0),
            ChunkPayload::PalettePacked { .. } => (0..CHUNK_VOLUME).any(|idx| {
                Self::chunk_payload_material_at(payload, idx)
                    .map(|material| material != 0)
                    .unwrap_or(false)
            }),
        }
    }

    fn chunk_payload_to_voxel_type(payload: &ChunkPayload, idx: usize) -> VoxelType {
        let material = Self::chunk_payload_material_at(payload, idx).unwrap_or(0);
        VoxelType(u8::try_from(material).unwrap_or(u8::MAX))
    }

    fn set_chunk_map_voxel(
        chunks: &mut HashMap<ChunkPos, DenseChunk>,
        wx: i32,
        wy: i32,
        wz: i32,
        ww: i32,
        v: VoxelType,
    ) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        let should_remove = {
            let chunk = chunks.entry(cp).or_insert_with(empty_dense_chunk);
            set_chunk_voxel_by_index(chunk, idx, v);
            dense_chunk_is_empty(chunk)
        };
        if should_remove {
            chunks.remove(&cp);
        }
    }

    fn fill_hypercube(
        chunks: &mut HashMap<ChunkPos, DenseChunk>,
        min: [i32; 4],
        edge: i32,
        material: VoxelType,
    ) {
        for x in min[0]..(min[0] + edge) {
            for y in min[1]..(min[1] + edge) {
                for z in min[2]..(min[2] + edge) {
                    for w in min[3]..(min[3] + edge) {
                        Self::set_chunk_map_voxel(chunks, x, y, z, w, material);
                    }
                }
            }
        }
    }

    fn place_material_showcase(chunks: &mut HashMap<ChunkPos, DenseChunk>, origin: [i32; 4]) {
        for (idx, material) in SHOWCASE_MATERIALS.iter().copied().enumerate() {
            let col = (idx % 6) as i32;
            let row = (idx / 6) as i32;
            let min = [
                origin[0] + col * 4,
                origin[1],
                origin[2] + row * 4,
                origin[3],
            ];
            Self::fill_hypercube(chunks, min, 2, VoxelType(material));
        }
    }

    fn build_scene_preset_world(
        preset: ScenePreset,
    ) -> (BaseWorldKind, HashMap<ChunkPos, DenseChunk>) {
        let mut chunks = HashMap::<ChunkPos, DenseChunk>::new();

        let base_kind = match preset {
            ScenePreset::Empty => BaseWorldKind::Empty,
            ScenePreset::Flat => {
                Self::place_material_showcase(&mut chunks, [-10, 0, -14, -4]);
                BaseWorldKind::FlatFloor {
                    material: FLAT_PRESET_FLOOR_MATERIAL,
                }
            }
            ScenePreset::DemoCubes => {
                let mut texture_rot = 0u8;
                for x in 0..2 {
                    for y in 0..2 {
                        for z in 0..2 {
                            for w in 0..2 {
                                let base = [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2];
                                let material = VoxelType((texture_rot % 5) + 1);
                                Self::fill_hypercube(&mut chunks, base, 2, material);
                                texture_rot = (texture_rot + 1) % 5;
                            }
                        }
                    }
                }
                Self::fill_hypercube(&mut chunks, [0, 0, 0, 0], 2, VoxelType(13));
                BaseWorldKind::Empty
            }
        };

        (base_kind, chunks)
    }

    fn scene_world_from_chunk_overrides(
        base_kind: BaseWorldKind,
        mut world_chunks: HashMap<ChunkPos, DenseChunk>,
    ) -> (
        RegionChunkTree,
        HashMap<ChunkPos, DenseChunk>,
        BaseWorldKind,
        DenseChunk,
        bool,
    ) {
        world_chunks.retain(|_, chunk| !dense_chunk_is_empty(chunk));
        let world_tree = RegionChunkTree::from_chunks(world_chunks.iter().map(|(&pos, chunk)| {
            (
                chunk_key_from_chunk_pos(pos),
                dense_chunk_to_payload_compact(chunk),
            )
        }));
        let flat_floor_chunk = Self::flat_floor_chunk_for_base(base_kind);
        (world_tree, world_chunks, base_kind, flat_floor_chunk, false)
    }

    fn build_flat_floor_chunk(material: VoxelType) -> DenseChunk {
        let mut chunk = empty_dense_chunk();
        if material.is_air() {
            return chunk;
        }

        let local_y_top = CHUNK_SIZE - 1;
        let local_y_bottom = CHUNK_SIZE - 2;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for w in 0..CHUNK_SIZE {
                    dense_chunk_set(&mut chunk, x, local_y_top, z, w, material);
                    dense_chunk_set(&mut chunk, x, local_y_bottom, z, w, material);
                }
            }
        }
        chunk
    }

    fn flat_floor_chunk_for_base(base_kind: BaseWorldKind) -> DenseChunk {
        match base_kind {
            BaseWorldKind::FlatFloor { material }
            | BaseWorldKind::MassivePlatforms { material } => {
                Self::build_flat_floor_chunk(material)
            }
            BaseWorldKind::Empty => empty_dense_chunk(),
        }
    }

    fn world_base_chunk_for_pos(&self, pos: ChunkPos) -> Option<&DenseChunk> {
        match self.world_base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } | BaseWorldKind::MassivePlatforms { .. }
                if pos.y == FLAT_FLOOR_CHUNK_Y =>
            {
                Some(&self.world_flat_floor_chunk)
            }
            BaseWorldKind::FlatFloor { .. } | BaseWorldKind::MassivePlatforms { .. } => None,
        }
    }

    fn world_base_voxel_at(&self, pos: ChunkPos, idx: usize) -> VoxelType {
        self.world_base_chunk_for_pos(pos)
            .map(|chunk| chunk[idx])
            .unwrap_or(VoxelType::AIR)
    }

    fn world_clone_base_chunk_or_empty(&self, pos: ChunkPos) -> DenseChunk {
        self.world_base_chunk_for_pos(pos)
            .cloned()
            .unwrap_or_else(empty_dense_chunk)
    }

    fn world_chunk_matches_base(&self, pos: ChunkPos, chunk: &DenseChunk) -> bool {
        match self.world_base_chunk_for_pos(pos) {
            Some(base) => chunk[..] == base[..],
            None => dense_chunk_is_empty(chunk),
        }
    }

    fn world_queue_chunk_update(&mut self, pos: ChunkPos) -> bool {
        if self.world_pending_chunk_update_set.insert(pos) {
            self.world_pending_chunk_updates.push(pos);
            true
        } else {
            false
        }
    }

    fn world_set_chunk_override(&mut self, pos: ChunkPos, chunk: Option<DenseChunk>) -> bool {
        let key = chunk_key_from_chunk_pos(pos);
        let payload = match chunk {
            Some(chunk) => {
                if self.world_chunk_matches_base(pos, &chunk) {
                    self.world_chunks.remove(&pos);
                    None
                } else {
                    if dense_chunk_is_empty(&chunk) {
                        self.world_chunks.remove(&pos);
                    } else {
                        self.world_chunks.insert(pos, chunk);
                    }
                    Some(dense_chunk_to_payload_compact(&chunk))
                }
            }
            None => {
                self.world_chunks.remove(&pos);
                None
            }
        };
        let changed = self.world_tree.set_chunk(key, payload);
        if changed {
            self.world_dirty = true;
            let _ = self.world_queue_chunk_update(pos);
        }
        changed
    }

    fn world_has_explicit_chunk(&self, pos: ChunkPos) -> bool {
        self.world_tree.has_chunk(chunk_key_from_chunk_pos(pos))
    }

    fn world_chunk_at(&mut self, pos: ChunkPos) -> Option<DenseChunk> {
        if let Some(chunk) = self.world_chunks.get(&pos) {
            return (!dense_chunk_is_empty(chunk)).then_some(*chunk);
        }

        if let Some(payload) = self.world_tree.chunk_payload(chunk_key_from_chunk_pos(pos)) {
            if !Self::chunk_payload_has_solid_material(&payload) {
                return None;
            }
            match dense_chunk_from_payload(&payload) {
                Ok(chunk) => {
                    if dense_chunk_is_empty(&chunk) {
                        return None;
                    }
                    self.world_chunks.insert(pos, chunk);
                    return Some(chunk);
                }
                Err(error) => {
                    eprintln!(
                        "Ignoring malformed world chunk payload at ({}, {}, {}, {}): {}",
                        pos.x, pos.y, pos.z, pos.w, error
                    );
                    return None;
                }
            }
        }
        self.world_base_chunk_for_pos(pos)
            .filter(|chunk| !dense_chunk_is_empty(chunk))
            .cloned()
    }

    fn world_get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        if let Some(chunk) = self.world_chunks.get(&cp) {
            return chunk[idx];
        }
        if let Some(payload) = self.world_tree.chunk_payload(chunk_key_from_chunk_pos(cp)) {
            return Self::chunk_payload_to_voxel_type(&payload, idx);
        }
        self.world_base_voxel_at(cp, idx)
    }

    fn world_set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, v: VoxelType) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        let old = self.world_get_voxel(wx, wy, wz, ww);
        if old == v {
            return;
        }

        let mut chunk = if self.world_has_explicit_chunk(cp) {
            self.world_chunk_at(cp).unwrap_or_else(empty_dense_chunk)
        } else {
            self.world_clone_base_chunk_or_empty(cp)
        };
        set_chunk_voxel_by_index(&mut chunk, idx, v);
        let _ = self.world_set_chunk_override(cp, Some(chunk));
    }

    fn base_floor_region_core_in_bounds(&self, bounds: Aabb4i) -> Option<RegionTreeCore> {
        if !bounds.is_valid() {
            return None;
        }
        let floor_material = match self.world_base_kind {
            BaseWorldKind::FlatFloor { material }
            | BaseWorldKind::MassivePlatforms { material } => material,
            BaseWorldKind::Empty => return None,
        };
        if floor_material.is_air() {
            return None;
        }
        if FLAT_FLOOR_CHUNK_Y < bounds.min[1] || FLAT_FLOOR_CHUNK_Y > bounds.max[1] {
            return None;
        }
        let floor_bounds = Aabb4i::new(
            [
                bounds.min[0],
                FLAT_FLOOR_CHUNK_Y,
                bounds.min[2],
                bounds.min[3],
            ],
            [
                bounds.max[0],
                FLAT_FLOOR_CHUNK_Y,
                bounds.max[2],
                bounds.max[3],
            ],
        );
        if !floor_bounds.is_valid() {
            return None;
        }

        let floor_payload = dense_chunk_to_payload_compact(&self.world_flat_floor_chunk);
        let floor_kind = match floor_payload {
            ChunkPayload::Empty => return None,
            ChunkPayload::Uniform(material) => RegionNodeKind::Uniform(material),
            payload => {
                let cell_count = floor_bounds.chunk_cell_count()?;
                let indices = vec![0u16; cell_count];
                let chunk_array =
                    ChunkArrayData::from_dense_indices(floor_bounds, vec![payload], indices, None)
                        .ok()?;
                RegionNodeKind::ChunkArray(chunk_array)
            }
        };

        Some(RegionTreeCore {
            bounds: floor_bounds,
            kind: floor_kind,
            generator_version_hash: 0,
        })
    }

    fn build_render_tree_core_in_bounds(&self, bounds: Aabb4i) -> RenderTreeCore {
        if !bounds.is_valid() {
            return RenderTreeCore::empty(bounds);
        }
        let mut composed = RegionChunkTree::new();
        if let Some(base_floor_core) = self.base_floor_region_core_in_bounds(bounds) {
            let _ =
                composed.splice_non_empty_core_in_bounds(base_floor_core.bounds, &base_floor_core);
        }
        let world_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = composed.overlay_core_in_bounds(bounds, &world_core);
        let composed_core = composed.slice_non_empty_core_in_bounds(bounds);
        render_tree::from_region_core(&composed_core)
    }

    fn collect_render_non_empty_chunks_in_bounds(&self, bounds: Aabb4i) -> Vec<ChunkPos> {
        let render_core = self.build_render_tree_core_in_bounds(bounds);
        render_tree::collect_non_empty_chunk_keys_in_bounds(&render_core, bounds)
            .into_iter()
            .map(chunk_pos_from_chunk_key)
            .collect()
    }

    fn world_drain_pending_chunk_updates(&mut self) -> Vec<ChunkPos> {
        self.world_pending_chunk_update_set.clear();
        std::mem::take(&mut self.world_pending_chunk_updates)
    }

    fn world_any_dirty(&self) -> bool {
        self.world_dirty
    }

    fn world_clear_dirty(&mut self) {
        self.world_dirty = false;
    }

    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        self.world_get_voxel(wx, wy, wz, ww)
    }

    pub fn collect_non_empty_explicit_chunk_positions(&self) -> Vec<[i32; 4]> {
        let Some(root) = self.world_tree.root() else {
            return Vec::new();
        };
        let mut out: Vec<[i32; 4]> = self
            .world_tree
            .collect_non_empty_chunk_keys_in_bounds(root.bounds)
            .into_iter()
            .map(|key| key)
            .collect();
        out.sort_unstable();
        out
    }

    pub fn apply_near_lod_region_patch(
        &mut self,
        bounds: Aabb4i,
        subtree: &RegionTreeCore,
    ) -> NearRegionPatchStats {
        if !bounds.is_valid() {
            return NearRegionPatchStats::default();
        }

        let collect_previous_start = Instant::now();
        let previous_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let collect_previous_ms = collect_previous_start.elapsed().as_secs_f64() * 1000.0;
        let collect_desired_start = Instant::now();
        let desired_core = slice_non_empty_region_core_in_bounds(subtree, bounds);
        let collect_desired_ms = collect_desired_start.elapsed().as_secs_f64() * 1000.0;
        if previous_core.kind == desired_core.kind {
            let previous_non_empty = Self::count_non_empty_chunks_in_core(&previous_core);
            return NearRegionPatchStats {
                previous_non_empty,
                desired_non_empty: previous_non_empty,
                upserts: 0,
                removals: 0,
                changed_chunks: 0,
                previous_total_chunks: previous_non_empty,
                desired_total_chunks: previous_non_empty,
                invalidated_cached_chunks: 0,
                queued_updates: 0,
                collect_previous_ms,
                splice_ms: 0.0,
                collect_desired_ms,
                diff_ms: 0.0,
            };
        }

        let previous_chunks = collect_non_empty_chunks_from_core_in_bounds(&previous_core, bounds);
        let mut previous_by_pos =
            HashMap::<ChunkPos, ChunkPayload>::with_capacity(previous_chunks.len());
        for (key, payload) in previous_chunks {
            previous_by_pos.insert(chunk_pos_from_chunk_key(key), payload);
        }
        let previous_non_empty = previous_by_pos.len();
        let previous_total_chunks = previous_by_pos.len();

        let desired_chunks = collect_non_empty_chunks_from_core_in_bounds(&desired_core, bounds);
        let mut desired_by_pos =
            HashMap::<ChunkPos, ChunkPayload>::with_capacity(desired_chunks.len());
        for (key, payload) in desired_chunks {
            desired_by_pos.insert(chunk_pos_from_chunk_key(key), payload);
        }
        let desired_non_empty = desired_by_pos.len();
        let desired_total_chunks = desired_by_pos.len();

        let diff_start = Instant::now();
        let mut changed_positions = Vec::<ChunkPos>::new();
        let mut upserts = 0usize;
        let mut removals = 0usize;

        for (&pos, previous_payload) in &previous_by_pos {
            let desired_payload = desired_by_pos.get(&pos);
            if desired_payload == Some(previous_payload) {
                continue;
            }

            changed_positions.push(pos);
            if desired_payload.is_some() {
                upserts = upserts.saturating_add(1);
            } else {
                removals = removals.saturating_add(1);
            }
        }

        for &pos in desired_by_pos.keys() {
            if previous_by_pos.contains_key(&pos) {
                continue;
            }

            changed_positions.push(pos);
            upserts = upserts.saturating_add(1);
        }
        let diff_ms = diff_start.elapsed().as_secs_f64() * 1000.0;
        let changed_chunks = changed_positions.len();

        if changed_chunks == 0 {
            return NearRegionPatchStats {
                previous_non_empty,
                desired_non_empty,
                upserts: 0,
                removals: 0,
                changed_chunks: 0,
                previous_total_chunks,
                desired_total_chunks,
                invalidated_cached_chunks: 0,
                queued_updates: 0,
                collect_previous_ms,
                splice_ms: 0.0,
                collect_desired_ms,
                diff_ms,
            };
        }

        let splice_start = Instant::now();
        let _ = self
            .world_tree
            .splice_non_empty_core_in_bounds(bounds, &desired_core);
        let splice_ms = splice_start.elapsed().as_secs_f64() * 1000.0;

        let mut invalidated_cached_chunks = 0usize;
        let mut queued_updates = 0usize;
        for pos in changed_positions {
            if self.world_chunks.remove(&pos).is_some() {
                invalidated_cached_chunks = invalidated_cached_chunks.saturating_add(1);
            }
            if self.world_queue_chunk_update(pos) {
                queued_updates = queued_updates.saturating_add(1);
            }
        }

        self.world_dirty = true;

        NearRegionPatchStats {
            previous_non_empty,
            desired_non_empty,
            upserts,
            removals,
            changed_chunks,
            previous_total_chunks,
            desired_total_chunks,
            invalidated_cached_chunks,
            queued_updates,
            collect_previous_ms,
            splice_ms,
            collect_desired_ms,
            diff_ms,
        }
    }

    fn surface_voxel_count(surface: &SurfaceData) -> u32 {
        surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum()
    }

    fn count_non_empty_chunks_in_core(core: &RegionTreeCore) -> usize {
        fn payload_has_non_empty_material(payload: &ChunkPayload) -> bool {
            match payload {
                ChunkPayload::Empty => false,
                ChunkPayload::Uniform(material) => *material != 0,
                ChunkPayload::Dense16 { materials } => materials.iter().any(|m| *m != 0),
                ChunkPayload::PalettePacked { .. } => payload
                    .dense_materials()
                    .map(|dense| dense.into_iter().any(|m| m != 0))
                    .unwrap_or(true),
            }
        }

        fn recurse(core: &RegionTreeCore) -> usize {
            match &core.kind {
                polychora::shared::region_tree::RegionNodeKind::Empty
                | polychora::shared::region_tree::RegionNodeKind::ProceduralRef(_) => 0,
                polychora::shared::region_tree::RegionNodeKind::Uniform(material) => {
                    if *material == 0 {
                        0
                    } else {
                        core.bounds.chunk_cell_count().unwrap_or(0)
                    }
                }
                polychora::shared::region_tree::RegionNodeKind::ChunkArray(chunk_array) => {
                    let Ok(indices) = chunk_array.decode_dense_indices() else {
                        return 0;
                    };
                    let palette_non_empty: Vec<bool> = chunk_array
                        .chunk_palette
                        .iter()
                        .map(payload_has_non_empty_material)
                        .collect();
                    indices
                        .into_iter()
                        .filter(|idx| {
                            palette_non_empty
                                .get(*idx as usize)
                                .copied()
                                .unwrap_or(true)
                        })
                        .count()
                }
                polychora::shared::region_tree::RegionNodeKind::Branch(children) => {
                    children.iter().map(recurse).sum()
                }
            }
        }

        recurse(core)
    }

    fn rebuild_surface(&mut self, label: &str) {
        self.surface = cull::extract_surfaces(
            &self.world_tree,
            self.world_base_kind,
            &self.world_flat_floor_chunk,
        );
        self.world_clear_dirty();
        let total_voxels = Self::surface_voxel_count(&self.surface);
        eprintln!(
            "{}: {} chunks, {} surface voxels",
            label,
            self.surface.chunks.len(),
            total_voxels
        );
    }

    pub fn new(preset: ScenePreset) -> Self {
        let (base_kind, world_chunks_init) = Self::build_scene_preset_world(preset);
        let (world_tree, world_chunks, world_base_kind, world_flat_floor_chunk, world_dirty) =
            Self::scene_world_from_chunk_overrides(base_kind, world_chunks_init);

        let surface = cull::extract_surfaces(&world_tree, world_base_kind, &world_flat_floor_chunk);
        let scene = Self {
            world_tree,
            world_chunks,
            world_base_kind,
            world_flat_floor_chunk,
            world_dirty,
            world_pending_chunk_updates: Vec::new(),
            world_pending_chunk_update_set: HashSet::new(),
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
            voxel_chunk_payload_cache: HashMap::new(),
            voxel_chunk_payloads: Vec::new(),
            voxel_chunk_payload_free_ids: Vec::new(),
            voxel_chunk_payload_hash_buckets: HashMap::new(),
            voxel_payload_slot_to_payload: Vec::new(),
            voxel_payload_free_slots: Vec::new(),
            voxel_pending_payload_uploads: Vec::new(),
            voxel_pending_payload_upload_set: HashSet::new(),
            voxel_active_chunks: Vec::new(),
            voxel_active_chunk_indices: HashMap::new(),
            voxel_world_revision: 0,
            voxel_visibility_generation: 0,
            voxel_cached_visibility_camera_chunk: None,
            voxel_cached_visibility_world_revision: 0,
            voxel_payload_slot_overflow_logged: false,
            voxel_frame_data: VoxelFrameData {
                metadata_generation: 0,
                chunk_headers: Vec::new(),
                payload_update_slots: Vec::new(),
                occupancy_words: Vec::new(),
                material_words: Vec::new(),
                macro_words: Vec::new(),
                visible_chunk_indices: Vec::new(),
                y_slice_bounds: Vec::new(),
                y_slice_lookup_entries: Vec::new(),
            },
        };
        let total_voxels = Self::surface_voxel_count(&scene.surface);
        eprintln!(
            "Voxel surface ({}): {} chunks, {} surface voxels",
            preset.label(),
            scene.surface.chunks.len(),
            total_voxels
        );
        scene
    }

    /// Rebuild surface data if any chunk is dirty.
    pub fn update_surfaces_if_dirty(&mut self) {
        if self.world_any_dirty() {
            self.rebuild_surface("Voxel surface rebuilt");
        }
    }

    /// Per-frame: cull and build ModelInstances for the current camera position.
    pub fn build_instances(&mut self, cam_pos: [f32; 4]) -> &[common::ModelInstance] {
        self.culled_instances.clear();
        cull::cull_and_build(
            &self.surface,
            cam_pos,
            RENDER_DISTANCE,
            &mut self.culled_instances,
        );

        let (num_instances, num_tets) = cull::mesh_stats(&self.culled_instances);
        if self.cull_log_counter == 0 || self.cull_log_counter % 120 == 0 {
            eprintln!("Culled: {num_instances} instances, {num_tets} tetrahedra");
        }
        self.cull_log_counter = self.cull_log_counter.wrapping_add(1);

        &self.culled_instances
    }

    fn player_aabb(pos: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let min = [
            pos[0] - PLAYER_RADIUS_XZW,
            pos[1] - PLAYER_HEIGHT,
            pos[2] - PLAYER_RADIUS_XZW,
            pos[3] - PLAYER_RADIUS_XZW,
        ];
        let max = [
            pos[0] + PLAYER_RADIUS_XZW,
            pos[1],
            pos[2] + PLAYER_RADIUS_XZW,
            pos[3] + PLAYER_RADIUS_XZW,
        ];
        (min, max)
    }

    fn aabb_intersects_solid(&self, min: [f32; 4], max: [f32; 4]) -> bool {
        // Infinite hard floor: disallow moving below this world Y plane.
        if min[1] < HARD_WORLD_FLOOR_Y {
            return true;
        }

        let max_epsilon = 1e-4f32;
        let lo = [
            min[0].floor() as i32,
            min[1].floor() as i32,
            min[2].floor() as i32,
            min[3].floor() as i32,
        ];
        let hi = [
            (max[0] - max_epsilon).floor() as i32,
            (max[1] - max_epsilon).floor() as i32,
            (max[2] - max_epsilon).floor() as i32,
            (max[3] - max_epsilon).floor() as i32,
        ];
        if hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2] || hi[3] < lo[3] {
            return false;
        }

        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    for w in lo[3]..=hi[3] {
                        if self.world_get_voxel(x, y, z, w).is_solid() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn player_collides_at(&self, pos: [f32; 4]) -> bool {
        let (min, max) = Self::player_aabb(pos);
        self.aabb_intersects_solid(min, max)
    }

    fn depenetrate_upward(&self, pos: &mut [f32; 4]) {
        if !self.player_collides_at(*pos) {
            return;
        }
        for _ in 0..COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += COLLISION_PUSHUP_STEP;
            if !self.player_collides_at(*pos) {
                return;
            }
        }
    }

    /// Resolve player movement against solid voxels.
    /// Returns `(resolved_position, grounded)`.
    pub fn resolve_player_collision(
        &self,
        old_pos: [f32; 4],
        attempted_pos: [f32; 4],
        velocity_y: &mut f32,
    ) -> ([f32; 4], bool) {
        let mut pos = old_pos;
        self.depenetrate_upward(&mut pos);

        for axis in [0usize, 2, 3, 1] {
            let target = attempted_pos[axis];
            if (target - pos[axis]).abs() <= 1e-6 {
                continue;
            }

            let mut candidate = pos;
            candidate[axis] = target;
            if !self.player_collides_at(candidate) {
                pos = candidate;
                continue;
            }

            let mut feasible = pos[axis];
            let mut blocked = target;
            for _ in 0..COLLISION_BINARY_STEPS {
                let mid = 0.5 * (feasible + blocked);
                let mut probe = pos;
                probe[axis] = mid;
                if self.player_collides_at(probe) {
                    blocked = mid;
                } else {
                    feasible = mid;
                }
            }
            pos[axis] = feasible;

            if axis == 1 {
                *velocity_y = 0.0;
            }
        }

        let mut ground_probe = pos;
        ground_probe[1] -= 0.02;
        let grounded = self.player_collides_at(ground_probe);
        if grounded && *velocity_y < 0.0 {
            *velocity_y = 0.0;
        }

        (pos, grounded)
    }

    fn trace_first_solid_voxel(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<VoxelRayHit> {
        let dir_len_sq = ray_direction[0] * ray_direction[0]
            + ray_direction[1] * ray_direction[1]
            + ray_direction[2] * ray_direction[2]
            + ray_direction[3] * ray_direction[3];
        if dir_len_sq <= 1e-8 || max_distance <= 0.0 {
            return None;
        }

        let inv_dir_len = dir_len_sq.sqrt().recip();
        let dir = [
            ray_direction[0] * inv_dir_len,
            ray_direction[1] * inv_dir_len,
            ray_direction[2] * inv_dir_len,
            ray_direction[3] * inv_dir_len,
        ];
        let origin = [
            ray_origin[0] + dir[0] * EDIT_RAY_EPSILON,
            ray_origin[1] + dir[1] * EDIT_RAY_EPSILON,
            ray_origin[2] + dir[2] * EDIT_RAY_EPSILON,
            ray_origin[3] + dir[3] * EDIT_RAY_EPSILON,
        ];

        let mut cell = [
            origin[0].floor() as i32,
            origin[1].floor() as i32,
            origin[2].floor() as i32,
            origin[3].floor() as i32,
        ];

        let mut step = [0i32; 4];
        let mut t_max = [f32::INFINITY; 4];
        let mut t_delta = [f32::INFINITY; 4];

        for axis in 0..4 {
            let d = dir[axis];
            if d > 1e-8 {
                step[axis] = 1;
                let next_boundary = cell[axis] as f32 + 1.0;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = 1.0 / d;
            } else if d < -1e-8 {
                step[axis] = -1;
                let next_boundary = cell[axis] as f32;
                t_max[axis] = (next_boundary - origin[axis]) / d;
                t_delta[axis] = -1.0 / d;
            }
        }

        let mut last_empty_voxel = None;
        let v0 = self.world_get_voxel(cell[0], cell[1], cell[2], cell[3]);
        if v0.is_solid() {
            return Some(VoxelRayHit {
                solid_voxel: cell,
                last_empty_voxel,
            });
        }
        last_empty_voxel = Some(cell);

        for _ in 0..EDIT_RAY_MAX_STEPS {
            let mut axis = 0usize;
            for candidate in 1..4 {
                if t_max[candidate] < t_max[axis] {
                    axis = candidate;
                }
            }

            let traversed_t = t_max[axis];
            if !traversed_t.is_finite() || traversed_t > max_distance {
                break;
            }

            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];

            let voxel = self.world_get_voxel(cell[0], cell[1], cell[2], cell[3]);
            if voxel.is_solid() {
                return Some(VoxelRayHit {
                    solid_voxel: cell,
                    last_empty_voxel,
                });
            }
            last_empty_voxel = Some(cell);
        }

        None
    }

    /// Remove the first solid voxel intersected by a camera ray.
    pub fn remove_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> Option<[i32; 4]> {
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let [x, y, z, w] = hit.solid_voxel;
        self.world_set_voxel(x, y, z, w, VoxelType::AIR);
        Some(hit.solid_voxel)
    }

    /// Query edit ray targets without mutating the world.
    pub fn block_edit_targets(
        &self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
    ) -> BlockEditTargets {
        match self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance) {
            Some(hit) => BlockEditTargets {
                hit_voxel: Some(hit.solid_voxel),
                place_voxel: hit.last_empty_voxel,
            },
            None => BlockEditTargets::default(),
        }
    }

    /// Fan-cast across the ZW viewing wedge and return the nearest solid
    /// voxel hit.  `view_z` and `view_w` are the camera's world-space Z and W
    /// basis vectors (obtained from the view basis).  The sweep mirrors the VTE
    /// shader: theta ranges from `PI/4 - viewAngle/2` to `PI/4 + viewAngle/2`
    /// where `viewAngle = (PI/2) / focal_length_zw`.
    pub fn fan_cast_nearest_block(
        &self,
        ray_origin: [f32; 4],
        view_z: [f32; 4],
        view_w: [f32; 4],
        focal_length_zw: f32,
        max_distance: f32,
        num_samples: usize,
    ) -> Option<[i32; 4]> {
        let pi = std::f32::consts::PI;
        let view_angle = (pi / 2.0) / focal_length_zw.max(0.01);
        let theta_min = pi / 4.0 - view_angle / 2.0;
        let theta_max = pi / 4.0 + view_angle / 2.0;

        let samples = num_samples.max(1);
        let mut best_voxel: Option<[i32; 4]> = None;
        let mut best_dist_sq = f32::INFINITY;

        for i in 0..samples {
            let t = if samples == 1 {
                0.5
            } else {
                i as f32 / (samples - 1) as f32
            };
            let theta = theta_min + t * (theta_max - theta_min);
            let cz = theta.cos();
            let sw = theta.sin();

            let dir = [
                cz * view_z[0] + sw * view_w[0],
                cz * view_z[1] + sw * view_w[1],
                cz * view_z[2] + sw * view_w[2],
                cz * view_z[3] + sw * view_w[3],
            ];

            if let Some(hit) = self.trace_first_solid_voxel(ray_origin, dir, max_distance) {
                let dx = hit.solid_voxel[0] as f32 + 0.5 - ray_origin[0];
                let dy = hit.solid_voxel[1] as f32 + 0.5 - ray_origin[1];
                let dz = hit.solid_voxel[2] as f32 + 0.5 - ray_origin[2];
                let dw = hit.solid_voxel[3] as f32 + 0.5 - ray_origin[3];
                let dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_voxel = Some(hit.solid_voxel);
                }
            }
        }

        best_voxel
    }

    /// Place a voxel in the last empty cell before the first solid hit.
    pub fn place_block_along_ray(
        &mut self,
        ray_origin: [f32; 4],
        ray_direction: [f32; 4],
        max_distance: f32,
        material: VoxelType,
    ) -> Option<[i32; 4]> {
        if material.is_air() {
            return None;
        }
        let hit = self.trace_first_solid_voxel(ray_origin, ray_direction, max_distance)?;
        let target = hit.last_empty_voxel?;
        let [x, y, z, w] = target;
        if self.world_get_voxel(x, y, z, w).is_solid() {
            return None;
        }
        self.world_set_voxel(x, y, z, w, material);
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene_with_voxels(voxels: &[([i32; 4], VoxelType)]) -> Scene {
        let mut scene = Scene::new(ScenePreset::Empty);
        for ([x, y, z, w], material) in voxels.iter().copied() {
            scene.world_set_voxel(x, y, z, w, material);
        }
        scene.update_surfaces_if_dirty();
        scene
    }

    #[test]
    fn remove_block_along_ray_hits_first_solid_voxel() {
        let mut scene = make_scene_with_voxels(&[([0, 0, 0, 0], VoxelType(7))]);

        let removed =
            scene.remove_block_along_ray([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

        assert_eq!(removed, Some([0, 0, 0, 0]));
        assert!(scene.get_voxel(0, 0, 0, 0).is_air());
    }

    #[test]
    fn place_block_along_ray_places_in_last_empty_before_hit() {
        let mut scene = make_scene_with_voxels(&[([0, 0, 0, 0], VoxelType(7))]);

        let placed = scene.place_block_along_ray(
            [0.5, 0.5, -2.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            8.0,
            VoxelType(3),
        );

        assert_eq!(placed, Some([0, 0, -1, 0]));
        assert!(scene.get_voxel(0, 0, -1, 0) == VoxelType(3));
        assert!(scene.get_voxel(0, 0, 0, 0) == VoxelType(7));
    }

    #[test]
    fn block_edit_targets_reports_hit_and_placement_voxels() {
        let scene = make_scene_with_voxels(&[([0, 0, 0, 0], VoxelType(7))]);

        let targets = scene.block_edit_targets([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

        assert_eq!(
            targets,
            BlockEditTargets {
                hit_voxel: Some([0, 0, 0, 0]),
                place_voxel: Some([0, 0, -1, 0]),
            }
        );
    }

    #[test]
    fn resolve_player_collision_lands_on_voxel_surface() {
        let scene = make_scene_with_voxels(&[([0, -1, 0, 0], VoxelType(3))]);

        let old_pos = [0.5, 2.4, 0.5, 0.5];
        let attempted = [0.5, 1.1, 0.5, 0.5];
        let mut velocity_y = -5.0;
        let (resolved, grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        assert!(grounded);
        assert!((resolved[1] - PLAYER_HEIGHT).abs() < 0.02);
        assert_eq!(velocity_y, 0.0);
    }

    #[test]
    fn resolve_player_collision_blocks_horizontal_motion() {
        let scene = make_scene_with_voxels(&[([1, 0, 0, 0], VoxelType(3))]);

        let old_pos = [0.35, 1.2, 0.5, 0.5];
        let attempted = [1.6, 1.2, 0.5, 0.5];
        let mut velocity_y = 0.0;
        let (resolved, _grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        // Player radius keeps center below x=0.7 when blocked by voxel slab at x>=1.
        assert!(resolved[0] < 0.72);
    }

    #[test]
    fn resolve_player_collision_lands_on_hard_world_floor() {
        let scene = make_scene_with_voxels(&[]);

        let old_pos = [0.0, 0.4, 0.0, 0.0];
        let attempted = [0.0, -6.0, 0.0, 0.0];
        let mut velocity_y = -9.0;
        let (resolved, grounded) =
            scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

        assert!(grounded);
        assert!((resolved[1] - (HARD_WORLD_FLOOR_Y + PLAYER_HEIGHT)).abs() < 0.03);
        assert_eq!(velocity_y, 0.0);
    }

    #[test]
    fn edit_back_to_air_rebuilds_surface_for_clean_world() {
        let mut scene = make_scene_with_voxels(&[([0, 0, 0, 0], VoxelType(3))]);
        assert!(!scene.surface.chunks.is_empty());

        scene.world_set_voxel(0, 0, 0, 0, VoxelType::AIR);
        scene.update_surfaces_if_dirty();
        assert!(scene.surface.chunks.is_empty());
    }

    #[test]
    fn apply_near_lod_region_patch_splices_chunk_and_queues_update() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut patch_tree = RegionChunkTree::new();
        let changed = patch_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(7)));
        assert!(changed);
        let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

        let stats = scene.apply_near_lod_region_patch(bounds, &patch_core);

        assert_eq!(stats.previous_non_empty, 0);
        assert_eq!(stats.desired_non_empty, 1);
        assert_eq!(stats.upserts, 1);
        assert_eq!(stats.removals, 0);
        assert_eq!(stats.changed_chunks, 1);
        assert_eq!(stats.previous_total_chunks, 0);
        assert_eq!(stats.desired_total_chunks, 1);
        assert_eq!(stats.queued_updates, 1);
        assert!(stats.collect_previous_ms >= 0.0);
        assert!(stats.splice_ms >= 0.0);
        assert!(stats.collect_desired_ms >= 0.0);
        assert!(stats.diff_ms >= 0.0);
        assert_eq!(scene.get_voxel(0, 0, 0, 0), VoxelType(7));
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            vec![ChunkPos::new(0, 0, 0, 0)]
        );
    }

    #[test]
    fn apply_near_lod_region_patch_removes_chunk_and_queues_update() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut seed_tree = RegionChunkTree::new();
        let _ = seed_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(5)));
        let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = scene.apply_near_lod_region_patch(bounds, &seed_core);
        let _ = scene.world_drain_pending_chunk_updates();

        let empty_core = RegionTreeCore {
            bounds,
            kind: polychora::shared::region_tree::RegionNodeKind::Empty,
            generator_version_hash: 0,
        };
        let stats = scene.apply_near_lod_region_patch(bounds, &empty_core);

        assert_eq!(stats.previous_non_empty, 1);
        assert_eq!(stats.desired_non_empty, 0);
        assert_eq!(stats.upserts, 0);
        assert_eq!(stats.removals, 1);
        assert_eq!(stats.changed_chunks, 1);
        assert_eq!(stats.previous_total_chunks, 1);
        assert_eq!(stats.desired_total_chunks, 0);
        assert_eq!(stats.queued_updates, 1);
        assert!(stats.collect_previous_ms >= 0.0);
        assert!(stats.splice_ms >= 0.0);
        assert!(stats.collect_desired_ms >= 0.0);
        assert!(stats.diff_ms >= 0.0);
        assert!(scene.get_voxel(0, 0, 0, 0).is_air());
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            vec![ChunkPos::new(0, 0, 0, 0)]
        );
    }

    #[test]
    fn apply_near_lod_region_patch_identical_patch_is_noop() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut patch_tree = RegionChunkTree::new();
        let _ = patch_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(7)));
        let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

        let _ = scene.apply_near_lod_region_patch(bounds, &patch_core);
        let _ = scene.world_drain_pending_chunk_updates();

        let stats = scene.apply_near_lod_region_patch(bounds, &patch_core);
        assert_eq!(stats.previous_non_empty, 1);
        assert_eq!(stats.desired_non_empty, 1);
        assert_eq!(stats.upserts, 0);
        assert_eq!(stats.removals, 0);
        assert_eq!(stats.changed_chunks, 0);
        assert_eq!(stats.previous_total_chunks, 1);
        assert_eq!(stats.desired_total_chunks, 1);
        assert_eq!(stats.invalidated_cached_chunks, 0);
        assert_eq!(stats.queued_updates, 0);
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            Vec::<ChunkPos>::new()
        );
    }

    #[test]
    fn apply_near_lod_region_patch_semantic_noop_skips_splice() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);

        let mut seed_tree = RegionChunkTree::new();
        let _ = seed_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(3)));
        let _ = seed_tree.set_chunk([1, 0, 0, 0], Some(ChunkPayload::Uniform(4)));
        let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = scene.apply_near_lod_region_patch(bounds, &seed_core);
        let _ = scene.world_drain_pending_chunk_updates();
        let before = scene.world_tree.root().cloned();

        let patch_chunk_array =
            polychora::shared::chunk_payload::ChunkArrayData::from_dense_indices(
                bounds,
                vec![ChunkPayload::Uniform(3), ChunkPayload::Uniform(4)],
                vec![0, 1],
                None,
            )
            .expect("chunk array");
        let patch_core = RegionTreeCore {
            bounds,
            kind: polychora::shared::region_tree::RegionNodeKind::ChunkArray(patch_chunk_array),
            generator_version_hash: 0,
        };

        let stats = scene.apply_near_lod_region_patch(bounds, &patch_core);
        assert_eq!(stats.changed_chunks, 0);
        assert_eq!(stats.upserts, 0);
        assert_eq!(stats.removals, 0);
        assert_eq!(stats.queued_updates, 0);
        assert_eq!(stats.invalidated_cached_chunks, 0);
        assert_eq!(stats.splice_ms, 0.0);
        assert_eq!(scene.world_tree.root().cloned(), before);
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            Vec::<ChunkPos>::new()
        );
    }
}
