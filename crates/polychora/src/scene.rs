use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkBvhNode, GpuVoxelChunkHeader, GpuVoxelLeafHeader, VoxelFrameDirtyRanges,
    VoxelFrameInput, VoxelMutationBatch, VTE_MAX_DENSE_CHUNKS, VTE_REGION_BVH_INVALID_NODE,
    VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY,
};
use polychora::shared::chunk_payload::ChunkPayload;
use polychora::shared::region_tree::{
    chunk_key_from_chunk_pos, slice_non_empty_region_core_in_bounds, RegionChunkTree,
    RegionNodeKind, RegionTreeCore,
};
use polychora::shared::render_tree::{self, RenderBvhChunkMutationDelta, RenderTreeCore};
use polychora::shared::spatial::Aabb4i;
use polychora::shared::voxel::world_to_chunk;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

mod collision;
mod editing;
mod patching;
mod render_cache;
mod voxel_runtime;

const RENDER_DISTANCE: f32 = 64.0;
const VOXEL_NEAR_ACTIVE_DISTANCE: f32 = 32.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 4; // packed 4x u8 per u32
const MACRO_CELLS_PER_AXIS: usize = CHUNK_SIZE / 2; // 2x2x2x2 macro cells
const MACRO_CELLS_PER_CHUNK: usize =
    MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS;
const MACRO_WORDS_PER_CHUNK: usize = MACRO_CELLS_PER_CHUNK / 32;
const EDIT_RAY_EPSILON: f32 = 1e-4;
const EDIT_RAY_MAX_STEPS: usize = 4096;
const VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET: usize = 4;
const COLLISION_PUSHUP_STEP: f32 = 0.05;
const COLLISION_MAX_PUSHUP_STEPS: usize = 80;
const COLLISION_BINARY_STEPS: usize = 14;
const HARD_WORLD_FLOOR_Y: f32 = -4.0;
const FLAT_FLOOR_CHUNK_Y: i32 = -1;
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
    pub region_bvh_root_index: u32,
    pub dirty_ranges: VoxelFrameDirtyRanges,
    pub mutation_batch: Option<VoxelMutationBatch>,
    pub chunk_headers: Vec<GpuVoxelChunkHeader>,
    pub occupancy_words: Vec<u32>,
    pub material_words: Vec<u32>,
    pub macro_words: Vec<u32>,
    pub region_bvh_nodes: Vec<GpuVoxelChunkBvhNode>,
    pub leaf_headers: Vec<GpuVoxelLeafHeader>,
    pub leaf_chunk_entries: Vec<u32>,
}

struct VoxelFrameDataBuffers {
    region_bvh_root_index: u32,
    dense_payload_encoded_cache: HashMap<ChunkPayload, u32>,
    chunk_headers: Vec<GpuVoxelChunkHeader>,
    occupancy_words: Vec<u32>,
    material_words: Vec<u32>,
    macro_words: Vec<u32>,
    region_bvh_nodes: Vec<GpuVoxelChunkBvhNode>,
    leaf_headers: Vec<GpuVoxelLeafHeader>,
    leaf_chunk_entries: Vec<u32>,
}

impl VoxelFrameData {
    pub fn as_input(&self) -> VoxelFrameInput<'_> {
        VoxelFrameInput {
            metadata_generation: self.metadata_generation,
            region_bvh_root_index: self.region_bvh_root_index,
            chunk_headers: &self.chunk_headers,
            occupancy_words: &self.occupancy_words,
            material_words: &self.material_words,
            macro_words: &self.macro_words,
            region_bvh_nodes: &self.region_bvh_nodes,
            leaf_headers: &self.leaf_headers,
            leaf_chunk_entries: &self.leaf_chunk_entries,
            mutation_batch: self.mutation_batch.as_ref(),
            dirty_ranges: Some(&self.dirty_ranges),
        }
    }
}

pub struct Scene {
    world_tree: RegionChunkTree,
    world_tree_revision: u64,
    // Dense decode cache for local edits/collision/surface extraction. The
    // authoritative world state is always `world_tree`.
    world_chunks: HashMap<ChunkPos, DenseChunk>,
    world_dirty: bool,
    world_pending_chunk_updates: Vec<ChunkPos>,
    world_pending_chunk_update_set: HashSet<ChunkPos>,
    surface: SurfaceData,
    culled_instances: Vec<common::ModelInstance>,
    cull_log_counter: u64,
    voxel_visibility_generation: u64,
    voxel_cached_visibility_bounds: Option<Aabb4i>,
    voxel_pending_scene_dirty_regions: Vec<Aabb4i>,
    render_region_cache_bounds: Option<Aabb4i>,
    render_region_cache: Option<RegionChunkTree>,
    render_bvh_cache_bounds: Option<Aabb4i>,
    render_bvh_cache: Option<render_tree::RenderBvh>,
    voxel_pending_render_bvh_rebuild: bool,
    voxel_pending_render_bvh_mutation_deltas: Vec<RenderBvhChunkMutationDelta>,
    voxel_dense_payload_encoded_cache: HashMap<ChunkPayload, u32>,
    voxel_leaf_entry_spans: Vec<Option<std::ops::Range<usize>>>,
    voxel_leaf_entry_free_spans: Vec<std::ops::Range<usize>>,
    voxel_last_rebuild_failure_signature: Option<(Aabb4i, usize, usize)>,
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
pub struct RegionPatchStats {
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

    fn build_scene_preset_world(preset: ScenePreset) -> HashMap<ChunkPos, DenseChunk> {
        let mut chunks = HashMap::<ChunkPos, DenseChunk>::new();

        match preset {
            ScenePreset::Empty => {}
            ScenePreset::Flat => {
                Self::place_material_showcase(&mut chunks, [-10, 0, -14, -4]);
                let floor_chunk = Self::build_flat_floor_chunk(FLAT_PRESET_FLOOR_MATERIAL);
                for x in -2..=2 {
                    for z in -2..=2 {
                        for w in -2..=2 {
                            chunks.insert(
                                ChunkPos {
                                    x,
                                    y: FLAT_FLOOR_CHUNK_Y,
                                    z,
                                    w,
                                },
                                floor_chunk,
                            );
                        }
                    }
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
            }
        };

        chunks
    }

    fn scene_world_from_chunks(
        mut world_chunks: HashMap<ChunkPos, DenseChunk>,
    ) -> (RegionChunkTree, HashMap<ChunkPos, DenseChunk>, bool) {
        world_chunks.retain(|_, chunk| !dense_chunk_is_empty(chunk));
        let world_tree = RegionChunkTree::from_chunks(world_chunks.iter().map(|(&pos, chunk)| {
            (
                chunk_key_from_chunk_pos(pos),
                dense_chunk_to_payload_compact(chunk),
            )
        }));
        (world_tree, world_chunks, false)
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

    fn world_queue_chunk_update(&mut self, pos: ChunkPos) -> bool {
        if self.world_pending_chunk_update_set.insert(pos) {
            self.world_pending_chunk_updates.push(pos);
            true
        } else {
            false
        }
    }

    fn world_set_chunk(&mut self, pos: ChunkPos, chunk: Option<DenseChunk>) -> bool {
        let key = chunk_key_from_chunk_pos(pos);
        let previous_payload = self.world_tree.chunk_payload(key);
        let payload = match chunk {
            Some(chunk) => {
                if dense_chunk_is_empty(&chunk) {
                    self.world_chunks.remove(&pos);
                    None
                } else {
                    self.world_chunks.insert(pos, chunk);
                    Some(dense_chunk_to_payload_compact(&chunk))
                }
            }
            None => {
                self.world_chunks.remove(&pos);
                None
            }
        };
        let changed_by_api = self.world_tree.set_chunk(key, payload.clone());
        let current_payload = self.world_tree.chunk_payload(key);
        let mut changed = changed_by_api || previous_payload != current_payload;
        if !changed {
            if current_payload != payload {
                let bounds = Aabb4i::new(key, key);
                let mut repair_tree = RegionChunkTree::new();
                if let Some(repair_payload) = payload.clone() {
                    let _ = repair_tree.set_chunk(key, Some(repair_payload));
                }
                let repair_core = repair_tree.root().cloned().unwrap_or(RegionTreeCore {
                    bounds,
                    kind: RegionNodeKind::Empty,
                    generator_version_hash: 0,
                });
                if self
                    .world_tree
                    .splice_non_empty_core_in_bounds(bounds, &repair_core)
                    .is_some()
                {
                    changed = true;
                    eprintln!(
                        "[scene-world-repair] forced chunk repair at {:?} (expected={:?} current={:?})",
                        key, payload, current_payload
                    );
                }
            }
        }
        if changed {
            self.world_tree_revision = self.world_tree_revision.wrapping_add(1);
            self.world_dirty = true;
            self.mark_voxel_scene_region_dirty(Aabb4i::new(key, key));
            let _ = self.world_queue_chunk_update(pos);
        }
        changed
    }

    fn world_chunk_at(&mut self, pos: ChunkPos) -> Option<DenseChunk> {
        let chunk_key = chunk_key_from_chunk_pos(pos);
        if let Some(payload) = self.world_tree.chunk_payload(chunk_key) {
            if let Some(chunk) = self.world_chunks.get(&pos) {
                return (!dense_chunk_is_empty(chunk)).then_some(*chunk);
            }
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
        if self.world_chunks.remove(&pos).is_some() {
            eprintln!(
                "[scene-world-cache-repair] dropped orphaned cached chunk at ({}, {}, {}, {})",
                pos.x, pos.y, pos.z, pos.w
            );
        }
        None
    }

    fn world_get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        if let Some(payload) = self.world_tree.chunk_payload(chunk_key_from_chunk_pos(cp)) {
            if let Some(chunk) = self.world_chunks.get(&cp) {
                return chunk[idx];
            }
            return Self::chunk_payload_to_voxel_type(&payload, idx);
        }
        VoxelType::AIR
    }

    fn world_set_voxel(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, v: VoxelType) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        let old = self.world_get_voxel(wx, wy, wz, ww);
        if old == v {
            return;
        }

        let mut chunk = self.world_chunk_at(cp).unwrap_or_else(empty_dense_chunk);
        set_chunk_voxel_by_index(&mut chunk, idx, v);
        let _ = self.world_set_chunk(cp, Some(chunk));
    }

    #[cfg(test)]
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

    pub fn debug_world_tree_chunk_payload(&self, chunk_key: [i32; 4]) -> Option<ChunkPayload> {
        self.world_tree.chunk_payload(chunk_key)
    }

    pub fn debug_world_tree_root_bounds(&self) -> Option<Aabb4i> {
        self.world_tree.root().map(|root| root.bounds)
    }

    pub fn debug_voxel_frame_buffer_lengths(&self) -> (usize, usize, usize, usize) {
        (
            self.voxel_frame_data.chunk_headers.len(),
            self.voxel_frame_data.leaf_headers.len(),
            self.voxel_frame_data.region_bvh_nodes.len(),
            self.voxel_frame_data.leaf_chunk_entries.len(),
        )
    }

    fn decode_voxel_frame_dense_chunk_materials(&self, chunk_index: usize) -> Option<Vec<u16>> {
        let header = self.voxel_frame_data.chunk_headers.get(chunk_index)?;
        let mut out = vec![0u16; CHUNK_VOLUME];
        let is_full = (header.flags & GpuVoxelChunkHeader::FLAG_FULL) != 0;
        for voxel_idx in 0..CHUNK_VOLUME {
            if !is_full {
                let occ_word_idx = header.occupancy_word_offset as usize + (voxel_idx / 32);
                let occ_word = *self.voxel_frame_data.occupancy_words.get(occ_word_idx)?;
                if (occ_word & (1u32 << (voxel_idx % 32))) == 0 {
                    continue;
                }
            }
            let mat_word_idx = header.material_word_offset as usize + (voxel_idx / 4);
            let mat_word = *self.voxel_frame_data.material_words.get(mat_word_idx)?;
            let material = ((mat_word >> ((voxel_idx % 4) * 8)) & 0xFF) as u16;
            out[voxel_idx] = material;
        }
        Some(out)
    }

    pub fn debug_voxel_frame_chunk_payloads(&self, chunk_key: [i32; 4]) -> Vec<ChunkPayload> {
        let mut out = Vec::<ChunkPayload>::new();
        for leaf in &self.voxel_frame_data.leaf_headers {
            if chunk_key[0] < leaf.min_chunk_coord[0]
                || chunk_key[0] > leaf.max_chunk_coord[0]
                || chunk_key[1] < leaf.min_chunk_coord[1]
                || chunk_key[1] > leaf.max_chunk_coord[1]
                || chunk_key[2] < leaf.min_chunk_coord[2]
                || chunk_key[2] > leaf.max_chunk_coord[2]
                || chunk_key[3] < leaf.min_chunk_coord[3]
                || chunk_key[3] > leaf.max_chunk_coord[3]
            {
                continue;
            }

            if leaf.leaf_kind == higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM {
                if leaf.uniform_material == 0 {
                    out.push(ChunkPayload::Empty);
                } else {
                    out.push(ChunkPayload::Uniform(leaf.uniform_material as u16));
                }
                continue;
            }
            if leaf.leaf_kind
                != higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY
            {
                continue;
            }

            let dim_x = (leaf.max_chunk_coord[0] - leaf.min_chunk_coord[0] + 1).max(0) as usize;
            let dim_y = (leaf.max_chunk_coord[1] - leaf.min_chunk_coord[1] + 1).max(0) as usize;
            let dim_z = (leaf.max_chunk_coord[2] - leaf.min_chunk_coord[2] + 1).max(0) as usize;
            let dim_w = (leaf.max_chunk_coord[3] - leaf.min_chunk_coord[3] + 1).max(0) as usize;
            if dim_x == 0 || dim_y == 0 || dim_z == 0 || dim_w == 0 {
                continue;
            }
            let local_x = (chunk_key[0] - leaf.min_chunk_coord[0]) as usize;
            let local_y = (chunk_key[1] - leaf.min_chunk_coord[1]) as usize;
            let local_z = (chunk_key[2] - leaf.min_chunk_coord[2]) as usize;
            let local_w = (chunk_key[3] - leaf.min_chunk_coord[3]) as usize;
            let linear = local_x + dim_x * (local_y + dim_y * (local_z + dim_z * local_w));
            let entry_index = leaf.chunk_entry_offset as usize + linear;
            let Some(&entry) = self.voxel_frame_data.leaf_chunk_entries.get(entry_index) else {
                continue;
            };

            if entry == higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY {
                out.push(ChunkPayload::Empty);
                continue;
            }
            if (entry & higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0
            {
                let material = (entry
                    & !higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG)
                    as u16;
                if material == 0 {
                    out.push(ChunkPayload::Empty);
                } else {
                    out.push(ChunkPayload::Uniform(material));
                }
                continue;
            }

            let chunk_index = entry.saturating_sub(1) as usize;
            if let Some(materials) = self.decode_voxel_frame_dense_chunk_materials(chunk_index) {
                out.push(
                    ChunkPayload::from_dense_materials_compact(&materials)
                        .unwrap_or(ChunkPayload::Dense16 { materials }),
                );
            }
        }
        out
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

    fn surface_voxel_count(surface: &SurfaceData) -> u32 {
        surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum()
    }

    fn rebuild_surface(&mut self, label: &str) {
        self.surface = cull::extract_surfaces(&self.world_tree);
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
        let world_chunks_init = Self::build_scene_preset_world(preset);
        let (world_tree, world_chunks, world_dirty) =
            Self::scene_world_from_chunks(world_chunks_init);

        let surface = cull::extract_surfaces(&world_tree);
        let scene = Self {
            world_tree,
            world_tree_revision: 1,
            world_chunks,
            world_dirty,
            world_pending_chunk_updates: Vec::new(),
            world_pending_chunk_update_set: HashSet::new(),
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
            voxel_visibility_generation: 0,
            voxel_cached_visibility_bounds: None,
            voxel_pending_scene_dirty_regions: Vec::new(),
            render_region_cache_bounds: None,
            render_region_cache: None,
            render_bvh_cache_bounds: None,
            render_bvh_cache: None,
            voxel_pending_render_bvh_rebuild: true,
            voxel_pending_render_bvh_mutation_deltas: Vec::new(),
            voxel_dense_payload_encoded_cache: HashMap::new(),
            voxel_leaf_entry_spans: Vec::new(),
            voxel_leaf_entry_free_spans: Vec::new(),
            voxel_last_rebuild_failure_signature: None,
            voxel_frame_data: VoxelFrameData {
                metadata_generation: 0,
                region_bvh_root_index:
                    higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                dirty_ranges: VoxelFrameDirtyRanges::default(),
                mutation_batch: None,
                chunk_headers: Vec::new(),
                occupancy_words: Vec::new(),
                material_words: Vec::new(),
                macro_words: Vec::new(),
                region_bvh_nodes: Vec::new(),
                leaf_headers: Vec::new(),
                leaf_chunk_entries: Vec::new(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_voxel_from_frame(scene: &Scene, wx: i32, wy: i32, wz: i32, ww: i32) -> Option<u8> {
        let (chunk_pos, voxel_idx) = world_to_chunk(wx, wy, wz, ww);
        let chunk_key = chunk_key_from_chunk_pos(chunk_pos);

        for leaf in &scene.voxel_frame_data.leaf_headers {
            if chunk_key[0] < leaf.min_chunk_coord[0]
                || chunk_key[0] > leaf.max_chunk_coord[0]
                || chunk_key[1] < leaf.min_chunk_coord[1]
                || chunk_key[1] > leaf.max_chunk_coord[1]
                || chunk_key[2] < leaf.min_chunk_coord[2]
                || chunk_key[2] > leaf.max_chunk_coord[2]
                || chunk_key[3] < leaf.min_chunk_coord[3]
                || chunk_key[3] > leaf.max_chunk_coord[3]
            {
                continue;
            }

            if leaf.leaf_kind == higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM {
                let material = leaf.uniform_material as u8;
                return (material != 0).then_some(material);
            }
            if leaf.leaf_kind
                != higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY
            {
                continue;
            }

            let dim_x = (leaf.max_chunk_coord[0] - leaf.min_chunk_coord[0] + 1) as usize;
            let dim_y = (leaf.max_chunk_coord[1] - leaf.min_chunk_coord[1] + 1) as usize;
            let dim_z = (leaf.max_chunk_coord[2] - leaf.min_chunk_coord[2] + 1) as usize;
            let local_x = (chunk_key[0] - leaf.min_chunk_coord[0]) as usize;
            let local_y = (chunk_key[1] - leaf.min_chunk_coord[1]) as usize;
            let local_z = (chunk_key[2] - leaf.min_chunk_coord[2]) as usize;
            let local_w = (chunk_key[3] - leaf.min_chunk_coord[3]) as usize;
            let linear = local_x + dim_x * (local_y + dim_y * (local_z + dim_z * local_w));
            let entry_index = leaf.chunk_entry_offset as usize + linear;
            let Some(&entry) = scene.voxel_frame_data.leaf_chunk_entries.get(entry_index) else {
                return None;
            };

            if entry == higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY {
                return None;
            }
            if (entry & higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0
            {
                let material = (entry
                    & !higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG)
                    as u8;
                return (material != 0).then_some(material);
            }

            let chunk_index = entry.saturating_sub(1) as usize;
            let Some(header) = scene.voxel_frame_data.chunk_headers.get(chunk_index) else {
                return None;
            };
            if (header.flags & GpuVoxelChunkHeader::FLAG_FULL) == 0 {
                let occ_word_index = header.occupancy_word_offset as usize + (voxel_idx / 32);
                let Some(&occ_word) = scene.voxel_frame_data.occupancy_words.get(occ_word_index)
                else {
                    return None;
                };
                if (occ_word & (1u32 << (voxel_idx % 32))) == 0 {
                    return None;
                }
            }
            let mat_word_index = header.material_word_offset as usize + (voxel_idx / 4);
            let Some(&mat_word) = scene.voxel_frame_data.material_words.get(mat_word_index) else {
                return None;
            };
            let material = ((mat_word >> ((voxel_idx % 4) * 8)) & 0xFF) as u8;
            return (material != 0).then_some(material);
        }

        None
    }

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
    fn apply_region_patch_splices_chunk_and_queues_update() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut patch_tree = RegionChunkTree::new();
        let changed = patch_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(7)));
        assert!(changed);
        let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

        let stats = scene.apply_region_patch(bounds, &patch_core);

        assert_eq!(stats.previous_non_empty, 0);
        assert_eq!(stats.desired_non_empty, 1);
        assert_eq!(stats.changed_chunks, 1);
        assert_eq!(stats.previous_total_chunks, 0);
        assert_eq!(stats.desired_total_chunks, 1);
        assert!(stats.collect_previous_ms >= 0.0);
        assert!(stats.splice_ms >= 0.0);
        assert!(stats.collect_desired_ms >= 0.0);
        assert!(stats.diff_ms >= 0.0);
        assert_eq!(scene.get_voxel(0, 0, 0, 0), VoxelType(7));
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            Vec::<ChunkPos>::new()
        );
    }

    #[test]
    fn apply_region_patch_removes_chunk_and_queues_update() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut seed_tree = RegionChunkTree::new();
        let _ = seed_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(5)));
        let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = scene.apply_region_patch(bounds, &seed_core);
        let _ = scene.world_drain_pending_chunk_updates();

        let empty_core = RegionTreeCore {
            bounds,
            kind: polychora::shared::region_tree::RegionNodeKind::Empty,
            generator_version_hash: 0,
        };
        let stats = scene.apply_region_patch(bounds, &empty_core);

        assert_eq!(stats.previous_non_empty, 1);
        assert_eq!(stats.desired_non_empty, 0);
        assert_eq!(stats.changed_chunks, 1);
        assert_eq!(stats.previous_total_chunks, 1);
        assert_eq!(stats.desired_total_chunks, 0);
        assert!(stats.collect_previous_ms >= 0.0);
        assert!(stats.splice_ms >= 0.0);
        assert!(stats.collect_desired_ms >= 0.0);
        assert!(stats.diff_ms >= 0.0);
        assert!(scene.get_voxel(0, 0, 0, 0).is_air());
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            Vec::<ChunkPos>::new()
        );
    }

    #[test]
    fn apply_region_patch_identical_patch_is_noop() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let mut patch_tree = RegionChunkTree::new();
        let _ = patch_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(7)));
        let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

        let _ = scene.apply_region_patch(bounds, &patch_core);
        let _ = scene.world_drain_pending_chunk_updates();

        let stats = scene.apply_region_patch(bounds, &patch_core);
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
    fn apply_region_patch_semantic_noop_skips_splice() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);

        let mut seed_tree = RegionChunkTree::new();
        let _ = seed_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(3)));
        let _ = seed_tree.set_chunk([1, 0, 0, 0], Some(ChunkPayload::Uniform(4)));
        let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = scene.apply_region_patch(bounds, &seed_core);
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

        let stats = scene.apply_region_patch(bounds, &patch_core);
        assert_eq!(stats.changed_chunks, 0);
        assert_eq!(stats.upserts, 0);
        assert_eq!(stats.removals, 0);
        assert_eq!(stats.queued_updates, 0);
        assert_eq!(stats.invalidated_cached_chunks, 0);
        assert!(stats.splice_ms >= 0.0);
        assert_eq!(scene.world_tree.root().cloned(), before);
        assert_eq!(
            scene.world_drain_pending_chunk_updates(),
            Vec::<ChunkPos>::new()
        );
    }

    #[test]
    fn apply_region_patch_fast_matches_full_splice_state() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let mut full_scene = Scene::new(ScenePreset::Empty);
        let mut fast_scene = Scene::new(ScenePreset::Empty);

        let mut seed_tree = RegionChunkTree::new();
        let _ = seed_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(3)));
        let _ = seed_tree.set_chunk([1, 0, 0, 0], Some(ChunkPayload::Uniform(4)));
        let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = full_scene.apply_region_patch(bounds, &seed_core);
        let _ = fast_scene.apply_region_patch(bounds, &seed_core);
        let _ = full_scene.world_drain_pending_chunk_updates();
        let _ = fast_scene.world_drain_pending_chunk_updates();

        let mut patch_tree = RegionChunkTree::new();
        let _ = patch_tree.set_chunk([0, 0, 0, 0], Some(ChunkPayload::Uniform(7)));
        let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

        let full_stats = full_scene.apply_region_patch(bounds, &patch_core);
        let fast_stats = fast_scene.apply_region_patch_fast(bounds, &patch_core);

        assert_eq!(full_stats.changed_chunks, 1);
        assert_eq!(fast_stats.changed_chunks, 1);
        assert_eq!(full_scene.world_tree.root(), fast_scene.world_tree.root());
        assert_eq!(
            full_scene.world_tree_revision,
            fast_scene.world_tree_revision
        );
        assert_eq!(
            full_scene.world_drain_pending_chunk_updates(),
            fast_scene.world_drain_pending_chunk_updates()
        );
        assert_eq!(full_scene.get_voxel(0, 0, 0, 0), VoxelType(7));
        assert_eq!(fast_scene.get_voxel(0, 0, 0, 0), VoxelType(7));
        assert!(full_scene.get_voxel(CHUNK_SIZE as i32, 0, 0, 0).is_air());
        assert!(fast_scene.get_voxel(CHUNK_SIZE as i32, 0, 0, 0).is_air());
    }

    #[test]
    fn voxel_scene_dirty_tracking_rebuild_clears_only_overlapping_chunks() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([-2, -2, -2, -2], [2, 2, 2, 2]);

        // Prime cache.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        assert_eq!(scene.render_bvh_cache_bounds, Some(bounds));
        assert!(scene.voxel_pending_scene_dirty_regions.is_empty());

        // Add one dirty chunk inside bounds and one far outside.
        scene.world_set_voxel(0, 0, 0, 0, VoxelType(3));
        let far_world_x = (CHUNK_SIZE as i32) * 50;
        scene.world_set_voxel(far_world_x, 0, 0, 0, VoxelType(4));

        let far_key = [50, 0, 0, 0];
        assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), 2);
        assert!(scene
            .voxel_pending_scene_dirty_regions
            .iter()
            .any(|region| region.contains_chunk([0, 0, 0, 0])));
        assert!(scene
            .voxel_pending_scene_dirty_regions
            .iter()
            .any(|region| region.contains_chunk(far_key)));

        // Rebuild for local bounds should consume local dirty key only.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), 1);
        assert!(scene
            .voxel_pending_scene_dirty_regions
            .iter()
            .any(|region| region.contains_chunk(far_key)));
    }

    #[test]
    fn voxel_scene_dirty_tracking_offscreen_edits_do_not_invalidate_local_cache() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([-2, -2, -2, -2], [2, 2, 2, 2]);

        // Prime cache once.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        assert_eq!(scene.render_bvh_cache_bounds, Some(bounds));

        // Edit a far chunk; local bounds should remain cache-valid.
        let far_world_x = (CHUNK_SIZE as i32) * 60;
        scene.world_set_voxel(far_world_x, 0, 0, 0, VoxelType(5));
        let far_key = [60, 0, 0, 0];
        assert!(scene
            .voxel_pending_scene_dirty_regions
            .iter()
            .any(|region| region.contains_chunk(far_key)));
        assert!(!scene.voxel_scene_bounds_has_pending_dirty_regions(bounds));

        // Re-requesting local cache should keep far dirty queued.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        assert!(scene
            .voxel_pending_scene_dirty_regions
            .iter()
            .any(|region| region.contains_chunk(far_key)));
        assert_eq!(scene.render_bvh_cache_bounds, Some(bounds));
    }

    #[test]
    fn voxel_scene_dirty_budget_limits_chunks_per_rebuild() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let bounds = Aabb4i::new([-2, -2, -2, -2], [200, 2, 2, 2]);

        // Prime cache.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        assert_eq!(scene.render_bvh_cache_bounds, Some(bounds));

        // Mark more dirty chunks than per-frame budget.
        let dirty_count = VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET + 5;
        for i in 0..dirty_count {
            let wx = (i as i32) * CHUNK_SIZE as i32;
            scene.world_set_voxel(wx, 0, 0, 0, VoxelType(7));
        }
        assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), dirty_count);

        // One rebuild pass should only consume the budgeted amount.
        scene.ensure_render_bvh_cache_for_bounds(bounds);
        let expected_remaining = dirty_count.saturating_sub(VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET);
        assert_eq!(
            scene.voxel_pending_scene_dirty_regions.len(),
            expected_remaining
        );

        // Additional passes should eventually consume the remainder.
        while !scene.voxel_pending_scene_dirty_regions.is_empty() {
            scene.ensure_render_bvh_cache_for_bounds(bounds);
        }
        assert!(scene.voxel_pending_scene_dirty_regions.is_empty());
    }

    #[test]
    fn voxel_frame_snapshot_path_clears_mutation_batch() {
        let mut scene = Scene::new(ScenePreset::Empty);
        scene.world_set_voxel(0, 0, 0, 0, VoxelType(3));
        let _ = scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0);
        assert!(scene.voxel_frame_data.mutation_batch.is_none());
    }

    #[test]
    fn voxel_frame_delta_path_emits_mutation_batch() {
        let mut scene = Scene::new(ScenePreset::Empty);
        scene.world_set_voxel(0, 0, 0, 0, VoxelType(3));
        let _ = scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0);
        assert!(scene.voxel_frame_data.mutation_batch.is_none());

        scene.world_set_voxel(CHUNK_SIZE as i32, 0, 0, 0, VoxelType(4));
        let _ = scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0);

        let batch = scene.voxel_frame_data.mutation_batch.as_ref();
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert!(
            !batch.chunk_header_writes.is_empty()
                || !batch.occupancy_word_writes.is_empty()
                || !batch.material_word_writes.is_empty()
                || !batch.macro_word_writes.is_empty()
                || !batch.region_bvh_node_writes.is_empty()
                || !batch.leaf_header_writes.is_empty()
                || !batch.leaf_chunk_entry_writes.is_empty()
        );
    }

    #[test]
    fn voxel_frame_delta_root_mismatch_forces_snapshot_rebuild() {
        let mut scene = Scene::new(ScenePreset::Empty);
        scene.world_set_voxel(0, 0, 0, 0, VoxelType(3));
        let _ = scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0);

        let frame_root = scene.voxel_frame_data.region_bvh_root_index;
        assert_ne!(frame_root, VTE_REGION_BVH_INVALID_NODE);
        assert!(scene.render_bvh_cache.is_some());

        scene.voxel_pending_render_bvh_rebuild = false;
        scene.voxel_pending_render_bvh_mutation_deltas.clear();
        scene
            .voxel_pending_render_bvh_mutation_deltas
            .push(RenderBvhChunkMutationDelta {
                key: [0, 0, 0, 0],
                expected_root: Some(frame_root.wrapping_add(1)),
                new_root: Some(frame_root),
                node_writes: Vec::new(),
                leaf_writes: Vec::new(),
                freed_node_ids: Vec::new(),
                freed_leaf_ids: Vec::new(),
            });
        scene.voxel_cached_visibility_bounds = None;

        let _ = scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0);

        assert!(scene.voxel_pending_render_bvh_mutation_deltas.is_empty());
        assert!(scene.voxel_frame_data.mutation_batch.is_none());
        let cpu_root = scene
            .render_bvh_cache
            .as_ref()
            .and_then(|bvh| bvh.root)
            .unwrap_or(VTE_REGION_BVH_INVALID_NODE);
        assert_eq!(scene.voxel_frame_data.region_bvh_root_index, cpu_root);
    }

    #[test]
    fn world_get_voxel_ignores_orphaned_cached_chunk_when_tree_lacks_payload() {
        let mut scene = Scene::new(ScenePreset::Empty);
        let chunk_pos = ChunkPos {
            x: 0,
            y: 0,
            z: 0,
            w: 0,
        };
        let key = chunk_key_from_chunk_pos(chunk_pos);
        assert!(!scene.world_tree.has_chunk(key));
        let mut chunk = [VoxelType::AIR; CHUNK_VOLUME];
        chunk[0] = VoxelType(7);
        scene.world_chunks.insert(chunk_pos, chunk);

        assert!(scene.get_voxel(0, 0, 0, 0).is_air());
    }

    #[test]
    fn voxel_frame_delta_updates_match_world_after_flat_floor_edit() {
        let mut scene = Scene::new(ScenePreset::Flat);
        let cam = [0.0, 2.0, 0.0, 0.0];

        let _ = scene.build_voxel_frame_data(cam, [0.0, 0.0, 1.0, 0.0], 96.0);

        let edit_voxel = [0, -1, 0, 0];
        let before_world =
            scene.get_voxel(edit_voxel[0], edit_voxel[1], edit_voxel[2], edit_voxel[3]);
        assert!(before_world.is_solid());
        let before_frame = sample_voxel_from_frame(
            &scene,
            edit_voxel[0],
            edit_voxel[1],
            edit_voxel[2],
            edit_voxel[3],
        );
        assert_eq!(before_frame, Some(before_world.0));

        scene.world_set_voxel(
            edit_voxel[0],
            edit_voxel[1],
            edit_voxel[2],
            edit_voxel[3],
            VoxelType::AIR,
        );
        let _ = scene.build_voxel_frame_data(cam, [0.0, 0.0, 1.0, 0.0], 96.0);

        let after_world =
            scene.get_voxel(edit_voxel[0], edit_voxel[1], edit_voxel[2], edit_voxel[3]);
        assert!(after_world.is_air());
        let after_frame = sample_voxel_from_frame(
            &scene,
            edit_voxel[0],
            edit_voxel[1],
            edit_voxel[2],
            edit_voxel[3],
        );
        assert!(after_frame.is_none());
    }
}
