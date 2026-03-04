use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::{CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkBvhNode, GpuVoxelChunkHeader, GpuVoxelLeafHeader, VoxelFrameDirtyRanges,
    VoxelFrameInput, VoxelMutationBatch, VTE_REGION_BVH_INVALID_NODE,
};
use polychora::shared::chunk_payload::{ChunkPayload, ResolvedChunkPayload};
use polychora::shared::protocol::WorldBounds;
use polychora::shared::region_tree::{
    chunk_key_i32, slice_non_empty_region_core_in_bounds, validate_tree_integrity, ChunkKey,
    RegionChunkTree, RegionNodeKind, RegionTreeCore, TreeIntegrityReport,
};
use polychora::shared::render_tree::{self, RenderBvhChunkMutationDelta};
use polychora::shared::spatial::{step_for_scale, Aabb4i, ChunkCoord};
use polychora::shared::voxel::{world_to_chunk_at_scale, BlockData};
use std::collections::HashMap;
#[cfg(test)]
use std::collections::HashSet;
use std::time::Instant;

mod collision;
mod editing;
mod patching;
pub(crate) mod render_cache;
mod voxel_runtime;

const VOXEL_NEAR_ACTIVE_DISTANCE: f32 = 32.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 2; // packed 2x u16 per u32
const ORIENTATION_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 2; // packed 2x u16 per u32
const MACRO_CELLS_PER_AXIS: usize = CHUNK_SIZE / 2; // 2x2x2x2 macro cells
const MACRO_CELLS_PER_CHUNK: usize =
    MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS * MACRO_CELLS_PER_AXIS;
const MACRO_WORDS_PER_CHUNK: usize = MACRO_CELLS_PER_CHUNK / 32;
const EDIT_RAY_EPSILON: f32 = 1e-4;
const VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET: usize = 4;
const COLLISION_PUSHUP_STEP: f32 = 0.05;
const COLLISION_MAX_PUSHUP_STEPS: usize = 80;
const COLLISION_BINARY_STEPS: usize = 14;
const FLAT_FLOOR_CHUNK_Y: i32 = -1;
const FLAT_PRESET_FLOOR_MATERIAL_BLOCK: BlockData = BlockData {
    namespace: polychora_plugin_api::content_ids::CONTENT_NS,
    block_type: polychora_plugin_api::content_ids::BLOCK_GRID_FLOOR,
    orientation: polychora::shared::voxel::TesseractOrientation::IDENTITY,
    extra_data: Vec::new(),
    scale_exp: 0,
};
const SHOWCASE_MATERIAL_TOKENS: [u8; 37] = [
    15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 45, 50, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
];

#[inline]
fn dense_blocks_local_index(x: usize, y: usize, z: usize, w: usize) -> usize {
    w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x
}

#[derive(Copy, Clone, Debug)]
pub enum ScenePreset {
    Empty,
    Flat,
    DemoCubes,
}

pub struct VoxelFrameData {
    pub metadata_generation: u64,
    pub mutation_base_generation: Option<u64>,
    pub region_bvh_root_index: u32,
    pub dirty_ranges: VoxelFrameDirtyRanges,
    pub mutation_batch: Option<VoxelMutationBatch>,
    pub chunk_headers: Vec<GpuVoxelChunkHeader>,
    pub occupancy_words: Vec<u32>,
    pub material_words: Vec<u32>,
    pub orientation_words: Vec<u32>,
    pub macro_words: Vec<u32>,
    pub region_bvh_nodes: Vec<GpuVoxelChunkBvhNode>,
    pub leaf_headers: Vec<GpuVoxelLeafHeader>,
    pub leaf_chunk_entries: Vec<u32>,
}

struct VoxelFrameDataBuffers {
    region_bvh_root_index: u32,
    dense_payload_encoded_cache: HashMap<DensePayloadCacheKey, u32>,
    chunk_headers: Vec<GpuVoxelChunkHeader>,
    occupancy_words: Vec<u32>,
    material_words: Vec<u32>,
    orientation_words: Vec<u32>,
    macro_words: Vec<u32>,
    region_bvh_nodes: Vec<GpuVoxelChunkBvhNode>,
    leaf_headers: Vec<GpuVoxelLeafHeader>,
    leaf_chunk_entries: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DensePayloadCacheKey {
    payload: ChunkPayload,
    block_palette: Vec<BlockData>,
}

struct BackgroundRebuildResult {
    render_bvh: render_tree::RenderBvh,
    frame_buffers: VoxelFrameDataBuffers,
}

struct BackgroundVoxelRebuild {
    receiver: std::sync::mpsc::Receiver<Result<BackgroundRebuildResult, String>>,
    bounds: Aabb4i,
}

impl DensePayloadCacheKey {
    fn new(payload: &ChunkPayload, block_palette: &[BlockData]) -> Self {
        Self {
            payload: payload.clone(),
            block_palette: block_palette.to_vec(),
        }
    }
}

impl VoxelFrameData {
    pub fn as_input(&self) -> VoxelFrameInput<'_> {
        VoxelFrameInput {
            metadata_generation: self.metadata_generation,
            mutation_base_generation: self.mutation_base_generation,
            region_bvh_root_index: self.region_bvh_root_index,
            chunk_headers: &self.chunk_headers,
            occupancy_words: &self.occupancy_words,
            material_words: &self.material_words,
            orientation_words: &self.orientation_words,
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
    pub world_bounds: WorldBounds,
    world_tree: RegionChunkTree,
    world_tree_revision: u64,
    #[cfg(test)]
    world_pending_chunk_updates: Vec<ChunkKey>,
    #[cfg(test)]
    world_pending_chunk_update_set: HashSet<ChunkKey>,
    voxel_visibility_generation: u64,
    voxel_cached_visibility_bounds: Option<Aabb4i>,
    voxel_pending_scene_dirty_regions: Vec<Aabb4i>,
    render_bvh_cache_bounds: Option<Aabb4i>,
    render_bvh_cache: Option<render_tree::RenderBvh>,
    voxel_pending_render_bvh_rebuild: bool,
    voxel_pending_render_bvh_mutation_deltas: Vec<RenderBvhChunkMutationDelta>,
    voxel_dense_payload_encoded_cache: HashMap<DensePayloadCacheKey, u32>,
    voxel_leaf_entry_spans: Vec<Option<std::ops::Range<usize>>>,
    voxel_leaf_entry_free_spans: Vec<std::ops::Range<usize>>,
    voxel_background_rebuild: Option<BackgroundVoxelRebuild>,
    voxel_frame_data: VoxelFrameData,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ScaleAwareBlockTarget {
    /// Block-aligned world origin as fixed-point coordinates.
    pub origin: [ChunkCoord; 4],
    /// Scale exponent. Block occupies 2^scale_exp world units per axis.
    pub scale_exp: i8,
}

impl ScaleAwareBlockTarget {
    /// Side length as a fixed-point value: `2^scale_exp`.
    pub fn size(&self) -> ChunkCoord {
        step_for_scale(self.scale_exp)
    }

    /// Integer origin for legacy code paths.
    pub fn origin_i32(&self) -> [i32; 4] {
        self.origin.map(|c| c.to_num::<i32>())
    }

    pub fn world_min(&self) -> [f32; 4] {
        self.origin.map(|c| c.to_num::<f32>())
    }

    pub fn world_max(&self) -> [f32; 4] {
        let s = self.size();
        std::array::from_fn(|i| (self.origin[i] + s).to_num::<f32>())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockEditTargets {
    pub hit: Option<ScaleAwareBlockTarget>,
    pub hit_block: Option<BlockData>,
    pub place: Option<ScaleAwareBlockTarget>,
    /// Which axis (0..3) the ray crossed to enter the hit block.
    pub face_axis: u8,
    /// -1 = entered through the min face, +1 = entered through the max face.
    pub face_sign: i8,
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
    fn set_dense_blocks_voxel(
        chunks: &mut HashMap<ChunkKey, Vec<BlockData>>,
        wx: i32,
        wy: i32,
        wz: i32,
        ww: i32,
        block: BlockData,
    ) {
        let (cp, idx) = world_to_chunk_at_scale(
            ChunkCoord::from_num(wx),
            ChunkCoord::from_num(wy),
            ChunkCoord::from_num(wz),
            ChunkCoord::from_num(ww),
            0,
        );
        let should_remove = {
            let chunk = chunks
                .entry(cp)
                .or_insert_with(|| vec![BlockData::AIR; CHUNK_VOLUME]);
            chunk[idx] = block;
            chunk.iter().all(|b| b.is_air())
        };
        if should_remove {
            chunks.remove(&cp);
        }
    }

    fn fill_hypercube(
        chunks: &mut HashMap<ChunkKey, Vec<BlockData>>,
        min: [i32; 4],
        edge: i32,
        block: BlockData,
    ) {
        for x in min[0]..(min[0] + edge) {
            for y in min[1]..(min[1] + edge) {
                for z in min[2]..(min[2] + edge) {
                    for w in min[3]..(min[3] + edge) {
                        Self::set_dense_blocks_voxel(chunks, x, y, z, w, block.clone());
                    }
                }
            }
        }
    }

    fn place_material_showcase(chunks: &mut HashMap<ChunkKey, Vec<BlockData>>, origin: [i32; 4]) {
        for (idx, token) in SHOWCASE_MATERIAL_TOKENS.iter().copied().enumerate() {
            let col = (idx % 6) as i32;
            let row = (idx / 6) as i32;
            let min = [
                origin[0] + col * 4,
                origin[1],
                origin[2] + row * 4,
                origin[3],
            ];
            Self::fill_hypercube(
                chunks,
                min,
                2,
                polychora::content_registry::block_data_from_material_token(token),
            );
        }
    }

    fn build_scene_preset_world(preset: ScenePreset) -> RegionChunkTree {
        let mut chunks = HashMap::<ChunkKey, Vec<BlockData>>::new();

        match preset {
            ScenePreset::Empty => {}
            ScenePreset::Flat => {
                Self::place_material_showcase(&mut chunks, [-10, 0, -14, -4]);
                let floor_payload =
                    Self::build_flat_floor_payload(FLAT_PRESET_FLOOR_MATERIAL_BLOCK.clone());
                for x in -2..=2 {
                    for z in -2..=2 {
                        for w in -2..=2 {
                            let key = chunk_key_i32(x, FLAT_FLOOR_CHUNK_Y, z, w);
                            // Insert a dummy entry so we know to emit this chunk;
                            // the actual payload is already built.
                            chunks.entry(key).or_default();
                        }
                    }
                }
                // Build the tree from the chunks map, but override floor chunks
                // with the prebuilt payload.
                let mut tree_entries: Vec<(ChunkKey, ResolvedChunkPayload)> = Vec::new();
                for (&key, blocks) in &chunks {
                    if key[1] == ChunkCoord::from_num(FLAT_FLOOR_CHUNK_Y) && blocks.is_empty() {
                        tree_entries.push((key, floor_payload.clone()));
                    } else if !blocks.is_empty() && blocks.len() == CHUNK_VOLUME {
                        if blocks.iter().all(|b| b.is_air()) {
                            continue;
                        }
                        let payload = ResolvedChunkPayload::from_dense_blocks(blocks)
                            .unwrap_or_else(|_| ResolvedChunkPayload::empty());
                        tree_entries.push((key, payload));
                    }
                }
                return RegionChunkTree::from_chunks(tree_entries);
            }
            ScenePreset::DemoCubes => {
                // Cycle through the first 5 block tokens for variety
                let mut texture_rot = 0u8;
                for x in 0..2 {
                    for y in 0..2 {
                        for z in 0..2 {
                            for w in 0..2 {
                                let base = [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2];
                                let token = (texture_rot % 5) + 1;
                                let block =
                                    polychora::content_registry::block_data_from_material_token(
                                        token,
                                    );
                                Self::fill_hypercube(&mut chunks, base, 2, block);
                                texture_rot = (texture_rot + 1) % 5;
                            }
                        }
                    }
                }
                Self::fill_hypercube(
                    &mut chunks,
                    [0, 0, 0, 0],
                    2,
                    polychora::content_registry::block_data_from_material_token(13), // LIGHT
                );
            }
        };

        Self::dense_blocks_map_to_tree(chunks)
    }

    fn dense_blocks_map_to_tree(chunks: HashMap<ChunkKey, Vec<BlockData>>) -> RegionChunkTree {
        RegionChunkTree::from_chunks(
            chunks
                .into_iter()
                .filter(|(_, blocks)| {
                    blocks.len() == CHUNK_VOLUME && !blocks.iter().all(|b| b.is_air())
                })
                .map(|(key, blocks)| {
                    let payload = ResolvedChunkPayload::from_dense_blocks(&blocks)
                        .unwrap_or_else(|_| ResolvedChunkPayload::empty());
                    (key, payload)
                }),
        )
    }

    fn build_flat_floor_payload(block: BlockData) -> ResolvedChunkPayload {
        if block.is_air() {
            return ResolvedChunkPayload::empty();
        }
        let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];
        let local_y_top = CHUNK_SIZE - 1;
        let local_y_bottom = CHUNK_SIZE - 2;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for w in 0..CHUNK_SIZE {
                    blocks[dense_blocks_local_index(x, local_y_top, z, w)] = block.clone();
                    blocks[dense_blocks_local_index(x, local_y_bottom, z, w)] = block.clone();
                }
            }
        }
        ResolvedChunkPayload::from_dense_blocks(&blocks)
            .unwrap_or_else(|_| ResolvedChunkPayload::empty())
    }

    #[cfg(test)]
    fn world_queue_chunk_update(&mut self, key: ChunkKey) -> bool {
        if self.world_pending_chunk_update_set.insert(key) {
            self.world_pending_chunk_updates.push(key);
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn world_set_block(&mut self, wx: i32, wy: i32, wz: i32, ww: i32, block: BlockData) {
        self.world_set_block_at(
            [
                ChunkCoord::from_num(wx),
                ChunkCoord::from_num(wy),
                ChunkCoord::from_num(wz),
                ChunkCoord::from_num(ww),
            ],
            block,
        );
    }

    /// Set a block at a fixed-point world position. Supports sub-voxel positions
    /// for negative `scale_exp`.
    #[cfg(test)]
    fn world_set_block_at(&mut self, pos: [ChunkCoord; 4], block: BlockData) {
        let scale_exp = block.scale_exp;
        let (key, idx) = world_to_chunk_at_scale(pos[0], pos[1], pos[2], pos[3], scale_exp);
        let old = self.get_block_data_at(pos);
        if old == block {
            return;
        }

        // Read the current chunk payload, decode it to dense blocks, edit, and write back.
        let mut blocks = if let Some((payload, _)) = self.world_tree.chunk_payload(key) {
            (0..CHUNK_VOLUME)
                .map(|i| payload.block_at(i))
                .collect::<Vec<_>>()
        } else {
            vec![BlockData::AIR; CHUNK_VOLUME]
        };
        blocks[idx] = block;

        let all_air = blocks.iter().all(|b| b.is_air());
        let payload = if all_air {
            None
        } else {
            Some(
                ResolvedChunkPayload::from_dense_blocks(&blocks)
                    .unwrap_or_else(|_| ResolvedChunkPayload::empty()),
            )
        };

        let previous_payload = self.world_tree.chunk_payload(key);
        let changed_by_api = self
            .world_tree
            .set_chunk_at_scale(key, payload.clone(), scale_exp)
            .is_some();
        let current_payload = self.world_tree.chunk_payload(key);
        let mut changed = changed_by_api || previous_payload != current_payload;
        if !changed && current_payload.as_ref().map(|(p, _)| p) != payload.as_ref() {
            let bounds = Aabb4i::chunk_world_bounds(key, scale_exp);
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
        if changed {
            self.world_tree_revision = self.world_tree_revision.wrapping_add(1);
            self.mark_voxel_scene_region_dirty(Aabb4i::chunk_world_bounds(key, scale_exp));
            let _ = self.world_queue_chunk_update(key);
        }
    }

    #[cfg(test)]
    fn world_drain_pending_chunk_updates(&mut self) -> Vec<ChunkKey> {
        self.world_pending_chunk_update_set.clear();
        std::mem::take(&mut self.world_pending_chunk_updates)
    }

    pub fn get_block_data(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> BlockData {
        self.get_block_data_at([
            ChunkCoord::from_num(wx),
            ChunkCoord::from_num(wy),
            ChunkCoord::from_num(wz),
            ChunkCoord::from_num(ww),
        ])
    }

    pub fn get_block_data_at(&self, pos: [ChunkCoord; 4]) -> BlockData {
        self.world_tree.block_at(pos)
    }

    pub fn debug_world_tree_chunk_payload(
        &self,
        chunk_key: ChunkKey,
    ) -> Option<(ResolvedChunkPayload, i8)> {
        self.world_tree.chunk_payload(chunk_key)
    }

    pub fn debug_render_bvh_cache_chunk_payloads(
        &self,
        chunk_key: ChunkKey,
    ) -> Vec<ResolvedChunkPayload> {
        self.render_bvh_cache
            .as_ref()
            .map(|bvh| render_tree::sample_chunk_payloads_from_bvh(bvh, chunk_key))
            .unwrap_or_default()
    }

    pub fn debug_pending_dirty_region_count(&self) -> usize {
        self.voxel_pending_scene_dirty_regions.len()
    }

    pub fn debug_pending_render_bvh_delta_count(&self) -> usize {
        self.voxel_pending_render_bvh_mutation_deltas.len()
    }

    pub fn debug_world_tree_root_bounds(&self) -> Option<Aabb4i> {
        self.world_tree.root().map(|root| root.bounds)
    }

    pub fn check_world_tree_integrity(&self) -> TreeIntegrityReport {
        validate_tree_integrity(&self.world_tree)
    }

    pub fn force_render_rebuild(&mut self) {
        self.force_render_bvh_rebuild();
    }

    pub fn dump_world_tree(&self) {
        eprintln!("\n=== WORLD TREE ===");
        if let Some(root) = self.world_tree.root() {
            Self::dump_world_tree_node(root, 0);
        } else {
            eprintln!("  (empty)");
        }
    }

    fn dump_world_tree_node(node: &RegionTreeCore, depth: usize) {
        let indent = "  ".repeat(depth + 1);
        let b = &node.bounds;
        let bounds_str = format!(
            "[{},{},{},{}]->[{},{},{},{}]",
            b.min[0].to_num::<i32>(),
            b.min[1].to_num::<i32>(),
            b.min[2].to_num::<i32>(),
            b.min[3].to_num::<i32>(),
            b.max[0].to_num::<i32>(),
            b.max[1].to_num::<i32>(),
            b.max[2].to_num::<i32>(),
            b.max[3].to_num::<i32>(),
        );
        match &node.kind {
            RegionNodeKind::Empty => eprintln!("{indent}Empty {bounds_str}"),
            RegionNodeKind::Uniform(block) => {
                eprintln!(
                    "{indent}Uniform(ns={},bt={}) {bounds_str}",
                    block.namespace, block.block_type
                );
            }
            RegionNodeKind::ProceduralRef(_) => eprintln!("{indent}ProceduralRef {bounds_str}"),
            RegionNodeKind::ChunkArray(ca) => {
                let cells = ca
                    .bounds
                    .chunk_cell_count_at_scale(ca.scale_exp)
                    .unwrap_or(0);
                eprintln!(
                    "{indent}ChunkArray(cells={}, palette={}, scale={}) {bounds_str}",
                    cells,
                    ca.chunk_palette.len(),
                    ca.scale_exp
                );
            }
            RegionNodeKind::Branch(children) => {
                eprintln!("{indent}Branch({} children) {bounds_str}", children.len());
                for child in children {
                    Self::dump_world_tree_node(child, depth + 1);
                }
            }
        }
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
        for (voxel_idx, material_slot) in out.iter_mut().enumerate().take(CHUNK_VOLUME) {
            if !is_full {
                let occ_word_idx = header.occupancy_word_offset as usize + (voxel_idx / 32);
                let occ_word = *self.voxel_frame_data.occupancy_words.get(occ_word_idx)?;
                if (occ_word & (1u32 << (voxel_idx % 32))) == 0 {
                    continue;
                }
            }
            let mat_word_idx = header.material_word_offset as usize + (voxel_idx / 2);
            let mat_word = *self.voxel_frame_data.material_words.get(mat_word_idx)?;
            let material = ((mat_word >> ((voxel_idx % 2) * 16)) & 0xFFFF) as u16;
            *material_slot = material;
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
                let material = (entry & 0xFFFF) as u16;
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

    pub fn collect_non_empty_explicit_chunk_positions(&self) -> Vec<ChunkKey> {
        let Some(root) = self.world_tree.root() else {
            return Vec::new();
        };
        let mut out: Vec<ChunkKey> = self
            .world_tree
            .collect_non_empty_chunk_keys_in_bounds(root.bounds);
        out.sort_unstable();
        out
    }

    pub fn new(preset: ScenePreset) -> Self {
        let world_tree = Self::build_scene_preset_world(preset);

        let world_bounds = match preset {
            ScenePreset::Flat => WorldBounds {
                min: [None, Some(-4.0), None, None],
                max: [None; 4],
            },
            ScenePreset::Empty | ScenePreset::DemoCubes => WorldBounds::default(),
        };

        Self {
            world_bounds,
            world_tree,
            world_tree_revision: 1,
            #[cfg(test)]
            world_pending_chunk_updates: Vec::new(),
            #[cfg(test)]
            world_pending_chunk_update_set: HashSet::new(),
            voxel_visibility_generation: 0,
            voxel_cached_visibility_bounds: None,
            voxel_pending_scene_dirty_regions: Vec::new(),
            render_bvh_cache_bounds: None,
            render_bvh_cache: None,
            voxel_pending_render_bvh_rebuild: true,
            voxel_pending_render_bvh_mutation_deltas: Vec::new(),
            voxel_dense_payload_encoded_cache: HashMap::new(),
            voxel_leaf_entry_spans: Vec::new(),
            voxel_leaf_entry_free_spans: Vec::new(),
            voxel_background_rebuild: None,
            voxel_frame_data: VoxelFrameData {
                metadata_generation: 0,
                mutation_base_generation: None,
                region_bvh_root_index:
                    higher_dimension_playground::render::VTE_REGION_BVH_INVALID_NODE,
                dirty_ranges: VoxelFrameDirtyRanges::default(),
                mutation_batch: None,
                chunk_headers: Vec::new(),
                occupancy_words: Vec::new(),
                material_words: Vec::new(),
                orientation_words: Vec::new(),
                macro_words: Vec::new(),
                region_bvh_nodes: Vec::new(),
                leaf_headers: Vec::new(),
                leaf_chunk_entries: Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests;
