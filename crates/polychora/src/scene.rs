use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::world::BaseWorldKind;
use crate::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkHeader, GpuVoxelYSliceBounds, VoxelFrameInput, VTE_MAX_CHUNKS,
};
use polychora::shared::voxel::world_to_chunk;
use polychora::shared::worldfield::{ChunkKey, ChunkPayload, RegionChunkTree};
use std::collections::{BTreeMap, HashMap, HashSet};

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
pub const VOXEL_LOD_LEVEL_MID: u8 = 1;
pub const VOXEL_LOD_LEVEL_FAR: u8 = 2;
const FLAT_PRESET_FLOOR_MATERIAL: VoxelType = VoxelType(11);
const SHOWCASE_MATERIALS: [u8; 37] = [
    15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 45, 50, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
];

fn set_chunk_voxel_by_index(chunk: &mut crate::voxel::chunk::Chunk, idx: usize, v: VoxelType) {
    let old = chunk.voxels[idx];
    if old == v {
        return;
    }
    if old.is_solid() && v.is_air() {
        chunk.solid_count = chunk.solid_count.saturating_sub(1);
    } else if old.is_air() && v.is_solid() {
        chunk.solid_count = chunk.solid_count.saturating_add(1);
    }
    chunk.voxels[idx] = v;
    chunk.dirty = true;
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
    world_chunks: HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
    world_base_kind: BaseWorldKind,
    world_flat_floor_chunk: crate::voxel::chunk::Chunk,
    world_dirty: bool,
    world_pending_chunk_updates: Vec<ChunkPos>,
    world_pending_chunk_update_set: HashSet<ChunkPos>,
    voxel_lod_chunks: HashMap<RuntimeChunkKey, crate::voxel::chunk::Chunk>,
    voxel_pending_lod_chunk_updates: Vec<RuntimeChunkKey>,
    voxel_pending_lod_chunk_update_set: HashSet<RuntimeChunkKey>,
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
    voxel_cached_visibility_camera_chunk: Option<[i32; 16]>,
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

impl Scene {
    fn set_chunk_map_voxel(
        chunks: &mut HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
        wx: i32,
        wy: i32,
        wz: i32,
        ww: i32,
        v: VoxelType,
    ) {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        let should_remove = {
            let chunk = chunks
                .entry(cp)
                .or_insert_with(crate::voxel::chunk::Chunk::new);
            set_chunk_voxel_by_index(chunk, idx, v);
            chunk.is_empty()
        };
        if should_remove {
            chunks.remove(&cp);
        }
    }

    fn fill_hypercube(
        chunks: &mut HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
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

    fn place_material_showcase(
        chunks: &mut HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
        origin: [i32; 4],
    ) {
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
    ) -> (BaseWorldKind, HashMap<ChunkPos, crate::voxel::chunk::Chunk>) {
        let mut chunks = HashMap::<ChunkPos, crate::voxel::chunk::Chunk>::new();

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

        for chunk in chunks.values_mut() {
            chunk.dirty = false;
        }
        (base_kind, chunks)
    }

    fn scene_world_from_chunk_overrides(
        base_kind: BaseWorldKind,
        mut world_chunks: HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
    ) -> (
        RegionChunkTree,
        HashMap<ChunkPos, crate::voxel::chunk::Chunk>,
        BaseWorldKind,
        crate::voxel::chunk::Chunk,
        bool,
    ) {
        world_chunks.retain(|_, chunk| !chunk.is_empty());
        let world_tree = RegionChunkTree::from_chunks(world_chunks.iter().map(|(&pos, chunk)| {
            (
                ChunkKey::from_chunk_pos(pos),
                ChunkPayload::from_chunk_compact(chunk),
            )
        }));
        let flat_floor_chunk = Self::flat_floor_chunk_for_base(base_kind);
        (world_tree, world_chunks, base_kind, flat_floor_chunk, false)
    }

    fn build_flat_floor_chunk(material: VoxelType) -> crate::voxel::chunk::Chunk {
        let mut chunk = crate::voxel::chunk::Chunk::new();
        if material.is_air() {
            chunk.dirty = false;
            return chunk;
        }

        let local_y_top = CHUNK_SIZE - 1;
        let local_y_bottom = CHUNK_SIZE - 2;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for w in 0..CHUNK_SIZE {
                    chunk.set(x, local_y_top, z, w, material);
                    chunk.set(x, local_y_bottom, z, w, material);
                }
            }
        }
        chunk.dirty = false;
        chunk
    }

    fn flat_floor_chunk_for_base(base_kind: BaseWorldKind) -> crate::voxel::chunk::Chunk {
        match base_kind {
            BaseWorldKind::FlatFloor { material } => Self::build_flat_floor_chunk(material),
            BaseWorldKind::Empty => crate::voxel::chunk::Chunk::new(),
        }
    }

    fn world_base_chunk_for_pos(&self, pos: ChunkPos) -> Option<&crate::voxel::chunk::Chunk> {
        match self.world_base_kind {
            BaseWorldKind::Empty => None,
            BaseWorldKind::FlatFloor { .. } if pos.y == FLAT_FLOOR_CHUNK_Y => {
                Some(&self.world_flat_floor_chunk)
            }
            BaseWorldKind::FlatFloor { .. } => None,
        }
    }

    fn world_base_voxel_at(&self, pos: ChunkPos, idx: usize) -> VoxelType {
        self.world_base_chunk_for_pos(pos)
            .map(|chunk| chunk.voxels[idx])
            .unwrap_or(VoxelType::AIR)
    }

    fn world_clone_base_chunk_or_empty(&self, pos: ChunkPos) -> crate::voxel::chunk::Chunk {
        self.world_base_chunk_for_pos(pos)
            .cloned()
            .unwrap_or_else(crate::voxel::chunk::Chunk::new)
    }

    fn world_chunk_matches_base(&self, pos: ChunkPos, chunk: &crate::voxel::chunk::Chunk) -> bool {
        match self.world_base_chunk_for_pos(pos) {
            Some(base) => {
                chunk.solid_count == base.solid_count && chunk.voxels[..] == base.voxels[..]
            }
            None => chunk.is_empty(),
        }
    }

    fn world_queue_chunk_update(&mut self, pos: ChunkPos) {
        if self.world_pending_chunk_update_set.insert(pos) {
            self.world_pending_chunk_updates.push(pos);
        }
    }

    fn world_set_chunk_override(
        &mut self,
        pos: ChunkPos,
        chunk: Option<crate::voxel::chunk::Chunk>,
    ) -> bool {
        let key = ChunkKey::from_chunk_pos(pos);
        let payload = match chunk {
            Some(chunk) => {
                if self.world_chunk_matches_base(pos, &chunk) {
                    self.world_chunks.remove(&pos);
                    None
                } else {
                    self.world_chunks.insert(pos, chunk.clone());
                    Some(ChunkPayload::from_chunk_compact(&chunk))
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
            self.world_queue_chunk_update(pos);
        }
        changed
    }

    fn world_has_explicit_chunk(&self, pos: ChunkPos) -> bool {
        self.world_tree.has_chunk(ChunkKey::from_chunk_pos(pos))
    }

    fn world_chunk_at(&self, pos: ChunkPos) -> Option<&crate::voxel::chunk::Chunk> {
        if let Some(chunk) = self.world_chunks.get(&pos) {
            return (!chunk.is_empty()).then_some(chunk);
        }
        if self.world_has_explicit_chunk(pos) {
            return None;
        }
        self.world_base_chunk_for_pos(pos)
            .filter(|chunk| !chunk.is_empty())
    }

    fn world_get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        let (cp, idx) = world_to_chunk(wx, wy, wz, ww);
        if let Some(chunk) = self.world_chunks.get(&cp) {
            return chunk.voxels[idx];
        }
        if self.world_has_explicit_chunk(cp) {
            return VoxelType::AIR;
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
            self.world_chunks
                .get(&cp)
                .cloned()
                .unwrap_or_else(crate::voxel::chunk::Chunk::new)
        } else {
            self.world_clone_base_chunk_or_empty(cp)
        };
        set_chunk_voxel_by_index(&mut chunk, idx, v);
        let _ = self.world_set_chunk_override(cp, Some(chunk));
    }

    fn world_insert_chunk(&mut self, pos: ChunkPos, mut chunk: crate::voxel::chunk::Chunk) {
        chunk.dirty = true;
        let _ = self.world_set_chunk_override(pos, Some(chunk));
    }

    fn world_remove_chunk_override(&mut self, pos: ChunkPos) -> bool {
        self.world_set_chunk_override(pos, None)
    }

    fn world_gather_non_empty_chunks_in_bounds(
        &self,
        min_chunk: [i32; 4],
        max_chunk: [i32; 4],
        out: &mut Vec<ChunkPos>,
    ) {
        out.clear();
        if min_chunk[0] > max_chunk[0]
            || min_chunk[1] > max_chunk[1]
            || min_chunk[2] > max_chunk[2]
            || min_chunk[3] > max_chunk[3]
        {
            return;
        }

        if let BaseWorldKind::FlatFloor { .. } = self.world_base_kind {
            let y = FLAT_FLOOR_CHUNK_Y;
            if y >= min_chunk[1] && y <= max_chunk[1] {
                for x in min_chunk[0]..=max_chunk[0] {
                    for z in min_chunk[2]..=max_chunk[2] {
                        for w in min_chunk[3]..=max_chunk[3] {
                            let pos = ChunkPos::new(x, y, z, w);
                            if let Some(override_chunk) = self.world_chunks.get(&pos) {
                                if !override_chunk.is_empty() {
                                    out.push(pos);
                                }
                            } else if !self.world_has_explicit_chunk(pos) {
                                out.push(pos);
                            }
                        }
                    }
                }
            }
        }

        for (&pos, chunk) in &self.world_chunks {
            if pos.x < min_chunk[0]
                || pos.x > max_chunk[0]
                || pos.y < min_chunk[1]
                || pos.y > max_chunk[1]
                || pos.z < min_chunk[2]
                || pos.z > max_chunk[2]
                || pos.w < min_chunk[3]
                || pos.w > max_chunk[3]
                || chunk.is_empty()
            {
                continue;
            }
            if matches!(self.world_base_kind, BaseWorldKind::FlatFloor { .. })
                && pos.y == FLAT_FLOOR_CHUNK_Y
            {
                continue;
            }
            out.push(pos);
        }
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
        for chunk in self.world_chunks.values_mut() {
            chunk.dirty = false;
        }
    }

    pub fn get_voxel(&self, wx: i32, wy: i32, wz: i32, ww: i32) -> VoxelType {
        self.world_get_voxel(wx, wy, wz, ww)
    }

    pub fn collect_non_empty_explicit_chunk_positions(&self) -> Vec<[i32; 4]> {
        let mut out: Vec<[i32; 4]> = self
            .world_chunks
            .iter()
            .filter(|(_, chunk)| !chunk.is_empty())
            .map(|(&chunk_pos, _)| [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w])
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
            voxel_lod_chunks: HashMap::new(),
            voxel_pending_lod_chunk_updates: Vec::new(),
            voxel_pending_lod_chunk_update_set: HashSet::new(),
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

    fn queue_lod_chunk_update(&mut self, key: RuntimeChunkKey) {
        if self.voxel_pending_lod_chunk_update_set.insert(key) {
            self.voxel_pending_lod_chunk_updates.push(key);
        }
    }

    pub fn insert_lod_chunk(
        &mut self,
        lod_level: u8,
        chunk_pos: ChunkPos,
        mut chunk: crate::voxel::chunk::Chunk,
    ) {
        if lod_level == VOXEL_LOD_LEVEL_NEAR {
            self.world_insert_chunk(chunk_pos, chunk);
            return;
        }

        let key = RuntimeChunkKey {
            lod_level,
            chunk_pos,
        };
        chunk.dirty = true;
        if chunk.is_empty() {
            self.voxel_lod_chunks.remove(&key);
        } else {
            self.voxel_lod_chunks.insert(key, chunk);
        }
        self.queue_lod_chunk_update(key);
    }

    pub fn remove_lod_chunk(&mut self, lod_level: u8, chunk_pos: ChunkPos) -> bool {
        if lod_level == VOXEL_LOD_LEVEL_NEAR {
            return self.world_remove_chunk_override(chunk_pos);
        }

        let key = RuntimeChunkKey {
            lod_level,
            chunk_pos,
        };
        let removed = self.voxel_lod_chunks.remove(&key).is_some();
        if removed {
            self.queue_lod_chunk_update(key);
        }
        removed
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
}
