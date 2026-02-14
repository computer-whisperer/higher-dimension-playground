use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::io as voxel_io;
use crate::voxel::worldgen;
use crate::voxel::{ChunkPos, VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{
    GpuVoxelChunkHeader, GpuVoxelYSliceBounds, VoxelFrameInput, VTE_MAX_CHUNKS,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

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
const GPU_PAYLOAD_SLOT_CAPACITY: usize = VTE_MAX_CHUNKS;
pub const VOXEL_LOD_LEVEL_NEAR: u8 = 0;
pub const VOXEL_LOD_LEVEL_MID: u8 = 1;
pub const VOXEL_LOD_LEVEL_FAR: u8 = 2;

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
    Flat,
    DemoCubes,
}

impl ScenePreset {
    fn label(self) -> &'static str {
        match self {
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
    pub world: crate::voxel::world::VoxelWorld,
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
    voxel_cached_visibility_camera_chunk: Option<[i32; 8]>,
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
    fn surface_voxel_count(surface: &SurfaceData) -> u32 {
        surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum()
    }

    fn rebuild_surface(&mut self, label: &str) {
        self.surface = cull::extract_surfaces(&self.world);
        self.world.clear_dirty();
        let total_voxels = Self::surface_voxel_count(&self.surface);
        eprintln!(
            "{}: {} chunks, {} surface voxels",
            label,
            self.surface.chunks.len(),
            total_voxels
        );
    }

    pub fn new(preset: ScenePreset) -> Self {
        let world = match preset {
            ScenePreset::Flat => worldgen::generate_flat_world(
                5,             // 5×5×5 chunks in X, Z, W
                VoxelType(11), // neutral grid floor
            ),
            ScenePreset::DemoCubes => worldgen::generate_demo_cube_layout_world(),
        };

        let surface = cull::extract_surfaces(&world);
        let scene = Self {
            world,
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

    pub fn replace_world(&mut self, world: crate::voxel::world::VoxelWorld) {
        self.world = world;
        self.voxel_lod_chunks.clear();
        self.voxel_pending_lod_chunk_updates.clear();
        self.voxel_pending_lod_chunk_update_set.clear();
        self.voxel_chunk_payload_cache.clear();
        self.voxel_chunk_payloads.clear();
        self.voxel_chunk_payload_free_ids.clear();
        self.voxel_chunk_payload_hash_buckets.clear();
        self.voxel_payload_slot_to_payload.clear();
        self.voxel_payload_free_slots.clear();
        self.voxel_pending_payload_uploads.clear();
        self.voxel_pending_payload_upload_set.clear();
        self.voxel_active_chunks.clear();
        self.voxel_active_chunk_indices.clear();
        self.voxel_world_revision = 0;
        self.voxel_visibility_generation = 0;
        self.voxel_cached_visibility_camera_chunk = None;
        self.voxel_cached_visibility_world_revision = 0;
        self.voxel_payload_slot_overflow_logged = false;
        self.voxel_frame_data.metadata_generation = 0;
        self.voxel_frame_data.chunk_headers.clear();
        self.voxel_frame_data.payload_update_slots.clear();
        self.voxel_frame_data.occupancy_words.clear();
        self.voxel_frame_data.material_words.clear();
        self.voxel_frame_data.macro_words.clear();
        self.voxel_frame_data.visible_chunk_indices.clear();
        self.voxel_frame_data.y_slice_bounds.clear();
        self.voxel_frame_data.y_slice_lookup_entries.clear();
        self.rebuild_surface("Voxel surface (loaded)");
    }

    pub fn save_world_to_path(&self, path: &Path) -> io::Result<usize> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let mut writer = BufWriter::new(File::create(path)?);
        voxel_io::save_world(&self.world, &mut writer)?;
        writer.flush()?;

        Ok(self
            .world
            .chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count())
    }

    pub fn load_world_from_path(&mut self, path: &Path) -> io::Result<usize> {
        let mut reader = BufReader::new(File::open(path)?);
        let world = voxel_io::load_world(&mut reader)?;
        let non_empty_chunks = world
            .chunks
            .values()
            .filter(|chunk| !chunk.is_empty())
            .count();
        self.replace_world(world);
        Ok(non_empty_chunks)
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
            self.world.insert_chunk(chunk_pos, chunk);
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
            return self.world.remove_chunk_override(chunk_pos);
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
        if self.world.any_dirty() {
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
                        if self.world.get_voxel(x, y, z, w).is_solid() {
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
        let v0 = self.world.get_voxel(cell[0], cell[1], cell[2], cell[3]);
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

            let voxel = self.world.get_voxel(cell[0], cell[1], cell[2], cell[3]);
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
        self.world.set_voxel(x, y, z, w, VoxelType::AIR);
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
        if self.world.get_voxel(x, y, z, w).is_solid() {
            return None;
        }
        self.world.set_voxel(x, y, z, w, material);
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::VoxelWorld;

    fn make_scene_with_world(world: VoxelWorld) -> Scene {
        let surface = cull::extract_surfaces(&world);
        Scene {
            world,
            voxel_lod_chunks: std::collections::HashMap::new(),
            voxel_pending_lod_chunk_updates: Vec::new(),
            voxel_pending_lod_chunk_update_set: std::collections::HashSet::new(),
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
            voxel_chunk_payload_cache: std::collections::HashMap::new(),
            voxel_chunk_payloads: Vec::new(),
            voxel_chunk_payload_free_ids: Vec::new(),
            voxel_chunk_payload_hash_buckets: std::collections::HashMap::new(),
            voxel_payload_slot_to_payload: Vec::new(),
            voxel_payload_free_slots: Vec::new(),
            voxel_pending_payload_uploads: Vec::new(),
            voxel_pending_payload_upload_set: std::collections::HashSet::new(),
            voxel_active_chunks: Vec::new(),
            voxel_active_chunk_indices: std::collections::HashMap::new(),
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
        }
    }

    #[test]
    fn remove_block_along_ray_hits_first_solid_voxel() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let mut scene = make_scene_with_world(world);

        let removed =
            scene.remove_block_along_ray([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

        assert_eq!(removed, Some([0, 0, 0, 0]));
        assert!(scene.world.get_voxel(0, 0, 0, 0).is_air());
    }

    #[test]
    fn place_block_along_ray_places_in_last_empty_before_hit() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let mut scene = make_scene_with_world(world);

        let placed = scene.place_block_along_ray(
            [0.5, 0.5, -2.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            8.0,
            VoxelType(3),
        );

        assert_eq!(placed, Some([0, 0, -1, 0]));
        assert!(scene.world.get_voxel(0, 0, -1, 0) == VoxelType(3));
        assert!(scene.world.get_voxel(0, 0, 0, 0) == VoxelType(7));
    }

    #[test]
    fn block_edit_targets_reports_hit_and_placement_voxels() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(7));
        let scene = make_scene_with_world(world);

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
        let mut world = VoxelWorld::new();
        world.set_voxel(0, -1, 0, 0, VoxelType(3));
        let scene = make_scene_with_world(world);

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
        let mut world = VoxelWorld::new();
        world.set_voxel(1, 0, 0, 0, VoxelType(3));
        let scene = make_scene_with_world(world);

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
        let world = VoxelWorld::new();
        let scene = make_scene_with_world(world);

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
    fn replace_world_rebuilds_surface_for_clean_loaded_world() {
        let mut world = VoxelWorld::new();
        world.set_voxel(0, 0, 0, 0, VoxelType(3));
        let mut scene = make_scene_with_world(world);
        assert!(!scene.surface.chunks.is_empty());

        scene.replace_world(VoxelWorld::new());
        assert!(scene.surface.chunks.is_empty());
    }
}
