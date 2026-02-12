use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::worldgen;
use crate::voxel::{VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use crate::camera::{PLAYER_HEIGHT, PLAYER_RADIUS_XZW};
use higher_dimension_playground::render::{GpuVoxelChunkHeader, VoxelFrameInput};

const RENDER_DISTANCE: f32 = 64.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 4; // packed 4x u8 per u32
const EDIT_RAY_EPSILON: f32 = 1e-4;
const EDIT_RAY_MAX_STEPS: usize = 4096;
const COLLISION_PUSHUP_STEP: f32 = 0.05;
const COLLISION_MAX_PUSHUP_STEPS: usize = 80;
const COLLISION_BINARY_STEPS: usize = 14;

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
    pub chunk_headers: Vec<GpuVoxelChunkHeader>,
    pub occupancy_words: Vec<u32>,
    pub material_words: Vec<u32>,
    pub visible_chunk_indices: Vec<u32>,
}

impl VoxelFrameData {
    pub fn as_input(&self) -> VoxelFrameInput<'_> {
        VoxelFrameInput {
            chunk_headers: &self.chunk_headers,
            occupancy_words: &self.occupancy_words,
            material_words: &self.material_words,
            visible_chunk_indices: &self.visible_chunk_indices,
        }
    }
}

pub struct Scene {
    pub world: crate::voxel::world::VoxelWorld,
    surface: SurfaceData,
    culled_instances: Vec<common::ModelInstance>,
    cull_log_counter: u64,
}

#[derive(Copy, Clone)]
struct VoxelRayHit {
    solid_voxel: [i32; 4],
    last_empty_voxel: Option<[i32; 4]>,
}

impl Scene {
    pub fn new(preset: ScenePreset) -> Self {
        let world = match preset {
            ScenePreset::Flat => worldgen::generate_flat_world(
                3,            // 3×3×3 chunks in X, Z, W
                VoxelType(3), // grass
            ),
            ScenePreset::DemoCubes => worldgen::generate_demo_cube_layout_world(),
        };

        let surface = cull::extract_surfaces(&world);
        let total_voxels: u32 = surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum();
        eprintln!(
            "Voxel surface ({}): {} chunks, {} surface voxels",
            preset.label(),
            surface.chunks.len(),
            total_voxels
        );

        Self {
            world,
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
        }
    }

    /// Rebuild surface data if any chunk is dirty.
    pub fn update_surfaces_if_dirty(&mut self) {
        if self.world.any_dirty() {
            self.surface = cull::extract_surfaces(&self.world);
            self.world.clear_dirty();
            let total_voxels: u32 = self
                .surface
                .chunks
                .iter()
                .map(|c| c.voxel_end - c.voxel_start)
                .sum();
            eprintln!(
                "Voxel surface rebuilt: {} chunks, {} surface voxels",
                self.surface.chunks.len(),
                total_voxels
            );
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

    /// Build the voxel-native frame payload for VTE.
    pub fn build_voxel_frame_data(&self, cam_pos: [f32; 4]) -> VoxelFrameData {
        let mut chunk_headers = Vec::new();
        let mut occupancy_words = Vec::new();
        let mut material_words = Vec::new();
        let mut visible_chunk_indices = Vec::new();
        let render_dist_sq = RENDER_DISTANCE * RENDER_DISTANCE;

        let mut chunk_entries: Vec<_> = self.world.chunks.iter().collect();
        chunk_entries.sort_by_key(|(pos, _)| (pos.x, pos.y, pos.z, pos.w));

        for (chunk_pos, chunk) in chunk_entries {
            if chunk.is_empty() {
                continue;
            }

            // L0 culling: skip chunks outside the same camera radius used by tetra path.
            let chunk_min = [
                chunk_pos.x * CHUNK_SIZE as i32,
                chunk_pos.y * CHUNK_SIZE as i32,
                chunk_pos.z * CHUNK_SIZE as i32,
                chunk_pos.w * CHUNK_SIZE as i32,
            ];
            let chunk_max = [
                chunk_min[0] + CHUNK_SIZE as i32,
                chunk_min[1] + CHUNK_SIZE as i32,
                chunk_min[2] + CHUNK_SIZE as i32,
                chunk_min[3] + CHUNK_SIZE as i32,
            ];

            let mut dist_sq = 0.0f32;
            for d in 0..4 {
                let lo = chunk_min[d] as f32;
                let hi = chunk_max[d] as f32;
                let c = cam_pos[d];
                if c < lo {
                    let delta = lo - c;
                    dist_sq += delta * delta;
                } else if c > hi {
                    let delta = c - hi;
                    dist_sq += delta * delta;
                }
            }
            if dist_sq > render_dist_sq {
                continue;
            }

            let occupancy_word_offset = occupancy_words.len() as u32;
            occupancy_words.resize(occupancy_words.len() + OCCUPANCY_WORDS_PER_CHUNK, 0);
            let material_word_offset = material_words.len() as u32;
            material_words.resize(material_words.len() + MATERIAL_WORDS_PER_CHUNK, 0);

            for (voxel_idx, voxel) in chunk.voxels.iter().enumerate() {
                let mat_word_idx = material_word_offset as usize + (voxel_idx / 4);
                let mat_shift = ((voxel_idx & 3) * 8) as u32;
                material_words[mat_word_idx] |= (voxel.0 as u32) << mat_shift;
                if voxel.is_solid() {
                    let word_idx = occupancy_word_offset as usize + (voxel_idx / 32);
                    occupancy_words[word_idx] |= 1u32 << (voxel_idx % 32);
                }
            }

            let mut flags = 0u32;
            if chunk.is_empty() {
                flags |= GpuVoxelChunkHeader::FLAG_EMPTY;
            }
            if chunk.is_full() {
                flags |= GpuVoxelChunkHeader::FLAG_FULL;
            }

            let chunk_index = chunk_headers.len() as u32;
            chunk_headers.push(GpuVoxelChunkHeader {
                chunk_coord: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
                occupancy_word_offset,
                material_word_offset,
                flags,
                _padding: 0,
            });
            visible_chunk_indices.push(chunk_index);
        }

        VoxelFrameData {
            chunk_headers,
            occupancy_words,
            material_words,
            visible_chunk_indices,
        }
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
            surface,
            culled_instances: Vec::new(),
            cull_log_counter: 0,
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
}
