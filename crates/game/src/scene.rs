use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::worldgen;
use crate::voxel::{VoxelType, CHUNK_SIZE, CHUNK_VOLUME};
use higher_dimension_playground::render::{GpuVoxelChunkHeader, VoxelFrameInput};

const RENDER_DISTANCE: f32 = 64.0;
const OCCUPANCY_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 32;
const MATERIAL_WORDS_PER_CHUNK: usize = CHUNK_VOLUME / 4; // packed 4x u8 per u32

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
}

impl Scene {
    pub fn new() -> Self {
        let world = worldgen::generate_flat_world(
            3,            // 3×3×3 chunks in X, Z, W
            VoxelType(3), // grass
        );

        let surface = cull::extract_surfaces(&world);
        let total_voxels: u32 = surface
            .chunks
            .iter()
            .map(|c| c.voxel_end - c.voxel_start)
            .sum();
        eprintln!(
            "Voxel surface: {} chunks, {} surface voxels",
            surface.chunks.len(),
            total_voxels
        );

        Self {
            world,
            surface,
            culled_instances: Vec::new(),
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
        eprintln!("Culled: {num_instances} instances, {num_tets} tetrahedra");

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
}
