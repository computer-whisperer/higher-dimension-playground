use crate::voxel::cull::{self, SurfaceData};
use crate::voxel::worldgen;
use crate::voxel::VoxelType;

const RENDER_DISTANCE: f32 = 64.0;

pub struct Scene {
    pub world: crate::voxel::world::VoxelWorld,
    surface: SurfaceData,
    culled_instances: Vec<common::ModelInstance>,
}

impl Scene {
    pub fn new() -> Self {
        let world = worldgen::generate_flat_world(
            3,             // 3×3×3 chunks in X, Z, W
            VoxelType(3),  // grass
        );

        let surface = cull::extract_surfaces(&world);
        let total_voxels: u32 = surface.chunks.iter().map(|c| c.voxel_end - c.voxel_start).sum();
        eprintln!("Voxel surface: {} chunks, {} surface voxels", surface.chunks.len(), total_voxels);

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
            let total_voxels: u32 = self.surface.chunks.iter().map(|c| c.voxel_end - c.voxel_start).sum();
            eprintln!("Voxel surface rebuilt: {} chunks, {} surface voxels", self.surface.chunks.len(), total_voxels);
        }
    }

    /// Per-frame: cull and build ModelInstances for the current camera position.
    pub fn build_instances(&mut self, cam_pos: [f32; 4]) -> &[common::ModelInstance] {
        self.culled_instances.clear();
        cull::cull_and_build(&self.surface, cam_pos, RENDER_DISTANCE, &mut self.culled_instances);

        let (num_instances, num_tets) = cull::mesh_stats(&self.culled_instances);
        eprintln!("Culled: {num_instances} instances, {num_tets} tetrahedra");

        &self.culled_instances
    }
}
