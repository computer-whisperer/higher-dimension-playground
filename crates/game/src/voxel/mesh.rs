use higher_dimension_playground::matrix_operations::translate_matrix_4d;

use super::world::VoxelWorld;
use super::{Face4D, CHUNK_SIZE};

/// Generate ModelInstances for all visible voxels in the world.
///
/// Each solid voxel with at least one exposed face becomes one ModelInstance.
/// Only faces adjacent to air get their material ID set; interior faces are 0
/// (which the rasterizer culls via early-return).
pub fn generate_mesh(world: &VoxelWorld) -> Vec<common::ModelInstance> {
    let mut instances = Vec::new();

    for (&chunk_pos, chunk) in &world.chunks {
        if chunk.is_empty() {
            continue;
        }

        let base_x = chunk_pos.x * CHUNK_SIZE as i32;
        let base_y = chunk_pos.y * CHUNK_SIZE as i32;
        let base_z = chunk_pos.z * CHUNK_SIZE as i32;
        let base_w = chunk_pos.w * CHUNK_SIZE as i32;

        for lw in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                for ly in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        let voxel = chunk.get(lx, ly, lz, lw);
                        if voxel.is_air() {
                            continue;
                        }

                        let wx = base_x + lx as i32;
                        let wy = base_y + ly as i32;
                        let wz = base_z + lz as i32;
                        let ww = base_w + lw as i32;

                        // Check each face — if neighbor is air, expose this face
                        let mut cell_material_ids = [0u32; 8];
                        let mut any_exposed = false;

                        for face in Face4D::ALL {
                            let [dx, dy, dz, dw] = face.neighbor_offset();
                            let neighbor = world.get_voxel(wx + dx, wy + dy, wz + dz, ww + dw);
                            if neighbor.is_air() {
                                cell_material_ids[face.cell_id()] = voxel.0 as u32;
                                any_exposed = true;
                            }
                        }

                        if !any_exposed {
                            continue;
                        }

                        // Unit tesseract model is [0,1]^4, translation positions it
                        let model_transform =
                            translate_matrix_4d(wx as f32, wy as f32, wz as f32, ww as f32);

                        instances.push(common::ModelInstance {
                            model_transform: model_transform.into(),
                            cell_material_ids,
                        });
                    }
                }
            }
        }
    }

    // Sort by chunk for better GPU cache locality
    instances.sort_unstable_by_key(|inst| {
        // Extract translation from the transform matrix (column 4, rows 0-3)
        let tx = inst.model_transform[[0, 4]] as i32;
        let ty = inst.model_transform[[1, 4]] as i32;
        let tz = inst.model_transform[[2, 4]] as i32;
        let tw = inst.model_transform[[3, 4]] as i32;
        (tw, tz, ty, tx)
    });

    instances
}

/// Report mesh statistics.
pub fn mesh_stats(instances: &[common::ModelInstance]) -> (usize, usize) {
    let num_instances = instances.len();
    // Each exposed face = 6 tetrahedra (from tesseract cell decomposition)
    let num_tets: usize = instances
        .iter()
        .map(|inst| {
            inst.cell_material_ids
                .iter()
                .filter(|&&id| id != 0)
                .count()
                * 6
        })
        .sum();
    (num_instances, num_tets)
}
