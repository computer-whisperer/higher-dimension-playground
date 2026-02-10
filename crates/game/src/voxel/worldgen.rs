use super::chunk::Chunk;
use super::world::VoxelWorld;
use super::{ChunkPos, VoxelType, CHUNK_SIZE};

/// Generate a flat 4D ground world — a single-voxel-thick floor at y = -1.
///
/// - `extent_xzw`: number of chunks in each of X, Z, W (centered around origin)
/// - `material`: material for the floor voxels
pub fn generate_flat_world(extent_xzw: i32, material: VoxelType) -> VoxelWorld {
    let mut world = VoxelWorld::new();
    let half = extent_xzw / 2;

    // Floor sits at y = -1 (the top voxel of chunk cy = -1)
    let cy = -1i32;
    let local_y = CHUNK_SIZE - 1; // y = -1 maps to local y = 7

    for cx in -half..extent_xzw - half {
        for cz in -half..extent_xzw - half {
            for cw in -half..extent_xzw - half {
                let mut chunk = Chunk::new();
                for x in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        for w in 0..CHUNK_SIZE {
                            chunk.set(x, local_y, z, w, material);
                        }
                    }
                }
                world.insert_chunk(ChunkPos::new(cx, cy, cz, cw), chunk);
            }
        }
    }

    world
}

fn fill_hypercube(world: &mut VoxelWorld, min: [i32; 4], edge: i32, material: VoxelType) {
    for x in min[0]..(min[0] + edge) {
        for y in min[1]..(min[1] + edge) {
            for z in min[2]..(min[2] + edge) {
                for w in min[3]..(min[3] + edge) {
                    world.set_voxel(x, y, z, w, material);
                }
            }
        }
    }
}

/// Generate a voxelized version of the demo "cube layout":
/// - 2x2x2x2 outer block lattice (colored materials 1..=5 cycling)
/// - one bright center block (material 13)
///
/// This is intended for reproducible VTE quality sweeps.
pub fn generate_demo_cube_layout_world() -> VoxelWorld {
    let mut world = VoxelWorld::new();
    let mut texture_rot = 0u8;

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                for w in 0..2 {
                    let base = [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2];
                    let material = VoxelType((texture_rot % 5) + 1);
                    fill_hypercube(&mut world, base, 2, material);
                    texture_rot = (texture_rot + 1) % 5;
                }
            }
        }
    }

    // Central bright cube, analogous to the demo light block.
    fill_hypercube(&mut world, [0, 0, 0, 0], 2, VoxelType(13));

    world
}
