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
