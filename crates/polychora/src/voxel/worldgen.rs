use super::world::{BaseWorldKind, VoxelWorld};
use super::VoxelType;

const SHOWCASE_MATERIALS: [u8; 37] = [
    15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 45, 50, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
];

/// Generate a flat 4D ground world as a two-voxel-thick slab.
///
/// - `_extent_xzw`: retained for API compatibility; no longer used because ground is implicit
/// - `material`: material for the replicated floor voxels
pub fn generate_flat_world(_extent_xzw: i32, material: VoxelType) -> VoxelWorld {
    let mut world = VoxelWorld::new_with_base(BaseWorldKind::FlatFloor { material });

    place_material_showcase(&mut world, [-10, 0, -14, -4]);

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

fn place_material_showcase(world: &mut VoxelWorld, origin: [i32; 4]) {
    for (idx, material) in SHOWCASE_MATERIALS.iter().copied().enumerate() {
        let col = (idx % 6) as i32;
        let row = (idx / 6) as i32;
        let min = [
            origin[0] + col * 4,
            origin[1],
            origin[2] + row * 4,
            origin[3],
        ];
        fill_hypercube(world, min, 2, VoxelType(material));
    }
}

/// Generate the menu/demo showcase world:
/// - one cube at each 4D corner of a 2x2x2x2 lattice
/// - one bright center cube
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

    fill_hypercube(&mut world, [0, 0, 0, 0], 2, VoxelType(13));

    world
}
