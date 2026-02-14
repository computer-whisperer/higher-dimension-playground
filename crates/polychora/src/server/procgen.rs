use crate::shared::voxel::{Chunk, ChunkPos, VoxelType, CHUNK_SIZE};
use serde::Deserialize;
use std::sync::OnceLock;

const STRUCTURE_CELL_SIZE: i32 = 32;
const STRUCTURE_CELL_JITTER: i32 = 10;
const STRUCTURE_SPAWN_NUMERATOR: u64 = 1;
const STRUCTURE_SPAWN_DENOMINATOR: u64 = 32;
const STRUCTURE_ORIGIN_EXCLUSION_RADIUS: i32 = 16;
const STRUCTURE_HASH_SALT: u64 = 0x9f07_c9ab_33f2_3a11;
const STRUCTURE_PICK_SALT: u64 = 0x2d99_1f4e_47ba_8c6d;
const STRUCTURE_ROTATION_SALT: u64 = 0x54c7_9be0_2f61_0d93;
const JITTER_X_SALT: u64 = 0x1c69_b3f7_4d87_2ba1;
const JITTER_Z_SALT: u64 = 0x8ab3_d165_52cc_91f3;
const JITTER_W_SALT: u64 = 0xf2c6_0c4a_0a8a_d53d;
const ROTATION_VARIANTS: u64 = 48;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureBlueprintFile {
    name: String,
    #[serde(default = "default_spawn_weight")]
    spawn_weight: u32,
    #[serde(default)]
    ground_offset_y: i32,
    #[serde(default)]
    fills: Vec<StructureFill>,
    #[serde(default)]
    voxels: Vec<StructureVoxel>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureFill {
    min: [i32; 4],
    size: [i32; 4],
    material: u8,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureVoxel {
    offset: [i32; 4],
    material: u8,
}

#[derive(Clone, Debug)]
struct StructureBlueprint {
    spawn_weight: u32,
    ground_offset_y: i32,
    fills: Vec<StructureFill>,
    voxels: Vec<StructureVoxel>,
    min_offset: [i32; 4],
    max_offset: [i32; 4],
}

#[derive(Clone, Debug)]
struct StructureSet {
    blueprints: Vec<StructureBlueprint>,
    total_weight: u64,
    max_abs_offset_xzw: i32,
    min_world_y: i32,
    max_world_y: i32,
}

fn default_spawn_weight() -> u32 {
    1
}

impl StructureBlueprint {
    fn from_embedded_json(file_name: &str, json: &str) -> Self {
        let parsed: StructureBlueprintFile = serde_json::from_str(json)
            .unwrap_or_else(|error| panic!("invalid structure blueprint {file_name}: {error}"));
        if parsed.name.trim().is_empty() {
            panic!("structure blueprint {file_name} has an empty name");
        }

        if parsed.spawn_weight == 0 {
            panic!("structure blueprint {file_name} has spawn_weight=0");
        }

        let mut initialized = false;
        let mut min_offset = [0i32; 4];
        let mut max_offset = [0i32; 4];

        let mut include_point = |point: [i32; 4]| {
            if !initialized {
                min_offset = point;
                max_offset = point;
                initialized = true;
                return;
            }
            for axis in 0..4 {
                min_offset[axis] = min_offset[axis].min(point[axis]);
                max_offset[axis] = max_offset[axis].max(point[axis]);
            }
        };

        for fill in &parsed.fills {
            if fill.material == VoxelType::AIR.0 {
                continue;
            }
            if fill.size.iter().any(|size| *size <= 0) {
                panic!("structure blueprint {file_name} has non-positive fill size");
            }
            let fill_min = fill.min;
            let fill_max = [
                fill.min[0] + fill.size[0] - 1,
                fill.min[1] + fill.size[1] - 1,
                fill.min[2] + fill.size[2] - 1,
                fill.min[3] + fill.size[3] - 1,
            ];
            include_point(fill_min);
            include_point(fill_max);
        }

        for voxel in &parsed.voxels {
            if voxel.material == VoxelType::AIR.0 {
                continue;
            }
            include_point(voxel.offset);
        }

        if !initialized {
            panic!("structure blueprint {file_name} has no solid voxels");
        }

        Self {
            spawn_weight: parsed.spawn_weight,
            ground_offset_y: parsed.ground_offset_y,
            fills: parsed.fills,
            voxels: parsed.voxels,
            min_offset,
            max_offset,
        }
    }

    fn oriented_bounds(&self, orientation: u8) -> ([i32; 4], [i32; 4]) {
        let mut min_offset = [i32::MAX; 4];
        let mut max_offset = [i32::MIN; 4];
        let corners = [
            [
                self.min_offset[0],
                self.min_offset[1],
                self.min_offset[2],
                self.min_offset[3],
            ],
            [
                self.min_offset[0],
                self.min_offset[1],
                self.min_offset[2],
                self.max_offset[3],
            ],
            [
                self.min_offset[0],
                self.min_offset[1],
                self.max_offset[2],
                self.min_offset[3],
            ],
            [
                self.min_offset[0],
                self.min_offset[1],
                self.max_offset[2],
                self.max_offset[3],
            ],
            [
                self.max_offset[0],
                self.min_offset[1],
                self.min_offset[2],
                self.min_offset[3],
            ],
            [
                self.max_offset[0],
                self.min_offset[1],
                self.min_offset[2],
                self.max_offset[3],
            ],
            [
                self.max_offset[0],
                self.min_offset[1],
                self.max_offset[2],
                self.min_offset[3],
            ],
            [
                self.max_offset[0],
                self.min_offset[1],
                self.max_offset[2],
                self.max_offset[3],
            ],
            [
                self.min_offset[0],
                self.max_offset[1],
                self.min_offset[2],
                self.min_offset[3],
            ],
            [
                self.min_offset[0],
                self.max_offset[1],
                self.min_offset[2],
                self.max_offset[3],
            ],
            [
                self.min_offset[0],
                self.max_offset[1],
                self.max_offset[2],
                self.min_offset[3],
            ],
            [
                self.min_offset[0],
                self.max_offset[1],
                self.max_offset[2],
                self.max_offset[3],
            ],
            [
                self.max_offset[0],
                self.max_offset[1],
                self.min_offset[2],
                self.min_offset[3],
            ],
            [
                self.max_offset[0],
                self.max_offset[1],
                self.min_offset[2],
                self.max_offset[3],
            ],
            [
                self.max_offset[0],
                self.max_offset[1],
                self.max_offset[2],
                self.min_offset[3],
            ],
            [
                self.max_offset[0],
                self.max_offset[1],
                self.max_offset[2],
                self.max_offset[3],
            ],
        ];
        for corner in corners {
            let rotated = rotate_offset_xzw(corner, orientation);
            for axis in 0..4 {
                min_offset[axis] = min_offset[axis].min(rotated[axis]);
                max_offset[axis] = max_offset[axis].max(rotated[axis]);
            }
        }
        (min_offset, max_offset)
    }

    fn intersects_chunk_oriented(
        &self,
        origin: [i32; 4],
        orientation: u8,
        chunk_min: [i32; 4],
        chunk_max: [i32; 4],
    ) -> bool {
        let (oriented_min, oriented_max) = self.oriented_bounds(orientation);
        for axis in 0..4 {
            let min = origin[axis] + oriented_min[axis];
            let max = origin[axis] + oriented_max[axis];
            if max < chunk_min[axis] || min > chunk_max[axis] {
                return false;
            }
        }
        true
    }

    fn place_into_chunk_oriented(
        &self,
        origin: [i32; 4],
        orientation: u8,
        chunk_min: [i32; 4],
        chunk: &mut Chunk,
    ) {
        for fill in &self.fills {
            if fill.material == VoxelType::AIR.0 {
                continue;
            }

            let fill_min_rotated = rotate_offset_xzw(fill.min, orientation);
            let fill_max_local = [
                fill.min[0] + fill.size[0] - 1,
                fill.min[1] + fill.size[1] - 1,
                fill.min[2] + fill.size[2] - 1,
                fill.min[3] + fill.size[3] - 1,
            ];
            let fill_max_rotated = rotate_offset_xzw(fill_max_local, orientation);
            let fill_world_min = [
                origin[0] + fill_min_rotated[0].min(fill_max_rotated[0]),
                origin[1] + fill_min_rotated[1].min(fill_max_rotated[1]),
                origin[2] + fill_min_rotated[2].min(fill_max_rotated[2]),
                origin[3] + fill_min_rotated[3].min(fill_max_rotated[3]),
            ];
            let fill_world_max = [
                origin[0] + fill_min_rotated[0].max(fill_max_rotated[0]),
                origin[1] + fill_min_rotated[1].max(fill_max_rotated[1]),
                origin[2] + fill_min_rotated[2].max(fill_max_rotated[2]),
                origin[3] + fill_min_rotated[3].max(fill_max_rotated[3]),
            ];

            let mut loop_min = [0i32; 4];
            let mut loop_max = [0i32; 4];
            let mut empty_intersection = false;
            for axis in 0..4 {
                loop_min[axis] = fill_world_min[axis].max(chunk_min[axis]);
                loop_max[axis] = fill_world_max[axis].min(chunk_min[axis] + CHUNK_SIZE as i32 - 1);
                if loop_min[axis] > loop_max[axis] {
                    empty_intersection = true;
                    break;
                }
            }
            if empty_intersection {
                continue;
            }

            let voxel = VoxelType(fill.material);
            for wx in loop_min[0]..=loop_max[0] {
                for wy in loop_min[1]..=loop_max[1] {
                    for wz in loop_min[2]..=loop_max[2] {
                        for ww in loop_min[3]..=loop_max[3] {
                            let lx = (wx - chunk_min[0]) as usize;
                            let ly = (wy - chunk_min[1]) as usize;
                            let lz = (wz - chunk_min[2]) as usize;
                            let lw = (ww - chunk_min[3]) as usize;
                            chunk.set(lx, ly, lz, lw, voxel);
                        }
                    }
                }
            }
        }

        for voxel in &self.voxels {
            if voxel.material == VoxelType::AIR.0 {
                continue;
            }

            let rotated = rotate_offset_xzw(voxel.offset, orientation);
            let wx = origin[0] + rotated[0];
            let wy = origin[1] + rotated[1];
            let wz = origin[2] + rotated[2];
            let ww = origin[3] + rotated[3];

            if wx < chunk_min[0]
                || wx >= chunk_min[0] + CHUNK_SIZE as i32
                || wy < chunk_min[1]
                || wy >= chunk_min[1] + CHUNK_SIZE as i32
                || wz < chunk_min[2]
                || wz >= chunk_min[2] + CHUNK_SIZE as i32
                || ww < chunk_min[3]
                || ww >= chunk_min[3] + CHUNK_SIZE as i32
            {
                continue;
            }

            let lx = (wx - chunk_min[0]) as usize;
            let ly = (wy - chunk_min[1]) as usize;
            let lz = (wz - chunk_min[2]) as usize;
            let lw = (ww - chunk_min[3]) as usize;
            chunk.set(lx, ly, lz, lw, VoxelType(voxel.material));
        }
    }

    fn writes_any_voxel_in_chunk_oriented(
        &self,
        origin: [i32; 4],
        orientation: u8,
        chunk_min: [i32; 4],
        chunk_max: [i32; 4],
    ) -> bool {
        for fill in &self.fills {
            if fill.material == VoxelType::AIR.0 {
                continue;
            }

            let fill_min_rotated = rotate_offset_xzw(fill.min, orientation);
            let fill_max_local = [
                fill.min[0] + fill.size[0] - 1,
                fill.min[1] + fill.size[1] - 1,
                fill.min[2] + fill.size[2] - 1,
                fill.min[3] + fill.size[3] - 1,
            ];
            let fill_max_rotated = rotate_offset_xzw(fill_max_local, orientation);
            let fill_world_min = [
                origin[0] + fill_min_rotated[0].min(fill_max_rotated[0]),
                origin[1] + fill_min_rotated[1].min(fill_max_rotated[1]),
                origin[2] + fill_min_rotated[2].min(fill_max_rotated[2]),
                origin[3] + fill_min_rotated[3].min(fill_max_rotated[3]),
            ];
            let fill_world_max = [
                origin[0] + fill_min_rotated[0].max(fill_max_rotated[0]),
                origin[1] + fill_min_rotated[1].max(fill_max_rotated[1]),
                origin[2] + fill_min_rotated[2].max(fill_max_rotated[2]),
                origin[3] + fill_min_rotated[3].max(fill_max_rotated[3]),
            ];

            let mut intersects = true;
            for axis in 0..4 {
                if fill_world_max[axis] < chunk_min[axis] || fill_world_min[axis] > chunk_max[axis]
                {
                    intersects = false;
                    break;
                }
            }
            if intersects {
                return true;
            }
        }

        for voxel in &self.voxels {
            if voxel.material == VoxelType::AIR.0 {
                continue;
            }

            let rotated = rotate_offset_xzw(voxel.offset, orientation);
            let wx = origin[0] + rotated[0];
            let wy = origin[1] + rotated[1];
            let wz = origin[2] + rotated[2];
            let ww = origin[3] + rotated[3];
            if wx >= chunk_min[0]
                && wx <= chunk_max[0]
                && wy >= chunk_min[1]
                && wy <= chunk_max[1]
                && wz >= chunk_min[2]
                && wz <= chunk_max[2]
                && ww >= chunk_min[3]
                && ww <= chunk_max[3]
            {
                return true;
            }
        }

        false
    }
}

impl StructureSet {
    fn pick_blueprint_index(&self, mut roll: u64) -> usize {
        for (idx, blueprint) in self.blueprints.iter().enumerate() {
            let weight = u64::from(blueprint.spawn_weight.max(1));
            if roll < weight {
                return idx;
            }
            roll -= weight;
        }
        self.blueprints.len().saturating_sub(1)
    }
}

static STRUCTURE_SET: OnceLock<StructureSet> = OnceLock::new();

fn structure_set() -> &'static StructureSet {
    STRUCTURE_SET.get_or_init(|| {
        let sources = [
            (
                "cross_shrine.json",
                include_str!("../../assets/structures/cross_shrine.json"),
            ),
            (
                "hyper_arch.json",
                include_str!("../../assets/structures/hyper_arch.json"),
            ),
            (
                "tetra_spire.json",
                include_str!("../../assets/structures/tetra_spire.json"),
            ),
            (
                "sky_bridge.json",
                include_str!("../../assets/structures/sky_bridge.json"),
            ),
            (
                "ring_keep.json",
                include_str!("../../assets/structures/ring_keep.json"),
            ),
            (
                "terraced_pyramid.json",
                include_str!("../../assets/structures/terraced_pyramid.json"),
            ),
            (
                "cathedral_spine.json",
                include_str!("../../assets/structures/cathedral_spine.json"),
            ),
            (
                "lattice_hall.json",
                include_str!("../../assets/structures/lattice_hall.json"),
            ),
            (
                "shard_garden.json",
                include_str!("../../assets/structures/shard_garden.json"),
            ),
            (
                "portal_axis.json",
                include_str!("../../assets/structures/portal_axis.json"),
            ),
        ];

        let blueprints: Vec<_> = sources
            .iter()
            .map(|(name, source)| StructureBlueprint::from_embedded_json(name, source))
            .collect();

        let total_weight = blueprints
            .iter()
            .map(|blueprint| blueprint.spawn_weight.max(1) as u64)
            .sum::<u64>()
            .max(1);

        let max_abs_offset_xzw = blueprints
            .iter()
            .flat_map(|blueprint| {
                [
                    blueprint.min_offset[0].abs(),
                    blueprint.max_offset[0].abs(),
                    blueprint.min_offset[2].abs(),
                    blueprint.max_offset[2].abs(),
                    blueprint.min_offset[3].abs(),
                    blueprint.max_offset[3].abs(),
                ]
            })
            .max()
            .unwrap_or(0);

        let min_world_y = blueprints
            .iter()
            .map(|blueprint| blueprint.ground_offset_y + blueprint.min_offset[1])
            .min()
            .unwrap_or(0);
        let max_world_y = blueprints
            .iter()
            .map(|blueprint| blueprint.ground_offset_y + blueprint.max_offset[1])
            .max()
            .unwrap_or(0);

        StructureSet {
            blueprints,
            total_weight,
            max_abs_offset_xzw,
            min_world_y,
            max_world_y,
        }
    })
}

pub fn structure_chunk_y_bounds() -> (i32, i32) {
    let set = structure_set();
    let chunk_size = CHUNK_SIZE as i32;
    (
        set.min_world_y.div_euclid(chunk_size),
        set.max_world_y.div_euclid(chunk_size),
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct StructurePlacement {
    blueprint_idx: usize,
    origin: [i32; 4],
    orientation: u8,
}

fn chunk_bounds(chunk_pos: ChunkPos) -> ([i32; 4], [i32; 4]) {
    let chunk_size = CHUNK_SIZE as i32;
    let chunk_min = [
        chunk_pos.x * chunk_size,
        chunk_pos.y * chunk_size,
        chunk_pos.z * chunk_size,
        chunk_pos.w * chunk_size,
    ];
    let chunk_max = [
        chunk_min[0] + chunk_size - 1,
        chunk_min[1] + chunk_size - 1,
        chunk_min[2] + chunk_size - 1,
        chunk_min[3] + chunk_size - 1,
    ];
    (chunk_min, chunk_max)
}

fn rotate_offset_xzw(offset: [i32; 4], orientation: u8) -> [i32; 4] {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let horizontal = [offset[0], offset[2], offset[3]];
    let perm = PERMUTATIONS[(orientation as usize) % PERMUTATIONS.len()];
    let sign_bits = (orientation as usize / PERMUTATIONS.len()) & 0b111;

    let mut out_h = [0i32; 3];
    for out_axis in 0..3 {
        let src_axis = perm[out_axis];
        let mut value = horizontal[src_axis];
        if (sign_bits & (1 << out_axis)) != 0 {
            value = -value;
        }
        out_h[out_axis] = value;
    }
    [out_h[0], offset[1], out_h[1], out_h[2]]
}

fn collect_structure_placements_for_chunk(
    world_seed: u64,
    chunk_pos: ChunkPos,
) -> Vec<StructurePlacement> {
    let set = structure_set();
    let (min_chunk_y, max_chunk_y) = structure_chunk_y_bounds();
    if chunk_pos.y < min_chunk_y || chunk_pos.y > max_chunk_y {
        return Vec::new();
    }

    let chunk_size = CHUNK_SIZE as i32;
    let (chunk_min, chunk_max) = chunk_bounds(chunk_pos);

    let search_margin = STRUCTURE_CELL_JITTER + set.max_abs_offset_xzw + chunk_size;
    let cell_min_x = (chunk_min[0] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_x = (chunk_max[0] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_z = (chunk_min[2] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_z = (chunk_max[2] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_w = (chunk_min[3] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_w = (chunk_max[3] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;

    let mut placements = Vec::new();

    for cell_x in cell_min_x..=cell_max_x {
        for cell_z in cell_min_z..=cell_max_z {
            for cell_w in cell_min_w..=cell_max_w {
                let cell_hash =
                    hash_structure_cell(world_seed, cell_x, cell_z, cell_w, STRUCTURE_HASH_SALT);
                if cell_hash % STRUCTURE_SPAWN_DENOMINATOR >= STRUCTURE_SPAWN_NUMERATOR {
                    continue;
                }

                let origin_x =
                    cell_x * STRUCTURE_CELL_SIZE + jitter_from_hash(cell_hash ^ JITTER_X_SALT);
                let origin_z =
                    cell_z * STRUCTURE_CELL_SIZE + jitter_from_hash(cell_hash ^ JITTER_Z_SALT);
                let origin_w =
                    cell_w * STRUCTURE_CELL_SIZE + jitter_from_hash(cell_hash ^ JITTER_W_SALT);

                if origin_x.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                    && origin_z.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                    && origin_w.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                {
                    continue;
                }

                let pick_roll = splitmix64(cell_hash ^ STRUCTURE_PICK_SALT) % set.total_weight;
                let blueprint_idx = set.pick_blueprint_index(pick_roll);
                let blueprint = &set.blueprints[blueprint_idx];
                let orientation =
                    (splitmix64(cell_hash ^ STRUCTURE_ROTATION_SALT) % ROTATION_VARIANTS) as u8;
                let origin = [origin_x, blueprint.ground_offset_y, origin_z, origin_w];
                if !blueprint.intersects_chunk_oriented(origin, orientation, chunk_min, chunk_max) {
                    continue;
                }

                placements.push(StructurePlacement {
                    blueprint_idx,
                    origin,
                    orientation,
                });
            }
        }
    }

    placements
}

pub fn generate_structure_chunk(world_seed: u64, chunk_pos: ChunkPos) -> Option<Chunk> {
    let set = structure_set();
    let (chunk_min, _) = chunk_bounds(chunk_pos);
    let placements = collect_structure_placements_for_chunk(world_seed, chunk_pos);

    let mut chunk = Chunk::new();
    for placement in placements {
        let blueprint = &set.blueprints[placement.blueprint_idx];
        blueprint.place_into_chunk_oriented(
            placement.origin,
            placement.orientation,
            chunk_min,
            &mut chunk,
        );
    }

    if chunk.is_empty() {
        None
    } else {
        Some(chunk)
    }
}

pub fn structure_chunk_has_content(world_seed: u64, chunk_pos: ChunkPos) -> bool {
    let set = structure_set();
    let (chunk_min, chunk_max) = chunk_bounds(chunk_pos);
    collect_structure_placements_for_chunk(world_seed, chunk_pos)
        .into_iter()
        .any(|placement| {
            let blueprint = &set.blueprints[placement.blueprint_idx];
            blueprint.writes_any_voxel_in_chunk_oriented(
                placement.origin,
                placement.orientation,
                chunk_min,
                chunk_max,
            )
        })
}

fn jitter_from_hash(hash: u64) -> i32 {
    let span = (STRUCTURE_CELL_JITTER * 2 + 1) as u64;
    (splitmix64(hash) % span) as i32 - STRUCTURE_CELL_JITTER
}

fn hash_structure_cell(seed: u64, x: i32, z: i32, w: i32, salt: u64) -> u64 {
    let mut mixed = seed ^ salt;
    mixed ^= (x as i64 as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
    mixed = splitmix64(mixed);
    mixed ^= (z as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    mixed = splitmix64(mixed);
    mixed ^= (w as i64 as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    splitmix64(mixed)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_chunk_generation_is_seed_deterministic() {
        let chunk_a = generate_structure_chunk(42, ChunkPos::new(6, 0, 4, -3));
        let chunk_b = generate_structure_chunk(42, ChunkPos::new(6, 0, 4, -3));

        assert_eq!(chunk_a.is_some(), chunk_b.is_some());
        if let (Some(a), Some(b)) = (chunk_a, chunk_b) {
            assert_eq!(a.solid_count, b.solid_count);
            assert_eq!(a.voxels[..], b.voxels[..]);
        }
    }

    #[test]
    fn structure_generation_produces_non_empty_chunks() {
        let mut found_non_empty = false;
        let (min_y, max_y) = structure_chunk_y_bounds();

        for x in -8..=8 {
            for z in -8..=8 {
                for w in -8..=8 {
                    for y in min_y..=max_y {
                        if generate_structure_chunk(1337, ChunkPos::new(x, y, z, w)).is_some() {
                            found_non_empty = true;
                            break;
                        }
                    }
                    if found_non_empty {
                        break;
                    }
                }
                if found_non_empty {
                    break;
                }
            }
            if found_non_empty {
                break;
            }
        }

        assert!(
            found_non_empty,
            "expected at least one structure chunk to be generated"
        );
    }

    #[test]
    fn structure_blueprints_are_loaded() {
        let set = structure_set();
        assert!(!set.blueprints.is_empty());
    }

    #[test]
    fn blueprints_can_span_multiple_chunks() {
        let set = structure_set();
        let mut found_spanning = false;

        for blueprint in &set.blueprints {
            let span_x = blueprint.max_offset[0] - blueprint.min_offset[0] + 1;
            if span_x <= CHUNK_SIZE as i32 {
                continue;
            }

            // Place one boundary voxel-plane at x=8 so content can hit both chunk x=0 and x=1.
            let origin = [
                8 - blueprint.max_offset[0],
                -blueprint.min_offset[1],
                -blueprint.min_offset[2],
                -blueprint.min_offset[3],
            ];

            let mut left_chunk = Chunk::new();
            let mut right_chunk = Chunk::new();
            blueprint.place_into_chunk_oriented(origin, 0, [0, 0, 0, 0], &mut left_chunk);
            blueprint.place_into_chunk_oriented(
                origin,
                0,
                [CHUNK_SIZE as i32, 0, 0, 0],
                &mut right_chunk,
            );

            if !left_chunk.is_empty() && !right_chunk.is_empty() {
                found_spanning = true;
                break;
            }
        }

        assert!(
            found_spanning,
            "expected at least one blueprint to place non-empty voxels in adjacent chunks"
        );
    }

    #[test]
    fn generated_placements_span_adjacent_chunks() {
        let (min_y, max_y) = structure_chunk_y_bounds();
        let mut found_spanning = false;

        'scan: for x in -12..=12 {
            for z in -12..=12 {
                for w in -12..=12 {
                    for y in min_y..=max_y {
                        let left = ChunkPos::new(x, y, z, w);
                        let right = ChunkPos::new(x + 1, y, z, w);
                        let left_placements = collect_structure_placements_for_chunk(2026, left);
                        if left_placements.is_empty() {
                            continue;
                        }
                        let right_placements = collect_structure_placements_for_chunk(2026, right);
                        if right_placements.is_empty() {
                            continue;
                        }

                        if left_placements
                            .iter()
                            .any(|placement| right_placements.contains(placement))
                        {
                            found_spanning = true;
                            break 'scan;
                        }
                    }
                }
            }
        }

        assert!(
            found_spanning,
            "expected at least one generated placement to intersect adjacent chunks"
        );
    }

    #[test]
    fn chunk_has_content_matches_chunk_generation() {
        let seed = 777_u64;
        let (min_y, max_y) = structure_chunk_y_bounds();
        for x in -8..=8 {
            for z in -8..=8 {
                for w in -8..=8 {
                    for y in min_y..=max_y {
                        let chunk_pos = ChunkPos::new(x, y, z, w);
                        let has_content = structure_chunk_has_content(seed, chunk_pos);
                        let generated = generate_structure_chunk(seed, chunk_pos).is_some();
                        assert_eq!(
                            has_content, generated,
                            "content mismatch for chunk ({x}, {y}, {z}, {w})"
                        );
                    }
                }
            }
        }
    }
}
