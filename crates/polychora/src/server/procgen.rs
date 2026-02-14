use crate::shared::voxel::{Chunk, ChunkPos, VoxelType, CHUNK_SIZE};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::HashSet;
use std::io::Read;
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

const MAZE_CELL_SIZE: i32 = 128;
const MAZE_CELL_JITTER: i32 = 20;
const MAZE_SPAWN_NUMERATOR: u64 = 1;
const MAZE_SPAWN_DENOMINATOR: u64 = 20;
const MAZE_ORIGIN_EXCLUSION_RADIUS: i32 = 24;
const MAZE_HASH_SALT: u64 = 0x6f1d_05ce_294a_719b;
const MAZE_LAYOUT_SALT: u64 = 0x49ec_66d6_0d13_9e75;
const MAZE_PARENT_SALT: u64 = 0x932f_b43c_f1f9_746d;
const MAZE_JITTER_X_SALT: u64 = 0x5fa7_6d28_0f8b_81c3;
const MAZE_JITTER_Z_SALT: u64 = 0x8af0_1f7a_cc99_20be;
const MAZE_JITTER_W_SALT: u64 = 0x163a_8b39_b0a2_6741;
const MAZE_GATE_X_NEG_Z_SALT: u64 = 0x7270_6f63_6765_6e31;
const MAZE_GATE_X_NEG_W_SALT: u64 = 0x7270_6f63_6765_6e32;
const MAZE_GATE_X_POS_Z_SALT: u64 = 0x7270_6f63_6765_6e33;
const MAZE_GATE_X_POS_W_SALT: u64 = 0x7270_6f63_6765_6e34;
const MAZE_GATE_Z_NEG_X_SALT: u64 = 0x7270_6f63_6765_6e35;
const MAZE_GATE_Z_NEG_W_SALT: u64 = 0x7270_6f63_6765_6e36;
const MAZE_GATE_Z_POS_X_SALT: u64 = 0x7270_6f63_6765_6e37;
const MAZE_GATE_Z_POS_W_SALT: u64 = 0x7270_6f63_6765_6e38;
const MAZE_GATE_W_NEG_X_SALT: u64 = 0x7270_6f63_6765_6e39;
const MAZE_GATE_W_NEG_Z_SALT: u64 = 0x7270_6f63_6765_6e3a;
const MAZE_GATE_W_POS_X_SALT: u64 = 0x7270_6f63_6765_6e3b;
const MAZE_GATE_W_POS_Z_SALT: u64 = 0x7270_6f63_6765_6e3c;
const MAZE_GRID_CELLS: i32 = 11;
const MAZE_STRIDE: i32 = 4;
const MAZE_SPAN: i32 = MAZE_GRID_CELLS * MAZE_STRIDE + 1;
const MAZE_HALF_SPAN: i32 = MAZE_SPAN / 2;
const MAZE_HEIGHT: i32 = 7;
const MAZE_WORLD_Y_MIN: i32 = 0;
const MAZE_FLOOR_MATERIAL: u8 = 55;
const MAZE_CEILING_MATERIAL: u8 = 63;
const MAZE_WALL_MATERIAL: u8 = 66;
const MAZE_GATE_FRAME_MATERIAL: u8 = 68;
const MAZE_BEACON_MATERIAL: u8 = 64;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureBlueprintFile {
    name: String,
    #[serde(default)]
    description: Option<String>,
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
    fn from_embedded_json_bytes(file_name: &str, json: &[u8]) -> Self {
        let parsed: StructureBlueprintFile = serde_json::from_slice(json)
            .unwrap_or_else(|error| panic!("invalid structure blueprint {file_name}: {error}"));
        if parsed.name.trim().is_empty() {
            panic!("structure blueprint {file_name} has an empty name");
        }
        if let Some(description) = parsed.description.as_deref() {
            if description.trim().is_empty() {
                panic!("structure blueprint {file_name} has an empty description");
            }
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

    fn from_embedded_source(file_name: &str, source: &[u8]) -> Self {
        if let Some(base_name) = file_name.strip_suffix(".gz") {
            let mut decoder = GzDecoder::new(source);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .unwrap_or_else(|error| {
                    panic!("failed to decompress structure blueprint {file_name}: {error}")
                });
            Self::from_embedded_json_bytes(base_name, &decompressed)
        } else {
            Self::from_embedded_json_bytes(file_name, source)
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

    fn intersects_chunk(&self, origin: [i32; 4], chunk_min: [i32; 4], chunk_max: [i32; 4]) -> bool {
        for axis in 0..4 {
            let min = origin[axis] + self.min_offset[axis];
            let max = origin[axis] + self.max_offset[axis];
            if max < chunk_min[axis] || min > chunk_max[axis] {
                return false;
            }
        }
        true
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
        let sources: [(&str, &[u8]); 18] = [
            (
                "cross_shrine.json",
                &include_bytes!("../../assets/structures/cross_shrine.json")[..],
            ),
            (
                "hyper_arch.json",
                &include_bytes!("../../assets/structures/hyper_arch.json")[..],
            ),
            (
                "tetra_spire.json",
                &include_bytes!("../../assets/structures/tetra_spire.json")[..],
            ),
            (
                "sky_bridge.json",
                &include_bytes!("../../assets/structures/sky_bridge.json")[..],
            ),
            (
                "ring_keep.json",
                &include_bytes!("../../assets/structures/ring_keep.json")[..],
            ),
            (
                "terraced_pyramid.json",
                &include_bytes!("../../assets/structures/terraced_pyramid.json")[..],
            ),
            (
                "cathedral_spine.json",
                &include_bytes!("../../assets/structures/cathedral_spine.json")[..],
            ),
            (
                "lattice_hall.json",
                &include_bytes!("../../assets/structures/lattice_hall.json")[..],
            ),
            (
                "shard_garden.json",
                &include_bytes!("../../assets/structures/shard_garden.json")[..],
            ),
            (
                "portal_axis.json",
                &include_bytes!("../../assets/structures/portal_axis.json")[..],
            ),
            (
                "resonance_forge.json",
                &include_bytes!("../../assets/structures/resonance_forge.json")[..],
            ),
            (
                "void_colonnade.json",
                &include_bytes!("../../assets/structures/void_colonnade.json")[..],
            ),
            (
                "lumen_orrery.json",
                &include_bytes!("../../assets/structures/lumen_orrery.json")[..],
            ),
            (
                "clifford_atrium.json",
                &include_bytes!("../../assets/structures/clifford_atrium.json")[..],
            ),
            (
                "braided_transit.json",
                &include_bytes!("../../assets/structures/braided_transit.json")[..],
            ),
            (
                "phase_ladder.json",
                &include_bytes!("../../assets/structures/phase_ladder.json")[..],
            ),
            (
                "orthoplex_nexus.json",
                &include_bytes!("../../assets/structures/orthoplex_nexus.json")[..],
            ),
            (
                "hyper_maze.json.gz",
                &include_bytes!("../../assets/structures/hyper_maze.json.gz")[..],
            ),
        ];

        let blueprints: Vec<_> = sources
            .iter()
            .map(|(name, source)| StructureBlueprint::from_embedded_source(name, source))
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
    let maze_max_world_y = MAZE_WORLD_Y_MIN + MAZE_HEIGHT - 1;
    let min_world_y = set.min_world_y.min(MAZE_WORLD_Y_MIN);
    let max_world_y = set.max_world_y.max(maze_max_world_y);
    (
        min_world_y.div_euclid(chunk_size),
        max_world_y.div_euclid(chunk_size),
    )
}

pub fn structure_chunk_y_bounds_for_scale(chunk_scale: i32) -> (i32, i32) {
    let (min_chunk_y, max_chunk_y) = structure_chunk_y_bounds();
    let scale = chunk_scale.max(1);
    (min_chunk_y.div_euclid(scale), max_chunk_y.div_euclid(scale))
}

pub type StructureCell = [i32; 3];

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct StructurePlacement {
    blueprint_idx: usize,
    origin: [i32; 4],
    orientation: u8,
    cell: StructureCell,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct MazePlacement {
    origin: [i32; 4],
    layout_seed: u64,
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
                    cell: [cell_x, cell_z, cell_w],
                });
            }
        }
    }

    placements
}

fn placement_allowed(
    placement: &StructurePlacement,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    blocked_cells
        .map(|blocked| !blocked.contains(&placement.cell))
        .unwrap_or(true)
}

pub fn structure_cells_affecting_chunk(world_seed: u64, chunk_pos: ChunkPos) -> Vec<StructureCell> {
    let mut unique = HashSet::new();
    for placement in collect_structure_placements_for_chunk(world_seed, chunk_pos) {
        unique.insert(placement.cell);
    }
    let mut out: Vec<_> = unique.into_iter().collect();
    out.sort_unstable();
    out
}

fn maze_bounds(origin: [i32; 4]) -> ([i32; 4], [i32; 4]) {
    (
        [
            origin[0] - MAZE_HALF_SPAN,
            MAZE_WORLD_Y_MIN,
            origin[2] - MAZE_HALF_SPAN,
            origin[3] - MAZE_HALF_SPAN,
        ],
        [
            origin[0] + MAZE_HALF_SPAN,
            MAZE_WORLD_Y_MIN + MAZE_HEIGHT - 1,
            origin[2] + MAZE_HALF_SPAN,
            origin[3] + MAZE_HALF_SPAN,
        ],
    )
}

fn maze_intersects_chunk(origin: [i32; 4], chunk_min: [i32; 4], chunk_max: [i32; 4]) -> bool {
    let (maze_min, maze_max) = maze_bounds(origin);
    for axis in 0..4 {
        if maze_max[axis] < chunk_min[axis] || maze_min[axis] > chunk_max[axis] {
            return false;
        }
    }
    true
}

fn maze_layout_seed(world_seed: u64, origin: [i32; 4]) -> u64 {
    let mut seed = hash_structure_cell(
        world_seed,
        origin[0],
        origin[2],
        origin[3],
        MAZE_LAYOUT_SALT,
    );
    seed ^= (origin[1] as i64 as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    splitmix64(seed)
}

fn maze_gate_cell(layout_seed: u64, salt: u64) -> i32 {
    (splitmix64(layout_seed ^ salt) % MAZE_GRID_CELLS as u64) as i32
}

fn maze_gate_band(cell_idx: i32) -> (i32, i32) {
    let start = cell_idx * MAZE_STRIDE + 1;
    (start, start + 2)
}

fn maze_gate_open(u0: i32, u1: i32, gate0: i32, gate1: i32) -> bool {
    let (g0_min, g0_max) = maze_gate_band(gate0);
    let (g1_min, g1_max) = maze_gate_band(gate1);
    u0 >= g0_min && u0 <= g0_max && u1 >= g1_min && u1 <= g1_max
}

fn maze_cell_in_bounds(cell: [i32; 3]) -> bool {
    cell[0] >= 0
        && cell[0] < MAZE_GRID_CELLS
        && cell[1] >= 0
        && cell[1] < MAZE_GRID_CELLS
        && cell[2] >= 0
        && cell[2] < MAZE_GRID_CELLS
}

fn maze_parent_cell(layout_seed: u64, cell: [i32; 3]) -> Option<[i32; 3]> {
    let center = MAZE_GRID_CELLS / 2;
    if cell == [center, center, center] {
        return None;
    }

    let mut candidates = [[0i32; 3]; 3];
    let mut count = 0usize;
    if cell[0] != center {
        candidates[count] = [
            cell[0] + if cell[0] < center { 1 } else { -1 },
            cell[1],
            cell[2],
        ];
        count += 1;
    }
    if cell[1] != center {
        candidates[count] = [
            cell[0],
            cell[1] + if cell[1] < center { 1 } else { -1 },
            cell[2],
        ];
        count += 1;
    }
    if cell[2] != center {
        candidates[count] = [
            cell[0],
            cell[1],
            cell[2] + if cell[2] < center { 1 } else { -1 },
        ];
        count += 1;
    }

    let roll_seed = hash_structure_cell(layout_seed, cell[0], cell[1], cell[2], MAZE_PARENT_SALT);
    let idx = (splitmix64(roll_seed) % count as u64) as usize;
    Some(candidates[idx])
}

fn maze_edge_open(layout_seed: u64, a: [i32; 3], b: [i32; 3]) -> bool {
    if !maze_cell_in_bounds(a) || !maze_cell_in_bounds(b) {
        return false;
    }

    let distance = (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs();
    if distance != 1 {
        return false;
    }

    maze_parent_cell(layout_seed, a) == Some(b) || maze_parent_cell(layout_seed, b) == Some(a)
}

fn collect_maze_placements_for_chunk(world_seed: u64, chunk_pos: ChunkPos) -> Vec<MazePlacement> {
    let maze_min_chunk_y = MAZE_WORLD_Y_MIN.div_euclid(CHUNK_SIZE as i32);
    let maze_max_chunk_y = (MAZE_WORLD_Y_MIN + MAZE_HEIGHT - 1).div_euclid(CHUNK_SIZE as i32);
    if chunk_pos.y < maze_min_chunk_y || chunk_pos.y > maze_max_chunk_y {
        return Vec::new();
    }

    let chunk_size = CHUNK_SIZE as i32;
    let (chunk_min, chunk_max) = chunk_bounds(chunk_pos);

    let search_margin = MAZE_CELL_JITTER + MAZE_HALF_SPAN + chunk_size;
    let cell_min_x = (chunk_min[0] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_x = (chunk_max[0] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;
    let cell_min_z = (chunk_min[2] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_z = (chunk_max[2] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;
    let cell_min_w = (chunk_min[3] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_w = (chunk_max[3] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;

    let mut placements = Vec::new();

    for cell_x in cell_min_x..=cell_max_x {
        for cell_z in cell_min_z..=cell_max_z {
            for cell_w in cell_min_w..=cell_max_w {
                let cell_hash =
                    hash_structure_cell(world_seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                    continue;
                }

                let origin_x = cell_x * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_X_SALT,
                        MAZE_CELL_JITTER,
                    );
                let origin_z = cell_z * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_Z_SALT,
                        MAZE_CELL_JITTER,
                    );
                let origin_w = cell_w * MAZE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ MAZE_JITTER_W_SALT,
                        MAZE_CELL_JITTER,
                    );
                if origin_x.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                    && origin_z.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                    && origin_w.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                {
                    continue;
                }

                let origin = [origin_x, MAZE_WORLD_Y_MIN, origin_z, origin_w];
                if !maze_intersects_chunk(origin, chunk_min, chunk_max) {
                    continue;
                }

                placements.push(MazePlacement {
                    origin,
                    layout_seed: maze_layout_seed(world_seed, origin),
                });
            }
        }
    }

    placements
}

fn place_maze_into_chunk(placement: MazePlacement, chunk_min: [i32; 4], chunk: &mut Chunk) {
    let chunk_max = [
        chunk_min[0] + CHUNK_SIZE as i32 - 1,
        chunk_min[1] + CHUNK_SIZE as i32 - 1,
        chunk_min[2] + CHUNK_SIZE as i32 - 1,
        chunk_min[3] + CHUNK_SIZE as i32 - 1,
    ];
    if !maze_intersects_chunk(placement.origin, chunk_min, chunk_max) {
        return;
    }

    let (maze_min, maze_max) = maze_bounds(placement.origin);
    let mut loop_min = [0i32; 4];
    let mut loop_max = [0i32; 4];
    for axis in 0..4 {
        loop_min[axis] = maze_min[axis].max(chunk_min[axis]);
        loop_max[axis] = maze_max[axis].min(chunk_max[axis]);
        if loop_min[axis] > loop_max[axis] {
            return;
        }
    }

    let layout_seed = placement.layout_seed;
    let x_neg_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_X_NEG_Z_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_X_NEG_W_SALT),
    ];
    let x_pos_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_X_POS_Z_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_X_POS_W_SALT),
    ];
    let z_neg_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_Z_NEG_X_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_Z_NEG_W_SALT),
    ];
    let z_pos_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_Z_POS_X_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_Z_POS_W_SALT),
    ];
    let w_neg_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_W_NEG_X_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_W_NEG_Z_SALT),
    ];
    let w_pos_gate = [
        maze_gate_cell(layout_seed, MAZE_GATE_W_POS_X_SALT),
        maze_gate_cell(layout_seed, MAZE_GATE_W_POS_Z_SALT),
    ];
    let center_u = (MAZE_GRID_CELLS / 2) * MAZE_STRIDE + 2;

    for wx in loop_min[0]..=loop_max[0] {
        for wy in loop_min[1]..=loop_max[1] {
            for wz in loop_min[2]..=loop_max[2] {
                for ww in loop_min[3]..=loop_max[3] {
                    let ux = wx - maze_min[0];
                    let uy = wy - MAZE_WORLD_Y_MIN;
                    let uz = wz - maze_min[2];
                    let uw = ww - maze_min[3];

                    let material = if uy == 0 {
                        Some(MAZE_FLOOR_MATERIAL)
                    } else if uy == MAZE_HEIGHT - 1 {
                        Some(MAZE_CEILING_MATERIAL)
                    } else {
                        let on_x_neg = ux == 0;
                        let on_x_pos = ux == MAZE_SPAN - 1;
                        let on_z_neg = uz == 0;
                        let on_z_pos = uz == MAZE_SPAN - 1;
                        let on_w_neg = uw == 0;
                        let on_w_pos = uw == MAZE_SPAN - 1;

                        if on_x_neg || on_x_pos || on_z_neg || on_z_pos || on_w_neg || on_w_pos {
                            let gate_open = if on_x_neg {
                                maze_gate_open(uz, uw, x_neg_gate[0], x_neg_gate[1])
                            } else if on_x_pos {
                                maze_gate_open(uz, uw, x_pos_gate[0], x_pos_gate[1])
                            } else if on_z_neg {
                                maze_gate_open(ux, uw, z_neg_gate[0], z_neg_gate[1])
                            } else if on_z_pos {
                                maze_gate_open(ux, uw, z_pos_gate[0], z_pos_gate[1])
                            } else if on_w_neg {
                                maze_gate_open(ux, uz, w_neg_gate[0], w_neg_gate[1])
                            } else {
                                maze_gate_open(ux, uz, w_pos_gate[0], w_pos_gate[1])
                            };

                            if gate_open {
                                None
                            } else {
                                Some(MAZE_GATE_FRAME_MATERIAL)
                            }
                        } else {
                            let wall_x = ux % MAZE_STRIDE == 0;
                            let wall_z = uz % MAZE_STRIDE == 0;
                            let wall_w = uw % MAZE_STRIDE == 0;
                            let wall_count = wall_x as i32 + wall_z as i32 + wall_w as i32;

                            if wall_count == 0 {
                                if ux == center_u
                                    && uz == center_u
                                    && uw == center_u
                                    && uy == MAZE_HEIGHT / 2
                                {
                                    Some(MAZE_BEACON_MATERIAL)
                                } else {
                                    None
                                }
                            } else if wall_count > 1 {
                                Some(MAZE_WALL_MATERIAL)
                            } else if wall_x {
                                if uz % MAZE_STRIDE != 0 && uw % MAZE_STRIDE != 0 {
                                    let left = ux / MAZE_STRIDE - 1;
                                    let right = left + 1;
                                    let cz = uz / MAZE_STRIDE;
                                    let cw = uw / MAZE_STRIDE;
                                    if maze_edge_open(layout_seed, [left, cz, cw], [right, cz, cw])
                                    {
                                        None
                                    } else {
                                        Some(MAZE_WALL_MATERIAL)
                                    }
                                } else {
                                    Some(MAZE_WALL_MATERIAL)
                                }
                            } else if wall_z {
                                if ux % MAZE_STRIDE != 0 && uw % MAZE_STRIDE != 0 {
                                    let near = uz / MAZE_STRIDE - 1;
                                    let far = near + 1;
                                    let cx = ux / MAZE_STRIDE;
                                    let cw = uw / MAZE_STRIDE;
                                    if maze_edge_open(layout_seed, [cx, near, cw], [cx, far, cw]) {
                                        None
                                    } else {
                                        Some(MAZE_WALL_MATERIAL)
                                    }
                                } else {
                                    Some(MAZE_WALL_MATERIAL)
                                }
                            } else if ux % MAZE_STRIDE != 0 && uz % MAZE_STRIDE != 0 {
                                let near = uw / MAZE_STRIDE - 1;
                                let far = near + 1;
                                let cx = ux / MAZE_STRIDE;
                                let cz = uz / MAZE_STRIDE;
                                if maze_edge_open(layout_seed, [cx, cz, near], [cx, cz, far]) {
                                    None
                                } else {
                                    Some(MAZE_WALL_MATERIAL)
                                }
                            } else {
                                Some(MAZE_WALL_MATERIAL)
                            }
                        }
                    };

                    let Some(material) = material else {
                        continue;
                    };

                    let lx = (wx - chunk_min[0]) as usize;
                    let ly = (wy - chunk_min[1]) as usize;
                    let lz = (wz - chunk_min[2]) as usize;
                    let lw = (ww - chunk_min[3]) as usize;
                    chunk.set(lx, ly, lz, lw, VoxelType(material));
                }
            }
        }
    }
}

pub fn generate_structure_chunk_with_keepout(
    world_seed: u64,
    chunk_pos: ChunkPos,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Option<Chunk> {
    let set = structure_set();
    let (chunk_min, _) = chunk_bounds(chunk_pos);
    let structure_placements = collect_structure_placements_for_chunk(world_seed, chunk_pos);
    let maze_placements = collect_maze_placements_for_chunk(world_seed, chunk_pos);

    let mut chunk = Chunk::new();
    for placement in structure_placements {
        if !placement_allowed(&placement, blocked_cells) {
            continue;
        }
        let blueprint = &set.blueprints[placement.blueprint_idx];
        blueprint.place_into_chunk_oriented(
            placement.origin,
            placement.orientation,
            chunk_min,
            &mut chunk,
        );
    }
    for placement in maze_placements {
        place_maze_into_chunk(placement, chunk_min, &mut chunk);
    }

    if chunk.is_empty() {
        None
    } else {
        Some(chunk)
    }
}

pub fn generate_structure_chunk(world_seed: u64, chunk_pos: ChunkPos) -> Option<Chunk> {
    generate_structure_chunk_with_keepout(world_seed, chunk_pos, None)
}

pub fn structure_chunk_has_content_with_keepout(
    world_seed: u64,
    chunk_pos: ChunkPos,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    let set = structure_set();
    let (chunk_min, chunk_max) = chunk_bounds(chunk_pos);
    collect_structure_placements_for_chunk(world_seed, chunk_pos)
        .into_iter()
        .any(|placement| {
            if !placement_allowed(&placement, blocked_cells) {
                return false;
            }
            let blueprint = &set.blueprints[placement.blueprint_idx];
            blueprint.writes_any_voxel_in_chunk_oriented(
                placement.origin,
                placement.orientation,
                chunk_min,
                chunk_max,
            )
        })
}

pub fn structure_chunk_has_content(world_seed: u64, chunk_pos: ChunkPos) -> bool {
    structure_chunk_has_content_with_keepout(world_seed, chunk_pos, None)
}

pub fn structure_chunk_has_content_for_scale_with_keepout(
    world_seed: u64,
    chunk_pos: ChunkPos,
    chunk_scale: i32,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    let scale = chunk_scale.max(1);
    if scale == 1 {
        return structure_chunk_has_content_with_keepout(world_seed, chunk_pos, blocked_cells);
    }

    for dw in 0..scale {
        for dz in 0..scale {
            for dy in 0..scale {
                for dx in 0..scale {
                    let child = ChunkPos::new(
                        chunk_pos.x * scale + dx,
                        chunk_pos.y * scale + dy,
                        chunk_pos.z * scale + dz,
                        chunk_pos.w * scale + dw,
                    );
                    if structure_chunk_has_content_with_keepout(world_seed, child, blocked_cells) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub fn structure_chunk_has_content_for_scale(
    world_seed: u64,
    chunk_pos: ChunkPos,
    chunk_scale: i32,
) -> bool {
    structure_chunk_has_content_for_scale_with_keepout(world_seed, chunk_pos, chunk_scale, None)
}

fn jitter_from_hash(hash: u64) -> i32 {
    jitter_from_hash_with_radius(hash, STRUCTURE_CELL_JITTER)
}

fn jitter_from_hash_with_radius(hash: u64, radius: i32) -> i32 {
    let span = (radius * 2 + 1) as u64;
    (splitmix64(hash) % span) as i32 - radius
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
    use std::collections::HashSet;

    fn find_chunk_with_maze(seed: u64) -> Option<ChunkPos> {
        let chunk_y = MAZE_WORLD_Y_MIN.div_euclid(CHUNK_SIZE as i32);
        for cell_x in -8..=8 {
            for cell_z in -8..=8 {
                for cell_w in -8..=8 {
                    let cell_hash =
                        hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                    if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                        continue;
                    }

                    let origin_x = cell_x * MAZE_CELL_SIZE
                        + jitter_from_hash_with_radius(
                            cell_hash ^ MAZE_JITTER_X_SALT,
                            MAZE_CELL_JITTER,
                        );
                    let origin_z = cell_z * MAZE_CELL_SIZE
                        + jitter_from_hash_with_radius(
                            cell_hash ^ MAZE_JITTER_Z_SALT,
                            MAZE_CELL_JITTER,
                        );
                    let origin_w = cell_w * MAZE_CELL_SIZE
                        + jitter_from_hash_with_radius(
                            cell_hash ^ MAZE_JITTER_W_SALT,
                            MAZE_CELL_JITTER,
                        );
                    if origin_x.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                        && origin_z.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                        && origin_w.abs() <= MAZE_ORIGIN_EXCLUSION_RADIUS
                    {
                        continue;
                    }

                    let chunk_pos = ChunkPos::new(
                        origin_x.div_euclid(CHUNK_SIZE as i32),
                        chunk_y,
                        origin_z.div_euclid(CHUNK_SIZE as i32),
                        origin_w.div_euclid(CHUNK_SIZE as i32),
                    );
                    if !collect_maze_placements_for_chunk(seed, chunk_pos).is_empty() {
                        return Some(chunk_pos);
                    }
                }
            }
        }
        None
    }

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
    fn procedural_mazes_generate_chunks() {
        let seed = 0x5eed_1234_5678_9abc;
        let chunk_pos =
            find_chunk_with_maze(seed).expect("expected to find at least one maze chunk");
        let chunk = generate_structure_chunk(seed, chunk_pos)
            .expect("maze placement should generate chunk");
        let has_maze_material = chunk.voxels.iter().any(|voxel| {
            matches!(
                voxel.0,
                MAZE_CEILING_MATERIAL
                    | MAZE_WALL_MATERIAL
                    | MAZE_GATE_FRAME_MATERIAL
                    | MAZE_BEACON_MATERIAL
            )
        });
        assert!(
            has_maze_material,
            "expected generated chunk to contain procedural maze materials"
        );
    }

    #[test]
    fn procedural_maze_generation_is_seed_deterministic() {
        let seed = 0x9d6f_21a5_7784_0031;
        let chunk_pos =
            find_chunk_with_maze(seed).expect("expected to find at least one maze chunk");
        let chunk_a = generate_structure_chunk(seed, chunk_pos)
            .expect("maze placement should generate chunk");
        let chunk_b = generate_structure_chunk(seed, chunk_pos)
            .expect("maze placement should generate chunk");
        assert_eq!(chunk_a.solid_count, chunk_b.solid_count);
        assert_eq!(chunk_a.voxels[..], chunk_b.voxels[..]);
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

    #[test]
    fn chunk_has_content_for_scale_matches_child_chunks() {
        let seed = 1717_u64;
        let scale = 2;
        let (min_y, max_y) = structure_chunk_y_bounds_for_scale(scale);

        for x in -6..=6 {
            for z in -6..=6 {
                for w in -6..=6 {
                    for y in min_y..=max_y {
                        let coarse = ChunkPos::new(x, y, z, w);
                        let mut expected = false;
                        for dw in 0..scale {
                            for dz in 0..scale {
                                for dy in 0..scale {
                                    for dx in 0..scale {
                                        let child = ChunkPos::new(
                                            coarse.x * scale + dx,
                                            coarse.y * scale + dy,
                                            coarse.z * scale + dz,
                                            coarse.w * scale + dw,
                                        );
                                        if structure_chunk_has_content(seed, child) {
                                            expected = true;
                                            break;
                                        }
                                    }
                                    if expected {
                                        break;
                                    }
                                }
                                if expected {
                                    break;
                                }
                            }
                            if expected {
                                break;
                            }
                        }

                        assert_eq!(
                            structure_chunk_has_content_for_scale(seed, coarse, scale),
                            expected,
                            "scaled content mismatch for chunk ({x}, {y}, {z}, {w})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn keepout_cells_block_generation_for_affected_chunk() {
        let seed = 4242_u64;
        let (min_y, max_y) = structure_chunk_y_bounds();

        let mut target_chunk = None;
        'search: for x in -10..=10 {
            for z in -10..=10 {
                for w in -10..=10 {
                    for y in min_y..=max_y {
                        let chunk_pos = ChunkPos::new(x, y, z, w);
                        let cells = structure_cells_affecting_chunk(seed, chunk_pos);
                        if !cells.is_empty() && generate_structure_chunk(seed, chunk_pos).is_some()
                        {
                            target_chunk = Some((chunk_pos, cells));
                            break 'search;
                        }
                    }
                }
            }
        }

        let (chunk_pos, cells) = target_chunk.expect("expected at least one structure chunk");
        let mut blocked = HashSet::new();
        blocked.insert(cells[0]);

        assert!(
            !structure_chunk_has_content_with_keepout(seed, chunk_pos, Some(&blocked)),
            "expected keepout cell to suppress structure content for chunk {:?}",
            chunk_pos
        );
        assert!(
            generate_structure_chunk_with_keepout(seed, chunk_pos, Some(&blocked)).is_none(),
            "expected keepout cell to suppress chunk generation for chunk {:?}",
            chunk_pos
        );
    }
}
