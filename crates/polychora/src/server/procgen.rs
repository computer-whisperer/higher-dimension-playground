use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::{Aabb4i, ChunkCoord};
use crate::shared::voxel::{BlockData, CHUNK_SIZE, CHUNK_VOLUME};
use flate2::read::GzDecoder;
use polychora_plugin_api::content_ids;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::{Arc, Mutex, OnceLock};

const STRUCTURE_CELL_SIZE: i32 = 32;
const STRUCTURE_CELL_JITTER: i32 = 10;
const STRUCTURE_SPAWN_NUMERATOR: u64 = 1;
const STRUCTURE_SPAWN_DENOMINATOR: u64 = 480;
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
const MAZE_SPAWN_DENOMINATOR: u64 = 36;
const MAZE_ORIGIN_EXCLUSION_RADIUS: i32 = 24;
const MAZE_HASH_SALT: u64 = 0x6f1d_05ce_294a_719b;
const MAZE_LAYOUT_SALT: u64 = 0x49ec_66d6_0d13_9e75;
const MAZE_VARIANT_SALT: u64 = 0x932f_b43c_f1f9_746d;
const MAZE_EDGE_SORT_SALT: u64 = 0xba04_7f9b_11d2_c638;
const MAZE_BRAID_SALT: u64 = 0x0d7c_88e1_a4f3_5b29;
const MAZE_EDGE_X_SALT: u64 = 0x3374_11a9_5f9c_2d01;
const MAZE_EDGE_Y_SALT: u64 = 0x40ad_6f0c_22be_8a17;
const MAZE_EDGE_Z_SALT: u64 = 0x9bc3_51d4_788a_f2ee;
const MAZE_EDGE_W_SALT: u64 = 0x1ec8_daa5_4b73_99c0;
const MAZE_JITTER_X_SALT: u64 = 0x5fa7_6d28_0f8b_81c3;
const MAZE_JITTER_Z_SALT: u64 = 0x8af0_1f7a_cc99_20be;
const MAZE_JITTER_W_SALT: u64 = 0x163a_8b39_b0a2_6741;
const MAZE_GATE_X_NEG_Y_SALT: u64 = 0x7270_6f63_6765_6e31;
const MAZE_GATE_X_NEG_Z_SALT: u64 = 0x7270_6f63_6765_6e32;
const MAZE_GATE_X_NEG_W_SALT: u64 = 0x7270_6f63_6765_6e33;
const MAZE_GATE_X_POS_Y_SALT: u64 = 0x7270_6f63_6765_6e34;
const MAZE_GATE_X_POS_Z_SALT: u64 = 0x7270_6f63_6765_6e35;
const MAZE_GATE_X_POS_W_SALT: u64 = 0x7270_6f63_6765_6e36;
const MAZE_GATE_Z_NEG_X_SALT: u64 = 0x7270_6f63_6765_6e37;
const MAZE_GATE_Z_NEG_Y_SALT: u64 = 0x7270_6f63_6765_6e38;
const MAZE_GATE_Z_NEG_W_SALT: u64 = 0x7270_6f63_6765_6e39;
const MAZE_GATE_Z_POS_X_SALT: u64 = 0x7270_6f63_6765_6e3a;
const MAZE_GATE_Z_POS_Y_SALT: u64 = 0x7270_6f63_6765_6e3b;
const MAZE_GATE_Z_POS_W_SALT: u64 = 0x7270_6f63_6765_6e3c;
const MAZE_GATE_W_NEG_X_SALT: u64 = 0x7270_6f63_6765_6e3d;
const MAZE_GATE_W_NEG_Y_SALT: u64 = 0x7270_6f63_6765_6e3e;
const MAZE_GATE_W_NEG_Z_SALT: u64 = 0x7270_6f63_6765_6e3f;
const MAZE_GATE_W_POS_X_SALT: u64 = 0x7270_6f63_6765_6e40;
const MAZE_GATE_W_POS_Y_SALT: u64 = 0x7270_6f63_6765_6e41;
const MAZE_GATE_W_POS_Z_SALT: u64 = 0x7270_6f63_6765_6e42;
const MAZE_GRID_X_SALT: u64 = 0x2c93_f17a_6540_8e11;
const MAZE_GRID_Y_SALT: u64 = 0x1720_8d59_3bf4_2c77;
const MAZE_GRID_Z_SALT: u64 = 0x85a4_5b0d_9926_f5c4;
const MAZE_GRID_W_SALT: u64 = 0xf30d_7c9e_4ab2_1688;
const MAZE_Y_JITTER_SALT: u64 = 0x6f5e_2914_a7cd_3b90;
const MAZE_GRID_XZW_CELLS_MIN: i32 = 9;
const MAZE_GRID_XZW_CELLS_MAX: i32 = 15;
const MAZE_GRID_Y_CELLS_MIN: i32 = 3;
const MAZE_GRID_Y_CELLS_MAX: i32 = 7;
const MAZE_STRIDE: i32 = 4;
const MAZE_LEVEL_HEIGHT: i32 = 4;
const MAZE_WORLD_Y_BASE: i32 = 0;
const MAZE_WORLD_Y_JITTER: i32 = 8;
const MAZE_MAX_HALF_SPAN_XZW: i32 = (MAZE_GRID_XZW_CELLS_MAX * MAZE_STRIDE + 1) / 2;
const MAZE_LAYOUT_CACHE_CAPACITY: usize = 96;
type DenseChunk = [u16; CHUNK_VOLUME];

#[inline]
fn dense_chunk_new() -> DenseChunk {
    [0u16; CHUNK_VOLUME]
}

#[inline]
fn dense_chunk_set(
    chunk: &mut DenseChunk,
    x: usize,
    y: usize,
    z: usize,
    w: usize,
    material: u16,
) {
    let idx =
        w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x;
    chunk[idx] = material;
}

#[inline]
fn dense_chunk_is_empty(chunk: &DenseChunk) -> bool {
    chunk.iter().all(|&v| v == 0)
}

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
    block: BlockData,
    #[serde(skip)]
    palette_idx: u16,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureVoxel {
    offset: [i32; 4],
    block: BlockData,
    #[serde(skip)]
    palette_idx: u16,
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
    block_palette: Vec<BlockData>,
    maze_floor_idx: u16,
    maze_ceiling_idx: u16,
    maze_wall_idx: u16,
    maze_gate_frame_idx: u16,
    maze_beacon_idx: u16,
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
            if fill.block.is_air() {
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
            if voxel.block.is_air() {
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
        chunk: &mut DenseChunk,
    ) {
        for fill in &self.fills {
            if fill.palette_idx == 0 {
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

            let material = fill.palette_idx;
            for wx in loop_min[0]..=loop_max[0] {
                for wy in loop_min[1]..=loop_max[1] {
                    for wz in loop_min[2]..=loop_max[2] {
                        for ww in loop_min[3]..=loop_max[3] {
                            let lx = (wx - chunk_min[0]) as usize;
                            let ly = (wy - chunk_min[1]) as usize;
                            let lz = (wz - chunk_min[2]) as usize;
                            let lw = (ww - chunk_min[3]) as usize;
                            dense_chunk_set(chunk, lx, ly, lz, lw, material);
                        }
                    }
                }
            }
        }

        for voxel in &self.voxels {
            if voxel.palette_idx == 0 {
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
            dense_chunk_set(chunk, lx, ly, lz, lw, voxel.palette_idx);
        }
    }

    fn oriented_world_bounds(&self, origin: [i32; 4], orientation: u8) -> ([i32; 4], [i32; 4]) {
        let x_vals = [self.min_offset[0], self.max_offset[0]];
        let y_vals = [self.min_offset[1], self.max_offset[1]];
        let z_vals = [self.min_offset[2], self.max_offset[2]];
        let w_vals = [self.min_offset[3], self.max_offset[3]];
        let mut min = [i32::MAX; 4];
        let mut max = [i32::MIN; 4];
        for x in x_vals {
            for y in y_vals {
                for z in z_vals {
                    for w in w_vals {
                        let rotated = rotate_offset_xzw([x, y, z, w], orientation);
                        let point = [
                            origin[0] + rotated[0],
                            origin[1] + rotated[1],
                            origin[2] + rotated[2],
                            origin[3] + rotated[3],
                        ];
                        for axis in 0..4 {
                            min[axis] = min[axis].min(point[axis]);
                            max[axis] = max[axis].max(point[axis]);
                        }
                    }
                }
            }
        }
        (min, max)
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

fn intern_block(
    palette: &mut Vec<BlockData>,
    index: &mut HashMap<BlockData, u16>,
    block: &BlockData,
) -> u16 {
    if block.is_air() {
        return 0;
    }
    if let Some(&idx) = index.get(block) {
        return idx;
    }
    let idx = palette.len() as u16;
    palette.push(block.clone());
    index.insert(block.clone(), idx);
    idx
}

static STRUCTURE_SET: OnceLock<StructureSet> = OnceLock::new();

fn structure_set() -> &'static StructureSet {
    STRUCTURE_SET.get_or_init(|| {
        let sources: [(&str, &[u8]); 26] = [
            (
                "cross_shrine.json",
                &include_bytes!("../../assets/structures/cross_shrine.json")[..],
            ),
            (
                "hyper_arch.json",
                &include_bytes!("../../assets/structures/hyper_arch.json")[..],
            ),
            (
                "duoprism_exchange.json",
                &include_bytes!("../../assets/structures/duoprism_exchange.json")[..],
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
                "phase_cloister.json",
                &include_bytes!("../../assets/structures/phase_cloister.json")[..],
            ),
            (
                "pentachord_spindle.json",
                &include_bytes!("../../assets/structures/pentachord_spindle.json")[..],
            ),
            (
                "orthoplex_nexus.json",
                &include_bytes!("../../assets/structures/orthoplex_nexus.json")[..],
            ),
            (
                "ore_cairn.json",
                &include_bytes!("../../assets/structures/ore_cairn.json")[..],
            ),
            (
                "frozen_spire.json",
                &include_bytes!("../../assets/structures/frozen_spire.json")[..],
            ),
            (
                "moss_ruins.json",
                &include_bytes!("../../assets/structures/moss_ruins.json")[..],
            ),
            (
                "sandstone_obelisk.json",
                &include_bytes!("../../assets/structures/sandstone_obelisk.json")[..],
            ),
            (
                "prism_fountain.json",
                &include_bytes!("../../assets/structures/prism_fountain.json")[..],
            ),
            (
                "hypercube_frame.json",
                &include_bytes!("../../assets/structures/hypercube_frame.json")[..],
            ),
        ];

        let mut blueprints: Vec<_> = sources
            .iter()
            .map(|(name, source)| StructureBlueprint::from_embedded_source(name, source))
            .collect();

        // Build the shared block palette and assign palette indices.
        let mut block_palette = vec![BlockData::AIR];
        let mut block_to_idx = HashMap::new();
        block_to_idx.insert(BlockData::AIR, 0u16);
        for blueprint in &mut blueprints {
            for fill in &mut blueprint.fills {
                fill.palette_idx = intern_block(&mut block_palette, &mut block_to_idx, &fill.block);
            }
            for voxel in &mut blueprint.voxels {
                voxel.palette_idx = intern_block(&mut block_palette, &mut block_to_idx, &voxel.block);
            }
        }

        // Intern maze materials.
        let maze_floor_idx = intern_block(&mut block_palette, &mut block_to_idx, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_BASALT_TILES));
        let maze_ceiling_idx = intern_block(&mut block_palette, &mut block_to_idx, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_SMOKED_GLASS));
        let maze_wall_idx = intern_block(&mut block_palette, &mut block_to_idx, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_RUNIC_ALLOY));
        let maze_gate_frame_idx = intern_block(&mut block_palette, &mut block_to_idx, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_OBSIDIAN));
        let maze_beacon_idx = intern_block(&mut block_palette, &mut block_to_idx, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_EVENTIDE_ALLOY));

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
            block_palette,
            maze_floor_idx,
            maze_ceiling_idx,
            maze_wall_idx,
            maze_gate_frame_idx,
            maze_beacon_idx,
        }
    })
}

pub fn structure_chunk_y_bounds() -> (i32, i32) {
    let set = structure_set();
    let chunk_size = CHUNK_SIZE as i32;
    let (maze_min_world_y, maze_max_world_y) = maze_world_y_bounds();
    let min_world_y = set.min_world_y.min(maze_min_world_y);
    let max_world_y = set.max_world_y.max(maze_max_world_y);
    (
        min_world_y.div_euclid(chunk_size),
        max_world_y.div_euclid(chunk_size),
    )
}

#[cfg(test)]
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
    shape: MazeShape,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct MazeShape {
    grid_cells: [i32; 4],
    span: [i32; 4],
    half_span_xzw: [i32; 3],
    world_y_min: i32,
    variant: MazeVariant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum MazeVariant {
    Catacomb,
    Vertical,
    Braided,
}

impl MazeVariant {
    fn from_cell_hash(cell_hash: u64) -> Self {
        match splitmix64(cell_hash ^ MAZE_VARIANT_SALT) % 3 {
            0 => Self::Catacomb,
            1 => Self::Vertical,
            _ => Self::Braided,
        }
    }

    fn axis_weights(self) -> [u32; 4] {
        match self {
            // Favors mostly horizontal corridors with fewer level transitions.
            Self::Catacomb => [2, 6, 2, 2],
            // Encourages vertical connectivity between maze levels.
            Self::Vertical => [3, 1, 3, 3],
            // Balanced tree plus loop-rich braid pass.
            Self::Braided => [2, 2, 2, 2],
        }
    }

    fn braid_chance(self) -> (u64, u64) {
        match self {
            Self::Catacomb => (0, 1),
            Self::Vertical => (1, 12),
            Self::Braided => (1, 4),
        }
    }

    fn seed_tag(self) -> u64 {
        match self {
            Self::Catacomb => 0x4bb3_136b_2d9f_8a01,
            Self::Vertical => 0xa574_1c90_ef22_4637,
            Self::Braided => 0x9f6d_3a1e_6cb5_78d2,
        }
    }
}

fn maze_random_odd(hash: u64, min_value: i32, max_value: i32) -> i32 {
    let min_odd = if min_value & 1 == 0 {
        min_value + 1
    } else {
        min_value
    };
    let max_odd = if max_value & 1 == 0 {
        max_value - 1
    } else {
        max_value
    };
    let choices = ((max_odd - min_odd) / 2 + 1).max(1) as u64;
    min_odd + 2 * (splitmix64(hash) % choices) as i32
}

fn maze_shape_from_cell_hash(cell_hash: u64) -> MazeShape {
    let variant = MazeVariant::from_cell_hash(cell_hash);
    let grid_x = maze_random_odd(
        cell_hash ^ MAZE_GRID_X_SALT,
        MAZE_GRID_XZW_CELLS_MIN,
        MAZE_GRID_XZW_CELLS_MAX,
    );
    let grid_y = maze_random_odd(
        cell_hash ^ MAZE_GRID_Y_SALT,
        MAZE_GRID_Y_CELLS_MIN,
        MAZE_GRID_Y_CELLS_MAX,
    );
    let grid_z = maze_random_odd(
        cell_hash ^ MAZE_GRID_Z_SALT,
        MAZE_GRID_XZW_CELLS_MIN,
        MAZE_GRID_XZW_CELLS_MAX,
    );
    let grid_w = maze_random_odd(
        cell_hash ^ MAZE_GRID_W_SALT,
        MAZE_GRID_XZW_CELLS_MIN,
        MAZE_GRID_XZW_CELLS_MAX,
    );
    let world_y_min = MAZE_WORLD_Y_BASE
        + jitter_from_hash_with_radius(cell_hash ^ MAZE_Y_JITTER_SALT, MAZE_WORLD_Y_JITTER);

    let span = [
        grid_x * MAZE_STRIDE + 1,
        grid_y * MAZE_LEVEL_HEIGHT + 1,
        grid_z * MAZE_STRIDE + 1,
        grid_w * MAZE_STRIDE + 1,
    ];
    let half_span_xzw = [span[0] / 2, span[2] / 2, span[3] / 2];

    MazeShape {
        grid_cells: [grid_x, grid_y, grid_z, grid_w],
        span,
        half_span_xzw,
        world_y_min,
        variant,
    }
}

fn maze_world_y_bounds() -> (i32, i32) {
    (
        MAZE_WORLD_Y_BASE - MAZE_WORLD_Y_JITTER,
        MAZE_WORLD_Y_BASE + MAZE_WORLD_Y_JITTER + MAZE_GRID_Y_CELLS_MAX * MAZE_LEVEL_HEIGHT,
    )
}

fn chunk_bounds(chunk_key: ChunkKey) -> ([i32; 4], [i32; 4]) {
    let chunk_size = CHUNK_SIZE as i32;
    let chunk_min = [
        chunk_key[0].to_num::<i32>() * chunk_size,
        chunk_key[1].to_num::<i32>() * chunk_size,
        chunk_key[2].to_num::<i32>() * chunk_size,
        chunk_key[3].to_num::<i32>() * chunk_size,
    ];
    let chunk_max = [
        chunk_min[0] + chunk_size - 1,
        chunk_min[1] + chunk_size - 1,
        chunk_min[2] + chunk_size - 1,
        chunk_min[3] + chunk_size - 1,
    ];
    (chunk_min, chunk_max)
}

/// Convert world-space half-open bounds to inclusive integer world bounds.
fn world_bounds_from_chunk_bounds(bounds: Aabb4i) -> ([i32; 4], [i32; 4]) {
    let min = bounds.min.map(|v| v.to_num::<i32>());
    let max = bounds.max.map(|v| v.to_num::<i32>() - 1);
    (min, max)
}

fn world_bounds_intersect(
    a_min: [i32; 4],
    a_max: [i32; 4],
    b_min: [i32; 4],
    b_max: [i32; 4],
) -> bool {
    for axis in 0..4 {
        if a_max[axis] < b_min[axis] || a_min[axis] > b_max[axis] {
            return false;
        }
    }
    true
}

fn intersect_world_bounds_as_chunk_bounds(
    world_min: [i32; 4],
    world_max: [i32; 4],
    query_chunk_bounds: Aabb4i,
) -> Option<Aabb4i> {
    let (query_world_min, query_world_max) = world_bounds_from_chunk_bounds(query_chunk_bounds);
    let mut clipped_world_min = [0i32; 4];
    let mut clipped_world_max = [0i32; 4];
    for axis in 0..4 {
        clipped_world_min[axis] = world_min[axis].max(query_world_min[axis]);
        clipped_world_max[axis] = world_max[axis].min(query_world_max[axis]);
        if clipped_world_min[axis] > clipped_world_max[axis] {
            return None;
        }
    }

    let chunk_min = [
        clipped_world_min[0].div_euclid(CHUNK_SIZE as i32),
        clipped_world_min[1].div_euclid(CHUNK_SIZE as i32),
        clipped_world_min[2].div_euclid(CHUNK_SIZE as i32),
        clipped_world_min[3].div_euclid(CHUNK_SIZE as i32),
    ];
    let chunk_max = [
        clipped_world_max[0].div_euclid(CHUNK_SIZE as i32),
        clipped_world_max[1].div_euclid(CHUNK_SIZE as i32),
        clipped_world_max[2].div_euclid(CHUNK_SIZE as i32),
        clipped_world_max[3].div_euclid(CHUNK_SIZE as i32),
    ];

    if chunk_min[0] > chunk_max[0]
        || chunk_min[1] > chunk_max[1]
        || chunk_min[2] > chunk_max[2]
        || chunk_min[3] > chunk_max[3]
    {
        return None;
    }
    Some(Aabb4i::from_lattice_bounds(chunk_min, chunk_max, 0))
}

fn collect_structure_placements_for_chunk_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Vec<StructurePlacement> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let set = structure_set();
    let (query_world_min, query_world_max) = world_bounds_from_chunk_bounds(bounds);
    if query_world_max[1] < set.min_world_y || query_world_min[1] > set.max_world_y {
        return Vec::new();
    }

    let search_margin = STRUCTURE_CELL_JITTER + set.max_abs_offset_xzw + CHUNK_SIZE as i32;
    let cell_min_x = (query_world_min[0] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_x = (query_world_max[0] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_z = (query_world_min[2] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_z = (query_world_max[2] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_w = (query_world_min[3] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_w = (query_world_max[3] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;

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
                if !placement_allowed(
                    &StructurePlacement {
                        blueprint_idx,
                        origin,
                        orientation,
                        cell: [cell_x, cell_z, cell_w],
                    },
                    blocked_cells,
                ) {
                    continue;
                }
                let (placement_min, placement_max) =
                    blueprint.oriented_world_bounds(origin, orientation);
                if !world_bounds_intersect(
                    placement_min,
                    placement_max,
                    query_world_min,
                    query_world_max,
                ) {
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

fn collect_maze_placements_for_chunk_bounds(world_seed: u64, bounds: Aabb4i) -> Vec<MazePlacement> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let (maze_world_min_y, maze_world_max_y) = maze_world_y_bounds();
    let (query_world_min, query_world_max) = world_bounds_from_chunk_bounds(bounds);
    if query_world_max[1] < maze_world_min_y || query_world_min[1] > maze_world_max_y {
        return Vec::new();
    }

    let search_margin = MAZE_CELL_JITTER + MAZE_MAX_HALF_SPAN_XZW + CHUNK_SIZE as i32;
    let cell_min_x = (query_world_min[0] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_x = (query_world_max[0] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;
    let cell_min_z = (query_world_min[2] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_z = (query_world_max[2] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;
    let cell_min_w = (query_world_min[3] - search_margin).div_euclid(MAZE_CELL_SIZE) - 1;
    let cell_max_w = (query_world_max[3] + search_margin).div_euclid(MAZE_CELL_SIZE) + 1;

    let mut placements = Vec::new();
    for cell_x in cell_min_x..=cell_max_x {
        for cell_z in cell_min_z..=cell_max_z {
            for cell_w in cell_min_w..=cell_max_w {
                let cell_hash =
                    hash_structure_cell(world_seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                    continue;
                }
                let shape = maze_shape_from_cell_hash(cell_hash);
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

                let origin = [origin_x, shape.world_y_min, origin_z, origin_w];
                let (maze_min, maze_max) = maze_bounds(origin, shape);
                if !world_bounds_intersect(maze_min, maze_max, query_world_min, query_world_max) {
                    continue;
                }
                placements.push(MazePlacement {
                    origin,
                    layout_seed: maze_layout_seed(world_seed, origin, shape),
                    shape,
                });
            }
        }
    }
    placements
}

pub fn structure_chunk_positions_for_bounds_with_keepout(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Vec<ChunkKey> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let set = structure_set();
    let mut chunk_positions = HashSet::<[i32; 4]>::new();

    for placement in
        collect_structure_placements_for_chunk_bounds(world_seed, bounds, blocked_cells)
    {
        let blueprint = &set.blueprints[placement.blueprint_idx];
        let (placement_min, placement_max) =
            blueprint.oriented_world_bounds(placement.origin, placement.orientation);
        let Some(covered_chunks) =
            intersect_world_bounds_as_chunk_bounds(placement_min, placement_max, bounds)
        else {
            continue;
        };
        let (cc_min, cc_max) = covered_chunks.to_chunk_lattice_bounds(0);
        for w in cc_min[3]..=cc_max[3] {
            for z in cc_min[2]..=cc_max[2] {
                for y in cc_min[1]..=cc_max[1] {
                    for x in cc_min[0]..=cc_max[0] {
                        chunk_positions.insert([x, y, z, w]);
                    }
                }
            }
        }
    }

    for placement in collect_maze_placements_for_chunk_bounds(world_seed, bounds) {
        let (maze_min, maze_max) = maze_bounds(placement.origin, placement.shape);
        let Some(covered_chunks) =
            intersect_world_bounds_as_chunk_bounds(maze_min, maze_max, bounds)
        else {
            continue;
        };
        let (cc_min, cc_max) = covered_chunks.to_chunk_lattice_bounds(0);
        for w in cc_min[3]..=cc_max[3] {
            for z in cc_min[2]..=cc_max[2] {
                for y in cc_min[1]..=cc_max[1] {
                    for x in cc_min[0]..=cc_max[0] {
                        chunk_positions.insert([x, y, z, w]);
                    }
                }
            }
        }
    }

    let mut out: Vec<ChunkKey> = chunk_positions
        .into_iter()
        .map(|[x, y, z, w]| [x, y, z, w].map(ChunkCoord::from_num))
        .collect();
    out.sort_unstable();
    out
}

/// Per-placement chunk data: a map of chunk positions to dense chunks generated from a single
/// structure placement.
pub struct PlacementChunkData {
    /// chunk_pos -> generated dense chunk (non-empty only, from this one placement).
    pub chunks: HashMap<[i32; 4], DenseChunk>,
}

/// Generate per-placement chunk data for all structure placements intersecting `bounds`.
///
/// Unlike `generate_structure_chunk_with_keepout` which merges all placements into each chunk,
/// this returns separate chunk data for each placement. Each placement's chunks contain only
/// voxels from that single structure, preventing overlap merging.
///
/// Mazes are NOT included — they remain on the per-chunk path.
pub fn generate_structure_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Vec<PlacementChunkData> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let set = structure_set();
    let placements = collect_structure_placements_for_chunk_bounds(world_seed, bounds, blocked_cells);

    let mut results = Vec::with_capacity(placements.len());
    for placement in placements {
        let blueprint = &set.blueprints[placement.blueprint_idx];
        let (placement_world_min, placement_world_max) =
            blueprint.oriented_world_bounds(placement.origin, placement.orientation);

        // Convert placement world bounds to chunk coordinates
        let chunk_size = CHUNK_SIZE as i32;
        let full_chunk_bounds = Aabb4i::from_lattice_bounds(
            [
                placement_world_min[0].div_euclid(chunk_size),
                placement_world_min[1].div_euclid(chunk_size),
                placement_world_min[2].div_euclid(chunk_size),
                placement_world_min[3].div_euclid(chunk_size),
            ],
            [
                placement_world_max[0].div_euclid(chunk_size),
                placement_world_max[1].div_euclid(chunk_size),
                placement_world_max[2].div_euclid(chunk_size),
                placement_world_max[3].div_euclid(chunk_size),
            ],
            0,
        );

        // Clip to query bounds
        let clipped = Aabb4i::new(
            [
                full_chunk_bounds.min[0].max(bounds.min[0]),
                full_chunk_bounds.min[1].max(bounds.min[1]),
                full_chunk_bounds.min[2].max(bounds.min[2]),
                full_chunk_bounds.min[3].max(bounds.min[3]),
            ],
            [
                full_chunk_bounds.max[0].min(bounds.max[0]),
                full_chunk_bounds.max[1].min(bounds.max[1]),
                full_chunk_bounds.max[2].min(bounds.max[2]),
                full_chunk_bounds.max[3].min(bounds.max[3]),
            ],
        );
        if !clipped.is_valid() {
            continue;
        }

        let (cl_min, cl_max) = clipped.to_chunk_lattice_bounds(0);
        let mut chunks = HashMap::new();
        for cw in cl_min[3]..=cl_max[3] {
            for cz in cl_min[2]..=cl_max[2] {
                for cy in cl_min[1]..=cl_max[1] {
                    for cx in cl_min[0]..=cl_max[0] {
                        let chunk_min = [
                            cx * chunk_size,
                            cy * chunk_size,
                            cz * chunk_size,
                            cw * chunk_size,
                        ];
                        let mut chunk = dense_chunk_new();
                        blueprint.place_into_chunk_oriented(
                            placement.origin,
                            placement.orientation,
                            chunk_min,
                            &mut chunk,
                        );
                        if !dense_chunk_is_empty(&chunk) {
                            chunks.insert([cx, cy, cz, cw], chunk);
                        }
                    }
                }
            }
        }

        if !chunks.is_empty() {
            results.push(PlacementChunkData { chunks });
        }
    }
    results
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
    chunk_key: ChunkKey,
) -> Vec<StructurePlacement> {
    let set = structure_set();
    let (min_chunk_y, max_chunk_y) = structure_chunk_y_bounds();
    let chunk_y = chunk_key[1].to_num::<i32>();
    if chunk_y < min_chunk_y || chunk_y > max_chunk_y {
        return Vec::new();
    }

    let chunk_size = CHUNK_SIZE as i32;
    let (chunk_min, chunk_max) = chunk_bounds(chunk_key);

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

#[cfg(test)]
pub fn structure_cells_affecting_chunk(world_seed: u64, chunk_key: ChunkKey) -> Vec<StructureCell> {
    let mut unique = HashSet::new();
    for placement in collect_structure_placements_for_chunk(world_seed, chunk_key) {
        unique.insert(placement.cell);
    }
    let mut out: Vec<_> = unique.into_iter().collect();
    out.sort_unstable();
    out
}

fn maze_bounds(origin: [i32; 4], shape: MazeShape) -> ([i32; 4], [i32; 4]) {
    (
        [
            origin[0] - shape.half_span_xzw[0],
            origin[1],
            origin[2] - shape.half_span_xzw[1],
            origin[3] - shape.half_span_xzw[2],
        ],
        [
            origin[0] + shape.half_span_xzw[0],
            origin[1] + shape.span[1] - 1,
            origin[2] + shape.half_span_xzw[1],
            origin[3] + shape.half_span_xzw[2],
        ],
    )
}

fn maze_intersects_chunk(
    origin: [i32; 4],
    shape: MazeShape,
    chunk_min: [i32; 4],
    chunk_max: [i32; 4],
) -> bool {
    let (maze_min, maze_max) = maze_bounds(origin, shape);
    for axis in 0..4 {
        if maze_max[axis] < chunk_min[axis] || maze_min[axis] > chunk_max[axis] {
            return false;
        }
    }
    true
}

fn maze_layout_seed(world_seed: u64, origin: [i32; 4], shape: MazeShape) -> u64 {
    let mut seed = hash_structure_cell(
        world_seed,
        origin[0],
        origin[2],
        origin[3],
        MAZE_LAYOUT_SALT,
    );
    seed ^= (origin[1] as i64 as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    seed ^= (shape.grid_cells[0] as i64 as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
    seed ^= (shape.grid_cells[1] as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    seed ^= (shape.grid_cells[2] as i64 as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    seed ^= (shape.grid_cells[3] as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    seed ^= shape.variant.seed_tag();
    splitmix64(seed)
}

fn maze_gate_cell(layout_seed: u64, salt: u64, cell_count: i32) -> i32 {
    (splitmix64(layout_seed ^ salt) % cell_count.max(1) as u64) as i32
}

fn maze_gate_band(cell_idx: i32, stride: i32) -> (i32, i32) {
    let start = cell_idx * stride + 1;
    (start, start + stride.saturating_sub(2))
}

fn maze_gate_open(
    u0: i32,
    u1: i32,
    u2: i32,
    gate0: i32,
    gate1: i32,
    gate2: i32,
    stride0: i32,
    stride1: i32,
    stride2: i32,
) -> bool {
    let (g0_min, g0_max) = maze_gate_band(gate0, stride0);
    let (g1_min, g1_max) = maze_gate_band(gate1, stride1);
    let (g2_min, g2_max) = maze_gate_band(gate2, stride2);
    u0 >= g0_min && u0 <= g0_max && u1 >= g1_min && u1 <= g1_max && u2 >= g2_min && u2 <= g2_max
}

fn maze_cell_in_bounds(cell: [i32; 4], grid_cells: [i32; 4]) -> bool {
    cell[0] >= 0
        && cell[0] < grid_cells[0]
        && cell[1] >= 0
        && cell[1] < grid_cells[1]
        && cell[2] >= 0
        && cell[2] < grid_cells[2]
        && cell[3] >= 0
        && cell[3] < grid_cells[3]
}

fn maze_hash_cell(seed: u64, cell: [i32; 4], salt: u64) -> u64 {
    let mut mixed = hash_structure_cell(seed, cell[0], cell[2], cell[3], salt);
    mixed ^= (cell[1] as i64 as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    splitmix64(mixed)
}

#[derive(Copy, Clone, Debug)]
struct MazeEdgeCandidate {
    axis: usize,
    base: [i32; 4],
    a_idx: usize,
    b_idx: usize,
    weighted_score: u128,
    tie_break: u64,
    braid_roll: u64,
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0u8; size],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        let parent = self.parent[node];
        if parent != node {
            let root = self.find(parent);
            self.parent[node] = root;
        }
        self.parent[node]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return false;
        }

        let rank_a = self.rank[root_a];
        let rank_b = self.rank[root_b];
        if rank_a < rank_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if rank_a == rank_b {
            self.rank[root_a] = self.rank[root_a].saturating_add(1);
        }
        true
    }
}

#[derive(Clone, Debug)]
struct MazeTopology {
    grid_cells: [i32; 4],
    open_x: Vec<bool>,
    open_y: Vec<bool>,
    open_z: Vec<bool>,
    open_w: Vec<bool>,
}

impl MazeTopology {
    fn new(grid_cells: [i32; 4]) -> Self {
        Self {
            grid_cells,
            open_x: vec![false; maze_edge_count(grid_cells, 0)],
            open_y: vec![false; maze_edge_count(grid_cells, 1)],
            open_z: vec![false; maze_edge_count(grid_cells, 2)],
            open_w: vec![false; maze_edge_count(grid_cells, 3)],
        }
    }

    fn set_edge_open(&mut self, axis: usize, base: [i32; 4]) {
        let idx = maze_edge_linear_index(self.grid_cells, axis, base);
        match axis {
            0 => self.open_x[idx] = true,
            1 => self.open_y[idx] = true,
            2 => self.open_z[idx] = true,
            3 => self.open_w[idx] = true,
            _ => unreachable!("invalid axis"),
        }
    }

    fn edge_open(&self, a: [i32; 4], b: [i32; 4]) -> bool {
        if !maze_cell_in_bounds(a, self.grid_cells) || !maze_cell_in_bounds(b, self.grid_cells) {
            return false;
        }

        let mut changed_axis = None;
        for axis in 0..4 {
            let delta = a[axis] - b[axis];
            if delta == 0 {
                continue;
            }
            if delta.abs() != 1 || changed_axis.is_some() {
                return false;
            }
            changed_axis = Some(axis);
        }

        let Some(axis) = changed_axis else {
            return false;
        };
        let mut base = a;
        if b[axis] < a[axis] {
            base[axis] = b[axis];
        }
        let idx = maze_edge_linear_index(self.grid_cells, axis, base);
        match axis {
            0 => self.open_x[idx],
            1 => self.open_y[idx],
            2 => self.open_z[idx],
            3 => self.open_w[idx],
            _ => unreachable!("invalid axis"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct MazeLayoutCacheKey {
    origin: [i32; 4],
    layout_seed: u64,
    shape: MazeShape,
}

#[derive(Clone, Debug)]
struct MazeCompiledLayout {
    topology: MazeTopology,
    x_neg_gate: [i32; 3],
    x_pos_gate: [i32; 3],
    z_neg_gate: [i32; 3],
    z_pos_gate: [i32; 3],
    w_neg_gate: [i32; 3],
    w_pos_gate: [i32; 3],
    center_u: [i32; 4],
}

#[derive(Clone, Debug)]
struct MazeLayoutCacheEntry {
    layout: Arc<MazeCompiledLayout>,
    last_used: u64,
}

#[derive(Clone, Debug)]
struct MazeLayoutCache {
    entries: HashMap<MazeLayoutCacheKey, MazeLayoutCacheEntry>,
    next_use_id: u64,
}

impl MazeLayoutCache {
    fn next_use_id(&mut self) -> u64 {
        self.next_use_id = self.next_use_id.wrapping_add(1);
        if self.next_use_id == 0 {
            self.next_use_id = 1;
        }
        self.next_use_id
    }

    fn evict_lru_entry(&mut self) {
        let Some(stale_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| *key)
        else {
            return;
        };
        self.entries.remove(&stale_key);
    }

    fn get_or_insert(&mut self, key: MazeLayoutCacheKey) -> Arc<MazeCompiledLayout> {
        let use_id = self.next_use_id();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = use_id;
            return Arc::clone(&entry.layout);
        }

        let layout = Arc::new(maze_compile_layout(key.layout_seed, key.shape));
        self.entries.insert(
            key,
            MazeLayoutCacheEntry {
                layout: Arc::clone(&layout),
                last_used: use_id,
            },
        );
        while self.entries.len() > MAZE_LAYOUT_CACHE_CAPACITY {
            self.evict_lru_entry();
        }
        layout
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.next_use_id = 0;
    }
}

impl Default for MazeLayoutCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            next_use_id: 0,
        }
    }
}

static MAZE_LAYOUT_CACHE: OnceLock<Mutex<MazeLayoutCache>> = OnceLock::new();

fn maze_layout_cache() -> &'static Mutex<MazeLayoutCache> {
    MAZE_LAYOUT_CACHE.get_or_init(|| Mutex::new(MazeLayoutCache::default()))
}

pub fn clear_runtime_maze_layout_cache() {
    if let Some(cache) = MAZE_LAYOUT_CACHE.get() {
        let mut guard = cache.lock().expect("maze layout cache lock poisoned");
        guard.clear();
    }
}

fn maze_axis_salt(axis: usize) -> u64 {
    match axis {
        0 => MAZE_EDGE_X_SALT,
        1 => MAZE_EDGE_Y_SALT,
        2 => MAZE_EDGE_Z_SALT,
        3 => MAZE_EDGE_W_SALT,
        _ => unreachable!("invalid axis"),
    }
}

fn maze_cell_linear_index(grid_cells: [i32; 4], cell: [i32; 4]) -> usize {
    (((cell[0] as usize * grid_cells[1] as usize + cell[1] as usize) * grid_cells[2] as usize
        + cell[2] as usize)
        * grid_cells[3] as usize)
        + cell[3] as usize
}

fn maze_edge_count(grid_cells: [i32; 4], axis: usize) -> usize {
    match axis {
        0 => ((grid_cells[0] - 1).max(0) * grid_cells[1] * grid_cells[2] * grid_cells[3]) as usize,
        1 => (grid_cells[0] * (grid_cells[1] - 1).max(0) * grid_cells[2] * grid_cells[3]) as usize,
        2 => (grid_cells[0] * grid_cells[1] * (grid_cells[2] - 1).max(0) * grid_cells[3]) as usize,
        3 => (grid_cells[0] * grid_cells[1] * grid_cells[2] * (grid_cells[3] - 1).max(0)) as usize,
        _ => unreachable!("invalid axis"),
    }
}

fn maze_edge_linear_index(grid_cells: [i32; 4], axis: usize, base: [i32; 4]) -> usize {
    match axis {
        0 => {
            (((base[0] as usize * grid_cells[1] as usize + base[1] as usize)
                * grid_cells[2] as usize
                + base[2] as usize)
                * grid_cells[3] as usize)
                + base[3] as usize
        }
        1 => {
            (((base[0] as usize * (grid_cells[1] - 1) as usize + base[1] as usize)
                * grid_cells[2] as usize
                + base[2] as usize)
                * grid_cells[3] as usize)
                + base[3] as usize
        }
        2 => {
            (((base[0] as usize * grid_cells[1] as usize + base[1] as usize)
                * (grid_cells[2] - 1) as usize
                + base[2] as usize)
                * grid_cells[3] as usize)
                + base[3] as usize
        }
        3 => {
            (((base[0] as usize * grid_cells[1] as usize + base[1] as usize)
                * grid_cells[2] as usize
                + base[2] as usize)
                * (grid_cells[3] - 1) as usize)
                + base[3] as usize
        }
        _ => unreachable!("invalid axis"),
    }
}

fn maze_edge_hash(layout_seed: u64, base: [i32; 4], axis: usize, salt: u64) -> u64 {
    let mut mixed = maze_hash_cell(layout_seed, base, salt ^ maze_axis_salt(axis));
    mixed ^= (axis as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
    splitmix64(mixed)
}

fn maze_collect_edge_candidates(layout_seed: u64, shape: MazeShape) -> Vec<MazeEdgeCandidate> {
    let axis_weights = shape.variant.axis_weights();
    let grid = shape.grid_cells;
    let estimated_edges = maze_edge_count(grid, 0)
        + maze_edge_count(grid, 1)
        + maze_edge_count(grid, 2)
        + maze_edge_count(grid, 3);
    let mut edges = Vec::with_capacity(estimated_edges);

    for x in 0..grid[0] {
        for y in 0..grid[1] {
            for z in 0..grid[2] {
                for w in 0..grid[3] {
                    let base = [x, y, z, w];
                    let a_idx = maze_cell_linear_index(grid, base);
                    for axis in 0..4 {
                        if base[axis] + 1 >= grid[axis] {
                            continue;
                        }
                        let mut neighbor = base;
                        neighbor[axis] += 1;
                        let b_idx = maze_cell_linear_index(grid, neighbor);
                        let sort_roll =
                            maze_edge_hash(layout_seed, base, axis, MAZE_EDGE_SORT_SALT);
                        let tie_break = maze_edge_hash(
                            layout_seed,
                            base,
                            axis,
                            MAZE_EDGE_SORT_SALT ^ 0xd6e8_feb8_6659_fd93,
                        );
                        let braid_roll = maze_edge_hash(layout_seed, base, axis, MAZE_BRAID_SALT);
                        edges.push(MazeEdgeCandidate {
                            axis,
                            base,
                            a_idx,
                            b_idx,
                            weighted_score: (sort_roll as u128) * axis_weights[axis] as u128,
                            tie_break,
                            braid_roll,
                        });
                    }
                }
            }
        }
    }

    edges
}

fn maze_build_topology(layout_seed: u64, shape: MazeShape) -> MazeTopology {
    let grid = shape.grid_cells;
    let total_cells =
        (grid[0] as usize) * (grid[1] as usize) * (grid[2] as usize) * (grid[3] as usize);
    let mut topology = MazeTopology::new(grid);
    let mut disjoint_set = DisjointSet::new(total_cells);
    let mut rejected_edges = Vec::new();
    let mut edges = maze_collect_edge_candidates(layout_seed, shape);
    edges.sort_unstable_by(|left, right| {
        left.weighted_score
            .cmp(&right.weighted_score)
            .then_with(|| left.tie_break.cmp(&right.tie_break))
    });

    for edge in edges {
        if disjoint_set.union(edge.a_idx, edge.b_idx) {
            topology.set_edge_open(edge.axis, edge.base);
        } else {
            rejected_edges.push(edge);
        }
    }

    let (braid_numerator, braid_denominator) = shape.variant.braid_chance();
    if braid_numerator > 0 {
        for edge in rejected_edges {
            if splitmix64(edge.braid_roll) % braid_denominator < braid_numerator {
                topology.set_edge_open(edge.axis, edge.base);
            }
        }
    }

    topology
}

fn maze_compile_layout(layout_seed: u64, shape: MazeShape) -> MazeCompiledLayout {
    MazeCompiledLayout {
        topology: maze_build_topology(layout_seed, shape),
        x_neg_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_X_NEG_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_X_NEG_Z_SALT, shape.grid_cells[2]),
            maze_gate_cell(layout_seed, MAZE_GATE_X_NEG_W_SALT, shape.grid_cells[3]),
        ],
        x_pos_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_X_POS_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_X_POS_Z_SALT, shape.grid_cells[2]),
            maze_gate_cell(layout_seed, MAZE_GATE_X_POS_W_SALT, shape.grid_cells[3]),
        ],
        z_neg_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_Z_NEG_X_SALT, shape.grid_cells[0]),
            maze_gate_cell(layout_seed, MAZE_GATE_Z_NEG_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_Z_NEG_W_SALT, shape.grid_cells[3]),
        ],
        z_pos_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_Z_POS_X_SALT, shape.grid_cells[0]),
            maze_gate_cell(layout_seed, MAZE_GATE_Z_POS_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_Z_POS_W_SALT, shape.grid_cells[3]),
        ],
        w_neg_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_W_NEG_X_SALT, shape.grid_cells[0]),
            maze_gate_cell(layout_seed, MAZE_GATE_W_NEG_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_W_NEG_Z_SALT, shape.grid_cells[2]),
        ],
        w_pos_gate: [
            maze_gate_cell(layout_seed, MAZE_GATE_W_POS_X_SALT, shape.grid_cells[0]),
            maze_gate_cell(layout_seed, MAZE_GATE_W_POS_Y_SALT, shape.grid_cells[1]),
            maze_gate_cell(layout_seed, MAZE_GATE_W_POS_Z_SALT, shape.grid_cells[2]),
        ],
        center_u: [
            (shape.grid_cells[0] / 2) * MAZE_STRIDE + MAZE_STRIDE / 2,
            (shape.grid_cells[1] / 2) * MAZE_LEVEL_HEIGHT + MAZE_LEVEL_HEIGHT / 2,
            (shape.grid_cells[2] / 2) * MAZE_STRIDE + MAZE_STRIDE / 2,
            (shape.grid_cells[3] / 2) * MAZE_STRIDE + MAZE_STRIDE / 2,
        ],
    }
}

fn maze_compiled_layout_for_placement(placement: MazePlacement) -> Arc<MazeCompiledLayout> {
    let key = MazeLayoutCacheKey {
        origin: placement.origin,
        layout_seed: placement.layout_seed,
        shape: placement.shape,
    };
    let mut cache = maze_layout_cache()
        .lock()
        .expect("maze layout cache lock poisoned");
    cache.get_or_insert(key)
}

fn collect_maze_placements_for_chunk(world_seed: u64, chunk_key: ChunkKey) -> Vec<MazePlacement> {
    let (maze_world_min_y, maze_world_max_y) = maze_world_y_bounds();
    let maze_min_chunk_y = maze_world_min_y.div_euclid(CHUNK_SIZE as i32);
    let maze_max_chunk_y = maze_world_max_y.div_euclid(CHUNK_SIZE as i32);
    let chunk_y = chunk_key[1].to_num::<i32>();
    if chunk_y < maze_min_chunk_y || chunk_y > maze_max_chunk_y {
        return Vec::new();
    }

    let chunk_size = CHUNK_SIZE as i32;
    let (chunk_min, chunk_max) = chunk_bounds(chunk_key);

    let search_margin = MAZE_CELL_JITTER + MAZE_MAX_HALF_SPAN_XZW + chunk_size;
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
                let shape = maze_shape_from_cell_hash(cell_hash);

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

                let origin = [origin_x, shape.world_y_min, origin_z, origin_w];
                if !maze_intersects_chunk(origin, shape, chunk_min, chunk_max) {
                    continue;
                }

                placements.push(MazePlacement {
                    origin,
                    layout_seed: maze_layout_seed(world_seed, origin, shape),
                    shape,
                });
            }
        }
    }

    placements
}

fn place_maze_into_chunk(placement: MazePlacement, chunk_min: [i32; 4], chunk: &mut DenseChunk, set: &StructureSet) {
    let chunk_max = [
        chunk_min[0] + CHUNK_SIZE as i32 - 1,
        chunk_min[1] + CHUNK_SIZE as i32 - 1,
        chunk_min[2] + CHUNK_SIZE as i32 - 1,
        chunk_min[3] + CHUNK_SIZE as i32 - 1,
    ];
    if !maze_intersects_chunk(placement.origin, placement.shape, chunk_min, chunk_max) {
        return;
    }

    let shape = placement.shape;
    let compiled_layout = maze_compiled_layout_for_placement(placement);
    let topology = &compiled_layout.topology;
    let x_neg_gate = compiled_layout.x_neg_gate;
    let x_pos_gate = compiled_layout.x_pos_gate;
    let z_neg_gate = compiled_layout.z_neg_gate;
    let z_pos_gate = compiled_layout.z_pos_gate;
    let w_neg_gate = compiled_layout.w_neg_gate;
    let w_pos_gate = compiled_layout.w_pos_gate;
    let center_u = compiled_layout.center_u;
    let (maze_min, maze_max) = maze_bounds(placement.origin, shape);
    let mut loop_min = [0i32; 4];
    let mut loop_max = [0i32; 4];
    for axis in 0..4 {
        loop_min[axis] = maze_min[axis].max(chunk_min[axis]);
        loop_max[axis] = maze_max[axis].min(chunk_max[axis]);
        if loop_min[axis] > loop_max[axis] {
            return;
        }
    }

    for wx in loop_min[0]..=loop_max[0] {
        for wy in loop_min[1]..=loop_max[1] {
            for wz in loop_min[2]..=loop_max[2] {
                for ww in loop_min[3]..=loop_max[3] {
                    let ux = wx - maze_min[0];
                    let uy = wy - maze_min[1];
                    let uz = wz - maze_min[2];
                    let uw = ww - maze_min[3];

                    let material: Option<u16> = if uy == 0 {
                        Some(set.maze_floor_idx)
                    } else if uy == shape.span[1] - 1 {
                        Some(set.maze_ceiling_idx)
                    } else {
                        let on_x_neg = ux == 0;
                        let on_x_pos = ux == shape.span[0] - 1;
                        let on_z_neg = uz == 0;
                        let on_z_pos = uz == shape.span[2] - 1;
                        let on_w_neg = uw == 0;
                        let on_w_pos = uw == shape.span[3] - 1;

                        if on_x_neg || on_x_pos || on_z_neg || on_z_pos || on_w_neg || on_w_pos {
                            let gate_open = if on_x_neg {
                                maze_gate_open(
                                    uy,
                                    uz,
                                    uw,
                                    x_neg_gate[0],
                                    x_neg_gate[1],
                                    x_neg_gate[2],
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                    MAZE_STRIDE,
                                )
                            } else if on_x_pos {
                                maze_gate_open(
                                    uy,
                                    uz,
                                    uw,
                                    x_pos_gate[0],
                                    x_pos_gate[1],
                                    x_pos_gate[2],
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                    MAZE_STRIDE,
                                )
                            } else if on_z_neg {
                                maze_gate_open(
                                    ux,
                                    uy,
                                    uw,
                                    z_neg_gate[0],
                                    z_neg_gate[1],
                                    z_neg_gate[2],
                                    MAZE_STRIDE,
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                )
                            } else if on_z_pos {
                                maze_gate_open(
                                    ux,
                                    uy,
                                    uw,
                                    z_pos_gate[0],
                                    z_pos_gate[1],
                                    z_pos_gate[2],
                                    MAZE_STRIDE,
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                )
                            } else if on_w_neg {
                                maze_gate_open(
                                    ux,
                                    uy,
                                    uz,
                                    w_neg_gate[0],
                                    w_neg_gate[1],
                                    w_neg_gate[2],
                                    MAZE_STRIDE,
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                )
                            } else {
                                maze_gate_open(
                                    ux,
                                    uy,
                                    uz,
                                    w_pos_gate[0],
                                    w_pos_gate[1],
                                    w_pos_gate[2],
                                    MAZE_STRIDE,
                                    MAZE_LEVEL_HEIGHT,
                                    MAZE_STRIDE,
                                )
                            };

                            if gate_open {
                                None
                            } else {
                                Some(set.maze_gate_frame_idx)
                            }
                        } else {
                            let wall_x = ux % MAZE_STRIDE == 0;
                            let wall_y = uy % MAZE_LEVEL_HEIGHT == 0;
                            let wall_z = uz % MAZE_STRIDE == 0;
                            let wall_w = uw % MAZE_STRIDE == 0;
                            let wall_count =
                                wall_x as i32 + wall_y as i32 + wall_z as i32 + wall_w as i32;

                            if wall_count == 0 {
                                if ux == center_u[0]
                                    && uy == center_u[1]
                                    && uz == center_u[2]
                                    && uw == center_u[3]
                                {
                                    Some(set.maze_beacon_idx)
                                } else {
                                    None
                                }
                            } else if wall_count > 1 {
                                Some(set.maze_wall_idx)
                            } else if wall_x {
                                if !wall_y && !wall_z && !wall_w {
                                    let left = ux / MAZE_STRIDE - 1;
                                    let right = left + 1;
                                    let cy = uy / MAZE_LEVEL_HEIGHT;
                                    let cz = uz / MAZE_STRIDE;
                                    let cw = uw / MAZE_STRIDE;
                                    if topology.edge_open([left, cy, cz, cw], [right, cy, cz, cw]) {
                                        None
                                    } else {
                                        Some(set.maze_wall_idx)
                                    }
                                } else {
                                    Some(set.maze_wall_idx)
                                }
                            } else if wall_y {
                                if !wall_x && !wall_z && !wall_w {
                                    let lower = uy / MAZE_LEVEL_HEIGHT - 1;
                                    let upper = lower + 1;
                                    let cx = ux / MAZE_STRIDE;
                                    let cz = uz / MAZE_STRIDE;
                                    let cw = uw / MAZE_STRIDE;
                                    if topology.edge_open([cx, lower, cz, cw], [cx, upper, cz, cw])
                                    {
                                        None
                                    } else {
                                        Some(set.maze_wall_idx)
                                    }
                                } else {
                                    Some(set.maze_wall_idx)
                                }
                            } else if wall_z {
                                if !wall_x && !wall_y && !wall_w {
                                    let near = uz / MAZE_STRIDE - 1;
                                    let far = near + 1;
                                    let cx = ux / MAZE_STRIDE;
                                    let cy = uy / MAZE_LEVEL_HEIGHT;
                                    let cw = uw / MAZE_STRIDE;
                                    if topology.edge_open([cx, cy, near, cw], [cx, cy, far, cw]) {
                                        None
                                    } else {
                                        Some(set.maze_wall_idx)
                                    }
                                } else {
                                    Some(set.maze_wall_idx)
                                }
                            } else if !wall_x && !wall_y && !wall_z {
                                let near = uw / MAZE_STRIDE - 1;
                                let far = near + 1;
                                let cx = ux / MAZE_STRIDE;
                                let cy = uy / MAZE_LEVEL_HEIGHT;
                                let cz = uz / MAZE_STRIDE;
                                if topology.edge_open([cx, cy, cz, near], [cx, cy, cz, far]) {
                                    None
                                } else {
                                    Some(set.maze_wall_idx)
                                }
                            } else {
                                Some(set.maze_wall_idx)
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
                    dense_chunk_set(chunk, lx, ly, lz, lw, material);
                }
            }
        }
    }
}

pub fn generate_structure_chunk_with_keepout(
    world_seed: u64,
    chunk_key: ChunkKey,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Option<DenseChunk> {
    let set = structure_set();
    let (chunk_min, _) = chunk_bounds(chunk_key);
    let structure_placements = collect_structure_placements_for_chunk(world_seed, chunk_key);
    let maze_placements = collect_maze_placements_for_chunk(world_seed, chunk_key);

    let mut chunk = dense_chunk_new();
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
        place_maze_into_chunk(placement, chunk_min, &mut chunk, set);
    }

    if dense_chunk_is_empty(&chunk) {
        None
    } else {
        Some(chunk)
    }
}

/// Get the block palette used by procgen DenseChunks.
/// Palette index 0 = AIR; indices 1..N map to the corresponding BlockData.
pub fn block_palette() -> &'static [BlockData] {
    &structure_set().block_palette
}

/// Intern a block into a palette, returning its palette index.
/// If the block already exists in the palette, returns the existing index.
pub fn intern_block_into_palette(palette: &mut Vec<BlockData>, block: &BlockData) -> u16 {
    if block.is_air() {
        return 0;
    }
    if let Some(idx) = palette.iter().position(|b| b == block) {
        return idx as u16;
    }
    let idx = palette.len() as u16;
    palette.push(block.clone());
    idx
}

/// Returns chunk positions that contain maze content within the given bounds.
/// This is the maze counterpart of structure_chunk_positions — excludes structures.
#[cfg(test)]
pub fn maze_chunk_positions_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
) -> Vec<ChunkKey> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let mut chunk_positions = HashSet::<[i32; 4]>::new();
    for placement in collect_maze_placements_for_chunk_bounds(world_seed, bounds) {
        let (maze_min, maze_max) = maze_bounds(placement.origin, placement.shape);
        let Some(covered_chunks) =
            intersect_world_bounds_as_chunk_bounds(maze_min, maze_max, bounds)
        else {
            continue;
        };
        let (cc_min, cc_max) = covered_chunks.to_chunk_lattice_bounds(0);
        for w in cc_min[3]..=cc_max[3] {
            for z in cc_min[2]..=cc_max[2] {
                for y in cc_min[1]..=cc_max[1] {
                    for x in cc_min[0]..=cc_max[0] {
                        chunk_positions.insert([x, y, z, w]);
                    }
                }
            }
        }
    }
    let mut out: Vec<ChunkKey> = chunk_positions
        .into_iter()
        .map(|[x, y, z, w]| [x, y, z, w].map(ChunkCoord::from_num))
        .collect();
    out.sort_unstable();
    out
}

/// Generate per-placement maze chunk data for all maze placements intersecting `bounds`.
///
/// Each maze placement returns its own set of chunks, allowing the caller to build
/// one ChunkArray per maze rather than inserting chunks one at a time.
pub fn generate_maze_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
) -> Vec<PlacementChunkData> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let placements = collect_maze_placements_for_chunk_bounds(world_seed, bounds);
    let chunk_size = CHUNK_SIZE as i32;

    let mut results = Vec::with_capacity(placements.len());
    for placement in placements {
        let (maze_min, maze_max) = maze_bounds(placement.origin, placement.shape);
        let Some(covered_chunks) =
            intersect_world_bounds_as_chunk_bounds(maze_min, maze_max, bounds)
        else {
            continue;
        };

        let (cc_min, cc_max) = covered_chunks.to_chunk_lattice_bounds(0);
        let mut chunks = HashMap::new();
        for cw in cc_min[3]..=cc_max[3] {
            for cz in cc_min[2]..=cc_max[2] {
                for cy in cc_min[1]..=cc_max[1] {
                    for cx in cc_min[0]..=cc_max[0] {
                        let chunk_min = [
                            cx * chunk_size,
                            cy * chunk_size,
                            cz * chunk_size,
                            cw * chunk_size,
                        ];
                        let mut chunk = dense_chunk_new();
                        place_maze_into_chunk(
                            MazePlacement {
                                origin: placement.origin,
                                layout_seed: placement.layout_seed,
                                shape: placement.shape,
                            },
                            chunk_min,
                            &mut chunk,
                            structure_set(),
                        );
                        if !dense_chunk_is_empty(&chunk) {
                            chunks.insert([cx, cy, cz, cw], chunk);
                        }
                    }
                }
            }
        }

        if !chunks.is_empty() {
            results.push(PlacementChunkData { chunks });
        }
    }
    results
}

/// Generate maze-only chunk content for a single chunk position (no structures).
#[cfg(test)]
pub fn generate_maze_chunk(
    world_seed: u64,
    chunk_key: ChunkKey,
) -> Option<DenseChunk> {
    let (chunk_min, _) = chunk_bounds(chunk_key);
    let maze_placements = collect_maze_placements_for_chunk(world_seed, chunk_key);
    if maze_placements.is_empty() {
        return None;
    }
    let mut chunk = dense_chunk_new();
    for placement in maze_placements {
        place_maze_into_chunk(placement, chunk_min, &mut chunk, structure_set());
    }
    if dense_chunk_is_empty(&chunk) {
        None
    } else {
        Some(chunk)
    }
}

#[cfg(test)]
pub fn generate_structure_chunk(world_seed: u64, chunk_key: ChunkKey) -> Option<DenseChunk> {
    generate_structure_chunk_with_keepout(world_seed, chunk_key, None)
}

#[cfg(test)]
pub fn structure_chunk_has_content_with_keepout(
    world_seed: u64,
    chunk_key: ChunkKey,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    let set = structure_set();
    let (chunk_min, chunk_max) = chunk_bounds(chunk_key);
    let has_structure = collect_structure_placements_for_chunk(world_seed, chunk_key)
        .into_iter()
        .any(|placement| {
            if !placement_allowed(&placement, blocked_cells) {
                return false;
            }
            let blueprint = &set.blueprints[placement.blueprint_idx];
            let (bp_min, bp_max) =
                blueprint.oriented_world_bounds(placement.origin, placement.orientation);
            (0..4).all(|a| bp_max[a] >= chunk_min[a] && bp_min[a] <= chunk_max[a])
        });
    if has_structure {
        return true;
    }
    // Also check for maze content in this chunk
    !collect_maze_placements_for_chunk(world_seed, chunk_key).is_empty()
}

#[cfg(test)]
pub fn structure_chunk_has_content(world_seed: u64, chunk_key: ChunkKey) -> bool {
    structure_chunk_has_content_with_keepout(world_seed, chunk_key, None)
}

#[cfg(test)]
pub fn structure_chunk_has_content_for_scale_with_keepout(
    world_seed: u64,
    chunk_key: ChunkKey,
    chunk_scale: i32,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    let scale = chunk_scale.max(1);
    if scale == 1 {
        return structure_chunk_has_content_with_keepout(world_seed, chunk_key, blocked_cells);
    }

    let cx = chunk_key[0].to_num::<i32>();
    let cy = chunk_key[1].to_num::<i32>();
    let cz = chunk_key[2].to_num::<i32>();
    let cw = chunk_key[3].to_num::<i32>();
    for dw in 0..scale {
        for dz in 0..scale {
            for dy in 0..scale {
                for dx in 0..scale {
                    let child = [
                        cx * scale + dx,
                        cy * scale + dy,
                        cz * scale + dz,
                        cw * scale + dw,
                    ].map(ChunkCoord::from_num);
                    if structure_chunk_has_content_with_keepout(world_seed, child, blocked_cells) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
pub fn structure_chunk_has_content_for_scale(
    world_seed: u64,
    chunk_key: ChunkKey,
    chunk_scale: i32,
) -> bool {
    structure_chunk_has_content_for_scale_with_keepout(world_seed, chunk_key, chunk_scale, None)
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
    use crate::shared::region_tree::chunk_key_i32;
    use std::collections::HashSet;

    fn ck(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
        chunk_key_i32(x, y, z, w)
    }

    fn find_chunk_with_maze(seed: u64) -> Option<ChunkKey> {
        for cell_x in -8..=8 {
            for cell_z in -8..=8 {
                for cell_w in -8..=8 {
                    let cell_hash =
                        hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                    if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                        continue;
                    }
                    let shape = maze_shape_from_cell_hash(cell_hash);

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

                    let chunk_key = [
                        origin_x.div_euclid(CHUNK_SIZE as i32),
                        shape.world_y_min.div_euclid(CHUNK_SIZE as i32),
                        origin_z.div_euclid(CHUNK_SIZE as i32),
                        origin_w.div_euclid(CHUNK_SIZE as i32),
                    ].map(ChunkCoord::from_num);
                    if !collect_maze_placements_for_chunk(seed, chunk_key).is_empty() {
                        return Some(chunk_key);
                    }
                }
            }
        }
        None
    }

    #[test]
    fn procedural_maze_bounds_vary_in_all_dimensions() {
        let seed = 0x4b31_c7a9_529d_17f2u64;
        let mut span_x_values = HashSet::new();
        let mut span_y_values = HashSet::new();
        let mut span_z_values = HashSet::new();
        let mut span_w_values = HashSet::new();
        let mut y_min_values = HashSet::new();

        'scan: for cell_x in -16..=16 {
            for cell_z in -16..=16 {
                for cell_w in -16..=16 {
                    let cell_hash =
                        hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                    if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                        continue;
                    }
                    let shape = maze_shape_from_cell_hash(cell_hash);
                    span_x_values.insert(shape.span[0]);
                    span_y_values.insert(shape.span[1]);
                    span_z_values.insert(shape.span[2]);
                    span_w_values.insert(shape.span[3]);
                    y_min_values.insert(shape.world_y_min);

                    if span_x_values.len() > 1
                        && span_y_values.len() > 1
                        && span_z_values.len() > 1
                        && span_w_values.len() > 1
                        && y_min_values.len() > 1
                    {
                        break 'scan;
                    }
                }
            }
        }

        assert!(
            span_x_values.len() > 1,
            "expected procedural mazes to vary x span"
        );
        assert!(
            span_y_values.len() > 1,
            "expected procedural mazes to vary y span"
        );
        assert!(
            span_z_values.len() > 1,
            "expected procedural mazes to vary z span"
        );
        assert!(
            span_w_values.len() > 1,
            "expected procedural mazes to vary w span"
        );
        assert!(
            y_min_values.len() > 1,
            "expected procedural mazes to vary y min bounds"
        );
    }

    #[test]
    fn procedural_maze_topology_uses_y_dimension() {
        let layout_seed = 0x5cf2_9b44_1de3_a871u64;
        let shape = MazeShape {
            grid_cells: [9, 5, 9, 9],
            span: [
                9 * MAZE_STRIDE + 1,
                5 * MAZE_LEVEL_HEIGHT + 1,
                9 * MAZE_STRIDE + 1,
                9 * MAZE_STRIDE + 1,
            ],
            half_span_xzw: [0, 0, 0],
            world_y_min: 0,
            variant: MazeVariant::Vertical,
        };
        let topology = maze_build_topology(layout_seed, shape);
        let mut open_vertical_edges = 0usize;
        for x in 0..shape.grid_cells[0] {
            for y in 0..(shape.grid_cells[1] - 1) {
                for z in 0..shape.grid_cells[2] {
                    for w in 0..shape.grid_cells[3] {
                        if topology.edge_open([x, y, z, w], [x, y + 1, z, w]) {
                            open_vertical_edges += 1;
                        }
                    }
                }
            }
        }
        assert!(
            open_vertical_edges > 0,
            "expected topology to include vertical connections"
        );
    }

    #[test]
    fn procedural_maze_variants_are_deterministic_and_diverse() {
        let seed = 0x2cab_4d10_3fc8_9912u64;
        let mut variants = HashSet::new();

        'scan: for cell_x in -20..=20 {
            for cell_z in -20..=20 {
                for cell_w in -20..=20 {
                    let cell_hash =
                        hash_structure_cell(seed, cell_x, cell_z, cell_w, MAZE_HASH_SALT);
                    if cell_hash % MAZE_SPAWN_DENOMINATOR >= MAZE_SPAWN_NUMERATOR {
                        continue;
                    }
                    let variant_a = maze_shape_from_cell_hash(cell_hash).variant;
                    let variant_b = maze_shape_from_cell_hash(cell_hash).variant;
                    assert_eq!(variant_a, variant_b, "maze variant must be deterministic");
                    variants.insert(variant_a);
                    if variants.len() == 3 {
                        break 'scan;
                    }
                }
            }
        }

        assert_eq!(
            variants.len(),
            3,
            "expected all maze variant presets to appear across sampled spawn cells"
        );
    }

    #[test]
    fn structure_chunk_generation_is_seed_deterministic() {
        let chunk_a = generate_structure_chunk(42, ck(6, 0, 4, -3));
        let chunk_b = generate_structure_chunk(42, ck(6, 0, 4, -3));

        assert_eq!(chunk_a.is_some(), chunk_b.is_some());
        if let (Some(a), Some(b)) = (chunk_a, chunk_b) {
            assert_eq!(a[..], b[..]);
        }
    }

    #[test]
    fn structure_generation_produces_non_empty_chunks() {
        let mut found_non_empty = false;
        let (min_y, max_y) = structure_chunk_y_bounds();

        for x in -25..=25 {
            for z in -25..=25 {
                for w in -25..=25 {
                    for y in min_y..=max_y {
                        if generate_structure_chunk(1337, ck(x, y, z, w)).is_some() {
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
        let set = structure_set();
        let maze_indices = [
            set.maze_ceiling_idx,
            set.maze_wall_idx,
            set.maze_gate_frame_idx,
            set.maze_beacon_idx,
        ];
        let has_maze_material = chunk.iter().any(|&mat| maze_indices.contains(&mat));
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
        assert_eq!(chunk_a[..], chunk_b[..]);
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

            let mut left_chunk = dense_chunk_new();
            let mut right_chunk = dense_chunk_new();
            blueprint.place_into_chunk_oriented(origin, 0, [0, 0, 0, 0], &mut left_chunk);
            blueprint.place_into_chunk_oriented(
                origin,
                0,
                [CHUNK_SIZE as i32, 0, 0, 0],
                &mut right_chunk,
            );

            if !dense_chunk_is_empty(&left_chunk) && !dense_chunk_is_empty(&right_chunk) {
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

        'scan: for x in -35..=35 {
            for z in -35..=35 {
                for w in -35..=35 {
                    for y in min_y..=max_y {
                        let left = ck(x, y, z, w);
                        let right = ck(x + 1, y, z, w);
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
                        let chunk_pos = ck(x, y, z, w);
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
    fn structure_chunk_positions_for_bounds_matches_bruteforce() {
        let seed = 0x4e73_9ac1_2f07_118du64;
        let bounds = Aabb4i::from_lattice_bounds([-3, -2, -3, -3], [3, 3, 3, 3], 0);

        let mut brute = Vec::new();
        let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
        for w in bmin[3]..=bmax[3] {
            for z in bmin[2]..=bmax[2] {
                for y in bmin[1]..=bmax[1] {
                    for x in bmin[0]..=bmax[0] {
                        let pos = ck(x, y, z, w);
                        if structure_chunk_has_content_with_keepout(seed, pos, None) {
                            brute.push(pos);
                        }
                    }
                }
            }
        }
        brute.sort_unstable();

        let fast = structure_chunk_positions_for_bounds_with_keepout(seed, bounds, None);
        assert_eq!(fast, brute);
    }

    #[test]
    fn structure_chunk_positions_for_bounds_respects_keepout_cells() {
        let seed = 0x2a19_6e8d_0cb4_7fd1u64;
        let (min_y, max_y) = structure_chunk_y_bounds();
        let mut target = None::<(ChunkKey, StructureCell)>;
        'search: for x in -30..=30 {
            for z in -30..=30 {
                for w in -30..=30 {
                    for y in min_y..=max_y {
                        let pos = ck(x, y, z, w);
                        if generate_structure_chunk(seed, pos).is_none() {
                            continue;
                        }
                        let cells = structure_cells_affecting_chunk(seed, pos);
                        if let Some(cell) = cells.first().copied() {
                            target = Some((pos, cell));
                            break 'search;
                        }
                    }
                }
            }
        }
        let (target_pos, blocked_cell) = target.expect("expected target structure chunk");
        let tp = [
            target_pos[0].to_num::<i32>(),
            target_pos[1].to_num::<i32>(),
            target_pos[2].to_num::<i32>(),
            target_pos[3].to_num::<i32>(),
        ];
        let bounds = Aabb4i::from_lattice_bounds(
            [tp[0] - 2, tp[1] - 2, tp[2] - 2, tp[3] - 2],
            [tp[0] + 2, tp[1] + 2, tp[2] + 2, tp[3] + 2],
            0,
        );
        let mut blocked = HashSet::new();
        blocked.insert(blocked_cell);

        let mut brute = Vec::new();
        let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
        for w in bmin[3]..=bmax[3] {
            for z in bmin[2]..=bmax[2] {
                for y in bmin[1]..=bmax[1] {
                    for x in bmin[0]..=bmax[0] {
                        let pos = ck(x, y, z, w);
                        if structure_chunk_has_content_with_keepout(seed, pos, Some(&blocked)) {
                            brute.push(pos);
                        }
                    }
                }
            }
        }
        brute.sort_unstable();

        let fast = structure_chunk_positions_for_bounds_with_keepout(seed, bounds, Some(&blocked));
        assert_eq!(fast, brute);
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
                        let coarse = ck(x, y, z, w);
                        let mut expected = false;
                        for dw in 0..scale {
                            for dz in 0..scale {
                                for dy in 0..scale {
                                    for dx in 0..scale {
                                        let child = ck(
                                            x * scale + dx,
                                            y * scale + dy,
                                            z * scale + dz,
                                            w * scale + dw,
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
        'search: for x in -30..=30 {
            for z in -30..=30 {
                for w in -30..=30 {
                    for y in min_y..=max_y {
                        let chunk_pos = ck(x, y, z, w);
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
