use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BlockData, CHUNK_SIZE};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::collections::HashSet;
use std::io::Read;

use super::{
    chunk_bounds, dense_chunk_is_empty, dense_chunk_new, dense_chunk_set, hash_structure_cell,
    jitter_from_hash, rotate_offset_xzw, splitmix64, structure_chunk_y_bounds, structure_set,
    world_bounds_from_chunk_bounds, world_bounds_intersect, DenseChunk, PlacementChunkData,
};

pub(super) const STRUCTURE_CELL_SIZE: i32 = 32;
pub(super) const STRUCTURE_CELL_JITTER: i32 = 10;
const STRUCTURE_SPAWN_NUMERATOR: u64 = 1;
const STRUCTURE_SPAWN_DENOMINATOR: u64 = 480;
const STRUCTURE_ORIGIN_EXCLUSION_RADIUS: i32 = 16;
const STRUCTURE_HASH_SALT: u64 = 0x9f07_c9ab_33f2_3a11;
const STRUCTURE_PICK_SALT: u64 = 0x2d99_1f4e_47ba_8c6d;
const STRUCTURE_ROTATION_SALT: u64 = 0x54c7_9be0_2f61_0d93;
pub(super) const JITTER_X_SALT: u64 = 0x1c69_b3f7_4d87_2ba1;
pub(super) const JITTER_Z_SALT: u64 = 0x8ab3_d165_52cc_91f3;
pub(super) const JITTER_W_SALT: u64 = 0xf2c6_0c4a_0a8a_d53d;
const ROTATION_VARIANTS: u64 = 48;

pub type StructureCell = [i32; 3];

fn default_spawn_weight() -> u32 {
    1
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
pub(super) struct StructureFill {
    pub(super) min: [i32; 4],
    pub(super) size: [i32; 4],
    pub(super) block: BlockData,
    #[serde(skip)]
    pub(super) palette_idx: u16,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StructureVoxel {
    pub(super) offset: [i32; 4],
    pub(super) block: BlockData,
    #[serde(skip)]
    pub(super) palette_idx: u16,
}

#[derive(Clone, Debug)]
pub(super) struct StructureBlueprint {
    pub(super) spawn_weight: u32,
    pub(super) ground_offset_y: i32,
    pub(super) fills: Vec<StructureFill>,
    pub(super) voxels: Vec<StructureVoxel>,
    pub(super) min_offset: [i32; 4],
    pub(super) max_offset: [i32; 4],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct StructurePlacement {
    pub(super) blueprint_idx: usize,
    pub(super) origin: [i32; 4],
    pub(super) orientation: u8,
    pub(super) cell: StructureCell,
}

impl StructureBlueprint {
    pub(super) fn from_embedded_json_bytes(file_name: &str, json: &[u8]) -> Self {
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

    pub(super) fn from_embedded_source(file_name: &str, source: &[u8]) -> Self {
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

    pub(super) fn intersects_chunk_oriented(
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

    pub(super) fn place_into_chunk_oriented(
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

    pub(super) fn oriented_world_bounds(
        &self,
        origin: [i32; 4],
        orientation: u8,
    ) -> ([i32; 4], [i32; 4]) {
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

pub(super) fn placement_allowed(
    placement: &StructurePlacement,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> bool {
    blocked_cells
        .map(|blocked| !blocked.contains(&placement.cell))
        .unwrap_or(true)
}

pub(super) fn collect_structure_placements_for_chunk_bounds(
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

pub(super) fn collect_structure_placements_for_chunk(
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

pub(super) fn generate_structure_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
) -> Vec<PlacementChunkData> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let set = structure_set();
    let placements =
        collect_structure_placements_for_chunk_bounds(world_seed, bounds, blocked_cells);

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
        let mut chunks = std::collections::HashMap::new();
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
