//! Placement grid logic for procedural structures.
//!
//! This module determines WHERE structures spawn in the world (cell hashing,
//! jitter, spawn rates, keepout zones). The actual generation — including
//! mazes, which are just another structure type — is handled by the WASM
//! content plugin.

use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::CHUNK_SIZE;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Placement constants
// ---------------------------------------------------------------------------

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

/// Conservative upper bound on how far a structure can extend from its origin
/// in any XZW direction (in voxels). Must cover the largest possible structure
/// including mazes (max half-span ~30 voxels).
const STRUCTURE_MAX_EXTENT_XZW: i32 = 34;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

pub type StructureCell = [i32; 3];

/// Configuration for structure placement, derived from the WASM plugin's
/// structure declarations.
pub struct StructurePlacementConfig {
    /// Per-blueprint spawn weights (index = blueprint_idx).
    pub blueprint_weights: Vec<u32>,
    /// Sum of all blueprint weights.
    pub total_weight: u64,
}

impl StructurePlacementConfig {
    pub fn from_weights(weights: &[u32]) -> Self {
        let total_weight = weights.iter().map(|&w| w.max(1) as u64).sum::<u64>().max(1);
        Self {
            blueprint_weights: weights.to_vec(),
            total_weight,
        }
    }

    fn pick_blueprint_index(&self, mut roll: u64) -> usize {
        for (idx, &w) in self.blueprint_weights.iter().enumerate() {
            let weight = w.max(1) as u64;
            if roll < weight {
                return idx;
            }
            roll -= weight;
        }
        self.blueprint_weights.len().saturating_sub(1)
    }
}

#[derive(Clone, Debug)]
pub struct StructurePlacementReport {
    /// Structure origin in local (procgen-grid) space. Y is always 0;
    /// the WASM plugin applies per-blueprint Y offsets internally.
    pub origin: [i32; 4],
    pub blueprint_idx: usize,
    pub orientation: u8,
    pub cell_hash: u64,
}

// ---------------------------------------------------------------------------
// Hash / jitter utilities
// ---------------------------------------------------------------------------

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
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

fn jitter_from_hash_with_radius(hash: u64, radius: i32) -> i32 {
    let span = (radius * 2 + 1) as u64;
    (splitmix64(hash) % span) as i32 - radius
}

fn world_bounds_from_chunk_bounds(bounds: Aabb4i) -> ([i32; 4], [i32; 4]) {
    let min = bounds.min.map(|v| v.to_num::<i32>());
    let max = bounds.max.map(|v| v.to_num::<i32>() - 1);
    (min, max)
}

fn placement_allowed(cell: StructureCell, blocked_cells: Option<&HashSet<StructureCell>>) -> bool {
    blocked_cells
        .map(|blocked| !blocked.contains(&cell))
        .unwrap_or(true)
}

// ---------------------------------------------------------------------------
// Public API: Y bounds and overhang
// ---------------------------------------------------------------------------

/// Conservative chunk-Y range where procgen content may exist.
pub fn structure_chunk_y_bounds() -> (i32, i32) {
    let cs = CHUNK_SIZE as i32;
    // Structures: ground_offset_y range is roughly -16..+8 in voxels, plus
    // blueprint vertical extent of ~20 voxels.
    // Mazes: Y base is 0, max height is ~29 voxels (7 levels * 4 + 1).
    let min_y: i32 = -16;
    let max_y: i32 = 32;
    (min_y.div_euclid(cs), max_y.div_euclid(cs))
}

/// Maximum distance (in chunks) that a structure can extend beyond
/// its placement origin on any XZW axis.
pub fn max_structure_overhang_chunks() -> i32 {
    let max_voxels = STRUCTURE_MAX_EXTENT_XZW + STRUCTURE_CELL_JITTER;
    (max_voxels + CHUNK_SIZE as i32 - 1) / CHUNK_SIZE as i32
}

// ---------------------------------------------------------------------------
// Public API: structure placement reports
// ---------------------------------------------------------------------------

pub fn collect_structure_placement_reports(
    seed: u64,
    bounds: Aabb4i,
    blocked: Option<&HashSet<StructureCell>>,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
    config: &StructurePlacementConfig,
) -> Vec<StructurePlacementReport> {
    if !bounds.is_valid() {
        return Vec::new();
    }

    let (spawn_num, spawn_den) = if let Some((scale_num, scale_den)) = spawn_rate_scale {
        (
            STRUCTURE_SPAWN_NUMERATOR * scale_num,
            STRUCTURE_SPAWN_DENOMINATOR * scale_den,
        )
    } else {
        (STRUCTURE_SPAWN_NUMERATOR, STRUCTURE_SPAWN_DENOMINATOR)
    };

    let (query_world_min, query_world_max) = world_bounds_from_chunk_bounds(bounds);
    let search_margin = STRUCTURE_CELL_JITTER + STRUCTURE_MAX_EXTENT_XZW + CHUNK_SIZE as i32;
    let cell_min_x = (query_world_min[0] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_x = (query_world_max[0] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_z = (query_world_min[2] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_z = (query_world_max[2] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;
    let cell_min_w = (query_world_min[3] - search_margin).div_euclid(STRUCTURE_CELL_SIZE) - 1;
    let cell_max_w = (query_world_max[3] + search_margin).div_euclid(STRUCTURE_CELL_SIZE) + 1;

    let mut reports = Vec::new();
    for cell_x in cell_min_x..=cell_max_x {
        for cell_z in cell_min_z..=cell_max_z {
            for cell_w in cell_min_w..=cell_max_w {
                let cell_hash =
                    hash_structure_cell(seed, cell_x, cell_z, cell_w, STRUCTURE_HASH_SALT);
                if cell_hash % spawn_den >= spawn_num {
                    continue;
                }

                let origin_x = cell_x * STRUCTURE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ JITTER_X_SALT,
                        STRUCTURE_CELL_JITTER,
                    );
                let origin_z = cell_z * STRUCTURE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ JITTER_Z_SALT,
                        STRUCTURE_CELL_JITTER,
                    );
                let origin_w = cell_w * STRUCTURE_CELL_SIZE
                    + jitter_from_hash_with_radius(
                        cell_hash ^ JITTER_W_SALT,
                        STRUCTURE_CELL_JITTER,
                    );

                if origin_x.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                    && origin_z.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                    && origin_w.abs() <= STRUCTURE_ORIGIN_EXCLUSION_RADIUS
                {
                    continue;
                }

                if let Some((min_xzw, max_xzw)) = origin_bounds_xzw {
                    if origin_x < min_xzw[0]
                        || origin_x > max_xzw[0]
                        || origin_z < min_xzw[1]
                        || origin_z > max_xzw[1]
                        || origin_w < min_xzw[2]
                        || origin_w > max_xzw[2]
                    {
                        continue;
                    }
                }

                let cell = [cell_x, cell_z, cell_w];
                if !placement_allowed(cell, blocked) {
                    continue;
                }

                let pick_roll = splitmix64(cell_hash ^ STRUCTURE_PICK_SALT) % config.total_weight;
                let blueprint_idx = config.pick_blueprint_index(pick_roll);
                let orientation =
                    (splitmix64(cell_hash ^ STRUCTURE_ROTATION_SALT) % ROTATION_VARIANTS) as u8;
                // Origin Y = 0: the WASM plugin applies per-blueprint Y offsets.
                let origin = [origin_x, 0, origin_z, origin_w];

                reports.push(StructurePlacementReport {
                    origin,
                    blueprint_idx,
                    orientation,
                    cell_hash,
                });
            }
        }
    }
    reports
}
