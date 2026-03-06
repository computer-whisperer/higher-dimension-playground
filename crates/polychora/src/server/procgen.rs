mod maze;
mod structure;

use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::{Aabb4i, ChunkCoord};
use crate::shared::voxel::{BlockData, CHUNK_SIZE, CHUNK_VOLUME};
use polychora_plugin_api::content_ids;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use self::maze::{
    collect_maze_placements_for_chunk, collect_maze_placements_for_chunk_bounds, maze_bounds,
    maze_world_y_bounds, place_maze_into_chunk, MAZE_CELL_JITTER, MAZE_MAX_HALF_SPAN_XZW,
};
#[cfg(test)]
use self::maze::{
    maze_build_topology, maze_shape_from_cell_hash, MazeShape, MazeVariant, MAZE_CELL_SIZE,
    MAZE_HASH_SALT, MAZE_JITTER_W_SALT, MAZE_JITTER_X_SALT, MAZE_JITTER_Z_SALT, MAZE_LEVEL_HEIGHT,
    MAZE_ORIGIN_EXCLUSION_RADIUS, MAZE_SPAWN_DENOMINATOR, MAZE_SPAWN_NUMERATOR, MAZE_STRIDE,
};
use self::structure::{
    collect_structure_placements_for_chunk, collect_structure_placements_for_chunk_bounds,
    placement_allowed, StructureBlueprint, STRUCTURE_CELL_JITTER,
};

// Re-export public API items used by maze and structure submodules.
pub use self::maze::clear_runtime_maze_layout_cache;
pub use self::structure::StructureCell;

type DenseChunk = [u16; CHUNK_VOLUME];

#[inline]
fn dense_chunk_new() -> DenseChunk {
    [0u16; CHUNK_VOLUME]
}

#[inline]
fn dense_chunk_set(chunk: &mut DenseChunk, x: usize, y: usize, z: usize, w: usize, material: u16) {
    let idx =
        w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x;
    chunk[idx] = material;
}

#[inline]
fn dense_chunk_is_empty(chunk: &DenseChunk) -> bool {
    chunk.iter().all(|&v| v == 0)
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
                voxel.palette_idx =
                    intern_block(&mut block_palette, &mut block_to_idx, &voxel.block);
            }
        }

        // Intern maze materials.
        let maze_floor_idx = intern_block(
            &mut block_palette,
            &mut block_to_idx,
            &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_BASALT_TILES),
        );
        let maze_ceiling_idx = intern_block(
            &mut block_palette,
            &mut block_to_idx,
            &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_SMOKED_GLASS),
        );
        let maze_wall_idx = intern_block(
            &mut block_palette,
            &mut block_to_idx,
            &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_RUNIC_ALLOY),
        );
        let maze_gate_frame_idx = intern_block(
            &mut block_palette,
            &mut block_to_idx,
            &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_OBSIDIAN),
        );
        let maze_beacon_idx = intern_block(
            &mut block_palette,
            &mut block_to_idx,
            &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_EVENTIDE_ALLOY),
        );

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

/// Maximum distance (in chunks) that a structure or maze can extend beyond
/// its platform edge on any XZW axis.
pub fn max_structure_overhang_chunks() -> i32 {
    let set = structure_set();
    // Structure overhang: max blueprint extent + jitter from cell center.
    let structure_voxels = set.max_abs_offset_xzw + STRUCTURE_CELL_JITTER;
    // Maze overhang: half-span + jitter from cell center.
    let maze_voxels = MAZE_MAX_HALF_SPAN_XZW + MAZE_CELL_JITTER;
    let max_voxels = structure_voxels.max(maze_voxels);
    (max_voxels + CHUNK_SIZE as i32 - 1) / CHUNK_SIZE as i32
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

/// Per-placement chunk data: a map of chunk positions to dense chunks generated from a single
/// structure placement.
pub struct PlacementChunkData {
    /// chunk_pos -> generated dense chunk (non-empty only, from this one placement).
    pub chunks: HashMap<[i32; 4], DenseChunk>,
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

// ---------------------------------------------------------------------------
// Public API: combined structure + maze generation
// ---------------------------------------------------------------------------

pub fn structure_chunk_positions_for_bounds_with_keepout(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<ChunkKey> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let set = structure_set();
    let mut chunk_positions = HashSet::<[i32; 4]>::new();

    for placement in
        collect_structure_placements_for_chunk_bounds(world_seed, bounds, blocked_cells, origin_bounds_xzw, spawn_rate_scale)
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

    for placement in collect_maze_placements_for_chunk_bounds(world_seed, bounds, origin_bounds_xzw, spawn_rate_scale) {
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

pub fn generate_structure_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    blocked_cells: Option<&HashSet<StructureCell>>,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<PlacementChunkData> {
    structure::generate_structure_placements_for_bounds(world_seed, bounds, blocked_cells, origin_bounds_xzw, spawn_rate_scale)
}

pub fn generate_maze_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<PlacementChunkData> {
    maze::generate_maze_placements_for_bounds(world_seed, bounds, origin_bounds_xzw, spawn_rate_scale)
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

// ---------------------------------------------------------------------------
// Placement report types (public API for seed scanning)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StructurePlacementReport {
    pub origin: [i32; 4],
    pub blueprint_idx: usize,
    pub orientation: u8,
}

#[derive(Clone, Debug)]
pub struct MazePlacementReport {
    pub origin: [i32; 4],
    pub grid_cells: [i32; 4],
    pub variant_name: &'static str,
}

pub fn collect_structure_placement_reports(
    seed: u64,
    bounds: Aabb4i,
    blocked: Option<&HashSet<StructureCell>>,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<StructurePlacementReport> {
    collect_structure_placements_for_chunk_bounds(seed, bounds, blocked, origin_bounds_xzw, spawn_rate_scale)
        .into_iter()
        .map(|p| StructurePlacementReport {
            origin: p.origin,
            blueprint_idx: p.blueprint_idx,
            orientation: p.orientation,
        })
        .collect()
}

pub fn collect_maze_placement_reports(
    seed: u64,
    bounds: Aabb4i,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<MazePlacementReport> {
    collect_maze_placements_for_chunk_bounds(seed, bounds, origin_bounds_xzw, spawn_rate_scale)
        .into_iter()
        .map(|p| MazePlacementReport {
            origin: p.origin,
            grid_cells: p.shape.grid_cells,
            variant_name: p.shape.variant.name(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test-only helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
pub fn generate_structure_chunk(world_seed: u64, chunk_key: ChunkKey) -> Option<DenseChunk> {
    generate_structure_chunk_with_keepout(world_seed, chunk_key, None)
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
                    ]
                    .map(ChunkCoord::from_num);
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

#[cfg(test)]
mod tests;
