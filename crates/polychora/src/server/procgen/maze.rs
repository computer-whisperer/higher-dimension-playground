use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::CHUNK_SIZE;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::{
    chunk_bounds, dense_chunk_is_empty, dense_chunk_new, dense_chunk_set, hash_structure_cell,
    intersect_world_bounds_as_chunk_bounds, jitter_from_hash_with_radius, splitmix64,
    structure_set, world_bounds_from_chunk_bounds, world_bounds_intersect, DenseChunk,
    PlacementChunkData, StructureSet,
};

pub(super) const MAZE_CELL_SIZE: i32 = 128;
pub(super) const MAZE_CELL_JITTER: i32 = 20;
pub(super) const MAZE_SPAWN_NUMERATOR: u64 = 1;
pub(super) const MAZE_SPAWN_DENOMINATOR: u64 = 36;
pub(super) const MAZE_ORIGIN_EXCLUSION_RADIUS: i32 = 24;
pub(super) const MAZE_HASH_SALT: u64 = 0x6f1d_05ce_294a_719b;
const MAZE_LAYOUT_SALT: u64 = 0x49ec_66d6_0d13_9e75;
pub(super) const MAZE_VARIANT_SALT: u64 = 0x932f_b43c_f1f9_746d;
const MAZE_EDGE_SORT_SALT: u64 = 0xba04_7f9b_11d2_c638;
const MAZE_BRAID_SALT: u64 = 0x0d7c_88e1_a4f3_5b29;
const MAZE_EDGE_X_SALT: u64 = 0x3374_11a9_5f9c_2d01;
const MAZE_EDGE_Y_SALT: u64 = 0x40ad_6f0c_22be_8a17;
const MAZE_EDGE_Z_SALT: u64 = 0x9bc3_51d4_788a_f2ee;
const MAZE_EDGE_W_SALT: u64 = 0x1ec8_daa5_4b73_99c0;
pub(super) const MAZE_JITTER_X_SALT: u64 = 0x5fa7_6d28_0f8b_81c3;
pub(super) const MAZE_JITTER_Z_SALT: u64 = 0x8af0_1f7a_cc99_20be;
pub(super) const MAZE_JITTER_W_SALT: u64 = 0x163a_8b39_b0a2_6741;
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
pub(super) const MAZE_GRID_X_SALT: u64 = 0x2c93_f17a_6540_8e11;
pub(super) const MAZE_GRID_Y_SALT: u64 = 0x1720_8d59_3bf4_2c77;
pub(super) const MAZE_GRID_Z_SALT: u64 = 0x85a4_5b0d_9926_f5c4;
pub(super) const MAZE_GRID_W_SALT: u64 = 0xf30d_7c9e_4ab2_1688;
pub(super) const MAZE_Y_JITTER_SALT: u64 = 0x6f5e_2914_a7cd_3b90;
pub(super) const MAZE_GRID_XZW_CELLS_MIN: i32 = 9;
pub(super) const MAZE_GRID_XZW_CELLS_MAX: i32 = 15;
pub(super) const MAZE_GRID_Y_CELLS_MIN: i32 = 3;
pub(super) const MAZE_GRID_Y_CELLS_MAX: i32 = 7;
pub(super) const MAZE_STRIDE: i32 = 4;
pub(super) const MAZE_LEVEL_HEIGHT: i32 = 4;
pub(super) const MAZE_WORLD_Y_BASE: i32 = 0;
pub(super) const MAZE_WORLD_Y_JITTER: i32 = 0;
pub(super) const MAZE_MAX_HALF_SPAN_XZW: i32 = (MAZE_GRID_XZW_CELLS_MAX * MAZE_STRIDE + 1) / 2;
const MAZE_LAYOUT_CACHE_CAPACITY: usize = 96;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct MazePlacement {
    pub(super) origin: [i32; 4],
    pub(super) layout_seed: u64,
    pub(super) shape: MazeShape,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct MazeShape {
    pub(super) grid_cells: [i32; 4],
    pub(super) span: [i32; 4],
    pub(super) half_span_xzw: [i32; 3],
    pub(super) world_y_min: i32,
    pub(super) variant: MazeVariant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum MazeVariant {
    Catacomb,
    Vertical,
    Braided,
}

impl MazeVariant {
    pub(super) fn from_cell_hash(cell_hash: u64) -> Self {
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

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Catacomb => "catacomb",
            Self::Vertical => "vertical",
            Self::Braided => "braided",
        }
    }
}

pub(super) fn maze_random_odd(hash: u64, min_value: i32, max_value: i32) -> i32 {
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

pub(super) fn maze_shape_from_cell_hash(cell_hash: u64) -> MazeShape {
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

pub(super) fn maze_world_y_bounds() -> (i32, i32) {
    (
        MAZE_WORLD_Y_BASE - MAZE_WORLD_Y_JITTER,
        MAZE_WORLD_Y_BASE + MAZE_WORLD_Y_JITTER + MAZE_GRID_Y_CELLS_MAX * MAZE_LEVEL_HEIGHT,
    )
}

pub(super) fn maze_bounds(origin: [i32; 4], shape: MazeShape) -> ([i32; 4], [i32; 4]) {
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
pub(super) struct MazeTopology {
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

    pub(super) fn edge_open(&self, a: [i32; 4], b: [i32; 4]) -> bool {
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
pub(super) struct MazeCompiledLayout {
    pub(super) topology: MazeTopology,
    x_neg_gate: [i32; 3],
    x_pos_gate: [i32; 3],
    z_neg_gate: [i32; 3],
    z_pos_gate: [i32; 3],
    w_neg_gate: [i32; 3],
    w_pos_gate: [i32; 3],
    pub(super) center_u: [i32; 4],
}

#[derive(Clone, Debug)]
struct MazeLayoutCacheEntry {
    layout: Arc<MazeCompiledLayout>,
    last_used: u64,
}

#[derive(Clone, Debug, Default)]
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

pub(super) fn maze_build_topology(layout_seed: u64, shape: MazeShape) -> MazeTopology {
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

pub(super) fn collect_maze_placements_for_chunk(
    world_seed: u64,
    chunk_key: ChunkKey,
) -> Vec<MazePlacement> {
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

pub(super) fn collect_maze_placements_for_chunk_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<MazePlacement> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let (maze_world_min_y, maze_world_max_y) = maze_world_y_bounds();
    let (query_world_min, query_world_max) = world_bounds_from_chunk_bounds(bounds);
    if query_world_max[1] < maze_world_min_y || query_world_min[1] > maze_world_max_y {
        return Vec::new();
    }

    let (spawn_num, spawn_den) = if let Some((scale_num, scale_den)) = spawn_rate_scale {
        (MAZE_SPAWN_NUMERATOR * scale_num, MAZE_SPAWN_DENOMINATOR * scale_den)
    } else {
        (MAZE_SPAWN_NUMERATOR, MAZE_SPAWN_DENOMINATOR)
    };

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
                if cell_hash % spawn_den >= spawn_num {
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

pub(super) fn place_maze_into_chunk(
    placement: MazePlacement,
    chunk_min: [i32; 4],
    chunk: &mut DenseChunk,
    set: &StructureSet,
) {
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

pub(super) fn generate_maze_placements_for_bounds(
    world_seed: u64,
    bounds: Aabb4i,
    origin_bounds_xzw: Option<([i32; 3], [i32; 3])>,
    spawn_rate_scale: Option<(u64, u64)>,
) -> Vec<PlacementChunkData> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let placements = collect_maze_placements_for_chunk_bounds(world_seed, bounds, origin_bounds_xzw, spawn_rate_scale);
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
