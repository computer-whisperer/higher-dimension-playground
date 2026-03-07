use alloc::vec;
use alloc::vec::Vec;
use polychora_plugin_api::content_ids;
use polychora_plugin_api::procgen_abi::{
    ProcgenGenerateInput, ProcgenGenerateOutput, ProcgenPrepareInput, ProcgenPrepareOutput,
};
use polychora_plugin_api::region_tree::{
    BlockData, ChunkArrayData, ChunkPayload, RegionNodeKind, RegionTreeCore,
};

use super::structures::aabb4_from_chunk_lattice;

const CHUNK_SIZE: usize = 8;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

const MAZE_STRIDE: i32 = 4;
const MAZE_LEVEL_HEIGHT: i32 = 4;
const MAZE_GRID_XZW_CELLS_MIN: i32 = 9;
const MAZE_GRID_XZW_CELLS_MAX: i32 = 15;
const MAZE_GRID_Y_CELLS_MIN: i32 = 3;
const MAZE_GRID_Y_CELLS_MAX: i32 = 7;

const MAZE_GRID_X_SALT: u64 = 0x2c93_f17a_6540_8e11;
const MAZE_GRID_Y_SALT: u64 = 0x1720_8d59_3bf4_2c77;
const MAZE_GRID_Z_SALT: u64 = 0x85a4_5b0d_9926_f5c4;
const MAZE_GRID_W_SALT: u64 = 0xf30d_7c9e_4ab2_1688;
const MAZE_VARIANT_SALT: u64 = 0x932f_b43c_f1f9_746d;
const MAZE_LAYOUT_SALT: u64 = 0x49ec_66d6_0d13_9e75;
const MAZE_EDGE_SORT_SALT: u64 = 0xba04_7f9b_11d2_c638;
const MAZE_BRAID_SALT: u64 = 0x0d7c_88e1_a4f3_5b29;
const MAZE_EDGE_X_SALT: u64 = 0x3374_11a9_5f9c_2d01;
const MAZE_EDGE_Y_SALT: u64 = 0x40ad_6f0c_22be_8a17;
const MAZE_EDGE_Z_SALT: u64 = 0x9bc3_51d4_788a_f2ee;
const MAZE_EDGE_W_SALT: u64 = 0x1ec8_daa5_4b73_99c0;
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

fn maze_random_odd(hash: u64, min_value: i32, max_value: i32) -> i32 {
    let min_odd = if min_value & 1 == 0 { min_value + 1 } else { min_value };
    let max_odd = if max_value & 1 == 0 { max_value - 1 } else { max_value };
    let choices = ((max_odd - min_odd) / 2 + 1).max(1) as u64;
    min_odd + 2 * (splitmix64(hash) % choices) as i32
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
            Self::Catacomb => [2, 6, 2, 2],
            Self::Vertical => [3, 1, 3, 3],
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

#[derive(Copy, Clone, Debug)]
struct MazeShape {
    grid_cells: [i32; 4],
    span: [i32; 4],
    half_span_xzw: [i32; 3],
    variant: MazeVariant,
}

fn maze_shape_from_seed(seed: u64) -> MazeShape {
    let cell_hash = seed;
    let variant = MazeVariant::from_cell_hash(cell_hash);
    let grid_x = maze_random_odd(cell_hash ^ MAZE_GRID_X_SALT, MAZE_GRID_XZW_CELLS_MIN, MAZE_GRID_XZW_CELLS_MAX);
    let grid_y = maze_random_odd(cell_hash ^ MAZE_GRID_Y_SALT, MAZE_GRID_Y_CELLS_MIN, MAZE_GRID_Y_CELLS_MAX);
    let grid_z = maze_random_odd(cell_hash ^ MAZE_GRID_Z_SALT, MAZE_GRID_XZW_CELLS_MIN, MAZE_GRID_XZW_CELLS_MAX);
    let grid_w = maze_random_odd(cell_hash ^ MAZE_GRID_W_SALT, MAZE_GRID_XZW_CELLS_MIN, MAZE_GRID_XZW_CELLS_MAX);

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
        variant,
    }
}

// -- Disjoint set for Kruskal's algorithm --

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
        if self.rank[root_a] < self.rank[root_b] {
            core::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] = self.rank[root_a].saturating_add(1);
        }
        true
    }
}

// -- Topology --

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
            _ => {}
        }
    }

    fn edge_open(&self, a: [i32; 4], b: [i32; 4]) -> bool {
        if !cell_in_bounds(a, self.grid_cells) || !cell_in_bounds(b, self.grid_cells) {
            return false;
        }
        let mut changed_axis = None;
        for axis in 0..4 {
            let delta = a[axis] - b[axis];
            if delta == 0 { continue; }
            if delta.abs() != 1 || changed_axis.is_some() { return false; }
            changed_axis = Some(axis);
        }
        let Some(axis) = changed_axis else { return false; };
        let mut base = a;
        if b[axis] < a[axis] { base[axis] = b[axis]; }
        let idx = maze_edge_linear_index(self.grid_cells, axis, base);
        match axis {
            0 => self.open_x[idx],
            1 => self.open_y[idx],
            2 => self.open_z[idx],
            3 => self.open_w[idx],
            _ => false,
        }
    }
}

fn cell_in_bounds(cell: [i32; 4], grid: [i32; 4]) -> bool {
    cell[0] >= 0 && cell[0] < grid[0]
        && cell[1] >= 0 && cell[1] < grid[1]
        && cell[2] >= 0 && cell[2] < grid[2]
        && cell[3] >= 0 && cell[3] < grid[3]
}

fn maze_cell_linear_index(grid: [i32; 4], cell: [i32; 4]) -> usize {
    (((cell[0] as usize * grid[1] as usize + cell[1] as usize) * grid[2] as usize
        + cell[2] as usize) * grid[3] as usize) + cell[3] as usize
}

fn maze_edge_count(grid: [i32; 4], axis: usize) -> usize {
    match axis {
        0 => ((grid[0] - 1).max(0) * grid[1] * grid[2] * grid[3]) as usize,
        1 => (grid[0] * (grid[1] - 1).max(0) * grid[2] * grid[3]) as usize,
        2 => (grid[0] * grid[1] * (grid[2] - 1).max(0) * grid[3]) as usize,
        3 => (grid[0] * grid[1] * grid[2] * (grid[3] - 1).max(0)) as usize,
        _ => 0,
    }
}

fn maze_edge_linear_index(grid: [i32; 4], axis: usize, base: [i32; 4]) -> usize {
    match axis {
        0 => (((base[0] as usize * grid[1] as usize + base[1] as usize) * grid[2] as usize + base[2] as usize) * grid[3] as usize) + base[3] as usize,
        1 => (((base[0] as usize * (grid[1] - 1) as usize + base[1] as usize) * grid[2] as usize + base[2] as usize) * grid[3] as usize) + base[3] as usize,
        2 => (((base[0] as usize * grid[1] as usize + base[1] as usize) * (grid[2] - 1) as usize + base[2] as usize) * grid[3] as usize) + base[3] as usize,
        3 => (((base[0] as usize * grid[1] as usize + base[1] as usize) * grid[2] as usize + base[2] as usize) * (grid[3] - 1) as usize) + base[3] as usize,
        _ => 0,
    }
}

fn maze_axis_salt(axis: usize) -> u64 {
    match axis {
        0 => MAZE_EDGE_X_SALT,
        1 => MAZE_EDGE_Y_SALT,
        2 => MAZE_EDGE_Z_SALT,
        3 => MAZE_EDGE_W_SALT,
        _ => 0,
    }
}

fn maze_hash_cell(seed: u64, cell: [i32; 4], salt: u64) -> u64 {
    let mut mixed = hash_structure_cell(seed, cell[0], cell[2], cell[3], salt);
    mixed ^= (cell[1] as i64 as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    splitmix64(mixed)
}

fn maze_edge_hash(layout_seed: u64, base: [i32; 4], axis: usize, salt: u64) -> u64 {
    let mut mixed = maze_hash_cell(layout_seed, base, salt ^ maze_axis_salt(axis));
    mixed ^= (axis as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
    splitmix64(mixed)
}

#[derive(Copy, Clone)]
struct MazeEdgeCandidate {
    axis: usize,
    base: [i32; 4],
    a_idx: usize,
    b_idx: usize,
    weighted_score: u128,
    tie_break: u64,
    braid_roll: u64,
}

fn build_topology(layout_seed: u64, shape: MazeShape) -> MazeTopology {
    let grid = shape.grid_cells;
    let total_cells = (grid[0] as usize) * (grid[1] as usize) * (grid[2] as usize) * (grid[3] as usize);
    let mut topology = MazeTopology::new(grid);
    let mut disjoint_set = DisjointSet::new(total_cells);
    let mut rejected_edges = Vec::new();

    let axis_weights = shape.variant.axis_weights();
    let mut edges = Vec::new();
    for x in 0..grid[0] {
        for y in 0..grid[1] {
            for z in 0..grid[2] {
                for w in 0..grid[3] {
                    let base = [x, y, z, w];
                    let a_idx = maze_cell_linear_index(grid, base);
                    for axis in 0..4 {
                        if base[axis] + 1 >= grid[axis] { continue; }
                        let mut neighbor = base;
                        neighbor[axis] += 1;
                        let b_idx = maze_cell_linear_index(grid, neighbor);
                        let sort_roll = maze_edge_hash(layout_seed, base, axis, MAZE_EDGE_SORT_SALT);
                        let tie_break = maze_edge_hash(layout_seed, base, axis, MAZE_EDGE_SORT_SALT ^ 0xd6e8_feb8_6659_fd93);
                        let braid_roll = maze_edge_hash(layout_seed, base, axis, MAZE_BRAID_SALT);
                        edges.push(MazeEdgeCandidate {
                            axis, base, a_idx, b_idx,
                            weighted_score: (sort_roll as u128) * axis_weights[axis] as u128,
                            tie_break, braid_roll,
                        });
                    }
                }
            }
        }
    }
    edges.sort_unstable_by(|a, b| a.weighted_score.cmp(&b.weighted_score).then_with(|| a.tie_break.cmp(&b.tie_break)));

    for edge in edges {
        if disjoint_set.union(edge.a_idx, edge.b_idx) {
            topology.set_edge_open(edge.axis, edge.base);
        } else {
            rejected_edges.push(edge);
        }
    }

    let (braid_num, braid_den) = shape.variant.braid_chance();
    if braid_num > 0 {
        for edge in rejected_edges {
            if splitmix64(edge.braid_roll) % braid_den < braid_num {
                topology.set_edge_open(edge.axis, edge.base);
            }
        }
    }

    topology
}

fn maze_gate_cell(layout_seed: u64, salt: u64, cell_count: i32) -> i32 {
    (splitmix64(layout_seed ^ salt) % cell_count.max(1) as u64) as i32
}

fn maze_gate_band(cell_idx: i32, stride: i32) -> (i32, i32) {
    let start = cell_idx * stride + 1;
    (start, start + stride.saturating_sub(2))
}

fn maze_gate_open(
    u0: i32, u1: i32, u2: i32,
    gate0: i32, gate1: i32, gate2: i32,
    stride0: i32, stride1: i32, stride2: i32,
) -> bool {
    let (g0_min, g0_max) = maze_gate_band(gate0, stride0);
    let (g1_min, g1_max) = maze_gate_band(gate1, stride1);
    let (g2_min, g2_max) = maze_gate_band(gate2, stride2);
    u0 >= g0_min && u0 <= g0_max && u1 >= g1_min && u1 <= g1_max && u2 >= g2_min && u2 <= g2_max
}

fn maze_layout_seed(seed: u64, origin: [i32; 4], shape: MazeShape) -> u64 {
    let mut s = hash_structure_cell(seed, origin[0], origin[2], origin[3], MAZE_LAYOUT_SALT);
    s ^= (origin[1] as i64 as u64).wrapping_mul(0xa24b_aed4_963e_e407);
    s ^= (shape.grid_cells[0] as i64 as u64).wrapping_mul(0x9e37_79b1_85eb_ca87);
    s ^= (shape.grid_cells[1] as i64 as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    s ^= (shape.grid_cells[2] as i64 as u64).wrapping_mul(0x1656_67b1_9e37_79f9);
    s ^= (shape.grid_cells[3] as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    s ^= shape.variant.seed_tag();
    splitmix64(s)
}

struct CompiledLayout {
    topology: MazeTopology,
    x_neg_gate: [i32; 3],
    x_pos_gate: [i32; 3],
    z_neg_gate: [i32; 3],
    z_pos_gate: [i32; 3],
    w_neg_gate: [i32; 3],
    w_pos_gate: [i32; 3],
    center_u: [i32; 4],
}

fn compile_layout(layout_seed: u64, shape: MazeShape) -> CompiledLayout {
    CompiledLayout {
        topology: build_topology(layout_seed, shape),
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

// -- State blob serialization --
// The state blob stores the maze topology as a compact binary format:
// [4 x i32 grid_cells] [4 x i32 span] [3 x i32 half_span_xzw] [variant u8]
// [u64 layout_seed] [topology open_x..open_w as packed bytes]

fn serialize_state(layout_seed: u64, shape: MazeShape, layout: &CompiledLayout) -> Vec<u8> {
    let mut buf = Vec::new();
    for &g in &shape.grid_cells { buf.extend_from_slice(&g.to_le_bytes()); }
    for &s in &shape.span { buf.extend_from_slice(&s.to_le_bytes()); }
    for &h in &shape.half_span_xzw { buf.extend_from_slice(&h.to_le_bytes()); }
    buf.push(match shape.variant {
        MazeVariant::Catacomb => 0,
        MazeVariant::Vertical => 1,
        MazeVariant::Braided => 2,
    });
    buf.extend_from_slice(&layout_seed.to_le_bytes());
    // Pack topology booleans
    fn pack_bools(bools: &[bool], buf: &mut Vec<u8>) {
        buf.extend_from_slice(&(bools.len() as u32).to_le_bytes());
        for chunk in bools.chunks(8) {
            let mut byte = 0u8;
            for (i, &b) in chunk.iter().enumerate() {
                if b { byte |= 1 << i; }
            }
            buf.push(byte);
        }
    }
    pack_bools(&layout.topology.open_x, &mut buf);
    pack_bools(&layout.topology.open_y, &mut buf);
    pack_bools(&layout.topology.open_z, &mut buf);
    pack_bools(&layout.topology.open_w, &mut buf);
    // Pack gate data
    for gate in [&layout.x_neg_gate, &layout.x_pos_gate, &layout.z_neg_gate, &layout.z_pos_gate, &layout.w_neg_gate, &layout.w_pos_gate] {
        for &g in gate { buf.extend_from_slice(&g.to_le_bytes()); }
    }
    for &c in &layout.center_u { buf.extend_from_slice(&c.to_le_bytes()); }
    buf
}

struct DeserializedState {
    shape: MazeShape,
    layout: CompiledLayout,
}

fn deserialize_state(data: &[u8]) -> DeserializedState {
    let mut pos = 0usize;
    let read_i32 = |pos: &mut usize| -> i32 {
        let v = i32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]);
        *pos += 4;
        v
    };
    let read_u64 = |pos: &mut usize| -> u64 {
        let v = u64::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3], data[*pos+4], data[*pos+5], data[*pos+6], data[*pos+7]]);
        *pos += 8;
        v
    };

    let grid_cells = [read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos)];
    let span = [read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos)];
    let half_span_xzw = [read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos)];
    let variant = match data[pos] {
        0 => MazeVariant::Catacomb,
        1 => MazeVariant::Vertical,
        _ => MazeVariant::Braided,
    };
    pos += 1;
    let _layout_seed = read_u64(&mut pos);

    let shape = MazeShape { grid_cells, span, half_span_xzw, variant };

    fn unpack_bools(data: &[u8], pos: &mut usize) -> Vec<bool> {
        let len = u32::from_le_bytes([data[*pos], data[*pos+1], data[*pos+2], data[*pos+3]]) as usize;
        *pos += 4;
        let byte_count = (len + 7) / 8;
        let mut bools = Vec::with_capacity(len);
        for i in 0..len {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bools.push((data[*pos + byte_idx] >> bit_idx) & 1 != 0);
        }
        *pos += byte_count;
        bools
    }

    let open_x = unpack_bools(data, &mut pos);
    let open_y = unpack_bools(data, &mut pos);
    let open_z = unpack_bools(data, &mut pos);
    let open_w = unpack_bools(data, &mut pos);

    let topology = MazeTopology { grid_cells, open_x, open_y, open_z, open_w };

    let read_gate = |pos: &mut usize| -> [i32; 3] {
        [read_i32(pos), read_i32(pos), read_i32(pos)]
    };
    let x_neg_gate = read_gate(&mut pos);
    let x_pos_gate = read_gate(&mut pos);
    let z_neg_gate = read_gate(&mut pos);
    let z_pos_gate = read_gate(&mut pos);
    let w_neg_gate = read_gate(&mut pos);
    let w_pos_gate = read_gate(&mut pos);
    let center_u = [read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos), read_i32(&mut pos)];

    let layout = CompiledLayout {
        topology,
        x_neg_gate, x_pos_gate, z_neg_gate, z_pos_gate, w_neg_gate, w_pos_gate,
        center_u,
    };

    DeserializedState { shape, layout }
}

// -- Rasterization --

fn rasterize_maze(
    origin: [i32; 4],
    shape: MazeShape,
    layout: &CompiledLayout,
    block_palette: &[BlockData],
    floor_idx: u16,
    ceiling_idx: u16,
    wall_idx: u16,
    gate_frame_idx: u16,
    beacon_idx: u16,
) -> RegionTreeCore {
    let maze_min = [
        origin[0] - shape.half_span_xzw[0],
        origin[1],
        origin[2] - shape.half_span_xzw[1],
        origin[3] - shape.half_span_xzw[2],
    ];
    let maze_max = [
        origin[0] + shape.half_span_xzw[0],
        origin[1] + shape.span[1] - 1,
        origin[2] + shape.half_span_xzw[1],
        origin[3] + shape.half_span_xzw[2],
    ];

    let cs = CHUNK_SIZE as i32;
    let chunk_min = [
        maze_min[0].div_euclid(cs),
        maze_min[1].div_euclid(cs),
        maze_min[2].div_euclid(cs),
        maze_min[3].div_euclid(cs),
    ];
    let chunk_max = [
        maze_max[0].div_euclid(cs),
        maze_max[1].div_euclid(cs),
        maze_max[2].div_euclid(cs),
        maze_max[3].div_euclid(cs),
    ];

    let bounds = aabb4_from_chunk_lattice(chunk_min, chunk_max);
    let dims = [
        (chunk_max[0] - chunk_min[0] + 1) as usize,
        (chunk_max[1] - chunk_min[1] + 1) as usize,
        (chunk_max[2] - chunk_min[2] + 1) as usize,
        (chunk_max[3] - chunk_min[3] + 1) as usize,
    ];

    let total_chunks = dims[0] * dims[1] * dims[2] * dims[3];
    let mut chunk_data: Vec<Vec<u16>> = (0..total_chunks).map(|_| vec![0u16; CHUNK_VOLUME]).collect();

    let topology = &layout.topology;
    let center_u = layout.center_u;

    for wx in maze_min[0]..=maze_max[0] {
        for wy in maze_min[1]..=maze_max[1] {
            for wz in maze_min[2]..=maze_max[2] {
                for ww in maze_min[3]..=maze_max[3] {
                    let ux = wx - maze_min[0];
                    let uy = wy - maze_min[1];
                    let uz = wz - maze_min[2];
                    let uw = ww - maze_min[3];

                    let material: Option<u16> = if uy == 0 {
                        Some(floor_idx)
                    } else if uy == shape.span[1] - 1 {
                        Some(ceiling_idx)
                    } else {
                        classify_interior_voxel(
                            ux, uy, uz, uw, shape, topology, layout, center_u,
                            wall_idx, gate_frame_idx, beacon_idx,
                        )
                    };

                    let Some(mat) = material else { continue; };
                    let cx = (wx.div_euclid(cs) - chunk_min[0]) as usize;
                    let cy = (wy.div_euclid(cs) - chunk_min[1]) as usize;
                    let cz = (wz.div_euclid(cs) - chunk_min[2]) as usize;
                    let cw = (ww.div_euclid(cs) - chunk_min[3]) as usize;
                    let chunk_idx = cx + dims[0] * (cy + dims[1] * (cz + dims[2] * cw));
                    let lx = wx.rem_euclid(cs) as usize;
                    let ly = wy.rem_euclid(cs) as usize;
                    let lz = wz.rem_euclid(cs) as usize;
                    let lw = ww.rem_euclid(cs) as usize;
                    let voxel_idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                        + lz * CHUNK_SIZE * CHUNK_SIZE
                        + ly * CHUNK_SIZE + lx;
                    chunk_data[chunk_idx][voxel_idx] = mat;
                }
            }
        }
    }

    // Build ChunkArrayData — deduplicate chunks by linear scan to avoid
    // cloning full 8KB chunk vectors into a BTreeMap.
    let mut palette: Vec<ChunkPayload> = vec![ChunkPayload::Empty];
    let mut dense_indices = Vec::with_capacity(total_chunks);
    let empty_chunk = vec![0u16; CHUNK_VOLUME];
    for chunk in chunk_data.drain(..) {
        if chunk == empty_chunk {
            dense_indices.push(0u16);
            continue;
        }
        // Linear scan for deduplication (few unique chunks in practice)
        let mut found = false;
        for (idx, existing) in palette.iter().enumerate() {
            if let ChunkPayload::Dense16 { materials } = existing {
                if materials == &chunk {
                    dense_indices.push(idx as u16);
                    found = true;
                    break;
                }
            }
        }
        if !found {
            let idx = palette.len() as u16;
            palette.push(ChunkPayload::Dense16 { materials: chunk });
            dense_indices.push(idx);
        }
    }

    if dense_indices.iter().all(|&idx| idx == 0) {
        return RegionTreeCore { bounds, kind: RegionNodeKind::Empty, generator_version_hash: 0 };
    }

    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::ChunkArray(ChunkArrayData {
            bounds,
            scale_exp: 0,
            chunk_palette: palette,
            dense_indices,
            block_palette: block_palette.to_vec(),
        }),
        generator_version_hash: 0,
    }
}

fn classify_interior_voxel(
    ux: i32, uy: i32, uz: i32, uw: i32,
    shape: MazeShape,
    topology: &MazeTopology,
    layout: &CompiledLayout,
    center_u: [i32; 4],
    wall_idx: u16, gate_frame_idx: u16, beacon_idx: u16,
) -> Option<u16> {
    let on_x_neg = ux == 0;
    let on_x_pos = ux == shape.span[0] - 1;
    let on_z_neg = uz == 0;
    let on_z_pos = uz == shape.span[2] - 1;
    let on_w_neg = uw == 0;
    let on_w_pos = uw == shape.span[3] - 1;

    if on_x_neg || on_x_pos || on_z_neg || on_z_pos || on_w_neg || on_w_pos {
        let gate_open = if on_x_neg {
            maze_gate_open(uy, uz, uw, layout.x_neg_gate[0], layout.x_neg_gate[1], layout.x_neg_gate[2], MAZE_LEVEL_HEIGHT, MAZE_STRIDE, MAZE_STRIDE)
        } else if on_x_pos {
            maze_gate_open(uy, uz, uw, layout.x_pos_gate[0], layout.x_pos_gate[1], layout.x_pos_gate[2], MAZE_LEVEL_HEIGHT, MAZE_STRIDE, MAZE_STRIDE)
        } else if on_z_neg {
            maze_gate_open(ux, uy, uw, layout.z_neg_gate[0], layout.z_neg_gate[1], layout.z_neg_gate[2], MAZE_STRIDE, MAZE_LEVEL_HEIGHT, MAZE_STRIDE)
        } else if on_z_pos {
            maze_gate_open(ux, uy, uw, layout.z_pos_gate[0], layout.z_pos_gate[1], layout.z_pos_gate[2], MAZE_STRIDE, MAZE_LEVEL_HEIGHT, MAZE_STRIDE)
        } else if on_w_neg {
            maze_gate_open(ux, uy, uz, layout.w_neg_gate[0], layout.w_neg_gate[1], layout.w_neg_gate[2], MAZE_STRIDE, MAZE_LEVEL_HEIGHT, MAZE_STRIDE)
        } else {
            maze_gate_open(ux, uy, uz, layout.w_pos_gate[0], layout.w_pos_gate[1], layout.w_pos_gate[2], MAZE_STRIDE, MAZE_LEVEL_HEIGHT, MAZE_STRIDE)
        };
        if gate_open { None } else { Some(gate_frame_idx) }
    } else {
        let wall_x = ux % MAZE_STRIDE == 0;
        let wall_y = uy % MAZE_LEVEL_HEIGHT == 0;
        let wall_z = uz % MAZE_STRIDE == 0;
        let wall_w = uw % MAZE_STRIDE == 0;
        let wall_count = wall_x as i32 + wall_y as i32 + wall_z as i32 + wall_w as i32;

        if wall_count == 0 {
            if ux == center_u[0] && uy == center_u[1] && uz == center_u[2] && uw == center_u[3] {
                Some(beacon_idx)
            } else {
                None
            }
        } else if wall_count > 1 {
            Some(wall_idx)
        } else if wall_x {
            let left = ux / MAZE_STRIDE - 1;
            let right = left + 1;
            let cy = uy / MAZE_LEVEL_HEIGHT;
            let cz = uz / MAZE_STRIDE;
            let cw = uw / MAZE_STRIDE;
            if topology.edge_open([left, cy, cz, cw], [right, cy, cz, cw]) { None } else { Some(wall_idx) }
        } else if wall_y {
            let lower = uy / MAZE_LEVEL_HEIGHT - 1;
            let upper = lower + 1;
            let cx = ux / MAZE_STRIDE;
            let cz = uz / MAZE_STRIDE;
            let cw = uw / MAZE_STRIDE;
            if topology.edge_open([cx, lower, cz, cw], [cx, upper, cz, cw]) { None } else { Some(wall_idx) }
        } else if wall_z {
            let near = uz / MAZE_STRIDE - 1;
            let far = near + 1;
            let cx = ux / MAZE_STRIDE;
            let cy = uy / MAZE_LEVEL_HEIGHT;
            let cw = uw / MAZE_STRIDE;
            if topology.edge_open([cx, cy, near, cw], [cx, cy, far, cw]) { None } else { Some(wall_idx) }
        } else {
            let near = uw / MAZE_STRIDE - 1;
            let far = near + 1;
            let cx = ux / MAZE_STRIDE;
            let cy = uy / MAZE_LEVEL_HEIGHT;
            let cz = uz / MAZE_STRIDE;
            if topology.edge_open([cx, cy, cz, near], [cx, cy, cz, far]) { None } else { Some(wall_idx) }
        }
    }
}

// -- Public API --

pub struct MazeGenerator {
    block_palette: Vec<BlockData>,
    floor_idx: u16,
    ceiling_idx: u16,
    wall_idx: u16,
    gate_frame_idx: u16,
    beacon_idx: u16,
}

fn intern_block(palette: &mut Vec<BlockData>, block: &BlockData) -> u16 {
    if block.is_air() { return 0; }
    if let Some(idx) = palette.iter().position(|b| b == block) { return idx as u16; }
    let idx = palette.len() as u16;
    palette.push(block.clone());
    idx
}

impl MazeGenerator {
    pub fn new() -> Self {
        let mut palette = vec![BlockData::AIR];
        let floor_idx = intern_block(&mut palette, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_BASALT_TILES));
        let ceiling_idx = intern_block(&mut palette, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_SMOKED_GLASS));
        let wall_idx = intern_block(&mut palette, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_RUNIC_ALLOY));
        let gate_frame_idx = intern_block(&mut palette, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_OBSIDIAN));
        let beacon_idx = intern_block(&mut palette, &BlockData::simple(content_ids::CONTENT_NS, content_ids::BLOCK_EVENTIDE_ALLOY));
        Self { block_palette: palette, floor_idx, ceiling_idx, wall_idx, gate_frame_idx, beacon_idx }
    }

    pub fn prepare(&self, input: &ProcgenPrepareInput) -> ProcgenPrepareOutput {
        let shape = maze_shape_from_seed(input.seed);
        let origin = input.origin;
        let layout_seed = maze_layout_seed(input.seed, origin, shape);
        let layout = compile_layout(layout_seed, shape);

        let maze_min = [
            origin[0] - shape.half_span_xzw[0],
            origin[1],
            origin[2] - shape.half_span_xzw[1],
            origin[3] - shape.half_span_xzw[2],
        ];
        let maze_max = [
            origin[0] + shape.half_span_xzw[0],
            origin[1] + shape.span[1] - 1,
            origin[2] + shape.half_span_xzw[1],
            origin[3] + shape.half_span_xzw[2],
        ];
        let cs = CHUNK_SIZE as i32;
        let bounds = aabb4_from_chunk_lattice(
            [maze_min[0].div_euclid(cs), maze_min[1].div_euclid(cs), maze_min[2].div_euclid(cs), maze_min[3].div_euclid(cs)],
            [maze_max[0].div_euclid(cs), maze_max[1].div_euclid(cs), maze_max[2].div_euclid(cs), maze_max[3].div_euclid(cs)],
        );

        let state = serialize_state(layout_seed, shape, &layout);
        ProcgenPrepareOutput { bounds, state }
    }

    pub fn generate(&self, input: &ProcgenGenerateInput) -> ProcgenGenerateOutput {
        let ds = deserialize_state(&input.state);
        let tree = rasterize_maze(
            input.origin,
            ds.shape,
            &ds.layout,
            &self.block_palette,
            self.floor_idx,
            self.ceiling_idx,
            self.wall_idx,
            self.gate_frame_idx,
            self.beacon_idx,
        );
        ProcgenGenerateOutput { tree }
    }
}
