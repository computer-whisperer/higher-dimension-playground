use fixed::types::I48F16;
use serde::{Deserialize, Serialize};

/// Fixed-point coordinate type for chunk positions: 48 integer bits, 16 fractional bits.
pub type ChunkCoord = I48F16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Aabb4 {
    pub min: [ChunkCoord; 4],
    pub max: [ChunkCoord; 4],
}

/// Temporary alias for incremental migration.
pub type Aabb4i = Aabb4;

impl Aabb4 {
    pub fn new(min: [ChunkCoord; 4], max: [ChunkCoord; 4]) -> Self {
        Self { min, max }
    }

    pub fn from_i32(min: [i32; 4], max: [i32; 4]) -> Self {
        Self {
            min: min.map(ChunkCoord::from_num),
            max: max.map(ChunkCoord::from_num),
        }
    }

    pub fn from_lattice_bounds(min: [i32; 4], max: [i32; 4], scale_exp: i8) -> Self {
        Self {
            min: min.map(|v| fixed_from_lattice(v, scale_exp)),
            max: max.map(|v| fixed_from_lattice(v, scale_exp)),
        }
    }

    pub fn from_fixed_bits(min: [i64; 4], max: [i64; 4]) -> Self {
        Self {
            min: min.map(ChunkCoord::from_bits),
            max: max.map(ChunkCoord::from_bits),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0]
            && self.min[1] <= self.max[1]
            && self.min[2] <= self.max[2]
            && self.min[3] <= self.max[3]
    }

    pub fn chunk_extents_at_scale(&self, scale_exp: i8) -> Option<[usize; 4]> {
        if !self.is_valid() {
            return None;
        }
        let mut extents = [0usize; 4];
        for axis in 0..4 {
            let lattice_min = lattice_from_fixed(self.min[axis], scale_exp);
            let lattice_max = lattice_from_fixed(self.max[axis], scale_exp);
            let span = i64::from(lattice_max) - i64::from(lattice_min) + 1;
            if span <= 0 {
                return None;
            }
            extents[axis] = usize::try_from(span).ok()?;
        }
        Some(extents)
    }

    pub fn chunk_cell_count_at_scale(&self, scale_exp: i8) -> Option<usize> {
        let extents = self.chunk_extents_at_scale(scale_exp)?;
        extents[0]
            .checked_mul(extents[1])?
            .checked_mul(extents[2])?
            .checked_mul(extents[3])
    }

    pub fn contains_chunk(&self, pos: [ChunkCoord; 4]) -> bool {
        self.is_valid()
            && pos[0] >= self.min[0]
            && pos[0] <= self.max[0]
            && pos[1] >= self.min[1]
            && pos[1] <= self.max[1]
            && pos[2] >= self.min[2]
            && pos[2] <= self.max[2]
            && pos[3] >= self.min[3]
            && pos[3] <= self.max[3]
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.is_valid()
            && other.is_valid()
            && self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
            && self.min[3] <= other.max[3]
            && self.max[3] >= other.min[3]
    }

    /// Convert to lattice-relative integer bounds at the given scale.
    pub fn to_lattice_bounds(&self, scale_exp: i8) -> ([i32; 4], [i32; 4]) {
        let min = [
            lattice_from_fixed(self.min[0], scale_exp),
            lattice_from_fixed(self.min[1], scale_exp),
            lattice_from_fixed(self.min[2], scale_exp),
            lattice_from_fixed(self.min[3], scale_exp),
        ];
        let max = [
            lattice_from_fixed(self.max[0], scale_exp),
            lattice_from_fixed(self.max[1], scale_exp),
            lattice_from_fixed(self.max[2], scale_exp),
            lattice_from_fixed(self.max[3], scale_exp),
        ];
        (min, max)
    }
}

/// Convert a lattice integer position to fixed-point at the given scale.
///
/// `fixed = lattice_pos * 2^scale_exp`
/// e.g. lattice 1 at scale -1 → 0.5
pub fn fixed_from_lattice(lattice_pos: i32, scale_exp: i8) -> ChunkCoord {
    let shift = 16i32 + scale_exp as i32;
    if shift < 0 {
        ChunkCoord::ZERO
    } else {
        ChunkCoord::from_bits((lattice_pos as i64) << shift)
    }
}

/// Convert a fixed-point position to a lattice integer at the given scale.
///
/// Inverse of `fixed_from_lattice`: `lattice = fixed / 2^scale_exp`
pub fn lattice_from_fixed(fixed_pos: ChunkCoord, scale_exp: i8) -> i32 {
    let shift = 16i32 + scale_exp as i32;
    if shift < 0 {
        0
    } else {
        (fixed_pos.to_bits() >> shift) as i32
    }
}

/// The grid step for a given scale exponent, as a fixed-point value.
///
/// `step = 2^scale_exp`
pub fn step_for_scale(scale_exp: i8) -> ChunkCoord {
    let shift = 16i32 + scale_exp as i32;
    if shift < 0 || shift > 62 {
        ChunkCoord::ZERO
    } else {
        ChunkCoord::from_bits(1i64 << shift)
    }
}

/// Build a ChunkKey from lattice integers at the given scale.
pub fn chunk_key_from_lattice(lattice: [i32; 4], scale_exp: i8) -> [ChunkCoord; 4] {
    [
        fixed_from_lattice(lattice[0], scale_exp),
        fixed_from_lattice(lattice[1], scale_exp),
        fixed_from_lattice(lattice[2], scale_exp),
        fixed_from_lattice(lattice[3], scale_exp),
    ]
}
