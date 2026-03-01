use crate::shared::region_tree::ChunkKey;
use crate::shared::spatial::fixed_from_lattice;
use serde::{Deserialize, Serialize};

pub const CHUNK_SIZE: usize = 8;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

// 4! = 24 permutations of [X, Y, Z, W] axes.
// Each row is a permutation: output[i] = input[PERMUTATIONS[perm][i]]
const PERMUTATIONS: [[u8; 4]; 24] = [
    [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1],
    [0, 3, 1, 2], [0, 3, 2, 1], [1, 0, 2, 3], [1, 0, 3, 2],
    [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
    [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0],
    [2, 3, 0, 1], [2, 3, 1, 0], [3, 0, 1, 2], [3, 0, 2, 1],
    [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
];

/// A tesseract (4D hypercube) orientation: one of 384 discrete rotations/reflections.
///
/// Encoding: `permutation_index * 16 + sign_bits`
/// - `permutation_index`: 0..23 (4! permutations of axes [X,Y,Z,W])
/// - `sign_bits`: 0..15 (bitmask of which axes are negated)
///
/// 24 permutations x 16 sign combos = 384 valid orientations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TesseractOrientation(pub u16);

impl TesseractOrientation {
    pub const IDENTITY: Self = Self(0);

    pub fn identity() -> Self {
        Self::IDENTITY
    }

    pub fn is_valid(self) -> bool {
        self.0 < 384
    }

    pub fn permutation_index(self) -> u8 {
        (self.0 / 16) as u8
    }

    pub fn sign_bits(self) -> u8 {
        (self.0 % 16) as u8
    }

    pub fn from_parts(perm: u8, signs: u8) -> Self {
        Self(perm as u16 * 16 + (signs & 0xF) as u16)
    }

    /// Apply this orientation to a 4D offset.
    pub fn apply(self, offset: [i32; 4]) -> [i32; 4] {
        let perm = &PERMUTATIONS[self.permutation_index() as usize];
        let signs = self.sign_bits();
        let mut result = [0i32; 4];
        for i in 0..4 {
            let val = offset[perm[i] as usize];
            result[i] = if signs & (1 << i) != 0 { -val } else { val };
        }
        result
    }

    /// Compose two orientations: `self` applied after `other`.
    pub fn compose(self, other: Self) -> Self {
        let a_perm = &PERMUTATIONS[self.permutation_index() as usize];
        let a_signs = self.sign_bits();
        let b_perm = &PERMUTATIONS[other.permutation_index() as usize];
        let b_signs = other.sign_bits();

        // Composed permutation: c_perm[i] = b_perm[a_perm[i]]
        let mut c_perm = [0u8; 4];
        let mut c_signs = 0u8;
        for i in 0..4 {
            c_perm[i] = b_perm[a_perm[i] as usize];
            // Sign: a flips output axis i, b flips the axis a_perm[i] selects
            let a_flip = (a_signs >> i) & 1;
            let b_flip = (b_signs >> a_perm[i]) & 1;
            c_signs |= (a_flip ^ b_flip) << i;
        }

        // Find the permutation index for c_perm
        let perm_idx = PERMUTATIONS
            .iter()
            .position(|p| *p == c_perm)
            .unwrap_or(0) as u8;
        Self::from_parts(perm_idx, c_signs)
    }

    /// Compute the inverse orientation.
    pub fn inverse(self) -> Self {
        let perm = &PERMUTATIONS[self.permutation_index() as usize];
        let signs = self.sign_bits();

        // Inverse permutation: inv_perm[perm[i]] = i
        let mut inv_perm = [0u8; 4];
        let mut inv_signs = 0u8;
        for i in 0..4 {
            inv_perm[perm[i] as usize] = i as u8;
        }
        // Inverse sign: the sign that undoes the original flip
        for i in 0..4 {
            let original_axis = inv_perm[i] as usize;
            if signs & (1 << original_axis) != 0 {
                inv_signs |= 1 << i;
            }
        }

        let perm_idx = PERMUTATIONS
            .iter()
            .position(|p| *p == inv_perm)
            .unwrap_or(0) as u8;
        Self::from_parts(perm_idx, inv_signs)
    }
}

/// Rich block data: namespace + type + orientation + optional extra payload.
///
/// Stored in `ChunkArrayData::block_palette` and `RegionNodeKind::Uniform`.
/// The u16 indices inside `ChunkPayload` variants reference entries in the block palette.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockData {
    pub namespace: u32,
    pub block_type: u32,
    #[serde(default = "TesseractOrientation::identity")]
    pub orientation: TesseractOrientation,
    #[serde(default)]
    pub extra_data: Vec<u8>,
    #[serde(default)]
    pub scale_exp: i8,
}

impl BlockData {
    pub const AIR: Self = Self {
        namespace: 0,
        block_type: 0,
        orientation: TesseractOrientation::IDENTITY,
        extra_data: Vec::new(),
        scale_exp: 0,
    };

    pub fn simple(namespace: u32, block_type: u32) -> Self {
        Self {
            namespace,
            block_type,
            orientation: TesseractOrientation::IDENTITY,
            extra_data: Vec::new(),
            scale_exp: 0,
        }
    }

    pub fn at_scale(mut self, scale_exp: i8) -> Self {
        self.scale_exp = scale_exp;
        self
    }

    pub fn is_air(&self) -> bool {
        self.namespace == 0 && self.block_type == 0
    }

    /// Compare two blocks ignoring `scale_exp`.
    ///
    /// Useful for pruning override chunks against virgin data: blocks may
    /// carry different `scale_exp` values (e.g. a scale-2 player edit vs
    /// scale-0 worldgen) while being semantically identical for the purpose
    /// of deciding whether an override can be removed.
    pub fn matches_ignoring_scale(&self, other: &Self) -> bool {
        self.namespace == other.namespace
            && self.block_type == other.block_type
            && self.orientation == other.orientation
            && self.extra_data == other.extra_data
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseWorldKind {
    Empty,
    FlatFloor { material: BlockData },
    MassivePlatforms { material: BlockData },
}

/// Compute a linear index from 4D coordinates and dimension extents.
///
/// Layout: `x + dims[0] * (y + dims[1] * (z + dims[2] * w))`
#[inline]
pub fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

/// Convert world coordinates to chunk key and local voxel index at a given scale.
///
/// - `scale_exp = 0`: cell_size = 1, chunk spans 8 world units (default behavior)
/// - `scale_exp = -1`: cell_size = 0.5, chunk spans 4 world units (half-scale)
/// - `scale_exp = 1`: cell_size = 2, chunk spans 16 world units (double-scale)
///
/// The returned ChunkKey is in fixed-point coordinates.
#[inline]
pub fn world_to_chunk_at_scale(
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
    scale_exp: i8,
) -> (ChunkKey, usize) {
    let cs = CHUNK_SIZE as i64;

    // Convert world coords to scaled-lattice coords using i64 to avoid overflow.
    let (sx, sy, sz, sw) = if scale_exp >= 0 {
        let shift = scale_exp as u32;
        (
            (wx as i64).div_euclid(1i64 << shift),
            (wy as i64).div_euclid(1i64 << shift),
            (wz as i64).div_euclid(1i64 << shift),
            (ww as i64).div_euclid(1i64 << shift),
        )
    } else {
        let shift = (-scale_exp) as u32;
        (
            (wx as i64) << shift,
            (wy as i64) << shift,
            (wz as i64) << shift,
            (ww as i64) << shift,
        )
    };

    let chunk_lattice = [
        sx.div_euclid(cs) as i32,
        sy.div_euclid(cs) as i32,
        sz.div_euclid(cs) as i32,
        sw.div_euclid(cs) as i32,
    ];
    let key = [
        fixed_from_lattice(chunk_lattice[0], scale_exp),
        fixed_from_lattice(chunk_lattice[1], scale_exp),
        fixed_from_lattice(chunk_lattice[2], scale_exp),
        fixed_from_lattice(chunk_lattice[3], scale_exp),
    ];
    let lx = sx.rem_euclid(cs) as usize;
    let ly = sy.rem_euclid(cs) as usize;
    let lz = sz.rem_euclid(cs) as usize;
    let lw = sw.rem_euclid(cs) as usize;
    let idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
        + lz * CHUNK_SIZE * CHUNK_SIZE
        + ly * CHUNK_SIZE
        + lx;
    (key, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::region_tree::chunk_key_i32;
    use crate::shared::spatial::{fixed_from_lattice, ChunkCoord};

    #[test]
    fn world_to_chunk_at_scale_negative_one() {
        // scale_exp=-1: cell_size=0.5, chunk spans 4 world units
        let (cp, _idx) = world_to_chunk_at_scale(0, 0, 0, 0, -1);
        assert_eq!(cp, chunk_key_i32(0, 0, 0, 0));

        // World coord 3 → scaled coord 6 → chunk 0, local 6
        let (cp, idx) = world_to_chunk_at_scale(3, 0, 0, 0, -1);
        assert_eq!(cp[0], ChunkCoord::from_num(0));
        assert_eq!(idx, 6);

        // World coord 4 → scaled coord 8 → chunk 1, local 0
        // At scale -1, lattice pos 1 → fixed 0.5
        let (cp, idx) = world_to_chunk_at_scale(4, 0, 0, 0, -1);
        assert_eq!(cp[0], ChunkCoord::from_bits(1i64 << 15)); // 0.5
        assert_eq!(idx, 0);

        // Negative: world coord -1 → scaled coord -2 → chunk -1, local 6
        let (cp, idx) = world_to_chunk_at_scale(-1, 0, 0, 0, -1);
        assert_eq!(cp[0], fixed_from_lattice(-1, -1));
        assert_eq!(idx, 6);
    }

    #[test]
    fn world_to_chunk_at_scale_positive_one() {
        // scale_exp=1: cell_size=2, chunk spans 16 world units
        let (cp, _) = world_to_chunk_at_scale(0, 0, 0, 0, 1);
        assert_eq!(cp, chunk_key_i32(0, 0, 0, 0));

        // World coord 15 → scaled coord 7 → chunk 0, local 7
        let (cp, idx) = world_to_chunk_at_scale(15, 0, 0, 0, 1);
        assert_eq!(cp[0], ChunkCoord::from_num(0));
        assert_eq!(idx, 7);

        // World coord 16 → scaled coord 8 → chunk 1, local 0
        // At scale 1, lattice pos 1 → fixed 2
        let (cp, idx) = world_to_chunk_at_scale(16, 0, 0, 0, 1);
        assert_eq!(cp[0], ChunkCoord::from_num(2));
        assert_eq!(idx, 0);
    }
}
