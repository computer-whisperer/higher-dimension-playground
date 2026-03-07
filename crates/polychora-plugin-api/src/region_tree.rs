use alloc::vec::Vec;
use fixed::types::I48F16;
use serde::{Deserialize, Serialize};

/// Fixed-point coordinate type for chunk positions: 48 integer bits, 16 fractional bits.
pub type ChunkCoord = I48F16;

/// Half-open 4D axis-aligned bounding box in world-space coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Aabb4 {
    pub min: [ChunkCoord; 4],
    pub max: [ChunkCoord; 4],
}

/// A tesseract orientation: one of 384 discrete rotations/reflections.
///
/// Encoding: `permutation_index * 16 + sign_bits`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TesseractOrientation(pub u16);

impl TesseractOrientation {
    pub const IDENTITY: Self = Self(0);

    pub fn is_valid(self) -> bool {
        self.0 < 384
    }

    pub fn permutation_index(self) -> u8 {
        (self.0 / 16) as u8
    }

    pub fn sign_bits(self) -> u8 {
        (self.0 % 16) as u8
    }
}

/// Rich block data: namespace + type + orientation + optional extra payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockData {
    pub namespace: u32,
    pub block_type: u32,
    #[serde(default = "TesseractOrientation::default")]
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

    /// Sentinel value meaning "no override — leave existing world data".
    pub const VIRGIN: Self = Self {
        namespace: u32::MAX,
        block_type: u32::MAX,
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

    pub fn is_air(&self) -> bool {
        self.namespace == 0 && self.block_type == 0
    }

    pub fn is_virgin(&self) -> bool {
        self.namespace == u32::MAX && self.block_type == u32::MAX
    }
}

/// Minimal region tree for procgen output from plugins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionTreeCore {
    pub bounds: Aabb4,
    pub kind: RegionNodeKind,
    pub generator_version_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionNodeKind {
    Empty,
    Uniform(BlockData),
    ChunkArray(ChunkArrayData),
    Branch(Vec<RegionTreeCore>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkArrayData {
    pub bounds: Aabb4,
    #[serde(default)]
    pub scale_exp: i8,
    pub chunk_palette: Vec<ChunkPayload>,
    /// Per-cell index into `chunk_palette`, laid out in XYZW order matching the bounds extents.
    pub dense_indices: Vec<u16>,
    pub block_palette: Vec<BlockData>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkPayload {
    Empty,
    Dense16 {
        materials: Vec<u16>,
    },
    /// No override — leave existing world content unchanged.
    /// Used in blueprint/structure overlays so that sub-chunk structures
    /// don't erase surrounding terrain.
    Virgin,
}
