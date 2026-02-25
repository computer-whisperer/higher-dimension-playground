use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{BlockData, ChunkPos};
use serde::{Deserialize, Serialize};

pub type ChunkKey = [i32; 4];

#[inline]
pub fn chunk_key_from_chunk_pos(chunk_pos: ChunkPos) -> ChunkKey {
    [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w]
}

#[inline]
pub fn chunk_pos_from_chunk_key(key: ChunkKey) -> ChunkPos {
    ChunkPos::new(key[0], key[1], key[2], key[3])
}

/// A chunk key paired with a scale exponent, identifying a chunk at a specific
/// position in a specific scale lattice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ScaledChunkKey {
    pub pos: ChunkKey,
    pub scale_exp: i8,
}

impl ScaledChunkKey {
    #[inline]
    pub fn new(pos: ChunkKey, scale_exp: i8) -> Self {
        Self { pos, scale_exp }
    }

    /// Shorthand for `scale_exp = 0` (the standard unit-scale lattice).
    #[inline]
    pub fn unit(pos: ChunkKey) -> Self {
        Self { pos, scale_exp: 0 }
    }
}

/// Wrap a `ChunkKey` as a `ScaledChunkKey` with `scale_exp = 0`.
#[inline]
pub fn chunk_key_to_scaled(key: ChunkKey) -> ScaledChunkKey {
    ScaledChunkKey::unit(key)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GeneratorRef {
    pub generator_id: String,
    pub params: Vec<u8>,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionTreeCore {
    pub bounds: Aabb4i,
    pub kind: RegionNodeKind,
    pub generator_version_hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionNodeKind {
    Empty,
    Uniform(BlockData),
    ProceduralRef(GeneratorRef),
    ChunkArray(ChunkArrayData),
    Branch(Vec<RegionTreeCore>),
}

mod tree;
pub use tree::{
    collect_non_empty_chunks_from_core_in_bounds, slice_non_empty_region_core_in_bounds,
    slice_region_core_in_bounds, validate_region_core_world_space_non_overlapping,
    RegionChunkTree,
};

#[cfg(test)]
mod tests;
