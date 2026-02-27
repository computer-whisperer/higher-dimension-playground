use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::{Aabb4i, ChunkCoord};
use crate::shared::voxel::BlockData;
use serde::{Deserialize, Serialize};

pub type ChunkKey = [ChunkCoord; 4];

/// Convenience constructor for a ChunkKey from integer coordinates (scale 0).
#[inline]
pub fn chunk_key_i32(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    [
        ChunkCoord::from_num(x),
        ChunkCoord::from_num(y),
        ChunkCoord::from_num(z),
        ChunkCoord::from_num(w),
    ]
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
    chunk_spatial_extent, collect_non_empty_chunks_from_core_in_bounds,
    slice_non_empty_region_core_in_bounds, slice_region_core_in_bounds,
    validate_region_core_world_space_non_overlapping, RegionChunkTree,
};

#[cfg(test)]
mod tests;
