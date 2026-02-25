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
    slice_region_core_in_bounds, validate_region_core_world_space_non_overlapping, RegionChunkTree,
};

#[cfg(test)]
mod tests;
