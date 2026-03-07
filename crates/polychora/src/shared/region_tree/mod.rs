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

mod chunk_array_ops;
mod integrity;
mod tree;
mod tree_edit;
mod tree_normalize;
mod tree_query;

pub use chunk_array_ops::{optimize_subtree, resample_chunk_array_to_bounds};
pub use integrity::{
    validate_region_core_world_space_non_overlapping, validate_tree_integrity, TreeIntegrityReport,
};
pub use tree::{chunk_spatial_extent, coarsest_aligned_scale, RegionChunkTree};
pub use tree_edit::{slice_non_empty_region_core_in_bounds, slice_region_core_in_bounds};
pub use tree_query::{collect_non_empty_chunks_from_core_in_bounds, BvhBlockHit, BvhRayHit};

mod plugin_convert;

pub use plugin_convert::region_tree_from_plugin;

#[cfg(test)]
mod tests;
