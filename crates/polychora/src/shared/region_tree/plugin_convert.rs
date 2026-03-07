use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::Aabb4;
use crate::shared::voxel::{BlockData, TesseractOrientation, CHUNK_VOLUME};
use polychora_plugin_api::region_tree as api;

use super::{RegionNodeKind, RegionTreeCore};

fn convert_aabb4(src: &api::Aabb4) -> Aabb4 {
    Aabb4 {
        min: src.min,
        max: src.max,
    }
}

fn convert_block_data(src: &api::BlockData) -> BlockData {
    BlockData {
        namespace: src.namespace,
        block_type: src.block_type,
        orientation: TesseractOrientation(src.orientation.0),
        extra_data: src.extra_data.clone(),
        scale_exp: src.scale_exp,
    }
}

fn convert_chunk_payload(src: &api::ChunkPayload) -> ChunkPayload {
    match src {
        api::ChunkPayload::Empty => ChunkPayload::Empty,
        api::ChunkPayload::Virgin => ChunkPayload::Virgin,
        api::ChunkPayload::Dense16 { materials } => {
            if materials.len() != CHUNK_VOLUME {
                return ChunkPayload::Empty;
            }
            ChunkPayload::from_dense_materials_compact(materials)
                .unwrap_or(ChunkPayload::Dense16 {
                    materials: materials.clone(),
                })
        }
    }
}

fn convert_chunk_array_data(src: &api::ChunkArrayData) -> ChunkArrayData {
    let chunk_palette: Vec<ChunkPayload> = src.chunk_palette.iter().map(convert_chunk_payload).collect();
    let block_palette: Vec<BlockData> = src.block_palette.iter().map(convert_block_data).collect();
    let bounds = convert_aabb4(&src.bounds);

    ChunkArrayData::from_dense_indices_with_block_palette(
        bounds, chunk_palette, src.dense_indices.clone(), Some(0), block_palette, src.scale_exp,
    )
    .unwrap_or_else(|_| empty_chunk_array(bounds, vec![BlockData::AIR], src.scale_exp))
}

fn empty_chunk_array(bounds: Aabb4, block_palette: Vec<BlockData>, scale_exp: i8) -> ChunkArrayData {
    ChunkArrayData::from_dense_indices_with_block_palette(
        bounds, vec![ChunkPayload::Empty], vec![0u16; 1], Some(0), block_palette, scale_exp,
    )
    .expect("creating empty chunk array should not fail")
}

fn convert_node_kind(src: &api::RegionNodeKind) -> RegionNodeKind {
    match src {
        api::RegionNodeKind::Empty => RegionNodeKind::Empty,
        api::RegionNodeKind::Uniform(block) => RegionNodeKind::Uniform(convert_block_data(block)),
        api::RegionNodeKind::ChunkArray(data) => RegionNodeKind::ChunkArray(convert_chunk_array_data(data)),
        api::RegionNodeKind::Branch(children) => {
            RegionNodeKind::Branch(children.iter().map(convert_region_tree).collect())
        }
    }
}

fn convert_region_tree(src: &api::RegionTreeCore) -> RegionTreeCore {
    RegionTreeCore {
        bounds: convert_aabb4(&src.bounds),
        kind: convert_node_kind(&src.kind),
        generator_version_hash: src.generator_version_hash,
    }
}

/// Convert a plugin API `RegionTreeCore` to the host's internal type.
pub fn region_tree_from_plugin(src: &api::RegionTreeCore) -> RegionTreeCore {
    convert_region_tree(src)
}
