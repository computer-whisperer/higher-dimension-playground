use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::region_tree::{
    collect_non_empty_chunks_from_core_in_bounds, validate_region_core_world_space_non_overlapping,
    ChunkKey, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::{fixed_from_lattice, lattice_from_fixed, Aabb4i, ChunkCoord};
use crate::shared::voxel::BlockData;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTreeCore {
    pub bounds: Aabb4i,
    pub kind: RenderNodeKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderNodeKind {
    Empty,
    Uniform(BlockData),
    VoxelChunkArray(ChunkArrayData),
    Branch(Vec<RenderTreeCore>),
}

impl RenderTreeCore {
    pub fn empty(bounds: Aabb4i) -> Self {
        Self {
            bounds,
            kind: RenderNodeKind::Empty,
        }
    }
}

pub fn from_region_core(core: &RegionTreeCore) -> RenderTreeCore {
    fn map_kind(kind: &RegionNodeKind) -> RenderNodeKind {
        match kind {
            RegionNodeKind::Empty => RenderNodeKind::Empty,
            RegionNodeKind::Uniform(block) => RenderNodeKind::Uniform(block.clone()),
            RegionNodeKind::ProceduralRef(_) => RenderNodeKind::Empty,
            RegionNodeKind::ChunkArray(chunk_array) => {
                RenderNodeKind::VoxelChunkArray(chunk_array.clone())
            }
            RegionNodeKind::Branch(children) => {
                let mapped_children: Vec<RenderTreeCore> = children
                    .iter()
                    .map(from_region_core)
                    .filter(|child| !matches!(child.kind, RenderNodeKind::Empty))
                    .collect();
                if mapped_children.is_empty() {
                    RenderNodeKind::Empty
                } else {
                    RenderNodeKind::Branch(mapped_children)
                }
            }
        }
    }

    let mut mapped = RenderTreeCore {
        bounds: core.bounds,
        kind: map_kind(&core.kind),
    };
    normalize_render_core(&mut mapped);
    #[cfg(debug_assertions)]
    if let Err(error) = validate_render_core_world_space_non_overlapping(&mapped) {
        eprintln!(
            "[render-tree-scale-overlap] BUG: from_region_core produced overlap: {}",
            error
        );
    }
    mapped
}

pub fn to_region_core(core: &RenderTreeCore) -> RegionTreeCore {
    fn map_kind(kind: &RenderNodeKind) -> RegionNodeKind {
        match kind {
            RenderNodeKind::Empty => RegionNodeKind::Empty,
            RenderNodeKind::Uniform(block) => RegionNodeKind::Uniform(block.clone()),
            RenderNodeKind::VoxelChunkArray(chunk_array) => {
                RegionNodeKind::ChunkArray(chunk_array.clone())
            }
            RenderNodeKind::Branch(children) => {
                RegionNodeKind::Branch(children.iter().map(to_region_core).collect())
            }
        }
    }

    RegionTreeCore {
        bounds: core.bounds,
        kind: map_kind(&core.kind),
        generator_version_hash: 0,
    }
}

pub fn collect_non_empty_chunks_in_bounds(
    core: &RenderTreeCore,
    bounds: Aabb4i,
) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let core_region = to_region_core(core);
    collect_non_empty_chunks_from_core_in_bounds(&core_region, bounds)
}

pub fn collect_non_empty_chunk_keys_in_bounds(
    core: &RenderTreeCore,
    bounds: Aabb4i,
) -> Vec<ChunkKey> {
    let mut keys: Vec<ChunkKey> = collect_non_empty_chunks_in_bounds(core, bounds)
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

pub fn repeated_voxel_leaf(
    bounds: Aabb4i,
    payload: ChunkPayload,
    block_palette: &[BlockData],
) -> Option<RenderTreeCore> {
    if !bounds.is_valid() {
        return None;
    }
    match payload {
        ChunkPayload::Empty => Some(RenderTreeCore::empty(bounds)),
        ChunkPayload::Uniform(idx) => {
            let block = block_palette
                .get(idx as usize)
                .cloned()
                .unwrap_or(BlockData::AIR);
            if block.is_air() {
                Some(RenderTreeCore::empty(bounds))
            } else {
                Some(RenderTreeCore {
                    bounds,
                    kind: RenderNodeKind::Uniform(block),
                })
            }
        }
        payload => {
            let cell_count = bounds.chunk_cell_count_at_scale(0)?;
            let indices = vec![0u16; cell_count];
            let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
                bounds,
                vec![payload],
                indices,
                None,
                block_palette.to_vec(),
                0,
            )
            .ok()?;
            Some(RenderTreeCore {
                bounds,
                kind: RenderNodeKind::VoxelChunkArray(chunk_array),
            })
        }
    }
}

mod bvh;
pub use bvh::*;

fn intersect_bounds(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    if !a.intersects(&b) {
        return None;
    }
    Some(Aabb4i::new(
        [
            a.min[0].max(b.min[0]),
            a.min[1].max(b.min[1]),
            a.min[2].max(b.min[2]),
            a.min[3].max(b.min[3]),
        ],
        [
            a.max[0].min(b.max[0]),
            a.max[1].min(b.max[1]),
            a.max[2].min(b.max[2]),
            a.max[3].min(b.max[3]),
        ],
    ))
}

fn union_bounds(a: Aabb4i, b: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
            a.min[3].min(b.min[3]),
        ],
        [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
            a.max[3].max(b.max[3]),
        ],
    )
}

fn bounds_contains_bounds(outer: Aabb4i, inner: Aabb4i) -> bool {
    outer.is_valid()
        && inner.is_valid()
        && outer.min[0] <= inner.min[0]
        && outer.min[1] <= inner.min[1]
        && outer.min[2] <= inner.min[2]
        && outer.min[3] <= inner.min[3]
        && outer.max[0] >= inner.max[0]
        && outer.max[1] >= inner.max[1]
        && outer.max[2] >= inner.max[2]
        && outer.max[3] >= inner.max[3]
}

fn normalize_render_core(core: &mut RenderTreeCore) {
    let RenderNodeKind::Branch(children) = &mut core.kind else {
        return;
    };
    for child in children.iter_mut() {
        normalize_render_core(child);
    }
    children.retain(|child| !matches!(child.kind, RenderNodeKind::Empty));
    if children.is_empty() {
        core.kind = RenderNodeKind::Empty;
    } else if children.len() == 1 && children[0].bounds == core.bounds {
        let child = children.pop().expect("single child");
        core.kind = child.kind;
    }
}

#[cfg(test)]
mod tests;
