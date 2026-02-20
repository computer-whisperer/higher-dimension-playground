use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::region_tree::{
    collect_non_empty_chunks_from_core_in_bounds, ChunkKey, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderTreeCore {
    pub bounds: Aabb4i,
    pub kind: RenderNodeKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderNodeKind {
    Empty,
    Uniform(u16),
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
            RegionNodeKind::Uniform(material) => RenderNodeKind::Uniform(*material),
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
    mapped
}

pub fn to_region_core(core: &RenderTreeCore) -> RegionTreeCore {
    fn map_kind(kind: &RenderNodeKind) -> RegionNodeKind {
        match kind {
            RenderNodeKind::Empty => RegionNodeKind::Empty,
            RenderNodeKind::Uniform(material) => RegionNodeKind::Uniform(*material),
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
) -> Vec<(ChunkKey, ChunkPayload)> {
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

pub fn compose_in_bounds(bounds: Aabb4i, mut children: Vec<RenderTreeCore>) -> RenderTreeCore {
    if !bounds.is_valid() {
        return RenderTreeCore::empty(bounds);
    }
    children
        .retain(|child| child.bounds.is_valid() && !matches!(child.kind, RenderNodeKind::Empty));
    if children.is_empty() {
        return RenderTreeCore::empty(bounds);
    }
    if children.len() == 1 && children[0].bounds == bounds {
        return children.pop().expect("single child");
    }
    let mut out = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(children),
    };
    normalize_render_core(&mut out);
    out
}

pub fn repeated_voxel_leaf(bounds: Aabb4i, payload: ChunkPayload) -> Option<RenderTreeCore> {
    if !bounds.is_valid() {
        return None;
    }
    match payload {
        ChunkPayload::Empty => Some(RenderTreeCore::empty(bounds)),
        ChunkPayload::Uniform(material) => Some(RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Uniform(material),
        }),
        payload => {
            let cell_count = bounds.chunk_cell_count()?;
            let indices = vec![0u16; cell_count];
            let chunk_array =
                ChunkArrayData::from_dense_indices(bounds, vec![payload], indices, None).ok()?;
            Some(RenderTreeCore {
                bounds,
                kind: RenderNodeKind::VoxelChunkArray(chunk_array),
            })
        }
    }
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
mod tests {
    use super::*;
    use crate::shared::chunk_payload::ChunkArrayData;
    use crate::shared::region_tree::GeneratorRef;

    #[test]
    fn from_region_core_drops_procedural_and_keeps_chunk_content() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let uniform_child = RegionTreeCore {
            bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
            kind: RegionNodeKind::Uniform(7),
            generator_version_hash: 1,
        };
        let procedural_child = RegionTreeCore {
            bounds: Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
            kind: RegionNodeKind::ProceduralRef(GeneratorRef {
                generator_id: "test".into(),
                params: vec![],
                seed: 1,
            }),
            generator_version_hash: 1,
        };
        let src = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Branch(vec![uniform_child, procedural_child]),
            generator_version_hash: 1,
        };
        let out = from_region_core(&src);
        match out.kind {
            RenderNodeKind::Uniform(7) => {}
            RenderNodeKind::Branch(children) => {
                assert_eq!(children.len(), 1);
                assert!(matches!(children[0].kind, RenderNodeKind::Uniform(7)));
            }
            other => panic!("unexpected mapped kind: {other:?}"),
        }
    }

    #[test]
    fn collect_non_empty_chunks_handles_uniform_and_voxel_chunk_array() {
        let uniform_bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let voxel_bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
        let chunk_array = ChunkArrayData::from_dense_indices(
            voxel_bounds,
            vec![ChunkPayload::Uniform(11)],
            vec![0],
            None,
        )
        .expect("chunk array");
        let core = RenderTreeCore {
            bounds: Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]),
            kind: RenderNodeKind::Branch(vec![
                RenderTreeCore {
                    bounds: uniform_bounds,
                    kind: RenderNodeKind::Uniform(5),
                },
                RenderTreeCore {
                    bounds: voxel_bounds,
                    kind: RenderNodeKind::VoxelChunkArray(chunk_array),
                },
            ]),
        };
        let chunks = collect_non_empty_chunks_in_bounds(&core, core.bounds);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, [0, 0, 0, 0]);
        assert_eq!(chunks[1].0, [1, 0, 0, 0]);
    }
}
