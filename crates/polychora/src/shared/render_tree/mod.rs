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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderBvh {
    pub bounds: Aabb4i,
    pub root: Option<u32>,
    pub nodes: Vec<RenderBvhNode>,
    pub leaves: Vec<RenderLeaf>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderBvhNode {
    pub bounds: Aabb4i,
    pub kind: RenderBvhNodeKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderBvhNodeKind {
    Internal { left: u32, right: u32 },
    Leaf { leaf_index: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderLeaf {
    pub bounds: Aabb4i,
    pub kind: RenderLeafKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderLeafKind {
    Uniform(u16),
    VoxelChunkArray(ChunkArrayData),
}

pub fn build_bvh_in_bounds(core: &RenderTreeCore, bounds: Aabb4i) -> RenderBvh {
    if !bounds.is_valid() {
        return RenderBvh {
            bounds,
            root: None,
            nodes: Vec::new(),
            leaves: Vec::new(),
        };
    }

    let mut leaves = Vec::<RenderLeaf>::new();
    collect_render_leaves_in_bounds(core, bounds, &mut leaves);
    if leaves.is_empty() {
        return RenderBvh {
            bounds,
            root: None,
            nodes: Vec::new(),
            leaves,
        };
    }

    let mut nodes = Vec::<RenderBvhNode>::new();
    let mut leaf_indices: Vec<usize> = (0..leaves.len()).collect();
    let root = Some(build_bvh_recursive(
        &leaves,
        &mut leaf_indices[..],
        &mut nodes,
    ));

    RenderBvh {
        bounds,
        root,
        nodes,
        leaves,
    }
}

pub fn collect_non_empty_chunk_keys_from_bvh_in_bounds(
    bvh: &RenderBvh,
    bounds: Aabb4i,
) -> Vec<ChunkKey> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let Some(root) = bvh.root else {
        return Vec::new();
    };

    let mut keys = Vec::<ChunkKey>::new();
    let mut stack = Vec::<u32>::with_capacity(64);
    stack.push(root);

    while let Some(node_idx) = stack.pop() {
        let Some(node) = bvh.nodes.get(node_idx as usize) else {
            continue;
        };
        if !node.bounds.intersects(&bounds) {
            continue;
        }

        match node.kind {
            RenderBvhNodeKind::Internal { left, right } => {
                stack.push(right);
                stack.push(left);
            }
            RenderBvhNodeKind::Leaf { leaf_index } => {
                let Some(leaf) = bvh.leaves.get(leaf_index as usize) else {
                    continue;
                };
                collect_non_empty_leaf_chunk_keys_in_bounds(leaf, bounds, &mut keys);
            }
        }
    }

    keys.sort_unstable();
    keys.dedup();
    keys
}

fn collect_render_leaves_in_bounds(
    core: &RenderTreeCore,
    bounds: Aabb4i,
    out: &mut Vec<RenderLeaf>,
) {
    let Some(intersection) = intersect_bounds(core.bounds, bounds) else {
        return;
    };
    match &core.kind {
        RenderNodeKind::Empty => {}
        RenderNodeKind::Uniform(material) => {
            if *material != 0 {
                out.push(RenderLeaf {
                    bounds: intersection,
                    kind: RenderLeafKind::Uniform(*material),
                });
            }
        }
        RenderNodeKind::VoxelChunkArray(chunk_array) => out.push(RenderLeaf {
            bounds: intersection,
            kind: RenderLeafKind::VoxelChunkArray(chunk_array.clone()),
        }),
        RenderNodeKind::Branch(children) => {
            for child in children {
                collect_render_leaves_in_bounds(child, intersection, out);
            }
        }
    }
}

fn build_bvh_recursive(
    leaves: &[RenderLeaf],
    leaf_indices: &mut [usize],
    nodes: &mut Vec<RenderBvhNode>,
) -> u32 {
    if leaf_indices.len() == 1 {
        let leaf_index = leaf_indices[0] as u32;
        let node_idx = nodes.len() as u32;
        nodes.push(RenderBvhNode {
            bounds: leaves[leaf_indices[0]].bounds,
            kind: RenderBvhNodeKind::Leaf { leaf_index },
        });
        return node_idx;
    }

    let aggregate_bounds = aggregate_bounds_for_leaf_indices(leaves, leaf_indices);
    let split_axis = longest_axis(aggregate_bounds);
    leaf_indices.sort_unstable_by(|a, b| {
        let ca = centroid_axis(leaves[*a].bounds, split_axis);
        let cb = centroid_axis(leaves[*b].bounds, split_axis);
        ca.total_cmp(&cb).then_with(|| {
            leaves[*a]
                .bounds
                .min
                .cmp(&leaves[*b].bounds.min)
                .then_with(|| leaves[*a].bounds.max.cmp(&leaves[*b].bounds.max))
        })
    });

    let mid = leaf_indices.len() / 2;
    let (left_indices, right_indices) = leaf_indices.split_at_mut(mid);
    let left = build_bvh_recursive(leaves, left_indices, nodes);
    let right = build_bvh_recursive(leaves, right_indices, nodes);
    let left_bounds = nodes[left as usize].bounds;
    let right_bounds = nodes[right as usize].bounds;

    let node_idx = nodes.len() as u32;
    nodes.push(RenderBvhNode {
        bounds: union_bounds(left_bounds, right_bounds),
        kind: RenderBvhNodeKind::Internal { left, right },
    });
    node_idx
}

fn collect_non_empty_leaf_chunk_keys_in_bounds(
    leaf: &RenderLeaf,
    bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(clipped_bounds) = intersect_bounds(leaf.bounds, bounds) else {
        return;
    };
    match &leaf.kind {
        RenderLeafKind::Uniform(material) => {
            if *material == 0 {
                return;
            }
            enumerate_chunk_keys(clipped_bounds, out);
        }
        RenderLeafKind::VoxelChunkArray(chunk_array) => {
            collect_non_empty_chunkarray_chunk_keys_in_bounds(chunk_array, clipped_bounds, out);
        }
    }
}

fn collect_non_empty_chunkarray_chunk_keys_in_bounds(
    chunk_array: &ChunkArrayData,
    bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(clipped_bounds) = intersect_bounds(bounds, chunk_array.bounds) else {
        return;
    };
    let Some(dims) = chunk_array.bounds.chunk_extents() else {
        return;
    };
    let Ok(indices) = chunk_array.decode_dense_indices() else {
        return;
    };
    if chunk_array.chunk_palette.is_empty() {
        return;
    }

    let palette_non_empty: Vec<bool> = chunk_array
        .chunk_palette
        .iter()
        .map(chunk_payload_has_solid_material)
        .collect();

    for w in clipped_bounds.min[3]..=clipped_bounds.max[3] {
        for z in clipped_bounds.min[2]..=clipped_bounds.max[2] {
            for y in clipped_bounds.min[1]..=clipped_bounds.max[1] {
                for x in clipped_bounds.min[0]..=clipped_bounds.max[0] {
                    let lx = (x - chunk_array.bounds.min[0]) as usize;
                    let ly = (y - chunk_array.bounds.min[1]) as usize;
                    let lz = (z - chunk_array.bounds.min[2]) as usize;
                    let lw = (w - chunk_array.bounds.min[3]) as usize;
                    let linear = lx + dims[0] * (ly + dims[1] * (lz + dims[2] * lw));
                    let Some(&palette_index) = indices.get(linear) else {
                        continue;
                    };
                    let palette_index = palette_index as usize;
                    if palette_non_empty
                        .get(palette_index)
                        .copied()
                        .unwrap_or(false)
                    {
                        out.push([x, y, z, w]);
                    }
                }
            }
        }
    }
}

fn chunk_payload_has_solid_material(payload: &ChunkPayload) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(material) => *material != 0,
        ChunkPayload::Dense16 { materials } => materials.iter().any(|material| *material != 0),
        ChunkPayload::PalettePacked { palette, .. } => {
            palette.iter().any(|material| *material != 0)
        }
    }
}

fn enumerate_chunk_keys(bounds: Aabb4i, out: &mut Vec<ChunkKey>) {
    for w in bounds.min[3]..=bounds.max[3] {
        for z in bounds.min[2]..=bounds.max[2] {
            for y in bounds.min[1]..=bounds.max[1] {
                for x in bounds.min[0]..=bounds.max[0] {
                    out.push([x, y, z, w]);
                }
            }
        }
    }
}

fn aggregate_bounds_for_leaf_indices(leaves: &[RenderLeaf], leaf_indices: &[usize]) -> Aabb4i {
    let mut bounds = leaves[leaf_indices[0]].bounds;
    for &leaf_idx in &leaf_indices[1..] {
        bounds = union_bounds(bounds, leaves[leaf_idx].bounds);
    }
    bounds
}

fn longest_axis(bounds: Aabb4i) -> usize {
    let mut axis = 0usize;
    let mut best_span = i64::MIN;
    for i in 0..4 {
        let span = i64::from(bounds.max[i]) - i64::from(bounds.min[i]);
        if span > best_span {
            best_span = span;
            axis = i;
        }
    }
    axis
}

fn centroid_axis(bounds: Aabb4i, axis: usize) -> f64 {
    (f64::from(bounds.min[axis]) + f64::from(bounds.max[axis])) * 0.5
}

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

    #[test]
    fn bvh_collect_matches_core_collect_for_mixed_leaves() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
        let chunk_array_bounds = Aabb4i::new([2, 0, 0, 0], [3, 0, 0, 0]);
        let chunk_array = ChunkArrayData::from_dense_indices(
            chunk_array_bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(9)],
            vec![1, 0],
            None,
        )
        .expect("chunk array");
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Branch(vec![
                RenderTreeCore {
                    bounds: Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(7),
                },
                RenderTreeCore {
                    bounds: chunk_array_bounds,
                    kind: RenderNodeKind::VoxelChunkArray(chunk_array),
                },
            ]),
        };

        let from_core = collect_non_empty_chunk_keys_in_bounds(&core, bounds);
        let bvh = build_bvh_in_bounds(&core, bounds);
        let from_bvh = collect_non_empty_chunk_keys_from_bvh_in_bounds(&bvh, bounds);
        assert_eq!(from_bvh, from_core);
    }

    #[test]
    fn bvh_collect_respects_query_bounds_clipping() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [4, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Uniform(3),
        };
        let bvh = build_bvh_in_bounds(&core, bounds);

        let query = Aabb4i::new([2, 0, 0, 0], [3, 0, 0, 0]);
        let keys = collect_non_empty_chunk_keys_from_bvh_in_bounds(&bvh, query);
        assert_eq!(keys, vec![[2, 0, 0, 0], [3, 0, 0, 0]]);
    }

    #[test]
    fn bvh_build_skips_air_uniform_leaves() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Uniform(0),
        };
        let bvh = build_bvh_in_bounds(&core, bounds);
        assert!(bvh.root.is_none());
        assert!(bvh.nodes.is_empty());
        assert!(bvh.leaves.is_empty());
    }
}
