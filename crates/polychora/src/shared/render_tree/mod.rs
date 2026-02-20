use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::region_tree::{
    collect_non_empty_chunks_from_core_in_bounds, ChunkKey, RegionNodeKind, RegionTreeCore,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::CHUNK_SIZE;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DebugRayBvhNodeKind {
    Internal,
    LeafUniform { material: u16 },
    LeafChunkArray,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugRayBvhNodeHit {
    pub bounds: Aabb4i,
    pub kind: DebugRayBvhNodeKind,
    pub t_enter: f32,
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

pub fn sample_chunk_payloads_from_bvh(bvh: &RenderBvh, key: ChunkKey) -> Vec<ChunkPayload> {
    if !bvh.bounds.contains_chunk(key) {
        return Vec::new();
    }
    let Some(root) = bvh.root else {
        return Vec::new();
    };

    let mut out = Vec::<ChunkPayload>::new();
    let mut stack = Vec::<u32>::with_capacity(64);
    stack.push(root);

    while let Some(node_idx) = stack.pop() {
        let Some(node) = bvh.nodes.get(node_idx as usize) else {
            continue;
        };
        if !node.bounds.contains_chunk(key) {
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
                if let Some(payload) = sample_leaf_payload_at_key(leaf, key) {
                    out.push(payload);
                }
            }
        }
    }
    out
}

pub fn collect_ray_intersected_nodes_from_bvh(
    bvh: &RenderBvh,
    ray_origin_world: [f32; 4],
    ray_direction_world: [f32; 4],
    max_distance_world: f32,
    max_nodes: usize,
) -> Vec<DebugRayBvhNodeHit> {
    if max_nodes == 0 {
        return Vec::new();
    }
    let Some(root) = bvh.root else {
        return Vec::new();
    };
    if !max_distance_world.is_finite() || max_distance_world <= 0.0 {
        return Vec::new();
    }
    let dir_len_sq = ray_direction_world[0] * ray_direction_world[0]
        + ray_direction_world[1] * ray_direction_world[1]
        + ray_direction_world[2] * ray_direction_world[2]
        + ray_direction_world[3] * ray_direction_world[3];
    if !dir_len_sq.is_finite() || dir_len_sq <= 1e-12 {
        return Vec::new();
    }
    let inv_dir_len = dir_len_sq.sqrt().recip();
    let ray_dir = [
        ray_direction_world[0] * inv_dir_len,
        ray_direction_world[1] * inv_dir_len,
        ray_direction_world[2] * inv_dir_len,
        ray_direction_world[3] * inv_dir_len,
    ];

    let root_enter =
        ray_intersects_chunk_bounds(ray_origin_world, ray_dir, bvh.bounds, max_distance_world);
    let Some(root_enter) = root_enter else {
        return Vec::new();
    };

    let mut out = Vec::<DebugRayBvhNodeHit>::new();
    let mut frontier = Vec::<(u32, f32)>::with_capacity(64);
    frontier.push((root, root_enter));

    while !frontier.is_empty() && out.len() < max_nodes {
        let mut best_idx = 0usize;
        for idx in 1..frontier.len() {
            if frontier[idx].1 < frontier[best_idx].1 {
                best_idx = idx;
            }
        }
        let (node_idx, t_enter) = frontier.swap_remove(best_idx);
        let Some(node) = bvh.nodes.get(node_idx as usize) else {
            continue;
        };

        let kind = match node.kind {
            RenderBvhNodeKind::Internal { left, right } => {
                if let Some(left_node) = bvh.nodes.get(left as usize) {
                    if let Some(left_enter) = ray_intersects_chunk_bounds(
                        ray_origin_world,
                        ray_dir,
                        left_node.bounds,
                        max_distance_world,
                    ) {
                        frontier.push((left, left_enter));
                    }
                }
                if let Some(right_node) = bvh.nodes.get(right as usize) {
                    if let Some(right_enter) = ray_intersects_chunk_bounds(
                        ray_origin_world,
                        ray_dir,
                        right_node.bounds,
                        max_distance_world,
                    ) {
                        frontier.push((right, right_enter));
                    }
                }
                DebugRayBvhNodeKind::Internal
            }
            RenderBvhNodeKind::Leaf { leaf_index } => {
                let Some(leaf) = bvh.leaves.get(leaf_index as usize) else {
                    continue;
                };
                match leaf.kind {
                    RenderLeafKind::Uniform(material) => {
                        DebugRayBvhNodeKind::LeafUniform { material }
                    }
                    RenderLeafKind::VoxelChunkArray(_) => DebugRayBvhNodeKind::LeafChunkArray,
                }
            }
        };

        out.push(DebugRayBvhNodeHit {
            bounds: node.bounds,
            kind,
            t_enter,
        });
    }

    out
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

fn sample_leaf_payload_at_key(leaf: &RenderLeaf, key: ChunkKey) -> Option<ChunkPayload> {
    if !leaf.bounds.contains_chunk(key) {
        return None;
    }
    match &leaf.kind {
        RenderLeafKind::Uniform(material) => Some(ChunkPayload::Uniform(*material)),
        RenderLeafKind::VoxelChunkArray(chunk_array) => {
            sample_chunkarray_payload_at_key(chunk_array, key)
        }
    }
}

fn sample_chunkarray_payload_at_key(
    chunk_array: &ChunkArrayData,
    key: ChunkKey,
) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk(key) {
        return None;
    }
    let dims = chunk_array.bounds.chunk_extents()?;
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    let lx = (key[0] - chunk_array.bounds.min[0]) as usize;
    let ly = (key[1] - chunk_array.bounds.min[1]) as usize;
    let lz = (key[2] - chunk_array.bounds.min[2]) as usize;
    let lw = (key[3] - chunk_array.bounds.min[3]) as usize;
    let linear = lx + dims[0] * (ly + dims[1] * (lz + dims[2] * lw));
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
}

fn ray_intersects_chunk_bounds(
    ray_origin_world: [f32; 4],
    ray_dir_world: [f32; 4],
    chunk_bounds: Aabb4i,
    max_distance_world: f32,
) -> Option<f32> {
    if !chunk_bounds.is_valid() {
        return None;
    }
    let chunk_size = CHUNK_SIZE as f32;
    let min = [
        chunk_bounds.min[0] as f32 * chunk_size,
        chunk_bounds.min[1] as f32 * chunk_size,
        chunk_bounds.min[2] as f32 * chunk_size,
        chunk_bounds.min[3] as f32 * chunk_size,
    ];
    let max = [
        (chunk_bounds.max[0] + 1) as f32 * chunk_size,
        (chunk_bounds.max[1] + 1) as f32 * chunk_size,
        (chunk_bounds.max[2] + 1) as f32 * chunk_size,
        (chunk_bounds.max[3] + 1) as f32 * chunk_size,
    ];

    let mut t_near = 0.0f32;
    let mut t_far = max_distance_world;
    for axis in 0..4 {
        let origin = ray_origin_world[axis];
        let dir = ray_dir_world[axis];
        let slab_min = min[axis];
        let slab_max = max[axis];
        if dir.abs() < 1e-8 {
            if origin < slab_min || origin > slab_max {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir;
        let mut t0 = (slab_min - origin) * inv;
        let mut t1 = (slab_max - origin) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_near = t_near.max(t0);
        t_far = t_far.min(t1);
        if t_far < t_near {
            return None;
        }
    }
    if t_far < 0.0 {
        return None;
    }
    Some(t_near.max(0.0))
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

    #[test]
    fn sample_chunk_payloads_from_bvh_reads_uniform_and_chunk_array_leaves() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let chunk_array_bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
        let chunk_array = ChunkArrayData::from_dense_indices(
            chunk_array_bounds,
            vec![ChunkPayload::Uniform(11)],
            vec![0],
            None,
        )
        .expect("chunk array");
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Branch(vec![
                RenderTreeCore {
                    bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(7),
                },
                RenderTreeCore {
                    bounds: chunk_array_bounds,
                    kind: RenderNodeKind::VoxelChunkArray(chunk_array),
                },
            ]),
        };
        let bvh = build_bvh_in_bounds(&core, bounds);

        assert_eq!(
            sample_chunk_payloads_from_bvh(&bvh, [0, 0, 0, 0]),
            vec![ChunkPayload::Uniform(7)]
        );
        assert_eq!(
            sample_chunk_payloads_from_bvh(&bvh, [1, 0, 0, 0]),
            vec![ChunkPayload::Uniform(11)]
        );
        assert!(sample_chunk_payloads_from_bvh(&bvh, [2, 0, 0, 0]).is_empty());
    }

    #[test]
    fn ray_bvh_hits_uniform_leaf() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Uniform(7),
        };
        let bvh = build_bvh_in_bounds(&core, bounds);
        let hits = collect_ray_intersected_nodes_from_bvh(
            &bvh,
            [0.5, 0.5, -2.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            16.0,
            16,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].bounds, bounds);
        assert_eq!(
            hits[0].kind,
            DebugRayBvhNodeKind::LeafUniform { material: 7 }
        );
    }

    #[test]
    fn ray_bvh_hits_internal_then_near_leaf() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Branch(vec![
                RenderTreeCore {
                    bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(3),
                },
                RenderTreeCore {
                    bounds: Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(9),
                },
            ]),
        };
        let bvh = build_bvh_in_bounds(&core, bounds);
        let hits = collect_ray_intersected_nodes_from_bvh(
            &bvh,
            [0.5, 0.5, -2.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
            16.0,
            16,
        );
        assert!(hits.len() >= 2);
        assert_eq!(hits[0].kind, DebugRayBvhNodeKind::Internal);
        assert_eq!(
            hits[1].kind,
            DebugRayBvhNodeKind::LeafUniform { material: 3 }
        );
    }
}
