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
    free_node_ids: Vec<u32>,
    free_leaf_ids: Vec<u32>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderBvhChunkMutationDelta {
    pub key: ChunkKey,
    pub expected_root: Option<u32>,
    pub new_root: Option<u32>,
    pub node_writes: Vec<(u32, RenderBvhNode)>,
    pub leaf_writes: Vec<(u32, RenderLeaf)>,
    pub freed_node_ids: Vec<u32>,
    pub freed_leaf_ids: Vec<u32>,
}

impl RenderBvhChunkMutationDelta {
    pub fn root_changed(&self) -> bool {
        self.expected_root != self.new_root
    }
}

pub fn build_bvh_in_bounds(core: &RenderTreeCore, bounds: Aabb4i) -> RenderBvh {
    if !bounds.is_valid() {
        return RenderBvh {
            bounds,
            root: None,
            nodes: Vec::new(),
            leaves: Vec::new(),
            free_node_ids: Vec::new(),
            free_leaf_ids: Vec::new(),
        };
    }

    let mut leaf_inputs = Vec::<RenderLeaf>::new();
    collect_render_leaves_in_bounds(core, bounds, &mut leaf_inputs);
    if leaf_inputs.is_empty() {
        return RenderBvh {
            bounds,
            root: None,
            nodes: Vec::new(),
            leaves: Vec::new(),
            free_node_ids: Vec::new(),
            free_leaf_ids: Vec::new(),
        };
    }

    let mut bvh = RenderBvh {
        bounds,
        root: None,
        nodes: Vec::new(),
        leaves: Vec::new(),
        free_node_ids: Vec::new(),
        free_leaf_ids: Vec::new(),
    };
    let mut init_delta = RenderBvhChunkMutationDelta {
        key: bounds.min,
        expected_root: None,
        new_root: None,
        node_writes: Vec::new(),
        leaf_writes: Vec::new(),
        freed_node_ids: Vec::new(),
        freed_leaf_ids: Vec::new(),
    };
    let root = append_leaves_subtree_with_delta(&mut bvh, leaf_inputs, &mut init_delta)
        .expect("initial BVH build should succeed");
    bvh.root = Some(root);
    bvh
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

pub fn apply_chunk_payload_mutations_in_bvh(
    bvh: &mut RenderBvh,
    mutations: &[(ChunkKey, Option<ChunkPayload>)],
) -> Result<usize, String> {
    Ok(apply_chunk_payload_mutations_with_deltas_in_bvh(bvh, mutations)?.len())
}

pub fn apply_chunk_payload_mutations_with_deltas_in_bvh(
    bvh: &mut RenderBvh,
    mutations: &[(ChunkKey, Option<ChunkPayload>)],
) -> Result<Vec<RenderBvhChunkMutationDelta>, String> {
    let mut deltas = Vec::<RenderBvhChunkMutationDelta>::new();
    for (key, payload) in mutations {
        if !bvh.bounds.contains_chunk(*key) {
            continue;
        }
        let key_bounds = Aabb4i::new(*key, *key);
        let patch_core = if let Some(payload) = payload.clone() {
            repeated_voxel_leaf(key_bounds, payload)
                .unwrap_or_else(|| RenderTreeCore::empty(key_bounds))
        } else {
            RenderTreeCore::empty(key_bounds)
        };
        if let Some(delta) =
            apply_core_patch_in_bounds_with_delta_in_bvh(bvh, key_bounds, &patch_core)?
        {
            deltas.push(delta);
        }
    }
    Ok(deltas)
}

pub fn apply_core_patch_in_bounds_with_delta_in_bvh(
    bvh: &mut RenderBvh,
    patch_bounds: Aabb4i,
    patch_core: &RenderTreeCore,
) -> Result<Option<RenderBvhChunkMutationDelta>, String> {
    if !patch_bounds.is_valid() {
        return Ok(None);
    }
    let Some(clipped_patch_bounds) = intersect_bounds(bvh.bounds, patch_bounds) else {
        return Ok(None);
    };

    let expected_root = bvh.root;
    let mut delta = RenderBvhChunkMutationDelta {
        key: clipped_patch_bounds.min,
        expected_root,
        new_root: expected_root,
        node_writes: Vec::new(),
        leaf_writes: Vec::new(),
        freed_node_ids: Vec::new(),
        freed_leaf_ids: Vec::new(),
    };
    let mut retired_node_ids = Vec::<u32>::new();
    let mut retired_leaf_ids = Vec::<u32>::new();

    let outside_root = if let Some(root_idx) = expected_root {
        clone_subtree_excluding_bounds(
            bvh,
            root_idx,
            clipped_patch_bounds,
            &mut delta,
            &mut retired_node_ids,
            &mut retired_leaf_ids,
        )?
    } else {
        None
    };

    let mut replacement_leaves = Vec::<RenderLeaf>::new();
    collect_render_leaves_in_bounds(patch_core, clipped_patch_bounds, &mut replacement_leaves);
    let replacement_root = if replacement_leaves.is_empty() {
        None
    } else {
        Some(append_leaves_subtree_with_delta(
            bvh,
            replacement_leaves,
            &mut delta,
        )?)
    };

    let new_root = match (outside_root, replacement_root) {
        (None, None) => None,
        (Some(root), None) | (None, Some(root)) => Some(root),
        (Some(left), Some(right)) => {
            // Instead of a simple single-node join (which degrades tree quality
            // over many incremental mutations), collect all leaf IDs from both
            // subtrees and rebuild with SAH for optimal structure.
            let mut left_nodes = Vec::new();
            let mut left_leaf_ids = Vec::new();
            collect_subtree_node_leaf_ids(bvh, left, &mut left_nodes, &mut left_leaf_ids);

            let mut right_nodes = Vec::new();
            let mut right_leaf_ids = Vec::new();
            collect_subtree_node_leaf_ids(bvh, right, &mut right_nodes, &mut right_leaf_ids);

            // Recycle node slots into the CPU free list so the SAH rebuild can
            // reuse them. We do NOT add them to delta.freed_node_ids because
            // the rebuild will immediately rewrite them (via delta.node_writes).
            // The GPU consumer applies writes to these slots, so they remain
            // valid in the GPU buffer.
            for &node_id in left_nodes.iter().chain(right_nodes.iter()) {
                bvh.free_node_ids.push(node_id);
            }

            let mut all_leaf_ids: Vec<u32> = left_leaf_ids;
            all_leaf_ids.extend(right_leaf_ids);
            Some(build_bvh_recursive_for_leaf_ids(
                bvh,
                &mut all_leaf_ids,
                &mut delta,
            )?)
        }
    };

    bvh.root = new_root;
    release_retired_nodes_and_leaves(bvh, retired_node_ids, retired_leaf_ids, &mut delta);
    delta.new_root = bvh.root;

    if delta.expected_root == delta.new_root
        && delta.node_writes.is_empty()
        && delta.leaf_writes.is_empty()
        && delta.freed_node_ids.is_empty()
        && delta.freed_leaf_ids.is_empty()
    {
        return Ok(None);
    }

    Ok(Some(delta))
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

fn clone_subtree_excluding_bounds(
    bvh: &mut RenderBvh,
    node_idx: u32,
    excluded_bounds: Aabb4i,
    delta: &mut RenderBvhChunkMutationDelta,
    retired_node_ids: &mut Vec<u32>,
    retired_leaf_ids: &mut Vec<u32>,
) -> Result<Option<u32>, String> {
    let Some(node) = bvh.nodes.get(node_idx as usize).cloned() else {
        return Ok(None);
    };
    if !node.bounds.intersects(&excluded_bounds) {
        return Ok(Some(node_idx));
    }
    if bounds_contains_bounds(excluded_bounds, node.bounds) {
        collect_subtree_node_leaf_ids(bvh, node_idx, retired_node_ids, retired_leaf_ids);
        return Ok(None);
    }
    retired_node_ids.push(node_idx);

    match node.kind {
        RenderBvhNodeKind::Leaf { leaf_index } => {
            let Some(leaf) = bvh.leaves.get(leaf_index as usize).cloned() else {
                return Ok(None);
            };
            retired_leaf_ids.push(leaf_index);
            let replacement_leaves = build_leaf_outside_bounds_pieces(&leaf, excluded_bounds)?;
            if replacement_leaves.is_empty() {
                Ok(None)
            } else {
                Ok(Some(append_leaves_subtree_with_delta(
                    bvh,
                    replacement_leaves,
                    delta,
                )?))
            }
        }
        RenderBvhNodeKind::Internal { left, right } => {
            let left_root = clone_subtree_excluding_bounds(
                bvh,
                left,
                excluded_bounds,
                delta,
                retired_node_ids,
                retired_leaf_ids,
            )?;
            let right_root = clone_subtree_excluding_bounds(
                bvh,
                right,
                excluded_bounds,
                delta,
                retired_node_ids,
                retired_leaf_ids,
            )?;
            match (left_root, right_root) {
                (None, None) => Ok(None),
                (Some(only), None) | (None, Some(only)) => Ok(Some(only)),
                (Some(new_left), Some(new_right)) => {
                    let bounds = union_bounds(
                        bvh.nodes[new_left as usize].bounds,
                        bvh.nodes[new_right as usize].bounds,
                    );
                    Ok(Some(append_internal_node_with_delta(
                        bvh, new_left, new_right, bounds, delta,
                    )))
                }
            }
        }
    }
}

fn build_leaf_outside_bounds_pieces(
    leaf: &RenderLeaf,
    excluded_bounds: Aabb4i,
) -> Result<Vec<RenderLeaf>, String> {
    let mut out = Vec::<RenderLeaf>::new();
    if !leaf.bounds.intersects(&excluded_bounds) {
        out.push(leaf.clone());
        return Ok(out);
    }
    let split_bounds = split_bounds_excluding_bounds(leaf.bounds, excluded_bounds);
    match &leaf.kind {
        RenderLeafKind::Uniform(material) => {
            if *material == 0 {
                return Ok(out);
            }
            for piece_bounds in split_bounds {
                out.push(RenderLeaf {
                    bounds: piece_bounds,
                    kind: RenderLeafKind::Uniform(*material),
                });
            }
        }
        RenderLeafKind::VoxelChunkArray(chunk_array) => {
            let source_indices = chunk_array
                .decode_dense_indices()
                .map_err(|error| format!("decode chunk-array leaf failed: {error:?}"))?;
            for piece_bounds in split_bounds {
                if let Some(piece_kind) =
                    slice_chunk_array_kind_to_bounds(chunk_array, &source_indices, piece_bounds)?
                {
                    out.push(RenderLeaf {
                        bounds: piece_bounds,
                        kind: piece_kind,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn split_bounds_excluding_bounds(bounds: Aabb4i, excluded_bounds: Aabb4i) -> Vec<Aabb4i> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let Some(clipped_excluded) = intersect_bounds(bounds, excluded_bounds) else {
        return vec![bounds];
    };

    let mut rem_min = bounds.min;
    let mut rem_max = bounds.max;
    let mut pieces = Vec::<Aabb4i>::new();
    for axis in 0..4 {
        if rem_min[axis] < clipped_excluded.min[axis] {
            let mut piece_max = rem_max;
            piece_max[axis] = clipped_excluded.min[axis] - 1;
            pieces.push(Aabb4i::new(rem_min, piece_max));
            rem_min[axis] = clipped_excluded.min[axis];
        }
        if rem_max[axis] > clipped_excluded.max[axis] {
            let mut piece_min = rem_min;
            piece_min[axis] = clipped_excluded.max[axis] + 1;
            pieces.push(Aabb4i::new(piece_min, rem_max));
            rem_max[axis] = clipped_excluded.max[axis];
        }
    }
    pieces
}

fn slice_chunk_array_kind_to_bounds(
    chunk_array: &ChunkArrayData,
    source_indices: &[u16],
    bounds: Aabb4i,
) -> Result<Option<RenderLeafKind>, String> {
    let Some(clipped_bounds) = intersect_bounds(bounds, chunk_array.bounds) else {
        return Ok(None);
    };
    if clipped_bounds != bounds {
        return Ok(None);
    }

    let Some(src_dims) = chunk_array.bounds.chunk_extents() else {
        return Ok(None);
    };
    let Some(piece_dims) = bounds.chunk_extents() else {
        return Ok(None);
    };
    let piece_cell_count = bounds
        .chunk_cell_count()
        .ok_or_else(|| "piece chunk count overflow".to_string())?;
    let mut piece_indices = Vec::<u16>::with_capacity(piece_cell_count);
    let palette_non_empty: Vec<bool> = chunk_array
        .chunk_palette
        .iter()
        .map(chunk_payload_has_solid_material)
        .collect();
    let mut any_non_empty = false;

    for w in 0..piece_dims[3] {
        for z in 0..piece_dims[2] {
            for y in 0..piece_dims[1] {
                for x in 0..piece_dims[0] {
                    let src_coord = [
                        bounds.min[0] + x as i32,
                        bounds.min[1] + y as i32,
                        bounds.min[2] + z as i32,
                        bounds.min[3] + w as i32,
                    ];
                    let lx = (src_coord[0] - chunk_array.bounds.min[0]) as usize;
                    let ly = (src_coord[1] - chunk_array.bounds.min[1]) as usize;
                    let lz = (src_coord[2] - chunk_array.bounds.min[2]) as usize;
                    let lw = (src_coord[3] - chunk_array.bounds.min[3]) as usize;
                    let src_linear =
                        lx + src_dims[0] * (ly + src_dims[1] * (lz + src_dims[2] * lw));
                    let Some(&palette_index) = source_indices.get(src_linear) else {
                        return Err("chunk-array source index out of bounds".to_string());
                    };
                    if palette_non_empty
                        .get(palette_index as usize)
                        .copied()
                        .unwrap_or(false)
                    {
                        any_non_empty = true;
                    }
                    piece_indices.push(palette_index);
                }
            }
        }
    }

    if !any_non_empty {
        return Ok(None);
    }

    let piece_chunk_array = ChunkArrayData::from_dense_indices(
        bounds,
        chunk_array.chunk_palette.clone(),
        piece_indices,
        None,
    )
    .map_err(|error| format!("slice chunk-array piece failed: {error:?}"))?;
    Ok(Some(RenderLeafKind::VoxelChunkArray(piece_chunk_array)))
}

fn append_leaves_subtree_with_delta(
    bvh: &mut RenderBvh,
    leaves: Vec<RenderLeaf>,
    delta: &mut RenderBvhChunkMutationDelta,
) -> Result<u32, String> {
    if leaves.is_empty() {
        return Err("cannot append empty leaf subtree".to_string());
    }
    let mut leaf_ids = Vec::<u32>::with_capacity(leaves.len());
    for leaf in leaves {
        let leaf_id = allocate_leaf_slot(bvh, leaf, delta)?;
        leaf_ids.push(leaf_id);
    }
    build_bvh_recursive_for_leaf_ids(bvh, &mut leaf_ids[..], delta)
}

fn append_internal_node_with_delta(
    bvh: &mut RenderBvh,
    left: u32,
    right: u32,
    bounds: Aabb4i,
    delta: &mut RenderBvhChunkMutationDelta,
) -> u32 {
    allocate_node_slot(
        bvh,
        RenderBvhNode {
            bounds,
            kind: RenderBvhNodeKind::Internal { left, right },
        },
        delta,
    )
    .expect("allocate internal node")
}

fn allocate_leaf_slot(
    bvh: &mut RenderBvh,
    leaf: RenderLeaf,
    delta: &mut RenderBvhChunkMutationDelta,
) -> Result<u32, String> {
    if let Some(leaf_id) = bvh.free_leaf_ids.pop() {
        let Some(slot) = bvh.leaves.get_mut(leaf_id as usize) else {
            return Err(format!("free leaf id {} out of range", leaf_id));
        };
        *slot = leaf.clone();
        delta.leaf_writes.push((leaf_id, leaf));
        Ok(leaf_id)
    } else {
        let leaf_id = bvh.leaves.len() as u32;
        bvh.leaves.push(leaf.clone());
        delta.leaf_writes.push((leaf_id, leaf));
        Ok(leaf_id)
    }
}

fn allocate_node_slot(
    bvh: &mut RenderBvh,
    node: RenderBvhNode,
    delta: &mut RenderBvhChunkMutationDelta,
) -> Result<u32, String> {
    if let Some(node_id) = bvh.free_node_ids.pop() {
        let Some(slot) = bvh.nodes.get_mut(node_id as usize) else {
            return Err(format!("free node id {} out of range", node_id));
        };
        *slot = node.clone();
        delta.node_writes.push((node_id, node));
        Ok(node_id)
    } else {
        let node_id = bvh.nodes.len() as u32;
        bvh.nodes.push(node.clone());
        delta.node_writes.push((node_id, node));
        Ok(node_id)
    }
}

fn build_bvh_recursive_for_leaf_ids(
    bvh: &mut RenderBvh,
    leaf_ids: &mut [u32],
    delta: &mut RenderBvhChunkMutationDelta,
) -> Result<u32, String> {
    if leaf_ids.is_empty() {
        return Err("cannot build BVH for empty leaf id slice".to_string());
    }
    if leaf_ids.len() == 1 {
        let leaf_id = leaf_ids[0];
        let Some(leaf) = bvh.leaves.get(leaf_id as usize) else {
            return Err(format!("leaf id {} out of range", leaf_id));
        };
        return allocate_node_slot(
            bvh,
            RenderBvhNode {
                bounds: leaf.bounds,
                kind: RenderBvhNodeKind::Leaf {
                    leaf_index: leaf_id,
                },
            },
            delta,
        );
    }

    // Find the best split across all 4 axes using SAH (Surface Area Heuristic).
    // For each axis, sort by centroid, then sweep to find the split position that
    // minimizes the SAH cost: cost = SA(left)*N_left + SA(right)*N_right
    // where SA is the 4D "surface area" (sum of 3-face volumes) of the child bounds.
    let n = leaf_ids.len();
    let mut best_axis = 0usize;
    let mut best_split = n / 2;
    let mut best_cost = f64::MAX;

    // Scratch space for per-axis evaluation. We collect bounds once and reuse.
    let leaf_bounds: Vec<Aabb4i> = leaf_ids
        .iter()
        .map(|&id| {
            bvh.leaves
                .get(id as usize)
                .map(|leaf| leaf.bounds)
                .unwrap_or(Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]))
        })
        .collect();

    // We need to try sorting by each axis independently. Since leaf_ids is mutable
    // and we need to try 4 axes, work with index arrays.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut right_bounds_suffix = vec![Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]); n];

    for axis in 0..4 {
        // Sort indices by centroid on this axis.
        indices.sort_unstable_by(|&a, &b| {
            let ca = centroid_axis(leaf_bounds[a], axis);
            let cb = centroid_axis(leaf_bounds[b], axis);
            ca.total_cmp(&cb)
                .then_with(|| leaf_bounds[a].min.cmp(&leaf_bounds[b].min))
                .then_with(|| leaf_bounds[a].max.cmp(&leaf_bounds[b].max))
        });

        // Build suffix array of right-side aggregate bounds (from right to left).
        right_bounds_suffix[n - 1] = leaf_bounds[indices[n - 1]];
        for i in (0..n - 1).rev() {
            right_bounds_suffix[i] =
                union_bounds(right_bounds_suffix[i + 1], leaf_bounds[indices[i]]);
        }

        // Sweep left to right, evaluating SAH cost at each split position.
        // Split at position i means left = [0..=i], right = [i+1..n-1].
        let mut left_agg = leaf_bounds[indices[0]];
        for split in 1..n {
            let left_count = split as f64;
            let right_count = (n - split) as f64;
            let left_sa = half_surface_area_4d(left_agg);
            let right_sa = half_surface_area_4d(right_bounds_suffix[split]);
            let cost = left_sa * left_count + right_sa * right_count;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = split;
            }

            if split < n - 1 {
                left_agg = union_bounds(left_agg, leaf_bounds[indices[split]]);
            }
        }
    }

    // Now apply the best split: sort leaf_ids by the chosen axis and split.
    leaf_ids.sort_unstable_by(|a, b| {
        let ab = bvh
            .leaves
            .get(*a as usize)
            .map(|leaf| leaf.bounds)
            .unwrap_or(Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]));
        let bb = bvh
            .leaves
            .get(*b as usize)
            .map(|leaf| leaf.bounds)
            .unwrap_or(Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]));
        let ca = centroid_axis(ab, best_axis);
        let cb = centroid_axis(bb, best_axis);
        ca.total_cmp(&cb)
            .then_with(|| ab.min.cmp(&bb.min))
            .then_with(|| ab.max.cmp(&bb.max))
    });

    let (left_ids, right_ids) = leaf_ids.split_at_mut(best_split);
    let left = build_bvh_recursive_for_leaf_ids(bvh, left_ids, delta)?;
    let right = build_bvh_recursive_for_leaf_ids(bvh, right_ids, delta)?;
    let left_bounds = bvh
        .nodes
        .get(left as usize)
        .map(|node| node.bounds)
        .ok_or_else(|| format!("left node {} out of range", left))?;
    let right_bounds = bvh
        .nodes
        .get(right as usize)
        .map(|node| node.bounds)
        .ok_or_else(|| format!("right node {} out of range", right))?;
    allocate_node_slot(
        bvh,
        RenderBvhNode {
            bounds: union_bounds(left_bounds, right_bounds),
            kind: RenderBvhNodeKind::Internal { left, right },
        },
        delta,
    )
}

fn collect_subtree_node_leaf_ids(
    bvh: &RenderBvh,
    root: u32,
    out_nodes: &mut Vec<u32>,
    out_leaves: &mut Vec<u32>,
) {
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        let idx = node_id as usize;
        let Some(node) = bvh.nodes.get(idx) else {
            continue;
        };
        out_nodes.push(node_id);
        match node.kind {
            RenderBvhNodeKind::Internal { left, right } => {
                stack.push(right);
                stack.push(left);
            }
            RenderBvhNodeKind::Leaf { leaf_index } => {
                out_leaves.push(leaf_index);
            }
        }
    }
}

fn release_retired_nodes_and_leaves(
    bvh: &mut RenderBvh,
    mut retired_node_ids: Vec<u32>,
    mut retired_leaf_ids: Vec<u32>,
    delta: &mut RenderBvhChunkMutationDelta,
) {
    retired_node_ids.sort_unstable();
    retired_node_ids.dedup();
    for node_id in retired_node_ids {
        let node_idx = node_id as usize;
        if node_idx >= bvh.nodes.len() {
            continue;
        }
        if bvh.free_node_ids.contains(&node_id) {
            continue;
        }
        bvh.free_node_ids.push(node_id);
        bvh.nodes[node_idx] = RenderBvhNode {
            bounds: Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]),
            kind: RenderBvhNodeKind::Leaf {
                leaf_index: u32::MAX,
            },
        };
        delta.freed_node_ids.push(node_id);
    }

    retired_leaf_ids.sort_unstable();
    retired_leaf_ids.dedup();
    for leaf_id in retired_leaf_ids {
        let leaf_idx = leaf_id as usize;
        if leaf_idx >= bvh.leaves.len() {
            continue;
        }
        if bvh.free_leaf_ids.contains(&leaf_id) {
            continue;
        }
        bvh.free_leaf_ids.push(leaf_id);
        bvh.leaves[leaf_idx] = RenderLeaf {
            bounds: Aabb4i::new([0, 0, 0, 0], [-1, -1, -1, -1]),
            kind: RenderLeafKind::Uniform(0),
        };
        delta.freed_leaf_ids.push(leaf_id);
    }
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

/// Compute the 4D analogue of "half surface area" for the SAH cost model.
/// For a 4D AABB with extents [dx, dy, dz, dw], this is the sum of all
/// 3-face volumes: dy*dz*dw + dx*dz*dw + dx*dy*dw + dx*dy*dz.
/// This measures how likely a random ray is to intersect the box.
fn half_surface_area_4d(bounds: Aabb4i) -> f64 {
    let dx = f64::from(bounds.max[0] - bounds.min[0] + 1).max(0.0);
    let dy = f64::from(bounds.max[1] - bounds.min[1] + 1).max(0.0);
    let dz = f64::from(bounds.max[2] - bounds.min[2] + 1).max(0.0);
    let dw = f64::from(bounds.max[3] - bounds.min[3] + 1).max(0.0);
    dy * dz * dw + dx * dz * dw + dx * dy * dw + dx * dy * dz
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
    fn bvh_chunk_mutation_delta_reports_root_change_for_outside_insert() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Branch(vec![RenderTreeCore {
                bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                kind: RenderNodeKind::Uniform(7),
            }]),
        };
        let mut bvh = build_bvh_in_bounds(&core, bounds);
        let old_root = bvh.root;
        let deltas = apply_chunk_payload_mutations_with_deltas_in_bvh(
            &mut bvh,
            &[([2, 0, 0, 0], Some(ChunkPayload::Uniform(9)))],
        )
        .expect("apply mutation");
        assert_eq!(deltas.len(), 1);
        let delta = &deltas[0];
        assert_eq!(delta.key, [2, 0, 0, 0]);
        assert_eq!(delta.expected_root, old_root);
        assert_eq!(delta.new_root, bvh.root);
        assert!(delta.root_changed());
        assert!(!delta.node_writes.is_empty());
        assert!(!delta.leaf_writes.is_empty());
    }

    #[test]
    fn bvh_chunk_mutation_delta_reports_touched_ancestors() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Branch(vec![
                RenderTreeCore {
                    bounds: Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(7),
                },
                RenderTreeCore {
                    bounds: Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]),
                    kind: RenderNodeKind::Uniform(7),
                },
            ]),
        };
        let mut bvh = build_bvh_in_bounds(&core, bounds);
        let old_root = bvh.root.expect("root");
        let deltas =
            apply_chunk_payload_mutations_with_deltas_in_bvh(&mut bvh, &[([0, 0, 0, 0], None)])
                .expect("apply mutation");
        assert_eq!(deltas.len(), 1);
        let delta = &deltas[0];
        assert_eq!(delta.key, [0, 0, 0, 0]);
        assert_eq!(delta.expected_root, Some(old_root));
        assert!(
            !delta.node_writes.is_empty()
                || !delta.freed_node_ids.is_empty()
                || delta.root_changed()
        );
        assert_eq!(delta.new_root, bvh.root);
    }

    #[test]
    fn bvh_patch_reuses_freed_ids_instead_of_append_only_growth() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
        let core = RenderTreeCore {
            bounds,
            kind: RenderNodeKind::Uniform(7),
        };
        let mut bvh = build_bvh_in_bounds(&core, bounds);
        let old_node_capacity = bvh.nodes.len();
        let old_leaf_capacity = bvh.leaves.len();

        let remove =
            apply_chunk_payload_mutations_with_deltas_in_bvh(&mut bvh, &[([1, 0, 0, 0], None)])
                .expect("remove patch");
        assert_eq!(remove.len(), 1);
        assert!(
            !remove[0].freed_node_ids.is_empty() || !remove[0].freed_leaf_ids.is_empty(),
            "expected removal to release at least one id"
        );
        let free_node_count_after_remove = bvh.free_node_ids.len();
        let free_leaf_count_after_remove = bvh.free_leaf_ids.len();
        assert!(free_node_count_after_remove > 0 || free_leaf_count_after_remove > 0);

        let insert = apply_chunk_payload_mutations_with_deltas_in_bvh(
            &mut bvh,
            &[([1, 0, 0, 0], Some(ChunkPayload::Uniform(9)))],
        )
        .expect("insert patch");
        assert_eq!(insert.len(), 1);
        let reused_node_slot = insert[0]
            .node_writes
            .iter()
            .any(|(node_id, _)| remove[0].freed_node_ids.contains(node_id));
        let reused_leaf_slot = insert[0]
            .leaf_writes
            .iter()
            .any(|(leaf_id, _)| remove[0].freed_leaf_ids.contains(leaf_id));
        assert!(
            reused_node_slot || reused_leaf_slot,
            "expected insertion patch to reuse at least one freed slot"
        );
        assert!(bvh.nodes.len() >= old_node_capacity);
        assert!(bvh.leaves.len() >= old_leaf_capacity);
        // The SAH rebuild needs 2N-1 nodes for N leaves while the two input
        // subtrees provide 2N-2 (one short since the old join node doesn't
        // exist). Allow a small margin for this extra allocation.
        assert!(
            bvh.free_node_ids.len() <= free_node_count_after_remove + 2,
            "free_node_ids grew too much: {} (was {})",
            bvh.free_node_ids.len(),
            free_node_count_after_remove,
        );
        assert!(
            bvh.free_leaf_ids.len() <= free_leaf_count_after_remove + 2,
            "free_leaf_ids grew too much: {} (was {})",
            bvh.free_leaf_ids.len(),
            free_leaf_count_after_remove,
        );
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
