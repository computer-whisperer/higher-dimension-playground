use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RenderBvh {
    pub bounds: Aabb4i,
    pub root: Option<u32>,
    pub nodes: Vec<RenderBvhNode>,
    pub leaves: Vec<RenderLeaf>,
    free_node_ids: Vec<u32>,
    free_leaf_ids: Vec<u32>,
}

impl RenderBvh {
    #[cfg(test)]
    pub(super) fn free_node_count(&self) -> usize {
        self.free_node_ids.len()
    }

    #[cfg(test)]
    pub(super) fn free_leaf_count(&self) -> usize {
        self.free_leaf_ids.len()
    }
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
    Uniform(BlockData),
    VoxelChunkArray(ChunkArrayData),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DebugRayBvhNodeKind {
    Internal,
    LeafUniform { block: BlockData },
    LeafChunkArray { scale_exp: i8 },
}

#[derive(Clone, Debug, PartialEq)]
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

pub fn validate_render_core_world_space_non_overlapping(
    core: &RenderTreeCore,
) -> Result<(), String> {
    let region_core = to_region_core(core);
    validate_region_core_world_space_non_overlapping(&region_core)
}

fn empty_render_bvh(bounds: Aabb4i) -> RenderBvh {
    RenderBvh {
        bounds,
        root: None,
        nodes: Vec::new(),
        leaves: Vec::new(),
        free_node_ids: Vec::new(),
        free_leaf_ids: Vec::new(),
    }
}

pub fn build_bvh_in_bounds(core: &RenderTreeCore, bounds: Aabb4i) -> RenderBvh {
    if !bounds.is_valid() {
        return empty_render_bvh(bounds);
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
    let root = build_bvh_from_render_tree(core, bounds, &mut bvh, &mut init_delta);
    bvh.root = root;
    bvh
}

/// Build a BVH by directly converting the render tree structure, preserving
/// its spatial partition hierarchy. Branch nodes are binarized using SAH on
/// the children (not flattened leaves), which guarantees zero sibling overlaps
/// since the children of any Branch are non-overlapping by construction.
fn build_bvh_from_render_tree(
    core: &RenderTreeCore,
    bounds: Aabb4i,
    bvh: &mut RenderBvh,
    delta: &mut RenderBvhChunkMutationDelta,
) -> Option<u32> {
    let Some(clipped) = core.bounds.intersection(&bounds) else {
        return None;
    };
    match &core.kind {
        RenderNodeKind::Empty => None,
        RenderNodeKind::Uniform(block) => {
            if block.is_air() {
                return None;
            }
            let leaf_id = allocate_leaf_slot(
                bvh,
                RenderLeaf {
                    bounds: clipped,
                    kind: RenderLeafKind::Uniform(block.clone()),
                },
                delta,
            )
            .expect("allocate uniform leaf");
            Some(
                allocate_node_slot(
                    bvh,
                    RenderBvhNode {
                        bounds: clipped,
                        kind: RenderBvhNodeKind::Leaf { leaf_index: leaf_id },
                    },
                    delta,
                )
                .expect("allocate leaf node"),
            )
        }
        RenderNodeKind::VoxelChunkArray(chunk_array) => {
            let leaf_id = allocate_leaf_slot(
                bvh,
                RenderLeaf {
                    bounds: clipped,
                    kind: RenderLeafKind::VoxelChunkArray(chunk_array.clone()),
                },
                delta,
            )
            .expect("allocate chunk array leaf");
            Some(
                allocate_node_slot(
                    bvh,
                    RenderBvhNode {
                        bounds: clipped,
                        kind: RenderBvhNodeKind::Leaf { leaf_index: leaf_id },
                    },
                    delta,
                )
                .expect("allocate leaf node"),
            )
        }
        RenderNodeKind::Branch(children) => {
            let child_roots: Vec<u32> = children
                .iter()
                .filter_map(|child| build_bvh_from_render_tree(child, clipped, bvh, delta))
                .collect();
            if child_roots.is_empty() {
                None
            } else {
                Some(binarize_subtree_roots(bvh, &child_roots, delta))
            }
        }
    }
}

/// Given a set of BVH subtree roots whose bounds are mutually non-overlapping
/// (from a spatial partition), produce a single binary BVH subtree using SAH
/// to choose the best partition at each level.
///
/// Because the input roots have non-overlapping bounds, the left and right
/// groups at each split also have non-overlapping union bounds — there exists
/// a separating hyperplane along the chosen axis.
fn binarize_subtree_roots(
    bvh: &mut RenderBvh,
    roots: &[u32],
    delta: &mut RenderBvhChunkMutationDelta,
) -> u32 {
    debug_assert!(!roots.is_empty());
    if roots.len() == 1 {
        return roots[0];
    }
    if roots.len() == 2 {
        let bounds = bvh.nodes[roots[0] as usize]
            .bounds
            .union(&bvh.nodes[roots[1] as usize].bounds);
        return allocate_node_slot(
            bvh,
            RenderBvhNode {
                bounds,
                kind: RenderBvhNodeKind::Internal {
                    left: roots[0],
                    right: roots[1],
                },
            },
            delta,
        )
        .expect("allocate internal node");
    }

    // SAH sweep over all 4 axes to find the best binary partition.
    let n = roots.len();
    let root_bounds: Vec<Aabb4i> = roots.iter().map(|&r| bvh.nodes[r as usize].bounds).collect();

    let mut best_axis = 0usize;
    let mut best_split = n / 2;
    let mut best_cost = f64::MAX;

    let mut indices: Vec<usize> = (0..n).collect();
    let mut right_bounds_suffix = vec![Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]); n];

    for axis in 0..4 {
        indices.sort_unstable_by(|&a, &b| {
            let ca = centroid_axis(root_bounds[a], axis);
            let cb = centroid_axis(root_bounds[b], axis);
            ca.total_cmp(&cb)
                .then_with(|| root_bounds[a].min.cmp(&root_bounds[b].min))
                .then_with(|| root_bounds[a].max.cmp(&root_bounds[b].max))
        });

        right_bounds_suffix[n - 1] = root_bounds[indices[n - 1]];
        for i in (0..n - 1).rev() {
            right_bounds_suffix[i] = right_bounds_suffix[i + 1].union(&root_bounds[indices[i]]);
        }

        let mut left_agg = root_bounds[indices[0]];
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
                left_agg = left_agg.union(&root_bounds[indices[split]]);
            }
        }
    }

    // Partition roots by the best axis and split position.
    let mut sorted_roots: Vec<u32> = roots.to_vec();
    sorted_roots.sort_unstable_by(|&a, &b| {
        let ab = bvh.nodes[a as usize].bounds;
        let bb = bvh.nodes[b as usize].bounds;
        let ca = centroid_axis(ab, best_axis);
        let cb = centroid_axis(bb, best_axis);
        ca.total_cmp(&cb)
            .then_with(|| ab.min.cmp(&bb.min))
            .then_with(|| ab.max.cmp(&bb.max))
    });

    let (left_roots, right_roots) = sorted_roots.split_at(best_split);
    let left = binarize_subtree_roots(bvh, left_roots, delta);
    let right = binarize_subtree_roots(bvh, right_roots, delta);
    let bounds = bvh.nodes[left as usize]
        .bounds
        .union(&bvh.nodes[right as usize].bounds);
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

pub fn sample_chunk_payloads_from_bvh(bvh: &RenderBvh, key: ChunkKey) -> Vec<ResolvedChunkPayload> {
    if !bvh.bounds.contains_chunk_world_min(key) {
        return Vec::new();
    }
    let Some(root) = bvh.root else {
        return Vec::new();
    };

    let mut out = Vec::<ResolvedChunkPayload>::new();
    let mut stack = Vec::<u32>::with_capacity(64);
    stack.push(root);

    while let Some(node_idx) = stack.pop() {
        let Some(node) = bvh.nodes.get(node_idx as usize) else {
            continue;
        };
        if !node.bounds.contains_chunk_world_min(key) {
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

pub fn apply_chunk_payload_mutations_with_deltas_in_bvh(
    bvh: &mut RenderBvh,
    mutations: &[(ChunkKey, Option<ChunkPayload>, Vec<BlockData>)],
) -> Result<Vec<RenderBvhChunkMutationDelta>, String> {
    let mut deltas = Vec::<RenderBvhChunkMutationDelta>::new();
    for (key, payload, block_palette) in mutations {
        if !bvh.bounds.contains_chunk_world_min(*key) {
            continue;
        }
        let key_bounds = Aabb4i::chunk_world_bounds(*key, 0);
        let patch_core = if let Some(payload) = payload.clone() {
            repeated_voxel_leaf(key_bounds, payload, block_palette)
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
    let Some(clipped_patch_bounds) = bvh.bounds.intersection(&patch_bounds) else {
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

    // Build replacement subtree using the partition-preserving binarizer,
    // which walks the render tree structure directly instead of flattening
    // to leaves + SAH.
    let replacement_root =
        build_bvh_from_render_tree(patch_core, clipped_patch_bounds, bvh, &mut delta);

    let new_root = match (outside_root, replacement_root) {
        (None, None) => None,
        (Some(root), None) | (None, Some(root)) => Some(root),
        (Some(left), Some(right)) => {
            // Collect all leaf IDs from both subtrees and rebuild with SAH
            // for optimal traversal structure. SAH minimizes expected ray
            // intersection cost via the surface area heuristic — its small
            // sibling overlaps are far less costly than suboptimal depth.
            let mut left_nodes = Vec::new();
            let mut left_leaf_ids = Vec::new();
            collect_subtree_node_leaf_ids(bvh, left, &mut left_nodes, &mut left_leaf_ids);

            let mut right_nodes = Vec::new();
            let mut right_leaf_ids = Vec::new();
            collect_subtree_node_leaf_ids(bvh, right, &mut right_nodes, &mut right_leaf_ids);

            // Recycle node slots into the CPU free list so the SAH rebuild
            // can reuse them. We do NOT add them to delta.freed_node_ids
            // because the rebuild will immediately rewrite them (via
            // delta.node_writes). The GPU consumer applies writes to these
            // slots, so they remain valid in the GPU buffer.
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
                match &leaf.kind {
                    RenderLeafKind::Uniform(block) => DebugRayBvhNodeKind::LeafUniform {
                        block: block.clone(),
                    },
                    RenderLeafKind::VoxelChunkArray(ca) => DebugRayBvhNodeKind::LeafChunkArray {
                        scale_exp: ca.scale_exp,
                    },
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

fn sample_leaf_payload_at_key(leaf: &RenderLeaf, key: ChunkKey) -> Option<ResolvedChunkPayload> {
    if !leaf.bounds.contains_chunk_world_min(key) {
        return None;
    }
    match &leaf.kind {
        RenderLeafKind::Uniform(block) => Some(ResolvedChunkPayload::uniform(block.clone())),
        RenderLeafKind::VoxelChunkArray(chunk_array) => {
            sample_chunkarray_payload_at_key(chunk_array, key)
        }
    }
}

fn sample_chunkarray_payload_at_key(
    chunk_array: &ChunkArrayData,
    key: ChunkKey,
) -> Option<ResolvedChunkPayload> {
    if !chunk_array.bounds.contains_chunk_world_min(key) {
        return None;
    }
    let se = chunk_array.scale_exp;
    let dims = chunk_array.bounds.chunk_extents_at_scale(se)?;
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    let (arr_min, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    let lx = (lattice_from_fixed(key[0], se) - arr_min[0]) as usize;
    let ly = (lattice_from_fixed(key[1], se) - arr_min[1]) as usize;
    let lz = (lattice_from_fixed(key[2], se) - arr_min[2]) as usize;
    let lw = (lattice_from_fixed(key[3], se) - arr_min[3]) as usize;
    let linear = lx + dims[0] * (ly + dims[1] * (lz + dims[2] * lw));
    let palette_idx = *dense_indices.get(linear)? as usize;
    let payload = chunk_array.chunk_palette.get(palette_idx)?.clone();
    Some(ResolvedChunkPayload {
        payload,
        block_palette: chunk_array.block_palette.clone(),
    })
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
    if excluded_bounds.contains_bounds(&node.bounds) {
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
                    let bounds = bvh.nodes[new_left as usize]
                        .bounds
                        .union(&bvh.nodes[new_right as usize].bounds);
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
        RenderLeafKind::Uniform(block) => {
            if block.is_air() {
                return Ok(out);
            }
            for piece_bounds in split_bounds {
                out.push(RenderLeaf {
                    bounds: piece_bounds,
                    kind: RenderLeafKind::Uniform(block.clone()),
                });
            }
        }
        RenderLeafKind::VoxelChunkArray(chunk_array) => {
            let se = chunk_array.scale_exp;
            let source_indices = chunk_array
                .decode_dense_indices()
                .map_err(|error| format!("decode chunk-array leaf failed: {error:?}"))?;
            for piece_bounds in split_bounds {
                // Verify piece bounds align to the chunk grid at this scale.
                // Misaligned bounds (e.g. [1,8) at scale=0) cannot be correctly
                // encoded for the GPU — the chunk world-space origins would fall
                // outside the piece bounds, producing empty entries. Signal an
                // error so the caller falls back to a full rebuild from the
                // authoritative render region cache.
                let (lmin, lmax) = piece_bounds.to_chunk_lattice_bounds(se);
                if Aabb4i::from_lattice_bounds(lmin, lmax, se) != piece_bounds {
                    return Err(format!(
                        "remnant bounds {:?}->{:?} misaligned at scale_exp={}, needs full rebuild",
                        piece_bounds.min, piece_bounds.max, se
                    ));
                }
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
    let Some(clipped_excluded) = bounds.intersection(&excluded_bounds) else {
        return vec![bounds];
    };

    let mut rem_min = bounds.min;
    let mut rem_max = bounds.max;
    let mut pieces = Vec::<Aabb4i>::new();
    for axis in 0..4 {
        if rem_min[axis] < clipped_excluded.min[axis] {
            let mut piece_max = rem_max;
            piece_max[axis] = clipped_excluded.min[axis];
            pieces.push(Aabb4i::new(rem_min, piece_max));
            rem_min[axis] = clipped_excluded.min[axis];
        }
        if rem_max[axis] > clipped_excluded.max[axis] {
            let mut piece_min = rem_min;
            piece_min[axis] = clipped_excluded.max[axis];
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
    let Some(clipped_bounds) = bounds.intersection(&chunk_array.bounds) else {
        return Ok(None);
    };
    if clipped_bounds != bounds {
        return Ok(None);
    }

    let se = chunk_array.scale_exp;
    let Some(src_dims) = chunk_array.bounds.chunk_extents_at_scale(se) else {
        return Ok(None);
    };
    let Some(piece_dims) = bounds.chunk_extents_at_scale(se) else {
        return Ok(None);
    };
    let piece_cell_count = bounds
        .chunk_cell_count_at_scale(se)
        .ok_or_else(|| "piece chunk count overflow".to_string())?;
    let mut piece_indices = Vec::<u16>::with_capacity(piece_cell_count);
    let palette_non_empty: Vec<bool> = chunk_array
        .chunk_palette
        .iter()
        .map(|p| chunk_payload_has_solid_material_in_context(p, &chunk_array.block_palette))
        .collect();
    let mut any_non_empty = false;

    let (piece_lat_min, _) = bounds.to_chunk_lattice_bounds(se);
    let (arr_lat_min, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    for w in 0..piece_dims[3] {
        for z in 0..piece_dims[2] {
            for y in 0..piece_dims[1] {
                for x in 0..piece_dims[0] {
                    let lx = ((piece_lat_min[0] + x as i32) - arr_lat_min[0]) as usize;
                    let ly = ((piece_lat_min[1] + y as i32) - arr_lat_min[1]) as usize;
                    let lz = ((piece_lat_min[2] + z as i32) - arr_lat_min[2]) as usize;
                    let lw = ((piece_lat_min[3] + w as i32) - arr_lat_min[3]) as usize;
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

    let piece_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        bounds,
        chunk_array.chunk_palette.clone(),
        piece_indices,
        None,
        chunk_array.block_palette.clone(),
        se,
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
                .unwrap_or(Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]))
        })
        .collect();

    // We need to try sorting by each axis independently. Since leaf_ids is mutable
    // and we need to try 4 axes, work with index arrays.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut right_bounds_suffix = vec![Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]); n];

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
            right_bounds_suffix[i] = right_bounds_suffix[i + 1].union(&leaf_bounds[indices[i]]);
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
                left_agg = left_agg.union(&leaf_bounds[indices[split]]);
            }
        }
    }

    // Now apply the best split: sort leaf_ids by the chosen axis and split.
    leaf_ids.sort_unstable_by(|a, b| {
        let ab = bvh
            .leaves
            .get(*a as usize)
            .map(|leaf| leaf.bounds)
            .unwrap_or(Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]));
        let bb = bvh
            .leaves
            .get(*b as usize)
            .map(|leaf| leaf.bounds)
            .unwrap_or(Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]));
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
            bounds: left_bounds.union(&right_bounds),
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
        if let Err(pos) = bvh.free_node_ids.binary_search(&node_id) {
            bvh.free_node_ids.insert(pos, node_id);
        } else {
            continue; // already free
        }
        bvh.nodes[node_idx] = RenderBvhNode {
            bounds: Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]),
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
        if let Err(pos) = bvh.free_leaf_ids.binary_search(&leaf_id) {
            bvh.free_leaf_ids.insert(pos, leaf_id);
        } else {
            continue; // already free
        }
        bvh.leaves[leaf_idx] = RenderLeaf {
            bounds: Aabb4i::from_i32([0, 0, 0, 0], [-1, -1, -1, -1]),
            kind: RenderLeafKind::Uniform(BlockData::AIR),
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
    let min = [
        chunk_bounds.min[0].to_num::<f32>(),
        chunk_bounds.min[1].to_num::<f32>(),
        chunk_bounds.min[2].to_num::<f32>(),
        chunk_bounds.min[3].to_num::<f32>(),
    ];
    let max = [
        chunk_bounds.max[0].to_num::<f32>(),
        chunk_bounds.max[1].to_num::<f32>(),
        chunk_bounds.max[2].to_num::<f32>(),
        chunk_bounds.max[3].to_num::<f32>(),
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
    let Some(clipped_bounds) = leaf.bounds.intersection(&bounds) else {
        return;
    };
    match &leaf.kind {
        RenderLeafKind::Uniform(block) => {
            if block.is_air() {
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
    let Some(clipped_bounds) = bounds.intersection(&chunk_array.bounds) else {
        return;
    };
    let se = chunk_array.scale_exp;
    let Some(dims) = chunk_array.bounds.chunk_extents_at_scale(se) else {
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
        .map(|p| chunk_payload_has_solid_material_in_context(p, &chunk_array.block_palette))
        .collect();

    let (cb_min, cb_max) = clipped_bounds.to_chunk_lattice_bounds(se);
    let (arr_min, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    for iw in cb_min[3]..=cb_max[3] {
        for iz in cb_min[2]..=cb_max[2] {
            for iy in cb_min[1]..=cb_max[1] {
                for ix in cb_min[0]..=cb_max[0] {
                    let lx = (ix - arr_min[0]) as usize;
                    let ly = (iy - arr_min[1]) as usize;
                    let lz = (iz - arr_min[2]) as usize;
                    let lw = (iw - arr_min[3]) as usize;
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
                        let x = fixed_from_lattice(ix, se);
                        let y = fixed_from_lattice(iy, se);
                        let z = fixed_from_lattice(iz, se);
                        let w = fixed_from_lattice(iw, se);
                        out.push([x, y, z, w]);
                    }
                }
            }
        }
    }
}

fn chunk_payload_has_solid_material_in_context(
    payload: &ChunkPayload,
    block_palette: &[BlockData],
) -> bool {
    let idx_is_solid = |idx: &u16| -> bool {
        block_palette
            .get(*idx as usize)
            .map(|b| !b.is_air())
            .unwrap_or(false)
    };
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(idx) => idx_is_solid(idx),
        ChunkPayload::Dense16 { materials } => materials.iter().any(idx_is_solid),
        ChunkPayload::PalettePacked { palette, .. } => palette.iter().any(idx_is_solid),
    }
}

fn enumerate_chunk_keys(bounds: Aabb4i, out: &mut Vec<ChunkKey>) {
    let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
    for iw in bmin[3]..=bmax[3] {
        for iz in bmin[2]..=bmax[2] {
            for iy in bmin[1]..=bmax[1] {
                for ix in bmin[0]..=bmax[0] {
                    out.push([
                        ChunkCoord::from_num(ix),
                        ChunkCoord::from_num(iy),
                        ChunkCoord::from_num(iz),
                        ChunkCoord::from_num(iw),
                    ]);
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
    let dx = (bounds.max[0] - bounds.min[0]).to_num::<f64>().max(0.0);
    let dy = (bounds.max[1] - bounds.min[1]).to_num::<f64>().max(0.0);
    let dz = (bounds.max[2] - bounds.min[2]).to_num::<f64>().max(0.0);
    let dw = (bounds.max[3] - bounds.min[3]).to_num::<f64>().max(0.0);
    dy * dz * dw + dx * dz * dw + dx * dy * dw + dx * dy * dz
}

fn centroid_axis(bounds: Aabb4i, axis: usize) -> f64 {
    (bounds.min[axis].to_num::<f64>() + bounds.max[axis].to_num::<f64>()) * 0.5
}
