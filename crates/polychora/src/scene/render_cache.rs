use super::*;
use higher_dimension_playground::render::{
    VTE_LEAF_KIND_UNIFORM, VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY, VTE_REGION_BVH_NODE_FLAG_LEAF,
};
use polychora::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;

#[derive(Default)]
struct RegionDumpStats {
    total_nodes: usize,
    empty: usize,
    uniform: usize,
    procedural: usize,
    chunk_array: usize,
    branch: usize,
    max_depth: usize,
}

#[derive(Default)]
struct GpuBvhDumpStats {
    internal_count: usize,
    leaf_count: usize,
    uniform_leaves: usize,
    chunk_array_leaves: usize,
    unknown_leaves: usize,
    max_depth: usize,
    sibling_overlaps: usize,
}

impl Scene {
    fn build_render_region_tree_in_bounds(&self, bounds: Aabb4i) -> RegionChunkTree {
        let mut composed = RegionChunkTree::new();
        if !bounds.is_valid() {
            return composed;
        }
        let world_core = self.world_tree.slice_non_empty_core_in_bounds(bounds);
        let _ = composed.splice_non_empty_core_in_bounds(bounds, &world_core);
        composed
    }

    fn build_render_tree_core_in_bounds(&self, bounds: Aabb4i) -> RenderTreeCore {
        if !bounds.is_valid() {
            return RenderTreeCore::empty(bounds);
        }
        let composed = self.build_render_region_tree_in_bounds(bounds);
        let composed_core = composed.slice_non_empty_core_in_bounds(bounds);
        render_tree::from_region_core(&composed_core)
    }

    fn rebuild_render_bvh_from_region_cache(&mut self, bounds: Aabb4i) {
        let render_core = if let Some(cache) = self.render_region_cache.as_ref() {
            let core = cache.slice_non_empty_core_in_bounds(bounds);
            render_tree::from_region_core(&core)
        } else {
            self.build_render_tree_core_in_bounds(bounds)
        };
        self.render_bvh_cache = Some(render_tree::build_bvh_in_bounds(&render_core, bounds));
        self.render_bvh_cache_bounds = Some(bounds);
    }

    fn intersect_bounds(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
        if !a.is_valid() || !b.is_valid() || !a.intersects(&b) {
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

    fn merge_bounds(a: Aabb4i, b: Aabb4i) -> Aabb4i {
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

    pub(crate) fn bounds_contains_bounds(outer: Aabb4i, inner: Aabb4i) -> bool {
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

    /// Dump the region cache tree (client-side) and the on-GPU BVH buffer
    /// structure to stderr for structural analysis.
    pub fn dump_render_trees(&self) {
        self.dump_region_cache_tree();
        self.dump_gpu_bvh();
    }

    fn dump_region_cache_tree(&self) {
        fn bounds_size(b: &Aabb4i) -> [i32; 4] {
            [
                (b.max[0] - b.min[0]).to_num::<i32>(),
                (b.max[1] - b.min[1]).to_num::<i32>(),
                (b.max[2] - b.min[2]).to_num::<i32>(),
                (b.max[3] - b.min[3]).to_num::<i32>(),
            ]
        }

        eprintln!("\n=== REGION CACHE TREE ===");
        if let Some(bounds) = self.render_region_cache_bounds {
            eprintln!(
                "  cache bounds: {:?} -> {:?}  size: {:?}",
                bounds.min, bounds.max, bounds_size(&bounds),
            );
        }
        if let Some(cache) = self.render_region_cache.as_ref() {
            if let Some(root) = cache.root() {
                let mut stats = RegionDumpStats::default();
                Self::dump_region_node(root, 0, &mut stats);
                eprintln!(
                    "--- region summary: {} nodes (empty={} uniform={} procedural={} chunk_array={} branch={}), max_depth={}",
                    stats.total_nodes,
                    stats.empty,
                    stats.uniform,
                    stats.procedural,
                    stats.chunk_array,
                    stats.branch,
                    stats.max_depth,
                );
            } else {
                eprintln!("  (empty root)");
            }
        } else {
            eprintln!("  (no region cache)");
        }
        eprintln!("=== END REGION CACHE TREE ===\n");
    }

    fn dump_region_node(
        node: &RegionTreeCore,
        depth: usize,
        stats: &mut RegionDumpStats,
    ) {
        fn bounds_size(b: &Aabb4i) -> [i32; 4] {
            [
                (b.max[0] - b.min[0]).to_num::<i32>(),
                (b.max[1] - b.min[1]).to_num::<i32>(),
                (b.max[2] - b.min[2]).to_num::<i32>(),
                (b.max[3] - b.min[3]).to_num::<i32>(),
            ]
        }

        let indent = "  ".repeat(depth + 1);
        let size = bounds_size(&node.bounds);
        stats.total_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);

        match &node.kind {
            RegionNodeKind::Empty => {
                stats.empty += 1;
                // Only print shallow empties to avoid flooding.
                if depth <= 3 {
                    eprintln!(
                        "{}Empty  bounds={:?}->{:?} size={:?}",
                        indent, node.bounds.min, node.bounds.max, size,
                    );
                }
            }
            RegionNodeKind::Uniform(block) => {
                stats.uniform += 1;
                eprintln!(
                    "{}Uniform(ns={},bt={})  bounds={:?}->{:?} size={:?}",
                    indent, block.namespace, block.block_type, node.bounds.min, node.bounds.max, size,
                );
            }
            RegionNodeKind::ProceduralRef(gen) => {
                stats.procedural += 1;
                eprintln!(
                    "{}ProceduralRef(gen={})  bounds={:?}->{:?} size={:?}",
                    indent, gen.generator_id, node.bounds.min, node.bounds.max, size,
                );
            }
            RegionNodeKind::ChunkArray(_) => {
                stats.chunk_array += 1;
                eprintln!(
                    "{}ChunkArray  bounds={:?}->{:?} size={:?}",
                    indent, node.bounds.min, node.bounds.max, size,
                );
            }
            RegionNodeKind::Branch(children) => {
                stats.branch += 1;
                eprintln!(
                    "{}Branch({} children)  bounds={:?}->{:?} size={:?}",
                    indent,
                    children.len(),
                    node.bounds.min,
                    node.bounds.max,
                    size,
                );
                for child in children {
                    Self::dump_region_node(child, depth + 1, stats);
                }
            }
        }
    }

    /// Dump the on-GPU BVH buffer structure to stderr.
    /// Walks `voxel_frame_data.region_bvh_nodes` (the actual GPU-side
    /// `GpuVoxelChunkBvhNode` array) and `leaf_headers`, showing the tree
    /// the shader traverses.
    fn dump_gpu_bvh(&self) {
        let fd = &self.voxel_frame_data;
        let nodes = &fd.region_bvh_nodes;
        let leaves = &fd.leaf_headers;
        let root_idx = fd.region_bvh_root_index;

        eprintln!("\n=== GPU BVH BUFFER DUMP ===");
        eprintln!(
            "  root_index: {}  node_count: {}  leaf_count: {}",
            if root_idx == VTE_REGION_BVH_INVALID_NODE {
                "INVALID".to_string()
            } else {
                root_idx.to_string()
            },
            nodes.len(),
            leaves.len(),
        );

        if root_idx == VTE_REGION_BVH_INVALID_NODE || root_idx as usize >= nodes.len() {
            eprintln!("  (no valid root)");
            eprintln!("=== END GPU BVH DUMP ===\n");
            return;
        }

        let mut stats = GpuBvhDumpStats::default();
        Self::dump_gpu_bvh_node(nodes, leaves, root_idx, 0, &mut stats);

        eprintln!(
            "--- gpu bvh summary: {} internal + {} leaves (uniform={} chunk_array={} unknown={}), max_depth={}, sibling_overlaps={}",
            stats.internal_count,
            stats.leaf_count,
            stats.uniform_leaves,
            stats.chunk_array_leaves,
            stats.unknown_leaves,
            stats.max_depth,
            stats.sibling_overlaps,
        );
        eprintln!("=== END GPU BVH DUMP ===\n");
    }

    fn dump_gpu_bvh_node(
        nodes: &[GpuVoxelChunkBvhNode],
        leaves: &[GpuVoxelLeafHeader],
        node_idx: u32,
        depth: usize,
        stats: &mut GpuBvhDumpStats,
    ) {
        if node_idx == VTE_REGION_BVH_INVALID_NODE || node_idx as usize >= nodes.len() {
            return;
        }
        let node = &nodes[node_idx as usize];
        let indent = "  ".repeat(depth + 1);
        let size = [
            node.world_max[0] - node.world_min[0],
            node.world_max[1] - node.world_min[1],
            node.world_max[2] - node.world_min[2],
            node.world_max[3] - node.world_min[3],
        ];
        stats.max_depth = stats.max_depth.max(depth);

        let is_leaf = (node.flags & VTE_REGION_BVH_NODE_FLAG_LEAF) != 0;

        if is_leaf {
            stats.leaf_count += 1;
            let leaf_idx = node.leaf_index;
            let leaf_label = if leaf_idx < leaves.len() as u32 {
                let lh = &leaves[leaf_idx as usize];
                if lh.leaf_kind == VTE_LEAF_KIND_UNIFORM {
                    stats.uniform_leaves += 1;
                    format!("Uniform(mat={})", lh.uniform_material)
                } else if lh.leaf_kind == VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY {
                    stats.chunk_array_leaves += 1;
                    format!(
                        "ChunkArray(entry_off={} bounds={:?}->{:?})",
                        lh.chunk_entry_offset, lh.min_chunk_coord, lh.max_chunk_coord
                    )
                } else {
                    stats.unknown_leaves += 1;
                    format!("Unknown(kind={})", lh.leaf_kind)
                }
            } else {
                stats.unknown_leaves += 1;
                format!("INVALID_LEAF(idx={})", leaf_idx)
            };
            eprintln!(
                "{}Leaf[{}->{}] {}  bounds={:?}->{:?} size={:?}",
                indent,
                node_idx,
                leaf_idx,
                leaf_label,
                node.world_min,
                node.world_max,
                size,
            );
        } else {
            stats.internal_count += 1;
            let left = node.left_child;
            let right = node.right_child;

            // Check sibling overlap.
            let overlaps = if left != VTE_REGION_BVH_INVALID_NODE
                && right != VTE_REGION_BVH_INVALID_NODE
                && (left as usize) < nodes.len()
                && (right as usize) < nodes.len()
            {
                let l = &nodes[left as usize];
                let r = &nodes[right as usize];
                // Strict inequality for half-open [min, max) bounds — adjacent
                // non-overlapping boxes have l.world_max == r.world_min.
                let o = l.world_min[0] < r.world_max[0]
                    && l.world_max[0] > r.world_min[0]
                    && l.world_min[1] < r.world_max[1]
                    && l.world_max[1] > r.world_min[1]
                    && l.world_min[2] < r.world_max[2]
                    && l.world_max[2] > r.world_min[2]
                    && l.world_min[3] < r.world_max[3]
                    && l.world_max[3] > r.world_min[3];
                if o {
                    stats.sibling_overlaps += 1;
                }
                o
            } else {
                false
            };

            if depth <= 10 {
                eprintln!(
                    "{}Internal[{}] L={} R={}  bounds={:?}->{:?} size={:?}{}",
                    indent,
                    node_idx,
                    left,
                    right,
                    node.world_min,
                    node.world_max,
                    size,
                    if overlaps { "  OVERLAP" } else { "" },
                );
            }
            Self::dump_gpu_bvh_node(nodes, leaves, left, depth + 1, stats);
            Self::dump_gpu_bvh_node(nodes, leaves, right, depth + 1, stats);
        }
    }

    /// Force a complete render BVH rebuild from the current region cache.
    /// This discards all incremental mutation state and reconstructs a fresh
    /// optimal tree, useful for profiling BVH quality degradation.
    pub fn force_render_bvh_rebuild(&mut self) {
        if let Some(bounds) = self.render_bvh_cache_bounds {
            self.rebuild_render_bvh_from_region_cache(bounds);
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
        }
    }

    pub(crate) fn ensure_render_bvh_cache_for_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            self.render_region_cache_bounds = None;
            self.render_region_cache = None;
            self.render_bvh_cache_bounds = None;
            self.render_bvh_cache = None;
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            return;
        }

        let has_cache = self.render_bvh_cache_bounds == Some(bounds)
            && self.render_region_cache_bounds == Some(bounds)
            && self.render_region_cache.is_some();
        if !has_cache {
            self.render_region_cache_bounds = Some(bounds);
            self.render_region_cache = Some(self.build_render_region_tree_in_bounds(bounds));
            self.rebuild_render_bvh_from_region_cache(bounds);
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            self.clear_voxel_scene_dirty_regions_in_bounds(bounds);
            return;
        }
        if self.voxel_frame_data.region_bvh_root_index == VTE_REGION_BVH_INVALID_NODE
            && self
                .render_bvh_cache
                .as_ref()
                .and_then(|bvh| bvh.root)
                .is_some()
        {
            // Recover from an invalid/empty frame upload by replaying a full
            // snapshot from the already cached render BVH.
            self.voxel_pending_render_bvh_rebuild = true;
        }

        let dirty_regions = self.take_voxel_scene_dirty_regions_in_bounds(
            bounds,
            VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET,
        );
        if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some()
            && dirty_regions.is_empty()
            && !self.voxel_pending_scene_dirty_regions.is_empty()
        {
            eprintln!(
                "[edit-sync-dirty-skip] bounds={:?}->{:?} pending_total={} pending_sample={:?}->{:?}",
                bounds.min,
                bounds.max,
                self.voxel_pending_scene_dirty_regions.len(),
                self.voxel_pending_scene_dirty_regions[0].min,
                self.voxel_pending_scene_dirty_regions[0].max
            );
        }
        if dirty_regions.is_empty() {
            return;
        }

        let mut changed = false;
        let mut deltas = Vec::<RenderBvhChunkMutationDelta>::new();
        let mut fallback_full_rebuild = false;
        let mut fallback_reason: Option<String> = None;
        let dirty_region_count = dirty_regions.len();
        for dirty_region in dirty_regions {
            let Some(dirty_bounds) = Self::intersect_bounds(dirty_region, bounds) else {
                continue;
            };
            let desired_fragment_tree = self.build_render_region_tree_in_bounds(dirty_bounds);
            let desired_fragment_core =
                desired_fragment_tree.slice_non_empty_core_in_bounds(dirty_bounds);
            if let Some(cache) = self.render_region_cache.as_mut() {
                let previous_fragment_core = cache.slice_non_empty_core_in_bounds(dirty_bounds);
                let splice_changed =
                    cache.splice_non_empty_core_in_bounds(dirty_bounds, &desired_fragment_core);
                if splice_changed.is_none() {
                    let current_fragment_core = cache.slice_non_empty_core_in_bounds(dirty_bounds);
                    if current_fragment_core.kind != desired_fragment_core.kind {
                        fallback_full_rebuild = true;
                        fallback_reason = Some(format!(
                            "render_region_splice_noop_mismatch bounds={:?}->{:?} prev_kind={:?} desired_kind={:?} current_kind={:?}",
                            dirty_bounds.min,
                            dirty_bounds.max,
                            previous_fragment_core.kind,
                            desired_fragment_core.kind,
                            current_fragment_core.kind
                        ));
                        break;
                    }
                    continue;
                }
            } else {
                fallback_full_rebuild = true;
                fallback_reason = Some("missing_render_region_cache".to_string());
                break;
            }

            changed = true;
            if let Some(render_bvh) = self.render_bvh_cache.as_mut() {
                let desired_render_core = render_tree::from_region_core(&desired_fragment_core);
                match render_tree::apply_core_patch_in_bounds_with_delta_in_bvh(
                    render_bvh,
                    dirty_bounds,
                    &desired_render_core,
                ) {
                    Ok(Some(delta)) => {
                        deltas.push(delta);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        fallback_full_rebuild = true;
                        fallback_reason = Some(format!(
                            "apply_core_patch_delta_failed bounds={:?}->{:?} error={}",
                            dirty_bounds.min, dirty_bounds.max, error
                        ));
                        break;
                    }
                }
            } else {
                fallback_full_rebuild = true;
                fallback_reason = Some("missing_render_bvh_cache".to_string());
                break;
            }
        }

        if fallback_full_rebuild {
            eprintln!(
                "[vte-bvh-delta-fallback] reason={} rebuild_bounds={:?}->{:?} pending_deltas={}",
                fallback_reason.as_deref().unwrap_or("unknown"),
                bounds.min,
                bounds.max,
                deltas.len()
            );
            self.render_region_cache_bounds = Some(bounds);
            self.render_region_cache = Some(self.build_render_region_tree_in_bounds(bounds));
            self.rebuild_render_bvh_from_region_cache(bounds);
            self.voxel_pending_render_bvh_rebuild = true;
            self.voxel_pending_render_bvh_mutation_deltas.clear();
            return;
        }
        if changed {
            if deltas.is_empty() {
                // If the render-region cache changed but the BVH delta path produced
                // no writes, the GPU frame can get stuck on stale data.
                eprintln!(
                    "[vte-bvh-delta-fallback] reason=changed_cache_without_deltas rebuild_bounds={:?}->{:?}",
                    bounds.min, bounds.max
                );
                self.rebuild_render_bvh_from_region_cache(bounds);
                self.voxel_pending_render_bvh_rebuild = true;
                self.voxel_pending_render_bvh_mutation_deltas.clear();
                return;
            }
            if std::env::var_os("R4D_EDIT_RENDER_SYNC_DIAG").is_some() {
                eprintln!(
                    "[edit-sync-dirty-apply] bounds={:?}->{:?} dirty_regions={} deltas={} pending_after={}",
                    bounds.min,
                    bounds.max,
                    dirty_region_count,
                    deltas.len(),
                    self.voxel_pending_scene_dirty_regions.len()
                );
            }
            self.voxel_pending_render_bvh_mutation_deltas.extend(deltas);
        }
    }

    pub(crate) fn mark_voxel_scene_region_dirty(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            return;
        }
        let mut merged = bounds;
        self.voxel_pending_scene_dirty_regions.retain(|existing| {
            if Self::bounds_contains_bounds(*existing, merged) {
                merged = *existing;
                return true;
            }
            if Self::bounds_contains_bounds(merged, *existing) {
                return false;
            }
            if existing.intersects(&merged) {
                merged = Self::merge_bounds(*existing, merged);
                return false;
            }
            true
        });
        self.voxel_pending_scene_dirty_regions.push(merged);
    }

    pub(crate) fn voxel_scene_bounds_has_pending_dirty_regions(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        self.voxel_pending_scene_dirty_regions
            .iter()
            .any(|pending| pending.intersects(&bounds))
    }

    fn clear_voxel_scene_dirty_regions_in_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() || self.voxel_pending_scene_dirty_regions.is_empty() {
            return;
        }
        self.voxel_pending_scene_dirty_regions
            .retain(|pending| !pending.intersects(&bounds));
    }

    fn take_voxel_scene_dirty_regions_in_bounds(
        &mut self,
        bounds: Aabb4i,
        max_regions: usize,
    ) -> Vec<Aabb4i> {
        if !bounds.is_valid()
            || max_regions == 0
            || self.voxel_pending_scene_dirty_regions.is_empty()
        {
            return Vec::new();
        }

        let mut taken = Vec::<Aabb4i>::new();
        self.voxel_pending_scene_dirty_regions.retain(|pending| {
            if taken.len() < max_regions && pending.intersects(&bounds) {
                taken.push(*pending);
                false
            } else {
                true
            }
        });
        taken
    }

    pub(crate) fn take_pending_render_bvh_update_flags(
        &mut self,
    ) -> (bool, Vec<RenderBvhChunkMutationDelta>) {
        let needs_rebuild = self.voxel_pending_render_bvh_rebuild;
        self.voxel_pending_render_bvh_rebuild = false;
        let deltas = std::mem::take(&mut self.voxel_pending_render_bvh_mutation_deltas);
        (needs_rebuild, deltas)
    }

    pub fn debug_render_bvh_chunk_payloads_in_bounds(
        &mut self,
        bounds: Aabb4i,
        chunk_key: ChunkKey,
    ) -> Vec<ResolvedChunkPayload> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        self.ensure_render_bvh_cache_for_bounds(bounds);
        self.render_bvh_cache
            .as_ref()
            .map(|bvh| render_tree::sample_chunk_payloads_from_bvh(bvh, chunk_key))
            .unwrap_or_default()
    }

    /// Comprehensive render tree integrity check: validates the render region
    /// cache, CPU BVH, and GPU BVH buffers, and cross-validates against the
    /// world tree.
    pub fn check_render_tree_integrity(&self) -> RenderTreeReport {
        let mut report = RenderTreeReport::default();

        // --- Render region cache ---
        let cache_bounds = self.render_region_cache_bounds;
        report.cache_bounds = cache_bounds;
        if let Some(cache) = self.render_region_cache.as_ref() {
            report.cache_tree_report = Some(validate_tree_integrity(cache));
        }

        // --- CPU BVH ---
        if let Some(bvh) = self.render_bvh_cache.as_ref() {
            report.cpu_bvh_bounds = Some(bvh.bounds);
            report.cpu_bvh_root = bvh.root;
            report.cpu_bvh_node_count = bvh.nodes.len();
            report.cpu_bvh_leaf_count = bvh.leaves.len();

            // Walk CPU BVH and validate structure
            if let Some(root_idx) = bvh.root {
                let mut ctx = CpuBvhWalkCtx::default();
                walk_cpu_bvh_node(bvh, root_idx, 0, &mut ctx);
                report.cpu_bvh_reachable_nodes = ctx.reachable_nodes;
                report.cpu_bvh_reachable_leaves = ctx.reachable_leaves;
                report.cpu_bvh_max_depth = ctx.max_depth;
                report.cpu_bvh_uniform_leaves = ctx.uniform_leaves;
                report.cpu_bvh_chunk_array_leaves = ctx.chunk_array_leaves;
                report.cpu_bvh_errors = ctx.errors;
            }
        }

        // --- GPU BVH buffers ---
        let fd = &self.voxel_frame_data;
        report.gpu_bvh_root = fd.region_bvh_root_index;
        report.gpu_bvh_node_count = fd.region_bvh_nodes.len();
        report.gpu_bvh_leaf_count = fd.leaf_headers.len();

        if fd.region_bvh_root_index != VTE_REGION_BVH_INVALID_NODE
            && (fd.region_bvh_root_index as usize) < fd.region_bvh_nodes.len()
        {
            let mut ctx = GpuBvhWalkCtx::default();
            walk_gpu_bvh_node(fd, fd.region_bvh_root_index, 0, &mut ctx);
            report.gpu_bvh_reachable_nodes = ctx.reachable_nodes;
            report.gpu_bvh_reachable_leaves = ctx.reachable_leaves;
            report.gpu_bvh_max_depth = ctx.max_depth;
            report.gpu_bvh_uniform_leaves = ctx.uniform_leaves;
            report.gpu_bvh_chunk_array_leaves = ctx.chunk_array_leaves;
            report.gpu_bvh_sibling_overlaps = ctx.sibling_overlaps;
            report.gpu_bvh_errors = ctx.errors;
        }

        // --- Cross-validate: render cache vs world tree ---
        if let (Some(bounds), Some(cache)) =
            (cache_bounds, self.render_region_cache.as_ref())
        {
            let fresh = self.build_render_region_tree_in_bounds(bounds);
            let fresh_core = fresh.slice_non_empty_core_in_bounds(bounds);
            let cached_core = cache.slice_non_empty_core_in_bounds(bounds);

            // Compare by collecting all non-empty chunks from both
            let mut fresh_chunks = collect_non_empty_chunks_from_core_in_bounds(
                &fresh_core, bounds,
            );
            fresh_chunks.sort_unstable_by_key(|(key, _)| *key);

            let mut cached_chunks = collect_non_empty_chunks_from_core_in_bounds(
                &cached_core, bounds,
            );
            cached_chunks.sort_unstable_by_key(|(key, _)| *key);

            report.cross_validate_fresh_chunk_count = fresh_chunks.len();
            report.cross_validate_cached_chunk_count = cached_chunks.len();

            // Find mismatches
            let mut fi = 0;
            let mut ci = 0;
            while fi < fresh_chunks.len() && ci < cached_chunks.len() {
                let (fk, fp) = &fresh_chunks[fi];
                let (ck, cp) = &cached_chunks[ci];
                match fk.cmp(ck) {
                    std::cmp::Ordering::Less => {
                        report.cross_validate_errors.push(format!(
                            "chunk {:?} in world_tree but missing from render cache",
                            fk,
                        ));
                        fi += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        report.cross_validate_errors.push(format!(
                            "chunk {:?} in render cache but missing from world_tree",
                            ck,
                        ));
                        ci += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        // Compare resolved block data, not internal palette
                        // structure — the cache may flatten branches into a
                        // single ChunkArray with a shared (larger) palette.
                        if fp.dense_blocks() != cp.dense_blocks() {
                            report.cross_validate_errors.push(format!(
                                "chunk {:?} payload mismatch: fresh={:?} cached={:?}",
                                fk, fp, cp,
                            ));
                        }
                        fi += 1;
                        ci += 1;
                    }
                }
            }
            while fi < fresh_chunks.len() {
                report.cross_validate_errors.push(format!(
                    "chunk {:?} in world_tree but missing from render cache",
                    fresh_chunks[fi].0,
                ));
                fi += 1;
            }
            while ci < cached_chunks.len() {
                report.cross_validate_errors.push(format!(
                    "chunk {:?} in render cache but missing from world_tree",
                    cached_chunks[ci].0,
                ));
                ci += 1;
            }
        }

        // Dump full trees to stderr for offline analysis
        self.dump_render_trees();

        report
    }

    pub fn debug_render_bvh_ray_node_hits(
        &mut self,
        ray_origin_world: [f32; 4],
        ray_direction_world: [f32; 4],
        max_distance_world: f32,
        max_nodes: usize,
    ) -> Vec<render_tree::DebugRayBvhNodeHit> {
        if max_nodes == 0 || !max_distance_world.is_finite() || max_distance_world <= 0.0 {
            return Vec::new();
        }
        // Diagnostics must not mutate the active render cache: forcing a
        // per-frame cache resize here can trigger pathological full snapshot
        // rebuild loops in the main voxel path.
        self.render_bvh_cache
            .as_ref()
            .map(|bvh| {
                render_tree::collect_ray_intersected_nodes_from_bvh(
                    bvh,
                    ray_origin_world,
                    ray_direction_world,
                    max_distance_world,
                    max_nodes,
                )
            })
            .unwrap_or_default()
    }
}

// --- Render tree integrity report ---

#[derive(Default)]
pub struct RenderTreeReport {
    // Render region cache
    pub cache_bounds: Option<Aabb4i>,
    pub cache_tree_report: Option<TreeIntegrityReport>,

    // CPU BVH
    pub cpu_bvh_bounds: Option<Aabb4i>,
    pub cpu_bvh_root: Option<u32>,
    pub cpu_bvh_node_count: usize,
    pub cpu_bvh_leaf_count: usize,
    pub cpu_bvh_reachable_nodes: usize,
    pub cpu_bvh_reachable_leaves: usize,
    pub cpu_bvh_max_depth: usize,
    pub cpu_bvh_uniform_leaves: usize,
    pub cpu_bvh_chunk_array_leaves: usize,
    pub cpu_bvh_errors: Vec<String>,

    // GPU BVH buffers
    pub gpu_bvh_root: u32,
    pub gpu_bvh_node_count: usize,
    pub gpu_bvh_leaf_count: usize,
    pub gpu_bvh_reachable_nodes: usize,
    pub gpu_bvh_reachable_leaves: usize,
    pub gpu_bvh_max_depth: usize,
    pub gpu_bvh_uniform_leaves: usize,
    pub gpu_bvh_chunk_array_leaves: usize,
    pub gpu_bvh_sibling_overlaps: usize,
    pub gpu_bvh_errors: Vec<String>,

    // Cross-validation: render cache vs world tree
    pub cross_validate_fresh_chunk_count: usize,
    pub cross_validate_cached_chunk_count: usize,
    pub cross_validate_errors: Vec<String>,
}

// --- CPU BVH walk helpers ---

#[derive(Default)]
struct CpuBvhWalkCtx {
    reachable_nodes: usize,
    reachable_leaves: usize,
    max_depth: usize,
    uniform_leaves: usize,
    chunk_array_leaves: usize,
    errors: Vec<String>,
}

fn walk_cpu_bvh_node(
    bvh: &render_tree::RenderBvh,
    node_idx: u32,
    depth: usize,
    ctx: &mut CpuBvhWalkCtx,
) {
    if node_idx as usize >= bvh.nodes.len() {
        ctx.errors.push(format!(
            "node index {} out of range (len={})",
            node_idx,
            bvh.nodes.len()
        ));
        return;
    }
    ctx.reachable_nodes += 1;
    ctx.max_depth = ctx.max_depth.max(depth);

    let node = &bvh.nodes[node_idx as usize];
    match node.kind {
        render_tree::RenderBvhNodeKind::Leaf { leaf_index } => {
            if leaf_index as usize >= bvh.leaves.len() {
                ctx.errors.push(format!(
                    "leaf index {} out of range (len={}) at node {}",
                    leaf_index,
                    bvh.leaves.len(),
                    node_idx
                ));
                return;
            }
            ctx.reachable_leaves += 1;
            let leaf = &bvh.leaves[leaf_index as usize];

            // Validate leaf bounds match node bounds
            if leaf.bounds != node.bounds {
                ctx.errors.push(format!(
                    "node {} bounds {:?}->{:?} != leaf {} bounds {:?}->{:?}",
                    node_idx,
                    node.bounds.min,
                    node.bounds.max,
                    leaf_index,
                    leaf.bounds.min,
                    leaf.bounds.max,
                ));
            }

            match &leaf.kind {
                render_tree::RenderLeafKind::Uniform(_) => ctx.uniform_leaves += 1,
                render_tree::RenderLeafKind::VoxelChunkArray(ca) => {
                    ctx.chunk_array_leaves += 1;
                    // Validate ChunkArrayData bounds vs leaf bounds
                    if ca.bounds != leaf.bounds {
                        ctx.errors.push(format!(
                            "leaf {} bounds {:?}->{:?} != chunk_array bounds {:?}->{:?}",
                            leaf_index,
                            leaf.bounds.min,
                            leaf.bounds.max,
                            ca.bounds.min,
                            ca.bounds.max,
                        ));
                    }
                }
            }
        }
        render_tree::RenderBvhNodeKind::Internal { left, right } => {
            // Validate children bounds within parent
            if (left as usize) < bvh.nodes.len() {
                let l = &bvh.nodes[left as usize];
                if !bounds_within(l.bounds, node.bounds) {
                    ctx.errors.push(format!(
                        "left child {} bounds {:?}->{:?} not within parent {} bounds {:?}->{:?}",
                        left, l.bounds.min, l.bounds.max,
                        node_idx, node.bounds.min, node.bounds.max,
                    ));
                }
            }
            if (right as usize) < bvh.nodes.len() {
                let r = &bvh.nodes[right as usize];
                if !bounds_within(r.bounds, node.bounds) {
                    ctx.errors.push(format!(
                        "right child {} bounds {:?}->{:?} not within parent {} bounds {:?}->{:?}",
                        right, r.bounds.min, r.bounds.max,
                        node_idx, node.bounds.min, node.bounds.max,
                    ));
                }
            }
            walk_cpu_bvh_node(bvh, left, depth + 1, ctx);
            walk_cpu_bvh_node(bvh, right, depth + 1, ctx);
        }
    }
}

fn bounds_within(inner: Aabb4i, outer: Aabb4i) -> bool {
    inner.min[0] >= outer.min[0]
        && inner.min[1] >= outer.min[1]
        && inner.min[2] >= outer.min[2]
        && inner.min[3] >= outer.min[3]
        && inner.max[0] <= outer.max[0]
        && inner.max[1] <= outer.max[1]
        && inner.max[2] <= outer.max[2]
        && inner.max[3] <= outer.max[3]
}

// --- GPU BVH walk helpers ---

#[derive(Default)]
struct GpuBvhWalkCtx {
    reachable_nodes: usize,
    reachable_leaves: usize,
    max_depth: usize,
    uniform_leaves: usize,
    chunk_array_leaves: usize,
    sibling_overlaps: usize,
    errors: Vec<String>,
}

fn walk_gpu_bvh_node(fd: &VoxelFrameData, node_idx: u32, depth: usize, ctx: &mut GpuBvhWalkCtx) {
    if node_idx == VTE_REGION_BVH_INVALID_NODE {
        return;
    }
    if node_idx as usize >= fd.region_bvh_nodes.len() {
        ctx.errors.push(format!(
            "gpu node index {} out of range (len={})",
            node_idx,
            fd.region_bvh_nodes.len()
        ));
        return;
    }
    ctx.reachable_nodes += 1;
    ctx.max_depth = ctx.max_depth.max(depth);

    let node = &fd.region_bvh_nodes[node_idx as usize];
    let is_leaf = (node.flags & VTE_REGION_BVH_NODE_FLAG_LEAF) != 0;

    // Validate bounds are finite and non-degenerate
    for i in 0..4 {
        if !node.world_min[i].is_finite() || !node.world_max[i].is_finite() {
            ctx.errors.push(format!(
                "gpu node {} has non-finite bounds: min={:?} max={:?}",
                node_idx, node.world_min, node.world_max,
            ));
            break;
        }
        if node.world_min[i] > node.world_max[i] {
            ctx.errors.push(format!(
                "gpu node {} has inverted bounds on axis {}: min={} max={}",
                node_idx, i, node.world_min[i], node.world_max[i],
            ));
            break;
        }
    }

    if is_leaf {
        let leaf_idx = node.leaf_index;
        if leaf_idx as usize >= fd.leaf_headers.len() {
            ctx.errors.push(format!(
                "gpu leaf index {} out of range (len={}) at node {}",
                leaf_idx,
                fd.leaf_headers.len(),
                node_idx,
            ));
            return;
        }
        ctx.reachable_leaves += 1;
        let lh = &fd.leaf_headers[leaf_idx as usize];
        if lh.leaf_kind == VTE_LEAF_KIND_UNIFORM {
            ctx.uniform_leaves += 1;
        } else if lh.leaf_kind == VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY {
            ctx.chunk_array_leaves += 1;
        } else {
            ctx.errors.push(format!(
                "gpu leaf {} has unknown kind {}",
                leaf_idx, lh.leaf_kind,
            ));
        }

        // Validate leaf header lattice bounds vs node world bounds.
        // Leaf stores inclusive lattice coords; node stores half-open world floats.
        // scale_exp is packed into bits 24-31 of uniform_orientation.
        // cell_size = CHUNK_SIZE * 2^scale_exp.
        // Lattice coords are: lat_min[i] = floor(world_min[i] / cell_size)
        //                     lat_max[i] = floor(world_max[i] / cell_size) - 1
        let leaf_min = lh.min_chunk_coord;
        let leaf_max = lh.max_chunk_coord;
        let node_min = node.world_min;
        let node_max = node.world_max;
        let scale_exp = (lh.uniform_orientation >> 24) as u8 as i8;
        let cell_size = (CHUNK_SIZE as f32) * (2.0f32).powi(scale_exp as i32);
        let mut bounds_ok = true;
        for i in 0..4 {
            let expected_min = (node_min[i] / cell_size).floor() as i32;
            let expected_max = (node_max[i] / cell_size).floor() as i32 - 1;
            if leaf_min[i] != expected_min || leaf_max[i] != expected_max {
                bounds_ok = false;
                break;
            }
        }
        if !bounds_ok {
            let expected_min: [i32; 4] = std::array::from_fn(|i| (node_min[i] / cell_size).floor() as i32);
            let expected_max: [i32; 4] = std::array::from_fn(|i| (node_max[i] / cell_size).floor() as i32 - 1);
            ctx.errors.push(format!(
                "gpu node {} bounds {:?}->{:?} diverges from leaf {} coords {:?}->{:?} (scale_exp={}, cell_size={}, expected {:?}->{:?})",
                node_idx, node_min, node_max, leaf_idx, leaf_min, leaf_max, scale_exp, cell_size, expected_min, expected_max,
            ));
        }
    } else {
        let left = node.left_child;
        let right = node.right_child;

        // Check sibling overlap
        if left != VTE_REGION_BVH_INVALID_NODE
            && right != VTE_REGION_BVH_INVALID_NODE
            && (left as usize) < fd.region_bvh_nodes.len()
            && (right as usize) < fd.region_bvh_nodes.len()
        {
            let l = &fd.region_bvh_nodes[left as usize];
            let r = &fd.region_bvh_nodes[right as usize];
            // Check child bounds within parent
            for (child_name, child_idx, c) in [("left", left, l), ("right", right, r)] {
                for i in 0..4 {
                    if c.world_min[i] < node.world_min[i] - 0.01
                        || c.world_max[i] > node.world_max[i] + 0.01
                    {
                        ctx.errors.push(format!(
                            "gpu {} child {} bounds {:?}->{:?} outside parent {} bounds {:?}->{:?}",
                            child_name, child_idx, c.world_min, c.world_max,
                            node_idx, node.world_min, node.world_max,
                        ));
                        break;
                    }
                }
            }
        }

        walk_gpu_bvh_node(fd, left, depth + 1, ctx);
        walk_gpu_bvh_node(fd, right, depth + 1, ctx);
    }
}
