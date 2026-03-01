use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::{
    chunk_key_from_lattice, fixed_from_lattice, lattice_from_fixed, step_for_scale, Aabb4i,
    ChunkCoord,
};
use crate::shared::voxel::{linear_cell_index, BlockData, CHUNK_SIZE, CHUNK_VOLUME};
use std::collections::HashMap;

/// A spatial tree storing voxel chunk data with world-space half-open AABB bounds.
///
/// # Invariants
///
/// - **Non-overlapping siblings**: children of the same branch never spatially overlap.
/// - **Containment**: every child's bounds are fully within its parent's bounds.
/// - **Overwrite semantics**: `set_chunk_at_scale` replaces the full spatial region
///   of the chunk. A scale-2 chunk occupies 32×32×32×32 world units; inserting it
///   removes any existing data (including finer-scale chunks) in that region.
///   Cross-scale composition (merging edits with virgin data) is handled at a
///   higher level (world_field), not here.
#[derive(Clone, Debug, Default)]
pub struct RegionChunkTree {
    root: Option<Box<RegionTreeCore>>,
}

impl RegionChunkTree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_chunks<I>(chunks: I) -> Self
    where
        I: IntoIterator<Item = (ChunkKey, ResolvedChunkPayload)>,
    {
        let mut tree = Self::new();
        for (key, resolved) in chunks {
            let _ = tree.set_chunk(key, Some(resolved));
        }
        tree
    }

    pub fn root(&self) -> Option<&RegionTreeCore> {
        self.root.as_deref()
    }

    pub fn has_chunk(&self, key: ChunkKey) -> bool {
        self.chunk_payload(key).is_some()
    }

    /// Look up the chunk payload at a fixed-point key, returning the payload
    /// alongside the scale it is stored at.
    ///
    /// The returned `i8` is the `scale_exp` of the underlying data.  Callers
    /// that index into the payload via `block_at(voxel_index)` **must** ensure
    /// that `voxel_index` was computed at the returned scale, not at some
    /// assumed scale.
    pub fn chunk_payload(&self, key: ChunkKey) -> Option<(ResolvedChunkPayload, i8)> {
        self.root
            .as_ref()
            .and_then(|node| query_chunk_payload_in_node(node, key))
    }

    /// Look up the block at a fixed-point position by walking the BVH.
    ///
    /// Returns `BlockData::AIR` if no solid block covers the position.
    pub fn block_at(&self, pos: [ChunkCoord; 4]) -> BlockData {
        match self.root.as_ref() {
            Some(node) => bvh_point_query(&node.kind, node.bounds, pos)
                .map(|hit| hit.block)
                .unwrap_or(BlockData::AIR),
            None => BlockData::AIR,
        }
    }

    /// Look up the block and its spatial AABB at a fixed-point position.
    ///
    /// The returned AABB reflects the block's own `scale_exp`, not the
    /// chunk cell size — a coarse block stored in a fine-scale chunk will
    /// return the full coarse-block extent.
    pub fn block_and_bounds_at(&self, pos: [ChunkCoord; 4]) -> Option<BvhBlockHit> {
        self.root
            .as_ref()
            .and_then(|node| bvh_point_query(&node.kind, node.bounds, pos))
    }

    /// Look up the block at a fixed-point position, distinguishing "no data"
    /// from "data says air".
    ///
    /// Returns `Some(block)` (including `Some(AIR)`) when the tree has data
    /// covering the position. Returns `None` only when no data-containing
    /// node covers the position.
    pub fn block_data_at(&self, pos: [ChunkCoord; 4]) -> Option<BlockData> {
        self.root
            .as_ref()
            .and_then(|node| bvh_block_data_at_point(&node.kind, node.bounds, pos))
    }

    /// Cast a ray through the BVH and return the nearest solid block hit.
    ///
    /// `origin` and `direction` are in fixed-point coordinates.
    /// `max_t` is the maximum ray parameter (in the same units as the
    /// coordinates — i.e. a `max_t` of 10.0 means 10 world units for scale 0).
    pub fn raycast(
        &self,
        origin: [ChunkCoord; 4],
        direction: [ChunkCoord; 4],
        max_t: ChunkCoord,
    ) -> Option<BvhRayHit> {
        let root = self.root.as_ref()?;
        bvh_raycast(&root.kind, root.bounds, origin, direction, max_t)
    }

    /// Set a chunk at scale 0 (the standard unit-scale lattice).
    pub fn set_chunk(&mut self, key: ChunkKey, resolved: Option<ResolvedChunkPayload>) -> bool {
        self.set_chunk_at_scale(key, resolved, 0)
    }

    /// Set a chunk at a specific scale. The key is already in fixed-point coordinates.
    ///
    /// The chunk's world-space extent is `chunk_world_bounds(key, scale_exp)`.
    /// All existing data within that spatial region is overwritten, regardless
    /// of what scale it was stored at. Passing `None` for `resolved` clears
    /// the region (sets it to empty/air).
    /// `scale_exp` is needed to create ChunkArrayData with correct cell resolution.
    pub fn set_chunk_at_scale(
        &mut self,
        key: ChunkKey,
        resolved: Option<ResolvedChunkPayload>,
        scale_exp: i8,
    ) -> bool {
        let payload = resolved.map(|r| canonicalize_resolved_payload(r));
        let chunk_bounds = Aabb4i::chunk_world_bounds(key, scale_exp);
        if self.root.is_none() {
            let Some(payload) = payload else {
                return false;
            };
            self.root = Some(Box::new(RegionTreeCore {
                bounds: chunk_bounds,
                kind: kind_from_resolved_value_at_scale(chunk_bounds, Some(payload), scale_exp),
                generator_version_hash: 0,
            }));
            return true;
        }

        if payload.is_none()
            && self
                .root
                .as_ref()
                .map(|root| !aabb_contains_aabb(root.bounds, chunk_bounds))
                .unwrap_or(false)
        {
            return false;
        }

        while self
            .root
            .as_ref()
            .map(|root| !aabb_contains_aabb(root.bounds, chunk_bounds))
            .unwrap_or(false)
        {
            let Some(root) = self.root.take() else {
                break;
            };
            let expand_alignment = chunk_world_size_for_kind(Some(&root.kind))
                .max(chunk_world_size_for_scale(scale_exp));
            self.root = Some(expand_root_once(root, chunk_bounds, expand_alignment));
        }

        let changed = if let Some(root) = self.root.as_mut() {
            set_chunk_recursive(root, key, chunk_bounds, payload, scale_exp)
        } else {
            false
        };

        if changed
            && self
                .root
                .as_ref()
                .map(|root| matches!(root.kind, RegionNodeKind::Empty))
                .unwrap_or(false)
        {
            self.root = None;
        }

        changed
    }

    pub fn remove_chunk(&mut self, key: ChunkKey) -> bool {
        self.set_chunk(key, None)
    }

    pub fn any_non_empty_chunk_in_bounds(&self, bounds: Aabb4i) -> bool {
        if !bounds.is_valid() {
            return false;
        }
        self.root
            .as_ref()
            .map(|node| kind_has_non_empty_chunk_intersection(&node.kind, node.bounds, bounds))
            .unwrap_or(false)
    }

    pub fn non_empty_chunk_count(&self) -> usize {
        self.root
            .as_ref()
            .map(|node| count_non_empty_chunks(&node.kind, node.bounds))
            .unwrap_or(0)
    }

    pub fn collect_chunks(&self) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunks_from_kind(&root.kind, root.bounds, &mut out);
        }
        out
    }

    pub fn collect_chunks_in_bounds(&self, bounds: Aabb4i) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunks_from_kind_in_bounds(&root.kind, root.bounds, bounds, &mut out);
        }
        out
    }

    pub fn collect_chunk_keys_in_bounds(&self, bounds: Aabb4i) -> Vec<ChunkKey> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunk_keys_from_kind_in_bounds(&root.kind, root.bounds, bounds, &mut out);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn collect_non_empty_chunk_keys_in_bounds(&self, bounds: Aabb4i) -> Vec<ChunkKey> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_non_empty_chunk_keys_from_kind_in_bounds(
                &root.kind,
                root.bounds,
                bounds,
                &mut out,
            );
        }
        out.sort_unstable_by_key(|key| *key);
        out
    }

    pub fn slice_core_in_bounds(&self, bounds: Aabb4i) -> RegionTreeCore {
        let Some(root) = self.root.as_ref() else {
            return empty_core_for_bounds(bounds);
        };
        slice_region_core_in_bounds(root.as_ref(), bounds)
    }

    pub fn slice_non_empty_core_in_bounds(&self, bounds: Aabb4i) -> RegionTreeCore {
        let Some(root) = self.root.as_ref() else {
            return empty_core_for_bounds(bounds);
        };
        slice_non_empty_region_core_in_bounds(root.as_ref(), bounds)
    }

    pub fn take_non_empty_core_in_bounds(&mut self, bounds: Aabb4i) -> RegionTreeCore {
        if !bounds.is_valid() {
            return empty_core_for_bounds(bounds);
        }

        let extracted = self.slice_non_empty_core_in_bounds(bounds);
        if matches!(extracted.kind, RegionNodeKind::Empty) {
            return extracted;
        }

        let empty_replacement = empty_core_for_bounds(bounds);
        let _ = self.splice_non_empty_core_in_bounds(bounds, &empty_replacement);
        extracted
    }

    pub fn lazy_drop_outside_bounds(
        &mut self,
        keep_bounds: Aabb4i,
        max_subtree_drops: usize,
    ) -> Option<Aabb4i> {
        if max_subtree_drops == 0 {
            return None;
        }
        let Some(root) = self.root.as_mut() else {
            return None;
        };

        let mut budget = max_subtree_drops;
        let changed_bounds = lazy_drop_outside_node(root, keep_bounds, &mut budget);
        if changed_bounds.is_some() {
            normalize_chunk_node(root);
        }
        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }
        changed_bounds
    }

    pub fn splice_non_empty_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        core: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let replacement = slice_non_empty_region_core_in_bounds(core, bounds);
        let replacement_is_empty = matches!(replacement.kind, RegionNodeKind::Empty);

        if self.root.is_none() {
            if replacement_is_empty {
                return None;
            }
            self.root = Some(Box::new(replacement));
            return Some(bounds);
        }

        if !replacement_is_empty {
            self.ensure_root_contains_bounds(bounds);
        }

        let changed_bounds = if let Some(root) = self.root.as_mut() {
            splice_node_with_non_empty_core(root, bounds, &replacement)
        } else {
            None
        };

        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }

        changed_bounds
    }

    pub fn splice_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        core: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let replacement = slice_region_core_in_bounds(core, bounds);
        let replacement_is_empty = matches!(replacement.kind, RegionNodeKind::Empty);

        if self.root.is_none() {
            if replacement_is_empty {
                return None;
            }
            self.root = Some(Box::new(replacement));
            return Some(bounds);
        }

        if !replacement_is_empty {
            self.ensure_root_contains_bounds(bounds);
        }

        let changed_bounds = if let Some(root) = self.root.as_mut() {
            splice_node_with_core(root, bounds, &replacement)
        } else {
            None
        };

        if self
            .root
            .as_ref()
            .map(|root| matches!(root.kind, RegionNodeKind::Empty))
            .unwrap_or(false)
        {
            self.root = None;
        }

        changed_bounds
    }

    pub fn overlay_core_in_bounds(
        &mut self,
        bounds: Aabb4i,
        overlay: &RegionTreeCore,
    ) -> Option<Aabb4i> {
        if !bounds.is_valid() {
            return None;
        }

        let overlay_slice = slice_region_core_in_bounds(overlay, bounds);
        if matches!(overlay_slice.kind, RegionNodeKind::Empty) {
            return None;
        }

        let mut changed = None::<Aabb4i>;
        overlay_non_empty_leaves(&overlay_slice, &mut |leaf| {
            if let Some(splice_changed) = self.splice_core_in_bounds(leaf.bounds, leaf) {
                changed = Some(match changed {
                    Some(existing) => merge_aabb(existing, splice_changed),
                    None => splice_changed,
                });
            }
        });
        changed
    }

    /// Find all non-empty leaf chunks whose spatial extent overlaps `query`.
    ///
    /// Returns `(ChunkKey, scale_exp)` for each overlapping leaf. For single-chunk
    /// leaves this is the key and scale directly. For multi-chunk ChunkArray or
    /// Uniform nodes, individual chunk positions within the overlapping region
    /// are enumerated.
    pub fn find_leaf_chunks_in_spatial_range(&self, query: &Aabb4i) -> Vec<(ChunkKey, i8)> {
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            find_leaf_chunks_in_spatial_range_recursive(
                &root.kind,
                root.bounds,
                query,
                &mut out,
            );
        }
        out
    }

    fn ensure_root_contains_bounds(&mut self, bounds: Aabb4i) {
        if !bounds.is_valid() {
            return;
        }

        if self.root.is_none() {
            self.root = Some(Box::new(RegionTreeCore {
                bounds,
                kind: RegionNodeKind::Empty,
                generator_version_hash: 0,
            }));
            return;
        }

        let alignment = chunk_world_size_for_kind(self.root.as_ref().map(|r| &r.kind));
        while self
            .root
            .as_ref()
            .map(|root| !aabb_contains_aabb(root.bounds, bounds))
            .unwrap_or(false)
        {
            let Some(root) = self.root.take() else {
                break;
            };
            self.root = Some(expand_root_once(root, bounds, alignment));
        }
    }

}

/// Summary of tree structure and integrity issues.
#[derive(Debug, Default)]
pub struct TreeIntegrityReport {
    pub root_bounds: Option<Aabb4i>,
    pub max_depth: usize,
    pub branch_count: usize,
    pub chunk_array_count: usize,
    pub uniform_count: usize,
    pub empty_count: usize,
    pub procedural_ref_count: usize,
    pub total_chunk_cells: usize,
    /// Pairs of sibling nodes whose BOUNDS overlap (query-routing hazard).
    pub bounds_overlaps: Vec<(Aabb4i, Aabb4i)>,
    /// Data-level overlaps (non-empty cells claimed by multiple leaves).
    pub data_overlap_error: Option<String>,
    /// Scale distribution: (scale_exp, count_of_chunk_arrays_at_that_scale).
    pub scale_histogram: Vec<(i8, usize)>,
}

pub fn validate_tree_integrity(tree: &RegionChunkTree) -> TreeIntegrityReport {
    let mut report = TreeIntegrityReport::default();
    let Some(root) = tree.root() else {
        return report;
    };
    report.root_bounds = Some(root.bounds);

    let mut scale_counts: std::collections::HashMap<i8, usize> = std::collections::HashMap::new();
    collect_integrity_stats(root, 0, &mut report, &mut scale_counts);

    let mut hist: Vec<(i8, usize)> = scale_counts.into_iter().collect();
    hist.sort_by_key(|(s, _)| *s);
    report.scale_histogram = hist;

    match validate_region_core_world_space_non_overlapping(root) {
        Ok(()) => {}
        Err(e) => report.data_overlap_error = Some(e),
    }

    report
}

fn collect_integrity_stats(
    node: &RegionTreeCore,
    depth: usize,
    report: &mut TreeIntegrityReport,
    scale_counts: &mut std::collections::HashMap<i8, usize>,
) {
    if depth > report.max_depth {
        report.max_depth = depth;
    }
    match &node.kind {
        RegionNodeKind::Empty => report.empty_count += 1,
        RegionNodeKind::Uniform(_) => report.uniform_count += 1,
        RegionNodeKind::ProceduralRef(_) => report.procedural_ref_count += 1,
        RegionNodeKind::ChunkArray(ca) => {
            report.chunk_array_count += 1;
            let cells = ca.bounds.chunk_cell_count_at_scale(ca.scale_exp).unwrap_or(0);
            report.total_chunk_cells += cells;
            *scale_counts.entry(ca.scale_exp).or_default() += 1;
        }
        RegionNodeKind::Branch(children) => {
            report.branch_count += 1;
            // Check sibling bounds overlaps.
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    if children[i].bounds.intersects(&children[j].bounds) {
                        report.bounds_overlaps.push((children[i].bounds, children[j].bounds));
                    }
                }
            }
            for child in children {
                collect_integrity_stats(child, depth + 1, report, scale_counts);
            }
        }
    }
}

pub fn validate_region_core_world_space_non_overlapping(
    core: &RegionTreeCore,
) -> Result<(), String> {
    if !core.bounds.is_valid() {
        return Ok(());
    }

    let mut occupied = Vec::<Aabb4i>::new();
    collect_world_occupied_cells_from_kind(&core.kind, core.bounds, &mut occupied)?;
    if occupied.len() < 2 {
        return Ok(());
    }

    for i in 0..occupied.len() {
        for j in (i + 1)..occupied.len() {
            let a = occupied[i];
            let b = occupied[j];
            if a.intersects(&b) {
                return Err(format!(
                    "overlap detected between bounds {:?}->{:?} and {:?}->{:?}",
                    a.min,
                    a.max,
                    b.min,
                    b.max,
                ));
            }
        }
    }

    Ok(())
}

fn collect_world_occupied_cells_from_kind(
    kind: &RegionNodeKind,
    kind_bounds: Aabb4i,
    out: &mut Vec<Aabb4i>,
) -> Result<(), String> {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => Ok(()),
        RegionNodeKind::Uniform(block) => {
            if !block.is_air() {
                out.push(kind_bounds);
            }
            Ok(())
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let indices = chunk_array
                .decode_dense_indices()
                .map_err(|error| format!("decode chunk array indices failed: {error:?}"))?;
            let extents = chunk_array
                .bounds
                .chunk_extents_at_scale(chunk_array.scale_exp)
                .ok_or_else(|| "chunk array bounds extents missing".to_string())?;
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            let se = chunk_array.scale_exp;
            let (lmin, lmax) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            let local = [
                                (lx - lmin[0]) as usize,
                                (ly - lmin[1]) as usize,
                                (lz - lmin[2]) as usize,
                                (lw - lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                return Err("chunk-array index out of bounds while validating overlap"
                                    .to_string());
                            };
                            if !palette_non_empty
                                .get(*palette_idx as usize)
                                .copied()
                                .unwrap_or(true)
                            {
                                continue;
                            }
                            let pos = chunk_key_from_lattice(
                                [lx, ly, lz, lw],
                                se,
                            );
                            // Half-open world-space bounds for this chunk cell.
                            out.push(Aabb4i::chunk_world_bounds(pos, se));
                        }
                    }
                }
            }
            Ok(())
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_world_occupied_cells_from_kind(&child.kind, child.bounds, out)?;
            }
            Ok(())
        }
    }
}

pub fn collect_non_empty_chunks_from_core_in_bounds(
    core: &RegionTreeCore,
    bounds: Aabb4i,
) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
    if !bounds.is_valid() {
        return Vec::new();
    }
    let mut out = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(&core.kind, core.bounds, bounds, &mut out);
    out.sort_unstable_by_key(|(key, _)| *key);
    out
}

pub fn slice_region_core_in_bounds(core: &RegionTreeCore, bounds: Aabb4i) -> RegionTreeCore {
    if !bounds.is_valid() {
        return empty_core_for_bounds(bounds);
    }

    let Some(intersection) = intersect_aabb(core.bounds, bounds) else {
        return empty_core_for_bounds(bounds);
    };
    let Some(clipped) = slice_node_to_bounds(core, intersection) else {
        return empty_core_for_bounds(bounds);
    };

    if clipped.bounds == bounds {
        clipped
    } else {
        RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Branch(vec![clipped]),
            generator_version_hash: core.generator_version_hash,
        }
    }
}

pub fn slice_non_empty_region_core_in_bounds(
    core: &RegionTreeCore,
    bounds: Aabb4i,
) -> RegionTreeCore {
    let mut sliced = slice_region_core_in_bounds(core, bounds);
    prune_empty_subtrees(&mut sliced);
    sliced
}

fn empty_core_for_bounds(bounds: Aabb4i) -> RegionTreeCore {
    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    }
}

fn slice_node_to_bounds(node: &RegionTreeCore, bounds: Aabb4i) -> Option<RegionTreeCore> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let kind = match &node.kind {
        RegionNodeKind::Empty => RegionNodeKind::Empty,
        RegionNodeKind::Uniform(block) => RegionNodeKind::Uniform(block.clone()),
        RegionNodeKind::ProceduralRef(generator_ref) => {
            RegionNodeKind::ProceduralRef(generator_ref.clone())
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let sliced = if intersection == chunk_array.bounds {
                chunk_array.clone()
            } else {
                slice_chunk_array_to_bounds(chunk_array, intersection)?
            };
            RegionNodeKind::ChunkArray(sliced)
        }
        RegionNodeKind::Branch(children) => {
            let mut clipped_children = Vec::new();
            for child in children {
                if let Some(clipped) = slice_node_to_bounds(child, intersection) {
                    clipped_children.push(clipped);
                }
            }
            if clipped_children.is_empty() {
                RegionNodeKind::Empty
            } else if clipped_children.len() == 1 && clipped_children[0].bounds == intersection {
                return Some(clipped_children.pop().expect("single clipped child"));
            } else {
                RegionNodeKind::Branch(clipped_children)
            }
        }
    };

    Some(RegionTreeCore {
        bounds: intersection,
        kind,
        generator_version_hash: node.generator_version_hash,
    })
}

fn slice_chunk_array_to_bounds(
    chunk_array: &ChunkArrayData,
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = intersect_aabb(chunk_array.bounds, bounds)?;
    let source_indices = chunk_array.decode_dense_indices().ok()?;
    slice_chunk_array_to_bounds_with_dense_indices(chunk_array, &source_indices, intersection)
}

fn slice_chunk_array_to_bounds_with_dense_indices(
    chunk_array: &ChunkArrayData,
    source_indices: &[u16],
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = intersect_aabb(chunk_array.bounds, bounds)?;
    let se = chunk_array.scale_exp;
    let source_extents = chunk_array.bounds.chunk_extents_at_scale(se)?;
    let target_extents = intersection.chunk_extents_at_scale(se)?;
    let target_cell_count = target_extents[0]
        .checked_mul(target_extents[1])?
        .checked_mul(target_extents[2])?
        .checked_mul(target_extents[3])?;
    let mut target_indices = Vec::with_capacity(target_cell_count);

    let (src_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    let (int_lmin, int_lmax) = intersection.to_chunk_lattice_bounds(se);
    for lw in int_lmin[3]..=int_lmax[3] {
        for lz in int_lmin[2]..=int_lmax[2] {
            for ly in int_lmin[1]..=int_lmax[1] {
                for lx in int_lmin[0]..=int_lmax[0] {
                    let source_local = [
                        (lx - src_lmin[0]) as usize,
                        (ly - src_lmin[1]) as usize,
                        (lz - src_lmin[2]) as usize,
                        (lw - src_lmin[3]) as usize,
                    ];
                    let source_linear = linear_cell_index(source_local, source_extents);
                    target_indices.push(*source_indices.get(source_linear)?);
                }
            }
        }
    }

    ChunkArrayData::from_dense_indices_with_block_palette(
        intersection,
        chunk_array.chunk_palette.clone(),
        target_indices,
        None,
        chunk_array.block_palette.clone(),
        chunk_array.scale_exp,
    )
    .ok()
}

/// Re-encode a ChunkArray's blocks into target_bounds at a (possibly finer) target_scale.
///
/// Used during tree carving to fill gap regions that don't align to the source's
/// scale grid. Rather than losing data (returning Empty), this resamples the source
/// blocks at a finer scale that does align to the target bounds.
///
/// Returns a full `RegionTreeCore`:
/// - `Uniform` if all resampled cells contain the same block
/// - `ChunkArray` otherwise
/// - `Empty` if target_bounds doesn't intersect source_bounds
pub fn resample_chunk_array_to_bounds(
    source: &ChunkArrayData,
    target_bounds: Aabb4i,
    target_scale: i8,
    generator_version_hash: u64,
) -> RegionTreeCore {
    // Intersect to get only the overlapping region.
    let Some(intersection) = intersect_aabb(target_bounds, source.bounds) else {
        return RegionTreeCore {
            bounds: target_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash,
        };
    };

    // The target bounds must align to target_scale (caller guarantees via coarsest_aligned_scale).
    let target_extents = intersection
        .chunk_extents_at_scale(target_scale)
        .expect("resample_chunk_array_to_bounds: target_bounds must align to target_scale");
    let target_cell_count = target_extents[0] * target_extents[1] * target_extents[2] * target_extents[3];

    // Source geometry.
    let source_scale = source.scale_exp;
    let source_extents = source.bounds
        .chunk_extents_at_scale(source_scale)
        .expect("resample_chunk_array_to_bounds: source bounds must align to source scale");

    // Decode source: indices → chunk payloads → dense block materials per chunk.
    let source_indices = source
        .decode_dense_indices()
        .expect("resample_chunk_array_to_bounds: failed to decode source indices");

    // Pre-expand each unique chunk payload to dense block palette indices.
    let chunk_materials: Vec<Vec<u16>> = source
        .chunk_palette
        .iter()
        .map(|cp| cp.dense_materials().unwrap_or_else(|_| vec![0u16; CHUNK_VOLUME]))
        .collect();

    // Scale ratio: how many target cells per source cell per axis.
    // target_scale <= source_scale (target is finer or equal).
    assert!(
        target_scale <= source_scale,
        "resample_chunk_array_to_bounds: target_scale ({}) must be <= source_scale ({})",
        target_scale,
        source_scale,
    );
    let scale_diff = (source_scale - target_scale) as u32;
    let ratio = 1usize << scale_diff; // target cells per source cell per axis

    // Lattice bounds for source and target at their respective scales.
    let (src_lmin, _) = source.bounds.to_chunk_lattice_bounds(source_scale);
    let (tgt_lmin, tgt_lmax) = intersection.to_chunk_lattice_bounds(target_scale);

    // Track uniformity: first block seen, and whether all match.
    let mut first_block: Option<BlockData> = None;
    let mut all_uniform = true;

    // Build dense block indices for the target. We build per-chunk payloads.
    // Target may span multiple chunks at target_scale.
    let target_chunk_count = target_cell_count;
    let mut target_blocks = Vec::with_capacity(target_chunk_count);

    for tw in tgt_lmin[3]..=tgt_lmax[3] {
        for tz in tgt_lmin[2]..=tgt_lmax[2] {
            for ty in tgt_lmin[1]..=tgt_lmax[1] {
                for tx in tgt_lmin[0]..=tgt_lmax[0] {
                    // This is a chunk position in the target grid at target_scale.
                    // Each chunk position contains CHUNK_SIZE^4 cells.
                    // But in a ChunkArrayData, each "chunk" in the grid is one
                    // entry in dense_indices pointing to a ChunkPayload.
                    // We need to build one ChunkPayload per target grid position.

                    // Map target chunk to source: each target chunk position maps
                    // to a region inside one source chunk.
                    // Target chunk abs lattice at target_scale: [tx, ty, tz, tw]
                    // Source chunk abs lattice at source_scale: [tx/ratio, ty/ratio, ...]
                    let src_abs = [
                        (tx as isize).div_euclid(ratio as isize) as i32,
                        (ty as isize).div_euclid(ratio as isize) as i32,
                        (tz as isize).div_euclid(ratio as isize) as i32,
                        (tw as isize).div_euclid(ratio as isize) as i32,
                    ];
                    let src_local = [
                        (src_abs[0] - src_lmin[0]) as usize,
                        (src_abs[1] - src_lmin[1]) as usize,
                        (src_abs[2] - src_lmin[2]) as usize,
                        (src_abs[3] - src_lmin[3]) as usize,
                    ];
                    let src_linear = linear_cell_index(src_local, source_extents);
                    let chunk_palette_idx = source_indices[src_linear] as usize;
                    let src_materials = &chunk_materials[chunk_palette_idx];
                    let cs = CHUNK_SIZE;

                    if scale_diff == 0 {
                        // Same scale — the source chunk maps 1:1 to the target chunk.
                        // Just resolve all blocks directly.
                        let mut chunk_blocks = Vec::with_capacity(CHUNK_VOLUME);
                        for voxel_idx in 0..CHUNK_VOLUME {
                            let block_palette_idx = src_materials[voxel_idx] as usize;
                            let block = source
                                .block_palette
                                .get(block_palette_idx)
                                .cloned()
                                .unwrap_or(BlockData::AIR);
                            if all_uniform {
                                match &first_block {
                                    None => first_block = Some(block.clone()),
                                    Some(fb) => {
                                        if !fb.matches_ignoring_scale(&block) {
                                            all_uniform = false;
                                        }
                                    }
                                }
                            }
                            chunk_blocks.push(block);
                        }
                        target_blocks.push(chunk_blocks);
                    } else if ratio >= cs {
                        // Fast path: ratio >= CHUNK_SIZE means every cell in the
                        // target chunk maps to the same single source voxel.
                        // Skip the O(CHUNK_VOLUME) inner loop entirely.
                        let src_voxel = [
                            ((tx as usize).rem_euclid(ratio) * cs) / ratio,
                            ((ty as usize).rem_euclid(ratio) * cs) / ratio,
                            ((tz as usize).rem_euclid(ratio) * cs) / ratio,
                            ((tw as usize).rem_euclid(ratio) * cs) / ratio,
                        ];
                        let src_voxel_idx = src_voxel[3] * cs * cs * cs
                            + src_voxel[2] * cs * cs
                            + src_voxel[1] * cs
                            + src_voxel[0];
                        let block_palette_idx = src_materials[src_voxel_idx] as usize;
                        let block = source
                            .block_palette
                            .get(block_palette_idx)
                            .cloned()
                            .unwrap_or(BlockData::AIR);
                        if all_uniform {
                            match &first_block {
                                None => first_block = Some(block.clone()),
                                Some(fb) => {
                                    if !fb.matches_ignoring_scale(&block) {
                                        all_uniform = false;
                                    }
                                }
                            }
                        }
                        // Uniform chunk: all CHUNK_VOLUME cells are the same block.
                        let mut chunk_blocks = Vec::with_capacity(CHUNK_VOLUME);
                        chunk_blocks.resize(CHUNK_VOLUME, block);
                        target_blocks.push(chunk_blocks);
                    } else {
                        // Target is finer — each target chunk maps to a sub-region
                        // of one source chunk. Multiple target cells may map to
                        // different source cells.

                        let mut chunk_blocks = Vec::with_capacity(CHUNK_VOLUME);

                        // Offset of target chunk within source chunk, in target-scale cells.
                        let tgt_offset_in_src = [
                            (tx as usize).rem_euclid(ratio) * cs,
                            (ty as usize).rem_euclid(ratio) * cs,
                            (tz as usize).rem_euclid(ratio) * cs,
                            (tw as usize).rem_euclid(ratio) * cs,
                        ];

                        for vw in 0..cs {
                            for vz in 0..cs {
                                for vy in 0..cs {
                                    for vx in 0..cs {
                                        let tgt_cell = [
                                            tgt_offset_in_src[0] + vx,
                                            tgt_offset_in_src[1] + vy,
                                            tgt_offset_in_src[2] + vz,
                                            tgt_offset_in_src[3] + vw,
                                        ];
                                        let src_voxel = [
                                            tgt_cell[0] / ratio,
                                            tgt_cell[1] / ratio,
                                            tgt_cell[2] / ratio,
                                            tgt_cell[3] / ratio,
                                        ];
                                        let src_voxel_idx = src_voxel[3] * cs * cs * cs
                                            + src_voxel[2] * cs * cs
                                            + src_voxel[1] * cs
                                            + src_voxel[0];
                                        let block_palette_idx =
                                            src_materials[src_voxel_idx] as usize;
                                        let block = source
                                            .block_palette
                                            .get(block_palette_idx)
                                            .cloned()
                                            .unwrap_or(BlockData::AIR);
                                        if all_uniform {
                                            match &first_block {
                                                None => first_block = Some(block.clone()),
                                                Some(fb) => {
                                                    if !fb.matches_ignoring_scale(&block) {
                                                        all_uniform = false;
                                                    }
                                                }
                                            }
                                        }
                                        chunk_blocks.push(block);
                                    }
                                }
                            }
                        }
                        target_blocks.push(chunk_blocks);
                    }
                }
            }
        }
    }

    // If all cells are the same block, return Uniform.
    if all_uniform {
        let block = first_block.unwrap_or(BlockData::AIR);
        return RegionTreeCore {
            bounds: intersection,
            kind: RegionNodeKind::Uniform(block),
            generator_version_hash,
        };
    }

    // Build ChunkArrayData from per-chunk block vectors.
    // Each target_blocks entry is one chunk's worth of blocks (CHUNK_VOLUME).
    // We need to build chunk_palette + block_palette + dense_indices.
    let mut block_palette = vec![BlockData::AIR];
    let mut block_to_idx: HashMap<BlockData, u16> = HashMap::new();
    block_to_idx.insert(BlockData::AIR, 0);

    let mut chunk_palette = Vec::new();
    let mut chunk_to_idx: HashMap<Vec<u16>, u16> = HashMap::new();
    let mut dense_indices = Vec::with_capacity(target_cell_count);

    for chunk_blocks in &target_blocks {
        // Convert blocks to block palette indices.
        let mut voxel_indices = Vec::with_capacity(CHUNK_VOLUME);
        for block in chunk_blocks {
            let idx = match block_to_idx.get(block) {
                Some(&idx) => idx,
                None => {
                    let idx = block_palette.len() as u16;
                    block_palette.push(block.clone());
                    block_to_idx.insert(block.clone(), idx);
                    idx
                }
            };
            voxel_indices.push(idx);
        }

        // Deduplicate chunk payloads.
        let chunk_idx = match chunk_to_idx.get(&voxel_indices) {
            Some(&idx) => idx,
            None => {
                let idx = chunk_palette.len() as u16;
                let payload = ChunkPayload::from_dense_materials_compact(&voxel_indices)
                    .unwrap_or(ChunkPayload::Empty);
                chunk_palette.push(payload);
                chunk_to_idx.insert(voxel_indices, idx);
                idx
            }
        };
        dense_indices.push(chunk_idx);
    }

    match ChunkArrayData::from_dense_indices_with_block_palette(
        intersection,
        chunk_palette,
        dense_indices,
        None,
        block_palette,
        target_scale,
    ) {
        Ok(chunk_array) => RegionTreeCore {
            bounds: intersection,
            kind: RegionNodeKind::ChunkArray(chunk_array),
            generator_version_hash,
        },
        Err(e) => {
            panic!(
                "resample_chunk_array_to_bounds: failed to build ChunkArrayData: {:?}",
                e
            );
        }
    }
}

fn prune_empty_subtrees(core: &mut RegionTreeCore) -> bool {
    match &mut core.kind {
        RegionNodeKind::Empty => false,
        RegionNodeKind::Uniform(block) => !block.is_air(),
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_has_non_empty_intersection(chunk_array, chunk_array.bounds)
        }
        RegionNodeKind::Branch(children) => {
            children.retain_mut(prune_empty_subtrees);
            if children.is_empty() {
                core.kind = RegionNodeKind::Empty;
                false
            } else if children.len() == 1 {
                let child = children.pop().expect("single child");
                core.bounds = child.bounds;
                core.kind = child.kind;
                true
            } else {
                true
            }
        }
    }
}

fn splice_node_with_non_empty_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let replacement_slice = slice_non_empty_region_core_in_bounds(replacement, intersection);

    if matches!(replacement_slice.kind, RegionNodeKind::Empty)
        && !kind_has_non_empty_chunk_intersection(&node.kind, node.bounds, intersection)
    {
        return None;
    }

    match (&node.kind, &replacement_slice.kind) {
        (RegionNodeKind::Empty, RegionNodeKind::Empty) => return None,
        (RegionNodeKind::Uniform(existing), RegionNodeKind::Uniform(incoming))
            if existing == incoming =>
        {
            return None;
        }
        (RegionNodeKind::Uniform(existing), RegionNodeKind::Empty) if existing.is_air() => {
            return None;
        }
        (RegionNodeKind::ProceduralRef(existing), RegionNodeKind::ProceduralRef(incoming))
            if existing == incoming =>
        {
            return None;
        }
        _ => {}
    }

    if matches!(
        node.kind,
        RegionNodeKind::Branch(_) | RegionNodeKind::ChunkArray(_)
    ) {
        let existing_slice = slice_non_empty_region_core_in_bounds(node, intersection);
        if existing_slice.kind == replacement_slice.kind
            && existing_slice.bounds == replacement_slice.bounds
        {
            return None;
        }
    }
    if non_empty_kinds_semantically_equal_in_bounds(
        &node.kind,
        node.bounds,
        &replacement_slice.kind,
        replacement_slice.bounds,
        intersection,
    ) {
        return None;
    }

    if intersection == node.bounds {
        if node.kind == replacement_slice.kind {
            return None;
        }
        *node = replacement_slice;
        return Some(node.bounds);
    }

    // --- Partial overlap: carve existing data, insert replacement. ---

    // For Branch nodes, try to recurse into a single child that fully
    // contains the intersection.  This avoids re-projecting all siblings
    // when only one child is affected.
    if let RegionNodeKind::Branch(children) = &mut node.kind {
        let containing_idx = children
            .iter()
            .position(|child| aabb_contains_aabb(child.bounds, intersection));
        if let Some(idx) = containing_idx {
            let result =
                splice_node_with_non_empty_core(&mut children[idx], bounds, replacement);
            if result.is_some() {
                normalize_chunk_node(node);
            }
            return result;
        }
        // No single child contains it — fall through to leaf-style carve
        // on the node itself.
    }

    // Single-pass carve-and-replace for leaf nodes (or Branch nodes where
    // the patch spans multiple children).  Modelled on carve_leaf_for_chunk_edit
    // to produce correctly-scaled remnants.
    splice_carve_and_replace(node, intersection, replacement_slice)
}

/// Recursively collect tree nodes whose bounds don't overlap `patch_bounds`.
/// For Branch nodes that partially overlap the patch, descend into children
/// rather than pushing the Branch itself (whose bounds may still claim coverage
/// over the carved-out region, creating false overlap with the replacement).
/// For leaf nodes that partially overlap, carve them and recurse on the result.
fn collect_non_overlapping_remnants(
    node: &RegionTreeCore,
    patch_bounds: Aabb4i,
    out: &mut Vec<RegionTreeCore>,
) {
    if !node.bounds.intersects(&patch_bounds) {
        out.push(node.clone());
        return;
    }
    if aabb_contains_aabb(patch_bounds, node.bounds) {
        // Fully covered by the patch — drop.
        return;
    }
    // Partially overlapping.
    match &node.kind {
        RegionNodeKind::Branch(children) => {
            // Recurse into children — each child either doesn't overlap,
            // is fully covered, or is handled recursively.
            for child in children {
                collect_non_overlapping_remnants(child, patch_bounds, out);
            }
        }
        _ => {
            // Non-Branch leaf that partially overlaps: produce remnant pieces
            // via subtract_aabb + project_node_to_bounds.  We intentionally
            // avoid carve_region_with_rechunk here because its normalize step
            // can consolidate ChunkArray pieces back to the original bounds,
            // restoring the overlap we're trying to eliminate.
            let intersection = match intersect_aabb(node.bounds, patch_bounds) {
                Some(i) => i,
                None => return,
            };
            for piece_bounds in subtract_aabb(node.bounds, intersection) {
                let piece = project_node_to_bounds(
                    &node.kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(piece.kind, RegionNodeKind::Empty) {
                    out.push(piece);
                }
            }
        }
    }
}

/// Single-pass carve: split the existing node around `patch_bounds`, then
/// insert `replacement` for the patched region.  Uses ChunkArray-aware
/// alignment splitting so remnants stay at their original scale when
/// possible, falling back to resample only for non-aligned pieces.
///
/// Always returns `patch_bounds` — the carved remnants have identical voxel
/// content (just reorganized in the tree), so only the patch region is
/// semantically changed.  Note: callers that feed this into the render BVH
/// delta path may need wider dirty bounds if the BVH can't correctly
/// represent misaligned remnants (see `build_leaf_outside_bounds_pieces`).
fn splice_carve_and_replace(
    node: &mut RegionTreeCore,
    patch_bounds: Aabb4i,
    replacement: RegionTreeCore,
) -> Option<Aabb4i> {
    let source_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let gen_hash = node.generator_version_hash;

    let mut children = Vec::with_capacity(9);

    match &source_kind {
        RegionNodeKind::ChunkArray(chunk_array) => {
            // ChunkArray-aware carving: keep remnants at original scale where
            // possible (aligned pieces), resample only for non-aligned pieces.
            if let Ok(source_indices) = chunk_array.decode_dense_indices() {
                let se = chunk_array.scale_exp;
                for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                    let piece_aligned = piece_bounds
                        .chunk_extents_at_scale(se)
                        .map(|_| {
                            let (lmin, lmax) = piece_bounds.to_chunk_lattice_bounds(se);
                            Aabb4i::from_lattice_bounds(lmin, lmax, se) == piece_bounds
                        })
                        .unwrap_or(false);

                    if piece_aligned {
                        if let Some(chunk_array_piece) =
                            slice_chunk_array_to_bounds_with_dense_indices(
                                chunk_array,
                                &source_indices,
                                piece_bounds,
                            )
                        {
                            children.push(RegionTreeCore {
                                bounds: piece_bounds,
                                kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                                generator_version_hash: gen_hash,
                            });
                            continue;
                        }
                    }
                    // Non-aligned: resample at coarsest aligned finer scale.
                    let aligned_scale =
                        coarsest_aligned_scale(piece_bounds, chunk_array.scale_exp);
                    let resampled = resample_chunk_array_to_bounds(
                        chunk_array,
                        piece_bounds,
                        aligned_scale,
                        gen_hash,
                    );
                    if !matches!(resampled.kind, RegionNodeKind::Empty) {
                        children.push(resampled);
                    }
                }
            } else {
                // Fallback: can't decode indices, use generic projection.
                for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                    let projected = project_node_to_bounds(
                        &source_kind,
                        node.bounds,
                        piece_bounds,
                        gen_hash,
                    );
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        children.push(projected);
                    }
                }
            }
        }
        RegionNodeKind::Branch(branch_children) => {
            // Recursively collect remnant pieces whose bounds don't overlap
            // the patch.  A simple one-level flatten is insufficient because
            // nested Branches may have bounds that still span the patch region
            // even though their data was carved away — the query would enter
            // the wrong Branch and miss the replacement.
            for child in branch_children {
                collect_non_overlapping_remnants(child, patch_bounds, &mut children);
            }
        }
        RegionNodeKind::Uniform(block) if !block.is_air() => {
            for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                let projected = project_node_to_bounds(
                    &source_kind,
                    node.bounds,
                    piece_bounds,
                    gen_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
        _ => {
            // Empty or air Uniform — no existing data to carve.
            // ProceduralRef — shouldn't appear on client scene trees.
            for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                let projected = project_node_to_bounds(
                    &source_kind,
                    node.bounds,
                    piece_bounds,
                    gen_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    for child in &children {
        debug_assert!(
            !child.bounds.intersects(&patch_bounds),
            "remnant child overlaps patch: child={:?} patch={:?}",
            child.bounds,
            patch_bounds,
        );
    }

    // Insert the replacement data.
    if !matches!(replacement.kind, RegionNodeKind::Empty) {
        children.push(replacement);
    }

    if children.is_empty() {
        node.kind = RegionNodeKind::Empty;
    } else {
        node.kind = RegionNodeKind::Branch(children);
        normalize_chunk_node(node);
    }

    #[cfg(debug_assertions)]
    if let RegionNodeKind::Branch(ref ch) = node.kind {
        for i in 0..ch.len() {
            for j in (i + 1)..ch.len() {
                debug_assert!(
                    !ch[i].bounds.intersects(&ch[j].bounds),
                    "post-normalize overlap: ch[{}]={:?} ch[{}]={:?}",
                    i, ch[i].bounds, j, ch[j].bounds
                );
            }
        }
    }

    // Return the patch bounds — carved remnants have the same voxel content
    // (just reorganized in the tree), so only the patch region changed.
    Some(patch_bounds)
}

/// Carve `clear_bounds` out of `node`, preserving data outside the cleared
/// region with proper ChunkArray-aware rechunking.
fn carve_region_with_rechunk(node: &mut RegionTreeCore, clear_bounds: Aabb4i) {
    let Some(intersection) = intersect_aabb(node.bounds, clear_bounds) else {
        return;
    };
    if matches!(node.kind, RegionNodeKind::Empty) {
        return;
    }
    if intersection == node.bounds {
        node.kind = RegionNodeKind::Empty;
        return;
    }

    match &mut node.kind {
        RegionNodeKind::Branch(children) => {
            let mut retained = Vec::with_capacity(children.len());
            for mut child in std::mem::take(children) {
                if !child.bounds.intersects(&intersection) {
                    retained.push(child);
                    continue;
                }
                if aabb_contains_aabb(intersection, child.bounds) {
                    continue;
                }
                carve_region_with_rechunk(&mut child, intersection);
                if !matches!(child.kind, RegionNodeKind::Empty) {
                    retained.push(child);
                }
            }
            *children = retained;
            normalize_chunk_node(node);
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let gen_hash = node.generator_version_hash;
            let se = chunk_array.scale_exp;
            let mut pieces = Vec::new();
            if let Ok(source_indices) = chunk_array.decode_dense_indices() {
                for piece_bounds in subtract_aabb(node.bounds, intersection) {
                    let piece_aligned = piece_bounds
                        .chunk_extents_at_scale(se)
                        .map(|_| {
                            let (lmin, lmax) = piece_bounds.to_chunk_lattice_bounds(se);
                            Aabb4i::from_lattice_bounds(lmin, lmax, se) == piece_bounds
                        })
                        .unwrap_or(false);

                    if piece_aligned {
                        if let Some(chunk_array_piece) =
                            slice_chunk_array_to_bounds_with_dense_indices(
                                chunk_array,
                                &source_indices,
                                piece_bounds,
                            )
                        {
                            pieces.push(RegionTreeCore {
                                bounds: piece_bounds,
                                kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                                generator_version_hash: gen_hash,
                            });
                            continue;
                        }
                    }
                    let aligned_scale = coarsest_aligned_scale(piece_bounds, se);
                    let resampled = resample_chunk_array_to_bounds(
                        chunk_array,
                        piece_bounds,
                        aligned_scale,
                        gen_hash,
                    );
                    if !matches!(resampled.kind, RegionNodeKind::Empty) {
                        pieces.push(resampled);
                    }
                }
            } else {
                let old_kind = node.kind.clone();
                for piece_bounds in subtract_aabb(node.bounds, intersection) {
                    let projected = project_node_to_bounds(
                        &old_kind,
                        node.bounds,
                        piece_bounds,
                        gen_hash,
                    );
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        pieces.push(projected);
                    }
                }
            }
            node.kind = RegionNodeKind::Branch(pieces);
            normalize_chunk_node(node);
        }
        _ => {
            // Uniform, ProceduralRef — generic projection.
            let old_kind = node.kind.clone();
            let mut pieces = Vec::new();
            for piece_bounds in subtract_aabb(node.bounds, intersection) {
                let projected = project_node_to_bounds(
                    &old_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    pieces.push(projected);
                }
            }
            node.kind = RegionNodeKind::Branch(pieces);
            normalize_chunk_node(node);
        }
    }
}

fn splice_node_with_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = intersect_aabb(node.bounds, bounds)?;
    let replacement_slice = slice_region_core_in_bounds(replacement, intersection);
    let existing_slice = slice_region_core_in_bounds(node, intersection);
    if existing_slice.kind == replacement_slice.kind {
        return None;
    }

    if intersection == node.bounds {
        if node.kind == replacement_slice.kind {
            return None;
        }
        *node = replacement_slice;
        return Some(node.bounds);
    }

    // For Branch nodes, try to recurse into a single child that fully
    // contains the intersection.
    if let RegionNodeKind::Branch(children) = &mut node.kind {
        let containing_idx = children
            .iter()
            .position(|child| aabb_contains_aabb(child.bounds, intersection));
        if let Some(idx) = containing_idx {
            let result = splice_node_with_core(&mut children[idx], bounds, replacement);
            if result.is_some() {
                normalize_chunk_node(node);
            }
            return result;
        }
    }

    // Single-pass carve-and-replace.
    splice_carve_and_replace(node, intersection, replacement_slice)
}

fn lazy_drop_outside_node(
    node: &mut RegionTreeCore,
    keep_bounds: Aabb4i,
    budget: &mut usize,
) -> Option<Aabb4i> {
    if *budget == 0 {
        return None;
    }
    if !keep_bounds.is_valid() || !node.bounds.intersects(&keep_bounds) {
        if matches!(node.kind, RegionNodeKind::Empty) {
            return None;
        }
        *budget -= 1;
        let changed = node.bounds;
        node.kind = RegionNodeKind::Empty;
        return Some(changed);
    }
    if aabb_contains_aabb(keep_bounds, node.bounds) || is_single_chunk_bounds_any_scale(node.bounds) {
        return None;
    }

    ensure_binary_children(node);
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return None;
    };

    let mut changed = None::<Aabb4i>;
    for child in children.iter_mut() {
        if *budget == 0 {
            break;
        }
        if let Some(child_changed) = lazy_drop_outside_node(child, keep_bounds, budget) {
            changed = Some(match changed {
                Some(acc) => merge_aabb(acc, child_changed),
                None => child_changed,
            });
        }
    }

    normalize_chunk_node(node);
    changed
}

fn overlay_non_empty_leaves<F>(core: &RegionTreeCore, visit: &mut F)
where
    F: FnMut(&RegionTreeCore),
{
    match &core.kind {
        RegionNodeKind::Empty => {}
        RegionNodeKind::Branch(children) => {
            for child in children {
                overlay_non_empty_leaves(child, visit);
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            // A consolidated ChunkArray may contain Empty gap entries at positions
            // where no original data existed. If we splice the whole array, those
            // Empty gaps overwrite base data (e.g. virgin terrain). Decompose arrays
            // that have Empty gaps so only positions with real data are overlayed.
            if chunk_array_has_empty_gap_default(chunk_array) {
                overlay_chunk_array_non_empty_cells(
                    chunk_array,
                    core.generator_version_hash,
                    visit,
                );
            } else {
                visit(core);
            }
        }
        _ => visit(core),
    }
}

/// Returns true if this ChunkArray uses a default_chunk_idx pointing to an
/// Empty palette entry — the signature of consolidation-created gap positions.
fn chunk_array_has_empty_gap_default(chunk_array: &ChunkArrayData) -> bool {
    if let Some(default_idx) = chunk_array.default_chunk_idx {
        chunk_array
            .chunk_palette
            .get(default_idx as usize)
            .map_or(false, |p| *p == ChunkPayload::Empty)
    } else {
        false
    }
}

/// Visit each non-Empty cell in a ChunkArray individually, skipping gap entries.
fn overlay_chunk_array_non_empty_cells<F>(
    chunk_array: &ChunkArrayData,
    generator_version_hash: u64,
    visit: &mut F,
) where
    F: FnMut(&RegionTreeCore),
{
    let Ok(indices) = chunk_array.decode_dense_indices() else {
        return;
    };
    let se = chunk_array.scale_exp;
    let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(se) else {
        return;
    };
    let default_idx = chunk_array.default_chunk_idx;
    let (lmin, lmax) = chunk_array.bounds.to_chunk_lattice_bounds(se);

    for lw in lmin[3]..=lmax[3] {
        for lz in lmin[2]..=lmax[2] {
            for ly in lmin[1]..=lmax[1] {
                for lx in lmin[0]..=lmax[0] {
                    let local = [
                        (lx - lmin[0]) as usize,
                        (ly - lmin[1]) as usize,
                        (lz - lmin[2]) as usize,
                        (lw - lmin[3]) as usize,
                    ];
                    let linear = linear_cell_index(local, extents);
                    let palette_idx = indices[linear];

                    if default_idx == Some(palette_idx) {
                        continue;
                    }

                    let payload = &chunk_array.chunk_palette[palette_idx as usize];
                    if *payload == ChunkPayload::Empty {
                        continue;
                    }

                    let pos = chunk_key_from_lattice([lx, ly, lz, lw], se);
                    let cell_bounds = Aabb4i::chunk_world_bounds(pos, se);
                    let Ok(cell_ca) = ChunkArrayData::from_dense_indices_with_block_palette(
                        cell_bounds,
                        vec![payload.clone()],
                        vec![0],
                        None,
                        chunk_array.block_palette.clone(),
                        se,
                    ) else {
                        continue;
                    };
                    visit(&RegionTreeCore {
                        bounds: cell_bounds,
                        kind: RegionNodeKind::ChunkArray(cell_ca),
                        generator_version_hash,
                    });
                }
            }
        }
    }
}

/// World-space chunk size for a given scale: CS * step_for_scale(s).
fn chunk_world_size_for_scale(scale_exp: i8) -> ChunkCoord {
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    cs.saturating_mul(step_for_scale(scale_exp))
}

/// World-space alignment for split/carve boundaries, derived from node content.
///
/// Uses the coarsest (largest) chunk world size among all leaf content so that
/// split boundaries land on a grid compatible with all children.
fn chunk_world_size_for_kind(kind: Option<&RegionNodeKind>) -> ChunkCoord {
    match kind {
        Some(RegionNodeKind::ChunkArray(ca)) => chunk_world_size_for_scale(ca.scale_exp),
        Some(RegionNodeKind::Branch(children)) => children
            .iter()
            .map(|c| chunk_world_size_for_kind(Some(&c.kind)))
            .max()
            .unwrap_or(chunk_world_size_for_scale(0)),
        _ => chunk_world_size_for_scale(0),
    }
}

fn aabb_contains_aabb(outer: Aabb4i, inner: Aabb4i) -> bool {
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

fn merge_aabb(a: Aabb4i, b: Aabb4i) -> Aabb4i {
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

/// Subtract `inner` from `outer`, producing up to 8 half-open pieces that tile
/// the remaining space with no gaps and no overlaps.
fn subtract_aabb(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    let Some(inner) = intersect_aabb(outer, inner) else {
        return vec![outer];
    };
    if inner == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;

    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis];
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis];
            if piece.is_valid() {
                pieces.push(piece);
            }
            core.max[axis] = inner.max[axis];
        }
    }

    pieces
}

fn collect_non_empty_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ResolvedChunkPayload)>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return;
            }
            let resolved = ResolvedChunkPayload::uniform(block.clone());
            // Uniform is always scale 0
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((chunk_key_from_lattice([lx, ly, lz, lw], 0), resolved.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(chunk_array.scale_exp) else {
                return;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            let se = chunk_array.scale_exp;
            let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_chunk_lattice_bounds(se);
            for lw in int_lmin[3]..=int_lmax[3] {
                for lz in int_lmin[2]..=int_lmax[2] {
                    for ly in int_lmin[1]..=int_lmax[1] {
                        for lx in int_lmin[0]..=int_lmax[0] {
                            let local = [
                                (lx - ca_lmin[0]) as usize,
                                (ly - ca_lmin[1]) as usize,
                                (lz - ca_lmin[2]) as usize,
                                (lw - ca_lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let palette_idx = *palette_idx as usize;
                            if !palette_non_empty.get(palette_idx).copied().unwrap_or(true) {
                                continue;
                            }
                            let Some(payload) = chunk_array.chunk_palette.get(palette_idx) else {
                                continue;
                            };
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], se),
                                ResolvedChunkPayload {
                                    payload: payload.clone(),
                                    block_palette: chunk_array.block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_chunks_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn collect_non_empty_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return;
            }
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push(chunk_key_from_lattice([lx, ly, lz, lw], 0));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(chunk_array.scale_exp) else {
                return;
            };
            let se = chunk_array.scale_exp;
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_chunk_lattice_bounds(se);
            for lw in int_lmin[3]..=int_lmax[3] {
                for lz in int_lmin[2]..=int_lmax[2] {
                    for ly in int_lmin[1]..=int_lmax[1] {
                        for lx in int_lmin[0]..=int_lmax[0] {
                            let local = [
                                (lx - ca_lmin[0]) as usize,
                                (ly - ca_lmin[1]) as usize,
                                (lz - ca_lmin[2]) as usize,
                                (lw - ca_lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let palette_idx = *palette_idx as usize;
                            if !palette_non_empty.get(palette_idx).copied().unwrap_or(true) {
                                continue;
                            }
                            out.push(chunk_key_from_lattice([lx, ly, lz, lw], se));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_empty_chunk_keys_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn set_chunk_recursive(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    chunk_bounds: Aabb4i,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    if !aabb_contains_aabb(node.bounds, chunk_bounds) {
        return false;
    }

    if is_single_chunk_bounds(node.bounds, scale_exp) {
        if let RegionNodeKind::ChunkArray(existing_ca) = &node.kind {
            if existing_ca.scale_exp != scale_exp {
                eprintln!(
                    "[set-chunk-scale-mismatch] replacing scale {} with {} at {:?} — \
                     caller should have resolved cross-scale overlap first",
                    existing_ca.scale_exp, scale_exp, key_pos
                );
            }
        }
        let new_kind = kind_from_resolved_value_at_scale(node.bounds, payload, scale_exp);
        if node.kind == new_kind {
            return false;
        }
        node.kind = new_kind;
        return true;
    }

    if matches!(node.kind, RegionNodeKind::Branch(_)) {
        return set_chunk_recursive_in_branch(node, key_pos, chunk_bounds, payload, scale_exp);
    }

    carve_leaf_for_chunk_edit(node, key_pos, chunk_bounds, payload, scale_exp)
}

fn set_chunk_recursive_in_branch(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    chunk_bounds: Aabb4i,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return false;
    };
    // Find a child that fully contains the chunk bounds.
    let target_idx = children
        .iter()
        .position(|child| aabb_contains_aabb(child.bounds, chunk_bounds));
    let changed = if let Some(target_idx) = target_idx {
        set_chunk_recursive(&mut children[target_idx], key_pos, chunk_bounds, payload, scale_exp)
    } else {
        // No child fully contains the chunk — it spans multiple children
        // (e.g. a scale-2 chunk that's larger than the branch's sub-regions).
        //
        // We carve the chunk's spatial region out of all overlapping children,
        // preserving data outside the chunk and discarding data inside it.
        // This is correct: set_chunk_at_scale has overwrite semantics — the
        // new chunk replaces ALL prior data in its world-space extent,
        // regardless of what scale that data was at.
        let gen_hash = node.generator_version_hash;
        let old_children = std::mem::take(children);
        let mut new_children = Vec::with_capacity(old_children.len() + 1);
        let mut changed = false;
        for child in old_children {
            if !child.bounds.intersects(&chunk_bounds) {
                new_children.push(child);
            } else if aabb_contains_aabb(chunk_bounds, child.bounds) {
                // Chunk fully covers this child — overwritten entirely.
                changed = true;
            } else {
                // Partial overlap — split child into pieces outside the chunk,
                // projecting existing data into each piece.
                changed = true;
                for piece_bounds in subtract_aabb(child.bounds, chunk_bounds) {
                    let projected = project_node_to_bounds(
                        &child.kind,
                        child.bounds,
                        piece_bounds,
                        child.generator_version_hash,
                    );
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        new_children.push(projected);
                    }
                }
            }
        }
        if let Some(payload) = payload {
            new_children.push(RegionTreeCore {
                bounds: chunk_bounds,
                kind: kind_from_resolved_value_at_scale(chunk_bounds, Some(payload), scale_exp),
                generator_version_hash: gen_hash,
            });
            changed = true;
        }
        *children = new_children;
        changed
    };

    if changed {
        normalize_chunk_node(node);
    }
    changed
}

fn carve_leaf_for_chunk_edit(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    chunk_bounds: Aabb4i,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    let source_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);

    let no_change = match &source_kind {
        RegionNodeKind::Empty => resolved_option_is_semantically_empty(payload.as_ref()),
        RegionNodeKind::Uniform(block) => {
            resolved_option_matches_block(block, payload.as_ref())
        }
        RegionNodeKind::ChunkArray(_) => false,
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::Branch(_) => false,
    };
    if no_change {
        node.kind = source_kind;
        return false;
    }

    let mut children = Vec::with_capacity(9);
    if let RegionNodeKind::ChunkArray(chunk_array) = &source_kind {
        if let Ok(source_indices) = chunk_array.decode_dense_indices() {
            let existing_resolved = chunk_array_payload_at_with_dense_indices(
                chunk_array,
                &source_indices,
                key_pos,
            )
            .map(|p| ResolvedChunkPayload {
                payload: p,
                block_palette: chunk_array.block_palette.clone(),
            });
            if resolved_option_matches_existing(existing_resolved, payload.as_ref()) {
                node.kind = source_kind;
                return false;
            }
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
                // Check if piece_bounds align to the source scale's chunk grid.
                // If not, the slice would produce semantically wrong data (sub-chunk
                // bounds with a full-chunk payload). Use resample instead.
                let se = chunk_array.scale_exp;
                let piece_aligned = piece_bounds
                    .chunk_extents_at_scale(se)
                    .map(|_| {
                        let (lmin, lmax) = piece_bounds.to_chunk_lattice_bounds(se);
                        Aabb4i::from_lattice_bounds(lmin, lmax, se) == piece_bounds
                    })
                    .unwrap_or(false);

                if piece_aligned {
                    if let Some(chunk_array_piece) =
                        slice_chunk_array_to_bounds_with_dense_indices(
                            chunk_array,
                            &source_indices,
                            piece_bounds,
                        )
                    {
                        children.push(RegionTreeCore {
                            bounds: piece_bounds,
                            kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                            generator_version_hash: node.generator_version_hash,
                        });
                        continue;
                    }
                }
                // Bounds don't align to source scale — resample at coarsest aligned finer scale.
                let aligned_scale =
                    coarsest_aligned_scale(piece_bounds, chunk_array.scale_exp);
                let resampled = resample_chunk_array_to_bounds(
                    chunk_array,
                    piece_bounds,
                    aligned_scale,
                    node.generator_version_hash,
                );
                if !matches!(resampled.kind, RegionNodeKind::Empty) {
                    children.push(resampled);
                }
            }
        } else {
            let existing_resolved =
                chunk_array_payload_at(chunk_array, key_pos).map(|p| ResolvedChunkPayload {
                    payload: p,
                    block_palette: chunk_array.block_palette.clone(),
                });
            if resolved_option_matches_existing(existing_resolved, payload.as_ref()) {
                node.kind = source_kind;
                return false;
            }
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
                let projected = project_node_to_bounds(
                    &source_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
    } else {
        for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
            let projected = project_node_to_bounds(
                &source_kind,
                node.bounds,
                piece_bounds,
                node.generator_version_hash,
            );
            if !matches!(projected.kind, RegionNodeKind::Empty) {
                children.push(projected);
            }
        }
    }

    if let Some(payload) = payload {
        children.push(RegionTreeCore {
            bounds: chunk_bounds,
            kind: kind_from_resolved_value_at_scale(chunk_bounds, Some(payload), scale_exp),
            generator_version_hash: node.generator_version_hash,
        });
    }

    node.kind = RegionNodeKind::Branch(children);
    normalize_chunk_node(node);
    true
}

fn ensure_binary_children(node: &mut RegionTreeCore) {
    let alignment = chunk_world_size_for_kind(Some(&node.kind));
    if let RegionNodeKind::Branch(children) = &mut node.kind {
        if branch_matches_split(node.bounds, children, alignment) {
            sort_children_canonical(children);
            return;
        }
    }

    let Some((left_bounds, right_bounds)) = split_bounds_longest_axis(node.bounds, alignment)
    else {
        return;
    };

    let generator_version_hash = node.generator_version_hash;
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let left = project_node_to_bounds(&old_kind, node.bounds, left_bounds, generator_version_hash);
    let right =
        project_node_to_bounds(&old_kind, node.bounds, right_bounds, generator_version_hash);
    let mut children = vec![left, right];
    sort_children_canonical(&mut children);
    node.kind = RegionNodeKind::Branch(children);
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum AdjacentMergeKind {
    Uniform(BlockData),
    ProceduralRef(GeneratorRef),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct AdjacentMergeGroupKey {
    axis: u8,
    kind: AdjacentMergeKind,
    generator_version_hash: u64,
    other_mins: [ChunkCoord; 3],
    other_maxs: [ChunkCoord; 3],
}

fn adjacent_merge_kind(kind: &RegionNodeKind) -> Option<AdjacentMergeKind> {
    match kind {
        RegionNodeKind::Uniform(block) => Some(AdjacentMergeKind::Uniform(block.clone())),
        RegionNodeKind::ProceduralRef(generator_ref) => {
            Some(AdjacentMergeKind::ProceduralRef(generator_ref.clone()))
        }
        _ => None,
    }
}

fn build_adjacent_merge_group_key(
    node: &RegionTreeCore,
    axis: usize,
) -> Option<AdjacentMergeGroupKey> {
    let kind = adjacent_merge_kind(&node.kind)?;
    let mut other_mins = [ChunkCoord::ZERO; 3];
    let mut other_maxs = [ChunkCoord::ZERO; 3];
    let mut write_idx = 0usize;
    for dim in 0..4 {
        if dim == axis {
            continue;
        }
        other_mins[write_idx] = node.bounds.min[dim];
        other_maxs[write_idx] = node.bounds.max[dim];
        write_idx += 1;
    }
    Some(AdjacentMergeGroupKey {
        axis: axis as u8,
        kind,
        generator_version_hash: node.generator_version_hash,
        other_mins,
        other_maxs,
    })
}

fn merge_adjacent_children_once(children: &mut Vec<RegionTreeCore>) -> bool {
    if children.len() < 2 {
        return false;
    }

    for axis in 0..4 {
        let mut grouped = HashMap::<AdjacentMergeGroupKey, Vec<RegionTreeCore>>::new();
        let mut passthrough = Vec::<RegionTreeCore>::new();
        for child in std::mem::take(children) {
            let Some(key) = build_adjacent_merge_group_key(&child, axis) else {
                passthrough.push(child);
                continue;
            };
            grouped.entry(key).or_default().push(child);
        }

        let mut merged_any = false;
        let mut rebuilt = passthrough;
        for (_key, mut group) in grouped {
            if group.len() < 2 {
                rebuilt.extend(group.into_iter());
                continue;
            }

            group.sort_unstable_by_key(|node| node.bounds.min[axis]);
            let mut iter = group.into_iter();
            let mut current = iter
                .next()
                .expect("group with len >= 2 must produce first child");
            for node in iter {
                // Half-open adjacency: current ends where node begins.
                if node.bounds.min[axis] == current.bounds.max[axis] {
                    current.bounds.max[axis] = node.bounds.max[axis];
                    merged_any = true;
                } else {
                    rebuilt.push(current);
                    current = node;
                }
            }
            rebuilt.push(current);
        }

        *children = rebuilt;
        if merged_any {
            sort_children_canonical(children);
            return true;
        }
    }
    false
}

/// Merge multiple ChunkArray children of a Branch into a single ChunkArray.
///
/// Collects all ChunkArray children, decodes their chunk payloads, and rebuilds
/// them as one consolidated ChunkArray spanning the combined bounds.  The merged
/// ChunkArray uses `default_chunk_idx = Some(0)` (Empty) so that positions outside
/// the original children's bounds are implicitly empty.
///
/// Skips consolidation when:
/// - Fewer than 2 ChunkArray children exist
/// - The merged bounding box would overlap a non-ChunkArray sibling (violating
///   the Branch non-overlapping invariant)
/// - The merged volume exceeds 4× the populated chunk count (too sparse)
fn consolidate_chunk_array_children(
    children: &mut Vec<RegionTreeCore>,
    generator_version_hash: u64,
) -> bool {
    let ca_count = children
        .iter()
        .filter(|c| matches!(c.kind, RegionNodeKind::ChunkArray(_)))
        .count();
    if ca_count < 2 {
        return false;
    }

    // Extract ChunkArray children, leaving others in place.
    let mut ca_children = Vec::with_capacity(ca_count);
    let mut other_children = Vec::with_capacity(children.len() - ca_count);
    for child in std::mem::take(children) {
        if matches!(child.kind, RegionNodeKind::ChunkArray(_)) {
            ca_children.push(child);
        } else {
            other_children.push(child);
        }
    }

    let common_scale_exp = ca_children
        .iter()
        .find_map(|child| match &child.kind {
            RegionNodeKind::ChunkArray(ca) => Some(ca.scale_exp),
            _ => None,
        })
        .unwrap_or(0);
    if ca_children.iter().any(|child| match &child.kind {
        RegionNodeKind::ChunkArray(ca) => ca.scale_exp != common_scale_exp,
        _ => false,
    }) {
        *children = ca_children;
        children.extend(other_children);
        return false;
    }

    // Compute combined bounds and total populated chunk count.
    let mut combined_bounds = ca_children[0].bounds;
    let mut total_chunks = 0usize;
    for ca_child in &ca_children {
        combined_bounds = merge_aabb(combined_bounds, ca_child.bounds);
        if let RegionNodeKind::ChunkArray(ca) = &ca_child.kind {
            if let Some(extents) = ca.bounds.chunk_extents_at_scale(ca.scale_exp) {
                total_chunks += extents.iter().copied().product::<usize>();
            }
        }
    }

    // Safety check: the merged AABB must not overlap any non-ChunkArray sibling.
    // The Branch invariant requires non-overlapping children.  Since the merged
    // ChunkArray's bounds can be wider than any individual child (it's the AABB
    // union), it could extend into space occupied by Uniform or other siblings.
    // If that happens, skip the merge entirely.
    if other_children
        .iter()
        .any(|c| c.bounds.intersects(&combined_bounds))
    {
        *children = ca_children;
        children.extend(other_children);
        return false;
    }

    let Some(combined_extents) = combined_bounds.chunk_extents_at_scale(common_scale_exp) else {
        *children = ca_children;
        children.extend(other_children);
        return false;
    };
    let combined_volume: usize = combined_extents.iter().copied().product();
    if combined_volume == 0 {
        *children = ca_children;
        children.extend(other_children);
        return false;
    }

    // Density check: skip if merged volume is much larger than chunk count.
    // Allow up to 4× overhead (25% density minimum).
    if combined_volume > total_chunks.saturating_mul(4) {
        *children = ca_children;
        children.extend(other_children);
        return false;
    }

    // First: merge block_palettes from all children and build per-child remap tables.
    let mut merged_block_palette = vec![BlockData::AIR];
    let mut block_palette_map: HashMap<BlockData, u16> = HashMap::new();
    block_palette_map.insert(BlockData::AIR, 0);
    let mut child_block_remaps: Vec<Vec<u16>> = Vec::with_capacity(ca_children.len());
    for ca_child in &ca_children {
        let RegionNodeKind::ChunkArray(ca) = &ca_child.kind else {
            child_block_remaps.push(Vec::new());
            continue;
        };
        let mut remap = Vec::with_capacity(ca.block_palette.len());
        for block in &ca.block_palette {
            let merged_idx = if let Some(&idx) = block_palette_map.get(block) {
                idx
            } else {
                let idx = merged_block_palette.len() as u16;
                block_palette_map.insert(block.clone(), idx);
                merged_block_palette.push(block.clone());
                idx
            };
            remap.push(merged_idx);
        }
        child_block_remaps.push(remap);
    }

    // Build chunk palette and dense index array for the merged ChunkArray.
    let empty_payload = ChunkPayload::Empty;
    let mut palette: Vec<ChunkPayload> = vec![empty_payload.clone()];
    let mut palette_map: HashMap<ChunkPayload, u16> = HashMap::new();
    palette_map.insert(empty_payload, 0u16);
    let mut dense_indices = vec![0u16; combined_volume];

    for (child_idx, ca_child) in ca_children.iter().enumerate() {
        let RegionNodeKind::ChunkArray(ca) = &ca_child.kind else {
            continue;
        };
        let Ok(child_indices) = ca.decode_dense_indices() else {
            // If we can't decode, bail out and restore original children.
            *children = ca_children;
            children.extend(other_children);
            return false;
        };
        let Some(child_extents) = ca.bounds.chunk_extents_at_scale(common_scale_exp) else {
            continue;
        };
        let block_remap = &child_block_remaps[child_idx];
        let (child_lmin, child_lmax) = ca.bounds.to_chunk_lattice_bounds(common_scale_exp);
        let (comb_lmin, _) = combined_bounds.to_chunk_lattice_bounds(common_scale_exp);

        // Iterate all positions in this child's bounds.
        for lw in child_lmin[3]..=child_lmax[3] {
            for lz in child_lmin[2]..=child_lmax[2] {
                for ly in child_lmin[1]..=child_lmax[1] {
                    for lx in child_lmin[0]..=child_lmax[0] {
                        let child_local = [
                            (lx - child_lmin[0]) as usize,
                            (ly - child_lmin[1]) as usize,
                            (lz - child_lmin[2]) as usize,
                            (lw - child_lmin[3]) as usize,
                        ];
                        let child_linear = linear_cell_index(child_local, child_extents);
                        let palette_idx = child_indices[child_linear] as usize;

                        let payload = ca.chunk_palette[palette_idx].clone();

                        if payload == ChunkPayload::Empty {
                            continue;
                        }

                        // Remap block palette indices within this payload.
                        let remapped_payload = remap_chunk_payload_block_indices(&payload, block_remap);

                        // Map child lattice coords to combined lattice coords.
                        let combined_local = [
                            (lx - comb_lmin[0]) as usize,
                            (ly - comb_lmin[1]) as usize,
                            (lz - comb_lmin[2]) as usize,
                            (lw - comb_lmin[3]) as usize,
                        ];
                        let combined_linear = linear_cell_index(
                            combined_local,
                            combined_extents,
                        );

                        // Deduplicate palette entries.
                        let merged_idx = if let Some(&idx) = palette_map.get(&remapped_payload) {
                            idx
                        } else {
                            let idx = palette.len() as u16;
                            palette_map.insert(remapped_payload.clone(), idx);
                            palette.push(remapped_payload);
                            idx
                        };
                        dense_indices[combined_linear] = merged_idx;
                    }
                }
            }
        }
    }

    // Build the merged ChunkArray.
    let Ok(merged_ca) = ChunkArrayData::from_dense_indices_with_block_palette(
        combined_bounds,
        palette,
        dense_indices,
        Some(0), // default = Empty
        merged_block_palette,
        common_scale_exp,
    ) else {
        *children = ca_children;
        children.extend(other_children);
        return false;
    };

    // Replace ChunkArray children with the single merged node.
    *children = other_children;
    children.push(RegionTreeCore {
        bounds: combined_bounds,
        kind: RegionNodeKind::ChunkArray(merged_ca),
        generator_version_hash,
    });
    true
}

fn normalize_chunk_node(node: &mut RegionTreeCore) {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return;
    };
    for child in children.iter_mut() {
        normalize_chunk_node(child);
    }
    children.retain(|child| !matches!(child.kind, RegionNodeKind::Empty));
    if children.is_empty() {
        node.kind = RegionNodeKind::Empty;
        return;
    }
    // Collapse single-child branches by shrink-wrapping bounds to the child.
    // After splice/clear carves away siblings, the remaining child is typically smaller
    // than the parent bounds.  Tightening the parent to the child and collapsing removes
    // degenerate Branch(1) chains that add BVH traversal depth with no spatial benefit.
    if children.len() == 1 {
        let child = children.pop().expect("single child");
        node.bounds = child.bounds;
        node.kind = child.kind;
        return;
    }

    let mut merge_passes = 0usize;
    while merge_adjacent_children_once(children) {
        merge_passes += 1;
        if merge_passes >= 64 {
            break;
        }
    }

    // Consolidate ChunkArray siblings into a single multi-chunk ChunkArray.
    // This merges N single-chunk (or small) ChunkArray children that form a dense
    // cluster into one ChunkArray leaf, drastically reducing BVH node count.
    consolidate_chunk_array_children(children, node.generator_version_hash);

    if children.len() == 1 {
        let child = children.pop().expect("single child");
        node.bounds = child.bounds;
        node.kind = child.kind;
        return;
    }

    sort_children_canonical(children);
    if children.len() != 2 {
        return;
    }
    let merge_alignment = children
        .iter()
        .map(|c| chunk_world_size_for_kind(Some(&c.kind)))
        .max()
        .unwrap_or(chunk_world_size_for_scale(0));
    if !branch_matches_split(node.bounds, children, merge_alignment) {
        // Only collapse two-child branches when they are the canonical binary partition
        // of this node's bounds. Two arbitrary non-overlapping children with the same kind
        // must not fill gaps in the parent region.
        return;
    }

    sort_children_canonical(children);
    let left_kind = children[0].kind.clone();
    let right_kind = children[1].kind.clone();
    match (left_kind, right_kind) {
        (RegionNodeKind::Empty, RegionNodeKind::Empty) => {
            node.kind = RegionNodeKind::Empty;
        }
        (RegionNodeKind::Uniform(a), RegionNodeKind::Uniform(b)) if a == b => {
            node.kind = RegionNodeKind::Uniform(a);
        }
        (RegionNodeKind::ProceduralRef(a), RegionNodeKind::ProceduralRef(b)) if a == b => {
            node.kind = RegionNodeKind::ProceduralRef(a);
        }
        _ => {}
    }
}

fn project_node_to_bounds(
    source_kind: &RegionNodeKind,
    source_bounds: Aabb4i,
    target_bounds: Aabb4i,
    generator_version_hash: u64,
) -> RegionTreeCore {
    if !target_bounds.is_valid() || !source_bounds.intersects(&target_bounds) {
        return RegionTreeCore {
            bounds: target_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash,
        };
    }

    let kind = match source_kind {
        RegionNodeKind::Empty => RegionNodeKind::Empty,
        RegionNodeKind::Uniform(block) => RegionNodeKind::Uniform(block.clone()),
        RegionNodeKind::ProceduralRef(generator_ref) => {
            RegionNodeKind::ProceduralRef(generator_ref.clone())
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            if let Some(sliced) = slice_chunk_array_to_bounds(chunk_array, target_bounds) {
                RegionNodeKind::ChunkArray(sliced)
            } else {
                // Bounds don't align to source scale — resample at coarsest aligned finer scale.
                let aligned_scale =
                    coarsest_aligned_scale(target_bounds, chunk_array.scale_exp);
                return resample_chunk_array_to_bounds(
                    chunk_array,
                    target_bounds,
                    aligned_scale,
                    generator_version_hash,
                );
            }
        }
        RegionNodeKind::Branch(children) => {
            let mut clipped_children = Vec::new();
            for child in children {
                let Some(clipped) = slice_node_to_bounds(child, target_bounds) else {
                    continue;
                };
                if matches!(clipped.kind, RegionNodeKind::Empty) {
                    continue;
                }
                clipped_children.push(clipped);
            }

            if clipped_children.is_empty() {
                RegionNodeKind::Empty
            } else if clipped_children.len() == 1 && clipped_children[0].bounds == target_bounds {
                return clipped_children.pop().expect("single projected child");
            } else {
                RegionNodeKind::Branch(clipped_children)
            }
        }
    };

    let mut projected = RegionTreeCore {
        bounds: target_bounds,
        kind,
        generator_version_hash,
    };
    if matches!(projected.kind, RegionNodeKind::Branch(_)) {
        normalize_chunk_node(&mut projected);
    }
    projected
}

fn query_chunk_payload_in_node(
    node: &RegionTreeCore,
    key_pos: ChunkKey,
) -> Option<(ResolvedChunkPayload, i8)> {
    query_chunk_payload_in_kind(&node.kind, node.bounds, key_pos)
}

fn query_chunk_payload_in_kind(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    key_pos: ChunkKey,
) -> Option<(ResolvedChunkPayload, i8)> {
    if !bounds.contains_chunk_world_min(key_pos) {
        return None;
    }
    match kind {
        RegionNodeKind::Empty => None,
        RegionNodeKind::Uniform(block) => {
            Some((ResolvedChunkPayload::uniform(block.clone()), block.scale_exp))
        }
        RegionNodeKind::ProceduralRef(_) => None,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_resolved_payload_at(chunk_array, key_pos)
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                if child.bounds.contains_chunk_world_min(key_pos) {
                    return query_chunk_payload_in_kind(&child.kind, child.bounds, key_pos);
                }
            }
            None
        }
    }
}

fn chunk_array_resolved_payload_at(
    chunk_array: &ChunkArrayData,
    key_pos: ChunkKey,
) -> Option<(ResolvedChunkPayload, i8)> {
    let payload = chunk_array_payload_at(chunk_array, key_pos)?;
    Some((
        ResolvedChunkPayload {
            payload,
            block_palette: chunk_array.block_palette.clone(),
        },
        chunk_array.scale_exp,
    ))
}

fn kind_has_non_empty_chunk_intersection(
    kind: &RegionNodeKind,
    kind_bounds: Aabb4i,
    query_bounds: Aabb4i,
) -> bool {
    if !kind_bounds.intersects(&query_bounds) {
        return false;
    }
    match kind {
        RegionNodeKind::Empty => false,
        RegionNodeKind::Uniform(block) => !block.is_air(),
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_has_non_empty_intersection(chunk_array, query_bounds)
        }
        RegionNodeKind::Branch(children) => children.iter().any(|child| {
            kind_has_non_empty_chunk_intersection(&child.kind, child.bounds, query_bounds)
        }),
    }
}

fn count_non_empty_chunks(kind: &RegionNodeKind, bounds: Aabb4i) -> usize {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => 0,
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                0
            } else {
                bounds.chunk_cell_count_at_scale(0).unwrap_or(0)
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return 0;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
            indices
                .into_iter()
                .filter(|idx| {
                    palette_non_empty
                        .get(*idx as usize)
                        .copied()
                        .unwrap_or(true)
                })
                .count()
        }
        RegionNodeKind::Branch(children) => children
            .iter()
            .map(|child| count_non_empty_chunks(&child.kind, child.bounds))
            .sum(),
    }
}

fn collect_chunks_from_kind(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ResolvedChunkPayload)>,
) {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            let resolved = ResolvedChunkPayload::uniform(block.clone());
            let (lmin, lmax) = bounds.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((chunk_key_from_lattice([lx, ly, lz, lw], 0), resolved.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let se = chunk_array.scale_exp;
            let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(se) else {
                return;
            };
            let (lmin, lmax) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            let local = [
                                (lx - lmin[0]) as usize,
                                (ly - lmin[1]) as usize,
                                (lz - lmin[2]) as usize,
                                (lw - lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let Some(payload) =
                                chunk_array.chunk_palette.get(*palette_idx as usize)
                            else {
                                continue;
                            };
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], se),
                                ResolvedChunkPayload {
                                    payload: payload.clone(),
                                    block_palette: chunk_array.block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunks_from_kind(&child.kind, child.bounds, out);
            }
        }
    }
}

fn collect_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ResolvedChunkPayload)>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            let resolved = ResolvedChunkPayload::uniform(block.clone());
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((chunk_key_from_lattice([lx, ly, lz, lw], 0), resolved.clone()));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = intersect_aabb(chunk_array.bounds, query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let se = chunk_array.scale_exp;
            let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(se) else {
                return;
            };
            let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_chunk_lattice_bounds(se);
            for lw in int_lmin[3]..=int_lmax[3] {
                for lz in int_lmin[2]..=int_lmax[2] {
                    for ly in int_lmin[1]..=int_lmax[1] {
                        for lx in int_lmin[0]..=int_lmax[0] {
                            let local = [
                                (lx - ca_lmin[0]) as usize,
                                (ly - ca_lmin[1]) as usize,
                                (lz - ca_lmin[2]) as usize,
                                (lw - ca_lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            let Some(palette_idx) = indices.get(linear) else {
                                continue;
                            };
                            let Some(payload) =
                                chunk_array.chunk_palette.get(*palette_idx as usize)
                            else {
                                continue;
                            };
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], se),
                                ResolvedChunkPayload {
                                    payload: payload.clone(),
                                    block_palette: chunk_array.block_palette.clone(),
                                },
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunks_from_kind_in_bounds(&child.kind, child.bounds, query_bounds, out);
            }
        }
    }
}

fn collect_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = intersect_aabb(bounds, query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(_) => {
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push(chunk_key_from_lattice([lx, ly, lz, lw], 0));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let se = chunk_array.scale_exp;
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(se);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push(chunk_key_from_lattice([lx, ly, lz, lw], se));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunk_keys_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

fn chunk_array_has_non_empty_intersection(
    chunk_array: &ChunkArrayData,
    query_bounds: Aabb4i,
) -> bool {
    let Some(intersection) = intersect_aabb(chunk_array.bounds, query_bounds) else {
        return false;
    };
    let Ok(indices) = chunk_array.decode_dense_indices() else {
        return true;
    };
    let se = chunk_array.scale_exp;
    let Some(extents) = chunk_array.bounds.chunk_extents_at_scale(se) else {
        return true;
    };
    let palette_non_empty = chunk_array_palette_non_empty_mask(chunk_array);
    let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    let (int_lmin, int_lmax) = intersection.to_chunk_lattice_bounds(se);

    for lw in int_lmin[3]..=int_lmax[3] {
        for lz in int_lmin[2]..=int_lmax[2] {
            for ly in int_lmin[1]..=int_lmax[1] {
                for lx in int_lmin[0]..=int_lmax[0] {
                    let local = [
                        (lx - ca_lmin[0]) as usize,
                        (ly - ca_lmin[1]) as usize,
                        (lz - ca_lmin[2]) as usize,
                        (lw - ca_lmin[3]) as usize,
                    ];
                    let linear = linear_cell_index(local, extents);
                    let Some(palette_idx) = indices.get(linear) else {
                        return true;
                    };
                    if palette_non_empty
                        .get(*palette_idx as usize)
                        .copied()
                        .unwrap_or(true)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn resolved_option_is_semantically_empty(resolved: Option<&ResolvedChunkPayload>) -> bool {
    resolved
        .map(|r| !r.has_solid_block())
        .unwrap_or(true)
}

fn resolved_option_matches_block(
    block: &BlockData,
    resolved: Option<&ResolvedChunkPayload>,
) -> bool {
    match resolved {
        Some(r) => {
            let r = canonicalize_resolved_payload(r.clone());
            match &r.payload {
                ChunkPayload::Uniform(idx) => {
                    let resolved_block = r
                        .block_palette
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or(BlockData::AIR);
                    &resolved_block == block
                }
                _ => false,
            }
        }
        None => block.is_air(),
    }
}

fn resolved_option_matches_existing(
    existing: Option<ResolvedChunkPayload>,
    incoming: Option<&ResolvedChunkPayload>,
) -> bool {
    match incoming {
        Some(incoming) => {
            let incoming_c = canonicalize_resolved_payload(incoming.clone());
            match existing {
                Some(existing) => {
                    let existing_c = canonicalize_resolved_payload(existing);
                    resolved_payloads_semantically_equal(&existing_c, &incoming_c)
                }
                None => !incoming_c.has_solid_block(),
            }
        }
        None => existing
            .as_ref()
            .map(|r| !r.has_solid_block())
            .unwrap_or(true),
    }
}

fn resolved_payloads_semantically_equal(a: &ResolvedChunkPayload, b: &ResolvedChunkPayload) -> bool {
    // Compare by resolving both through their block palettes
    let a_dense = a.payload.dense_materials();
    let b_dense = b.payload.dense_materials();
    match (a_dense, b_dense) {
        (Ok(ad), Ok(bd)) => {
            if ad.len() != bd.len() {
                return false;
            }
            ad.iter().zip(bd.iter()).all(|(ai, bi)| {
                let a_block = a.block_palette.get(*ai as usize).cloned().unwrap_or(BlockData::AIR);
                let b_block = b.block_palette.get(*bi as usize).cloned().unwrap_or(BlockData::AIR);
                a_block == b_block
            })
        }
        _ => false,
    }
}

fn non_empty_kinds_semantically_equal_in_bounds(
    lhs_kind: &RegionNodeKind,
    lhs_bounds: Aabb4i,
    rhs_kind: &RegionNodeKind,
    rhs_bounds: Aabb4i,
    bounds: Aabb4i,
) -> bool {
    const MAX_COMPARE_CELLS: usize = 4096;
    if bounds
        .chunk_cell_count_at_scale(0)
        .map(|count| count > MAX_COMPARE_CELLS)
        .unwrap_or(true)
    {
        return false;
    }

    let mut lhs_chunks = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(lhs_kind, lhs_bounds, bounds, &mut lhs_chunks);
    lhs_chunks.sort_unstable_by_key(|(key, _)| *key);

    let mut rhs_chunks = Vec::new();
    collect_non_empty_chunks_from_kind_in_bounds(rhs_kind, rhs_bounds, bounds, &mut rhs_chunks);
    rhs_chunks.sort_unstable_by_key(|(key, _)| *key);

    if lhs_chunks.len() != rhs_chunks.len() {
        return false;
    }
    lhs_chunks.iter().zip(rhs_chunks.iter()).all(|((lk, lp), (rk, rp))| {
        lk == rk && resolved_payloads_semantically_equal(lp, rp)
    })
}

fn chunk_array_palette_non_empty_mask(chunk_array: &ChunkArrayData) -> Vec<bool> {
    chunk_array
        .chunk_palette
        .iter()
        .map(|p| payload_has_solid_material_in_context(p, &chunk_array.block_palette))
        .collect()
}

fn chunk_array_payload_at(chunk_array: &ChunkArrayData, key_pos: ChunkKey) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk_world_min(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    chunk_array_payload_at_with_dense_indices(chunk_array, &dense_indices, key_pos)
}

fn chunk_array_payload_at_with_dense_indices(
    chunk_array: &ChunkArrayData,
    dense_indices: &[u16],
    key_pos: ChunkKey,
) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk_world_min(key_pos) {
        return None;
    }
    let se = chunk_array.scale_exp;
    let extents = chunk_array.bounds.chunk_extents_at_scale(se)?;
    let (ca_lmin, _) = chunk_array.bounds.to_chunk_lattice_bounds(se);
    let lpos = [
        lattice_from_fixed(key_pos[0], se),
        lattice_from_fixed(key_pos[1], se),
        lattice_from_fixed(key_pos[2], se),
        lattice_from_fixed(key_pos[3], se),
    ];
    let local = [
        (lpos[0] - ca_lmin[0]) as usize,
        (lpos[1] - ca_lmin[1]) as usize,
        (lpos[2] - ca_lmin[2]) as usize,
        (lpos[3] - ca_lmin[3]) as usize,
    ];
    let linear = linear_cell_index(local, extents);
    let palette_idx = *dense_indices.get(linear)? as usize;
    chunk_array.chunk_palette.get(palette_idx).cloned()
}


/// Canonicalize the storage format of a ChunkPayload (opaque palette indices).
/// Converts Empty → Uniform(0), collapses all-same dense into Uniform.
fn canonicalize_payload_format(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        other => other,
    };
    let Ok(dense) = payload.dense_materials() else {
        return payload;
    };
    if dense.is_empty() {
        return payload;
    }
    let first = dense[0];
    if dense.iter().all(|m| *m == first) {
        ChunkPayload::Uniform(first)
    } else {
        payload
    }
}

fn canonicalize_resolved_payload(resolved: ResolvedChunkPayload) -> ResolvedChunkPayload {
    ResolvedChunkPayload {
        payload: canonicalize_payload_format(resolved.payload),
        block_palette: resolved.block_palette,
    }
}

fn repeated_payload_kind_resolved_at_scale(
    bounds: Aabb4i,
    resolved: ResolvedChunkPayload,
    scale_exp: i8,
) -> RegionNodeKind {
    let Some(cell_count) = bounds.chunk_cell_count_at_scale(scale_exp) else {
        return RegionNodeKind::Empty;
    };
    let indices = vec![0u16; cell_count];
    match ChunkArrayData::from_dense_indices_with_block_palette(
        bounds,
        vec![resolved.payload],
        indices,
        Some(0),
        resolved.block_palette,
        scale_exp,
    ) {
        Ok(chunk_array) => RegionNodeKind::ChunkArray(chunk_array),
        Err(_) => RegionNodeKind::Empty,
    }
}

fn kind_from_resolved_value_at_scale(
    bounds: Aabb4i,
    value: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> RegionNodeKind {
    let Some(resolved) = value else {
        return RegionNodeKind::Empty;
    };
    let resolved = canonicalize_resolved_payload(resolved);
    match &resolved.payload {
        ChunkPayload::Uniform(idx) => {
            let block = resolved
                .block_palette
                .get(*idx as usize)
                .cloned()
                .unwrap_or(BlockData::AIR);
            // Preserve the block's original scale_exp — it's metadata about the
            // block's visual/semantic scale (e.g. a scale-3 block placed by the
            // player), not the tree node's storage granularity.
            RegionNodeKind::Uniform(block)
        }
        _ => repeated_payload_kind_resolved_at_scale(bounds, resolved, scale_exp),
    }
}

/// Remap block-palette indices inside a single `ChunkPayload` using the given remap table.
/// `remap[old_idx]` gives the new index in the merged block palette.
fn remap_chunk_payload_block_indices(payload: &ChunkPayload, remap: &[u16]) -> ChunkPayload {
    match payload {
        ChunkPayload::Empty => ChunkPayload::Empty,
        ChunkPayload::Uniform(idx) => {
            let new_idx = remap.get(*idx as usize).copied().unwrap_or(*idx);
            ChunkPayload::Uniform(new_idx)
        }
        ChunkPayload::Dense16 { materials } => ChunkPayload::Dense16 {
            materials: materials
                .iter()
                .map(|idx| remap.get(*idx as usize).copied().unwrap_or(*idx))
                .collect(),
        },
        ChunkPayload::PalettePacked {
            palette,
            bit_width,
            packed_indices,
        } => ChunkPayload::PalettePacked {
            palette: palette
                .iter()
                .map(|idx| remap.get(*idx as usize).copied().unwrap_or(*idx))
                .collect(),
            bit_width: *bit_width,
            packed_indices: packed_indices.clone(),
        },
    }
}

fn payload_has_solid_material_in_context(
    payload: &ChunkPayload,
    block_palette: &[BlockData],
) -> bool {
    match payload {
        ChunkPayload::Empty => false,
        ChunkPayload::Uniform(idx) => block_palette
            .get(*idx as usize)
            .map(|b| !b.is_air())
            .unwrap_or(false),
        ChunkPayload::Dense16 { materials } => materials.iter().any(|idx| {
            block_palette
                .get(*idx as usize)
                .map(|b| !b.is_air())
                .unwrap_or(false)
        }),
        ChunkPayload::PalettePacked { palette, .. } => palette.iter().any(|idx| {
            block_palette
                .get(*idx as usize)
                .map(|b| !b.is_air())
                .unwrap_or(false)
        }),
    }
}

/// Check if bounds represent exactly one chunk at the given scale.
fn is_single_chunk_bounds(bounds: Aabb4i, scale_exp: i8) -> bool {
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    let step = step_for_scale(scale_exp);
    let chunk_world_size = cs.saturating_mul(step);
    (0..4).all(|i| bounds.max[i] - bounds.min[i] == chunk_world_size)
}

/// Check if bounds represent a single chunk at any possible scale.
/// Used when the exact scale is unknown (e.g. lazy_drop).
fn is_single_chunk_bounds_any_scale(bounds: Aabb4i) -> bool {
    if !bounds.is_valid() {
        return false;
    }
    // All axes must have the same span, and the span must be CS * 2^s for some s.
    let span = bounds.max[0] - bounds.min[0];
    if !(1..4).all(|i| bounds.max[i] - bounds.min[i] == span) {
        return false;
    }
    // span must be CS * step = CS * 2^s = 2^(3+s). Check it's a power of 2 >= CS.
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    if span < cs {
        return false;
    }
    let bits = span.to_bits();
    bits > 0 && (bits & (bits - 1)) == 0
}

/// Compute the world-space extent of a chunk given its key and scale.
///
/// Now that tree node bounds ARE world-space, this is equivalent to
/// `Aabb4i::chunk_world_bounds(key, scale_exp)` for single-chunk bounds.
/// For multi-chunk regions, the bounds already represent the world extent.
///
/// Retained for callers outside tree.rs that construct extents from keys.
pub fn chunk_spatial_extent(key: ChunkKey, scale_exp: i8) -> Aabb4i {
    Aabb4i::chunk_world_bounds(key, scale_exp)
}

/// Find the coarsest scale at which `bounds` aligns to the chunk grid.
///
/// A bounds "aligns" at scale `s` when `bounds.chunk_extents_at_scale(s)` returns
/// `Some` — i.e. each axis extent is an exact integer number of chunks at that scale.
///
/// Searches from `ceiling` downward (finer scales). Panics if no valid scale is
/// found within 10 levels — this is structurally impossible for bounds produced by
/// `subtract_aabb` against chunk-aligned inputs.
pub fn coarsest_aligned_scale(bounds: Aabb4i, ceiling: i8) -> i8 {
    for s in (ceiling.saturating_sub(10)..=ceiling).rev() {
        if bounds.chunk_extents_at_scale(s).is_some() {
            // Verify the bounds actually align: the lattice round-trip must
            // reconstruct the original bounds exactly. A sub-chunk overlap
            // produces valid extents but a different reconstructed AABB.
            let (lmin, lmax) = bounds.to_chunk_lattice_bounds(s);
            let reconstructed = Aabb4i::from_lattice_bounds(lmin, lmax, s);
            if reconstructed == bounds {
                return s;
            }
        }
    }
    panic!(
        "coarsest_aligned_scale: no valid scale found for bounds {:?} with ceiling {}",
        bounds, ceiling
    );
}

// ---------------------------------------------------------------------------
// BVH spatial queries — point query and raycast
// ---------------------------------------------------------------------------

/// Result of a BVH point query: the block and its world-space AABB.
#[derive(Clone, Debug)]
pub struct BvhBlockHit {
    pub block: BlockData,
    /// World-space extent of the block, accounting for `block.scale_exp`.
    pub bounds: Aabb4i,
}

/// Result of a BVH raycast: block hit info plus the ray parameter.
#[derive(Clone, Debug)]
pub struct BvhRayHit {
    pub block: BlockData,
    /// World-space extent of the hit block.
    pub bounds: Aabb4i,
    /// Ray parameter at entry to the block's AABB.
    pub t: ChunkCoord,
}

/// Given a fixed-point position and a leaf's scale, compute the chunk key,
/// voxel index, and the cell's own AABB (at the chunk's cell resolution).
fn cell_at_point(pos: [ChunkCoord; 4], scale_exp: i8) -> (ChunkKey, usize, Aabb4i) {
    let step = step_for_scale(scale_exp);
    let cs = CHUNK_SIZE as i32;
    let mut chunk_key = [ChunkCoord::ZERO; 4];
    let mut local = [0usize; 4];
    let mut cell_min = [ChunkCoord::ZERO; 4];
    let mut cell_max = [ChunkCoord::ZERO; 4];
    for axis in 0..4 {
        // Cell lattice coordinate at this scale
        let cell_lat = lattice_from_fixed(pos[axis], scale_exp);
        // Chunk lattice coordinate
        let chunk_lat = (cell_lat as i64).div_euclid(cs as i64) as i32;
        chunk_key[axis] = fixed_from_lattice(chunk_lat, scale_exp);
        local[axis] = (cell_lat as i64).rem_euclid(cs as i64) as usize;
        cell_min[axis] = fixed_from_lattice(cell_lat, scale_exp);
        cell_max[axis] = cell_min[axis].saturating_add(step);
    }
    let idx = local[3] * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
        + local[2] * CHUNK_SIZE * CHUNK_SIZE
        + local[1] * CHUNK_SIZE
        + local[0];
    (chunk_key, idx, Aabb4i::new(cell_min, cell_max))
}

/// Expand a cell-level AABB to the full block extent using `block.scale_exp`.
///
/// When a block has a coarser `scale_exp` than the chunk it's stored in, it
/// occupies multiple cells. This snaps the cell bounds outward to the block's
/// own grid.
fn block_bounds(cell_min: [ChunkCoord; 4], chunk_scale_exp: i8, block: &BlockData) -> Aabb4i {
    let bs = block.scale_exp;
    if bs <= chunk_scale_exp {
        // Block is at same or finer resolution than the chunk cell — the cell
        // AABB is already the block extent.
        let step = step_for_scale(chunk_scale_exp);
        return Aabb4i::new(cell_min, std::array::from_fn(|i| cell_min[i].saturating_add(step)));
    }
    // Block is coarser than the cell: snap to the block's grid.
    let block_step = step_for_scale(bs);
    let mut bmin = [ChunkCoord::ZERO; 4];
    let mut bmax = [ChunkCoord::ZERO; 4];
    for axis in 0..4 {
        let lat = lattice_from_fixed(cell_min[axis], bs);
        bmin[axis] = fixed_from_lattice(lat, bs);
        bmax[axis] = bmin[axis].saturating_add(block_step);
    }
    Aabb4i::new(bmin, bmax)
}

/// Does the half-open AABB `[min, max)` contain the point?
fn aabb_contains_point(bounds: &Aabb4i, pos: [ChunkCoord; 4]) -> bool {
    bounds.contains_point(pos)
}

/// Recursive BVH point query.
fn bvh_point_query(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    pos: [ChunkCoord; 4],
) -> Option<BvhBlockHit> {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => None,

        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return None;
            }
            if !aabb_contains_point(&bounds, pos) {
                return None;
            }
            // For Uniform, every cell has the same block. Compute cell bounds
            // at scale 0 and expand to the block's own scale.
            let (_, _, cell_aabb) = cell_at_point(pos, 0);
            let bb = block_bounds(cell_aabb.min, 0, block);
            Some(BvhBlockHit { block: block.clone(), bounds: bb })
        }

        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
            if !aabb_contains_point(&ca.bounds, pos) {
                return None;
            }
            let (chunk_key, cell_idx, cell_aabb) = cell_at_point(pos, se);
            if !ca.bounds.contains_chunk_world_min(chunk_key) {
                return None;
            }
            let (resolved, _) = chunk_array_resolved_payload_at(ca, chunk_key)?;
            let block = resolved.block_at(cell_idx);
            if block.is_air() {
                return None;
            }
            let bb = block_bounds(cell_aabb.min, se, &block);
            Some(BvhBlockHit { block, bounds: bb })
        }

        RegionNodeKind::Branch(children) => {
            for child in children {
                if let Some(hit) = bvh_point_query(&child.kind, child.bounds, pos) {
                    return Some(hit);
                }
            }
            None
        }
    }
}

/// BVH point query that distinguishes "no data" from "data says air".
///
/// Returns `Some(block)` when a data-containing node covers `pos`, including
/// `Some(AIR)` for air cells within ChunkArrays/Uniforms. Returns `None` only
/// when no data node covers the position (Empty or out of bounds).
fn bvh_block_data_at_point(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    pos: [ChunkCoord; 4],
) -> Option<BlockData> {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => None,

        RegionNodeKind::Uniform(block) => {
            if !aabb_contains_point(&bounds, pos) {
                return None;
            }
            Some(block.clone())
        }

        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
            if !aabb_contains_point(&ca.bounds, pos) {
                return None;
            }
            let (chunk_key, cell_idx, _) = cell_at_point(pos, se);
            if !ca.bounds.contains_chunk_world_min(chunk_key) {
                return None;
            }
            let (resolved, _) = chunk_array_resolved_payload_at(ca, chunk_key)?;
            Some(resolved.block_at(cell_idx))
        }

        RegionNodeKind::Branch(children) => {
            for child in children {
                if let Some(block) = bvh_block_data_at_point(&child.kind, child.bounds, pos) {
                    return Some(block);
                }
            }
            None
        }
    }
}

/// 4D ray–AABB slab intersection. Returns `(t_enter, t_exit)` or `None`.
///
/// `direction` components may be zero; those axes use position-only checks.
fn ray_aabb_intersect(
    origin: [ChunkCoord; 4],
    direction: [ChunkCoord; 4],
    aabb: &Aabb4i,
) -> Option<(ChunkCoord, ChunkCoord)> {
    let mut t_min = ChunkCoord::MIN;
    let mut t_max = ChunkCoord::MAX;
    for axis in 0..4 {
        let d = direction[axis];
        if d == ChunkCoord::ZERO {
            // Ray is parallel to this axis — check if origin is inside slab.
            if origin[axis] < aabb.min[axis] || origin[axis] >= aabb.max[axis] {
                return None;
            }
        } else {
            let inv_d_positive = d > ChunkCoord::ZERO;
            let (t_near, t_far) = if inv_d_positive {
                (
                    (aabb.min[axis] - origin[axis]) / d,
                    (aabb.max[axis] - origin[axis]) / d,
                )
            } else {
                (
                    (aabb.max[axis] - origin[axis]) / d,
                    (aabb.min[axis] - origin[axis]) / d,
                )
            };
            if t_near > t_min { t_min = t_near; }
            if t_far < t_max { t_max = t_far; }
            if t_min > t_max {
                return None;
            }
        }
    }
    Some((t_min, t_max))
}

/// Recursive BVH raycast. Returns the nearest solid hit.
fn bvh_raycast(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    origin: [ChunkCoord; 4],
    direction: [ChunkCoord; 4],
    max_t: ChunkCoord,
) -> Option<BvhRayHit> {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => None,

        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return None;
            }
            let (t_enter, _) = ray_aabb_intersect(origin, direction, &bounds)?;
            if t_enter > max_t {
                return None;
            }
            // Compute the point where the ray enters the uniform region.
            // Nudge t_sample slightly past t_enter so the sample lands just
            // inside the AABB rather than on its boundary (where cell_at_point
            // could snap to a cell outside the region).
            let nudge = ChunkCoord::from_bits(1);
            let t_sample = (if t_enter > ChunkCoord::ZERO { t_enter } else { ChunkCoord::ZERO })
                .saturating_add(nudge);
            let hit_pos: [ChunkCoord; 4] = std::array::from_fn(|i| {
                origin[i].saturating_add(direction[i].saturating_mul(t_sample))
            });
            let (_, _, cell_aabb) = cell_at_point(hit_pos, 0);
            let bb = block_bounds(cell_aabb.min, 0, block);
            Some(BvhRayHit { block: block.clone(), bounds: bb, t: t_enter.max(ChunkCoord::ZERO) })
        }

        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
            let (t_enter, t_exit) = ray_aabb_intersect(origin, direction, &ca.bounds)?;
            if t_enter > max_t {
                return None;
            }
            // Step through cells along the ray within this ChunkArray.
            // Use a simple parametric march: sample cells at each crossing.
            let step = step_for_scale(se);
            let cell_size_f64 = step.to_num::<f64>();
            let dir_f64: [f64; 4] = std::array::from_fn(|i| direction[i].to_num::<f64>());

            // Compute step size: half a cell diagonal to avoid skipping cells.
            let dir_len_sq: f64 = dir_f64.iter().map(|d| d * d).sum();
            if dir_len_sq < 1e-20 {
                return None;
            }
            let dt = cell_size_f64 * 0.4 / dir_len_sq.sqrt();

            let t_start = t_enter.max(ChunkCoord::ZERO).to_num::<f64>();
            let t_end = t_exit.min(max_t).to_num::<f64>();
            if t_start > t_end {
                return None;
            }

            let mut t = t_start;
            let mut prev_cell = ([ChunkCoord::MIN; 4], usize::MAX); // sentinel
            while t <= t_end {
                let hit_pos: [ChunkCoord; 4] = std::array::from_fn(|i| {
                    ChunkCoord::from_num(origin[i].to_num::<f64>() + dir_f64[i] * t)
                });
                let (chunk_key, cell_idx, cell_aabb) = cell_at_point(hit_pos, se);
                // Skip if we're still in the same cell as last step.
                if (chunk_key, cell_idx) != prev_cell || t == t_start {
                    prev_cell = (chunk_key, cell_idx);
                    if ca.bounds.contains_chunk_world_min(chunk_key) {
                        if let Some((resolved, _)) = chunk_array_resolved_payload_at(ca, chunk_key) {
                            let block = resolved.block_at(cell_idx);
                            if !block.is_air() {
                                let bb = block_bounds(cell_aabb.min, se, &block);
                                // Compute true t_enter for the block's bounds.
                                let block_t = ray_aabb_intersect(origin, direction, &bb)
                                    .map(|(te, _)| te.max(ChunkCoord::ZERO))
                                    .unwrap_or(ChunkCoord::from_num(t));
                                return Some(BvhRayHit { block, bounds: bb, t: block_t });
                            }
                        }
                    }
                }
                t += dt;
            }
            None
        }

        RegionNodeKind::Branch(children) => {
            // Collect children that the ray intersects, sorted by t_enter.
            let mut candidates: Vec<(ChunkCoord, usize)> = Vec::new();
            for (idx, child) in children.iter().enumerate() {
                if let Some((t_enter, _)) = ray_aabb_intersect(origin, direction, &child.bounds) {
                    if t_enter <= max_t {
                        candidates.push((t_enter, idx));
                    }
                }
            }
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut best: Option<BvhRayHit> = None;
            for (_, idx) in candidates {
                let child = &children[idx];
                if let Some(ref b) = best {
                    if let Some((t_enter, _)) = ray_aabb_intersect(origin, direction, &child.bounds) {
                        if t_enter > b.t {
                            continue;
                        }
                    }
                }
                let effective_max = best.as_ref().map(|b| b.t).unwrap_or(max_t);
                if let Some(hit) = bvh_raycast(&child.kind, child.bounds, origin, direction, effective_max) {
                    best = Some(match best {
                        Some(prev) if prev.t <= hit.t => prev,
                        _ => hit,
                    });
                }
            }
            best
        }
    }
}

fn find_leaf_chunks_in_spatial_range_recursive(
    kind: &RegionNodeKind,
    kind_bounds: Aabb4i,
    query: &Aabb4i,
    out: &mut Vec<(ChunkKey, i8)>,
) {
    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return;
            }
            let se = block.scale_exp;
            // Bounds are already world-space.
            if !kind_bounds.intersects(query) {
                return;
            }
            if is_single_chunk_bounds(kind_bounds, 0) {
                out.push((kind_bounds.chunk_key_from_world_bounds(), se));
            } else {
                let (lmin, lmax) = kind_bounds.to_chunk_lattice_bounds(0);
                for lw in lmin[3]..=lmax[3] {
                    for lz in lmin[2]..=lmax[2] {
                        for ly in lmin[1]..=lmax[1] {
                            for lx in lmin[0]..=lmax[0] {
                                let ck = chunk_key_from_lattice([lx, ly, lz, lw], 0);
                                let ck_world = Aabb4i::chunk_world_bounds(ck, 0);
                                if ck_world.intersects(query) {
                                    out.push((ck, se));
                                }
                            }
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
            // Bounds are already world-space.
            if !ca.bounds.intersects(query) {
                return;
            }
            let Ok(indices) = ca.decode_dense_indices() else {
                return;
            };
            let palette_non_empty = chunk_array_palette_non_empty_mask(ca);
            let Some(extents) = ca.bounds.chunk_extents_at_scale(se) else {
                return;
            };
            let (ca_lmin, _) = ca.bounds.to_chunk_lattice_bounds(se);
            let (lmin, lmax) = ca.bounds.to_chunk_lattice_bounds(se);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            let local = [
                                (lx - ca_lmin[0]) as usize,
                                (ly - ca_lmin[1]) as usize,
                                (lz - ca_lmin[2]) as usize,
                                (lw - ca_lmin[3]) as usize,
                            ];
                            let linear = linear_cell_index(local, extents);
                            if let Some(&palette_idx) = indices.get(linear) {
                                if !palette_non_empty
                                    .get(palette_idx as usize)
                                    .copied()
                                    .unwrap_or(true)
                                {
                                    continue;
                                }
                            }
                            let ck = chunk_key_from_lattice([lx, ly, lz, lw], se);
                            let ck_world = Aabb4i::chunk_world_bounds(ck, se);
                            if ck_world.intersects(query) {
                                out.push((ck, se));
                            }
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                find_leaf_chunks_in_spatial_range_recursive(
                    &child.kind,
                    child.bounds,
                    query,
                    out,
                );
            }
        }
    }
}

fn branch_matches_split(bounds: Aabb4i, children: &[RegionTreeCore], alignment: ChunkCoord) -> bool {
    if children.len() != 2 {
        return false;
    }
    let Some((left, right)) = split_bounds_longest_axis(bounds, alignment) else {
        return false;
    };
    (children[0].bounds == left && children[1].bounds == right)
        || (children[0].bounds == right && children[1].bounds == left)
}

/// Split half-open bounds along the longest axis into two halves.
///
/// `alignment` is the world-space chunk size (CS * step) for the coarsest
/// scale present. The midpoint is snapped to an alignment boundary.
/// With half-open ranges: left = [min, mid), right = [mid, max).
fn split_bounds_longest_axis(bounds: Aabb4i, alignment: ChunkCoord) -> Option<(Aabb4i, Aabb4i)> {
    if !bounds.is_valid() {
        return None;
    }

    let align_bits = alignment.to_bits();
    if align_bits == 0 {
        return None;
    }
    let to_lattice = |coord: ChunkCoord| -> i64 { coord.to_bits() / align_bits };
    let from_lattice = |lattice: i64| -> ChunkCoord { ChunkCoord::from_bits(lattice * align_bits) };

    // Half-open span: no +1 needed.
    let mut spans = [0i64; 4];
    for i in 0..4 {
        spans[i] = to_lattice(bounds.max[i]) - to_lattice(bounds.min[i]);
    }
    let mut axis = 0usize;
    for idx in 1..4 {
        if spans[idx] > spans[axis] {
            axis = idx;
        }
    }
    if spans[axis] <= 1 {
        return None;
    }

    let left_len = spans[axis] / 2;
    let mid = from_lattice(to_lattice(bounds.min[axis]) + left_len);

    let mut left = bounds;
    left.max[axis] = mid;

    let mut right = bounds;
    right.min[axis] = mid;

    Some((left, right))
}

fn sort_children_canonical(children: &mut [RegionTreeCore]) {
    children.sort_unstable_by_key(|child| {
        (
            child.bounds.min[0],
            child.bounds.min[1],
            child.bounds.min[2],
            child.bounds.min[3],
            child.bounds.max[0],
            child.bounds.max[1],
            child.bounds.max[2],
            child.bounds.max[3],
        )
    });
}

fn expand_root_once(root: Box<RegionTreeCore>, target: Aabb4i, _alignment: ChunkCoord) -> Box<RegionTreeCore> {
    if aabb_contains_aabb(root.bounds, target) {
        return root;
    }

    let mut old_root = *root;
    let old_bounds = old_root.bounds;
    normalize_chunk_node(&mut old_root);

    // Find the first axis where the target exceeds the root bounds.
    let axis = (0..4)
        .find(|&axis| {
            target.min[axis] < old_bounds.min[axis] || target.max[axis] > old_bounds.max[axis]
        })
        .unwrap_or(0);

    // Half-open span of the current root along this axis.
    let span = old_bounds.max[axis] - old_bounds.min[axis];

    let mut new_bounds = old_bounds;
    let mut sibling_bounds = old_bounds;
    if target.min[axis] < old_bounds.min[axis] {
        // Expand leftward: sibling covers [expanded, old_min).
        let expanded = old_bounds.min[axis].saturating_sub(span);
        let expanded = if expanded >= old_bounds.min[axis] {
            target.min[axis]
        } else {
            expanded
        };
        new_bounds.min[axis] = expanded;
        sibling_bounds.min[axis] = expanded;
        sibling_bounds.max[axis] = old_bounds.min[axis]; // half-open: sibling ends where root starts
    } else {
        // Expand rightward: sibling covers [old_max, expanded).
        let expanded = old_bounds.max[axis].saturating_add(span);
        let expanded = if expanded <= old_bounds.max[axis] {
            target.max[axis]
        } else {
            expanded
        };
        new_bounds.max[axis] = expanded;
        sibling_bounds.min[axis] = old_bounds.max[axis]; // half-open: sibling starts where root ends
        sibling_bounds.max[axis] = expanded;
    }

    let sibling = RegionTreeCore {
        bounds: sibling_bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: old_root.generator_version_hash,
    };
    let mut children = vec![old_root, sibling];
    sort_children_canonical(&mut children);
    Box::new(RegionTreeCore {
        bounds: new_bounds,
        kind: RegionNodeKind::Branch(children),
        generator_version_hash: 0,
    })
}

fn intersect_aabb(a: Aabb4i, b: Aabb4i) -> Option<Aabb4i> {
    let min = [
        a.min[0].max(b.min[0]),
        a.min[1].max(b.min[1]),
        a.min[2].max(b.min[2]),
        a.min[3].max(b.min[3]),
    ];
    let max = [
        a.max[0].min(b.max[0]),
        a.max[1].min(b.max[1]),
        a.max[2].min(b.max[2]),
        a.max[3].min(b.max[3]),
    ];
    (min[0] < max[0] && min[1] < max[1] && min[2] < max[2] && min[3] < max[3])
        .then_some(Aabb4i::new(min, max))
}
