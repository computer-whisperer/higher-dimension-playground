use super::chunk_array_ops::{
    optimize_subtree_in_bounds as optimize_subtree_in_bounds_impl, overlay_non_empty_leaves,
};
use super::tree_edit::{
    lazy_drop_outside_node, set_chunk_recursive, slice_non_empty_region_core_in_bounds,
    slice_region_core_in_bounds, splice_node_with_core, splice_node_with_non_empty_core,
};
use super::tree_normalize::normalize_chunk_node;
use super::tree_query::{
    bvh_block_data_at_point, bvh_point_query, bvh_raycast,
    collect_chunk_entries_from_kind_in_bounds, collect_chunk_keys_from_kind_in_bounds,
    collect_chunks_from_kind, collect_chunks_from_kind_in_bounds,
    collect_non_empty_chunk_keys_from_kind_in_bounds, count_non_empty_chunks,
    find_leaf_chunks_in_spatial_range_recursive, kind_has_non_empty_chunk_intersection,
    query_chunk_payload_in_node, BvhBlockHit, BvhRayHit,
};
use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::{step_for_scale, Aabb4i, ChunkCoord};
use crate::shared::voxel::{BlockData, CHUNK_SIZE};

/// A spatial tree storing voxel chunk data with world-space half-open AABB bounds.
///
/// # Invariants
///
/// - **Non-overlapping siblings**: children of the same branch never spatially overlap.
/// - **Containment**: every child's bounds are fully within its parent's bounds.
/// - **Overwrite semantics**: `set_chunk_at_scale` replaces the full spatial region
///   of the chunk. A scale-2 chunk occupies 32x32x32x32 world units; inserting it
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
    pub fn block_and_bounds_at(&self, pos: [ChunkCoord; 4]) -> Option<BvhBlockHit> {
        self.root
            .as_ref()
            .and_then(|node| bvh_point_query(&node.kind, node.bounds, pos))
    }

    /// Look up the block at a fixed-point position, distinguishing "no data"
    /// from "data says air".
    pub fn block_data_at(&self, pos: [ChunkCoord; 4]) -> Option<BlockData> {
        self.root
            .as_ref()
            .and_then(|node| bvh_block_data_at_point(&node.kind, node.bounds, pos))
    }

    /// Cast a ray through the BVH and return the nearest solid block hit.
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
        self.set_chunk_at_scale(key, resolved, 0).is_some()
    }

    /// Set a chunk at a specific scale. The key is already in fixed-point coordinates.
    ///
    /// The chunk's world-space extent is `chunk_world_bounds(key, scale_exp)`.
    /// All existing data within that spatial region is overwritten, regardless
    /// of what scale it was stored at. Passing `None` for `resolved` clears
    /// the region (sets it to empty/air).
    ///
    /// Returns `Some(affected_bounds)` if the tree was modified, where
    /// `affected_bounds` is the world-space region that was carved or replaced.
    /// Returns `None` if no change was made.
    pub fn set_chunk_at_scale(
        &mut self,
        key: ChunkKey,
        resolved: Option<ResolvedChunkPayload>,
        scale_exp: i8,
    ) -> Option<Aabb4i> {
        let payload = resolved.map(canonicalize_resolved_payload);
        let chunk_bounds = Aabb4i::chunk_world_bounds(key, scale_exp);
        if self.root.is_none() {
            let payload = payload?;
            self.root = Some(Box::new(RegionTreeCore {
                bounds: chunk_bounds,
                kind: kind_from_resolved_value_at_scale(chunk_bounds, Some(payload), scale_exp),
                generator_version_hash: 0,
            }));
            return Some(chunk_bounds);
        }

        if payload.is_none()
            && self
                .root
                .as_ref()
                .map(|root| !aabb_contains_aabb(root.bounds, chunk_bounds))
                .unwrap_or(false)
        {
            return None;
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

        let affected = if let Some(root) = self.root.as_mut() {
            set_chunk_recursive(root, key, chunk_bounds, payload, scale_exp)
        } else {
            None
        };

        if affected.is_some()
            && self
                .root
                .as_ref()
                .map(|root| matches!(root.kind, RegionNodeKind::Empty))
                .unwrap_or(false)
        {
            self.root = None;
        }

        affected
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

    pub fn collect_chunks_in_bounds(
        &self,
        bounds: Aabb4i,
    ) -> Vec<(ChunkKey, ResolvedChunkPayload)> {
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

    /// Enumerate all non-empty leaf chunks in `bounds`, returning each chunk's
    /// key and the scale it is stored at.
    pub fn collect_chunk_entries_in_bounds(&self, bounds: Aabb4i) -> Vec<(ChunkKey, i8)> {
        if !bounds.is_valid() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            collect_chunk_entries_from_kind_in_bounds(&root.kind, root.bounds, bounds, &mut out);
        }
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
        let root = self.root.as_mut()?;

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
    pub fn find_leaf_chunks_in_spatial_range(&self, query: &Aabb4i) -> Vec<(ChunkKey, i8)> {
        let mut out = Vec::new();
        if let Some(root) = self.root.as_ref() {
            find_leaf_chunks_in_spatial_range_recursive(&root.kind, root.bounds, query, &mut out);
        }
        out
    }

    /// Optimize the subtree rooted at this tree by coarsening ChunkArrays
    /// where all contained blocks have a `scale_exp` above the chunk's own
    /// scale, allowing representation with fewer, coarser chunks.
    pub fn optimize_subtree_in_bounds(&mut self, bounds: Aabb4i) {
        if let Some(root) = self.root.as_mut() {
            optimize_subtree_in_bounds_impl(root, bounds);
        }
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

// ===========================================================================
// Shared helper functions (pub(super) for use by submodules)
// ===========================================================================

pub(super) fn empty_core_for_bounds(bounds: Aabb4i) -> RegionTreeCore {
    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    }
}

pub(super) fn aabb_contains_aabb(outer: Aabb4i, inner: Aabb4i) -> bool {
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

pub(super) fn merge_aabb(a: Aabb4i, b: Aabb4i) -> Aabb4i {
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

/// Subtract `inner` from `outer`, producing up to 8 half-open pieces.
pub(super) fn subtract_aabb(outer: Aabb4i, inner: Aabb4i) -> Vec<Aabb4i> {
    let Some(inner) = outer.intersection(&inner) else {
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

/// World-space chunk size for a given scale: CS * step_for_scale(s).
pub(super) fn chunk_world_size_for_scale(scale_exp: i8) -> ChunkCoord {
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    cs.saturating_mul(step_for_scale(scale_exp))
}

/// World-space alignment for split/carve boundaries, derived from node content.
pub(super) fn chunk_world_size_for_kind(kind: Option<&RegionNodeKind>) -> ChunkCoord {
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

/// Check if bounds represent exactly one chunk at the given scale.
pub(super) fn is_single_chunk_bounds(bounds: Aabb4i, scale_exp: i8) -> bool {
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    let step = step_for_scale(scale_exp);
    let chunk_world_size = cs.saturating_mul(step);
    (0..4).all(|i| bounds.max[i] - bounds.min[i] == chunk_world_size)
}

/// Check if bounds represent a single chunk at any possible scale.
pub(super) fn is_single_chunk_bounds_any_scale(bounds: Aabb4i) -> bool {
    if !bounds.is_valid() {
        return false;
    }
    let span = bounds.max[0] - bounds.min[0];
    if !(1..4).all(|i| bounds.max[i] - bounds.min[i] == span) {
        return false;
    }
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    if span < cs {
        return false;
    }
    let bits = span.to_bits();
    bits > 0 && (bits & (bits - 1)) == 0
}

/// Compute the world-space extent of a chunk given its key and scale.
pub fn chunk_spatial_extent(key: ChunkKey, scale_exp: i8) -> Aabb4i {
    Aabb4i::chunk_world_bounds(key, scale_exp)
}

/// Find the coarsest scale at which `bounds` aligns to the chunk grid.
pub fn coarsest_aligned_scale(bounds: Aabb4i, ceiling: i8) -> i8 {
    for s in (ceiling.saturating_sub(10)..=ceiling).rev() {
        if bounds.chunk_extents_at_scale(s).is_some() {
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

pub(super) fn sort_children_canonical(children: &mut [RegionTreeCore]) {
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

pub(super) fn branch_matches_split(
    bounds: Aabb4i,
    children: &[RegionTreeCore],
    alignment: ChunkCoord,
) -> bool {
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
pub(super) fn split_bounds_longest_axis(
    bounds: Aabb4i,
    alignment: ChunkCoord,
) -> Option<(Aabb4i, Aabb4i)> {
    if !bounds.is_valid() {
        return None;
    }

    let align_bits = alignment.to_bits();
    if align_bits == 0 {
        return None;
    }
    let to_lattice = |coord: ChunkCoord| -> i64 { coord.to_bits() / align_bits };
    let from_lattice = |lattice: i64| -> ChunkCoord { ChunkCoord::from_bits(lattice * align_bits) };

    let mut spans = [0i64; 4];
    for (i, span) in spans.iter_mut().enumerate() {
        *span = to_lattice(bounds.max[i]) - to_lattice(bounds.min[i]);
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

fn expand_root_once(
    root: Box<RegionTreeCore>,
    target: Aabb4i,
    _alignment: ChunkCoord,
) -> Box<RegionTreeCore> {
    if aabb_contains_aabb(root.bounds, target) {
        return root;
    }

    let mut old_root = *root;
    let old_bounds = old_root.bounds;
    normalize_chunk_node(&mut old_root);

    let axis = (0..4)
        .find(|&axis| {
            target.min[axis] < old_bounds.min[axis] || target.max[axis] > old_bounds.max[axis]
        })
        .unwrap_or(0);

    let span = old_bounds.max[axis] - old_bounds.min[axis];

    let mut new_bounds = old_bounds;
    let mut sibling_bounds = old_bounds;
    if target.min[axis] < old_bounds.min[axis] {
        let expanded = old_bounds.min[axis].saturating_sub(span);
        let expanded = if expanded >= old_bounds.min[axis] {
            target.min[axis]
        } else {
            expanded
        };
        new_bounds.min[axis] = expanded;
        sibling_bounds.min[axis] = expanded;
        sibling_bounds.max[axis] = old_bounds.min[axis];
    } else {
        let expanded = old_bounds.max[axis].saturating_add(span);
        let expanded = if expanded <= old_bounds.max[axis] {
            target.max[axis]
        } else {
            expanded
        };
        new_bounds.max[axis] = expanded;
        sibling_bounds.min[axis] = old_bounds.max[axis];
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

// ---------------------------------------------------------------------------
// Payload canonicalization and helpers
// ---------------------------------------------------------------------------

pub(super) fn canonicalize_resolved_payload(
    resolved: ResolvedChunkPayload,
) -> ResolvedChunkPayload {
    ResolvedChunkPayload {
        payload: canonicalize_payload_format(resolved.payload),
        block_palette: resolved.block_palette,
    }
}

fn canonicalize_payload_format(payload: ChunkPayload) -> ChunkPayload {
    let payload = match payload {
        ChunkPayload::Empty => ChunkPayload::Uniform(0),
        ChunkPayload::Virgin => return ChunkPayload::Virgin,
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

pub(super) fn kind_from_resolved_value_at_scale(
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
            RegionNodeKind::Uniform(block)
        }
        _ => repeated_payload_kind_resolved_at_scale(bounds, resolved, scale_exp),
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

pub(super) fn resolved_option_is_semantically_empty(
    resolved: Option<&ResolvedChunkPayload>,
) -> bool {
    resolved.map(|r| !r.has_solid_block()).unwrap_or(true)
}

pub(super) fn resolved_option_matches_block(
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

pub(super) fn resolved_option_matches_existing(
    existing: Option<ResolvedChunkPayload>,
    incoming: Option<&ResolvedChunkPayload>,
) -> bool {
    match incoming {
        Some(incoming) => {
            let incoming_c = canonicalize_resolved_payload(incoming.clone());
            match existing {
                Some(existing) => {
                    let existing_c = canonicalize_resolved_payload(existing);
                    super::tree_query::resolved_payloads_semantically_equal(
                        &existing_c,
                        &incoming_c,
                    )
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
