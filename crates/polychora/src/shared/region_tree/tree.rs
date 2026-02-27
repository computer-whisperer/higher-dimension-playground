use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::{
    chunk_key_from_lattice, lattice_from_fixed, step_for_scale, ChunkCoord,
};
use crate::shared::voxel::BlockData;
use std::collections::HashMap;

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

    pub fn chunk_payload(&self, key: ChunkKey) -> Option<ResolvedChunkPayload> {
        self.root
            .as_ref()
            .and_then(|node| query_chunk_payload_in_node(node, key))
    }

    /// Set a chunk at scale 0 (the standard unit-scale lattice).
    pub fn set_chunk(&mut self, key: ChunkKey, resolved: Option<ResolvedChunkPayload>) -> bool {
        self.set_chunk_at_scale(key, resolved, 0)
    }

    /// Set a chunk at a specific scale. The key is already in fixed-point coordinates.
    /// `scale_exp` is needed to create ChunkArrayData with correct cell resolution.
    pub fn set_chunk_at_scale(
        &mut self,
        key: ChunkKey,
        resolved: Option<ResolvedChunkPayload>,
        scale_exp: i8,
    ) -> bool {
        let payload = resolved.map(|r| canonicalize_resolved_payload(r));
        let chunk_step = step_for_scale(scale_exp);
        if self.root.is_none() {
            let Some(payload) = payload else {
                return false;
            };
            let bounds = Aabb4i::new(key, key);
            self.root = Some(Box::new(RegionTreeCore {
                bounds,
                kind: kind_from_resolved_value_at_scale(bounds, Some(payload), scale_exp),
                generator_version_hash: 0,
            }));
            return true;
        }

        if payload.is_none()
            && self
                .root
                .as_ref()
                .map(|root| !root.bounds.contains_chunk(key))
                .unwrap_or(false)
        {
            return false;
        }

        while self
            .root
            .as_ref()
            .map(|root| !root.bounds.contains_chunk(key))
            .unwrap_or(false)
        {
            let Some(root) = self.root.take() else {
                break;
            };
            // Use the coarsest step between the new chunk and the existing tree
            // so sibling boundaries stay on a grid compatible with all content.
            let expand_step = chunk_step.max(step_for_kind(Some(&root.kind)));
            self.root = Some(expand_root_once(root, key, expand_step));
        }

        let changed = if let Some(root) = self.root.as_mut() {
            set_chunk_recursive(root, key, payload, scale_exp)
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

        let step = step_for_kind(self.root.as_ref().map(|r| &r.kind));
        while self
            .root
            .as_ref()
            .map(|root| {
                !root.bounds.contains_chunk(bounds.min) || !root.bounds.contains_chunk(bounds.max)
            })
            .unwrap_or(false)
        {
            let grow_key = self
                .root
                .as_ref()
                .map(|root| {
                    if !root.bounds.contains_chunk(bounds.min) {
                        bounds.min
                    } else {
                        bounds.max
                    }
                })
                .unwrap_or(bounds.min);
            let Some(root) = self.root.take() else {
                break;
            };
            self.root = Some(expand_root_once(root, grow_key, step));
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
            let step = step_for_scale(chunk_array.scale_exp);
            let epsilon = ChunkCoord::from_bits(1);
            let (lmin, lmax) = chunk_array.bounds.to_lattice_bounds(chunk_array.scale_exp);
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
                                chunk_array.scale_exp,
                            );
                            // Push the world-space extent of this cell, not just a point.
                            // A cell at `pos` with step `s` covers [pos, pos + s - epsilon].
                            let cell_max = pos.map(|c| c + step - epsilon);
                            out.push(Aabb4i::new(pos, cell_max));
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

    let (src_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
    let (int_lmin, int_lmax) = intersection.to_lattice_bounds(se);
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

    // Save original bounds: clear_node_region may shrink them via normalize,
    // but insert needs the full extent to place replacement data correctly.
    let original_bounds = node.bounds;
    let mut changed = clear_node_region(node, intersection);
    if node.bounds != original_bounds {
        // Normalize inside clear shrunk the node (single-child collapse).
        // Re-wrap so the node keeps its original spatial extent.
        let gen_hash = node.generator_version_hash;
        if matches!(node.kind, RegionNodeKind::Empty) {
            node.bounds = original_bounds;
        } else {
            let inner = std::mem::replace(
                node,
                RegionTreeCore {
                    bounds: original_bounds,
                    kind: RegionNodeKind::Empty,
                    generator_version_hash: gen_hash,
                },
            );
            node.kind = RegionNodeKind::Branch(vec![inner]);
        }
    }
    if !matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        changed |= insert_replacement_slice_into_node(node, replacement_slice);
    }

    if !changed {
        return None;
    }
    normalize_chunk_node(node);
    Some(intersection)
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

    let original_bounds = node.bounds;
    let mut changed = clear_node_region(node, intersection);
    if node.bounds != original_bounds {
        let gen_hash = node.generator_version_hash;
        if matches!(node.kind, RegionNodeKind::Empty) {
            node.bounds = original_bounds;
        } else {
            let inner = std::mem::replace(
                node,
                RegionTreeCore {
                    bounds: original_bounds,
                    kind: RegionNodeKind::Empty,
                    generator_version_hash: gen_hash,
                },
            );
            node.kind = RegionNodeKind::Branch(vec![inner]);
        }
    }
    if !matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        changed |= insert_replacement_slice_into_node(node, replacement_slice);
    }

    if !changed {
        return None;
    }
    normalize_chunk_node(node);
    Some(intersection)
}

fn clear_node_region(node: &mut RegionTreeCore, clear_bounds: Aabb4i) -> bool {
    let Some(intersection) = intersect_aabb(node.bounds, clear_bounds) else {
        return false;
    };

    if matches!(node.kind, RegionNodeKind::Empty) {
        return false;
    }

    if intersection == node.bounds {
        node.kind = RegionNodeKind::Empty;
        return true;
    }

    match &mut node.kind {
        RegionNodeKind::Branch(children) => {
            let mut changed = false;
            let mut retained = Vec::with_capacity(children.len());
            for mut child in std::mem::take(children) {
                if !child.bounds.intersects(&intersection) {
                    retained.push(child);
                    continue;
                }
                if aabb_contains_aabb(intersection, child.bounds) {
                    changed = true;
                    continue;
                }
                if clear_node_region(&mut child, intersection) {
                    changed = true;
                }
                if !matches!(child.kind, RegionNodeKind::Empty) {
                    retained.push(child);
                }
            }
            *children = retained;
            if changed {
                normalize_chunk_node(node);
            }
            changed
        }
        _ => {
            let old_kind = node.kind.clone();
            let step = step_for_kind(Some(&old_kind));
            let mut pieces = Vec::new();
            for piece_bounds in subtract_aabb(node.bounds, intersection, step) {
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
            true
        }
    }
}

fn insert_replacement_slice_into_node(
    node: &mut RegionTreeCore,
    replacement_slice: RegionTreeCore,
) -> bool {
    if matches!(replacement_slice.kind, RegionNodeKind::Empty) {
        return false;
    }

    if replacement_slice.bounds == node.bounds {
        if node.kind == replacement_slice.kind {
            return false;
        }
        *node = replacement_slice;
        return true;
    }

    let replacement_bounds = replacement_slice.bounds;
    match &mut node.kind {
        RegionNodeKind::Empty => {
            node.kind = RegionNodeKind::Branch(vec![replacement_slice]);
            normalize_chunk_node(node);
            return !matches!(node.kind, RegionNodeKind::Empty);
        }
        RegionNodeKind::Branch(children)
            if !children
                .iter()
                .any(|child| child.bounds.intersects(&replacement_bounds)) =>
        {
            children.push(replacement_slice);
            normalize_chunk_node(node);
            return true;
        }
        _ => {}
    }

    insert_replacement_slice_into_node_with_rebuild(node, replacement_slice)
}

fn insert_replacement_slice_into_node_with_rebuild(
    node: &mut RegionTreeCore,
    replacement_slice: RegionTreeCore,
) -> bool {
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let old_kind_snapshot = old_kind.clone();
    let mut children = Vec::new();
    match old_kind {
        RegionNodeKind::Empty => {}
        RegionNodeKind::Branch(existing_children) => {
            for child in existing_children {
                let child_step = step_for_kind(Some(&child.kind));
                for piece_bounds in subtract_aabb(child.bounds, replacement_slice.bounds, child_step) {
                    let projected = project_node_to_bounds(
                        &child.kind,
                        child.bounds,
                        piece_bounds,
                        node.generator_version_hash,
                    );
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        children.push(projected);
                    }
                }
            }
        }
        other_kind => {
            let other_step = step_for_kind(Some(&other_kind));
            for piece_bounds in subtract_aabb(node.bounds, replacement_slice.bounds, other_step) {
                let projected = project_node_to_bounds(
                    &other_kind,
                    node.bounds,
                    piece_bounds,
                    node.generator_version_hash,
                );
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
    }
    children.push(replacement_slice);
    node.kind = RegionNodeKind::Branch(children);
    normalize_chunk_node(node);
    node.kind != old_kind_snapshot
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
    if aabb_contains_aabb(keep_bounds, node.bounds) || is_single_chunk_bounds(node.bounds) {
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
    let (lmin, lmax) = chunk_array.bounds.to_lattice_bounds(se);

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
                    let cell_bounds = Aabb4i::new(pos, pos);
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

fn step_for_kind(kind: Option<&RegionNodeKind>) -> ChunkCoord {
    match kind {
        Some(RegionNodeKind::ChunkArray(ca)) => step_for_scale(ca.scale_exp),
        // Use MAX of children's steps so split/carve boundaries land on the
        // coarsest grid. Every fine grid is a superset of the coarse grid, so
        // coarse boundaries are always valid for fine-grid children. Using MIN
        // would create fractional boundaries that corrupt coarse-grid siblings
        // (e.g. Uniform nodes gaining half-integer bounds).
        Some(RegionNodeKind::Branch(children)) => children
            .iter()
            .map(|c| step_for_kind(Some(&c.kind)))
            .max()
            .unwrap_or(step_for_scale(0)),
        _ => step_for_scale(0),
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

fn subtract_aabb(outer: Aabb4i, inner: Aabb4i, step: ChunkCoord) -> Vec<Aabb4i> {
    let Some(inner) = intersect_aabb(outer, inner) else {
        return vec![outer];
    };
    if inner == outer {
        return Vec::new();
    }

    let mut pieces = Vec::with_capacity(8);
    let mut core = outer;

    let step_bits = step.to_bits();
    for axis in 0..4 {
        if core.min[axis] < inner.min[axis] {
            let mut piece = core;
            piece.max[axis] = inner.min[axis] - step;
            if piece.is_valid() {
                debug_assert!(
                    step_bits != 0 && piece.max[axis].to_bits() % step_bits == 0,
                    "subtract_aabb: piece max[{axis}]={} not on step={step} lattice \
                     (outer={outer:?} inner={inner:?})",
                    piece.max[axis],
                );
                pieces.push(piece);
            }
            core.min[axis] = inner.min[axis];
        }
        if core.max[axis] > inner.max[axis] {
            let mut piece = core;
            piece.min[axis] = inner.max[axis] + step;
            if piece.is_valid() {
                debug_assert!(
                    step_bits != 0 && piece.min[axis].to_bits() % step_bits == 0,
                    "subtract_aabb: piece min[{axis}]={} not on step={step} lattice \
                     (outer={outer:?} inner={inner:?})",
                    piece.min[axis],
                );
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
            let (lmin, lmax) = intersection.to_lattice_bounds(0);
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
            let (ca_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_lattice_bounds(se);
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
            let (lmin, lmax) = intersection.to_lattice_bounds(0);
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
            let (ca_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_lattice_bounds(se);
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
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    if !node.bounds.contains_chunk(key_pos) {
        return false;
    }

    if is_single_chunk_bounds(node.bounds) {
        let new_kind = kind_from_resolved_value_at_scale(node.bounds, payload, scale_exp);
        if node.kind == new_kind {
            return false;
        }
        node.kind = new_kind;
        return true;
    }

    if matches!(node.kind, RegionNodeKind::Branch(_)) {
        return set_chunk_recursive_in_branch(node, key_pos, payload, scale_exp);
    }

    carve_leaf_for_chunk_edit(node, key_pos, payload, scale_exp)
}

fn set_chunk_recursive_in_branch(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return false;
    };
    let target_idx = children
        .iter()
        .position(|child| child.bounds.contains_chunk(key_pos));
    let changed = if let Some(target_idx) = target_idx {
        set_chunk_recursive(&mut children[target_idx], key_pos, payload, scale_exp)
    } else if let Some(payload) = payload {
        let chunk_bounds = Aabb4i::new(key_pos, key_pos);
        children.push(RegionTreeCore {
            bounds: chunk_bounds,
            kind: kind_from_resolved_value_at_scale(chunk_bounds, Some(payload), scale_exp),
            generator_version_hash: node.generator_version_hash,
        });
        true
    } else {
        false
    };

    if changed {
        normalize_chunk_node(node);
    }
    changed
}

fn carve_leaf_for_chunk_edit(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> bool {
    let chunk_bounds = Aabb4i::new(key_pos, key_pos);
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
            let ca_step = step_for_scale(chunk_array.scale_exp);
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds, ca_step) {
                let Some(chunk_array_piece) = slice_chunk_array_to_bounds_with_dense_indices(
                    chunk_array,
                    &source_indices,
                    piece_bounds,
                ) else {
                    continue;
                };
                children.push(RegionTreeCore {
                    bounds: piece_bounds,
                    kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                    generator_version_hash: node.generator_version_hash,
                });
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
            let ca_step = step_for_scale(chunk_array.scale_exp);
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds, ca_step) {
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
        let step = step_for_kind(Some(&source_kind));
        for piece_bounds in subtract_aabb(node.bounds, chunk_bounds, step) {
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
    if is_single_chunk_bounds(node.bounds) {
        return;
    }

    let step = step_for_kind(Some(&node.kind));
    if let RegionNodeKind::Branch(children) = &mut node.kind {
        if branch_matches_split(node.bounds, children, step) {
            sort_children_canonical(children);
            return;
        }
    }

    let Some((left_bounds, right_bounds)) = split_bounds_longest_axis(node.bounds, step) else {
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
            // Uniform/ProceduralRef are always scale 0
            let step = step_for_scale(0);
            for node in iter {
                if node.bounds.min[axis] == current.bounds.max[axis] + step {
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
        let (child_lmin, child_lmax) = ca.bounds.to_lattice_bounds(common_scale_exp);
        let (comb_lmin, _) = combined_bounds.to_lattice_bounds(common_scale_exp);

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
    let merge_step = children
        .iter()
        .map(|c| step_for_kind(Some(&c.kind)))
        .max()
        .unwrap_or(step_for_scale(0));
    if !branch_matches_split(node.bounds, children, merge_step) {
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
            match slice_chunk_array_to_bounds(chunk_array, target_bounds) {
                Some(sliced) => RegionNodeKind::ChunkArray(sliced),
                None => RegionNodeKind::Empty,
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
) -> Option<ResolvedChunkPayload> {
    query_chunk_payload_in_kind(&node.kind, node.bounds, key_pos)
}

fn query_chunk_payload_in_kind(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    key_pos: ChunkKey,
) -> Option<ResolvedChunkPayload> {
    if !bounds.contains_chunk(key_pos) {
        return None;
    }
    match kind {
        RegionNodeKind::Empty => None,
        RegionNodeKind::Uniform(block) => Some(ResolvedChunkPayload::uniform(block.clone())),
        RegionNodeKind::ProceduralRef(_) => None,
        RegionNodeKind::ChunkArray(chunk_array) => {
            chunk_array_resolved_payload_at(chunk_array, key_pos)
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                if child.bounds.contains_chunk(key_pos) {
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
) -> Option<ResolvedChunkPayload> {
    let payload = chunk_array_payload_at(chunk_array, key_pos)?;
    Some(ResolvedChunkPayload {
        payload,
        block_palette: chunk_array.block_palette.clone(),
    })
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
            let (lmin, lmax) = bounds.to_lattice_bounds(0);
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
            let (lmin, lmax) = chunk_array.bounds.to_lattice_bounds(se);
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
            let (lmin, lmax) = intersection.to_lattice_bounds(0);
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
            let (ca_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
            let (int_lmin, int_lmax) = chunk_array_intersection.to_lattice_bounds(se);
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
            let (lmin, lmax) = intersection.to_lattice_bounds(0);
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
            let (lmin, lmax) = intersection.to_lattice_bounds(se);
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
    let (ca_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
    let (int_lmin, int_lmax) = intersection.to_lattice_bounds(se);

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
    if !chunk_array.bounds.contains_chunk(key_pos) {
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
    if !chunk_array.bounds.contains_chunk(key_pos) {
        return None;
    }
    let se = chunk_array.scale_exp;
    let extents = chunk_array.bounds.chunk_extents_at_scale(se)?;
    let (ca_lmin, _) = chunk_array.bounds.to_lattice_bounds(se);
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

fn kind_from_resolved_value(
    bounds: Aabb4i,
    value: Option<ResolvedChunkPayload>,
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
        _ => repeated_payload_kind_resolved(bounds, resolved),
    }
}

fn repeated_payload_kind_resolved(
    bounds: Aabb4i,
    resolved: ResolvedChunkPayload,
) -> RegionNodeKind {
    repeated_payload_kind_resolved_at_scale(bounds, resolved, 0)
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
    if scale_exp == 0 {
        return kind_from_resolved_value(bounds, value);
    }
    let Some(resolved) = value else {
        return RegionNodeKind::Empty;
    };
    let resolved = canonicalize_resolved_payload(resolved);
    // For non-zero scales, always create a ChunkArray to carry the scale_exp.
    // We cannot use Uniform(block) because Uniform has no scale_exp field.
    repeated_payload_kind_resolved_at_scale(bounds, resolved, scale_exp)
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

fn is_single_chunk_bounds(bounds: Aabb4i) -> bool {
    bounds.min == bounds.max
}

fn linear_cell_index(coords: [usize; 4], dims: [usize; 4]) -> usize {
    coords[0] + dims[0] * (coords[1] + dims[1] * (coords[2] + dims[2] * coords[3]))
}

fn branch_matches_split(bounds: Aabb4i, children: &[RegionTreeCore], step: ChunkCoord) -> bool {
    if children.len() != 2 {
        return false;
    }
    let Some((left, right)) = split_bounds_longest_axis(bounds, step) else {
        return false;
    };
    (children[0].bounds == left && children[1].bounds == right)
        || (children[0].bounds == right && children[1].bounds == left)
}

fn split_bounds_longest_axis(bounds: Aabb4i, step: ChunkCoord) -> Option<(Aabb4i, Aabb4i)> {
    if !bounds.is_valid() {
        return None;
    }

    // Convert to step-relative lattice integers for arithmetic
    let step_bits = step.to_bits();
    let to_lattice = |coord: ChunkCoord| -> i64 { coord.to_bits() / step_bits };
    let from_lattice = |lattice: i64| -> ChunkCoord { ChunkCoord::from_bits(lattice * step_bits) };

    let mut spans = [0i64; 4];
    for i in 0..4 {
        spans[i] = to_lattice(bounds.max[i]) - to_lattice(bounds.min[i]) + 1;
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
    let left_max_axis = from_lattice(to_lattice(bounds.min[axis]) + left_len - 1);
    let right_min_axis = from_lattice(to_lattice(bounds.min[axis]) + left_len);

    let mut left_max = bounds.max;
    left_max[axis] = left_max_axis;

    let mut right_min = bounds.min;
    right_min[axis] = right_min_axis;

    let left = Aabb4i::new(bounds.min, left_max);
    let right = Aabb4i::new(right_min, bounds.max);

    // Verify split boundaries are on the step lattice.
    debug_assert!(
        left_max_axis.to_bits() % step_bits == 0,
        "split_bounds_longest_axis: left_max[{axis}]={left_max_axis} not on step={step} lattice \
         (bounds={bounds:?})",
    );
    debug_assert!(
        right_min_axis.to_bits() % step_bits == 0,
        "split_bounds_longest_axis: right_min[{axis}]={right_min_axis} not on step={step} lattice \
         (bounds={bounds:?})",
    );

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

fn expand_root_once(root: Box<RegionTreeCore>, key_pos: ChunkKey, step: ChunkCoord) -> Box<RegionTreeCore> {
    if root.bounds.contains_chunk(key_pos) {
        return root;
    }

    let mut old_root = *root;
    let old_bounds = old_root.bounds;
    normalize_chunk_node(&mut old_root);
    let axis = (0..4)
        .find(|axis| {
            key_pos[*axis] < old_bounds.min[*axis] || key_pos[*axis] > old_bounds.max[*axis]
        })
        .unwrap_or(0);
    let span = (old_bounds.max[axis] - old_bounds.min[axis] + step).max(step);

    let mut new_bounds = old_bounds;
    let mut sibling_bounds = old_bounds;
    if key_pos[axis] < old_bounds.min[axis] {
        let expanded = old_bounds.min[axis].saturating_sub(span);
        let expanded = if expanded >= old_bounds.min[axis] {
            key_pos[axis]
        } else {
            expanded
        };
        new_bounds.min[axis] = expanded;
        sibling_bounds.min[axis] = expanded;
        sibling_bounds.max[axis] = old_bounds.min[axis] - step;
    } else {
        let expanded = old_bounds.max[axis].saturating_add(span);
        let expanded = if expanded <= old_bounds.max[axis] {
            key_pos[axis]
        } else {
            expanded
        };
        new_bounds.max[axis] = expanded;
        sibling_bounds.min[axis] = old_bounds.max[axis] + step;
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
    (min[0] <= max[0] && min[1] <= max[1] && min[2] <= max[2] && min[3] <= max[3])
        .then_some(Aabb4i::new(min, max))
}
