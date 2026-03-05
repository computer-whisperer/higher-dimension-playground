use super::chunk_array_ops::{
    chunk_array_min_block_scale, chunk_array_payload_at, chunk_array_payload_at_with_dense_indices,
    resample_chunk_array_hierarchical, slice_chunk_array_to_bounds,
    slice_chunk_array_to_bounds_with_dense_indices, try_coarsen_chunk_array,
};
use super::tree::*;
use super::tree_normalize::{normalize_chunk_node, prune_empty_subtrees};
use super::tree_query::{
    kind_has_non_empty_chunk_intersection, non_empty_kinds_semantically_equal_in_bounds,
};
use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::Aabb4i;

// ---------------------------------------------------------------------------
// Slicing (core/node level)
// ---------------------------------------------------------------------------

pub(super) fn slice_node_to_bounds(
    node: &RegionTreeCore,
    bounds: Aabb4i,
) -> Option<RegionTreeCore> {
    let intersection = node.bounds.intersection(&bounds)?;
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

pub fn slice_region_core_in_bounds(core: &RegionTreeCore, bounds: Aabb4i) -> RegionTreeCore {
    if !bounds.is_valid() {
        return empty_core_for_bounds(bounds);
    }

    let Some(intersection) = core.bounds.intersection(&bounds) else {
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

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

pub(super) fn project_node_to_bounds(
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
                let aligned_scale = coarsest_aligned_scale(target_bounds, chunk_array.scale_exp);
                return super::chunk_array_ops::resample_chunk_array_to_bounds(
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

// ---------------------------------------------------------------------------
// set_chunk_recursive
// ---------------------------------------------------------------------------

pub(super) fn set_chunk_recursive(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    chunk_bounds: Aabb4i,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> Option<Aabb4i> {
    if !aabb_contains_aabb(node.bounds, chunk_bounds) {
        return None;
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
            return None;
        }
        node.kind = new_kind;
        return Some(node.bounds);
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
) -> Option<Aabb4i> {
    let RegionNodeKind::Branch(children) = &mut node.kind else {
        return None;
    };
    let target_idx = children
        .iter()
        .position(|child| aabb_contains_aabb(child.bounds, chunk_bounds));
    let affected = if let Some(target_idx) = target_idx {
        set_chunk_recursive(
            &mut children[target_idx],
            key_pos,
            chunk_bounds,
            payload,
            scale_exp,
        )
    } else {
        let gen_hash = node.generator_version_hash;
        let old_children = std::mem::take(children);
        let mut new_children = Vec::with_capacity(old_children.len() + 1);
        let mut changed = false;
        for child in old_children {
            if !child.bounds.intersects(&chunk_bounds) {
                new_children.push(child);
            } else if aabb_contains_aabb(chunk_bounds, child.bounds) {
                changed = true;
            } else {
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
        if changed {
            Some(node.bounds)
        } else {
            None
        }
    };

    if affected.is_some() {
        normalize_chunk_node(node);
    }
    affected
}

fn carve_leaf_for_chunk_edit(
    node: &mut RegionTreeCore,
    key_pos: ChunkKey,
    chunk_bounds: Aabb4i,
    payload: Option<ResolvedChunkPayload>,
    scale_exp: i8,
) -> Option<Aabb4i> {
    let source_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);

    let no_change = match &source_kind {
        RegionNodeKind::Empty => resolved_option_is_semantically_empty(payload.as_ref()),
        RegionNodeKind::Uniform(block) => resolved_option_matches_block(block, payload.as_ref()),
        RegionNodeKind::ChunkArray(_) => false,
        RegionNodeKind::ProceduralRef(_) => false,
        RegionNodeKind::Branch(_) => false,
    };
    if no_change {
        node.kind = source_kind;
        return None;
    }

    let mut children = Vec::with_capacity(9);
    if let RegionNodeKind::ChunkArray(chunk_array) = &source_kind {
        if let Ok(source_indices) = chunk_array.decode_dense_indices() {
            let existing_resolved =
                chunk_array_payload_at_with_dense_indices(chunk_array, &source_indices, key_pos)
                    .map(|p| ResolvedChunkPayload {
                        payload: p,
                        block_palette: chunk_array.block_palette.clone(),
                    });
            if resolved_option_matches_existing(existing_resolved, payload.as_ref()) {
                node.kind = source_kind;
                return None;
            }
            let min_block_scale = chunk_array_min_block_scale(chunk_array, chunk_array.scale_exp)
                .unwrap_or(chunk_array.scale_exp);
            for piece_bounds in subtract_aabb(node.bounds, chunk_bounds) {
                let se = chunk_array.scale_exp;
                let piece_aligned = piece_bounds
                    .chunk_extents_at_scale(se)
                    .map(|_| {
                        let (lmin, lmax) = piece_bounds.to_chunk_lattice_bounds(se);
                        Aabb4i::from_lattice_bounds(lmin, lmax, se) == piece_bounds
                    })
                    .unwrap_or(false);

                if piece_aligned {
                    if let Some(chunk_array_piece) = slice_chunk_array_to_bounds_with_dense_indices(
                        chunk_array,
                        &source_indices,
                        piece_bounds,
                    ) {
                        children.push(RegionTreeCore {
                            bounds: piece_bounds,
                            kind: RegionNodeKind::ChunkArray(chunk_array_piece),
                            generator_version_hash: node.generator_version_hash,
                        });
                        continue;
                    }
                }
                resample_chunk_array_hierarchical(
                    chunk_array,
                    piece_bounds,
                    min_block_scale,
                    node.generator_version_hash,
                    &mut children,
                );
            }
        } else {
            let existing_resolved =
                chunk_array_payload_at(chunk_array, key_pos).map(|p| ResolvedChunkPayload {
                    payload: p,
                    block_palette: chunk_array.block_palette.clone(),
                });
            if resolved_option_matches_existing(existing_resolved, payload.as_ref()) {
                node.kind = source_kind;
                return None;
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

    // Optimize aligned-sliced children that may be at a finer scale than
    // necessary.
    for child in &mut children {
        if matches!(child.kind, RegionNodeKind::ChunkArray(_)) {
            try_coarsen_chunk_array(child);
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
    Some(node.bounds)
}

// ---------------------------------------------------------------------------
// ensure_binary_children
// ---------------------------------------------------------------------------

pub(super) fn ensure_binary_children(node: &mut RegionTreeCore) {
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

// ---------------------------------------------------------------------------
// Splice operations
// ---------------------------------------------------------------------------

pub(super) fn splice_node_with_non_empty_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = node.bounds.intersection(&bounds)?;
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

    if let RegionNodeKind::Branch(children) = &mut node.kind {
        let containing_idx = children
            .iter()
            .position(|child| aabb_contains_aabb(child.bounds, intersection));
        if let Some(idx) = containing_idx {
            let result = splice_node_with_non_empty_core(&mut children[idx], bounds, replacement);
            if result.is_some() {
                normalize_chunk_node(node);
            }
            return result;
        }
    }

    splice_carve_and_replace(node, intersection, replacement_slice)
}

pub(super) fn splice_node_with_core(
    node: &mut RegionTreeCore,
    bounds: Aabb4i,
    replacement: &RegionTreeCore,
) -> Option<Aabb4i> {
    let intersection = node.bounds.intersection(&bounds)?;
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

    splice_carve_and_replace(node, intersection, replacement_slice)
}

/// Recursively collect tree nodes whose bounds don't overlap `patch_bounds`.
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
        return;
    }
    match &node.kind {
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_non_overlapping_remnants(child, patch_bounds, out);
            }
        }
        _ => {
            let intersection = match node.bounds.intersection(&patch_bounds) {
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
/// insert `replacement` for the patched region.
fn splice_carve_and_replace(
    node: &mut RegionTreeCore,
    patch_bounds: Aabb4i,
    replacement: RegionTreeCore,
) -> Option<Aabb4i> {
    let source_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let gen_hash = node.generator_version_hash;
    let mut carved_chunk_array = false;

    let mut children = Vec::with_capacity(9);

    match &source_kind {
        RegionNodeKind::ChunkArray(chunk_array) => {
            if let Ok(source_indices) = chunk_array.decode_dense_indices() {
                carved_chunk_array = true;
                let se = chunk_array.scale_exp;
                let min_block_scale = chunk_array_min_block_scale(chunk_array, se).unwrap_or(se);
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
                    resample_chunk_array_hierarchical(
                        chunk_array,
                        piece_bounds,
                        min_block_scale,
                        gen_hash,
                        &mut children,
                    );
                }
            } else {
                for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                    let projected =
                        project_node_to_bounds(&source_kind, node.bounds, piece_bounds, gen_hash);
                    if !matches!(projected.kind, RegionNodeKind::Empty) {
                        children.push(projected);
                    }
                }
            }
        }
        RegionNodeKind::Branch(branch_children) => {
            for child in branch_children {
                collect_non_overlapping_remnants(child, patch_bounds, &mut children);
            }
        }
        RegionNodeKind::Uniform(block) if !block.is_air() => {
            for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                let projected =
                    project_node_to_bounds(&source_kind, node.bounds, piece_bounds, gen_hash);
                if !matches!(projected.kind, RegionNodeKind::Empty) {
                    children.push(projected);
                }
            }
        }
        _ => {
            for piece_bounds in subtract_aabb(node.bounds, patch_bounds) {
                let projected =
                    project_node_to_bounds(&source_kind, node.bounds, piece_bounds, gen_hash);
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
                    i,
                    ch[i].bounds,
                    j,
                    ch[j].bounds
                );
            }
        }
    }

    if carved_chunk_array {
        Some(node.bounds)
    } else {
        Some(patch_bounds)
    }
}

// ---------------------------------------------------------------------------
// Lazy drop
// ---------------------------------------------------------------------------

pub(super) fn lazy_drop_outside_node(
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
    if aabb_contains_aabb(keep_bounds, node.bounds) || is_single_chunk_bounds_any_scale(node.bounds)
    {
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
