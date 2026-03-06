use super::tree::*;
use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::{
    chunk_key_from_lattice, lattice_from_fixed, step_for_scale, Aabb4i, ChunkCoord,
};
use crate::shared::voxel::{linear_cell_index, BlockData, CHUNK_SIZE, CHUNK_VOLUME};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Slicing
// ---------------------------------------------------------------------------

pub(super) fn slice_chunk_array_to_bounds(
    chunk_array: &ChunkArrayData,
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = chunk_array.bounds.intersection(&bounds)?;
    let source_indices = chunk_array.decode_dense_indices().ok()?;
    slice_chunk_array_to_bounds_with_dense_indices(chunk_array, &source_indices, intersection)
}

pub(super) fn slice_chunk_array_to_bounds_with_dense_indices(
    chunk_array: &ChunkArrayData,
    source_indices: &[u16],
    bounds: Aabb4i,
) -> Option<ChunkArrayData> {
    let intersection = chunk_array.bounds.intersection(&bounds)?;
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
        chunk_array.default_chunk_idx,
        chunk_array.block_palette.clone(),
        chunk_array.scale_exp,
    )
    .ok()
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

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
    let Some(intersection) = target_bounds.intersection(&source.bounds) else {
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
    let target_cell_count =
        target_extents[0] * target_extents[1] * target_extents[2] * target_extents[3];

    // Source geometry.
    let source_scale = source.scale_exp;
    let source_extents = source
        .bounds
        .chunk_extents_at_scale(source_scale)
        .expect("resample_chunk_array_to_bounds: source bounds must align to source scale");

    // Decode source: indices -> chunk payloads -> dense block materials per chunk.
    let source_indices = source
        .decode_dense_indices()
        .expect("resample_chunk_array_to_bounds: failed to decode source indices");

    // Pre-expand each unique chunk payload to dense block palette indices.
    let chunk_materials: Vec<Vec<u16>> = source
        .chunk_palette
        .iter()
        .map(|cp| {
            cp.dense_materials()
                .unwrap_or_else(|_| vec![0u16; CHUNK_VOLUME])
        })
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
                    // Map target chunk to source: each target chunk position maps
                    // to a region inside one source chunk.
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
                        // Same scale -- the source chunk maps 1:1 to the target chunk.
                        let mut chunk_blocks = Vec::with_capacity(CHUNK_VOLUME);
                        for &mat_idx in src_materials.iter().take(CHUNK_VOLUME) {
                            let block_palette_idx = mat_idx as usize;
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
                        // Target is finer -- each target chunk maps to a sub-region
                        // of one source chunk.
                        let mut chunk_blocks = Vec::with_capacity(CHUNK_VOLUME);

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

// ---------------------------------------------------------------------------
// Hierarchical resampling
// ---------------------------------------------------------------------------

/// Hierarchically resample a ChunkArray into sub-pieces at the coarsest possible
/// scales, avoiding a monolithic fine-scale intermediate.
pub(super) fn resample_chunk_array_hierarchical(
    source: &ChunkArrayData,
    piece_bounds: Aabb4i,
    min_block_scale: i8,
    gen_hash: u64,
    children: &mut Vec<RegionTreeCore>,
) {
    let source_scale = source.scale_exp;

    // Try direct coarsening: check if the whole piece aligns at a coarser scale.
    if let Some(target_scale) =
        coarsest_aligned_scale_in_range(piece_bounds, source_scale, min_block_scale)
    {
        let resampled =
            resample_chunk_array_to_bounds(source, piece_bounds, target_scale, gen_hash);
        if !matches!(resampled.kind, RegionNodeKind::Empty) {
            children.push(resampled);
        }
        return;
    }

    // Find the finest aligned scale (base case for resampling).
    let finest_aligned = coarsest_aligned_scale(piece_bounds, source_scale);

    // Try to find a split boundary at a scale coarser than finest_aligned.
    if finest_aligned < min_block_scale {
        let cs_fp = ChunkCoord::from_num(CHUNK_SIZE as i32);
        let mut best_axis = None;
        let mut best_scale = finest_aligned;
        let mut best_boundary = ChunkCoord::ZERO;

        for axis in 0..4 {
            let lo = piece_bounds.min[axis];
            let hi = piece_bounds.max[axis];
            if lo >= hi {
                continue;
            }

            for s in (finest_aligned + 1..=min_block_scale).rev() {
                let step = step_for_scale(s);
                let chunk_world_size = cs_fp.saturating_mul(step);
                let cws_bits = chunk_world_size.to_bits();
                if cws_bits <= 0 {
                    continue;
                }

                let lo_bits = lo.to_bits();
                let boundary_lattice = lo_bits.div_euclid(cws_bits) + 1;
                let Some(boundary_bits) = boundary_lattice.checked_mul(cws_bits) else {
                    continue;
                };
                let boundary = ChunkCoord::from_bits(boundary_bits);

                if boundary > lo && boundary < hi {
                    if s > best_scale {
                        best_axis = Some(axis);
                        best_scale = s;
                        best_boundary = boundary;
                    }
                    break;
                }
            }
        }

        if let Some(axis) = best_axis {
            let mut left_bounds = piece_bounds;
            left_bounds.max[axis] = best_boundary;
            let mut right_bounds = piece_bounds;
            right_bounds.min[axis] = best_boundary;

            resample_chunk_array_hierarchical(
                source,
                left_bounds,
                min_block_scale,
                gen_hash,
                children,
            );
            resample_chunk_array_hierarchical(
                source,
                right_bounds,
                min_block_scale,
                gen_hash,
                children,
            );
            return;
        }
    }

    // No coarser representation possible. Resample at the finest aligned scale.
    let resampled = resample_chunk_array_to_bounds(source, piece_bounds, finest_aligned, gen_hash);
    if !matches!(resampled.kind, RegionNodeKind::Empty) {
        children.push(resampled);
    }
}

// ---------------------------------------------------------------------------
// Scale optimization (coarsening)
// ---------------------------------------------------------------------------

/// Optimize a subtree, coarsening ChunkArrays and collapsing empty/uniform
/// regions where the data permits.  Recurses into Branch children, then
/// re-normalizes.
pub fn optimize_subtree(node: &mut RegionTreeCore) {
    match &mut node.kind {
        RegionNodeKind::Branch(children) => {
            for child in children.iter_mut() {
                optimize_subtree(child);
            }
            super::tree_normalize::normalize_chunk_node(node);
        }
        RegionNodeKind::ChunkArray(_) => {
            try_coarsen_chunk_array(node);
        }
        _ => {}
    }
}

/// Like [`optimize_subtree`] but only descends into nodes that overlap `bounds`.
pub(super) fn optimize_subtree_in_bounds(node: &mut RegionTreeCore, bounds: Aabb4i) {
    if !node.bounds.intersects(&bounds) {
        return;
    }
    match &mut node.kind {
        RegionNodeKind::Branch(children) => {
            for child in children.iter_mut() {
                optimize_subtree_in_bounds(child, bounds);
            }
            super::tree_normalize::normalize_chunk_node(node);
        }
        RegionNodeKind::ChunkArray(_) => {
            try_coarsen_chunk_array(node);
        }
        _ => {}
    }
}

/// Scan a ChunkArray's block palette references to find the minimum
/// `scale_exp` across all voxels.  Returns `None` if the array is empty
/// or if the minimum equals or is below `floor`.
pub(super) fn chunk_array_min_block_scale(ca: &ChunkArrayData, floor: i8) -> Option<i8> {
    let mut min_s = i8::MAX;
    for cp in &ca.chunk_palette {
        match cp {
            ChunkPayload::Empty | ChunkPayload::Virgin => {
                if let Some(b) = ca.block_palette.first() {
                    min_s = min_s.min(b.scale_exp);
                }
            }
            ChunkPayload::Uniform(idx) => {
                if let Some(b) = ca.block_palette.get(*idx as usize) {
                    min_s = min_s.min(b.scale_exp);
                }
            }
            ChunkPayload::PalettePacked { palette, .. } => {
                for &idx in palette {
                    if let Some(b) = ca.block_palette.get(idx as usize) {
                        min_s = min_s.min(b.scale_exp);
                    }
                }
            }
            ChunkPayload::Dense16 { materials } => {
                for &idx in materials {
                    if let Some(b) = ca.block_palette.get(idx as usize) {
                        min_s = min_s.min(b.scale_exp);
                    }
                }
            }
        }
        if min_s <= floor {
            return None;
        }
    }
    if min_s <= floor || min_s == i8::MAX {
        None
    } else {
        Some(min_s)
    }
}

/// Find the coarsest scale in `(source_scale, ceiling]` at which `bounds`
/// aligns to a chunk lattice.  Returns `None` if no coarser alignment exists.
pub(super) fn coarsest_aligned_scale_in_range(
    bounds: Aabb4i,
    source_scale: i8,
    ceiling: i8,
) -> Option<i8> {
    for s in (source_scale + 1..=ceiling).rev() {
        if bounds.chunk_extents_at_scale(s).is_some() {
            let (lmin, lmax) = bounds.to_chunk_lattice_bounds(s);
            let reconstructed = Aabb4i::from_lattice_bounds(lmin, lmax, s);
            if reconstructed == bounds {
                return Some(s);
            }
        }
    }
    None
}

/// Try to coarsen a ChunkArray node to a coarser scale.
pub(super) fn try_coarsen_chunk_array(node: &mut RegionTreeCore) -> bool {
    let RegionNodeKind::ChunkArray(ca) = &node.kind else {
        return false;
    };
    let source_scale = ca.scale_exp;

    // --- Phase 1: find min block scale_exp across all referenced voxels ---
    let Some(min_block_scale) = chunk_array_min_block_scale(ca, source_scale) else {
        return false;
    };

    // --- Phase 2: try direct coarsening (bounds align at a coarser scale) ---
    if let Some(target_scale) =
        coarsest_aligned_scale_in_range(node.bounds, source_scale, min_block_scale)
    {
        return rebuild_chunk_array_at_scale(node, target_scale);
    }

    // --- Phase 3: hierarchical split ---
    try_split_and_coarsen(node, source_scale, min_block_scale)
}

/// Split a ChunkArray node along a power-of-2-aligned boundary, creating a
/// Branch, then recursively optimize each half.
fn try_split_and_coarsen(node: &mut RegionTreeCore, source_scale: i8, min_block_scale: i8) -> bool {
    let bounds = node.bounds;

    let mut best_axis = None;
    let mut best_scale = source_scale;
    let mut best_boundary = ChunkCoord::ZERO;

    let cs_fp = ChunkCoord::from_num(CHUNK_SIZE as i32);

    for axis in 0..4 {
        let lo = bounds.min[axis];
        let hi = bounds.max[axis];
        if lo >= hi {
            continue;
        }

        for s in (source_scale + 1..=min_block_scale).rev() {
            let step = step_for_scale(s);
            let chunk_world_size = cs_fp.saturating_mul(step);
            let cws_bits = chunk_world_size.to_bits();
            if cws_bits <= 0 {
                continue;
            }

            // Smallest multiple of chunk_world_size that's strictly > lo.
            let lo_bits = lo.to_bits();
            let boundary_lattice = lo_bits.div_euclid(cws_bits) + 1;
            let boundary_bits = boundary_lattice.checked_mul(cws_bits);
            let Some(boundary_bits) = boundary_bits else {
                continue;
            };
            let boundary = ChunkCoord::from_bits(boundary_bits);

            if boundary > lo && boundary < hi {
                if s > best_scale {
                    best_axis = Some(axis);
                    best_scale = s;
                    best_boundary = boundary;
                }
                break; // Found for this axis at this scale, try next axis
            }
        }
    }

    let Some(axis) = best_axis else {
        return false;
    };

    // Split the bounds.
    let mut left_bounds = bounds;
    left_bounds.max[axis] = best_boundary;
    let mut right_bounds = bounds;
    right_bounds.min[axis] = best_boundary;

    // Take the old ChunkArray and slice it.
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let RegionNodeKind::ChunkArray(ca) = old_kind else {
        unreachable!();
    };
    let gen_hash = node.generator_version_hash;

    let left = RegionTreeCore {
        bounds: left_bounds,
        kind: slice_chunk_array_to_bounds(&ca, left_bounds)
            .map(RegionNodeKind::ChunkArray)
            .unwrap_or(RegionNodeKind::Empty),
        generator_version_hash: gen_hash,
    };
    let right = RegionTreeCore {
        bounds: right_bounds,
        kind: slice_chunk_array_to_bounds(&ca, right_bounds)
            .map(RegionNodeKind::ChunkArray)
            .unwrap_or(RegionNodeKind::Empty),
        generator_version_hash: gen_hash,
    };

    let mut children = vec![left, right];

    // Recursively optimize each half (may coarsen or split further).
    for child in &mut children {
        optimize_subtree(child);
    }

    children.retain(|c| !matches!(c.kind, RegionNodeKind::Empty));

    if children.is_empty() {
        node.kind = RegionNodeKind::Empty;
    } else if children.len() == 1 {
        let child = children.pop().unwrap();
        node.bounds = child.bounds;
        node.kind = child.kind;
    } else {
        node.kind = RegionNodeKind::Branch(children);
        super::tree_normalize::normalize_chunk_node(node);
    }

    true
}

/// Rebuild a ChunkArray node at a coarser scale by sampling representative
/// cells.
fn rebuild_chunk_array_at_scale(node: &mut RegionTreeCore, target_scale: i8) -> bool {
    let old_kind = std::mem::replace(&mut node.kind, RegionNodeKind::Empty);
    let RegionNodeKind::ChunkArray(ca) = old_kind else {
        unreachable!();
    };
    let source_scale = ca.scale_exp;

    let source_extents = match ca.bounds.chunk_extents_at_scale(source_scale) {
        Some(e) => e,
        None => {
            node.kind = RegionNodeKind::ChunkArray(ca);
            return false;
        }
    };
    let source_indices = match ca.decode_dense_indices() {
        Ok(i) => i,
        Err(_) => {
            node.kind = RegionNodeKind::ChunkArray(ca);
            return false;
        }
    };

    // Pre-expand each unique source chunk payload to dense block palette indices.
    let chunk_materials: Vec<Vec<u16>> = ca
        .chunk_palette
        .iter()
        .map(|cp| {
            cp.dense_materials()
                .unwrap_or_else(|_| vec![0u16; CHUNK_VOLUME])
        })
        .collect();

    let scale_diff = (target_scale - source_scale) as u32;
    let ratio = 1usize << scale_diff;
    let cs = CHUNK_SIZE;

    let (src_lmin, _) = ca.bounds.to_chunk_lattice_bounds(source_scale);
    let (tgt_lmin, tgt_lmax) = ca.bounds.to_chunk_lattice_bounds(target_scale);

    let mut first_block_idx: Option<u16> = None;
    let mut all_uniform = true;
    let mut target_chunk_voxels: Vec<Vec<u16>> = Vec::new();

    for tw in tgt_lmin[3]..=tgt_lmax[3] {
        for tz in tgt_lmin[2]..=tgt_lmax[2] {
            for ty in tgt_lmin[1]..=tgt_lmax[1] {
                for tx in tgt_lmin[0]..=tgt_lmax[0] {
                    let mut voxels = Vec::with_capacity(CHUNK_VOLUME);

                    for vw in 0..cs {
                        for vz in 0..cs {
                            for vy in 0..cs {
                                for vx in 0..cs {
                                    let tgt_abs = [
                                        tx as isize * cs as isize + vx as isize,
                                        ty as isize * cs as isize + vy as isize,
                                        tz as isize * cs as isize + vz as isize,
                                        tw as isize * cs as isize + vw as isize,
                                    ];
                                    let src_abs = [
                                        tgt_abs[0] * ratio as isize,
                                        tgt_abs[1] * ratio as isize,
                                        tgt_abs[2] * ratio as isize,
                                        tgt_abs[3] * ratio as isize,
                                    ];
                                    let src_chunk_rel = [
                                        (src_abs[0].div_euclid(cs as isize) as i32 - src_lmin[0])
                                            as usize,
                                        (src_abs[1].div_euclid(cs as isize) as i32 - src_lmin[1])
                                            as usize,
                                        (src_abs[2].div_euclid(cs as isize) as i32 - src_lmin[2])
                                            as usize,
                                        (src_abs[3].div_euclid(cs as isize) as i32 - src_lmin[3])
                                            as usize,
                                    ];
                                    let src_voxel = [
                                        src_abs[0].rem_euclid(cs as isize) as usize,
                                        src_abs[1].rem_euclid(cs as isize) as usize,
                                        src_abs[2].rem_euclid(cs as isize) as usize,
                                        src_abs[3].rem_euclid(cs as isize) as usize,
                                    ];

                                    let src_linear =
                                        linear_cell_index(src_chunk_rel, source_extents);
                                    let cp_idx = source_indices[src_linear] as usize;
                                    let src_mat = &chunk_materials[cp_idx];
                                    let sv_idx = src_voxel[3] * cs * cs * cs
                                        + src_voxel[2] * cs * cs
                                        + src_voxel[1] * cs
                                        + src_voxel[0];
                                    let block_idx = src_mat[sv_idx];

                                    if all_uniform {
                                        match first_block_idx {
                                            None => first_block_idx = Some(block_idx),
                                            Some(fb) if fb != block_idx => {
                                                all_uniform = false;
                                            }
                                            _ => {}
                                        }
                                    }
                                    voxels.push(block_idx);
                                }
                            }
                        }
                    }
                    target_chunk_voxels.push(voxels);
                }
            }
        }
    }

    // Collapse to Empty or Uniform if possible.
    if all_uniform {
        let block_idx = first_block_idx.unwrap_or(0);
        let block = ca
            .block_palette
            .get(block_idx as usize)
            .cloned()
            .unwrap_or(BlockData::AIR);
        node.kind = if block.is_air() {
            RegionNodeKind::Empty
        } else {
            RegionNodeKind::Uniform(block)
        };
        return true;
    }

    // Build new ChunkArrayData at the coarser scale.
    let mut new_chunk_palette = Vec::new();
    let mut chunk_to_idx: HashMap<Vec<u16>, u16> = HashMap::new();
    let mut dense_indices = Vec::with_capacity(target_chunk_voxels.len());

    for chunk_voxels in &target_chunk_voxels {
        let chunk_idx = match chunk_to_idx.get(chunk_voxels) {
            Some(&idx) => idx,
            None => {
                let idx = new_chunk_palette.len() as u16;
                let payload = ChunkPayload::from_dense_materials_compact(chunk_voxels)
                    .unwrap_or(ChunkPayload::Empty);
                new_chunk_palette.push(payload);
                chunk_to_idx.insert(chunk_voxels.clone(), idx);
                idx
            }
        };
        dense_indices.push(chunk_idx);
    }

    match ChunkArrayData::from_dense_indices_with_block_palette(
        node.bounds,
        new_chunk_palette,
        dense_indices,
        None,
        ca.block_palette,
        target_scale,
    ) {
        Ok(new_ca) => {
            node.kind = RegionNodeKind::ChunkArray(new_ca);
            true
        }
        Err(e) => {
            panic!(
                "rebuild_chunk_array_at_scale: failed to build coarsened ChunkArray: {:?}",
                e
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Overlay helpers
// ---------------------------------------------------------------------------

/// Visit each non-empty leaf in a core, decomposing ChunkArrays with gap defaults.
pub(super) fn overlay_non_empty_leaves<F>(core: &RegionTreeCore, visit: &mut F)
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
            if chunk_array_has_transparent_gap_default(chunk_array) {
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
/// Empty or Virgin palette entry -- the signature of consolidation-created gap positions.
fn chunk_array_has_transparent_gap_default(chunk_array: &ChunkArrayData) -> bool {
    if let Some(default_idx) = chunk_array.default_chunk_idx {
        chunk_array
            .chunk_palette
            .get(default_idx as usize)
            .is_some_and(|p| matches!(p, ChunkPayload::Empty | ChunkPayload::Virgin))
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
                    if matches!(payload, ChunkPayload::Empty | ChunkPayload::Virgin) {
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

// ---------------------------------------------------------------------------
// Consolidation (merging multiple ChunkArrays into one)
// ---------------------------------------------------------------------------

/// Merge multiple ChunkArray children of a Branch into a single ChunkArray.
pub(super) fn consolidate_chunk_array_children(
    children: &mut Vec<RegionTreeCore>,
    generator_version_hash: u64,
    gap_payload: ChunkPayload,
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
    let mut palette: Vec<ChunkPayload> = vec![gap_payload.clone()];
    let mut palette_map: HashMap<ChunkPayload, u16> = HashMap::new();
    palette_map.insert(gap_payload, 0u16);
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

                        if matches!(payload, ChunkPayload::Empty | ChunkPayload::Virgin) {
                            continue;
                        }

                        // Remap block palette indices within this payload.
                        let remapped_payload =
                            remap_chunk_payload_block_indices(&payload, block_remap);

                        // Map child lattice coords to combined lattice coords.
                        let combined_local = [
                            (lx - comb_lmin[0]) as usize,
                            (ly - comb_lmin[1]) as usize,
                            (lz - comb_lmin[2]) as usize,
                            (lw - comb_lmin[3]) as usize,
                        ];
                        let combined_linear = linear_cell_index(combined_local, combined_extents);

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

// ---------------------------------------------------------------------------
// Chunk payload helpers
// ---------------------------------------------------------------------------

pub(super) fn chunk_array_palette_non_empty_mask(chunk_array: &ChunkArrayData) -> Vec<bool> {
    chunk_array
        .chunk_palette
        .iter()
        .map(|p| payload_has_solid_material_in_context(p, &chunk_array.block_palette))
        .collect()
}

pub(super) fn chunk_array_has_non_empty_intersection(
    chunk_array: &ChunkArrayData,
    query_bounds: Aabb4i,
) -> bool {
    let Some(intersection) = chunk_array.bounds.intersection(&query_bounds) else {
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

pub(super) fn chunk_array_payload_at(
    chunk_array: &ChunkArrayData,
    key_pos: ChunkKey,
) -> Option<ChunkPayload> {
    if !chunk_array.bounds.contains_chunk_world_min(key_pos) {
        return None;
    }
    let dense_indices = chunk_array.decode_dense_indices().ok()?;
    chunk_array_payload_at_with_dense_indices(chunk_array, &dense_indices, key_pos)
}

pub(super) fn chunk_array_payload_at_with_dense_indices(
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

pub(super) fn chunk_array_resolved_payload_at(
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

fn payload_has_solid_material_in_context(
    payload: &ChunkPayload,
    block_palette: &[BlockData],
) -> bool {
    match payload {
        ChunkPayload::Empty | ChunkPayload::Virgin => false,
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

/// Remap block-palette indices inside a single `ChunkPayload` using the given remap table.
pub(super) fn remap_chunk_payload_block_indices(
    payload: &ChunkPayload,
    remap: &[u16],
) -> ChunkPayload {
    match payload {
        ChunkPayload::Empty => ChunkPayload::Empty,
        ChunkPayload::Virgin => ChunkPayload::Virgin,
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
