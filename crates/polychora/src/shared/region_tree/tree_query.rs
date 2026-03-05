use super::chunk_array_ops::{
    chunk_array_has_non_empty_intersection, chunk_array_palette_non_empty_mask,
    chunk_array_resolved_payload_at,
};
use super::tree::*;
use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::spatial::{
    chunk_key_from_lattice, fixed_from_lattice, lattice_from_fixed, step_for_scale, Aabb4i,
    ChunkCoord,
};
use crate::shared::voxel::{linear_cell_index, BlockData, CHUNK_SIZE};

// ---------------------------------------------------------------------------
// Chunk payload query
// ---------------------------------------------------------------------------

pub(super) fn query_chunk_payload_in_node(
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
            let chunk_scale = bounds.coarsest_lattice_aligned_scale(block.scale_exp);
            Some((ResolvedChunkPayload::uniform(block.clone()), chunk_scale))
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

// ---------------------------------------------------------------------------
// Intersection / counting queries
// ---------------------------------------------------------------------------

pub(super) fn kind_has_non_empty_chunk_intersection(
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

pub(super) fn count_non_empty_chunks(kind: &RegionNodeKind, bounds: Aabb4i) -> usize {
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

// ---------------------------------------------------------------------------
// Collect chunks
// ---------------------------------------------------------------------------

pub(super) fn collect_chunks_from_kind(
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
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], 0),
                                resolved.clone(),
                            ));
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

pub(super) fn collect_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ResolvedChunkPayload)>,
) {
    let Some(_intersection) = bounds.intersection(&query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            let resolved = ResolvedChunkPayload::uniform(block.clone());
            let intersection = bounds.intersection(&query_bounds).unwrap();
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], 0),
                                resolved.clone(),
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = chunk_array.bounds.intersection(&query_bounds)
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

pub(super) fn collect_non_empty_chunks_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, ResolvedChunkPayload)>,
) {
    let Some(_intersection) = bounds.intersection(&query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return;
            }
            let intersection = bounds.intersection(&query_bounds).unwrap();
            let resolved = ResolvedChunkPayload::uniform(block.clone());
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(0);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((
                                chunk_key_from_lattice([lx, ly, lz, lw], 0),
                                resolved.clone(),
                            ));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = chunk_array.bounds.intersection(&query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array
                .bounds
                .chunk_extents_at_scale(chunk_array.scale_exp)
            else {
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

pub(super) fn collect_non_empty_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = bounds.intersection(&query_bounds) else {
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
            let Some(chunk_array_intersection) = chunk_array.bounds.intersection(&query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array
                .bounds
                .chunk_extents_at_scale(chunk_array.scale_exp)
            else {
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

pub(super) fn collect_chunk_entries_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<(ChunkKey, i8)>,
) {
    let Some(intersection) = bounds.intersection(&query_bounds) else {
        return;
    };

    match kind {
        RegionNodeKind::Empty | RegionNodeKind::ProceduralRef(_) => {}
        RegionNodeKind::Uniform(block) => {
            if block.is_air() {
                return;
            }
            let se = bounds.coarsest_lattice_aligned_scale(block.scale_exp);
            let (lmin, lmax) = intersection.to_chunk_lattice_bounds(se);
            for lw in lmin[3]..=lmax[3] {
                for lz in lmin[2]..=lmax[2] {
                    for ly in lmin[1]..=lmax[1] {
                        for lx in lmin[0]..=lmax[0] {
                            out.push((chunk_key_from_lattice([lx, ly, lz, lw], se), se));
                        }
                    }
                }
            }
        }
        RegionNodeKind::ChunkArray(chunk_array) => {
            let Some(chunk_array_intersection) = chunk_array.bounds.intersection(&query_bounds)
            else {
                return;
            };
            let Ok(indices) = chunk_array.decode_dense_indices() else {
                return;
            };
            let Some(extents) = chunk_array
                .bounds
                .chunk_extents_at_scale(chunk_array.scale_exp)
            else {
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
                            out.push((chunk_key_from_lattice([lx, ly, lz, lw], se), se));
                        }
                    }
                }
            }
        }
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunk_entries_from_kind_in_bounds(
                    &child.kind,
                    child.bounds,
                    query_bounds,
                    out,
                );
            }
        }
    }
}

pub(super) fn collect_chunk_keys_from_kind_in_bounds(
    kind: &RegionNodeKind,
    bounds: Aabb4i,
    query_bounds: Aabb4i,
    out: &mut Vec<ChunkKey>,
) {
    let Some(intersection) = bounds.intersection(&query_bounds) else {
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

pub(super) fn find_leaf_chunks_in_spatial_range_recursive(
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
            let se = kind_bounds.coarsest_lattice_aligned_scale(block.scale_exp);
            if !kind_bounds.intersects(query) {
                return;
            }
            if is_single_chunk_bounds(kind_bounds, se) {
                out.push((kind_bounds.chunk_key_from_world_bounds(), se));
            } else {
                let (lmin, lmax) = kind_bounds.to_chunk_lattice_bounds(se);
                for lw in lmin[3]..=lmax[3] {
                    for lz in lmin[2]..=lmax[2] {
                        for ly in lmin[1]..=lmax[1] {
                            for lx in lmin[0]..=lmax[0] {
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
        }
        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
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
                find_leaf_chunks_in_spatial_range_recursive(&child.kind, child.bounds, query, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Non-empty chunks from core (public)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Semantic equality helpers
// ---------------------------------------------------------------------------

pub(super) fn non_empty_kinds_semantically_equal_in_bounds(
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
    lhs_chunks
        .iter()
        .zip(rhs_chunks.iter())
        .all(|((lk, lp), (rk, rp))| lk == rk && resolved_payloads_semantically_equal(lp, rp))
}

pub(super) fn resolved_payloads_semantically_equal(
    a: &ResolvedChunkPayload,
    b: &ResolvedChunkPayload,
) -> bool {
    let a_dense = a.payload.dense_materials();
    let b_dense = b.payload.dense_materials();
    match (a_dense, b_dense) {
        (Ok(ad), Ok(bd)) => {
            if ad.len() != bd.len() {
                return false;
            }
            ad.iter().zip(bd.iter()).all(|(ai, bi)| {
                let a_block = a
                    .block_palette
                    .get(*ai as usize)
                    .cloned()
                    .unwrap_or(BlockData::AIR);
                let b_block = b
                    .block_palette
                    .get(*bi as usize)
                    .cloned()
                    .unwrap_or(BlockData::AIR);
                a_block == b_block
            })
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// BVH spatial queries -- point query and raycast
// ---------------------------------------------------------------------------

/// Result of a BVH point query: the block and its world-space AABB.
#[derive(Clone, Debug)]
pub struct BvhBlockHit {
    pub block: BlockData,
    /// World-space extent of the block, accounting for `block.scale_exp`.
    pub bounds: Aabb4i,
}

/// Result of a 4D ray-AABB slab intersection with face tracking.
#[derive(Clone, Debug)]
pub struct RayAabbHit {
    pub t_enter: ChunkCoord,
    pub t_exit: ChunkCoord,
    /// Which axis (0..3) the ray last crossed to enter the AABB.
    pub face_axis: u8,
    /// -1 = entered through the min face, +1 = entered through the max face.
    /// 0 = ray started inside the AABB.
    pub face_sign: i8,
}

/// Result of a BVH raycast: block hit info plus the ray parameter.
#[derive(Clone, Debug)]
pub struct BvhRayHit {
    pub block: BlockData,
    /// World-space extent of the hit block.
    pub bounds: Aabb4i,
    /// Ray parameter at entry to the block's AABB.
    pub t: ChunkCoord,
    /// World-space hit point: `origin + direction * t`.
    pub hit_point: [ChunkCoord; 4],
    /// Which axis (0..3) the ray crossed to enter the hit block.
    pub face_axis: u8,
    /// -1 = entered through the min face, +1 = entered through the max face.
    pub face_sign: i8,
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
        let cell_lat = lattice_from_fixed(pos[axis], scale_exp);
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
fn block_bounds(cell_min: [ChunkCoord; 4], chunk_scale_exp: i8, block: &BlockData) -> Aabb4i {
    let bs = block.scale_exp;
    if bs <= chunk_scale_exp {
        let step = step_for_scale(chunk_scale_exp);
        return Aabb4i::new(
            cell_min,
            std::array::from_fn(|i| cell_min[i].saturating_add(step)),
        );
    }
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
pub(super) fn bvh_point_query(
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
            let (_, _, cell_aabb) = cell_at_point(pos, 0);
            let bb = block_bounds(cell_aabb.min, 0, block);
            Some(BvhBlockHit {
                block: block.clone(),
                bounds: bb,
            })
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
pub(super) fn bvh_block_data_at_point(
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

/// 4D ray-AABB slab intersection with face tracking.
pub(super) fn ray_aabb_intersect(
    origin: [ChunkCoord; 4],
    direction: [ChunkCoord; 4],
    aabb: &Aabb4i,
) -> Option<RayAabbHit> {
    let mut t_min = ChunkCoord::MIN;
    let mut t_max = ChunkCoord::MAX;
    let mut face_axis: u8 = 0;
    let mut face_sign: i8 = 0;
    for axis in 0..4 {
        let d = direction[axis];
        if d == ChunkCoord::ZERO {
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
            if t_near > t_min {
                t_min = t_near;
                face_axis = axis as u8;
                face_sign = if inv_d_positive { -1 } else { 1 };
            }
            if t_far < t_max {
                t_max = t_far;
            }
            if t_min > t_max {
                return None;
            }
        }
    }
    Some(RayAabbHit {
        t_enter: t_min,
        t_exit: t_max,
        face_axis,
        face_sign,
    })
}

/// Recursive BVH raycast. Returns the nearest solid hit.
pub(super) fn bvh_raycast(
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
            let region_hit = ray_aabb_intersect(origin, direction, &bounds)?;
            if region_hit.t_enter > max_t || region_hit.t_exit < ChunkCoord::ZERO {
                return None;
            }
            let nudge = ChunkCoord::from_bits(1);
            let t_sample = (if region_hit.t_enter > ChunkCoord::ZERO {
                region_hit.t_enter
            } else {
                ChunkCoord::ZERO
            })
            .saturating_add(nudge);
            let hit_pos: [ChunkCoord; 4] = std::array::from_fn(|i| {
                origin[i].saturating_add(direction[i].saturating_mul(t_sample))
            });
            let (_, _, cell_aabb) = cell_at_point(hit_pos, 0);
            let bb = block_bounds(cell_aabb.min, 0, block);
            let t = region_hit.t_enter.max(ChunkCoord::ZERO);
            let hit_point =
                std::array::from_fn(|i| origin[i].saturating_add(direction[i].saturating_mul(t)));
            Some(BvhRayHit {
                block: block.clone(),
                bounds: bb,
                t,
                hit_point,
                face_axis: region_hit.face_axis,
                face_sign: region_hit.face_sign,
            })
        }

        RegionNodeKind::ChunkArray(ca) => {
            let se = ca.scale_exp;
            let ca_hit = ray_aabb_intersect(origin, direction, &ca.bounds)?;
            if ca_hit.t_enter > max_t {
                return None;
            }
            let t_end = ca_hit.t_exit.min(max_t);
            let t_start = ca_hit.t_enter.max(ChunkCoord::ZERO);
            if t_start > t_end {
                return None;
            }

            let dense_indices = ca.decode_dense_indices().ok()?;
            let chunk_extents = ca.bounds.chunk_extents_at_scale(se)?;
            let step = step_for_scale(se);
            let cs = CHUNK_SIZE as i64;

            let cell_lmin: [i32; 4] =
                std::array::from_fn(|i| lattice_from_fixed(ca.bounds.min[i], se));
            let cell_lmax: [i32; 4] =
                std::array::from_fn(|i| lattice_from_fixed(ca.bounds.max[i], se) - 1);

            let nudge = ChunkCoord::from_bits(1);
            let t_entry_clamped = t_start.saturating_add(nudge);
            let entry_pos: [ChunkCoord; 4] = std::array::from_fn(|i| {
                origin[i].saturating_add(direction[i].saturating_mul(t_entry_clamped))
            });

            let mut cell_lat: [i32; 4] =
                std::array::from_fn(|i| lattice_from_fixed(entry_pos[i], se));

            for axis in 0..4 {
                cell_lat[axis] = cell_lat[axis].clamp(cell_lmin[axis], cell_lmax[axis]);
            }

            // --- DDA initialization ---
            let mut step_dir = [0i32; 4];
            let mut t_max_arr = [ChunkCoord::MAX; 4];
            let mut t_delta = [ChunkCoord::MAX; 4];

            for axis in 0..4 {
                let d = direction[axis];
                if d == ChunkCoord::ZERO {
                    continue;
                }
                if d > ChunkCoord::ZERO {
                    step_dir[axis] = 1;
                    let next_boundary = fixed_from_lattice(cell_lat[axis] + 1, se);
                    t_max_arr[axis] = (next_boundary - origin[axis]) / d;
                } else {
                    step_dir[axis] = -1;
                    let next_boundary = fixed_from_lattice(cell_lat[axis], se);
                    t_max_arr[axis] = (next_boundary - origin[axis]) / d;
                }
                t_delta[axis] = step / d.abs();
            }

            let mut face_axis = ca_hit.face_axis;
            let mut face_sign = ca_hit.face_sign;
            let mut current_t = t_start;

            loop {
                if (0..4).any(|i| cell_lat[i] < cell_lmin[i] || cell_lat[i] > cell_lmax[i]) {
                    break;
                }

                let cell_rel: [i64; 4] =
                    std::array::from_fn(|i| (cell_lat[i] - cell_lmin[i]) as i64);
                let chunk_local: [usize; 4] =
                    std::array::from_fn(|i| cell_rel[i].div_euclid(cs) as usize);
                let voxel_local: [usize; 4] =
                    std::array::from_fn(|i| cell_rel[i].rem_euclid(cs) as usize);

                let chunk_linear = linear_cell_index(chunk_local, chunk_extents);

                if let Some(&palette_idx) = dense_indices.get(chunk_linear) {
                    let palette_idx = palette_idx as usize;
                    if let Some(chunk_payload) = ca.chunk_palette.get(palette_idx) {
                        let resolved = ResolvedChunkPayload {
                            payload: chunk_payload.clone(),
                            block_palette: ca.block_palette.clone(),
                        };
                        let voxel_linear = linear_cell_index(voxel_local, [CHUNK_SIZE; 4]);
                        let block = resolved.block_at(voxel_linear);

                        if !block.is_air() {
                            let cell_min: [ChunkCoord; 4] =
                                std::array::from_fn(|i| fixed_from_lattice(cell_lat[i], se));
                            let bb = block_bounds(cell_min, se, &block);

                            if block.scale_exp > se {
                                if let Some(bh) = ray_aabb_intersect(origin, direction, &bb) {
                                    let block_t = bh.t_enter.max(ChunkCoord::ZERO);
                                    let hit_point = std::array::from_fn(|i| {
                                        origin[i]
                                            .saturating_add(direction[i].saturating_mul(block_t))
                                    });
                                    return Some(BvhRayHit {
                                        block,
                                        bounds: bb,
                                        t: block_t,
                                        hit_point,
                                        face_axis: bh.face_axis,
                                        face_sign: bh.face_sign,
                                    });
                                }
                            } else {
                                let hit_point = std::array::from_fn(|i| {
                                    origin[i].saturating_add(direction[i].saturating_mul(current_t))
                                });
                                return Some(BvhRayHit {
                                    block,
                                    bounds: bb,
                                    t: current_t,
                                    hit_point,
                                    face_axis,
                                    face_sign,
                                });
                            }
                        }
                    }
                }

                // Advance DDA: step the axis with the smallest t_max.
                let mut min_axis = 0;
                let mut min_t = t_max_arr[0];
                for (axis, &t_val) in t_max_arr.iter().enumerate().skip(1) {
                    if t_val < min_t {
                        min_t = t_val;
                        min_axis = axis;
                    }
                }

                if min_t > t_end {
                    break;
                }

                cell_lat[min_axis] += step_dir[min_axis];
                current_t = min_t;
                face_axis = min_axis as u8;
                face_sign = -(step_dir[min_axis] as i8);
                t_max_arr[min_axis] = t_max_arr[min_axis].saturating_add(t_delta[min_axis]);
            }
            None
        }

        RegionNodeKind::Branch(children) => {
            let mut candidates: Vec<(ChunkCoord, usize)> = Vec::new();
            for (idx, child) in children.iter().enumerate() {
                if let Some(ch) = ray_aabb_intersect(origin, direction, &child.bounds) {
                    if ch.t_enter <= max_t && ch.t_exit >= ChunkCoord::ZERO {
                        candidates.push((ch.t_enter, idx));
                    }
                }
            }
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut best: Option<BvhRayHit> = None;
            for (_, idx) in candidates {
                let child = &children[idx];
                if let Some(ref b) = best {
                    if let Some(ch) = ray_aabb_intersect(origin, direction, &child.bounds) {
                        if ch.t_enter > b.t {
                            continue;
                        }
                    }
                }
                let effective_max = best.as_ref().map(|b| b.t).unwrap_or(max_t);
                if let Some(hit) =
                    bvh_raycast(&child.kind, child.bounds, origin, direction, effective_max)
                {
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

#[cfg(test)]
mod ray_aabb_tests {
    use super::*;

    fn cc(v: i32) -> ChunkCoord {
        ChunkCoord::from_num(v)
    }

    fn unit_aabb() -> Aabb4i {
        Aabb4i::new([cc(0); 4], [cc(1); 4])
    }

    #[test]
    fn ray_from_each_face_direction() {
        let aabb = unit_aabb();
        for axis in 0..4u8 {
            for &(sign, start_val, expected_face_sign) in &[(-1i8, 2, 1i8), (1i8, -1, -1i8)] {
                let mut origin = [cc(0); 4];
                for a in 0..4 {
                    if a != axis as usize {
                        origin[a] = ChunkCoord::from_num(0.5f32);
                    }
                }
                origin[axis as usize] = cc(start_val);

                let mut direction = [cc(0); 4];
                direction[axis as usize] = cc(sign as i32);

                let hit = ray_aabb_intersect(origin, direction, &aabb)
                    .unwrap_or_else(|| panic!("Expected hit for axis={axis} sign={sign}"));

                assert_eq!(hit.face_axis, axis, "face_axis: axis={axis} sign={sign}");
                assert_eq!(
                    hit.face_sign, expected_face_sign,
                    "face_sign: axis={axis} sign={sign}"
                );
                assert!(
                    hit.t_enter <= hit.t_exit,
                    "t_enter <= t_exit: axis={axis} sign={sign}"
                );
            }
        }
    }

    #[test]
    fn ray_starting_inside_has_zero_face_sign() {
        let aabb = unit_aabb();
        let origin = [ChunkCoord::from_num(0.5f32); 4];
        let direction = [cc(1), cc(0), cc(0), cc(0)];
        let hit = ray_aabb_intersect(origin, direction, &aabb).unwrap();
        assert!(hit.t_enter < cc(0));
        assert_eq!(hit.face_sign, -1);
    }

    #[test]
    fn miss_returns_none() {
        let aabb = unit_aabb();
        let origin = [cc(0), cc(5), cc(0), cc(0)];
        let direction = [cc(1), cc(0), cc(0), cc(0)];
        assert!(ray_aabb_intersect(origin, direction, &aabb).is_none());
    }
}
