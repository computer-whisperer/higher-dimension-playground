use super::chunk_array_ops::chunk_array_palette_non_empty_mask;
use super::tree::RegionChunkTree;
use super::*;
use crate::shared::spatial::{chunk_key_from_lattice, Aabb4i};
use crate::shared::voxel::linear_cell_index;

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
            let cells = ca
                .bounds
                .chunk_cell_count_at_scale(ca.scale_exp)
                .unwrap_or(0);
            report.total_chunk_cells += cells;
            *scale_counts.entry(ca.scale_exp).or_default() += 1;
        }
        RegionNodeKind::Branch(children) => {
            report.branch_count += 1;
            // Check sibling bounds overlaps.
            for i in 0..children.len() {
                for j in (i + 1)..children.len() {
                    if children[i].bounds.intersects(&children[j].bounds) {
                        report
                            .bounds_overlaps
                            .push((children[i].bounds, children[j].bounds));
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
                    a.min, a.max, b.min, b.max,
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
                                return Err(
                                    "chunk-array index out of bounds while validating overlap"
                                        .to_string(),
                                );
                            };
                            if !palette_non_empty
                                .get(*palette_idx as usize)
                                .copied()
                                .unwrap_or(true)
                            {
                                continue;
                            }
                            let pos = chunk_key_from_lattice([lx, ly, lz, lw], se);
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
