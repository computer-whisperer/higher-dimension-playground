use super::chunk_array_ops::{
    chunk_array_has_non_empty_intersection, consolidate_chunk_array_children,
};
use super::tree::*;
use super::*;
use crate::shared::spatial::ChunkCoord;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Adjacent merge types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum AdjacentMergeKind {
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

pub(super) fn adjacent_merge_kind(kind: &RegionNodeKind) -> Option<AdjacentMergeKind> {
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

// ---------------------------------------------------------------------------
// Core normalization
// ---------------------------------------------------------------------------

pub(super) fn normalize_chunk_node(node: &mut RegionTreeCore) {
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

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

pub(super) fn prune_empty_subtrees(core: &mut RegionTreeCore) -> bool {
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
