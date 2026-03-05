//! Temp-tree construction and serialization for the save-v4 index.

use super::*;

#[derive(Clone, Debug)]
pub(super) struct LeafDescriptor {
    pub min: ChunkKey,
    pub max: ChunkKey,
    pub scale_exp: i8,
    pub kind: IndexNodeKind,
}

#[derive(Clone, Debug)]
pub(super) enum TempNodeKind {
    Leaf(IndexNodeKind),
    Branch(Vec<TempNode>),
}

#[derive(Clone, Debug)]
pub(super) struct TempNode {
    pub min: ChunkKey,
    pub max: ChunkKey,
    pub scale_exp: i8,
    pub kind: TempNodeKind,
}

pub(super) fn bounds_key(min: ChunkKey, max: ChunkKey) -> [i64; 8] {
    [
        min[0].to_bits(),
        min[1].to_bits(),
        min[2].to_bits(),
        min[3].to_bits(),
        max[0].to_bits(),
        max[1].to_bits(),
        max[2].to_bits(),
        max[3].to_bits(),
    ]
}

pub(super) fn fixed_bounds_key(min: [i64; 4], max: [i64; 4]) -> [i64; 8] {
    [
        min[0], min[1], min[2], min[3], max[0], max[1], max[2], max[3],
    ]
}

pub(super) fn make_empty_branch_root(min: ChunkKey, max: ChunkKey) -> TempNode {
    TempNode {
        min,
        max,
        scale_exp: 0,
        kind: TempNodeKind::Branch(Vec::new()),
    }
}

pub(super) fn build_temp_tree_from_leaves(leaves: &[LeafDescriptor]) -> Option<TempNode> {
    if leaves.is_empty() {
        return None;
    }

    if leaves.len() == 1 {
        let leaf = &leaves[0];
        return Some(TempNode {
            min: leaf.min,
            max: leaf.max,
            scale_exp: leaf.scale_exp,
            kind: TempNodeKind::Leaf(leaf.kind.clone()),
        });
    }

    let mut min = [ChunkCoord::MAX; 4];
    let mut max = [ChunkCoord::MIN; 4];
    for leaf in leaves {
        for axis in 0..4 {
            min[axis] = min[axis].min(leaf.min[axis]);
            max[axis] = max[axis].max(leaf.max[axis]);
        }
    }

    // Save-v4 index validation requires direct branch siblings to have non-overlapping
    // hypervolumes. A BVH split can create overlapping sibling bounds even when leaf
    // volumes themselves are disjoint, so emit direct leaf children here.
    let mut children = Vec::with_capacity(leaves.len());
    for leaf in leaves {
        children.push(TempNode {
            min: leaf.min,
            max: leaf.max,
            scale_exp: leaf.scale_exp,
            kind: TempNodeKind::Leaf(leaf.kind.clone()),
        });
    }
    children.sort_unstable_by_key(temp_node_order_key);

    Some(TempNode {
        min,
        max,
        scale_exp: 0, // Branch nodes don't have a meaningful scale_exp
        kind: TempNodeKind::Branch(children),
    })
}

pub(super) fn build_temp_tree_from_index_subtree(
    node_by_id: &HashMap<u32, &IndexNode>,
    node_id: u32,
) -> io::Result<TempNode> {
    let Some(node) = node_by_id.get(&node_id).copied() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("index references missing node id {node_id}"),
        ));
    };
    let kind = match &node.kind {
        IndexNodeKind::Branch { child_node_ids } => {
            let mut children = Vec::with_capacity(child_node_ids.len());
            for child_id in child_node_ids {
                children.push(build_temp_tree_from_index_subtree(node_by_id, *child_id)?);
            }
            TempNodeKind::Branch(children)
        }
        leaf_kind => TempNodeKind::Leaf(leaf_kind.clone()),
    };
    Ok(TempNode {
        min: node.bounds_min_fixed.map(ChunkCoord::from_bits),
        max: node.bounds_max_fixed.map(ChunkCoord::from_bits),
        scale_exp: node.scale_exp,
        kind,
    })
}

pub(super) fn canonicalize_temp_tree(node: &mut TempNode) {
    let TempNodeKind::Branch(children) = &mut node.kind else {
        return;
    };
    for child in children.iter_mut() {
        canonicalize_temp_tree(child);
    }
    children.sort_unstable_by_key(temp_node_order_key);
}

pub(super) fn temp_node_order_key(node: &TempNode) -> ([i64; 8], u8, u64, u64, u64, u64, u64) {
    let (kind_rank, v0, v1, v2, v3, v4) = temp_kind_order_key(&node.kind);
    (
        bounds_key(node.min, node.max),
        kind_rank,
        v0,
        v1,
        v2,
        v3,
        v4,
    )
}

fn temp_kind_order_key(kind: &TempNodeKind) -> (u8, u64, u64, u64, u64, u64) {
    match kind {
        TempNodeKind::Branch(_) => (0, 0, 0, 0, 0, 0),
        TempNodeKind::Leaf(index_kind) => index_kind_order_key(index_kind),
    }
}

pub(super) fn index_kind_order_key(kind: &IndexNodeKind) -> (u8, u64, u64, u64, u64, u64) {
    match kind {
        IndexNodeKind::Branch { child_node_ids } => (0, child_node_ids.len() as u64, 0, 0, 0, 0),
        IndexNodeKind::LeafEmpty => (1, 0, 0, 0, 0, 0),
        IndexNodeKind::LeafUniform { block } => {
            use std::hash::{Hash, Hasher as StdHasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            block.hash(&mut hasher);
            let h = hasher.finish();
            (2, h, 0, 0, 0, 0)
        }
        IndexNodeKind::LeafChunkArray { chunk_array_ref } => (
            3,
            u64::from(chunk_array_ref.data_file_id),
            chunk_array_ref.record_offset,
            u64::from(chunk_array_ref.record_len),
            u64::from(chunk_array_ref.record_crc32),
            u64::from(chunk_array_ref.blob_type) | (u64::from(chunk_array_ref.blob_version) << 32),
        ),
    }
}

pub(super) fn flatten_temp_tree(node: &TempNode, out: &mut Vec<IndexNode>) -> u32 {
    let kind = match &node.kind {
        TempNodeKind::Leaf(kind) => kind.clone(),
        TempNodeKind::Branch(children) => {
            let mut child_ids = Vec::with_capacity(children.len());
            for child in children {
                child_ids.push(flatten_temp_tree(child, out));
            }
            IndexNodeKind::Branch {
                child_node_ids: child_ids,
            }
        }
    };

    let node_id = out.len() as u32;
    out.push(IndexNode {
        node_id,
        bounds_min_fixed: node.min.map(|c| c.to_bits()),
        bounds_max_fixed: node.max.map(|c| c.to_bits()),
        scale_exp: node.scale_exp,
        kind,
    });
    node_id
}
