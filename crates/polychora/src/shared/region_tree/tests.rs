use super::*;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::Aabb4i;

fn key(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    [x, y, z, w]
}

fn max_depth(node: &RegionTreeCore) -> usize {
    match &node.kind {
        RegionNodeKind::Branch(children) => 1 + children.iter().map(max_depth).max().unwrap_or(0),
        _ => 1,
    }
}

#[test]
fn set_get_remove_single_chunk_roundtrip() {
    let mut tree = RegionChunkTree::new();
    assert!(!tree.has_chunk(key(0, 0, 0, 0)));

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(12))));
    assert!(tree.has_chunk(key(0, 0, 0, 0)));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(12))
    );

    assert!(tree.remove_chunk(key(0, 0, 0, 0)));
    assert!(!tree.has_chunk(key(0, 0, 0, 0)));
    assert!(tree.root().is_none());
}

#[test]
fn set_chunk_uniform_zero_on_empty_tree_is_explicit_override() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(0))));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(0))
    );
}

#[test]
fn uniform_merge_and_fragment_behavior_is_stable() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(7))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(7))));

    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(7)));

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(9))));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(9))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)),
        Some(ChunkPayload::Uniform(7))
    );

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(7))));
    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(7)));
}

#[test]
fn splice_non_empty_core_replaces_window_contents() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

    let bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
    let patch_core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(9),
        generator_version_hash: 0,
    };

    let changed_bounds = tree.splice_non_empty_core_in_bounds(bounds, &patch_core);
    assert_eq!(changed_bounds, Some(bounds));
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)),
        Some(ChunkPayload::Uniform(9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(2))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)),
        Some(ChunkPayload::Uniform(4))
    );
}

#[test]
fn take_non_empty_core_extracts_and_clears_region() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(3))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

    let bounds = Aabb4i::new([1, 0, 0, 0], [2, 0, 0, 0]);
    let extracted = tree.take_non_empty_core_in_bounds(bounds);
    let mut extracted_chunks = collect_non_empty_chunks_from_core_in_bounds(&extracted, bounds);
    extracted_chunks.sort_unstable_by_key(|(key, _)| *key);
    assert_eq!(
        extracted_chunks,
        vec![
            (key(1, 0, 0, 0), ChunkPayload::Uniform(3)),
            (key(2, 0, 0, 0), ChunkPayload::Uniform(4)),
        ]
    );

    assert_eq!(tree.chunk_payload(key(1, 0, 0, 0)), None);
    assert_eq!(tree.chunk_payload(key(2, 0, 0, 0)), None);
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(2))
    );
}

#[test]
fn lazy_drop_outside_bounds_respects_budget() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(-8, 0, 0, 0), Some(ChunkPayload::Uniform(6))));
    assert!(tree.set_chunk(key(8, 0, 0, 0), Some(ChunkPayload::Uniform(7))));

    let keep_bounds = Aabb4i::new([-9, -1, -1, -1], [-7, 1, 1, 1]);
    assert!(tree.lazy_drop_outside_bounds(keep_bounds, 0).is_none());
    let changed = tree.lazy_drop_outside_bounds(keep_bounds, 8);
    assert!(changed.is_some());
    assert!(tree.chunk_payload(key(-8, 0, 0, 0)).is_some());
    assert!(tree.chunk_payload(key(8, 0, 0, 0)).is_none());
}

#[test]
fn slice_preserves_query_bounds() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(2))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(4))));

    let bounds = Aabb4i::new([2, 0, 0, 0], [3, 0, 0, 0]);
    let slice = tree.slice_core_in_bounds(bounds);
    assert_eq!(slice.bounds, bounds);
    let mut chunks = collect_non_empty_chunks_from_core_in_bounds(&slice, bounds);
    chunks.sort_unstable_by_key(|(key, _)| *key);
    assert_eq!(chunks, vec![(key(2, 0, 0, 0), ChunkPayload::Uniform(4))]);
}

#[test]
fn set_chunk_carves_large_uniform_leaf_locally() {
    let mut tree = RegionChunkTree::new();
    let bounds = Aabb4i::new([0, 0, 0, 0], [7, 7, 7, 7]);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(4),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &core),
        Some(bounds)
    );

    assert!(tree.set_chunk(key(3, 4, 2, 5), Some(ChunkPayload::Uniform(9))));
    assert_eq!(
        tree.chunk_payload(key(3, 4, 2, 5)),
        Some(ChunkPayload::Uniform(9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(4))
    );

    let root = tree.root().expect("root exists");
    assert_eq!(max_depth(root), 2);
    let RegionNodeKind::Branch(children) = &root.kind else {
        panic!("expected branch after carving");
    };
    assert!(
        children.len() <= 9,
        "unexpected child count {}",
        children.len()
    );
    assert!(
        children
            .iter()
            .all(|child| !matches!(child.kind, RegionNodeKind::Branch(_))),
        "carve should not create nested branch path"
    );
}

#[test]
fn set_chunk_same_uniform_value_is_noop_without_fragmentation() {
    let mut tree = RegionChunkTree::new();
    let bounds = Aabb4i::new([0, 0, 0, 0], [7, 7, 7, 7]);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(6),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &core),
        Some(bounds)
    );

    assert!(!tree.set_chunk(key(4, 3, 2, 1), Some(ChunkPayload::Uniform(6))));
    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(6)));
}

#[test]
fn set_chunk_carves_large_procedural_leaf_locally() {
    let mut tree = RegionChunkTree::new();
    let bounds = Aabb4i::new([-4, -4, -4, -4], [4, 4, 4, 4]);
    let generator = GeneratorRef {
        generator_id: "test-generator".to_string(),
        params: vec![1, 2, 3],
        seed: 42,
    };
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::ProceduralRef(generator.clone()),
        generator_version_hash: 17,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &core),
        Some(bounds)
    );

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(12))));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(12))
    );

    let root = tree.root().expect("root exists");
    assert_eq!(max_depth(root), 2);
    let RegionNodeKind::Branch(children) = &root.kind else {
        panic!("expected branch after procedural carve");
    };
    assert!(
        children.len() <= 9,
        "unexpected child count {}",
        children.len()
    );
    assert!(children.iter().any(|child| {
        child.bounds == Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0])
            && matches!(child.kind, RegionNodeKind::Uniform(12))
    }));
    assert!(children.iter().any(|child| {
        child.bounds != Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0])
            && matches!(child.kind, RegionNodeKind::ProceduralRef(ref g) if *g == generator)
    }));
}

#[test]
fn splice_identical_partial_uniform_region_is_noop() {
    let mut tree = RegionChunkTree::new();
    let root_bounds = Aabb4i::new([0, 0, 0, 0], [31, 7, 31, 7]);
    let root_core = RegionTreeCore {
        bounds: root_bounds,
        kind: RegionNodeKind::Uniform(2),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(root_bounds, &root_core),
        Some(root_bounds)
    );
    let before = tree.root().expect("root").clone();

    let patch_bounds = Aabb4i::new([4, 1, 4, 1], [20, 5, 20, 5]);
    let patch_core = RegionTreeCore {
        bounds: patch_bounds,
        kind: RegionNodeKind::Uniform(2),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core),
        None
    );

    let after = tree.root().expect("root");
    assert_eq!(*after, before);
    assert!(matches!(after.kind, RegionNodeKind::Uniform(2)));
}

#[test]
fn splice_non_empty_semantic_noop_skips_structural_rewrite() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(3))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ChunkPayload::Uniform(4))));
    let before = tree.root().expect("root").clone();

    let patch_bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
    let patch_chunk_array = ChunkArrayData::from_dense_indices(
        patch_bounds,
        vec![ChunkPayload::Uniform(3), ChunkPayload::Uniform(4)],
        vec![0, 1],
        None,
    )
    .expect("chunk array");
    let patch_core = RegionTreeCore {
        bounds: patch_bounds,
        kind: RegionNodeKind::ChunkArray(patch_chunk_array),
        generator_version_hash: 0,
    };

    assert_eq!(
        tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core),
        None
    );
    assert_eq!(tree.root(), Some(&before));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(3))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)),
        Some(ChunkPayload::Uniform(4))
    );
}

#[test]
fn overlay_core_applies_explicit_uniform_zero_and_non_zero_leaves() {
    let bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
    let mut tree = RegionChunkTree::new();
    let base = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(7),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &base),
        Some(bounds)
    );

    let zero_leaf_bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
    let nine_leaf_bounds = Aabb4i::new([2, 0, 0, 0], [2, 0, 0, 0]);
    let overlay = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Branch(vec![
            RegionTreeCore {
                bounds: zero_leaf_bounds,
                kind: RegionNodeKind::Uniform(0),
                generator_version_hash: 0,
            },
            RegionTreeCore {
                bounds: nine_leaf_bounds,
                kind: RegionNodeKind::Uniform(9),
                generator_version_hash: 0,
            },
        ]),
        generator_version_hash: 0,
    };

    assert!(tree.overlay_core_in_bounds(bounds, &overlay).is_some());
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(7))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)),
        Some(ChunkPayload::Uniform(0))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)),
        Some(ChunkPayload::Uniform(9))
    );
    assert_eq!(
        tree.chunk_payload(key(3, 0, 0, 0)),
        Some(ChunkPayload::Uniform(7))
    );
}
