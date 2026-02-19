use super::*;
use crate::shared::chunk_payload::ChunkPayload;
use crate::shared::spatial::Aabb4i;

fn key(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    [x, y, z, w]
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
