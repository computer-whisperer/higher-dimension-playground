use super::*;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::BlockData;
use std::collections::HashMap;

fn key(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    [x, y, z, w]
}

fn max_depth(node: &RegionTreeCore) -> usize {
    match &node.kind {
        RegionNodeKind::Branch(children) => 1 + children.iter().map(max_depth).max().unwrap_or(0),
        _ => 1,
    }
}

#[derive(Clone, Copy)]
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_inclusive_i32(&mut self, lo: i32, hi: i32) -> i32 {
        debug_assert!(lo <= hi);
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }
}

fn keys_in_bounds(bounds: Aabb4i) -> Vec<ChunkKey> {
    let mut keys = Vec::new();
    for w in bounds.min[3]..=bounds.max[3] {
        for z in bounds.min[2]..=bounds.max[2] {
            for y in bounds.min[1]..=bounds.max[1] {
                for x in bounds.min[0]..=bounds.max[0] {
                    keys.push([x, y, z, w]);
                }
            }
        }
    }
    keys
}

fn random_sub_bounds(rng: &mut TestRng, outer: Aabb4i) -> Aabb4i {
    let mut min = [0i32; 4];
    let mut max = [0i32; 4];
    for axis in 0..4 {
        let lo = rng.next_inclusive_i32(outer.min[axis], outer.max[axis]);
        let hi = rng.next_inclusive_i32(lo, outer.max[axis]);
        min[axis] = lo;
        max[axis] = hi;
    }
    Aabb4i::new(min, max)
}

fn linear_index_in_bounds(bounds: Aabb4i, key: ChunkKey) -> usize {
    let extents = bounds
        .chunk_extents()
        .expect("bounds used in tests must be valid");
    let lx = (key[0] - bounds.min[0]) as usize;
    let ly = (key[1] - bounds.min[1]) as usize;
    let lz = (key[2] - bounds.min[2]) as usize;
    let lw = (key[3] - bounds.min[3]) as usize;
    lx + extents[0] * (ly + extents[1] * (lz + extents[2] * lw))
}

fn chunk_array_uniform_palette_core(bounds: Aabb4i, materials: &[u16]) -> RegionTreeCore {
    let expected_cells = bounds
        .chunk_cell_count()
        .expect("bounds used in tests must be valid");
    assert_eq!(materials.len(), expected_cells);

    // Build block palette: index 0 = AIR, then each unique non-zero material.
    let mut block_palette = vec![BlockData::AIR];
    let mut material_to_block_idx = HashMap::<u16, u16>::new();
    material_to_block_idx.insert(0, 0);
    for &material in materials {
        if !material_to_block_idx.contains_key(&material) {
            let idx = block_palette.len() as u16;
            block_palette.push(BlockData::simple(0, material as u32));
            material_to_block_idx.insert(material, idx);
        }
    }

    // Build chunk palette: each unique material maps to a ChunkPayload::Uniform(block_idx).
    let mut chunk_palette_materials = Vec::<u16>::new();
    let mut chunk_palette_lookup = HashMap::<u16, u16>::new();
    let mut dense_indices = Vec::<u16>::with_capacity(materials.len());
    for &material in materials {
        let chunk_idx = if let Some(&existing) = chunk_palette_lookup.get(&material) {
            existing
        } else {
            let next = chunk_palette_materials.len() as u16;
            chunk_palette_materials.push(material);
            chunk_palette_lookup.insert(material, next);
            next
        };
        dense_indices.push(chunk_idx);
    }
    let palette_payloads = chunk_palette_materials
        .into_iter()
        .map(|m| ChunkPayload::Uniform(material_to_block_idx[&m]))
        .collect::<Vec<_>>();
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        bounds,
        palette_payloads,
        dense_indices,
        None,
        block_palette,
    )
    .expect("chunk array must build");
    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::ChunkArray(chunk_array),
        generator_version_hash: 0,
    }
}

fn assert_tree_matches_expected_uniform_map(
    tree: &RegionChunkTree,
    global_bounds: Aabb4i,
    expected: &HashMap<ChunkKey, u16>,
) {
    for key in keys_in_bounds(global_bounds) {
        let actual = match tree.chunk_payload(key) {
            Some(resolved) => match resolved.uniform_block() {
                Some(block) if !block.is_air() => Some(block.block_type as u16),
                _ => None,
            },
            None => None,
        };
        let expected_payload = expected.get(&key).copied();
        assert_eq!(
            actual, expected_payload,
            "mismatch at key {:?}: actual={:?} expected={:?}",
            key, actual, expected_payload
        );
    }

    let sliced = tree.slice_non_empty_core_in_bounds(global_bounds);
    let mut sliced_map = HashMap::<ChunkKey, u16>::new();
    for (key, resolved) in collect_non_empty_chunks_from_core_in_bounds(&sliced, global_bounds) {
        match resolved.uniform_block() {
            Some(block) if !block.is_air() => {
                sliced_map.insert(key, block.block_type as u16);
            }
            _ => {}
        }
    }
    assert_eq!(&sliced_map, expected, "slice_non_empty snapshot mismatch");
}

fn assert_tree_non_overlapping(node: &RegionTreeCore) {
    fn contains_bounds(outer: Aabb4i, inner: Aabb4i) -> bool {
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

    if let RegionNodeKind::Branch(children) = &node.kind {
        for child in children {
            assert!(
                contains_bounds(node.bounds, child.bounds),
                "child {:?}->{:?} escapes parent {:?}->{:?}",
                child.bounds.min,
                child.bounds.max,
                node.bounds.min,
                node.bounds.max
            );
            assert_tree_non_overlapping(child);
        }
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                assert!(
                    !children[i].bounds.intersects(&children[j].bounds),
                    "overlapping siblings: {} {:?}->{:?} and {} {:?}->{:?}",
                    i,
                    children[i].bounds.min,
                    children[i].bounds.max,
                    j,
                    children[j].bounds.min,
                    children[j].bounds.max
                );
            }
        }
    }
}

#[test]
fn set_get_remove_single_chunk_roundtrip() {
    let mut tree = RegionChunkTree::new();
    assert!(!tree.has_chunk(key(0, 0, 0, 0)));

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 12)))));
    assert!(tree.has_chunk(key(0, 0, 0, 0)));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 12))
    );

    assert!(tree.remove_chunk(key(0, 0, 0, 0)));
    assert!(!tree.has_chunk(key(0, 0, 0, 0)));
    assert!(tree.root().is_none());
}

#[test]
fn set_chunk_uniform_zero_on_empty_tree_is_explicit_override() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::empty())));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::AIR)
    );
}

#[test]
fn uniform_merge_and_fragment_behavior_is_stable() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7)))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7)))));

    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 7));

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 9)))));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 7))
    );

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7)))));
    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 7));
}

#[test]
fn non_covering_uniform_children_do_not_fill_parent_gaps() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 11)))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 11)))));

    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 11))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 11))
    );
    assert_eq!(tree.chunk_payload(key(1, 0, 0, 0)), None);

    let bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
    let non_empty = collect_non_empty_chunks_from_core_in_bounds(
        &tree.slice_non_empty_core_in_bounds(bounds),
        bounds,
    );
    let mut keys: Vec<[i32; 4]> = non_empty
        .into_iter()
        .map(|(chunk_key, _)| chunk_key)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, vec![key(0, 0, 0, 0), key(2, 0, 0, 0)]);
}

#[test]
fn adjacent_uniform_children_merge_even_when_branch_partition_is_non_canonical() {
    let mut tree = RegionChunkTree::new();
    let full_bounds = Aabb4i::new([0, 0, 0, 0], [2, 0, 0, 0]);
    let full_uniform = RegionTreeCore {
        bounds: full_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 11)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(full_bounds, &full_uniform),
        Some(full_bounds)
    );

    let center_bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
    let center_empty = RegionTreeCore {
        bounds: center_bounds,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(center_bounds, &center_empty),
        Some(center_bounds)
    );

    let center_uniform = RegionTreeCore {
        bounds: center_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 11)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(center_bounds, &center_uniform),
        Some(center_bounds)
    );

    let root = tree.root().expect("root exists");
    assert_eq!(root.bounds, full_bounds);
    assert!(matches!(root.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 11));
}

#[test]
fn randomized_mutations_preserve_non_overlapping_branches() {
    let mut rng = TestRng::new(0x8A5F_3D71_C2B4_9E10);
    let mut tree = RegionChunkTree::new();
    let global = Aabb4i::new([-6, -3, -6, -3], [6, 3, 6, 3]);

    for _step in 0..5000 {
        let op = rng.next_u32() % 10;
        if op < 7 {
            let chunk = [
                rng.next_inclusive_i32(global.min[0], global.max[0]),
                rng.next_inclusive_i32(global.min[1], global.max[1]),
                rng.next_inclusive_i32(global.min[2], global.max[2]),
                rng.next_inclusive_i32(global.min[3], global.max[3]),
            ];
            let payload = match rng.next_u32() % 4 {
                0 => None,
                1 => Some(ResolvedChunkPayload::empty()),
                _ => Some(ResolvedChunkPayload::uniform(BlockData::simple(0, (rng.next_u32() % 15) + 1))),
            };
            let _ = tree.set_chunk(chunk, payload);
        } else {
            let bounds = random_sub_bounds(&mut rng, global);
            let core = RegionTreeCore {
                bounds,
                kind: if (rng.next_u32() & 1) == 0 {
                    RegionNodeKind::Empty
                } else {
                    RegionNodeKind::Uniform(BlockData::simple(0, (rng.next_u32() % 15) + 1))
                },
                generator_version_hash: 0,
            };
            let _ = tree.splice_non_empty_core_in_bounds(bounds, &core);
        }

        if let Some(root) = tree.root() {
            assert_tree_non_overlapping(root);
        }
    }
}

#[test]
fn splice_non_empty_core_replaces_window_contents() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 2)))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4)))));

    let bounds = Aabb4i::new([1, 0, 0, 0], [1, 0, 0, 0]);
    let patch_core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 9)),
        generator_version_hash: 0,
    };

    let changed_bounds = tree.splice_non_empty_core_in_bounds(bounds, &patch_core);
    assert_eq!(changed_bounds, Some(bounds));
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 2))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

#[test]
fn take_non_empty_core_extracts_and_clears_region() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 2)))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 3)))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4)))));

    let bounds = Aabb4i::new([1, 0, 0, 0], [2, 0, 0, 0]);
    let extracted = tree.take_non_empty_core_in_bounds(bounds);
    let mut extracted_chunks = collect_non_empty_chunks_from_core_in_bounds(&extracted, bounds);
    extracted_chunks.sort_unstable_by_key(|(key, _)| *key);
    assert_eq!(extracted_chunks.len(), 2);
    assert_eq!(extracted_chunks[0].0, key(1, 0, 0, 0));
    assert_eq!(extracted_chunks[0].1.uniform_block(), Some(&BlockData::simple(0, 3)));
    assert_eq!(extracted_chunks[1].0, key(2, 0, 0, 0));
    assert_eq!(extracted_chunks[1].1.uniform_block(), Some(&BlockData::simple(0, 4)));

    assert_eq!(tree.chunk_payload(key(1, 0, 0, 0)), None);
    assert_eq!(tree.chunk_payload(key(2, 0, 0, 0)), None);
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 2))
    );
}

#[test]
fn lazy_drop_outside_bounds_respects_budget() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(-8, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 6)))));
    assert!(tree.set_chunk(key(8, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7)))));

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
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 2)))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4)))));

    let bounds = Aabb4i::new([2, 0, 0, 0], [3, 0, 0, 0]);
    let slice = tree.slice_core_in_bounds(bounds);
    assert_eq!(slice.bounds, bounds);
    let mut chunks = collect_non_empty_chunks_from_core_in_bounds(&slice, bounds);
    chunks.sort_unstable_by_key(|(key, _)| *key);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].0, key(2, 0, 0, 0));
    assert_eq!(chunks[0].1.uniform_block(), Some(&BlockData::simple(0, 4)));
}

#[test]
fn set_chunk_carves_large_uniform_leaf_locally() {
    let mut tree = RegionChunkTree::new();
    let bounds = Aabb4i::new([0, 0, 0, 0], [7, 7, 7, 7]);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 4)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &core),
        Some(bounds)
    );

    assert!(tree.set_chunk(key(3, 4, 2, 5), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 9)))));
    assert_eq!(
        tree.chunk_payload(key(3, 4, 2, 5)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
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
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 6)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(bounds, &core),
        Some(bounds)
    );

    assert!(!tree.set_chunk(key(4, 3, 2, 1), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 6)))));
    let root = tree.root().expect("root exists");
    assert!(matches!(root.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 6));
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

    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 12)))));
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 12))
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
            && matches!(child.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 12)
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
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 2)),
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
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 2)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core),
        None
    );

    let after = tree.root().expect("root");
    assert_eq!(*after, before);
    assert!(matches!(after.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 2));
}

#[test]
fn splice_non_empty_semantic_noop_skips_structural_rewrite() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 3)))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4)))));
    let before = tree.root().expect("root").clone();

    let patch_bounds = Aabb4i::new([0, 0, 0, 0], [1, 0, 0, 0]);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, 3), BlockData::simple(0, 4)];
    let patch_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        patch_bounds,
        vec![ChunkPayload::Uniform(1), ChunkPayload::Uniform(2)],
        vec![0, 1],
        None,
        block_palette,
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 3))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

#[test]
fn overlay_core_applies_explicit_uniform_zero_and_non_zero_leaves() {
    let bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
    let mut tree = RegionChunkTree::new();
    let base = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 7)),
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
                kind: RegionNodeKind::Uniform(BlockData::simple(0, 0)),
                generator_version_hash: 0,
            },
            RegionTreeCore {
                bounds: nine_leaf_bounds,
                kind: RegionNodeKind::Uniform(BlockData::simple(0, 9)),
                generator_version_hash: 0,
            },
        ]),
        generator_version_hash: 0,
    };

    assert!(tree.overlay_core_in_bounds(bounds, &overlay).is_some());
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 7))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::AIR)
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(3, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 7))
    );
}

#[test]
fn splice_non_empty_randomized_window_replacements_match_reference_grid() {
    let global_bounds = Aabb4i::new([0, 0, 0, 0], [5, 2, 5, 1]);
    let mut tree = RegionChunkTree::new();
    let base_core = RegionTreeCore {
        bounds: global_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 7)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(global_bounds, &base_core),
        Some(global_bounds)
    );

    let mut expected = HashMap::<ChunkKey, u16>::new();
    for key in keys_in_bounds(global_bounds) {
        expected.insert(key, 7);
    }

    let mut rng = TestRng::new(0xD04F_5EED_u64);
    for _step in 0..120 {
        let patch_bounds = random_sub_bounds(&mut rng, global_bounds);
        let patch_keys = keys_in_bounds(patch_bounds);
        let mut materials = Vec::<u16>::with_capacity(patch_keys.len());
        for _ in 0..patch_keys.len() {
            let v = rng.next_u32() % 9;
            let material = match v {
                0..=4 => 0,
                5 => 3,
                6 => 7,
                7 => 11,
                _ => 13,
            };
            materials.push(material);
        }

        let patch_core = if materials.iter().all(|m| *m == materials[0]) {
            RegionTreeCore {
                bounds: patch_bounds,
                kind: RegionNodeKind::Uniform(BlockData::simple(0, materials[0] as u32)),
                generator_version_hash: 0,
            }
        } else {
            chunk_array_uniform_palette_core(patch_bounds, &materials)
        };
        let _ = tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core);

        for key in patch_keys {
            expected.remove(&key);
            let idx = linear_index_in_bounds(patch_bounds, key);
            let material = materials[idx];
            if material != 0 {
                expected.insert(key, material);
            }
        }

        assert_tree_matches_expected_uniform_map(&tree, global_bounds, &expected);
    }
}

#[test]
fn overlay_core_randomized_uniform_leaf_layers_match_reference_grid() {
    let global_bounds = Aabb4i::new([0, 0, 0, 0], [4, 1, 4, 1]);
    let mut tree = RegionChunkTree::new();
    let base_core = RegionTreeCore {
        bounds: global_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 5)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(global_bounds, &base_core),
        Some(global_bounds)
    );

    let mut expected = HashMap::<ChunkKey, u16>::new();
    for key in keys_in_bounds(global_bounds) {
        expected.insert(key, 5);
    }

    let mut rng = TestRng::new(0x0B5E_1A7E_u64);
    for _step in 0..60 {
        let leaf_count = 2 + (rng.next_u32() % 4) as usize;
        let mut leaves = Vec::<RegionTreeCore>::new();
        for _ in 0..leaf_count {
            let leaf_bounds = random_sub_bounds(&mut rng, global_bounds);
            let material = match rng.next_u32() % 5 {
                0 => 0,
                1 => 2,
                2 => 6,
                3 => 9,
                _ => 14,
            };
            leaves.push(RegionTreeCore {
                bounds: leaf_bounds,
                kind: RegionNodeKind::Uniform(BlockData::simple(0, material as u32)),
                generator_version_hash: 0,
            });
        }
        let overlay = RegionTreeCore {
            bounds: global_bounds,
            kind: RegionNodeKind::Branch(leaves.clone()),
            generator_version_hash: 0,
        };
        let _ = tree.overlay_core_in_bounds(global_bounds, &overlay);

        for leaf in leaves {
            let RegionNodeKind::Uniform(ref block) = leaf.kind else {
                unreachable!("test builds only uniform leaves");
            };
            for key in keys_in_bounds(leaf.bounds) {
                if block.is_air() {
                    expected.remove(&key);
                } else {
                    expected.insert(key, block.block_type as u16);
                }
            }
        }

        assert_tree_matches_expected_uniform_map(&tree, global_bounds, &expected);
    }
}

#[test]
fn splice_non_empty_randomized_negative_coordinate_windows_match_reference_grid() {
    let global_bounds = Aabb4i::new([-4, -2, -3, -2], [3, 1, 4, 1]);
    let mut tree = RegionChunkTree::new();
    let base_core = RegionTreeCore {
        bounds: global_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 9)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(global_bounds, &base_core),
        Some(global_bounds)
    );

    let mut expected = HashMap::<ChunkKey, u16>::new();
    for key in keys_in_bounds(global_bounds) {
        expected.insert(key, 9);
    }

    let mut rng = TestRng::new(0xA11C_E551_u64);
    for _step in 0..140 {
        let patch_bounds = random_sub_bounds(&mut rng, global_bounds);
        let patch_keys = keys_in_bounds(patch_bounds);
        let mut materials = Vec::<u16>::with_capacity(patch_keys.len());
        for _ in 0..patch_keys.len() {
            let v = rng.next_u32() % 11;
            let material = match v {
                0..=5 => 0,
                6 => 1,
                7 => 4,
                8 => 9,
                9 => 12,
                _ => 15,
            };
            materials.push(material);
        }

        let patch_core = if materials.iter().all(|m| *m == materials[0]) {
            RegionTreeCore {
                bounds: patch_bounds,
                kind: RegionNodeKind::Uniform(BlockData::simple(0, materials[0] as u32)),
                generator_version_hash: 0,
            }
        } else {
            chunk_array_uniform_palette_core(patch_bounds, &materials)
        };
        let _ = tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core);

        for key in patch_keys {
            expected.remove(&key);
            let idx = linear_index_in_bounds(patch_bounds, key);
            let material = materials[idx];
            if material != 0 {
                expected.insert(key, material);
            }
        }

        assert_tree_matches_expected_uniform_map(&tree, global_bounds, &expected);
    }
}

#[test]
fn replaying_sliced_non_empty_windows_over_synced_tree_is_semantically_stable() {
    let global_bounds = Aabb4i::new([-5, -1, -5, -1], [6, 2, 6, 2]);
    let mut server_tree = RegionChunkTree::new();
    let base_core = RegionTreeCore {
        bounds: global_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 7)),
        generator_version_hash: 0,
    };
    let _ = server_tree.splice_non_empty_core_in_bounds(global_bounds, &base_core);

    let mut expected = HashMap::<ChunkKey, u16>::new();
    for key in keys_in_bounds(global_bounds) {
        expected.insert(key, 7);
    }

    let mut rng = TestRng::new(0xC11E_177E_u64);
    for _step in 0..120 {
        let patch_bounds = random_sub_bounds(&mut rng, global_bounds);
        let patch_keys = keys_in_bounds(patch_bounds);
        let mut materials = Vec::<u16>::with_capacity(patch_keys.len());
        for _ in 0..patch_keys.len() {
            let v = rng.next_u32() % 8;
            let material = match v {
                0..=3 => 0,
                4 => 2,
                5 => 7,
                6 => 10,
                _ => 13,
            };
            materials.push(material);
        }
        let patch_core = if materials.iter().all(|m| *m == materials[0]) {
            RegionTreeCore {
                bounds: patch_bounds,
                kind: RegionNodeKind::Uniform(BlockData::simple(0, materials[0] as u32)),
                generator_version_hash: 0,
            }
        } else {
            chunk_array_uniform_palette_core(patch_bounds, &materials)
        };
        let _ = server_tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core);

        for key in patch_keys {
            expected.remove(&key);
            let idx = linear_index_in_bounds(patch_bounds, key);
            let material = materials[idx];
            if material != 0 {
                expected.insert(key, material);
            }
        }
    }

    let mut client_tree = RegionChunkTree::new();
    let full_patch = server_tree.slice_non_empty_core_in_bounds(global_bounds);
    let _ = client_tree.splice_non_empty_core_in_bounds(global_bounds, &full_patch);
    assert_tree_matches_expected_uniform_map(&client_tree, global_bounds, &expected);

    for _step in 0..220 {
        let window = random_sub_bounds(&mut rng, global_bounds);
        let patch = server_tree.slice_non_empty_core_in_bounds(window);
        let _ = client_tree.splice_non_empty_core_in_bounds(window, &patch);
        assert_tree_matches_expected_uniform_map(&client_tree, global_bounds, &expected);
    }
}

/// Regression test: placing two Dense16 blocks in adjacent chunks on a Uniform
/// platform must preserve both edits after tree normalization (which includes
/// ChunkArray sibling consolidation).  A previous bug caused
/// `consolidate_chunk_array_children` to discard Dense16 payloads whose palette
/// index matched the ChunkArray's `default_chunk_idx`, silently destroying every
/// voxel edit.
#[test]
fn consolidation_preserves_dense_chunk_edits_on_uniform_platform() {
    let mut tree = RegionChunkTree::new();
    let platform_bounds = Aabb4i::new([0, 0, 0, 0], [7, 0, 7, 0]);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 4)),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Place a dense block at chunk [3,0,3,0].
    let mut blocks_a = vec![BlockData::AIR; 4096];
    blocks_a[0] = BlockData::simple(0, 9);
    let resolved_a = ResolvedChunkPayload::from_dense_blocks(&blocks_a).expect("dense blocks");
    assert!(tree.set_chunk(key(3, 0, 3, 0), Some(resolved_a.clone())));
    {
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("chunk must exist");
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }

    // Place a dense block at adjacent chunk [4,0,3,0].
    let mut blocks_b = vec![BlockData::AIR; 4096];
    blocks_b[0] = BlockData::simple(0, 10);
    let resolved_b = ResolvedChunkPayload::from_dense_blocks(&blocks_b).expect("dense blocks");
    assert!(tree.set_chunk(key(4, 0, 3, 0), Some(resolved_b.clone())));

    // Both edits must survive consolidation.
    {
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("first edit disappeared after second edit");
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }
    {
        let retrieved_b = tree.chunk_payload(key(4, 0, 3, 0)).expect("second edit not present");
        assert_eq!(retrieved_b.block_at(0), BlockData::simple(0, 10));
        assert_eq!(retrieved_b.block_at(1), BlockData::AIR);
    }
    // Platform Uniform elsewhere must be intact.
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

/// Same scenario via splice (the multiplayer patch path) instead of set_chunk.
#[test]
fn splice_consolidation_preserves_dense_chunk_edits_on_uniform_platform() {
    let mut tree = RegionChunkTree::new();
    let platform_bounds = Aabb4i::new([0, 0, 0, 0], [7, 0, 7, 0]);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 4)),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Splice edit A at [3,0,3,0].
    // block_palette: [AIR, block_type=9]; Dense16 indices reference this palette.
    let block_palette_a = vec![BlockData::AIR, BlockData::simple(0, 9)];
    let mut dense_indices_a = vec![0u16; 4096];
    dense_indices_a[0] = 1; // block_palette[1] = block_type=9
    let payload_a = ChunkPayload::Dense16 {
        materials: dense_indices_a,
    };
    let a_bounds = Aabb4i::new([3, 0, 3, 0], [3, 0, 3, 0]);
    let a_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        a_bounds,
        vec![payload_a.clone()],
        vec![0u16],
        Some(0),
        block_palette_a,
    )
    .unwrap();
    let a_core = RegionTreeCore {
        bounds: a_bounds,
        kind: RegionNodeKind::ChunkArray(a_ca),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(a_bounds, &a_core);
    {
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("chunk A must exist after splice");
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }

    // Splice edit B at [4,0,3,0].
    let block_palette_b = vec![BlockData::AIR, BlockData::simple(0, 10)];
    let mut dense_indices_b = vec![0u16; 4096];
    dense_indices_b[0] = 1; // block_palette[1] = block_type=10
    let payload_b = ChunkPayload::Dense16 {
        materials: dense_indices_b,
    };
    let b_bounds = Aabb4i::new([4, 0, 3, 0], [4, 0, 3, 0]);
    let b_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        b_bounds,
        vec![payload_b.clone()],
        vec![0u16],
        Some(0),
        block_palette_b,
    )
    .unwrap();
    let b_core = RegionTreeCore {
        bounds: b_bounds,
        kind: RegionNodeKind::ChunkArray(b_ca),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(b_bounds, &b_core);

    // Both must survive.
    {
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("first splice disappeared after second splice");
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }
    {
        let retrieved_b = tree.chunk_payload(key(4, 0, 3, 0)).expect("second splice not present");
        assert_eq!(retrieved_b.block_at(0), BlockData::simple(0, 10));
        assert_eq!(retrieved_b.block_at(1), BlockData::AIR);
    }
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}
