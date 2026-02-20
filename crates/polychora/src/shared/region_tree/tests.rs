use super::*;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
use crate::shared::spatial::Aabb4i;
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

    let mut palette_materials = Vec::<u16>::new();
    let mut palette_lookup = HashMap::<u16, u16>::new();
    let mut dense_indices = Vec::<u16>::with_capacity(materials.len());
    for material in materials {
        let palette_idx = if let Some(existing) = palette_lookup.get(material) {
            *existing
        } else {
            let next = palette_materials.len() as u16;
            palette_materials.push(*material);
            palette_lookup.insert(*material, next);
            next
        };
        dense_indices.push(palette_idx);
    }
    let palette_payloads = palette_materials
        .into_iter()
        .map(ChunkPayload::Uniform)
        .collect::<Vec<_>>();
    let chunk_array =
        ChunkArrayData::from_dense_indices(bounds, palette_payloads, dense_indices, None)
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
            Some(ChunkPayload::Uniform(material)) if material != 0 => Some(material),
            Some(ChunkPayload::Uniform(0) | ChunkPayload::Empty) | None => None,
            Some(other) => panic!(
                "unexpected payload in uniform-map assertion at {:?}: {:?}",
                key, other
            ),
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
    for (key, payload) in collect_non_empty_chunks_from_core_in_bounds(&sliced, global_bounds) {
        match payload {
            ChunkPayload::Uniform(material) if material != 0 => {
                sliced_map.insert(key, material);
            }
            ChunkPayload::Uniform(0) | ChunkPayload::Empty => {}
            other => panic!("unexpected payload in uniform-map assertion: {:?}", other),
        }
    }
    assert_eq!(&sliced_map, expected, "slice_non_empty snapshot mismatch");
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
fn non_covering_uniform_children_do_not_fill_parent_gaps() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ChunkPayload::Uniform(11))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ChunkPayload::Uniform(11))));

    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)),
        Some(ChunkPayload::Uniform(11))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)),
        Some(ChunkPayload::Uniform(11))
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

#[test]
fn splice_non_empty_randomized_window_replacements_match_reference_grid() {
    let global_bounds = Aabb4i::new([0, 0, 0, 0], [5, 2, 5, 1]);
    let mut tree = RegionChunkTree::new();
    let base_core = RegionTreeCore {
        bounds: global_bounds,
        kind: RegionNodeKind::Uniform(7),
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
                kind: RegionNodeKind::Uniform(materials[0]),
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
        kind: RegionNodeKind::Uniform(5),
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
                kind: RegionNodeKind::Uniform(material),
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
            let RegionNodeKind::Uniform(material) = leaf.kind else {
                unreachable!("test builds only uniform leaves");
            };
            for key in keys_in_bounds(leaf.bounds) {
                if material == 0 {
                    expected.remove(&key);
                } else {
                    expected.insert(key, material);
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
        kind: RegionNodeKind::Uniform(9),
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
                kind: RegionNodeKind::Uniform(materials[0]),
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
        kind: RegionNodeKind::Uniform(7),
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
                kind: RegionNodeKind::Uniform(materials[0]),
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
