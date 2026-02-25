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

fn single_cell_chunk_array_core_with_scale(
    key: ChunkKey,
    material: u16,
    scale_exp: i8,
) -> RegionTreeCore {
    let bounds = Aabb4i::new(key, key);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, material as u32)];
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette_and_scale(
        bounds,
        vec![ChunkPayload::Uniform(1)],
        vec![0],
        None,
        block_palette,
        scale_exp,
    )
    .expect("single-cell chunk array");
    RegionTreeCore {
        bounds,
        kind: RegionNodeKind::ChunkArray(chunk_array),
        generator_version_hash: 0,
    }
}

fn collect_chunk_array_scale_exps(core: &RegionTreeCore, out: &mut Vec<i8>) {
    match &core.kind {
        RegionNodeKind::ChunkArray(chunk_array) => out.push(chunk_array.scale_exp),
        RegionNodeKind::Branch(children) => {
            for child in children {
                collect_chunk_array_scale_exps(child, out);
            }
        }
        _ => {}
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
fn mixed_scale_non_overlapping_splice_is_accepted() {
    let mut tree = RegionChunkTree::new();
    let coarse = single_cell_chunk_array_core_with_scale(key(0, 0, 0, 0), 11, 0);
    let fine_non_overlapping = single_cell_chunk_array_core_with_scale(key(2, 0, 0, 0), 22, -1);

    assert_eq!(
        tree.splice_core_in_bounds(coarse.bounds, &coarse),
        Some(coarse.bounds)
    );
    assert_eq!(
        tree.splice_core_in_bounds(fine_non_overlapping.bounds, &fine_non_overlapping),
        Some(fine_non_overlapping.bounds)
    );

    let root = tree.root().expect("root");
    validate_region_core_world_space_non_overlapping(root).expect("tree must be world-space valid");
}

#[test]
fn mixed_scale_world_overlapping_splice_is_detected() {
    let mut tree = RegionChunkTree::new();
    let coarse = single_cell_chunk_array_core_with_scale(key(0, 0, 0, 0), 11, 0);
    let fine_overlapping = single_cell_chunk_array_core_with_scale(key(1, 0, 0, 0), 22, -1);

    assert_eq!(
        tree.splice_core_in_bounds(coarse.bounds, &coarse),
        Some(coarse.bounds)
    );
    // Overlapping splice proceeds (no rollback) but validation detects the overlap.
    assert_eq!(
        tree.splice_core_in_bounds(fine_overlapping.bounds, &fine_overlapping),
        Some(fine_overlapping.bounds)
    );

    let root = tree.root().expect("root");
    assert!(validate_region_core_world_space_non_overlapping(root).is_err());
}

#[test]
fn chunk_array_consolidation_does_not_cross_scale_boundaries() {
    let mut tree = RegionChunkTree::new();
    let coarse = single_cell_chunk_array_core_with_scale(key(0, 0, 0, 0), 11, 0);
    let fine = single_cell_chunk_array_core_with_scale(key(2, 0, 0, 0), 22, -1);

    assert_eq!(
        tree.splice_core_in_bounds(coarse.bounds, &coarse),
        Some(coarse.bounds)
    );
    assert_eq!(tree.splice_core_in_bounds(fine.bounds, &fine), Some(fine.bounds));

    let root = tree.root().expect("root");
    let mut scale_exps = Vec::new();
    collect_chunk_array_scale_exps(root, &mut scale_exps);
    scale_exps.sort_unstable();
    assert_eq!(scale_exps, vec![-1, 0]);
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

/// Simulates placing 4+ dense blocks in a cluster on a Uniform platform.
/// After consolidation, edit another adjacent chunk and verify all data survives.
/// This exercises the path: set_chunk → normalize → consolidate → set_chunk → carve.
#[test]
fn cluster_edits_on_uniform_platform_survive_consolidation_and_recarve() {
    let chunk_volume = crate::shared::voxel::CHUNK_VOLUME;
    let mut tree = RegionChunkTree::new();
    let platform_bounds = Aabb4i::new([0, 0, 0, 0], [7, 0, 7, 0]);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 4)),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Place dense blocks in a cluster of 4 adjacent chunks (x=2..5, y=0, z=3, w=0).
    let mut expected_blocks: HashMap<ChunkKey, Vec<BlockData>> = HashMap::new();
    for x in 2..=5i32 {
        let mut blocks = vec![BlockData::simple(0, 4); chunk_volume]; // Start with platform block.
        blocks[0] = BlockData::simple(0, (10 + x) as u32);
        blocks[1] = BlockData::simple(0, (20 + x) as u32);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        assert!(
            tree.set_chunk(key(x, 0, 3, 0), Some(resolved)),
            "set_chunk [{x},0,3,0] should succeed"
        );
        expected_blocks.insert(key(x, 0, 3, 0), blocks);

        // Verify ALL previously placed chunks still exist.
        for prev_x in 2..=x {
            let payload = tree
                .chunk_payload(key(prev_x, 0, 3, 0))
                .unwrap_or_else(|| {
                    panic!(
                        "chunk [{prev_x},0,3,0] disappeared after editing [{x},0,3,0]"
                    )
                });
            let actual = payload.dense_blocks();
            let expected = &expected_blocks[&key(prev_x, 0, 3, 0)];
            assert_eq!(
                actual[0], expected[0],
                "chunk [{prev_x},0,3,0] block[0] wrong after editing [{x},0,3,0]"
            );
            assert_eq!(
                actual[1], expected[1],
                "chunk [{prev_x},0,3,0] block[1] wrong after editing [{x},0,3,0]"
            );
        }
    }

    // Verify platform chunks are still uniform.
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
    assert_eq!(
        tree.chunk_payload(key(7, 0, 7, 0)).unwrap().uniform_block(),
        Some(&BlockData::simple(0, 4))
    );

    // Now edit chunk [3,0,3,0] again (already edited above) — triggers re-carve of
    // consolidated ChunkArray.
    {
        let mut blocks = tree
            .chunk_payload(key(3, 0, 3, 0))
            .expect("chunk should exist")
            .dense_blocks();
        blocks[2] = BlockData::simple(0, 99);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        tree.set_chunk(key(3, 0, 3, 0), Some(resolved));
        expected_blocks.get_mut(&key(3, 0, 3, 0)).unwrap()[2] = BlockData::simple(0, 99);
    }

    // Verify ALL cluster chunks survived the re-edit.
    for x in 2..=5i32 {
        let payload = tree
            .chunk_payload(key(x, 0, 3, 0))
            .unwrap_or_else(|| {
                panic!(
                    "chunk [{x},0,3,0] disappeared after re-editing [3,0,3,0]"
                )
            });
        let actual = payload.dense_blocks();
        let expected = &expected_blocks[&key(x, 0, 3, 0)];
        assert_eq!(
            actual[0], expected[0],
            "chunk [{x},0,3,0] block[0] wrong after re-edit"
        );
        assert_eq!(
            actual[1], expected[1],
            "chunk [{x},0,3,0] block[1] wrong after re-edit"
        );
    }

    // Add an adjacent chunk at [6,0,3,0] — outside the original cluster.
    {
        let mut blocks = vec![BlockData::simple(0, 4); chunk_volume];
        blocks[0] = BlockData::simple(0, 77);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        tree.set_chunk(key(6, 0, 3, 0), Some(resolved));
        expected_blocks.insert(key(6, 0, 3, 0), blocks);
    }

    // Final verification of ALL chunks.
    for (&k, expected) in &expected_blocks {
        let payload = tree
            .chunk_payload(k)
            .unwrap_or_else(|| panic!("chunk {k:?} missing in final check"));
        let actual = payload.dense_blocks();
        assert_eq!(
            actual[0], expected[0],
            "chunk {k:?} block[0] final check"
        );
        assert_eq!(
            actual[1], expected[1],
            "chunk {k:?} block[1] final check"
        );
    }

    // Verify tree structure is valid (no overlapping children).
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Simulates the server-side flow: save data containing a ChunkArray is loaded
/// via `splice_core_in_bounds`, then individual chunks are edited via `set_chunk`.
/// Verifies that editing one chunk within a ChunkArray does not lose sibling chunks.
#[test]
fn set_chunk_in_spliced_chunk_array_preserves_siblings() {
    let ca_bounds = Aabb4i::new([0, 0, 0, 0], [7, 0, 0, 0]);
    // 8 chunks along x, each with a unique uniform material (1..=8).
    let materials: Vec<u16> = (1..=8u16).collect();
    let core = chunk_array_uniform_palette_core(ca_bounds, &materials);

    // Splice the ChunkArray into an empty tree (simulating save load).
    let mut tree = RegionChunkTree::new();
    tree.splice_non_empty_core_in_bounds(ca_bounds, &core);

    // Verify all 8 chunks are present.
    for x in 0..8i32 {
        let payload = tree
            .chunk_payload(key(x, 0, 0, 0))
            .unwrap_or_else(|| panic!("chunk [{x},0,0,0] missing before any edit"));
        assert_eq!(
            payload.uniform_block(),
            Some(&BlockData::simple(0, (x + 1) as u32)),
            "chunk [{x},0,0,0] has wrong material before edit"
        );
    }

    // Edit chunk x=3 (in the middle of the array).
    tree.set_chunk(
        key(3, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 99))),
    );

    // Verify the edited chunk.
    assert_eq!(
        tree.chunk_payload(key(3, 0, 0, 0))
            .expect("edited chunk missing")
            .uniform_block(),
        Some(&BlockData::simple(0, 99))
    );

    // Verify ALL other chunks survived the edit.
    for x in 0..8i32 {
        if x == 3 {
            continue;
        }
        let payload = tree
            .chunk_payload(key(x, 0, 0, 0))
            .unwrap_or_else(|| panic!("chunk [{x},0,0,0] disappeared after editing chunk [3,0,0,0]"));
        assert_eq!(
            payload.uniform_block(),
            Some(&BlockData::simple(0, (x + 1) as u32)),
            "chunk [{x},0,0,0] has wrong material after editing chunk [3,0,0,0]"
        );
    }

    // Edit chunk x=0 (start of array).
    tree.set_chunk(
        key(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 100))),
    );

    // Edit chunk x=7 (end of array).
    tree.set_chunk(
        key(7, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 101))),
    );

    // Verify all chunks after multiple edits.
    let expected = [100, 2, 3, 99, 5, 6, 7, 101];
    for (x, &expected_mat) in expected.iter().enumerate() {
        let payload = tree
            .chunk_payload(key(x as i32, 0, 0, 0))
            .unwrap_or_else(|| panic!("chunk [{x},0,0,0] missing after cluster edits"));
        assert_eq!(
            payload.uniform_block(),
            Some(&BlockData::simple(0, expected_mat)),
            "chunk [{x},0,0,0] has wrong material after cluster edits"
        );
    }
}

/// Same scenario as above but with non-uniform (dense) chunk payloads
/// instead of uniform ones, since the bug might only manifest with
/// complex payloads.
#[test]
fn set_chunk_in_spliced_chunk_array_preserves_dense_siblings() {
    let ca_bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
    let chunk_volume = crate::shared::voxel::CHUNK_VOLUME;

    // Build 4 non-uniform chunks, each with a unique pattern.
    let mut block_palette = vec![BlockData::AIR];
    let mut chunk_payloads = Vec::new();
    let mut dense_indices = Vec::new();
    for x in 0..4u32 {
        // Each chunk has block_type = (x+1) at voxel 0, rest is AIR.
        let block = BlockData::simple(0, x + 1);
        let block_idx = block_palette.len() as u16;
        block_palette.push(block.clone());

        let mut materials = vec![0u16; chunk_volume];
        materials[0] = block_idx;
        let payload = ChunkPayload::Dense16 {
            materials: materials.clone(),
        };
        let chunk_idx = chunk_payloads.len() as u16;
        chunk_payloads.push(payload);
        dense_indices.push(chunk_idx);
    }

    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        ca_bounds,
        chunk_payloads,
        dense_indices,
        None,
        block_palette,
    )
    .expect("chunk array must build");

    let core = RegionTreeCore {
        bounds: ca_bounds,
        kind: RegionNodeKind::ChunkArray(chunk_array),
        generator_version_hash: 0,
    };

    let mut tree = RegionChunkTree::new();
    tree.splice_non_empty_core_in_bounds(ca_bounds, &core);

    // Verify all 4 chunks are present with correct first voxel.
    for x in 0..4i32 {
        let payload = tree
            .chunk_payload(key(x, 0, 0, 0))
            .unwrap_or_else(|| panic!("chunk [{x},0,0,0] missing before edit"));
        let blocks = payload.dense_blocks();
        assert_eq!(
            blocks[0],
            BlockData::simple(0, (x + 1) as u32),
            "chunk [{x},0,0,0] block[0] before edit"
        );
    }

    // Edit chunk x=1 (middle of array).
    let mut new_blocks = vec![BlockData::AIR; chunk_volume];
    new_blocks[0] = BlockData::simple(0, 50);
    new_blocks[1] = BlockData::simple(0, 51);
    let new_payload = ResolvedChunkPayload::from_dense_blocks(&new_blocks).expect("from dense");
    tree.set_chunk(key(1, 0, 0, 0), Some(new_payload));

    // Verify edited chunk.
    {
        let payload = tree
            .chunk_payload(key(1, 0, 0, 0))
            .expect("edited chunk missing");
        let blocks = payload.dense_blocks();
        assert_eq!(blocks[0], BlockData::simple(0, 50));
        assert_eq!(blocks[1], BlockData::simple(0, 51));
    }

    // Verify all other chunks survived.
    for x in [0, 2, 3] {
        let payload = tree
            .chunk_payload(key(x, 0, 0, 0))
            .unwrap_or_else(|| panic!("chunk [{x},0,0,0] disappeared after edit"));
        let blocks = payload.dense_blocks();
        assert_eq!(
            blocks[0],
            BlockData::simple(0, (x + 1) as u32),
            "chunk [{x},0,0,0] block[0] after edit"
        );
    }

    // Edit chunk x=2.
    tree.set_chunk(
        key(2, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 60))),
    );

    // Verify ALL chunks.
    {
        let p0 = tree.chunk_payload(key(0, 0, 0, 0)).expect("x=0 missing after second edit");
        assert_eq!(p0.dense_blocks()[0], BlockData::simple(0, 1));
    }
    {
        let p1 = tree.chunk_payload(key(1, 0, 0, 0)).expect("x=1 missing after second edit");
        assert_eq!(p1.dense_blocks()[0], BlockData::simple(0, 50));
    }
    {
        let p2 = tree.chunk_payload(key(2, 0, 0, 0)).expect("x=2 missing after second edit");
        assert_eq!(
            p2.uniform_block(),
            Some(&BlockData::simple(0, 60))
        );
    }
    {
        let p3 = tree.chunk_payload(key(3, 0, 0, 0)).expect("x=3 missing after second edit");
        assert_eq!(p3.dense_blocks()[0], BlockData::simple(0, 4));
    }
}

/// Simulates the full server lifecycle: bulk load from save → edits →
/// persist_dirty_overrides query → simulate reload → verify all data.
/// This exercises dense chunk array consolidation + dirty set interaction.
#[test]
fn server_lifecycle_bulk_load_edit_save_reload_preserves_all_chunks() {
    use std::collections::HashSet;
    let chunk_volume = crate::shared::voxel::CHUNK_VOLUME;

    // Build 6 dense chunks in a 3×2 cluster at y=0..1, z=0, w=0.
    // Each has a unique block at voxel[0] and voxel[1].
    let positions: Vec<[i32; 4]> = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [2, 1, 0, 0],
    ];

    // Build a "save tree" containing all 6 chunks as a single ChunkArray.
    let mut save_block_palette = vec![BlockData::AIR];
    let mut save_chunk_palette = Vec::new();
    let mut save_dense_indices = Vec::new();
    let mut original_blocks: HashMap<[i32; 4], Vec<BlockData>> = HashMap::new();

    for (i, pos) in positions.iter().enumerate() {
        let block_a = BlockData::simple(0, (i as u32 + 1) * 10);
        let block_b = BlockData::simple(0, (i as u32 + 1) * 10 + 1);
        let idx_a = save_block_palette.len() as u16;
        save_block_palette.push(block_a.clone());
        let idx_b = save_block_palette.len() as u16;
        save_block_palette.push(block_b.clone());

        let mut materials = vec![0u16; chunk_volume];
        materials[0] = idx_a;
        materials[1] = idx_b;
        let payload = ChunkPayload::Dense16 {
            materials: materials.clone(),
        };
        let chunk_idx = save_chunk_palette.len() as u16;
        save_chunk_palette.push(payload);
        save_dense_indices.push(chunk_idx);

        let mut blocks = vec![BlockData::AIR; chunk_volume];
        blocks[0] = block_a;
        blocks[1] = block_b;
        original_blocks.insert(*pos, blocks);
    }

    let save_bounds = Aabb4i::new([0, 0, 0, 0], [2, 1, 0, 0]);
    let save_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        save_bounds,
        save_chunk_palette,
        save_dense_indices,
        None,
        save_block_palette,
    )
    .expect("save chunk array must build");
    let save_core = RegionTreeCore {
        bounds: save_bounds,
        kind: RegionNodeKind::ChunkArray(save_chunk_array),
        generator_version_hash: 0,
    };

    // Phase 1: Bulk load from "save" into override tree.
    let mut tree = RegionChunkTree::new();
    tree.splice_non_empty_core_in_bounds(save_bounds, &save_core);

    // Verify all 6 chunks loaded correctly.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after bulk load"));
        let blocks = payload.dense_blocks();
        let expected = &original_blocks[pos];
        assert_eq!(blocks[0], expected[0], "chunk {pos:?} block[0] after load");
        assert_eq!(blocks[1], expected[1], "chunk {pos:?} block[1] after load");
    }

    // Phase 2: Edit 3 chunks (simulating apply_voxel_edit).
    let mut dirty_save_chunks: HashSet<[i32; 4]> = HashSet::new();
    let edited_positions = [[0, 0, 0, 0], [1, 1, 0, 0], [2, 0, 0, 0]];
    for pos in &edited_positions {
        let mut blocks = tree
            .chunk_payload(*pos)
            .expect("chunk should exist for edit")
            .dense_blocks();
        blocks[0] = BlockData::simple(0, 999);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(*pos, Some(resolved));
        dirty_save_chunks.insert(*pos);
        original_blocks.get_mut(pos).unwrap()[0] = BlockData::simple(0, 999);
    }

    // Verify all 6 chunks survive after edits.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after edits"));
        let blocks = payload.dense_blocks();
        let expected = &original_blocks[pos];
        assert_eq!(
            blocks[0], expected[0],
            "chunk {pos:?} block[0] after edits"
        );
        assert_eq!(
            blocks[1], expected[1],
            "chunk {pos:?} block[1] after edits"
        );
    }

    // Phase 3: Simulate persist_dirty_overrides — query only dirty chunks.
    let mut dirty_payloads: Vec<([i32; 4], Option<ResolvedChunkPayload>)> = Vec::new();
    for pos in &dirty_save_chunks {
        let resolved = tree.chunk_payload(*pos);
        dirty_payloads.push((*pos, resolved));
    }

    // All dirty payloads should be present and non-empty.
    for (pos, payload) in &dirty_payloads {
        assert!(
            payload.is_some(),
            "dirty chunk {pos:?} returned None from override tree"
        );
        let blocks = payload.as_ref().unwrap().dense_blocks();
        assert_eq!(
            blocks[0],
            BlockData::simple(0, 999),
            "dirty chunk {pos:?} has wrong edit data"
        );
    }

    // Phase 4: Simulate "save + reload" by building a new tree from:
    // - Non-dirty chunks from the original save
    // - Dirty chunks from the current override tree
    let mut reload_tree = RegionChunkTree::new();

    // Load the non-dirty chunks from "old save".
    for pos in &positions {
        if dirty_save_chunks.contains(pos) {
            continue;
        }
        // Extract from the original save_core (simulating reading from disk).
        let chunk_bounds = Aabb4i::new(*pos, *pos);
        let sliced = slice_region_core_in_bounds(&save_core, chunk_bounds);
        reload_tree.splice_non_empty_core_in_bounds(chunk_bounds, &sliced);
    }

    // Load the dirty chunks from the "new save" (which would have their updated data).
    for (pos, payload) in &dirty_payloads {
        if let Some(resolved) = payload {
            reload_tree.set_chunk(*pos, Some(resolved.clone()));
        }
    }

    // Phase 5: Verify all 6 chunks in the reloaded tree.
    for pos in &positions {
        let payload = reload_tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after reload"));
        let blocks = payload.dense_blocks();
        let expected = &original_blocks[pos];
        assert_eq!(
            blocks[0], expected[0],
            "chunk {pos:?} block[0] after reload"
        );
        assert_eq!(
            blocks[1], expected[1],
            "chunk {pos:?} block[1] after reload"
        );
    }

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
    if let Some(root) = reload_tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Tests the scenario where chunks are loaded INDIVIDUALLY (one at a time)
/// from a save ChunkArray, triggering progressive consolidation, then edited.
/// This is the exact server flow when ensure_persisted_bounds_loaded loads
/// one chunk at a time as the player edits different positions.
#[test]
fn individual_chunk_loads_with_progressive_consolidation_preserve_data() {
    let chunk_volume = crate::shared::voxel::CHUNK_VOLUME;

    // Build a save ChunkArray with 4 dense chunks.
    let positions: Vec<[i32; 4]> = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
    ];

    let mut save_block_palette = vec![BlockData::AIR];
    let mut save_chunk_palette = Vec::new();
    let mut save_dense_indices = Vec::new();
    let mut expected_blocks: HashMap<[i32; 4], Vec<BlockData>> = HashMap::new();

    for (i, pos) in positions.iter().enumerate() {
        let block = BlockData::simple(0, (i as u32 + 1) * 100);
        let block_idx = save_block_palette.len() as u16;
        save_block_palette.push(block.clone());

        let mut materials = vec![0u16; chunk_volume];
        materials[0] = block_idx;
        materials[i + 1] = block_idx; // unique position per chunk
        let payload = ChunkPayload::Dense16 { materials };
        let chunk_idx = save_chunk_palette.len() as u16;
        save_chunk_palette.push(payload);
        save_dense_indices.push(chunk_idx);

        let mut blocks = vec![BlockData::AIR; chunk_volume];
        blocks[0] = block.clone();
        blocks[i + 1] = block;
        expected_blocks.insert(*pos, blocks);
    }

    let save_bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
    let save_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        save_bounds,
        save_chunk_palette,
        save_dense_indices,
        None,
        save_block_palette,
    )
    .expect("save chunk array must build");
    let save_core = RegionTreeCore {
        bounds: save_bounds,
        kind: RegionNodeKind::ChunkArray(save_chunk_array),
        generator_version_hash: 0,
    };

    // Load chunks ONE AT A TIME (simulating individual ensure_persisted_bounds_loaded calls).
    let mut tree = RegionChunkTree::new();
    for (load_idx, pos) in positions.iter().enumerate() {
        let chunk_bounds = Aabb4i::new(*pos, *pos);
        let sliced = slice_region_core_in_bounds(&save_core, chunk_bounds);
        tree.splice_non_empty_core_in_bounds(chunk_bounds, &sliced);

        // After each load, verify ALL previously loaded chunks are still correct.
        for prev_pos in &positions[..=load_idx] {
            let payload = tree.chunk_payload(*prev_pos).unwrap_or_else(|| {
                panic!(
                    "chunk {prev_pos:?} disappeared after loading {pos:?} (load #{load_idx})"
                )
            });
            let blocks = payload.dense_blocks();
            let expected = &expected_blocks[prev_pos];
            assert_eq!(
                blocks[0], expected[0],
                "chunk {prev_pos:?} block[0] wrong after loading {pos:?}"
            );
        }
    }

    // Now edit chunk [1,0,0,0].
    {
        let mut blocks = tree
            .chunk_payload(key(1, 0, 0, 0))
            .expect("chunk should exist")
            .dense_blocks();
        blocks[0] = BlockData::simple(0, 777);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected_blocks.get_mut(&[1, 0, 0, 0]).unwrap()[0] = BlockData::simple(0, 777);
    }

    // Verify ALL chunks after edit.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after editing [1,0,0,0]"));
        let blocks = payload.dense_blocks();
        let expected = &expected_blocks[pos];
        assert_eq!(
            blocks[0], expected[0],
            "chunk {pos:?} block[0] after editing [1,0,0,0]"
        );
    }

    // Edit chunk [3,0,0,0].
    {
        let mut blocks = tree
            .chunk_payload(key(3, 0, 0, 0))
            .expect("chunk should exist")
            .dense_blocks();
        blocks[0] = BlockData::simple(0, 888);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(3, 0, 0, 0), Some(resolved));
        expected_blocks.get_mut(&[3, 0, 0, 0]).unwrap()[0] = BlockData::simple(0, 888);
    }

    // Verify ALL chunks after second edit.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after editing [3,0,0,0]"));
        let blocks = payload.dense_blocks();
        let expected = &expected_blocks[pos];
        assert_eq!(
            blocks[0], expected[0],
            "chunk {pos:?} block[0] after second edit"
        );
    }

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Tests a critical edge case: after consolidation creates a ChunkArray with
/// default_chunk_idx=Some(0), subsequent carving and re-consolidation preserves
/// all data correctly through multiple cycles.
#[test]
fn repeated_carve_consolidation_cycles_preserve_dense_data() {
    let chunk_volume = crate::shared::voxel::CHUNK_VOLUME;

    let mut tree = RegionChunkTree::new();
    let mut expected: HashMap<[i32; 4], Vec<BlockData>> = HashMap::new();

    // Insert 4 chunks one at a time (each triggers potential consolidation).
    for i in 0..4i32 {
        let pos = [i, 0, 0, 0];
        let mut blocks = vec![BlockData::AIR; chunk_volume];
        blocks[0] = BlockData::simple(0, (i + 1) as u32);
        blocks[1] = BlockData::simple(0, (i + 1) as u32 * 10);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(pos, Some(resolved));
        expected.insert(pos, blocks);
    }

    // Cycle 1: edit chunk [1,0,0,0] → carves consolidated array
    {
        let mut blocks = tree.chunk_payload(key(1, 0, 0, 0)).unwrap().dense_blocks();
        blocks[2] = BlockData::simple(0, 42);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected.get_mut(&[1, 0, 0, 0]).unwrap()[2] = BlockData::simple(0, 42);
    }

    // Verify after cycle 1.
    for (pos, exp) in &expected {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after cycle 1"));
        let blocks = payload.dense_blocks();
        assert_eq!(blocks[0], exp[0], "cycle 1: chunk {pos:?} block[0]");
        assert_eq!(blocks[1], exp[1], "cycle 1: chunk {pos:?} block[1]");
        assert_eq!(blocks[2], exp[2], "cycle 1: chunk {pos:?} block[2]");
    }

    // Cycle 2: edit chunk [3,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(3, 0, 0, 0)).unwrap().dense_blocks();
        blocks[3] = BlockData::simple(0, 77);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(3, 0, 0, 0), Some(resolved));
        expected.get_mut(&[3, 0, 0, 0]).unwrap()[3] = BlockData::simple(0, 77);
    }

    // Cycle 3: edit chunk [0,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(0, 0, 0, 0)).unwrap().dense_blocks();
        blocks[4] = BlockData::simple(0, 55);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(0, 0, 0, 0), Some(resolved));
        expected.get_mut(&[0, 0, 0, 0]).unwrap()[4] = BlockData::simple(0, 55);
    }

    // Cycle 4: edit chunk [2,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(2, 0, 0, 0)).unwrap().dense_blocks();
        blocks[5] = BlockData::simple(0, 33);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(2, 0, 0, 0), Some(resolved));
        expected.get_mut(&[2, 0, 0, 0]).unwrap()[5] = BlockData::simple(0, 33);
    }

    // Cycle 5: re-edit chunk [1,0,0,0] again
    {
        let mut blocks = tree.chunk_payload(key(1, 0, 0, 0)).unwrap().dense_blocks();
        blocks[6] = BlockData::simple(0, 11);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected.get_mut(&[1, 0, 0, 0]).unwrap()[6] = BlockData::simple(0, 11);
    }

    // Final verification of ALL chunks and ALL voxels.
    for (pos, exp) in &expected {
        let payload = tree
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after 5 carve cycles"));
        let blocks = payload.dense_blocks();
        for voxel_idx in 0..7 {
            assert_eq!(
                blocks[voxel_idx], exp[voxel_idx],
                "chunk {pos:?} block[{voxel_idx}] after 5 carve cycles"
            );
        }
    }

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Regression test for the chunk disappearance bug.
///
/// Scenario: The override tree has scattered chunks that consolidate into a
/// ChunkArray spanning gap positions (consolidation requires >= 25% density).
/// The virgin field has solid terrain. overlay_core_in_bounds must NOT overwrite
/// virgin terrain at gap positions with Empty entries from the consolidated array.
#[test]
fn overlay_preserves_virgin_terrain_at_chunk_array_gaps() {
    use crate::shared::voxel::CHUNK_VOLUME;

    let stone = BlockData::simple(0, 11);
    let dirt = BlockData::simple(0, 7);

    // Build a "virgin terrain" tree: a Uniform solid block.
    let virgin_bounds = Aabb4i::new([-1, 0, -1, 0], [0, 0, 0, 0]);
    let virgin_core = RegionTreeCore {
        bounds: virgin_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 42,
    };

    // Build an override tree with chunks at diagonal positions [-1,0,-1,0] and
    // [0,0,0,0]. The bounding box is 2×1×2×1 = 4 cells, with 2 populated.
    // Density = 50% >= 25%, so consolidation triggers.
    // Positions [-1,0,0,0] and [0,0,-1,0] become Empty gap entries.
    let mut override_tree = RegionChunkTree::new();

    let mut dirt_voxels = vec![BlockData::AIR; CHUNK_VOLUME];
    dirt_voxels[0] = dirt.clone();
    let dirt_payload = ResolvedChunkPayload::from_dense_blocks(&dirt_voxels).expect("from dense");

    override_tree.set_chunk(key(-1, 0, -1, 0), Some(dirt_payload.clone()));
    override_tree.set_chunk(key(0, 0, 0, 0), Some(dirt_payload.clone()));

    // Verify that consolidation actually happened: the override tree should have
    // a ChunkArray (not individual branches/leaves) as the root or near the root.
    let override_slice = override_tree.slice_core_in_bounds(virgin_bounds);
    let has_chunk_array = has_chunk_array_node(&override_slice);
    assert!(
        has_chunk_array,
        "override tree should consolidate into a ChunkArray for this test to be valid"
    );

    // Compose: virgin + overrides (same as query_region_core does).
    let mut composed = RegionChunkTree::new();
    let _ = composed.splice_non_empty_core_in_bounds(virgin_bounds, &virgin_core);
    let _ = composed.overlay_core_in_bounds(virgin_bounds, &override_slice);

    // Verify: overridden positions should have dirt at voxel 0.
    let overridden_positions = [key(-1, 0, -1, 0), key(0, 0, 0, 0)];
    for pos in &overridden_positions {
        let payload = composed
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("overridden chunk {pos:?} missing"));
        let blocks = payload.dense_blocks();
        assert_eq!(
            blocks[0], dirt,
            "overridden chunk {pos:?} voxel[0] should be dirt"
        );
    }

    // Verify: gap positions should still have the virgin stone block.
    let gap_positions = [key(-1, 0, 0, 0), key(0, 0, -1, 0)];
    for pos in &gap_positions {
        let payload = composed
            .chunk_payload(*pos)
            .unwrap_or_else(|| panic!("virgin chunk {pos:?} missing after overlay"));
        let blocks = payload.dense_blocks();
        assert_eq!(
            blocks[0], stone,
            "gap position {pos:?} should still have virgin stone, not be overwritten by Empty"
        );
    }
}

fn has_chunk_array_node(core: &RegionTreeCore) -> bool {
    match &core.kind {
        RegionNodeKind::ChunkArray(_) => true,
        RegionNodeKind::Branch(children) => children.iter().any(has_chunk_array_node),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Phase B: scale-aware tests
// ---------------------------------------------------------------------------

#[test]
fn scaled_chunk_key_equality_and_hashing() {
    let a = ScaledChunkKey::new([0, 0, 0, 0], 0);
    let b = ScaledChunkKey::unit([0, 0, 0, 0]);
    assert_eq!(a, b);
    assert_eq!(chunk_key_to_scaled([1, 2, 3, 4]), ScaledChunkKey::new([1, 2, 3, 4], 0));

    // Different scale_exp → different key
    let c = ScaledChunkKey::new([0, 0, 0, 0], -1);
    assert_ne!(a, c);

    // Hash consistency
    use std::hash::{Hash, Hasher};
    let mut ha = std::collections::hash_map::DefaultHasher::new();
    a.hash(&mut ha);
    let mut hb = std::collections::hash_map::DefaultHasher::new();
    b.hash(&mut hb);
    assert_eq!(ha.finish(), hb.finish());
}

#[test]
fn block_data_scale_exp_default_and_at_scale() {
    let b = BlockData::simple(1, 2);
    assert_eq!(b.scale_exp, 0);
    let b2 = b.at_scale(-2);
    assert_eq!(b2.scale_exp, -2);
    assert_eq!(b2.namespace, 1);
    assert_eq!(b2.block_type, 2);
}

#[test]
fn block_data_scale_exp_serde_roundtrip() {
    let b = BlockData::simple(1, 42).at_scale(-3);
    let bytes = postcard::to_allocvec(&b).unwrap();
    let b2: BlockData = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(b, b2);
    assert_eq!(b2.scale_exp, -3);
}

#[test]
fn block_data_scale_exp_serde_missing_field_defaults_to_zero() {
    // Simulate legacy data: serialize without scale_exp, deserialize with default
    let legacy_json = r#"{"namespace":1,"block_type":2}"#;
    let b: BlockData = serde_json::from_str(legacy_json).unwrap();
    assert_eq!(b.scale_exp, 0);
    assert_eq!(b.namespace, 1);
    assert_eq!(b.block_type, 2);
}

#[test]
fn set_and_get_chunk_scaled_round_trip() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 7);
    let payload = ResolvedChunkPayload::uniform(block.clone());
    let scaled_key = ScaledChunkKey::new([0, 0, 0, 0], -1);

    assert!(!tree.has_chunk_scaled(scaled_key));
    assert!(tree.set_chunk_scaled(scaled_key, Some(payload.clone())));
    assert!(tree.has_chunk_scaled(scaled_key));

    let result = tree.chunk_payload_scaled(scaled_key).unwrap();
    assert_eq!(result.uniform_block(), Some(&block));
}

#[test]
fn scaled_chunks_at_non_overlapping_positions() {
    let mut tree = RegionChunkTree::new();
    let block_a = BlockData::simple(1, 10);
    let block_b = BlockData::simple(1, 20);

    // Unit chunk at [0,0,0,0] covers world [0..8)^4
    let unit_key = ScaledChunkKey::unit([0, 0, 0, 0]);
    // Half-scale chunk at [2,0,0,0] covers world [8..12) × [0..4)^3
    // (scale_exp=-1: cell_size=0.5, chunk spans 4 world units per axis)
    let half_key = ScaledChunkKey::new([2, 0, 0, 0], -1);

    tree.set_chunk_scaled(unit_key, Some(ResolvedChunkPayload::uniform(block_a.clone())));
    tree.set_chunk_scaled(half_key, Some(ResolvedChunkPayload::uniform(block_b.clone())));

    // Both should be retrievable at their respective scales
    let result_unit = tree.chunk_payload_scaled(unit_key).unwrap();
    assert_eq!(result_unit.uniform_block(), Some(&block_a));

    let result_half = tree.chunk_payload_scaled(half_key).unwrap();
    assert_eq!(result_half.uniform_block(), Some(&block_b));

    // Querying the wrong scale at each position should return None
    assert!(tree.chunk_payload_scaled(ScaledChunkKey::new([0, 0, 0, 0], -1)).is_none());
    assert!(tree.chunk_payload_scaled(ScaledChunkKey::unit([2, 0, 0, 0])).is_none());
}

#[test]
fn set_chunk_scaled_at_zero_delegates_to_set_chunk() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 5);
    let payload = ResolvedChunkPayload::uniform(block.clone());

    let key = ScaledChunkKey::unit([2, 3, 0, 0]);
    tree.set_chunk_scaled(key, Some(payload));

    // Should be queryable via both the scaled and unscaled APIs
    assert!(tree.has_chunk(key.pos));
    assert!(tree.has_chunk_scaled(key));
    let result = tree.chunk_payload(key.pos).unwrap();
    assert_eq!(result.uniform_block(), Some(&block));
}

#[test]
fn delete_scaled_chunk_removes_it() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 7);
    let key = ScaledChunkKey::new([0, 0, 0, 0], -1);

    tree.set_chunk_scaled(key, Some(ResolvedChunkPayload::uniform(block)));
    assert!(tree.has_chunk_scaled(key));

    tree.set_chunk_scaled(key, None);
    assert!(!tree.has_chunk_scaled(key));
}
