use super::*;
use crate::shared::chunk_payload::{ChunkArrayData, ChunkPayload, ResolvedChunkPayload};
use crate::shared::spatial::{chunk_key_from_lattice, Aabb4i};
use crate::shared::voxel::BlockData;
use std::collections::HashMap;

fn key(x: i32, y: i32, z: i32, w: i32) -> ChunkKey {
    chunk_key_i32(x, y, z, w)
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
    let (lmin, lmax) = bounds.to_chunk_lattice_bounds(0);
    let mut keys = Vec::new();
    for w in lmin[3]..=lmax[3] {
        for z in lmin[2]..=lmax[2] {
            for y in lmin[1]..=lmax[1] {
                for x in lmin[0]..=lmax[0] {
                    keys.push(chunk_key_i32(x, y, z, w));
                }
            }
        }
    }
    keys
}

fn random_sub_bounds(rng: &mut TestRng, outer: Aabb4i) -> Aabb4i {
    let (omin, omax) = outer.to_chunk_lattice_bounds(0);
    let mut min = [0i32; 4];
    let mut max = [0i32; 4];
    for axis in 0..4 {
        let lo = rng.next_inclusive_i32(omin[axis], omax[axis]);
        let hi = rng.next_inclusive_i32(lo, omax[axis]);
        min[axis] = lo;
        max[axis] = hi;
    }
    Aabb4i::from_lattice_bounds(min, max, 0)
}

fn linear_index_in_bounds(bounds: Aabb4i, key: ChunkKey) -> usize {
    let extents = bounds
        .chunk_extents_at_scale(0)
        .expect("bounds used in tests must be valid");
    let (lmin, _) = bounds.to_chunk_lattice_bounds(0);
    let lx = (key[0].to_num::<i32>() - lmin[0]) as usize;
    let ly = (key[1].to_num::<i32>() - lmin[1]) as usize;
    let lz = (key[2].to_num::<i32>() - lmin[2]) as usize;
    let lw = (key[3].to_num::<i32>() - lmin[3]) as usize;
    lx + extents[0] * (ly + extents[1] * (lz + extents[2] * lw))
}

fn chunk_array_uniform_palette_core(bounds: Aabb4i, materials: &[u16]) -> RegionTreeCore {
    let expected_cells = bounds
        .chunk_cell_count_at_scale(0)
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
        0,
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
    let bounds = Aabb4i::chunk_world_bounds(key, scale_exp);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, material as u32)];
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
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
            Some((resolved, _)) => match resolved.uniform_block() {
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
    // scale-(-1) lattice [2,0,0,0] = fixed [1.0, 0, 0, 0] — non-overlapping with coarse [0,0,0,0]
    let fine_non_overlapping = single_cell_chunk_array_core_with_scale(chunk_key_from_lattice([2, 0, 0, 0], -1), 22, -1);

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
fn mixed_scale_world_overlapping_splice_carves_coarse_correctly() {
    let mut tree = RegionChunkTree::new();
    let coarse = single_cell_chunk_array_core_with_scale(key(0, 0, 0, 0), 11, 0);
    // scale-(-1) lattice [1,0,0,0] = fixed [0.5, 0, 0, 0] — overlaps with coarse [0,0,0,0]
    let fine_overlapping = single_cell_chunk_array_core_with_scale(chunk_key_from_lattice([1, 0, 0, 0], -1), 22, -1);

    assert_eq!(
        tree.splice_core_in_bounds(coarse.bounds, &coarse),
        Some(coarse.bounds)
    );
    // Overlapping splice carves the coarse chunk and inserts the fine chunk.
    assert_eq!(
        tree.splice_core_in_bounds(fine_overlapping.bounds, &fine_overlapping),
        Some(fine_overlapping.bounds)
    );

    let root = tree.root().expect("root");
    // The tree should be non-overlapping after carving.
    assert!(validate_region_core_world_space_non_overlapping(root).is_ok());
    // The fine-scale chunk should be queryable.
    let fine_key = chunk_key_from_lattice([1, 0, 0, 0], -1);
    let (payload, scale) = tree.chunk_payload(fine_key).expect("fine chunk present");
    assert_eq!(scale, -1);
    assert_eq!(payload.uniform_block(), Some(&BlockData::simple(0, 22)));
}

#[test]
fn chunk_array_consolidation_does_not_cross_scale_boundaries() {
    let mut tree = RegionChunkTree::new();
    let coarse = single_cell_chunk_array_core_with_scale(key(0, 0, 0, 0), 11, 0);
    // scale-(-1) lattice [2,0,0,0] = fixed [1.0, 0, 0, 0] — non-overlapping
    let fine = single_cell_chunk_array_core_with_scale(chunk_key_from_lattice([2, 0, 0, 0], -1), 22, -1);

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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.uniform_block(),
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 11))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 11))
    );
    assert_eq!(tree.chunk_payload(key(1, 0, 0, 0)), None);

    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 0, 0, 0], 0);
    let non_empty = collect_non_empty_chunks_from_core_in_bounds(
        &tree.slice_non_empty_core_in_bounds(bounds),
        bounds,
    );
    let mut keys: Vec<ChunkKey> = non_empty
        .into_iter()
        .map(|(chunk_key, _)| chunk_key)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, vec![key(0, 0, 0, 0), key(2, 0, 0, 0)]);
}

#[test]
fn adjacent_uniform_children_merge_even_when_branch_partition_is_non_canonical() {
    let mut tree = RegionChunkTree::new();
    let full_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 0, 0, 0], 0);
    let full_uniform = RegionTreeCore {
        bounds: full_bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 11)),
        generator_version_hash: 0,
    };
    assert_eq!(
        tree.splice_non_empty_core_in_bounds(full_bounds, &full_uniform),
        Some(full_bounds)
    );

    let center_bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0);
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
    let global = Aabb4i::from_lattice_bounds([-6, -3, -6, -3], [6, 3, 6, 3], 0);

    for _step in 0..500 {
        let op = rng.next_u32() % 10;
        if op < 7 {
            let (gmin, gmax) = global.to_chunk_lattice_bounds(0);
            let chunk = chunk_key_i32(
                rng.next_inclusive_i32(gmin[0], gmax[0]),
                rng.next_inclusive_i32(gmin[1], gmax[1]),
                rng.next_inclusive_i32(gmin[2], gmax[2]),
                rng.next_inclusive_i32(gmin[3], gmax[3]),
            );
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

    let bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0);
    let patch_core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 9)),
        generator_version_hash: 0,
    };

    let changed_bounds = tree.splice_non_empty_core_in_bounds(bounds, &patch_core);
    assert_eq!(changed_bounds, Some(bounds));
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 2))
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

#[test]
fn take_non_empty_core_extracts_and_clears_region() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 2)))));
    assert!(tree.set_chunk(key(1, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 3)))));
    assert!(tree.set_chunk(key(2, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4)))));

    let bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [2, 0, 0, 0], 0);
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 2))
    );
}

#[test]
fn lazy_drop_outside_bounds_respects_budget() {
    let mut tree = RegionChunkTree::new();
    assert!(tree.set_chunk(key(-8, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 6)))));
    assert!(tree.set_chunk(key(8, 0, 0, 0), Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7)))));

    let keep_bounds = Aabb4i::from_lattice_bounds([-9, -1, -1, -1], [-7, 1, 1, 1], 0);
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

    let bounds = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [3, 0, 0, 0], 0);
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
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 7, 7, 7], 0);
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
        tree.chunk_payload(key(3, 4, 2, 5)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
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
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 7, 7, 7], 0);
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
    let bounds = Aabb4i::from_lattice_bounds([-4, -4, -4, -4], [4, 4, 4, 4], 0);
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
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
        child.bounds == Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0)
            && matches!(child.kind, RegionNodeKind::Uniform(ref b) if b.block_type == 12)
    }));
    assert!(children.iter().any(|child| {
        child.bounds != Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0)
            && matches!(child.kind, RegionNodeKind::ProceduralRef(ref g) if *g == generator)
    }));
}

#[test]
fn splice_identical_partial_uniform_region_is_noop() {
    let mut tree = RegionChunkTree::new();
    let root_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [31, 7, 31, 7], 0);
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

    let patch_bounds = Aabb4i::from_lattice_bounds([4, 1, 4, 1], [20, 5, 20, 5], 0);
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

    let patch_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, 3), BlockData::simple(0, 4)];
    let patch_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        patch_bounds,
        vec![ChunkPayload::Uniform(1), ChunkPayload::Uniform(2)],
        vec![0, 1],
        None,
        block_palette,
        0,
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 3))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

#[test]
fn overlay_core_applies_explicit_uniform_zero_and_non_zero_leaves() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
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

    let zero_leaf_bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0);
    let nine_leaf_bounds = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [2, 0, 0, 0], 0);
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 7))
    );
    assert_eq!(
        tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::AIR)
    );
    assert_eq!(
        tree.chunk_payload(key(2, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 9))
    );
    assert_eq!(
        tree.chunk_payload(key(3, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 7))
    );
}

#[test]
fn splice_non_empty_randomized_window_replacements_match_reference_grid() {
    let global_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [5, 2, 5, 1], 0);
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

        for key in patch_keys.iter().copied() {
            expected.remove(&key);
            let idx = linear_index_in_bounds(patch_bounds, key);
            let material = materials[idx];
            if material != 0 {
                expected.insert(key, material);
            }
        }
    }
}

#[test]
fn overlay_core_randomized_uniform_leaf_layers_match_reference_grid() {
    let global_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [4, 1, 4, 1], 0);
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
    let global_bounds = Aabb4i::from_lattice_bounds([-4, -2, -3, -2], [3, 1, 4, 1], 0);
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
    let global_bounds = Aabb4i::from_lattice_bounds([-5, -1, -5, -1], [6, 2, 6, 2], 0);
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
    let platform_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 7, 0], 0);
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
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("chunk must exist").0;
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
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("first edit disappeared after second edit").0;
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }
    {
        let retrieved_b = tree.chunk_payload(key(4, 0, 3, 0)).expect("second edit not present").0;
        assert_eq!(retrieved_b.block_at(0), BlockData::simple(0, 10));
        assert_eq!(retrieved_b.block_at(1), BlockData::AIR);
    }
    // Platform Uniform elsewhere must be intact.
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
}

/// Same scenario via splice (the multiplayer patch path) instead of set_chunk.
#[test]
fn splice_consolidation_preserves_dense_chunk_edits_on_uniform_platform() {
    let mut tree = RegionChunkTree::new();
    let platform_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 7, 0], 0);
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
    let a_bounds = Aabb4i::from_lattice_bounds([3, 0, 3, 0], [3, 0, 3, 0], 0);
    let a_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        a_bounds,
        vec![payload_a.clone()],
        vec![0u16],
        Some(0),
        block_palette_a,
        0,
    )
    .unwrap();
    let a_core = RegionTreeCore {
        bounds: a_bounds,
        kind: RegionNodeKind::ChunkArray(a_ca),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(a_bounds, &a_core);
    {
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("chunk A must exist after splice").0;
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
    let b_bounds = Aabb4i::from_lattice_bounds([4, 0, 3, 0], [4, 0, 3, 0], 0);
    let b_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        b_bounds,
        vec![payload_b.clone()],
        vec![0u16],
        Some(0),
        block_palette_b,
        0,
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
        let retrieved_a = tree.chunk_payload(key(3, 0, 3, 0)).expect("first splice disappeared after second splice").0;
        assert_eq!(retrieved_a.block_at(0), BlockData::simple(0, 9));
        assert_eq!(retrieved_a.block_at(1), BlockData::AIR);
    }
    {
        let retrieved_b = tree.chunk_payload(key(4, 0, 3, 0)).expect("second splice not present").0;
        assert_eq!(retrieved_b.block_at(0), BlockData::simple(0, 10));
        assert_eq!(retrieved_b.block_at(1), BlockData::AIR);
    }
    assert_eq!(
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
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
    let platform_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 7, 0], 0);
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
                .map(|(p, _)| p)
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
        tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 4))
    );
    assert_eq!(
        tree.chunk_payload(key(7, 0, 7, 0)).unwrap().0.uniform_block(),
        Some(&BlockData::simple(0, 4))
    );

    // Now edit chunk [3,0,3,0] again (already edited above) — triggers re-carve of
    // consolidated ChunkArray.
    {
        let mut blocks = tree
            .chunk_payload(key(3, 0, 3, 0))
            .expect("chunk should exist")
            .0
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
            .map(|(p, _)| p)
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
            .map(|(p, _)| p)
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
    let ca_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 0, 0], 0);
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
            .map(|(p, _)| p)
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
            .0
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
            .map(|(p, _)| p)
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
            .map(|(p, _)| p)
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
    let ca_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
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
        0,
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
            .map(|(p, _)| p)
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
        let (payload, _) = tree
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
            .map(|(p, _)| p)
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
        let p0 = tree.chunk_payload(key(0, 0, 0, 0)).expect("x=0 missing after second edit").0;
        assert_eq!(p0.dense_blocks()[0], BlockData::simple(0, 1));
    }
    {
        let p1 = tree.chunk_payload(key(1, 0, 0, 0)).expect("x=1 missing after second edit").0;
        assert_eq!(p1.dense_blocks()[0], BlockData::simple(0, 50));
    }
    {
        let p2 = tree.chunk_payload(key(2, 0, 0, 0)).expect("x=2 missing after second edit").0;
        assert_eq!(
            p2.uniform_block(),
            Some(&BlockData::simple(0, 60))
        );
    }
    {
        let p3 = tree.chunk_payload(key(3, 0, 0, 0)).expect("x=3 missing after second edit").0;
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
    let positions: Vec<ChunkKey> = vec![
        chunk_key_i32(0, 0, 0, 0),
        chunk_key_i32(1, 0, 0, 0),
        chunk_key_i32(2, 0, 0, 0),
        chunk_key_i32(0, 1, 0, 0),
        chunk_key_i32(1, 1, 0, 0),
        chunk_key_i32(2, 1, 0, 0),
    ];

    // Build a "save tree" containing all 6 chunks as a single ChunkArray.
    let mut save_block_palette = vec![BlockData::AIR];
    let mut save_chunk_palette = Vec::new();
    let mut save_dense_indices = Vec::new();
    let mut original_blocks: HashMap<ChunkKey, Vec<BlockData>> = HashMap::new();

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

    let save_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 1, 0, 0], 0);
    let save_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        save_bounds,
        save_chunk_palette,
        save_dense_indices,
        None,
        save_block_palette,
        0,
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
            .map(|(p, _)| p)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after bulk load"));
        let blocks = payload.dense_blocks();
        let expected = &original_blocks[pos];
        assert_eq!(blocks[0], expected[0], "chunk {pos:?} block[0] after load");
        assert_eq!(blocks[1], expected[1], "chunk {pos:?} block[1] after load");
    }

    // Phase 2: Edit 3 chunks (simulating apply_voxel_edit).
    let mut dirty_save_chunks: HashSet<ChunkKey> = HashSet::new();
    let edited_positions = [chunk_key_i32(0, 0, 0, 0), chunk_key_i32(1, 1, 0, 0), chunk_key_i32(2, 0, 0, 0)];
    for pos in &edited_positions {
        let mut blocks = tree
            .chunk_payload(*pos)
            .expect("chunk should exist for edit")
            .0
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
            .map(|(p, _)| p)
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
    let mut dirty_payloads: Vec<(ChunkKey, Option<ResolvedChunkPayload>)> = Vec::new();
    for pos in &dirty_save_chunks {
        let resolved = tree.chunk_payload(*pos).map(|(p, _)| p);
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
        let chunk_bounds = Aabb4i::chunk_world_bounds(*pos, 0);
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
            .map(|(p, _)| p)
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
    let positions: Vec<ChunkKey> = vec![
        chunk_key_i32(0, 0, 0, 0),
        chunk_key_i32(1, 0, 0, 0),
        chunk_key_i32(2, 0, 0, 0),
        chunk_key_i32(3, 0, 0, 0),
    ];

    let mut save_block_palette = vec![BlockData::AIR];
    let mut save_chunk_palette = Vec::new();
    let mut save_dense_indices = Vec::new();
    let mut expected_blocks: HashMap<ChunkKey, Vec<BlockData>> = HashMap::new();

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

    let save_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let save_chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        save_bounds,
        save_chunk_palette,
        save_dense_indices,
        None,
        save_block_palette,
        0,
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
        let chunk_bounds = Aabb4i::chunk_world_bounds(*pos, 0);
        let sliced = slice_region_core_in_bounds(&save_core, chunk_bounds);
        tree.splice_non_empty_core_in_bounds(chunk_bounds, &sliced);

        // After each load, verify ALL previously loaded chunks are still correct.
        for prev_pos in &positions[..=load_idx] {
            let (payload, _) = tree.chunk_payload(*prev_pos).unwrap_or_else(|| {
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
            .0
            .dense_blocks();
        blocks[0] = BlockData::simple(0, 777);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected_blocks.get_mut(&chunk_key_i32(1, 0, 0, 0)).unwrap()[0] = BlockData::simple(0, 777);
    }

    // Verify ALL chunks after edit.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .map(|(p, _)| p)
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
            .0
            .dense_blocks();
        blocks[0] = BlockData::simple(0, 888);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(3, 0, 0, 0), Some(resolved));
        expected_blocks.get_mut(&chunk_key_i32(3, 0, 0, 0)).unwrap()[0] = BlockData::simple(0, 888);
    }

    // Verify ALL chunks after second edit.
    for pos in &positions {
        let payload = tree
            .chunk_payload(*pos)
            .map(|(p, _)| p)
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
    let mut expected: HashMap<ChunkKey, Vec<BlockData>> = HashMap::new();

    // Insert 4 chunks one at a time (each triggers potential consolidation).
    for i in 0..4i32 {
        let pos = chunk_key_i32(i, 0, 0, 0);
        let mut blocks = vec![BlockData::AIR; chunk_volume];
        blocks[0] = BlockData::simple(0, (i + 1) as u32);
        blocks[1] = BlockData::simple(0, (i + 1) as u32 * 10);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(pos, Some(resolved));
        expected.insert(pos, blocks);
    }

    // Cycle 1: edit chunk [1,0,0,0] → carves consolidated array
    {
        let mut blocks = tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.dense_blocks();
        blocks[2] = BlockData::simple(0, 42);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected.get_mut(&chunk_key_i32(1, 0, 0, 0)).unwrap()[2] = BlockData::simple(0, 42);
    }

    // Verify after cycle 1.
    for (pos, exp) in &expected {
        let payload = tree
            .chunk_payload(*pos)
            .map(|(p, _)| p)
            .unwrap_or_else(|| panic!("chunk {pos:?} missing after cycle 1"));
        let blocks = payload.dense_blocks();
        assert_eq!(blocks[0], exp[0], "cycle 1: chunk {pos:?} block[0]");
        assert_eq!(blocks[1], exp[1], "cycle 1: chunk {pos:?} block[1]");
        assert_eq!(blocks[2], exp[2], "cycle 1: chunk {pos:?} block[2]");
    }

    // Cycle 2: edit chunk [3,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(3, 0, 0, 0)).unwrap().0.dense_blocks();
        blocks[3] = BlockData::simple(0, 77);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(3, 0, 0, 0), Some(resolved));
        expected.get_mut(&chunk_key_i32(3, 0, 0, 0)).unwrap()[3] = BlockData::simple(0, 77);
    }

    // Cycle 3: edit chunk [0,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(0, 0, 0, 0)).unwrap().0.dense_blocks();
        blocks[4] = BlockData::simple(0, 55);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(0, 0, 0, 0), Some(resolved));
        expected.get_mut(&chunk_key_i32(0, 0, 0, 0)).unwrap()[4] = BlockData::simple(0, 55);
    }

    // Cycle 4: edit chunk [2,0,0,0] → carves again
    {
        let mut blocks = tree.chunk_payload(key(2, 0, 0, 0)).unwrap().0.dense_blocks();
        blocks[5] = BlockData::simple(0, 33);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(2, 0, 0, 0), Some(resolved));
        expected.get_mut(&chunk_key_i32(2, 0, 0, 0)).unwrap()[5] = BlockData::simple(0, 33);
    }

    // Cycle 5: re-edit chunk [1,0,0,0] again
    {
        let mut blocks = tree.chunk_payload(key(1, 0, 0, 0)).unwrap().0.dense_blocks();
        blocks[6] = BlockData::simple(0, 11);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense");
        tree.set_chunk(key(1, 0, 0, 0), Some(resolved));
        expected.get_mut(&chunk_key_i32(1, 0, 0, 0)).unwrap()[6] = BlockData::simple(0, 11);
    }

    // Final verification of ALL chunks and ALL voxels.
    for (pos, exp) in &expected {
        let payload = tree
            .chunk_payload(*pos)
            .map(|(p, _)| p)
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
    let virgin_bounds = Aabb4i::from_lattice_bounds([-1, 0, -1, 0], [0, 0, 0, 0], 0);
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
            .map(|(p, _)| p)
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
            .map(|(p, _)| p)
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
// Phase B: scale-aware tests (using fixed-point ChunkKey)
// ---------------------------------------------------------------------------

#[test]
fn fixed_point_chunk_key_equality_and_hashing() {
    use crate::shared::spatial::{chunk_key_from_lattice, ChunkCoord};

    // Scale 0: lattice [1,2,3,4] → fixed [1.0, 2.0, 3.0, 4.0]
    let a = chunk_key_i32(1, 2, 3, 4);
    let b = chunk_key_from_lattice([1, 2, 3, 4], 0);
    assert_eq!(a, b);

    // Scale -1: lattice [1,0,0,0] → fixed [0.5, 0, 0, 0]
    let c = chunk_key_from_lattice([1, 0, 0, 0], -1);
    assert_eq!(c[0], ChunkCoord::from_bits(1i64 << 15)); // 0.5
    assert_ne!(a, c); // Different from [1, 2, 3, 4]

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
fn set_and_get_chunk_at_scale_round_trip() {
    use crate::shared::spatial::chunk_key_from_lattice;

    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 7);
    let payload = ResolvedChunkPayload::uniform(block.clone());
    // Scale -1, lattice [0,0,0,0] → fixed [0,0,0,0]
    let key = chunk_key_from_lattice([0, 0, 0, 0], -1);

    assert!(!tree.has_chunk(key));
    assert!(tree.set_chunk_at_scale(key, Some(payload.clone()), -1));
    assert!(tree.has_chunk(key));

    let (result, _) = tree.chunk_payload(key).unwrap();
    // Block preserves its original scale_exp (0 from BlockData::simple),
    // NOT the chunk's storage scale. scale_exp is block metadata, not tree metadata.
    assert_eq!(result.uniform_block(), Some(&block));
}

#[test]
fn scaled_chunks_at_non_overlapping_positions() {
    use crate::shared::spatial::chunk_key_from_lattice;

    let mut tree = RegionChunkTree::new();
    let block_a = BlockData::simple(1, 10);
    let block_b = BlockData::simple(1, 20);

    // Unit chunk at [0,0,0,0] covers world [0..8)^4
    let unit_key = chunk_key_i32(0, 0, 0, 0);
    // Half-scale chunk at lattice [2,0,0,0] at scale -1 → fixed [1, 0, 0, 0]
    // Covers world [4..8) × [0..4)^3 at half-scale cell size
    let half_key = chunk_key_from_lattice([2, 0, 0, 0], -1);

    tree.set_chunk(unit_key, Some(ResolvedChunkPayload::uniform(block_a.clone())));
    tree.set_chunk_at_scale(half_key, Some(ResolvedChunkPayload::uniform(block_b.clone())), -1);

    // Both should be retrievable
    let (result_unit, _) = tree.chunk_payload(unit_key).unwrap();
    assert_eq!(result_unit.uniform_block(), Some(&block_a));

    // Block preserves its original scale_exp (0), not the chunk's storage scale.
    let (result_half, _) = tree.chunk_payload(half_key).unwrap();
    assert_eq!(result_half.uniform_block(), Some(&block_b));
}

#[test]
fn set_chunk_at_scale_zero_delegates_to_set_chunk() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 5);
    let payload = ResolvedChunkPayload::uniform(block.clone());

    let key = chunk_key_i32(2, 3, 0, 0);
    tree.set_chunk_at_scale(key, Some(payload), 0);

    // Should be queryable via both set_chunk and set_chunk_at_scale APIs
    assert!(tree.has_chunk(key));
    let (result, _) = tree.chunk_payload(key).unwrap();
    assert_eq!(result.uniform_block(), Some(&block));
}

#[test]
fn delete_scaled_chunk_removes_it() {
    use crate::shared::spatial::chunk_key_from_lattice;

    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 7);
    let key = chunk_key_from_lattice([0, 0, 0, 0], -1);

    tree.set_chunk_at_scale(key, Some(ResolvedChunkPayload::uniform(block)), -1);
    assert!(tree.has_chunk(key));

    tree.set_chunk_at_scale(key, None, -1);
    assert!(!tree.has_chunk(key));
}

/// Simulate the server generating the origin platform (MassivePlatforms style)
/// and verify that chunks across the full extent are accessible.
#[test]
fn origin_platform_uniform_accessible_at_positive_x() {
    let stone = BlockData::simple(0, 11);
    let platform_bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);
    let core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };

    let mut tree = RegionChunkTree::new();
    let _ = tree.splice_non_empty_core_in_bounds(platform_bounds, &core);

    // Check origin chunk
    let payload_origin = tree.chunk_payload(key(0, 0, 0, 0));
    assert!(payload_origin.is_some(), "origin chunk [0,0,0,0] should be present");

    // Check +x chunks
    for x in 1..=17 {
        let payload = tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "chunk [{x},0,0,0] should be present in Uniform platform"
        );
    }

    // Check -x chunks
    for x in -18..0 {
        let payload = tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "chunk [{x},0,0,0] should be present in Uniform platform"
        );
    }
}

/// Simulate the multiplayer patch flow: server generates a subtree,
/// client receives it via splice_non_empty_core_in_bounds.
#[test]
fn multiplayer_patch_flow_preserves_platform_chunks() {
    let stone = BlockData::simple(0, 11);
    let platform_bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);

    // Server-side: generate world for a query region
    let query_bounds = Aabb4i::from_lattice_bounds([-10, -5, -10, -10], [10, 5, 10, 10], 0);
    let mut server_tree = RegionChunkTree::new();
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    let _ = server_tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Server slices the subtree for the client
    let subtree = server_tree.slice_non_empty_core_in_bounds(query_bounds);

    // Client receives the patch
    let mut client_tree = RegionChunkTree::new();
    let _ = client_tree.splice_non_empty_core_in_bounds(query_bounds, &subtree);

    // Verify chunks in the platform region within the query bounds
    for x in -10..=10 {
        let payload = client_tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "chunk [{x},0,0,0] should be present after multiplayer patch"
        );
        let payload_neg_y = client_tree.chunk_payload(key(x, -1, 0, 0));
        assert!(
            payload_neg_y.is_some(),
            "chunk [{x},-1,0,0] should be present after multiplayer patch"
        );
    }

    // Also verify +x edge chunks that are AT the query boundary
    let edge_payload = client_tree.chunk_payload(key(10, 0, 0, 0));
    assert!(
        edge_payload.is_some(),
        "edge chunk [10,0,0,0] should be present"
    );
}

/// Simulate the client receiving multiple overlapping patches as the player moves.
#[test]
fn sequential_multiplayer_patches_preserve_platform_data() {
    let stone = BlockData::simple(0, 11);
    let platform_bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);

    // Build the server world
    let mut server_tree = RegionChunkTree::new();
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    let _ = server_tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // First client interest: around spawn
    let first_bounds = Aabb4i::from_lattice_bounds([-8, -4, -8, -8], [8, 4, 8, 8], 0);
    let first_subtree = server_tree.slice_non_empty_core_in_bounds(first_bounds);

    let mut client_tree = RegionChunkTree::new();
    let _ = client_tree.splice_non_empty_core_in_bounds(first_bounds, &first_subtree);

    // Verify initial chunks
    for x in -8..=8 {
        let payload = client_tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "chunk [{x},0,0,0] missing after first patch"
        );
    }

    // Second client interest: shifted +x
    let second_bounds = Aabb4i::from_lattice_bounds([-4, -4, -8, -8], [14, 4, 8, 8], 0);
    let second_subtree = server_tree.slice_non_empty_core_in_bounds(second_bounds);
    let _ = client_tree.splice_non_empty_core_in_bounds(second_bounds, &second_subtree);

    // Verify chunks after second patch
    // The platform extends to x=17, so chunks up to x=14 should all be solid
    let mut missing = Vec::new();
    for x in -8..=14 {
        let payload = client_tree.chunk_payload(key(x, 0, 0, 0));
        if payload.is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after second patch at x={missing:?}"
    );

    // Verify that old chunks from the first patch that are OUTSIDE the second patch
    // are still present (they shouldn't be cleared)
    let old_edge = client_tree.chunk_payload(key(-8, 0, 0, 0));
    assert!(
        old_edge.is_some(),
        "chunk [-8,0,0,0] should still be present from first patch"
    );
}

// ============================================================================
// Regression tests for I48F16 / multi-scale migration at scale_exp=0
// ============================================================================

/// Basic set_chunk + chunk_payload round-trip at scale_exp=0.
/// Regression: if fixed-point coordinate comparison or tree insertion is broken,
/// chunks will be lost or placed at wrong positions.
#[test]
fn set_chunk_query_roundtrip_at_scale_zero() {
    let stone = BlockData::simple(0, 11);
    let mut tree = RegionChunkTree::new();

    // Set chunks at various positions including negative coords
    let positions = [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (-1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
        (5, -3, 2, -1),
        (-10, -10, -10, -10),
    ];

    for &(x, y, z, w) in &positions {
        let payload = ResolvedChunkPayload {
            payload: ChunkPayload::Uniform(1),
            block_palette: vec![BlockData::AIR, stone.clone()],
        };
        tree.set_chunk(key(x, y, z, w), Some(payload));
    }

    // Verify every chunk is retrievable
    for &(x, y, z, w) in &positions {
        let payload = tree.chunk_payload(key(x, y, z, w));
        assert!(
            payload.is_some(),
            "chunk [{x},{y},{z},{w}] missing after set_chunk"
        );
    }

    // Verify a position that was never set returns None
    let missing = tree.chunk_payload(key(100, 100, 100, 100));
    assert!(missing.is_none(), "unset chunk should return None");
}

/// Verify that set_chunk (which calls set_chunk_at_scale(0)) doesn't lose data
/// when multiple chunks are added sequentially, forcing tree growth.
#[test]
fn sequential_set_chunk_preserves_all_chunks() {
    let stone = BlockData::simple(0, 11);
    let mut tree = RegionChunkTree::new();
    let mut expected = Vec::new();

    // Add chunks in a line along x-axis, forcing repeated root expansion
    for x in -20..=20 {
        let payload = ResolvedChunkPayload {
            payload: ChunkPayload::Uniform(1),
            block_palette: vec![BlockData::AIR, stone.clone()],
        };
        tree.set_chunk(key(x, 0, 0, 0), Some(payload));
        expected.push(x);
    }

    let mut missing = Vec::new();
    for &x in &expected {
        if tree.chunk_payload(key(x, 0, 0, 0)).is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after sequential set_chunk: x={missing:?}"
    );
}

/// Verify that expand_root_once correctly preserves existing tree content.
/// Regression: if the step parameter or bounds arithmetic is wrong,
/// the old root may be placed incorrectly in the expanded tree.
#[test]
fn expand_root_preserves_existing_content() {
    let stone = BlockData::simple(0, 11);
    let mut tree = RegionChunkTree::new();

    // Place a chunk at origin
    let payload = ResolvedChunkPayload {
        payload: ChunkPayload::Uniform(1),
        block_palette: vec![BlockData::AIR, stone.clone()],
    };
    tree.set_chunk(key(0, 0, 0, 0), Some(payload.clone()));

    // Verify it exists
    assert!(tree.chunk_payload(key(0, 0, 0, 0)).is_some());

    // Place a chunk far away, forcing root expansion
    tree.set_chunk(key(100, 0, 0, 0), Some(payload.clone()));

    // Both chunks must be present
    assert!(
        tree.chunk_payload(key(0, 0, 0, 0)).is_some(),
        "origin chunk lost after root expansion"
    );
    assert!(
        tree.chunk_payload(key(100, 0, 0, 0)).is_some(),
        "far chunk not found after root expansion"
    );

    // Expand in negative direction too
    tree.set_chunk(key(-100, 0, 0, 0), Some(payload));
    assert!(
        tree.chunk_payload(key(0, 0, 0, 0)).is_some(),
        "origin chunk lost after second root expansion"
    );
    assert!(
        tree.chunk_payload(key(100, 0, 0, 0)).is_some(),
        "far chunk lost after second root expansion"
    );
    assert!(
        tree.chunk_payload(key(-100, 0, 0, 0)).is_some(),
        "negative chunk not found after second root expansion"
    );
}

/// subtract_aabb via clear: clearing a center region from a Uniform leaf
/// should produce pieces that cover exactly the outer minus inner.
/// Regression: if subtract_aabb with step=1 breaks, clear_node_region corrupts the tree.
#[test]
fn clear_center_of_uniform_produces_correct_coverage() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // Create a 6x6x6x6 uniform platform
    let outer = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [5, 5, 5, 5], 0);
    let core = RegionTreeCore {
        bounds: outer,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(outer, &core);

    // Clear a center region by splicing Empty
    let inner = Aabb4i::from_lattice_bounds([2, 2, 2, 2], [3, 3, 3, 3], 0);
    let empty_core = RegionTreeCore {
        bounds: inner,
        kind: RegionNodeKind::Empty,
        generator_version_hash: 0,
    };
    tree.splice_core_in_bounds(inner, &empty_core);

    // Verify: every cell in outer that's NOT in inner should be present
    let mut missing = Vec::new();
    let mut false_present = Vec::new();
    for k in keys_in_bounds(outer) {
        let in_inner = inner.contains_chunk_world_min(k);
        let present = tree.chunk_payload(k).is_some();
        if in_inner && present {
            false_present.push(k);
        } else if !in_inner && !present {
            missing.push(k);
        }
    }
    assert!(
        false_present.is_empty(),
        "{} cells in cleared inner region are still present",
        false_present.len()
    );
    assert!(
        missing.is_empty(),
        "{} cells outside cleared inner region are missing",
        missing.len()
    );
}

/// splice_non_empty with a Uniform region extending beyond the existing tree.
/// Simplified version of the failing sequential_multiplayer_patches test.
/// Regression: splice drops data from the replacement that extends beyond the original.
#[test]
fn splice_uniform_extending_beyond_existing_tree() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // First splice: small region
    let first_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let first_core = RegionTreeCore {
        bounds: first_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(first_bounds, &first_core);

    // Verify first splice
    for x in 0..=3 {
        assert!(
            tree.chunk_payload(key(x, 0, 0, 0)).is_some(),
            "chunk [{x},0,0,0] missing after first splice"
        );
    }

    // Second splice: overlapping and extending further in x
    let second_bounds = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [7, 0, 0, 0], 0);
    let second_core = RegionTreeCore {
        bounds: second_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(second_bounds, &second_core);

    // Verify all chunks present: 0..3 from first, 4..7 from second
    let mut missing = Vec::new();
    for x in 0..=7 {
        if tree.chunk_payload(key(x, 0, 0, 0)).is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after second splice: x={missing:?}"
    );
}

/// Splice where the second patch is entirely outside the first.
/// Tests root expansion + splice into empty expanded space.
#[test]
fn splice_into_disjoint_region_preserves_both() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // First region: x=0..3
    let first_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let first_core = RegionTreeCore {
        bounds: first_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(first_bounds, &first_core);

    // Second region: x=10..13, entirely disjoint
    let second_bounds = Aabb4i::from_lattice_bounds([10, 0, 0, 0], [13, 0, 0, 0], 0);
    let second_core = RegionTreeCore {
        bounds: second_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(second_bounds, &second_core);

    // Verify first region
    for x in 0..=3 {
        assert!(
            tree.chunk_payload(key(x, 0, 0, 0)).is_some(),
            "first-region chunk [{x},0,0,0] missing"
        );
    }
    // Verify second region
    for x in 10..=13 {
        assert!(
            tree.chunk_payload(key(x, 0, 0, 0)).is_some(),
            "second-region chunk [{x},0,0,0] missing"
        );
    }
    // Verify gap is empty
    for x in 4..=9 {
        assert!(
            tree.chunk_payload(key(x, 0, 0, 0)).is_none(),
            "gap chunk [{x},0,0,0] should be empty"
        );
    }
}

/// clear_node_region on a Uniform leaf should produce correct pieces.
/// Regression: if subtract_aabb with step=1 doesn't produce valid pieces,
/// the clear operation will corrupt the tree.
#[test]
fn clear_region_of_uniform_leaf_preserves_surrounding() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // Create a uniform region
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [5, 0, 5, 0], 0);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(bounds, &core);

    // Delete a chunk in the middle by setting it to None
    tree.set_chunk(key(2, 0, 2, 0), None);

    // The deleted chunk should be gone
    assert!(
        tree.chunk_payload(key(2, 0, 2, 0)).is_none(),
        "deleted chunk should be None"
    );

    // All surrounding chunks should still be present
    let mut missing = Vec::new();
    for x in 0..=5 {
        for z in 0..=5 {
            if x == 2 && z == 2 {
                continue; // skip deleted
            }
            if tree.chunk_payload(key(x, 0, z, 0)).is_none() {
                missing.push((x, z));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after single-cell clear: {missing:?}"
    );
}

/// Verify that the overlap validation isn't causing pathological performance.
/// This test simulates a realistic world loading pattern with many set_chunk calls
/// and checks that it completes in a reasonable time.
#[test]
fn bulk_set_chunk_completes_in_reasonable_time() {
    let stone = BlockData::simple(0, 11);
    let mut tree = RegionChunkTree::new();

    let start = std::time::Instant::now();
    // Simulate loading a small world: 8x2x8x2 = 256 chunks
    for w in 0..2 {
        for z in 0..8 {
            for y in 0..2 {
                for x in 0..8 {
                    let payload = ResolvedChunkPayload {
                        payload: ChunkPayload::Uniform(1),
                        block_palette: vec![BlockData::AIR, stone.clone()],
                    };
                    tree.set_chunk(key(x, y, z, w), Some(payload));
                }
            }
        }
    }
    let elapsed = start.elapsed();

    // 256 set_chunk calls should complete in well under 10 seconds.
    // The O(n²) overlap validation makes this blow up if it runs on every call.
    assert!(
        elapsed.as_secs() < 10,
        "bulk set_chunk took {elapsed:?} — likely pathological overlap validation"
    );

    // Verify all chunks are present
    let mut missing = Vec::new();
    for w in 0..2 {
        for z in 0..8 {
            for y in 0..2 {
                for x in 0..8 {
                    if tree.chunk_payload(key(x, y, z, w)).is_none() {
                        missing.push((x, y, z, w));
                    }
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "{} chunks missing after bulk load",
        missing.len()
    );
}

/// Splice a ChunkArray (not just Uniform) into a tree that already has content.
/// This tests the code path where clear_node_region must split a ChunkArray,
/// not just a Uniform leaf.
#[test]
fn splice_chunk_array_over_existing_chunk_array() {
    let _stone = BlockData::simple(0, 11);
    let _dirt = BlockData::simple(0, 3);

    let mut tree = RegionChunkTree::new();

    // Build a 4x1x1x1 ChunkArray with stone
    let first_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let materials: Vec<u16> = vec![11, 11, 11, 11]; // all stone
    let first_core = chunk_array_uniform_palette_core(first_bounds, &materials);
    tree.splice_non_empty_core_in_bounds(first_bounds, &first_core);

    // Splice dirt over the middle two chunks
    let second_bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [2, 0, 0, 0], 0);
    let dirt_materials: Vec<u16> = vec![3, 3];
    let second_core = chunk_array_uniform_palette_core(second_bounds, &dirt_materials);
    tree.splice_non_empty_core_in_bounds(second_bounds, &second_core);

    // x=0 should be stone
    let p0 = tree.chunk_payload(key(0, 0, 0, 0));
    assert!(p0.is_some(), "chunk [0,0,0,0] missing");

    // x=1,2 should be dirt (replaced)
    let p1 = tree.chunk_payload(key(1, 0, 0, 0));
    assert!(p1.is_some(), "chunk [1,0,0,0] missing");

    // x=3 should be stone
    let p3 = tree.chunk_payload(key(3, 0, 0, 0));
    assert!(p3.is_some(), "chunk [3,0,0,0] missing");
}

/// Test the exact multiplayer scenario: server has large world, client receives
/// successive overlapping patches as they move, and chunks from new patches
/// that extend beyond old patches must be present.
#[test]
fn multiplayer_sliding_window_preserves_new_chunks_simplified() {
    let stone = BlockData::simple(0, 11);

    // Server world: platform at y=0, x=-20..20 (1D for simplicity)
    let server_bounds = Aabb4i::from_lattice_bounds([-20, 0, 0, 0], [20, 0, 0, 0], 0);
    let mut server_tree = RegionChunkTree::new();
    let server_core = RegionTreeCore {
        bounds: server_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    server_tree.splice_non_empty_core_in_bounds(server_bounds, &server_core);

    let mut client_tree = RegionChunkTree::new();

    // Patch 1: x=-5..5
    let p1_bounds = Aabb4i::from_lattice_bounds([-5, 0, 0, 0], [5, 0, 0, 0], 0);
    let p1_core = server_tree.slice_non_empty_core_in_bounds(p1_bounds);
    client_tree.splice_non_empty_core_in_bounds(p1_bounds, &p1_core);

    for x in -5..=5 {
        assert!(
            client_tree.chunk_payload(key(x, 0, 0, 0)).is_some(),
            "patch1: chunk [{x},0,0,0] missing"
        );
    }

    // Patch 2: x=0..10 (overlapping, extends right)
    let p2_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [10, 0, 0, 0], 0);
    let p2_core = server_tree.slice_non_empty_core_in_bounds(p2_bounds);
    client_tree.splice_non_empty_core_in_bounds(p2_bounds, &p2_core);

    let mut missing = Vec::new();
    for x in -5..=10 {
        if client_tree.chunk_payload(key(x, 0, 0, 0)).is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "after patch2: chunks missing at x={missing:?}"
    );

    // Patch 3: x=5..15 (extends further right)
    let p3_bounds = Aabb4i::from_lattice_bounds([5, 0, 0, 0], [15, 0, 0, 0], 0);
    let p3_core = server_tree.slice_non_empty_core_in_bounds(p3_bounds);
    client_tree.splice_non_empty_core_in_bounds(p3_bounds, &p3_core);

    let mut missing = Vec::new();
    for x in -5..=15 {
        if client_tree.chunk_payload(key(x, 0, 0, 0)).is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "after patch3: chunks missing at x={missing:?}"
    );
}

/// Edit (set_chunk) after splice_non_empty — the pattern used during gameplay.
/// Player places/breaks blocks in a world that was loaded via splice patches.
#[test]
fn set_chunk_edit_after_splice_works() {
    let stone = BlockData::simple(0, 11);
    let dirt = BlockData::simple(0, 3);

    let mut tree = RegionChunkTree::new();

    // Load a platform via splice
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 7, 0], 0);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(bounds, &core);

    // Edit: place dirt at (3,0,3,0)
    let edit_payload = ResolvedChunkPayload {
        payload: ChunkPayload::Uniform(1),
        block_palette: vec![BlockData::AIR, dirt.clone()],
    };
    let changed = tree.set_chunk(key(3, 0, 3, 0), Some(edit_payload));
    assert!(changed, "set_chunk should report change");

    // Verify the edit stuck
    let payload = tree.chunk_payload(key(3, 0, 3, 0));
    assert!(payload.is_some(), "edited chunk should be present");

    // Verify surrounding chunks weren't lost
    let mut missing = Vec::new();
    for x in 0..=7 {
        for z in 0..=7 {
            if tree.chunk_payload(key(x, 0, z, 0)).is_none() {
                missing.push((x, z));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after edit: {missing:?}"
    );
}

/// Delete (set_chunk None) after splice — break a block in the world.
#[test]
fn delete_chunk_after_splice_preserves_neighbors() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // Load a platform via splice
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 3, 0], 0);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(bounds, &core);

    // Delete chunk at (1,0,1,0)
    let changed = tree.set_chunk(key(1, 0, 1, 0), None);
    assert!(changed, "set_chunk(None) should report change");

    // Verify deleted chunk is gone
    assert!(tree.chunk_payload(key(1, 0, 1, 0)).is_none());

    // Verify all other chunks remain
    let mut missing = Vec::new();
    for x in 0..=3 {
        for z in 0..=3 {
            if x == 1 && z == 1 {
                continue;
            }
            if tree.chunk_payload(key(x, 0, z, 0)).is_none() {
                missing.push((x, z));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "neighbors missing after delete: {missing:?}"
    );
}

/// Test splice of a partially-empty patch (like a worldgen region with both
/// terrain and air chunks). The splice should only overwrite non-empty cells.
#[test]
fn splice_partial_patch_preserves_existing_at_empty_positions() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // Existing: stone at x=0..3
    let existing_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let existing_core = RegionTreeCore {
        bounds: existing_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(existing_bounds, &existing_core);

    // Incoming patch only covers x=2 with dirt
    // (splice_non_empty only writes non-empty, so x=0,1,3 should keep stone)
    let dirt = BlockData::simple(0, 3);
    let patch_bounds = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [2, 0, 0, 0], 0);
    let patch_core = RegionTreeCore {
        bounds: patch_bounds,
        kind: RegionNodeKind::Uniform(dirt),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(patch_bounds, &patch_core);

    // All chunks should be present
    for x in 0..=3 {
        assert!(
            tree.chunk_payload(key(x, 0, 0, 0)).is_some(),
            "chunk [{x},0,0,0] missing after partial splice"
        );
    }
}

/// Test that splice_non_empty properly clears the intersection before inserting.
/// Without clearing, overlapping splices produce duplicate data.
#[test]
fn splice_overlapping_uniform_regions_no_duplicates() {
    let stone = BlockData::simple(0, 11);
    let dirt = BlockData::simple(0, 3);

    let mut tree = RegionChunkTree::new();

    // Initial: stone at [0,0,0,0]..[5,0,0,0]
    let first_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [5, 0, 0, 0], 0);
    let first_core = RegionTreeCore {
        bounds: first_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(first_bounds, &first_core);

    // Replace overlap with dirt at [3,0,0,0]..[8,0,0,0]
    let second_bounds = Aabb4i::from_lattice_bounds([3, 0, 0, 0], [8, 0, 0, 0], 0);
    let second_core = RegionTreeCore {
        bounds: second_bounds,
        kind: RegionNodeKind::Uniform(dirt.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(second_bounds, &second_core);

    // Everything should be present: 0..2 = stone, 3..8 = dirt
    let mut missing = Vec::new();
    for x in 0..=8 {
        if tree.chunk_payload(key(x, 0, 0, 0)).is_none() {
            missing.push(x);
        }
    }
    assert!(
        missing.is_empty(),
        "chunks missing after overlapping splice: x={missing:?}"
    );
}

/// Multi-dimensional splice: test with a 2D platform (x,z) and overlapping patches.
/// Exercises the 4D bounds math more thoroughly than 1D tests.
#[test]
fn splice_2d_platform_with_overlapping_patches() {
    let stone = BlockData::simple(0, 11);

    let mut tree = RegionChunkTree::new();

    // Server: 10x10 platform at y=0
    let server_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [9, 0, 9, 0], 0);
    let server_core = RegionTreeCore {
        bounds: server_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    let mut server_tree = RegionChunkTree::new();
    server_tree.splice_non_empty_core_in_bounds(server_bounds, &server_core);

    // Client patch 1: x=0..4, z=0..4
    let p1 = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [4, 0, 4, 0], 0);
    let p1_core = server_tree.slice_non_empty_core_in_bounds(p1);
    tree.splice_non_empty_core_in_bounds(p1, &p1_core);

    // Client patch 2: x=3..7, z=3..7 (overlapping)
    let p2 = Aabb4i::from_lattice_bounds([3, 0, 3, 0], [7, 0, 7, 0], 0);
    let p2_core = server_tree.slice_non_empty_core_in_bounds(p2);
    tree.splice_non_empty_core_in_bounds(p2, &p2_core);

    // Verify all chunks in the union of both patches
    let mut missing = Vec::new();
    for x in 0..=7 {
        for z in 0..=7 {
            // Only check positions that are in at least one patch
            let in_p1 = x <= 4 && z <= 4;
            let in_p2 = x >= 3 && x <= 7 && z >= 3 && z <= 7;
            if (in_p1 || in_p2) && tree.chunk_payload(key(x, 0, z, 0)).is_none() {
                missing.push((x, z));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "2D platform chunks missing after overlapping patches: {missing:?}"
    );
}

/// Simulates the exact client bootstrap scenario where the server sends a large
/// Uniform platform node and the client clips it to the authoritative bounds.
/// Verifies that chunks at y=0 (part of the Uniform spanning y=-1..=0) are
/// present in the client tree after patching.
#[test]
fn client_bootstrap_uniform_platform_clipped_to_interest_bounds_preserves_all_y_chunks() {
    // Server's platform: Uniform(Grid Floor) spanning y=-1..=0 in chunk coords,
    // similar to MassivePlatforms origin anchor.
    let platform_material = BlockData::simple(1, 0xc45ed1f0);
    let platform_bounds = Aabb4i::from_lattice_bounds([-4, -1, -4, -4], [3, 0, 3, 3], 0);
    let server_subtree = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(platform_material.clone()),
        generator_version_hash: 0,
    };

    // Client's interest bounds centered at chunk (0, 0, 0, 0) with radius 2.
    let interest_bounds = Aabb4i::from_lattice_bounds([-2, -2, -2, -2], [2, 2, 2, 2], 0);

    // Simulate: server sends the full platform subtree, client clips to interest bounds.
    let clipped = slice_non_empty_region_core_in_bounds(&server_subtree, interest_bounds);

    // Apply to client tree.
    let mut client_tree = RegionChunkTree::new();
    client_tree.splice_non_empty_core_in_bounds(interest_bounds, &clipped);

    // Both y=-1 and y=0 should be present (they're within both platform and interest bounds).
    let mut missing_y_neg1 = Vec::new();
    let mut missing_y_0 = Vec::new();
    for x in -2..=2 {
        for z in -2..=2 {
            for w in -2..=2 {
                // Check y=-1 (should be in platform)
                let payload = client_tree.chunk_payload(key(x, -1, z, w));
                if payload.is_none() {
                    missing_y_neg1.push((x, z, w));
                }
                // Check y=0 (should be in platform)
                let payload = client_tree.chunk_payload(key(x, 0, z, w));
                if payload.is_none() {
                    missing_y_0.push((x, z, w));
                }
            }
        }
    }
    assert!(
        missing_y_neg1.is_empty(),
        "chunks at y=-1 missing after bootstrap: {:?} (out of 125)",
        missing_y_neg1
    );
    assert!(
        missing_y_0.is_empty(),
        "chunks at y=0 missing after bootstrap: {:?} (out of 125)",
        missing_y_0
    );

    // Also verify y=1 and y=-2 are NOT present (outside platform bounds).
    let y1_present = client_tree.chunk_payload(key(0, 1, 0, 0));
    assert!(
        y1_present.is_none(),
        "chunk at y=1 should NOT be present (outside platform)"
    );
    let y_neg2_present = client_tree.chunk_payload(key(0, -2, 0, 0));
    assert!(
        y_neg2_present.is_none(),
        "chunk at y=-2 should NOT be present (outside platform)"
    );
}

/// Simulates the case where the server splits the interest bounds into two
/// patches (y-negative half and y-positive half) and the client applies them
/// sequentially. The Uniform platform spans y=-1..=0 so each half gets one
/// y-slice. Verifies both survive splicing.
#[test]
fn client_bootstrap_split_patches_both_y_halves_preserve_uniform_platform() {
    let platform_material = BlockData::simple(1, 0xc45ed1f0);
    let platform_bounds = Aabb4i::from_lattice_bounds([-4, -1, -4, -4], [3, 0, 3, 3], 0);
    let server_subtree = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(platform_material.clone()),
        generator_version_hash: 0,
    };

    let mut client_tree = RegionChunkTree::new();

    // Patch 1: y-negative half of interest bounds (y=-2..=-1)
    let patch1_bounds = Aabb4i::from_lattice_bounds([-2, -2, -2, -2], [2, -1, 2, 2], 0);
    let clipped1 = slice_non_empty_region_core_in_bounds(&server_subtree, patch1_bounds);
    client_tree.splice_non_empty_core_in_bounds(patch1_bounds, &clipped1);

    // Verify y=-1 is present after first patch
    assert!(
        client_tree.chunk_payload(key(0, -1, 0, 0)).is_some(),
        "y=-1 chunk should be present after first patch"
    );

    // Patch 2: y-positive half of interest bounds (y=0..=2)
    let patch2_bounds = Aabb4i::from_lattice_bounds([-2, 0, -2, -2], [2, 2, 2, 2], 0);
    let clipped2 = slice_non_empty_region_core_in_bounds(&server_subtree, patch2_bounds);
    client_tree.splice_non_empty_core_in_bounds(patch2_bounds, &clipped2);

    // Verify BOTH y=-1 and y=0 are present after second patch
    assert!(
        client_tree.chunk_payload(key(0, -1, 0, 0)).is_some(),
        "y=-1 chunk should STILL be present after second patch (must not be clobbered)"
    );
    assert!(
        client_tree.chunk_payload(key(0, 0, 0, 0)).is_some(),
        "y=0 chunk should be present after second patch"
    );

    // Spot-check: both should return the platform material
    let (payload_neg1, _) = client_tree.chunk_payload(key(0, -1, 0, 0)).unwrap();
    assert_eq!(
        payload_neg1.block_at(0),
        platform_material,
        "y=-1 payload should be platform material"
    );
    let (payload_0, _) = client_tree.chunk_payload(key(0, 0, 0, 0)).unwrap();
    assert_eq!(
        payload_0.block_at(0),
        platform_material,
        "y=0 payload should be platform material"
    );
}

#[test]
fn block_data_serde_i48f16_bounds_roundtrip() {
    use crate::shared::spatial::ChunkCoord;

    // ── Part 1: Individual ChunkCoord round-trips ──

    let test_values: &[i32] = &[-18, -1, 0, 17];

    for &v in test_values {
        let coord = ChunkCoord::from_num(v);

        // Round-trip via raw bits (i64)
        let bits = coord.to_bits();
        let bits_bytes = postcard::to_stdvec(&bits).unwrap();
        let bits_back: i64 = postcard::from_bytes(&bits_bytes).unwrap();
        assert_eq!(
            bits, bits_back,
            "i64 bits round-trip failed for value {v}: original bits {bits:#x}, got {bits_back:#x}"
        );
        let coord_from_bits = ChunkCoord::from_bits(bits_back);
        assert_eq!(
            coord, coord_from_bits,
            "ChunkCoord from round-tripped bits differs for value {v}"
        );

        // Round-trip the ChunkCoord directly via serde
        let coord_bytes = postcard::to_stdvec(&coord).unwrap();
        let coord_back: ChunkCoord = postcard::from_bytes(&coord_bytes).unwrap();
        assert_eq!(
            coord, coord_back,
            "Direct ChunkCoord serde round-trip failed for value {v}: \
             original bits {:#x}, deserialized bits {:#x}",
            coord.to_bits(),
            coord_back.to_bits()
        );
    }

    // ── Part 2: RegionTreeCore with Aabb4i bounds round-trip ──

    let bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);
    let core = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Uniform(BlockData::simple(1, 0xc45ed1f0)),
        generator_version_hash: 0,
    };

    let bytes = postcard::to_stdvec(&core).unwrap();
    let core_back: RegionTreeCore = postcard::from_bytes(&bytes).unwrap();

    // Check bounds survived round-trip
    assert_eq!(
        core.bounds, core_back.bounds,
        "RegionTreeCore bounds mismatch after serde round-trip!\n\
         Original min bits: {:?}\n\
         Original max bits: {:?}\n\
         Deserialized min bits: {:?}\n\
         Deserialized max bits: {:?}",
        core.bounds.min.map(|c| c.to_bits()),
        core.bounds.max.map(|c| c.to_bits()),
        core_back.bounds.min.map(|c| c.to_bits()),
        core_back.bounds.max.map(|c| c.to_bits()),
    );

    // Explicitly check specific max values (world-space half-open: (lattice_max+1)*8)
    assert_eq!(
        core_back.bounds.max[0],
        ChunkCoord::from_num(144),
        "max[0] should be 144 ((17+1)*8), got bits {:#x}",
        core_back.bounds.max[0].to_bits()
    );
    assert_eq!(
        core_back.bounds.max[1],
        ChunkCoord::from_num(8),
        "max[1] should be 8 ((0+1)*8), got bits {:#x}",
        core_back.bounds.max[1].to_bits()
    );
    assert_eq!(
        core_back.bounds.max[2],
        ChunkCoord::from_num(144),
        "max[2] should be 144 ((17+1)*8), got bits {:#x}",
        core_back.bounds.max[2].to_bits()
    );
    assert_eq!(
        core_back.bounds.max[3],
        ChunkCoord::from_num(144),
        "max[3] should be 144 ((17+1)*8), got bits {:#x}",
        core_back.bounds.max[3].to_bits()
    );

    // Also verify the kind survived
    assert_eq!(core.kind, core_back.kind, "RegionNodeKind mismatch after round-trip");

    // Full equality check
    assert_eq!(core, core_back, "Full RegionTreeCore equality failed after round-trip");
}

/// Verify that no node in the tree has fractional (non-integer) bounds.
/// This catches the bug where `step_for_kind` / `subtract_aabb` produce
/// bounds like -0.5 instead of integer-aligned coordinates.
fn assert_no_fractional_bounds(core: &RegionTreeCore, context: &str) {
    fn check_node(node: &RegionTreeCore, path: &str) {
        for axis in 0..4 {
            let min_val = node.bounds.min[axis];
            let max_val = node.bounds.max[axis];
            assert_eq!(
                min_val.frac().to_bits(),
                0,
                "{path}: bounds.min[{axis}] = {} has fractional part (bits={:#x})",
                min_val,
                min_val.to_bits()
            );
            assert_eq!(
                max_val.frac().to_bits(),
                0,
                "{path}: bounds.max[{axis}] = {} has fractional part (bits={:#x})",
                max_val,
                max_val.to_bits()
            );
        }
        if let RegionNodeKind::Branch(children) = &node.kind {
            for (i, child) in children.iter().enumerate() {
                check_node(child, &format!("{path}/branch[{i}]"));
            }
        }
    }
    if let Some(root) = std::iter::once(core).next() {
        check_node(root, context);
    }
}

fn assert_tree_no_fractional_bounds(tree: &RegionChunkTree, context: &str) {
    if let Some(root) = tree.root() {
        assert_no_fractional_bounds(root, context);
    }
}

/// Simulate the MassivePlatforms generator building a tree:
/// 1. Insert platform Uniform
/// 2. Splice procgen ChunkArray that overlaps with platform on y=0
/// Verify that the splice does not produce fractional bounds.
#[test]
fn procgen_splice_over_platform_uniform_preserves_integer_bounds() {
    let stone = BlockData::simple(0, 11);
    let wood = BlockData::simple(0, 4);

    // Step 1: Platform Uniform (matches origin anchor from MassivePlatforms)
    let platform_bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };

    let mut tree = RegionChunkTree::new();
    let _ = tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);
    assert_tree_no_fractional_bounds(&tree, "after platform splice");

    // Step 2: Procgen structure — a small ChunkArray overlapping with the platform at y=0
    // (procgen structures sit above the platform but may extend into y=0)
    let procgen_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 2, 2, 2], 0);
    let procgen_chunk_palette = vec![
        ChunkPayload::Empty,
        ChunkPayload::Uniform(1), // wood material at palette index 1
    ];
    // 3x3x3x3 = 81 cells, most empty, a few have wood
    let cell_count = 3 * 3 * 3 * 3;
    let mut indices = vec![0u16; cell_count];
    indices[0] = 1; // one wood block at (0,0,0,0) relative
    indices[1] = 1; // wood at (1,0,0,0)
    let procgen_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        procgen_bounds,
        procgen_chunk_palette,
        indices,
        Some(0),
        vec![BlockData::AIR, wood.clone()],
        0,
    )
    .expect("valid chunk array");
    let procgen_core = RegionTreeCore {
        bounds: procgen_bounds,
        kind: RegionNodeKind::ChunkArray(procgen_ca),
        generator_version_hash: 0,
    };

    let _ = tree.splice_non_empty_core_in_bounds(procgen_bounds, &procgen_core);
    assert_tree_no_fractional_bounds(&tree, "after procgen splice");

    // Verify platform chunks are still accessible at y=0 outside the procgen area
    for x in -18..0 {
        let payload = tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing after procgen splice"
        );
    }
    for x in 3..=17 {
        let payload = tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing after procgen splice"
        );
    }
    // Verify platform chunks at y=-1 (untouched by procgen)
    for x in -18..=17 {
        let payload = tree.chunk_payload(key(x, -1, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},-1,0,0] missing after procgen splice"
        );
    }
}

/// Simulate the full flow: server generates tree (platform + procgen),
/// client receives it via streaming patches, then lazy_drop shrinks bounds.
/// Verify no fractional bounds at each step.
#[test]
fn full_server_client_flow_with_procgen_no_fractional_bounds() {
    let stone = BlockData::simple(0, 11);
    let wood = BlockData::simple(0, 4);

    // === Server side: build world tree ===
    let platform_bounds = Aabb4i::from_lattice_bounds([-18, -1, -18, -18], [17, 0, 17, 17], 0);

    let mut server_tree = RegionChunkTree::new();
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    let _ = server_tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Add several procgen structures at various positions above the platform
    let procgen_positions: &[[i32; 4]] = &[
        [-5, 1, -3, -3],   // above platform, inside x/z/w range
        [3, 1, 5, 2],      // above platform, inside x/z/w range
        [0, 0, 0, 0],      // overlapping with platform at y=0!
        [-10, 1, -10, -10], // above platform, at edge
    ];

    for (i, pos) in procgen_positions.iter().enumerate() {
        let bounds = Aabb4i::from_lattice_bounds(
            *pos,
            [pos[0] + 1, pos[1] + 1, pos[2] + 1, pos[3] + 1],
            0,
        );
        let cell_count = 2 * 2 * 2 * 2; // 16 cells
        let mut indices = vec![0u16; cell_count];
        indices[0] = 1;
        let ca = ChunkArrayData::from_dense_indices_with_block_palette(
            bounds,
            vec![ChunkPayload::Empty, ChunkPayload::Uniform(1)],
            indices,
            Some(0),
            vec![BlockData::AIR, wood.clone()],
            0,
        )
        .expect("valid chunk array");
        let core = RegionTreeCore {
            bounds,
            kind: RegionNodeKind::ChunkArray(ca),
            generator_version_hash: 0,
        };
        let _ = server_tree.splice_non_empty_core_in_bounds(bounds, &core);
        assert_tree_no_fractional_bounds(
            &server_tree,
            &format!("server: after procgen[{i}] at {pos:?}"),
        );
    }

    // === Client side: receive patches ===
    let _interest_bounds = Aabb4i::from_lattice_bounds([-10, -5, -10, -10], [10, 5, 10, 10], 0);

    // Simulate split_bounds_for_streaming: split interest into halves along x
    let left_bounds = Aabb4i::from_lattice_bounds([-10, -5, -10, -10], [-1, 5, 10, 10], 0);
    let right_bounds = Aabb4i::from_lattice_bounds([0, -5, -10, -10], [10, 5, 10, 10], 0);

    let left_subtree = server_tree.slice_non_empty_core_in_bounds(left_bounds);
    let right_subtree = server_tree.slice_non_empty_core_in_bounds(right_bounds);

    let mut client_tree = RegionChunkTree::new();

    // Apply left patch
    let _ = client_tree.splice_non_empty_core_in_bounds(left_bounds, &left_subtree);
    assert_tree_no_fractional_bounds(&client_tree, "client: after left patch");

    // Apply right patch
    let _ = client_tree.splice_non_empty_core_in_bounds(right_bounds, &right_subtree);
    assert_tree_no_fractional_bounds(&client_tree, "client: after right patch");

    // Verify platform chunks at y=0 are accessible
    for x in -10..=10 {
        let payload = client_tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing on client after patches"
        );
    }
    for x in -10..=10 {
        let payload = client_tree.chunk_payload(key(x, -1, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},-1,0,0] missing on client after patches"
        );
    }

    // === Client shrinks interest (player moved) ===
    let shrunk_bounds = Aabb4i::from_lattice_bounds([-5, -3, -5, -5], [5, 3, 5, 5], 0);
    let _ = client_tree.lazy_drop_outside_bounds(shrunk_bounds, 1000);
    assert_tree_no_fractional_bounds(&client_tree, "client: after lazy_drop");

    // Verify platform chunks within shrunk bounds still accessible
    for x in -5..=5 {
        let payload = client_tree.chunk_payload(key(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing after lazy_drop"
        );
        let payload_neg_y = client_tree.chunk_payload(key(x, -1, 0, 0));
        assert!(
            payload_neg_y.is_some(),
            "platform chunk [{x},-1,0,0] missing after lazy_drop"
        );
    }
}

#[test]
fn mixed_scale_lazy_drop_does_not_corrupt_uniform_bounds() {
    use crate::shared::spatial::chunk_key_from_lattice;

    // Simulate a tree with a Uniform platform and a fine-scale (scale_exp=-1) chunk.
    // Before the fix, step_for_kind on the Branch used min() which returned step=0.5,
    // causing ensure_binary_children to split the Uniform at half-integer boundaries.
    let mut tree = RegionChunkTree::new();
    let stone = BlockData::simple(0, 1);
    let platform_bounds = Aabb4i::from_lattice_bounds([-8, -1, -8, -8], [7, 0, 7, 7], 0);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);

    // Insert a fine-scale chunk above the platform.
    // This creates a Branch with both Uniform (step=1) and ChunkArray(scale=-1, step=0.5).
    let fine_key = chunk_key_from_lattice([0, 2, 0, 0], -1);
    let fine_block = BlockData::simple(0, 42);
    tree.set_chunk_at_scale(
        fine_key,
        Some(ResolvedChunkPayload::uniform(fine_block.clone())),
        -1,
    );

    // Now call lazy_drop_outside_node with bounds that only partially cover the platform.
    // This forces ensure_binary_children to split nodes. Before the fix, the split would
    // use step=0.5 (from the fine-scale child), corrupting the Uniform's bounds.
    let keep_bounds = Aabb4i::from_lattice_bounds([-4, -1, -4, -4], [4, 2, 4, 4], 0);
    tree.lazy_drop_outside_bounds(keep_bounds, 100);

    // Verify no fractional bounds in the tree.
    assert_tree_no_fractional_bounds(&tree, "after_mixed_scale_lazy_drop");

    // Verify that platform chunks at y=0 are still accessible.
    for x in -4..=4 {
        let payload = tree.chunk_payload(chunk_key_i32(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing after mixed-scale lazy_drop"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-scale block placement integrity tests
// ---------------------------------------------------------------------------

/// Helper: place a single block at a world position and scale, mimicking the
/// server's `apply_voxel_edit_at_scale` flow but operating directly on a
/// `RegionChunkTree`.
fn place_block_at_scale(
    tree: &mut RegionChunkTree,
    position: [i32; 4],
    block: BlockData,
    scale_exp: i8,
) {
    use crate::shared::spatial::{step_for_scale, ChunkCoord};
    use crate::shared::voxel::{world_to_chunk_at_scale, CHUNK_SIZE, CHUNK_VOLUME};

    let (chunk_key, voxel_index) =
        world_to_chunk_at_scale(position[0], position[1], position[2], position[3], scale_exp);

    // Read existing blocks via BVH point queries (scale-agnostic).
    let step = step_for_scale(scale_exp);
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    let origin = chunk_key.map(|k| k.saturating_mul(cs));
    let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];
    for lw in 0..CHUNK_SIZE {
        for lz in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lx in 0..CHUNK_SIZE {
                    let pos = [
                        origin[0] + step * ChunkCoord::from_num(lx as i32),
                        origin[1] + step * ChunkCoord::from_num(ly as i32),
                        origin[2] + step * ChunkCoord::from_num(lz as i32),
                        origin[3] + step * ChunkCoord::from_num(lw as i32),
                    ];
                    let idx = lw * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE
                        + lz * CHUNK_SIZE * CHUNK_SIZE
                        + ly * CHUNK_SIZE
                        + lx;
                    blocks[idx] = tree.block_at(pos);
                }
            }
        }
    }
    blocks[voxel_index] = block;

    let desired_payload = if blocks.iter().all(|b| b.is_air()) {
        None
    } else {
        ResolvedChunkPayload::from_dense_blocks(&blocks).ok()
    };

    tree.set_chunk_at_scale(chunk_key, desired_payload, scale_exp);
}

/// Helper: validate structural tree invariants (branch non-overlap, containment).
/// Does NOT check world-space overlap or fractional bounds, since those are
/// expected when mixing scales in the same region.
fn assert_tree_structure(tree: &RegionChunkTree, context: &str) {
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
        // World-space overlap and fractional bounds are not checked here;
        // mixing scales intentionally produces sub-integer bounds.
        let _ = context;
    }
}

/// Stricter variant: also checks world-space non-overlap and integer bounds.
/// Only use for single-scale or known non-overlapping scenarios.
fn assert_tree_integrity_strict(tree: &RegionChunkTree, context: &str) {
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
        validate_region_core_world_space_non_overlapping(root)
            .unwrap_or_else(|e| panic!("{context}: world-space overlap: {e}"));
        assert_no_fractional_bounds(root, context);
    }
}

/// Helper: read back a block at a world position using BVH point query (scale-agnostic).
fn read_block_at_scale(
    tree: &RegionChunkTree,
    position: [i32; 4],
    _scale_exp: i8,
) -> BlockData {
    use crate::shared::spatial::ChunkCoord;
    let pos = [
        ChunkCoord::from_num(position[0]),
        ChunkCoord::from_num(position[1]),
        ChunkCoord::from_num(position[2]),
        ChunkCoord::from_num(position[3]),
    ];
    tree.block_at(pos)
}

#[test]
fn place_single_block_at_scale_zero() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 42);
    place_block_at_scale(&mut tree, [3, 0, 5, 1], block.clone(), 0);

    assert_tree_integrity_strict(&tree, "single_block_s0");
    let read = read_block_at_scale(&tree, [3, 0, 5, 1], 0);
    assert_eq!(read.namespace, block.namespace);
    assert_eq!(read.block_type, block.block_type);
}

#[test]
fn place_single_block_at_negative_scale() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 7);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), -1);

    assert_tree_structure(&tree, "single_block_s-1");
    let read = read_block_at_scale(&tree, [0, 0, 0, 0], -1);
    assert_eq!(read.namespace, block.namespace);
    assert_eq!(read.block_type, block.block_type);
}

#[test]
fn place_single_block_at_positive_scale() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 99);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), 1);

    assert_tree_structure(&tree, "single_block_s1");
    let read = read_block_at_scale(&tree, [0, 0, 0, 0], 1);
    assert_eq!(read.namespace, block.namespace);
    assert_eq!(read.block_type, block.block_type);
}

#[test]
fn place_blocks_at_multiple_scales_same_region() {
    let mut tree = RegionChunkTree::new();
    let stone = BlockData::simple(0, 1);
    let brick = BlockData::simple(0, 2);
    let glass = BlockData::simple(0, 3);

    // Scale 0 block at origin
    place_block_at_scale(&mut tree, [0, 0, 0, 0], stone.clone(), 0);
    assert_tree_structure(&tree, "multi_scale_after_s0");

    // Scale 1 block nearby (occupies 2x2x2x2 world units starting at 16,0,0,0)
    place_block_at_scale(&mut tree, [16, 0, 0, 0], brick.clone(), 1);
    assert_tree_structure(&tree, "multi_scale_after_s1");

    // Scale -1 block (occupies 0.5 world units, at world 8,0,0,0)
    place_block_at_scale(&mut tree, [8, 0, 0, 0], glass.clone(), -1);
    assert_tree_structure(&tree, "multi_scale_after_s-1");

    // All three readable
    let r0 = read_block_at_scale(&tree, [0, 0, 0, 0], 0);
    assert_eq!(r0.block_type, stone.block_type);

    let r1 = read_block_at_scale(&tree, [16, 0, 0, 0], 1);
    assert_eq!(r1.block_type, brick.block_type);

    let r_neg = read_block_at_scale(&tree, [8, 0, 0, 0], -1);
    assert_eq!(r_neg.block_type, glass.block_type);
}

#[test]
fn place_and_remove_block_at_each_scale() {
    for scale_exp in -3i8..=3 {
        let mut tree = RegionChunkTree::new();
        let block = BlockData::simple(1, 50);
        place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), scale_exp);
        assert_tree_structure(&tree, &format!("place_remove_s{scale_exp}_after_place"));

        let read = read_block_at_scale(&tree, [0, 0, 0, 0], scale_exp);
        assert_eq!(read.block_type, block.block_type, "scale_exp={scale_exp}: read mismatch");

        // Remove by placing air
        place_block_at_scale(&mut tree, [0, 0, 0, 0], BlockData::AIR, scale_exp);
        assert_tree_structure(&tree, &format!("place_remove_s{scale_exp}_after_remove"));
    }
}

#[test]
fn many_blocks_at_same_negative_scale_fill_chunk() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(0, 5);

    // Fill a full chunk at scale -1 (8^4 = 4096 half-size voxels).
    // At scale -1, world coord 0 → scaled lattice 0, world coord 1 → scaled lattice 2, etc.
    // So fill voxels at world positions [0..8) in each axis, stepping by 1 (but each maps
    // to a unique half-scale voxel).
    for w in 0..4 {
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    place_block_at_scale(&mut tree, [x, y, z, w], block.clone(), -1);
                }
            }
        }
    }

    assert_tree_structure(&tree, "fill_chunk_s-1");

    // Spot-check several positions
    for pos in [[0, 0, 0, 0], [1, 2, 3, 0], [3, 3, 3, 3]] {
        let read = read_block_at_scale(&tree, pos, -1);
        assert_eq!(read.block_type, block.block_type, "at {pos:?}");
    }
}

#[test]
fn place_block_at_scale_2_roundtrips() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(0, 77);

    // scale_exp=2: each voxel is 4 world units, chunk spans 32 world units.
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), 2);
    assert_tree_structure(&tree, "s2_roundtrip");

    let read = read_block_at_scale(&tree, [0, 0, 0, 0], 2);
    assert_eq!(read.block_type, block.block_type);
}

#[test]
fn place_block_at_negative_world_coords_all_scales() {
    for scale_exp in -2i8..=2 {
        let mut tree = RegionChunkTree::new();
        let block = BlockData::simple(1, 33);
        place_block_at_scale(&mut tree, [-5, -3, -7, -1], block.clone(), scale_exp);
        assert_tree_structure(&tree, &format!("neg_coords_s{scale_exp}"));

        let read = read_block_at_scale(&tree, [-5, -3, -7, -1], scale_exp);
        assert_eq!(
            read.block_type, block.block_type,
            "scale_exp={scale_exp}: block at negative coords lost"
        );
    }
}

#[test]
fn interleaved_multi_scale_placements_preserve_all_blocks() {
    let mut tree = RegionChunkTree::new();
    let mut placements = Vec::new();

    // Place blocks at alternating scales across a range of positions.
    // Spacing must be large enough that chunk extents at scale 2 (32 world units)
    // never overlap with adjacent placements.
    for i in 0..20 {
        let scale_exp = (i % 5) as i8 - 2; // -2, -1, 0, 1, 2
        let x = i * 64; // 64 units apart — enough for even scale 2 (chunk = 32)
        let block = BlockData::simple(0, (i + 1) as u32);
        place_block_at_scale(&mut tree, [x, 0, 0, 0], block.clone(), scale_exp);
        placements.push((x, scale_exp, block));
        assert_tree_structure(&tree, &format!("interleaved_step_{i}"));
    }

    // Read all back
    for (x, scale_exp, expected) in &placements {
        let read = read_block_at_scale(&tree, [*x, 0, 0, 0], *scale_exp);
        assert_eq!(
            read.block_type, expected.block_type,
            "lost block at x={x} scale={scale_exp}"
        );
    }
}

#[test]
fn place_on_uniform_platform_at_different_scales_preserves_platform() {
    let mut tree = RegionChunkTree::new();
    let stone = BlockData::simple(0, 1);

    // Create a uniform platform
    let platform_bounds = Aabb4i::from_lattice_bounds([-4, -1, -4, -4], [3, 0, 3, 3], 0);
    let platform_core = RegionTreeCore {
        bounds: platform_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    tree.splice_non_empty_core_in_bounds(platform_bounds, &platform_core);
    assert_tree_integrity_strict(&tree, "platform_before_edits");

    // Place blocks at various scales on top of the platform
    let brick = BlockData::simple(0, 5);
    place_block_at_scale(&mut tree, [0, 1, 0, 0], brick.clone(), 0);
    assert_tree_structure(&tree, "platform_after_s0_edit");

    let glass = BlockData::simple(0, 6);
    place_block_at_scale(&mut tree, [0, 2, 0, 0], glass.clone(), -1);
    assert_tree_structure(&tree, "platform_after_s-1_edit");

    let big = BlockData::simple(0, 7);
    place_block_at_scale(&mut tree, [0, 2, 0, 0], big.clone(), 1);
    assert_tree_structure(&tree, "platform_after_s1_edit");

    // Platform chunks should still be accessible
    for x in -4..=3 {
        let payload = tree.chunk_payload(chunk_key_i32(x, 0, 0, 0));
        assert!(
            payload.is_some(),
            "platform chunk [{x},0,0,0] missing after multi-scale edits"
        );
    }
}

#[test]
fn scale_exp_preserved_in_block_palette_roundtrip() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(1, 10).at_scale(-2);

    // Place the block — the tree should store it with scale_exp = -2.
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), -2);
    let read = read_block_at_scale(&tree, [0, 0, 0, 0], -2);
    assert_eq!(read.scale_exp, -2, "scale_exp lost in roundtrip");
    assert_eq!(read.block_type, 10);
}

#[test]
fn adjacent_chunks_at_same_non_zero_scale_stay_distinct() {
    use crate::shared::spatial::chunk_key_from_lattice;

    let mut tree = RegionChunkTree::new();
    let block_a = BlockData::simple(0, 10);
    let block_b = BlockData::simple(0, 20);

    // Two adjacent scale -1 chunks. At scale -1, chunk lattice [0,0,0,0] and [1,0,0,0]
    // map to fixed-point [0,0,0,0] and [0.5,0,0,0].
    let key_a = chunk_key_from_lattice([0, 0, 0, 0], -1);
    let key_b = chunk_key_from_lattice([1, 0, 0, 0], -1);

    tree.set_chunk_at_scale(key_a, Some(ResolvedChunkPayload::uniform(block_a.clone())), -1);
    tree.set_chunk_at_scale(key_b, Some(ResolvedChunkPayload::uniform(block_b.clone())), -1);
    assert_tree_structure(&tree, "adjacent_s-1");

    let (read_a, _) = tree.chunk_payload(key_a).unwrap();
    let (read_b, _) = tree.chunk_payload(key_b).unwrap();
    assert_eq!(read_a.uniform_block().unwrap().block_type, 10);
    assert_eq!(read_b.uniform_block().unwrap().block_type, 20);
}

#[test]
fn world_to_chunk_at_scale_coordinate_consistency() {
    use crate::shared::voxel::world_to_chunk_at_scale;
    use crate::shared::spatial::chunk_key_from_lattice;

    // Scale 0: world (0,0,0,0) → chunk (0,0,0,0), voxel 0
    let (k, idx) = world_to_chunk_at_scale(0, 0, 0, 0, 0);
    assert_eq!(k, chunk_key_i32(0, 0, 0, 0));
    assert_eq!(idx, 0);

    // Scale 0: world (7,7,7,7) → same chunk, last voxel
    let (k, idx) = world_to_chunk_at_scale(7, 7, 7, 7, 0);
    assert_eq!(k, chunk_key_i32(0, 0, 0, 0));
    assert_eq!(idx, 7 + 8 * (7 + 8 * (7 + 8 * 7)));

    // Scale 0: world (8,0,0,0) → chunk (1,0,0,0), voxel 0
    let (k, _) = world_to_chunk_at_scale(8, 0, 0, 0, 0);
    assert_eq!(k, chunk_key_i32(1, 0, 0, 0));

    // Scale 1: world (0,0,0,0) → lattice (0,0,0,0) at scale 1
    let (k, idx) = world_to_chunk_at_scale(0, 0, 0, 0, 1);
    assert_eq!(k, chunk_key_from_lattice([0, 0, 0, 0], 1));
    assert_eq!(idx, 0);

    // Scale 1: world (2,0,0,0) → scaled lattice x=1 → still chunk 0, voxel index 1
    let (k, idx) = world_to_chunk_at_scale(2, 0, 0, 0, 1);
    assert_eq!(k, chunk_key_from_lattice([0, 0, 0, 0], 1));
    assert_eq!(idx, 1);

    // Scale -1: world (0,0,0,0) → scaled lattice x=0
    let (k, idx) = world_to_chunk_at_scale(0, 0, 0, 0, -1);
    assert_eq!(k, chunk_key_from_lattice([0, 0, 0, 0], -1));
    assert_eq!(idx, 0);

    // Scale -1: world (1,0,0,0) → scaled lattice x=2 → voxel index 2
    let (k, idx) = world_to_chunk_at_scale(1, 0, 0, 0, -1);
    assert_eq!(k, chunk_key_from_lattice([0, 0, 0, 0], -1));
    assert_eq!(idx, 2);

    // Scale -1: world (4,0,0,0) → scaled lattice x=8 → chunk lattice 1
    let (k, _) = world_to_chunk_at_scale(4, 0, 0, 0, -1);
    assert_eq!(k, chunk_key_from_lattice([1, 0, 0, 0], -1));

    // Negative world: scale 0, world (-1,0,0,0) → chunk (-1,0,0,0), last x voxel
    let (k, idx) = world_to_chunk_at_scale(-1, 0, 0, 0, 0);
    assert_eq!(k, chunk_key_i32(-1, 0, 0, 0));
    assert_eq!(idx, 7); // -1 mod 8 = 7 (Euclidean)
}

#[test]
fn randomized_multi_scale_placements_preserve_integrity() {
    use crate::shared::voxel::world_to_chunk_at_scale;

    let mut rng = TestRng::new(0xDEAD_BEEF_CAFE);
    let mut tree = RegionChunkTree::new();
    let mut placed = Vec::new();

    for i in 0..100 {
        let scale_exp = rng.next_inclusive_i32(-2, 2) as i8;
        let x = rng.next_inclusive_i32(-20, 20);
        let y = rng.next_inclusive_i32(-5, 5);
        let z = rng.next_inclusive_i32(-20, 20);
        let w = rng.next_inclusive_i32(-20, 20);
        let block_type = (i + 1) as u32;
        let block = BlockData::simple(0, block_type);

        place_block_at_scale(&mut tree, [x, y, z, w], block.clone(), scale_exp);
        placed.push(([x, y, z, w], scale_exp, block));

        // Check structural integrity every 10 steps
        if i % 10 == 9 {
            assert_tree_structure(&tree, &format!("random_step_{i}"));
        }
    }

    assert_tree_structure(&tree, "random_final");

    // Build the world-space chunk bounds for each placement.
    let chunk_regions: Vec<Aabb4i> = placed
        .iter()
        .map(|(pos, scale, _)| {
            let (ck, _) = world_to_chunk_at_scale(pos[0], pos[1], pos[2], pos[3], *scale);
            Aabb4i::chunk_world_bounds(ck, *scale)
        })
        .collect();

    // Verify data for placements whose chunk was NOT later intersected by
    // any other placement. The tree's contract: set_chunk_at_scale overwrites
    // the full spatial region of the chunk, and carving existing data at a
    // different scale may lose non-grid-aligned fragments. Cross-scale data
    // preservation is the world_field's responsibility (via rechunking), not
    // the tree's. Here we only verify chunks that survived completely intact.
    let mut latest: HashMap<([i32; 4], i8), (usize, BlockData)> = HashMap::new();
    for (i, (pos, scale, block)) in placed.iter().enumerate() {
        latest.insert((*pos, *scale), (i, block.clone()));
    }
    let mut verified = 0;
    for ((pos, scale), (last_idx, expected)) in &latest {
        let my_chunk = &chunk_regions[*last_idx];
        let chunk_was_carved = chunk_regions[last_idx + 1..]
            .iter()
            .any(|later| later.intersects(my_chunk));
        if chunk_was_carved {
            continue;
        }
        let read = read_block_at_scale(&tree, *pos, *scale);
        assert_eq!(
            read.block_type, expected.block_type,
            "mismatch at pos={pos:?} scale={scale}"
        );
        verified += 1;
    }
    // Sanity: we should verify a meaningful number of placements.
    assert!(verified > 10, "only verified {verified} placements — test is too lax");
}

#[test]
fn mixed_scale_edits_then_remove_all_leaves_empty_tree() {
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(0, 1);

    let edits: Vec<([i32; 4], i8)> = vec![
        ([0, 0, 0, 0], 0),
        ([0, 0, 0, 0], -1),
        ([0, 0, 0, 0], 1),
        ([10, 0, 0, 0], -2),
        ([-5, -5, -5, -5], 2),
    ];

    for (pos, scale) in &edits {
        place_block_at_scale(&mut tree, *pos, block.clone(), *scale);
    }
    assert_tree_structure(&tree, "mixed_before_remove");

    // Remove all
    for (pos, scale) in &edits {
        place_block_at_scale(&mut tree, *pos, BlockData::AIR, *scale);
    }
    assert_tree_structure(&tree, "mixed_after_remove");
}

#[test]
fn overwrite_block_at_same_position_different_content() {
    for scale_exp in [-2i8, -1, 0, 1, 2] {
        let mut tree = RegionChunkTree::new();
        let block_a = BlockData::simple(0, 10);
        let block_b = BlockData::simple(0, 20);

        place_block_at_scale(&mut tree, [0, 0, 0, 0], block_a, scale_exp);
        let read = read_block_at_scale(&tree, [0, 0, 0, 0], scale_exp);
        assert_eq!(read.block_type, 10, "scale={scale_exp} initial");

        place_block_at_scale(&mut tree, [0, 0, 0, 0], block_b, scale_exp);
        assert_tree_structure(&tree, &format!("overwrite_s{scale_exp}"));
        let read = read_block_at_scale(&tree, [0, 0, 0, 0], scale_exp);
        assert_eq!(read.block_type, 20, "scale={scale_exp} after overwrite");
    }
}

#[test]
fn scale_exp_3_extreme_range_roundtrip() {
    // scale_exp = 3: voxels are 8 world units, chunk spans 64 world units
    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(0, 88);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), 3);
    assert_tree_structure(&tree, "s3_place");

    let read = read_block_at_scale(&tree, [0, 0, 0, 0], 3);
    assert_eq!(read.block_type, 88);

    // scale_exp = -3: voxels are 1/8 world unit, chunk spans 1 world unit
    let mut tree = RegionChunkTree::new();
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block.clone(), -3);
    assert_tree_structure(&tree, "s-3_place");

    let read = read_block_at_scale(&tree, [0, 0, 0, 0], -3);
    assert_eq!(read.block_type, 88);
}

#[test]
fn multiple_blocks_in_same_chunk_at_non_zero_scale() {
    let mut tree = RegionChunkTree::new();

    // At scale 1, world positions 0 and 2 map to voxel indices 0 and 1 in the same chunk.
    let a = BlockData::simple(0, 10);
    let b = BlockData::simple(0, 20);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], a.clone(), 1);
    place_block_at_scale(&mut tree, [2, 0, 0, 0], b.clone(), 1);
    assert_tree_structure(&tree, "same_chunk_s1");

    let read_a = read_block_at_scale(&tree, [0, 0, 0, 0], 1);
    let read_b = read_block_at_scale(&tree, [2, 0, 0, 0], 1);
    assert_eq!(read_a.block_type, 10);
    assert_eq!(read_b.block_type, 20);
}

#[test]
fn consolidation_after_multi_scale_edits_preserves_data() {
    let mut tree = RegionChunkTree::new();

    // Place blocks at different scales, then consolidate and verify.
    let stone = BlockData::simple(0, 1);
    let brick = BlockData::simple(0, 2);
    let glass = BlockData::simple(0, 3);

    place_block_at_scale(&mut tree, [0, 0, 0, 0], stone.clone(), 0);
    place_block_at_scale(&mut tree, [0, 8, 0, 0], brick.clone(), -1);
    place_block_at_scale(&mut tree, [0, 16, 0, 0], glass.clone(), 1);

    assert_tree_structure(&tree, "consolidation_pre");

    // Trigger consolidation by splicing the tree's own content back
    if let Some(root) = tree.root() {
        let bounds = root.bounds;
        let sliced = tree.slice_non_empty_core_in_bounds(bounds);
        let mut tree2 = RegionChunkTree::new();
        tree2.splice_non_empty_core_in_bounds(bounds, &sliced);
        assert_tree_structure(&tree2, "consolidation_post");

        // Data should survive
        let r0 = read_block_at_scale(&tree2, [0, 0, 0, 0], 0);
        assert_eq!(r0.block_type, 1, "stone lost after consolidation");
    }
}

// ── Cross-scale overlap detection tests ─────────────────────────────────

#[test]
fn find_leaf_chunks_detects_same_scale_overlap() {
    use crate::shared::region_tree::tree::chunk_spatial_extent;

    let mut tree = RegionChunkTree::new();
    let block = BlockData::simple(0, 42);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block, -1);

    // Query for the exact spatial extent of the chunk we just placed.
    let query_key = chunk_key_from_lattice([0, 0, 0, 0], -1);
    let query = chunk_spatial_extent(query_key, -1);
    let found = tree.find_leaf_chunks_in_spatial_range(&query);

    assert_eq!(found.len(), 1, "expected exactly one overlapping chunk");
    assert_eq!(found[0].1, -1i8, "expected scale -1");
}

#[test]
fn find_leaf_chunks_detects_cross_scale_overlap() {
    use crate::shared::region_tree::tree::chunk_spatial_extent;

    let mut tree = RegionChunkTree::new();
    // Place a block at scale -1 at origin.
    // Scale -1 chunk at key [0,0,0,0] covers spatial [0, 3.5].
    place_block_at_scale(&mut tree, [0, 0, 0, 0], BlockData::simple(0, 10), -1);

    // Now query for a scale -2 chunk at [0,0,0,0].
    // Scale -2 chunk at key [0,0,0,0] covers spatial [0, 1.75].
    // These overlap in [0, 1.75].
    let query_key = chunk_key_from_lattice([0, 0, 0, 0], -2);
    let query = chunk_spatial_extent(query_key, -2);
    let found = tree.find_leaf_chunks_in_spatial_range(&query);

    assert!(
        !found.is_empty(),
        "scale -1 chunk at [0,0,0,0] should overlap scale -2 query at [0,0,0,0]"
    );
    assert_eq!(found[0].1, -1i8, "overlapping chunk should be at scale -1");
}

#[test]
fn find_leaf_chunks_no_overlap_when_spatially_disjoint() {
    use crate::shared::region_tree::tree::chunk_spatial_extent;

    let mut tree = RegionChunkTree::new();
    // Place at scale -1, chunk at [0,0,0,0], spatial extent [0, 3.5].
    place_block_at_scale(&mut tree, [0, 0, 0, 0], BlockData::simple(0, 10), -1);

    // Query a scale -2 chunk far away: key [8,0,0,0], spatial [8, 9.75].
    // No overlap.
    let query_key = chunk_key_from_lattice([32, 0, 0, 0], -2);
    let query = chunk_spatial_extent(query_key, -2);
    let found = tree.find_leaf_chunks_in_spatial_range(&query);

    assert!(
        found.is_empty(),
        "disjoint chunks should not overlap"
    );
}

/// Simulates a flat world floor (scale-0 chunks) then placing a scale-2 block.
/// The scale-2 chunk covers a 4×4×4×4 region of scale-0 chunks, so the tree
/// must carve the floor data and insert the coarser chunk without creating
/// overlapping siblings.
#[test]
fn set_chunk_at_scale2_over_scale0_floor_non_overlapping() {
    let mut tree = RegionChunkTree::new();

    // Create a floor: a grid of scale-0 chunks at y=0.
    // Scale-0 chunk at key [x,0,z,w] covers world [x*8, x*8+8) etc.
    let floor_block = BlockData::simple(0, 1);
    for w in 0..4 {
        for z in 0..4 {
            for x in 0..4 {
                tree.set_chunk(
                    key(x, 0, z, w),
                    Some(ResolvedChunkPayload::uniform(floor_block.clone())),
                );
            }
        }
    }

    // Verify tree is valid before the multi-scale edit.
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }

    // Place a scale-2 chunk at key [0,1,0,0].
    // Scale-2 step = 4, so world bounds = [0*8, 0*8+8*4) = [0, 32) in each axis.
    // This is above the floor (y=1 in chunk coords = y=8 in world coords).
    let scale2_block = BlockData::simple(0, 42);
    tree.set_chunk_at_scale(
        key(0, 1, 0, 0),
        Some(ResolvedChunkPayload::uniform(scale2_block.clone())),
        2,
    );

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Place a scale-2 chunk that OVERLAPS with existing scale-0 floor chunks.
/// The tree must carve the floor and produce non-overlapping siblings.
#[test]
fn set_chunk_at_scale2_overlapping_scale0_floor() {
    let mut tree = RegionChunkTree::new();

    // Create a floor: scale-0 chunks covering a 4x1x4x4 region at y=0.
    let floor_block = BlockData::simple(0, 1);
    for w in 0..4 {
        for z in 0..4 {
            for x in 0..4 {
                tree.set_chunk(
                    key(x, 0, z, w),
                    Some(ResolvedChunkPayload::uniform(floor_block.clone())),
                );
            }
        }
    }

    // Place a scale-2 chunk at key [0,0,0,0] — this directly overlaps floor chunks.
    // Scale-2 world bounds = [0, 32) in all axes. The floor chunks occupy
    // y ∈ [0, 8), so the scale-2 chunk overlaps them spatially.
    let scale2_block = BlockData::simple(0, 42);
    tree.set_chunk_at_scale(
        key(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(scale2_block.clone())),
        2,
    );

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Multiple scale-2 placements in sequence.
#[test]
fn multiple_scale2_placements_non_overlapping() {
    let mut tree = RegionChunkTree::new();

    // Floor
    let floor_block = BlockData::simple(0, 1);
    for w in 0..4 {
        for z in 0..4 {
            for x in 0..4 {
                tree.set_chunk(
                    key(x, 0, z, w),
                    Some(ResolvedChunkPayload::uniform(floor_block.clone())),
                );
            }
        }
    }

    // Place several scale-2 chunks at different positions
    let block_a = BlockData::simple(0, 10);
    let block_b = BlockData::simple(0, 20);
    let block_c = BlockData::simple(0, 30);

    tree.set_chunk_at_scale(key(0, 0, 0, 0), Some(ResolvedChunkPayload::uniform(block_a)), 2);
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }

    // Adjacent scale-2 chunk (key [4,0,0,0] at scale 2 = world [32, 64))
    tree.set_chunk_at_scale(key(4, 0, 0, 0), Some(ResolvedChunkPayload::uniform(block_b)), 2);
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }

    // Another at a different y level
    tree.set_chunk_at_scale(key(0, 4, 0, 0), Some(ResolvedChunkPayload::uniform(block_c)), 2);
    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Scale-3 placement over scale-0 floor.
#[test]
fn set_chunk_at_scale3_over_scale0_floor() {
    let mut tree = RegionChunkTree::new();

    // Floor — larger to cover more of the scale-3 region
    let floor_block = BlockData::simple(0, 1);
    for w in 0..8 {
        for z in 0..8 {
            for x in 0..8 {
                tree.set_chunk(
                    key(x, 0, z, w),
                    Some(ResolvedChunkPayload::uniform(floor_block.clone())),
                );
            }
        }
    }

    // Scale-3: step = 8, world bounds = [0, 64) in each axis.
    let scale3_block = BlockData::simple(0, 99);
    tree.set_chunk_at_scale(
        key(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(scale3_block)),
        3,
    );

    if let Some(root) = tree.root() {
        assert_tree_non_overlapping(root);
    }
}

/// Inserting a fine-scale chunk into a region covered by a coarser ChunkArray
/// must resample gap pieces instead of dropping them.
#[test]
fn carve_chunk_array_resamples_unaligned_gaps() {
    use crate::shared::spatial::ChunkCoord;

    let mut tree = RegionChunkTree::new();

    // Place two scale-1 blocks in the same chunk. Scale-1 chunk at origin
    // covers [0, 16)^4 in world space. Place one at [0,0,0,0] (cell [0,...])
    // and one at [8,0,0,0] (cell [4,0,0,0] in scale-1 lattice, since 8/2=4).
    let coarse_a = BlockData::simple(0, 10);
    let coarse_b = BlockData::simple(0, 11);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], coarse_a.clone(), 1);
    place_block_at_scale(&mut tree, [8, 0, 0, 0], coarse_b.clone(), 1);

    // Verify both blocks are present.
    assert_eq!(read_block_at_scale(&tree, [0, 0, 0, 0], 1).block_type, 10);
    assert_eq!(read_block_at_scale(&tree, [8, 0, 0, 0], 1).block_type, 11);

    // Now place a scale-0 block at [0,0,0,0]. The scale-0 chunk covers [0, 8)^4,
    // which cuts into the scale-1 chunk [0, 16)^4. The gap piece covering
    // [8, 16) in the x-axis doesn't align to the scale-1 grid and must be
    // resampled to preserve coarse_b.
    let fine_block = BlockData::simple(0, 20);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], fine_block.clone(), 0);

    // The fine block should be readable.
    assert_eq!(
        read_block_at_scale(&tree, [0, 0, 0, 0], 0).block_type, 20,
        "fine block lost after carve"
    );

    // Coarse block B at [8,0,0,0] is OUTSIDE the scale-0 chunk [0,8)^4
    // but inside the original scale-1 chunk. It must survive the carve.
    let read_b = tree.block_at([
        ChunkCoord::from_num(8),
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
    ]);
    assert_eq!(
        read_b.block_type, 11,
        "coarse block outside edit chunk was lost during carve — gap resample failed"
    );

    assert_tree_structure(&tree, "carve_resample_gaps");
}

/// `project_node_to_bounds` on a ChunkArray with bounds that don't align to
/// the ChunkArray's scale must resample instead of returning Empty.
#[test]
fn project_to_unaligned_bounds_resamples() {
    use crate::shared::spatial::ChunkCoord;

    let mut tree = RegionChunkTree::new();

    // Create a scale-1 ChunkArray by placing two blocks at different positions.
    // This ensures the data is non-uniform (can't be collapsed to Uniform).
    let block_a = BlockData::simple(0, 11);
    let block_b = BlockData::simple(0, 22);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], block_a.clone(), 1);
    place_block_at_scale(&mut tree, [2, 0, 0, 0], block_b.clone(), 1);

    // Place a scale-0 block that forces the tree to carve and project.
    // The scale-0 chunk at origin covers [0, 8)^4, which cuts into the
    // scale-1 chunk covering [0, 16)^4.
    let fine_block = BlockData::simple(0, 33);
    place_block_at_scale(&mut tree, [0, 0, 0, 0], fine_block.clone(), 0);

    // Block A at [0,0,0,0] was overwritten by the fine edit.
    let read_fine = read_block_at_scale(&tree, [0, 0, 0, 0], 0);
    assert_eq!(read_fine.block_type, 33, "fine block should overwrite");

    // Block B at [2,0,0,0] should survive — it's in the scale-1 chunk
    // but may be in a gap piece that requires resample.
    let read_b = tree.block_at([
        ChunkCoord::from_num(2),
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
    ]);
    assert_eq!(
        read_b.block_type, 22,
        "block B outside edit region was lost — project_node_to_bounds returned Empty"
    );

    // Verify a position in the scale-1 region beyond the scale-0 chunk
    // (world x=8..16 is definitely outside the scale-0 [0,8) chunk).
    let read_far = tree.block_at([
        ChunkCoord::from_num(8),
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
        ChunkCoord::ZERO,
    ]);
    // This was AIR in the original placement, so should still be AIR.
    assert!(read_far.is_air(), "position [8,0,0,0] should be air");

    assert_tree_structure(&tree, "project_resample");
}
