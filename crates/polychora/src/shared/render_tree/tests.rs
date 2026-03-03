use super::*;
use crate::shared::chunk_payload::ChunkArrayData;
use crate::shared::region_tree::{chunk_key_i32, GeneratorRef};
use crate::shared::spatial::chunk_key_from_lattice;

#[test]
fn from_region_core_drops_procedural_and_keeps_chunk_content() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let uniform_child = RegionTreeCore {
        bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0),
        kind: RegionNodeKind::Uniform(BlockData::simple(0, 7)),
        generator_version_hash: 1,
    };
    let procedural_child = RegionTreeCore {
        bounds: Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0),
        kind: RegionNodeKind::ProceduralRef(GeneratorRef {
            generator_id: "test".into(),
            params: vec![],
            seed: 1,
        }),
        generator_version_hash: 1,
    };
    let src = RegionTreeCore {
        bounds,
        kind: RegionNodeKind::Branch(vec![uniform_child, procedural_child]),
        generator_version_hash: 1,
    };
    let out = from_region_core(&src);
    match out.kind {
        RenderNodeKind::Uniform(ref b) if *b == BlockData::simple(0, 7) => {}
        RenderNodeKind::Branch(ref children) => {
            assert_eq!(children.len(), 1);
            assert!(
                matches!(children[0].kind, RenderNodeKind::Uniform(ref b) if *b == BlockData::simple(0, 7))
            );
        }
        other => panic!("unexpected mapped kind: {other:?}"),
    }
}

#[test]
fn collect_non_empty_chunks_handles_uniform_and_voxel_chunk_array() {
    let uniform_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0);
    let voxel_bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, 11)];
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        voxel_bounds,
        vec![ChunkPayload::Uniform(1)],
        vec![0],
        None,
        block_palette,
        0,
    )
    .expect("chunk array");
    let core = RenderTreeCore {
        bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0),
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: uniform_bounds,
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 5)),
            },
            RenderTreeCore {
                bounds: voxel_bounds,
                kind: RenderNodeKind::VoxelChunkArray(chunk_array),
            },
        ]),
    };
    let chunks = collect_non_empty_chunks_in_bounds(&core, core.bounds);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].0, chunk_key_i32(0, 0, 0, 0));
    assert_eq!(chunks[1].0, chunk_key_i32(1, 0, 0, 0));
}

#[test]
fn bvh_collect_matches_core_collect_for_mixed_leaves() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 0, 0], 0);
    let chunk_array_bounds = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [3, 0, 0, 0], 0);
    let block_palette = vec![BlockData::AIR, BlockData::simple(0, 9)];
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        chunk_array_bounds,
        vec![ChunkPayload::Empty, ChunkPayload::Uniform(1)],
        vec![1, 0],
        None,
        block_palette,
        0,
    )
    .expect("chunk array");
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
            },
            RenderTreeCore {
                bounds: chunk_array_bounds,
                kind: RenderNodeKind::VoxelChunkArray(chunk_array),
            },
        ]),
    };

    let from_core = collect_non_empty_chunk_keys_in_bounds(&core, bounds);
    let bvh = build_bvh_in_bounds(&core, bounds);
    let from_bvh = collect_non_empty_chunk_keys_from_bvh_in_bounds(&bvh, bounds);
    assert_eq!(from_bvh, from_core);
}

#[test]
fn bvh_collect_respects_query_bounds_clipping() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [4, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Uniform(BlockData::simple(0, 3)),
    };
    let bvh = build_bvh_in_bounds(&core, bounds);

    let query = Aabb4i::from_lattice_bounds([2, 0, 0, 0], [3, 0, 0, 0], 0);
    let keys = collect_non_empty_chunk_keys_from_bvh_in_bounds(&bvh, query);
    assert_eq!(
        keys,
        vec![chunk_key_i32(2, 0, 0, 0), chunk_key_i32(3, 0, 0, 0)]
    );
}

#[test]
fn bvh_build_skips_air_uniform_leaves() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Uniform(BlockData::AIR),
    };
    let bvh = build_bvh_in_bounds(&core, bounds);
    assert!(bvh.root.is_none());
    assert!(bvh.nodes.is_empty());
    assert!(bvh.leaves.is_empty());
}

#[test]
fn validate_render_core_rejects_mixed_scale_world_overlap() {
    let coarse_bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0);
    // scale-(-1) lattice [1,0,0,0] = world-space [4, 0, 0, 0]..[8, 0, 0, 0) — overlaps coarse [0,0,0,0]..[8,8,8,8)
    let fine_key = chunk_key_from_lattice([1, 0, 0, 0], -1);
    let fine_bounds = Aabb4i::chunk_world_bounds(fine_key, -1);
    let coarse = ChunkArrayData::from_dense_indices_with_block_palette(
        coarse_bounds,
        vec![ChunkPayload::Uniform(1)],
        vec![0],
        None,
        vec![BlockData::AIR, BlockData::simple(0, 7)],
        0,
    )
    .expect("coarse chunk array");
    let fine = ChunkArrayData::from_dense_indices_with_block_palette(
        fine_bounds,
        vec![ChunkPayload::Uniform(1)],
        vec![0],
        None,
        vec![BlockData::AIR, BlockData::simple(0, 9)],
        -1,
    )
    .expect("fine chunk array");

    let parent_bounds = union_bounds(coarse_bounds, fine_bounds);
    let core = RenderTreeCore {
        bounds: parent_bounds,
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: coarse_bounds,
                kind: RenderNodeKind::VoxelChunkArray(coarse),
            },
            RenderTreeCore {
                bounds: fine_bounds,
                kind: RenderNodeKind::VoxelChunkArray(fine),
            },
        ]),
    };

    assert!(validate_render_core_world_space_non_overlapping(&core).is_err());
}

#[test]
fn sample_chunk_payloads_from_bvh_reads_uniform_and_chunk_array_leaves() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let chunk_array_bounds = Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0);
    let chunk_array = ChunkArrayData::from_dense_indices_with_block_palette(
        chunk_array_bounds,
        vec![ChunkPayload::Uniform(11)],
        vec![0],
        None,
        vec![BlockData::AIR],
        0,
    )
    .expect("chunk array");
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
            },
            RenderTreeCore {
                bounds: chunk_array_bounds,
                kind: RenderNodeKind::VoxelChunkArray(chunk_array),
            },
        ]),
    };
    let bvh = build_bvh_in_bounds(&core, bounds);

    let results_0 = sample_chunk_payloads_from_bvh(&bvh, chunk_key_i32(0, 0, 0, 0));
    assert_eq!(results_0.len(), 1);
    assert_eq!(results_0[0].uniform_block(), Some(&BlockData::simple(0, 7)));
    let results_1 = sample_chunk_payloads_from_bvh(&bvh, chunk_key_i32(1, 0, 0, 0));
    assert_eq!(results_1.len(), 1);
    assert_eq!(results_1[0].payload, ChunkPayload::Uniform(11));
    assert!(sample_chunk_payloads_from_bvh(&bvh, chunk_key_i32(2, 0, 0, 0)).is_empty());
}

#[test]
fn bvh_chunk_mutation_delta_reports_root_change_for_outside_insert() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(vec![RenderTreeCore {
            bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0),
            kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
        }]),
    };
    let mut bvh = build_bvh_in_bounds(&core, bounds);
    let old_root = bvh.root;
    let deltas = apply_chunk_payload_mutations_with_deltas_in_bvh(
        &mut bvh,
        &[(
            chunk_key_i32(2, 0, 0, 0),
            Some(ChunkPayload::Uniform(1)),
            vec![BlockData::AIR, BlockData::simple(0, 9)],
        )],
    )
    .expect("apply mutation");
    assert_eq!(deltas.len(), 1);
    let delta = &deltas[0];
    assert_eq!(delta.key, chunk_key_i32(16, 0, 0, 0));
    assert_eq!(delta.expected_root, old_root);
    assert_eq!(delta.new_root, bvh.root);
    assert!(delta.root_changed());
    assert!(!delta.node_writes.is_empty());
    assert!(!delta.leaf_writes.is_empty());
}

#[test]
fn bvh_chunk_mutation_delta_reports_touched_ancestors() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
            },
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
            },
        ]),
    };
    let mut bvh = build_bvh_in_bounds(&core, bounds);
    let old_root = bvh.root.expect("root");
    let deltas = apply_chunk_payload_mutations_with_deltas_in_bvh(
        &mut bvh,
        &[(chunk_key_i32(0, 0, 0, 0), None, vec![])],
    )
    .expect("apply mutation");
    assert_eq!(deltas.len(), 1);
    let delta = &deltas[0];
    assert_eq!(delta.key, chunk_key_i32(0, 0, 0, 0));
    assert_eq!(delta.expected_root, Some(old_root));
    assert!(
        !delta.node_writes.is_empty() || !delta.freed_node_ids.is_empty() || delta.root_changed()
    );
    assert_eq!(delta.new_root, bvh.root);
}

#[test]
fn bvh_patch_reuses_freed_ids_instead_of_append_only_growth() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [2, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
    };
    let mut bvh = build_bvh_in_bounds(&core, bounds);
    let old_node_capacity = bvh.nodes.len();
    let old_leaf_capacity = bvh.leaves.len();

    let remove = apply_chunk_payload_mutations_with_deltas_in_bvh(
        &mut bvh,
        &[(chunk_key_i32(1, 0, 0, 0), None, vec![])],
    )
    .expect("remove patch");
    assert_eq!(remove.len(), 1);
    assert!(
        !remove[0].freed_node_ids.is_empty() || !remove[0].freed_leaf_ids.is_empty(),
        "expected removal to release at least one id"
    );
    let free_node_count_after_remove = bvh.free_node_count();
    let free_leaf_count_after_remove = bvh.free_leaf_count();
    assert!(free_node_count_after_remove > 0 || free_leaf_count_after_remove > 0);

    let insert = apply_chunk_payload_mutations_with_deltas_in_bvh(
        &mut bvh,
        &[(
            chunk_key_i32(1, 0, 0, 0),
            Some(ChunkPayload::Uniform(1)),
            vec![BlockData::AIR, BlockData::simple(0, 9)],
        )],
    )
    .expect("insert patch");
    assert_eq!(insert.len(), 1);
    let reused_node_slot = insert[0]
        .node_writes
        .iter()
        .any(|(node_id, _)| remove[0].freed_node_ids.contains(node_id));
    let reused_leaf_slot = insert[0]
        .leaf_writes
        .iter()
        .any(|(leaf_id, _)| remove[0].freed_leaf_ids.contains(leaf_id));
    assert!(
        reused_node_slot || reused_leaf_slot,
        "expected insertion patch to reuse at least one freed slot"
    );
    assert!(bvh.nodes.len() >= old_node_capacity);
    assert!(bvh.leaves.len() >= old_leaf_capacity);
    // The SAH rebuild needs 2N-1 nodes for N leaves while the two input
    // subtrees provide 2N-2 (one short since the old join node doesn't
    // exist). Allow a small margin for this extra allocation.
    assert!(
        bvh.free_node_count() <= free_node_count_after_remove + 2,
        "free_node_ids grew too much: {} (was {})",
        bvh.free_node_count(),
        free_node_count_after_remove,
    );
    assert!(
        bvh.free_leaf_count() <= free_leaf_count_after_remove + 2,
        "free_leaf_ids grew too much: {} (was {})",
        bvh.free_leaf_count(),
        free_leaf_count_after_remove,
    );
}

#[test]
fn ray_bvh_hits_uniform_leaf() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Uniform(BlockData::simple(0, 7)),
    };
    let bvh = build_bvh_in_bounds(&core, bounds);
    let hits = collect_ray_intersected_nodes_from_bvh(
        &bvh,
        [0.5, 0.5, -2.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        16.0,
        16,
    );
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].bounds, bounds);
    assert_eq!(
        hits[0].kind,
        DebugRayBvhNodeKind::LeafUniform {
            block: BlockData::simple(0, 7)
        }
    );
}

#[test]
fn ray_bvh_hits_internal_then_near_leaf() {
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [1, 0, 0, 0], 0);
    let core = RenderTreeCore {
        bounds,
        kind: RenderNodeKind::Branch(vec![
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([0, 0, 0, 0], [0, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 3)),
            },
            RenderTreeCore {
                bounds: Aabb4i::from_lattice_bounds([1, 0, 0, 0], [1, 0, 0, 0], 0),
                kind: RenderNodeKind::Uniform(BlockData::simple(0, 9)),
            },
        ]),
    };
    let bvh = build_bvh_in_bounds(&core, bounds);
    let hits = collect_ray_intersected_nodes_from_bvh(
        &bvh,
        [0.5, 0.5, -2.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        16.0,
        16,
    );
    assert!(hits.len() >= 2);
    assert_eq!(hits[0].kind, DebugRayBvhNodeKind::Internal);
    assert_eq!(
        hits[1].kind,
        DebugRayBvhNodeKind::LeafUniform {
            block: BlockData::simple(0, 3)
        }
    );
}
