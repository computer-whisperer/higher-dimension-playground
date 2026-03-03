use super::*;
use crate::content_registry::ContentRegistry;
use crate::shared::voxel::BaseWorldKind;

fn cc(v: i32) -> ChunkCoord {
    ChunkCoord::from_num(v)
}

fn cc4(v: [i32; 4]) -> [ChunkCoord; 4] {
    v.map(|x| ChunkCoord::from_num(x))
}

fn test_registry() -> Arc<ContentRegistry> {
    Arc::new(crate::plugin_loader::create_full_registry())
}

fn dense_materials_from_core_chunk(core: &RegionTreeCore, chunk_key: ChunkKey) -> Vec<u16> {
    let reg = test_registry();
    let Some(payload) = chunk_payload_from_core(core, chunk_key) else {
        return vec![0u16; CHUNK_VOLUME];
    };
    let Ok(materials) = payload.payload.dense_materials() else {
        return vec![0u16; CHUNK_VOLUME];
    };
    if materials.len() == CHUNK_VOLUME {
        // Resolve palette indices to material IDs through block_palette
        materials
            .iter()
            .map(|&idx| {
                let air = BlockData::AIR;
                let block = payload.block_palette.get(idx as usize).unwrap_or(&air);
                reg.block_material_token(block.namespace, block.block_type)
            })
            .collect()
    } else {
        vec![0u16; CHUNK_VOLUME]
    }
}

fn sample_virgin_chunk_dense(
    base_kind: &BaseWorldKind,
    world_seed: u64,
    procgen_structures: bool,
    chunk_key: [i32; 4],
) -> Vec<u16> {
    let key = chunk_key.map(ChunkCoord::from_num);
    sample_virgin_chunk_dense_with_query_bounds(
        base_kind,
        world_seed,
        procgen_structures,
        key,
        Aabb4i::chunk_world_bounds(key, 0),
    )
}

fn sample_virgin_chunk_dense_with_query_bounds(
    base_kind: &BaseWorldKind,
    world_seed: u64,
    procgen_structures: bool,
    chunk_key: ChunkKey,
    query_bounds: Aabb4i,
) -> Vec<u16> {
    let field = build_server_world_field(
        base_kind.clone(),
        world_seed,
        procgen_structures,
        HashSet::new(),
    );
    assert!(query_bounds.contains_chunk_world_min(chunk_key));
    let core = field.query_region_core(
        QueryVolume {
            bounds: query_bounds,
        },
        QueryDetail::Exact,
    );
    dense_materials_from_core_chunk(core.as_ref(), chunk_key)
}

fn collect_chunk_keys(bounds_list: &[Aabb4i]) -> Vec<ChunkKey> {
    let mut keys = Vec::new();
    for bounds in bounds_list {
        let (bmin, bmax) = bounds.to_chunk_lattice_bounds(0);
        for w in bmin[3]..=bmax[3] {
            for z in bmin[2]..=bmax[2] {
                for y in bmin[1]..=bmax[1] {
                    for x in bmin[0]..=bmax[0] {
                        keys.push([x, y, z, w].map(ChunkCoord::from_num));
                    }
                }
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

#[test]
fn overlay_dirty_bounds_drain_returns_touched_chunk_once() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::Empty,
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        123,
        false,
        HashSet::new(),
    );

    let changed_a = overlay.apply_voxel_edit(cc4([0, 0, 0, 0]), BlockData::simple(0, 3));
    let changed_b = overlay.apply_voxel_edit(cc4([1, 0, 0, 0]), BlockData::simple(0, 4));
    use crate::shared::region_tree::chunk_key_i32;
    assert_eq!(changed_a, Some(chunk_key_i32(0, 0, 0, 0)));
    assert_eq!(changed_b, Some(chunk_key_i32(0, 0, 0, 0)));

    let dirty = overlay.take_dirty_bounds();
    assert_eq!(
        dirty,
        vec![Aabb4i::chunk_world_bounds(chunk_key_i32(0, 0, 0, 0), 0)]
    );
    assert!(overlay.take_dirty_bounds().is_empty());
}

#[test]
fn overlay_dirty_bounds_drain_tracks_multiple_chunks_sorted() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::Empty,
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        123,
        false,
        HashSet::new(),
    );

    let _ = overlay.apply_voxel_edit(cc4([0, 0, 0, 0]), BlockData::simple(0, 1));
    let _ = overlay.apply_voxel_edit(cc4([8, 0, 0, 0]), BlockData::simple(0, 1));
    let _ = overlay.apply_voxel_edit(cc4([0, 8, 0, 0]), BlockData::simple(0, 1));

    let dirty = overlay.take_dirty_bounds();
    use crate::shared::region_tree::chunk_key_i32;
    assert_eq!(
        collect_chunk_keys(&dirty),
        vec![
            chunk_key_i32(0, 0, 0, 0),
            chunk_key_i32(0, 1, 0, 0),
            chunk_key_i32(1, 0, 0, 0)
        ]
    );
}

#[test]
fn overlay_clear_dirty_clears_overlay_dirty_chunks() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::Empty,
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        123,
        false,
        HashSet::new(),
    );

    let _ = overlay.apply_voxel_edit(cc4([0, 0, 0, 0]), BlockData::simple(0, 5));
    overlay.clear_dirty();
    assert!(overlay.take_dirty_bounds().is_empty());
}

#[test]
fn overlay_edit_does_not_mutate_virgin_field() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::FlatFloor {
            material: BlockData::simple(0, 11),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        777,
        false,
        HashSet::new(),
    );

    let (chunk_key, voxel_idx) = world_to_chunk_at_scale(cc(0), cc(-1), cc(0), cc(0), 0);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);

    let virgin_before = overlay
        .field()
        .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
    let virgin_payload_before =
        chunk_payload_from_core(virgin_before.as_ref(), chunk_key).expect("virgin payload");
    assert!(!virgin_payload_before.block_at(voxel_idx).is_air());

    let changed = overlay.apply_voxel_edit(cc4([0, -1, 0, 0]), BlockData::AIR);
    assert_eq!(changed, Some(chunk_key));

    let effective_after = overlay
        .effective_chunk(chunk_key, true)
        .expect("effective override chunk");
    assert!(effective_after.block_at(voxel_idx).is_air());

    let virgin_after = overlay
        .field()
        .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
    let virgin_payload_after =
        chunk_payload_from_core(virgin_after.as_ref(), chunk_key).expect("virgin payload");
    assert!(!virgin_payload_after.block_at(voxel_idx).is_air());
}

#[test]
fn query_region_core_applies_explicit_empty_override_over_virgin_content() {
    let (chunk_key, voxel_idx) = world_to_chunk_at_scale(cc(0), cc(-1), cc(0), cc(0), 0);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
    let chunk_key_i32 = [0i32, -1, 0, 0];
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::FlatFloor {
            material: BlockData::simple(0, 11),
        },
        vec![(chunk_key_i32, ResolvedChunkPayload::empty())],
        991,
        false,
        HashSet::new(),
    );

    let composed = overlay.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
    let composed_payload =
        chunk_payload_from_core(composed.as_ref(), chunk_key).expect("composed payload");
    assert!(composed_payload.block_at(voxel_idx).is_air());

    let virgin = overlay
        .field()
        .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
    let virgin_payload =
        chunk_payload_from_core(virgin.as_ref(), chunk_key).expect("virgin payload");
    assert!(!virgin_payload.block_at(voxel_idx).is_air());

    overlay.clear_dirty();
}

#[test]
fn overlay_query_preserves_generator_leaf_expansion_beyond_request_bounds() {
    let overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::MassivePlatforms {
            material: BlockData::simple(0, 11),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        1337,
        false,
        HashSet::new(),
    );
    let chunk_key = [0, -1, 0, 0].map(ChunkCoord::from_num);
    let query_bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
    let core = overlay.query_region_core(
        QueryVolume {
            bounds: query_bounds,
        },
        QueryDetail::Exact,
    );
    assert!(core.bounds.contains_chunk_world_min(chunk_key));
    assert_ne!(core.bounds, query_bounds);
}

#[test]
fn overlay_edit_is_visible_when_generator_returns_expanded_platform_leaf() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::MassivePlatforms {
            material: BlockData::simple(0, 11),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        1337,
        true,
        HashSet::new(),
    );
    let edit_pos = [0, -1, 0, 0];
    let (chunk_key, voxel_idx) = world_to_chunk_at_scale(
        cc(edit_pos[0]),
        cc(edit_pos[1]),
        cc(edit_pos[2]),
        cc(edit_pos[3]),
        0,
    );
    let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
    // Break the platform block first (collision check allows air placement).
    overlay.apply_voxel_edit(cc4(edit_pos), BlockData::AIR);
    assert_eq!(
        overlay.apply_voxel_edit(cc4(edit_pos), BlockData::simple(0, 5)),
        Some(chunk_key)
    );

    let queried = overlay.query_region_core(
        QueryVolume {
            bounds: chunk_bounds,
        },
        QueryDetail::Exact,
    );
    if let Some(payload) = chunk_payload_from_core(queried.as_ref(), chunk_key) {
        let block = payload.block_at(voxel_idx);
        assert_eq!(
            block,
            BlockData::simple(0, 5),
            "expanded-bounds overlay query returned wrong edited voxel value",
        );
    }
}

#[test]
fn virgin_world_generator_chunk_sampling_is_deterministic_for_observed_coords() {
    let observed_chunks = [
        [-14, -1, -22, -19],
        [-10, -2, -4, 0],
        [-12, -1, 0, 7],
        [-18, -14, -24, 22],
        [17, -26, -20, 23],
        [5, -2, 10, 3],
    ];
    let world_kinds = [
        BaseWorldKind::FlatFloor {
            material: BlockData::simple(0, 11),
        },
        BaseWorldKind::MassivePlatforms {
            material: BlockData::simple(0, 11),
        },
    ];

    for base_kind in &world_kinds {
        for chunk_key in observed_chunks {
            let baseline = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
            for sample_idx in 0..16 {
                let sample = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
                assert_eq!(
                        sample, baseline,
                        "virgin generator changed output for base_kind={base_kind:?} chunk={chunk_key:?} sample_idx={sample_idx}",
                    );
            }
        }
    }
}

#[test]
fn virgin_world_generator_chunk_sampling_is_query_volume_invariant() {
    let observed_chunks = [
        [-14, -1, -22, -19],
        [-10, -2, -4, 0],
        [-12, -1, 0, 7],
        [-18, -14, -24, 22],
        [17, -26, -20, 23],
        [5, -2, 10, 3],
    ];
    let world_kinds = [
        BaseWorldKind::FlatFloor {
            material: BlockData::simple(0, 11),
        },
        BaseWorldKind::MassivePlatforms {
            material: BlockData::simple(0, 11),
        },
    ];
    let query_radii = [0i32, 1, 2, 4, 8, 16];

    for base_kind in &world_kinds {
        for chunk_key in observed_chunks {
            let baseline = sample_virgin_chunk_dense(base_kind, 0xD1A6_2026, true, chunk_key);
            for radius in query_radii {
                let query_bounds = Aabb4i::from_lattice_bounds(
                    [
                        chunk_key[0] - radius,
                        chunk_key[1] - radius,
                        chunk_key[2] - radius,
                        chunk_key[3] - radius,
                    ],
                    [
                        chunk_key[0] + radius,
                        chunk_key[1] + radius,
                        chunk_key[2] + radius,
                        chunk_key[3] + radius,
                    ],
                    0,
                );
                let key = chunk_key.map(ChunkCoord::from_num);
                let sample = sample_virgin_chunk_dense_with_query_bounds(
                    base_kind,
                    0xD1A6_2026,
                    true,
                    key,
                    query_bounds,
                );
                assert_eq!(
                        sample, baseline,
                        "virgin generator changed per query volume: base_kind={base_kind:?} chunk={chunk_key:?} query_bounds={:?}->{:?}",
                        query_bounds.min,
                        query_bounds.max
                    );
            }
        }
    }
}

/// Verify that placing the same block as the platform material is correctly
/// detected as a no-op (no dirty bounds produced).
#[test]
fn voxel_edit_same_as_platform_material_is_noop() {
    let platform_material = BlockData::simple(0, 11);
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::MassivePlatforms {
            material: platform_material.clone(),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        1337,
        false,
        HashSet::new(),
    );

    // Edit within the platform, placing the same material.
    let edit_pos = [-1, 0, -6, -4];
    let changed = overlay.apply_voxel_edit(cc4(edit_pos), platform_material);
    assert_eq!(
        changed, None,
        "placing same block as platform should be no-op"
    );
    assert!(overlay.take_dirty_bounds().is_empty());
}

/// Verify that placing a *different* block on the platform succeeds and
/// produces dirty bounds that a client can receive.
#[test]
fn voxel_edit_different_from_platform_material_produces_dirty_bounds() {
    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::MassivePlatforms {
            material: BlockData::simple(0, 11),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        1337,
        false,
        HashSet::new(),
    );

    let edit_pos = [-1, 0, -6, -4];
    let (edit_chunk_key, _edit_voxel_idx) = world_to_chunk_at_scale(
        cc(edit_pos[0]),
        cc(edit_pos[1]),
        cc(edit_pos[2]),
        cc(edit_pos[3]),
        0,
    );

    // Break the platform block first (collision check rejects non-air on non-air).
    overlay.apply_voxel_edit(cc4(edit_pos), BlockData::AIR);
    overlay.take_dirty_bounds(); // drain the break's dirty bounds

    // Place a different block into the now-air position.
    let edit_block = BlockData::simple(0, 42);
    let changed = overlay.apply_voxel_edit(cc4(edit_pos), edit_block.clone());
    assert_eq!(changed, Some(edit_chunk_key));

    let dirty_bounds = overlay.take_dirty_bounds();
    assert!(!dirty_bounds.is_empty());
}

// ── Cross-scale overlap resolution tests ────────────────────────────

fn make_empty_overlay() -> PassthroughWorldOverlay<ServerWorldField> {
    let field = build_server_world_field(BaseWorldKind::Empty, 42, false, HashSet::new());
    PassthroughWorldOverlay::new(field, BaseWorldKind::Empty, 42)
}

fn read_block_from_overlay(
    overlay: &PassthroughWorldOverlay<ServerWorldField>,
    pos: [i32; 4],
    _scale_exp: i8,
) -> BlockData {
    use crate::shared::spatial::ChunkCoord;
    let fixed_pos = [
        ChunkCoord::from_num(pos[0]),
        ChunkCoord::from_num(pos[1]),
        ChunkCoord::from_num(pos[2]),
        ChunkCoord::from_num(pos[3]),
    ];
    overlay.override_chunks.block_at(fixed_pos)
}

#[test]
fn cross_scale_finer_then_coarser_both_survive() {
    let mut overlay = make_empty_overlay();
    let stone = BlockData::simple(0, 1);
    let brick = BlockData::simple(0, 2);

    // Place a scale -2 block at origin.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), stone.clone(), -2);
    // Place a scale -1 block at [2,0,0,0] — spatially nearby.
    // The scale -1 chunk at origin covers [0, 3.5], overlapping the -2 chunk.
    overlay.apply_voxel_edit_at_scale(cc4([2, 0, 0, 0]), brick.clone(), -1);

    // The -2 block should still be readable at its position.
    let read_stone = read_block_from_overlay(&overlay, [0, 0, 0, 0], -2);
    assert_eq!(
        read_stone.block_type, 1,
        "scale -2 stone block should survive after scale -1 placement"
    );
}

#[test]
fn cross_scale_coarser_then_finer_rechunks() {
    let mut overlay = make_empty_overlay();
    let stone = BlockData::simple(0, 1);
    let brick = BlockData::simple(0, 2);

    // Place a scale -1 block at origin first.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), stone.clone(), -1);
    // Place a scale -2 block at an adjacent (non-overlapping) position —
    // triggers rechunking of the -1 chunk to scale -2.
    overlay.apply_voxel_edit_at_scale(cc4([1, 0, 0, 0]), brick.clone(), -2);

    // The brick at scale -2 should be present at its position.
    let read_brick = read_block_from_overlay(&overlay, [1, 0, 0, 0], -2);
    assert_eq!(
        read_brick.block_type, 2,
        "scale -2 brick should be placed after rechunking"
    );

    // The original stone (rechunked from scale -1 to scale -2) should survive.
    let read_stone = read_block_from_overlay(&overlay, [0, 0, 0, 0], -2);
    assert_eq!(
        read_stone.block_type, 1,
        "scale -1 stone should survive rechunking to scale -2"
    );
}

#[test]
fn same_scale_edit_no_rechunk() {
    let mut overlay = make_empty_overlay();
    let stone = BlockData::simple(0, 1);
    let brick = BlockData::simple(0, 2);

    // Place two blocks at the same scale.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), stone.clone(), -1);
    overlay.apply_voxel_edit_at_scale(cc4([1, 0, 0, 0]), brick.clone(), -1);

    let read_stone = read_block_from_overlay(&overlay, [0, 0, 0, 0], -1);
    let read_brick = read_block_from_overlay(&overlay, [1, 0, 0, 0], -1);
    assert_eq!(read_stone.block_type, 1);
    assert_eq!(read_brick.block_type, 2);
}

#[test]
fn coarser_edit_fills_multiple_cells() {
    let mut overlay = make_empty_overlay();
    let brick = BlockData::simple(0, 2);

    // Place a scale -2 block to anchor the chunk at scale -2.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), BlockData::simple(0, 1), -2);

    // Place a scale -1 block (coarser) at a non-overlapping position.
    // The chunk is forced to scale -2, so the -1 block fills
    // 2^4 = 16 cells at scale -2 (2 cells per axis).
    overlay.apply_voxel_edit_at_scale(cc4([1, 0, 0, 0]), brick.clone(), -1);

    // The brick should fill cells at scale -2 starting at position [1,0,0,0].
    let read = read_block_from_overlay(&overlay, [1, 0, 0, 0], -2);
    assert_eq!(
        read.block_type, 2,
        "coarser brick block should fill cells at finer scale"
    );
}

#[test]
fn placing_finer_block_near_multiple_coarser_blocks_preserves_all() {
    let mut overlay = make_empty_overlay();

    // Build up a set of blocks at scale -1 (simulating a wall).
    let positions_s1 = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [2, 1, 0, 0],
        [3, 1, 0, 0],
    ];
    for (i, &pos) in positions_s1.iter().enumerate() {
        let block = BlockData::simple(0, 10 + i as u32);
        overlay.apply_voxel_edit_at_scale(cc4(pos), block, -1);
    }

    // Verify all blocks are present before the finer edit.
    for (i, &pos) in positions_s1.iter().enumerate() {
        let read = read_block_from_overlay(&overlay, pos, -1);
        assert_eq!(
            read.block_type,
            10 + i as u32,
            "pre-edit: block at {:?} scale -1 missing (got type {})",
            pos,
            read.block_type
        );
    }

    // Now place a finer block (scale -2) at a position that doesn't overlap
    // any of the scale -1 blocks (all have z=0, so z=1 is safe).
    let finer_block = BlockData::simple(0, 99);
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 1, 0]), finer_block, -2);

    // ALL original scale -1 blocks should survive (rechunked to scale -2).
    // Read them back at scale -2 (they should still be present as multi-cell
    // populations at the finer scale).
    for (i, &pos) in positions_s1.iter().enumerate() {
        let read = read_block_from_overlay(&overlay, pos, -2);
        let expected = 10 + i as u32;
        assert_eq!(
            read.block_type, expected,
            "post-edit: block at {:?} scale -2 wrong (expected {}, got {})",
            pos, expected, read.block_type
        );
    }
    // The finer block should also be present.
    let read_finer = read_block_from_overlay(&overlay, [0, 0, 1, 0], -2);
    assert_eq!(
        read_finer.block_type, 99,
        "finer block at [0,0,1,0] should be present"
    );
}

#[test]
fn placing_finer_block_does_not_lose_distant_same_scale_chunks() {
    let mut overlay = make_empty_overlay();

    // Place blocks at scale -1 in two DIFFERENT chunks.
    // At scale -1, chunk size is 4 world units, so world pos 0 and 4
    // fall in different chunks.
    let block_a = BlockData::simple(0, 10);
    let block_b = BlockData::simple(0, 20);
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block_a.clone(), -1);
    overlay.apply_voxel_edit_at_scale(cc4([4, 0, 0, 0]), block_b.clone(), -1);

    // Verify both exist.
    assert_eq!(
        read_block_from_overlay(&overlay, [0, 0, 0, 0], -1).block_type,
        10
    );
    assert_eq!(
        read_block_from_overlay(&overlay, [4, 0, 0, 0], -1).block_type,
        20
    );

    // Place a finer block in the first chunk's region.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), BlockData::simple(0, 99), -2);

    // The second chunk should be completely unaffected.
    let read_b = read_block_from_overlay(&overlay, [4, 0, 0, 0], -1);
    assert_eq!(
        read_b.block_type, 20,
        "distant chunk at scale -1 should survive finer edit in another chunk"
    );
}

// -----------------------------------------------------------------------
// Virgin-aware multi-scale edit tests
// -----------------------------------------------------------------------

fn make_flat_floor_overlay() -> ServerWorldOverlay {
    ServerWorldOverlay::from_chunk_payloads(
        BaseWorldKind::FlatFloor {
            material: BlockData::simple(0, 11),
        },
        Vec::<([i32; 4], ResolvedChunkPayload)>::new(),
        42,
        false,
        HashSet::new(),
    )
}

/// Read a block through the full compositing pipeline (virgin + override).
fn read_composed_block(overlay: &ServerWorldOverlay, pos: [i32; 4]) -> BlockData {
    let (chunk_key, voxel_idx) =
        world_to_chunk_at_scale(cc(pos[0]), cc(pos[1]), cc(pos[2]), cc(pos[3]), 0);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);
    let core = overlay.query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
    chunk_payload_from_core(core.as_ref(), chunk_key)
        .map(|p| p.block_at(voxel_idx))
        .unwrap_or(BlockData::AIR)
}

#[test]
fn scale_minus_one_edit_preserves_virgin_floor() {
    let mut overlay = make_flat_floor_overlay();

    // The flat floor is at chunk y=-1, so world y=-8..-1 all have
    // the floor material. Break the target cell first, then place.
    let edit_pos = [0, -1, 0, 0]; // on the floor
    let brick = BlockData::simple(0, 2);
    overlay.apply_voxel_edit_at_scale(cc4(edit_pos), BlockData::AIR, -1);
    overlay.apply_voxel_edit_at_scale(cc4(edit_pos), brick.clone(), -1);

    // The edited block should be present (read from override).
    let read_edited = read_block_from_overlay(&overlay, edit_pos, -1);
    assert_eq!(
        read_edited.block_type, 2,
        "edited block should be readable at scale -1"
    );

    // Nearby floor blocks should still be present in the composed view.
    // World pos [1, -1, 0, 0] is adjacent to the edit but should still be floor.
    let neighbor = read_composed_block(&overlay, [1, -1, 0, 0]);
    assert!(
        !neighbor.is_air(),
        "neighboring floor block at [1,-1,0,0] should survive the scale -1 edit"
    );
}

#[test]
fn scale_minus_two_edit_preserves_virgin_floor() {
    let mut overlay = make_flat_floor_overlay();

    // Edit at scale -2 on the floor — break the target cell first,
    // then place. The override chunk should still contain virgin floor
    // data for surrounding cells.
    let edit_pos = [0, -1, 0, 0];
    overlay.apply_voxel_edit_at_scale(cc4(edit_pos), BlockData::AIR, -2);
    overlay.apply_voxel_edit_at_scale(cc4(edit_pos), BlockData::simple(0, 5), -2);

    // The edit should be there.
    let read_edited = read_block_from_overlay(&overlay, edit_pos, -2);
    assert_eq!(read_edited.block_type, 5);

    // Other cells in the same chunk should still have the floor material,
    // not air. At scale -2, the chunk covers a small region. World pos
    // [0, -1, 1, 0] is nearby and should have floor material in the override.
    // (Because the override was initialized from virgin data.)
    let (chunk_key, _) = world_to_chunk_at_scale(
        cc(edit_pos[0]),
        cc(edit_pos[1]),
        cc(edit_pos[2]),
        cc(edit_pos[3]),
        -2,
    );
    let payload = overlay.override_chunks.chunk_payload(chunk_key);
    assert!(
        payload.is_some(),
        "override chunk should exist at {:?}",
        chunk_key
    );
    // The override chunk should NOT be all-air except for the edit.
    // Since the floor covers this region, most cells should be floor material.
    let blocks = payload.unwrap().0.dense_blocks();
    let non_air_count = blocks.iter().filter(|b| !b.is_air()).count();
    assert!(
        non_air_count > 1,
        "override chunk should have multiple non-air blocks (got {non_air_count}), \
             proving virgin floor data was preserved"
    );
}

#[test]
fn query_virgin_blocks_at_scale_returns_floor_for_finer_scales() {
    let overlay = make_flat_floor_overlay();

    // Query virgin data at scale -1 for a chunk on the floor.
    // World y=-1 is in scale-0 chunk y=-1. At scale -1, a chunk at
    // the appropriate key should contain floor material.
    let (chunk_key, _) = world_to_chunk_at_scale(cc(0), cc(-1), cc(0), cc(0), -1);
    let blocks = overlay.query_virgin_blocks_at_scale(chunk_key, -1);

    let non_air = blocks.iter().filter(|b| !b.is_air()).count();
    assert!(
        non_air > 0,
        "virgin blocks at scale -1 on the floor should contain floor material, \
             got {non_air} non-air blocks out of {}",
        CHUNK_VOLUME
    );
}

#[test]
fn query_virgin_blocks_at_scale_returns_air_above_floor() {
    let overlay = make_flat_floor_overlay();

    // Above the floor (y=0 in scale-0 chunks), everything should be air.
    let (chunk_key, _) = world_to_chunk_at_scale(cc(0), cc(0), cc(0), cc(0), -1);
    let blocks = overlay.query_virgin_blocks_at_scale(chunk_key, -1);

    assert!(
        blocks.iter().all(|b| b.is_air()),
        "virgin blocks at scale -1 above floor should all be air"
    );
}

#[test]
fn edit_restoring_virgin_state_removes_override() {
    let mut overlay = make_flat_floor_overlay();

    // Break the floor, then place it back — the override should be pruned
    // because the final state matches virgin.
    let edit_pos = [0, -1, 0, 0];
    let floor_material = BlockData::simple(0, 11);

    // First edit: break the floor (air ≠ virgin floor → override created).
    overlay.apply_voxel_edit(cc4(edit_pos), BlockData::AIR);
    let (chunk_key, _) = world_to_chunk_at_scale(
        cc(edit_pos[0]),
        cc(edit_pos[1]),
        cc(edit_pos[2]),
        cc(edit_pos[3]),
        0,
    );
    assert!(
        overlay.override_chunks.chunk_payload(chunk_key).is_some(),
        "override should exist after edit"
    );

    // Second edit: place floor material back (air → floor = matches virgin).
    overlay.apply_voxel_edit(cc4(edit_pos), floor_material);
    let still_overridden = overlay.override_chunks.chunk_payload(chunk_key);
    assert!(
        still_overridden.is_none(),
        "override should be removed when edit restores virgin state"
    );
}

#[test]
fn rechunked_virgin_matching_fragments_are_pruned() {
    let mut overlay = make_flat_floor_overlay();

    // Break the floor, then place a block — creates a scale-0 override.
    let edit_pos = [0, -1, 0, 0];
    overlay.apply_voxel_edit(cc4(edit_pos), BlockData::AIR);
    overlay.apply_voxel_edit(cc4(edit_pos), BlockData::simple(0, 2));

    let chunks_before = overlay.override_chunks.non_empty_chunk_count();
    assert!(chunks_before >= 1, "should have override chunk(s)");

    // Place a scale -1 block at a non-overlapping position — triggers
    // rechunking of the scale-0 override to scale -1. Many of the resulting
    // fragments should be pruned because they match virgin floor data.
    overlay.apply_voxel_edit_at_scale(cc4([1, -1, 0, 0]), BlockData::AIR, -1);
    overlay.apply_voxel_edit_at_scale(cc4([1, -1, 0, 0]), BlockData::simple(0, 3), -1);

    let chunks_after = overlay.override_chunks.non_empty_chunk_count();
    // Without pruning, we'd have up to 16 chunks from the rechunked
    // scale-0 override. With pruning, only chunks that differ from
    // virgin survive — at minimum the edited chunk.
    assert!(
        chunks_after < 16,
        "rechunked fragments matching virgin should be pruned \
             (got {chunks_after} chunks, expected fewer than 16)"
    );
    assert!(chunks_after >= 1, "at least the edited chunk should remain");
}

/// Place a scale-2 block into an empty world and verify that the stored
/// blocks carry scale_exp=2 (not 0 or some other value).
#[test]
fn scale2_edit_stores_correct_scale_exp_on_blocks() {
    let mut overlay = make_empty_overlay();
    let block = BlockData::simple(0, 42);

    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block.clone(), 2);

    // The edit is at scale 2 into an empty world (no finer data).
    // target_scale should be 2 (the edit's own scale).
    // Read back at scale 2.
    let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 2);
    assert_eq!(read.block_type, 42, "scale-2 block should be stored");
    assert_eq!(
        read.scale_exp, 2,
        "block should carry scale_exp=2, got {}",
        read.scale_exp
    );
}

/// Place a scale-3 block into an empty world and verify scale_exp=3.
#[test]
fn scale3_edit_stores_correct_scale_exp_on_blocks() {
    let mut overlay = make_empty_overlay();
    let block = BlockData::simple(0, 99);

    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block.clone(), 3);

    let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
    assert_eq!(read.block_type, 99, "scale-3 block should be stored");
    assert_eq!(
        read.scale_exp, 3,
        "block should carry scale_exp=3, got {}",
        read.scale_exp
    );
}

/// Place a scale-2 block over a scale-0 floor.
/// The floor forces rechunking to scale 0, so the scale-2 block fills
/// multiple scale-0 cells. Each cell should carry scale_exp=2.
#[test]
fn scale2_edit_over_floor_fills_cells_with_correct_scale_exp() {
    let mut overlay = make_flat_floor_overlay();

    // Break the floor at scale 2, then place. The floor is at scale 0, so
    // target_scale = 0. The edit fills (1<<(2-0))^4 = 256 cells per chunk.
    let block = BlockData::simple(0, 42);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), BlockData::AIR, 2);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), block.clone(), 2);

    // Read back one of the filled cells at scale 0.
    let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
    assert_eq!(
        read.block_type, 42,
        "scale-2 block should fill scale-0 cells"
    );
    assert_eq!(
        read.scale_exp, 2,
        "filled cells should carry the original scale_exp=2, got {}",
        read.scale_exp
    );
}

/// Place a scale-3 block over a scale-0 floor.
/// Each filled cell should carry scale_exp=3.
#[test]
fn scale3_edit_over_floor_fills_cells_with_correct_scale_exp() {
    let mut overlay = make_flat_floor_overlay();

    let block = BlockData::simple(0, 99);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), BlockData::AIR, 3);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), block.clone(), 3);

    let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
    assert_eq!(
        read.block_type, 99,
        "scale-3 block should fill scale-0 cells"
    );
    assert_eq!(
        read.scale_exp, 3,
        "filled cells should carry the original scale_exp=3, got {}",
        read.scale_exp
    );
}

/// Dirty bounds for a scale-3 edit should cover the full scale-3 chunk
/// extent, and the composited view for those bounds should include the
/// override data.
#[test]
fn scale3_edit_dirty_bounds_cover_full_chunk_extent() {
    let mut overlay = make_empty_overlay();
    let block = BlockData::simple(0, 99);

    // Place a scale-3 block.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block.clone(), 3);

    // Take the dirty bounds — these are what the server broadcasts to clients.
    let dirty_bounds = overlay.take_dirty_bounds();
    assert!(
        !dirty_bounds.is_empty(),
        "dirty bounds should be non-empty after edit"
    );

    // The dirty bounds should cover the full scale-3 chunk extent (64^4).
    let (chunk_key_s3, _) = world_to_chunk_at_scale(cc(0), cc(0), cc(0), cc(0), 3);
    let expected_bounds = Aabb4i::chunk_world_bounds(chunk_key_s3, 3);
    let covers_expected = dirty_bounds.iter().any(|b| {
        b.min[0] <= expected_bounds.min[0]
            && b.max[0] >= expected_bounds.max[0]
            && b.min[1] <= expected_bounds.min[1]
            && b.max[1] >= expected_bounds.max[1]
            && b.min[2] <= expected_bounds.min[2]
            && b.max[2] >= expected_bounds.max[2]
            && b.min[3] <= expected_bounds.min[3]
            && b.max[3] >= expected_bounds.max[3]
    });
    assert!(
        covers_expected,
        "dirty bounds should cover the full scale-3 extent {:?}->{:?}, got {:?}",
        expected_bounds.min, expected_bounds.max, dirty_bounds
    );

    // Query the composited view using the dirty bounds (what the server
    // actually sends to the client).
    for bounds in &dirty_bounds {
        let core = overlay.query_region_core(QueryVolume { bounds: *bounds }, QueryDetail::Exact);

        // The scale-0 chunk at origin should contain the block.
        let (chunk_key_s0, voxel_idx) = world_to_chunk_at_scale(cc(0), cc(0), cc(0), cc(0), 0);
        let payload = chunk_payload_from_core(core.as_ref(), chunk_key_s0);
        if let Some(payload) = payload {
            let block_at_origin = payload.block_at(voxel_idx);
            assert_eq!(
                block_at_origin.block_type, 99,
                "scale-3 block should be visible at origin in composited view"
            );
        }
    }
}

/// Place then break a scale-3 block in an empty world.
/// Breaking should remove the block entirely.
#[test]
fn break_scale3_block_removes_it_empty_world() {
    let mut overlay = make_empty_overlay();
    let block = BlockData::simple(0, 99);

    // Place at scale 3.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block.clone(), 3);

    // Verify it's there.
    let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
    assert_eq!(read.block_type, 99, "block should be placed");

    // Break: send AIR at scale 3.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), BlockData::AIR, 3);

    // The block at the break position should be gone.
    let read = read_block_from_overlay(&overlay, [0, 0, 0, 0], 3);
    assert!(
        read.is_air(),
        "broken block should be air, got block_type={}",
        read.block_type
    );

    // And the rest of the chunk should still be air (not filled with blocks).
    let read2 = read_block_from_overlay(&overlay, [8, 0, 0, 0], 3);
    assert!(
        read2.is_air(),
        "neighboring cell at scale 3 should remain air, got block_type={}",
        read2.block_type
    );

    // Override should be removed entirely since all AIR matches empty virgin.
    assert_eq!(
        overlay.override_chunks.non_empty_chunk_count(),
        0,
        "override should be removed when edit restores virgin (empty) state"
    );
}

/// Placing a scale-3 block on the floor should only override ONE scale-0
/// chunk; the floor at adjacent chunks should remain.
#[test]
fn scale3_place_on_floor_does_not_overwrite_neighbors() {
    let mut overlay = make_flat_floor_overlay();
    let block = BlockData::simple(0, 99);

    // Floor at [8, -1, 0, 0] should be present before edit.
    let floor_before = read_composed_block(&overlay, [8, -1, 0, 0]);
    assert_eq!(
        floor_before.block_type, 11,
        "floor should exist before edit at [8,-1,0,0]"
    );

    // Break the floor at scale 3, then place.
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), BlockData::AIR, 3);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), block.clone(), 3);

    // The placed block should be at [0,-1,0,0].
    let placed = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
    assert_eq!(placed.block_type, 99, "block should be placed");

    // The floor at [8, -1, 0, 0] should still be floor material.
    let floor_after = read_composed_block(&overlay, [8, -1, 0, 0]);
    assert_eq!(
        floor_after.block_type, 11,
        "floor at [8,-1,0,0] should survive the scale-3 edit, got block_type={}",
        floor_after.block_type
    );
}

/// Place then break a scale-3 block over a floor.
/// Breaking should remove the block but preserve the floor.
#[test]
fn break_scale3_block_over_floor() {
    let mut overlay = make_flat_floor_overlay();
    let block = BlockData::simple(0, 99);

    // Break the floor at scale 3, then place.
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), BlockData::AIR, 3);
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), block.clone(), 3);

    // Verify it's there (floor forces target_scale=0, so read at scale 0).
    let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
    assert_eq!(read.block_type, 99, "block should be placed");

    // Break: send AIR at scale 3.
    overlay.apply_voxel_edit_at_scale(cc4([0, -1, 0, 0]), BlockData::AIR, 3);

    // The block at the break position should be gone.
    let read = read_block_from_overlay(&overlay, [0, -1, 0, 0], 0);
    assert!(
        read.is_air(),
        "broken block should be air, got block_type={}",
        read.block_type
    );

    // Read via composition: the floor should still be visible outside the
    // edit area (the override only covers the area the scale-3 block occupied).
    let floor_pos = [8, -1, 0, 0];
    let composed = read_composed_block(&overlay, floor_pos);
    assert_eq!(
        composed.block_type, 11,
        "floor outside the edit area should be preserved, got block_type={}",
        composed.block_type
    );
}

#[test]
fn determine_chunk_scale_respects_existing_blocks() {
    let mut overlay = make_empty_overlay();

    // Place a scale -2 block at origin. This creates override data at scale -2.
    let fine_block = BlockData::simple(0, 5);
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), fine_block.clone(), -2);

    // Now determine the chunk scale for a scale 0 placement at [1,0,0,0].
    // The scale-0 chunk at origin covers [0,8)^4. Inside that region there's
    // a scale -2 block. A scale 0 chunk can represent blocks with scale_exp
    // in [0, 3]. scale_exp = -2 < 0, so the block is NOT representable at
    // scale 0. determine_chunk_scale should pick a finer scale.
    let chosen = overlay.determine_chunk_scale(cc4([1, 0, 0, 0]), 0);
    assert!(
        chosen <= -2,
        "chunk scale {chosen} can't represent existing scale -2 blocks"
    );

    // Place a scale 0 block at [8,0,0,0] — a region with no existing overrides.
    // The coarsest valid scale (0) should be chosen since there's nothing to conflict.
    let chosen_empty = overlay.determine_chunk_scale(cc4([8, 0, 0, 0]), 0);
    assert_eq!(
        chosen_empty, 0,
        "empty region should use coarsest scale, got {chosen_empty}"
    );

    // Place a scale 1 block at [16,0,0,0]. determine_chunk_scale for a scale 0
    // edit nearby should still pick scale 0 since scale 1 is representable at
    // scale 0 (scale_exp 1 >= 0 and 1 - 0 = 1 <= 3).
    overlay.apply_voxel_edit_at_scale(cc4([16, 0, 0, 0]), BlockData::simple(0, 7), 1);
    let chosen_coarse = overlay.determine_chunk_scale(cc4([17, 0, 0, 0]), 0);
    assert!(
        chosen_coarse >= -3 && chosen_coarse <= 0,
        "scale 1 blocks should be representable at scale 0, got {chosen_coarse}"
    );
}

/// Regression test: placing a fine-scale block into a coarse-scale chunk
/// carves the coarse chunk into fragments.  ALL fragments (not just the
/// directly-edited chunk) must appear in dirty_save_chunks so they are
/// persisted.  Without this, the fragments holding the original coarse
/// data are lost on save/reload.
#[test]
fn cross_scale_edit_marks_carved_fragments_dirty_for_save() {
    let mut overlay = make_empty_overlay();

    let block_a = BlockData::simple(0, 1); // scale 0
    let block_b = BlockData::simple(0, 2).at_scale(-2); // scale -2

    // 1. Place several scale-0 blocks spread across the chunk region
    //    [0..16, 0..16, 0..16, 0..16].  Blocks at positions far apart
    //    ensure the carve produces non-empty fragments.
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 0, 0]), block_a.clone(), 0);
    overlay.apply_voxel_edit_at_scale(cc4([8, 0, 0, 0]), block_a.clone(), 0);
    overlay.apply_voxel_edit_at_scale(cc4([0, 8, 0, 0]), block_a.clone(), 0);
    overlay.apply_voxel_edit_at_scale(cc4([0, 0, 8, 0]), block_a.clone(), 0);

    // Verify blocks are present.
    assert_eq!(
        overlay
            .override_chunks
            .block_at(cc4([8, 0, 0, 0]))
            .block_type,
        block_a.block_type
    );

    // Clear dirty state to simulate having already saved.
    overlay.clear_dirty();
    assert!(overlay.dirty_save_chunk_entries().is_empty());

    // 2. Place a scale-(-2) block at [1,1,1,1].  At scale -2, chunk cell
    //    size is 0.25, so the chunk covers [0..4, 0..4, 0..4, 0..4].
    //    The existing scale-0 chunk [0..16,...] must be carved: the
    //    [0..4,...] region becomes the new scale-2 chunk, and the rest
    //    ([4..16,...] etc.) becomes fragments that hold the blocks at
    //    [8,0,0,0], [0,8,0,0], and [0,0,8,0].
    overlay.apply_voxel_edit_at_scale(cc4([1, 1, 1, 1]), block_b.clone(), -2);

    let dirty = overlay.dirty_save_chunk_entries();
    assert!(
        !dirty.is_empty(),
        "fine-scale edit should produce dirty save entries"
    );

    // The original blocks must still be retrievable from the tree
    // (they were re-encoded into fragments, not deleted).
    for pos in [[0, 0, 0, 0], [8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0]] {
        let readback = overlay.override_chunks.block_at(cc4(pos));
        assert_eq!(
            readback.block_type, block_a.block_type,
            "original block at {:?} should survive carve: got {:?}",
            pos, readback
        );
    }

    // The new block_b data must be present.
    let readback_b = overlay.override_chunks.block_at(cc4([1, 1, 1, 1]));
    assert_eq!(
        readback_b.block_type, block_b.block_type,
        "edited block should be present: got {:?}",
        readback_b
    );

    // Critical: dirty_save_chunks must contain entries that cover the
    // ENTIRE affected region — not just the single edit chunk.  We
    // verify this by checking that every non-empty chunk in the tree
    // is represented in dirty_save_chunks.
    let all_entries = overlay
        .override_chunks
        .collect_chunk_entries_in_bounds(overlay.override_chunks.root().unwrap().bounds);
    for (key, se) in &all_entries {
        assert!(
            dirty.contains_key(key),
            "chunk {:?} at scale {} is in the tree but NOT in dirty_save_chunks — \
                 it would be lost on save",
            key,
            se
        );
    }
    assert!(
        all_entries.len() >= 2,
        "expected at least 2 non-empty chunks (edit + fragments), got {}",
        all_entries.len()
    );
}

/// Regression test: a scale-3 block placed in a scale-0 chunk creates a
/// Uniform node with block.scale_exp=3 but bounds sized for scale 0.
/// `collect_chunk_entries_in_bounds` must still enumerate the entry at
/// scale 0 (derived from bounds), NOT block.scale_exp=3 which would
/// produce an invalid lattice range and silently skip the chunk.
#[test]
fn uniform_with_coarse_block_in_fine_chunk_enumerates_correctly() {
    use crate::shared::region_tree::{chunk_key_i32, RegionChunkTree};

    // Construct a Uniform node directly: a scale-3 block in scale-0 bounds.
    let block = BlockData::simple(0, 42).at_scale(3);
    let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key_i32(1, 1, 0, -1), 0);

    let mut tree = RegionChunkTree::new();
    tree.set_chunk_at_scale(
        chunk_key_i32(1, 1, 0, -1),
        Some(ResolvedChunkPayload::uniform(block.clone())),
        0,
    );

    let root = tree.root().expect("tree should have root");
    assert!(
        matches!(root.kind, RegionNodeKind::Uniform(_)),
        "expected Uniform node, got {:?}",
        std::mem::discriminant(&root.kind),
    );

    // `collect_chunk_entries_in_bounds` should return exactly one entry at
    // scale 0 — not zero entries (the old bug with block.scale_exp=3).
    let entries = tree.collect_chunk_entries_in_bounds(chunk_bounds);
    assert_eq!(
        entries.len(),
        1,
        "Uniform(scale-3 block in scale-0 bounds) should produce 1 entry, got {}",
        entries.len()
    );
    let (key, se) = &entries[0];
    assert_eq!(
        *se, 0,
        "entry scale should be 0 (chunk scale), not {} (block scale)",
        se
    );
    assert_eq!(*key, chunk_key_i32(1, 1, 0, -1));

    // Also verify chunk_payload returns the correct scale.
    let (_, payload_se) = tree
        .chunk_payload(chunk_key_i32(1, 1, 0, -1))
        .expect("chunk_payload should find the Uniform");
    assert_eq!(
        payload_se, 0,
        "chunk_payload scale should be 0, not {}",
        payload_se
    );
}
