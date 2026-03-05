use super::*;
use polychora::content_registry::{ContentRegistry, MaterialResolver};
use polychora::shared::region_tree::chunk_key_i32;
use polychora::shared::voxel::BlockData;

fn cc(v: i32) -> ChunkCoord {
    ChunkCoord::from_num(v)
}

fn cc4(v: [i32; 4]) -> [ChunkCoord; 4] {
    v.map(|x| ChunkCoord::from_num(x))
}

fn test_registry() -> ContentRegistry {
    let (registry, _pending) = polychora::plugin_loader::create_full_registry();
    registry
}

fn test_resolver() -> MaterialResolver {
    MaterialResolver::from_registry(&test_registry())
}

fn sample_voxel_from_frame(scene: &Scene, wx: i32, wy: i32, wz: i32, ww: i32) -> Option<u8> {
    let (chunk_key_fixed, voxel_idx) = world_to_chunk_at_scale(
        ChunkCoord::from_num(wx),
        ChunkCoord::from_num(wy),
        ChunkCoord::from_num(wz),
        ChunkCoord::from_num(ww),
        0,
    );
    let chunk_key: [i32; 4] = chunk_key_fixed.map(|c: ChunkCoord| c.to_num::<i32>());

    for leaf in &scene.active_config.frame_data.leaf_headers {
        if chunk_key[0] < leaf.min_chunk_coord[0]
            || chunk_key[0] > leaf.max_chunk_coord[0]
            || chunk_key[1] < leaf.min_chunk_coord[1]
            || chunk_key[1] > leaf.max_chunk_coord[1]
            || chunk_key[2] < leaf.min_chunk_coord[2]
            || chunk_key[2] > leaf.max_chunk_coord[2]
            || chunk_key[3] < leaf.min_chunk_coord[3]
            || chunk_key[3] > leaf.max_chunk_coord[3]
        {
            continue;
        }

        if leaf.leaf_kind == higher_dimension_playground::render::VTE_LEAF_KIND_UNIFORM {
            let material = leaf.uniform_material as u8;
            return (material != 0).then_some(material);
        }
        if leaf.leaf_kind != higher_dimension_playground::render::VTE_LEAF_KIND_VOXEL_CHUNK_ARRAY {
            continue;
        }

        let dim_x = (leaf.max_chunk_coord[0] - leaf.min_chunk_coord[0] + 1) as usize;
        let dim_y = (leaf.max_chunk_coord[1] - leaf.min_chunk_coord[1] + 1) as usize;
        let dim_z = (leaf.max_chunk_coord[2] - leaf.min_chunk_coord[2] + 1) as usize;
        let local_x = (chunk_key[0] - leaf.min_chunk_coord[0]) as usize;
        let local_y = (chunk_key[1] - leaf.min_chunk_coord[1]) as usize;
        let local_z = (chunk_key[2] - leaf.min_chunk_coord[2]) as usize;
        let local_w = (chunk_key[3] - leaf.min_chunk_coord[3]) as usize;
        let linear = local_x + dim_x * (local_y + dim_y * (local_z + dim_z * local_w));
        let entry_index = leaf.chunk_entry_offset as usize + linear;
        let &entry = scene
            .active_config
            .frame_data
            .leaf_chunk_entries
            .get(entry_index)?;

        if entry == higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_EMPTY {
            return None;
        }
        if (entry & higher_dimension_playground::render::VTE_LEAF_CHUNK_ENTRY_UNIFORM_FLAG) != 0 {
            let material = (entry & 0xFFFF) as u16;
            return (material != 0).then_some(material as u8);
        }

        let chunk_index = entry.saturating_sub(1) as usize;
        let header = scene
            .active_config
            .frame_data
            .chunk_headers
            .get(chunk_index)?;
        if (header.flags & GpuVoxelChunkHeader::FLAG_FULL) == 0 {
            let occ_word_index = header.occupancy_word_offset as usize + (voxel_idx / 32);
            let &occ_word = scene
                .active_config
                .frame_data
                .occupancy_words
                .get(occ_word_index)?;
            if (occ_word & (1u32 << (voxel_idx % 32))) == 0 {
                return None;
            }
        }
        let mat_word_index = header.material_word_offset as usize + (voxel_idx / 2);
        let &mat_word = scene
            .active_config
            .frame_data
            .material_words
            .get(mat_word_index)?;
        let material = ((mat_word >> ((voxel_idx % 2) * 16)) & 0xFFFF) as u8;
        return (material != 0).then_some(material);
    }

    None
}

fn make_scene_with_blocks(voxels: &[([i32; 4], BlockData)]) -> Scene {
    let mut scene = Scene::new(ScenePreset::Empty);
    for ([x, y, z, w], material) in voxels.iter() {
        scene.world_set_block(*x, *y, *z, *w, material.clone());
    }
    scene
}

#[test]
fn remove_block_along_ray_hits_first_solid_voxel() {
    let mut scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let removed = scene.remove_block_along_ray([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0);

    assert_eq!(removed, Some(cc4([0, 0, 0, 0])));
    assert!(scene.get_block_data(0, 0, 0, 0).is_air());
}

#[test]
fn place_block_along_ray_places_in_last_empty_before_hit() {
    let mut scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let placed = scene.place_block_along_ray(
        [0.5, 0.5, -2.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        8.0,
        BlockData::simple(0, 3),
    );

    assert_eq!(placed, Some(cc4([0, 0, -1, 0])));
    assert_eq!(scene.get_block_data(0, 0, -1, 0), BlockData::simple(0, 3));
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 7));
}

#[test]
fn block_edit_targets_reports_hit_and_placement_voxels() {
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets([0.5, 0.5, -2.0, 0.5], [0.0, 0.0, 1.0, 0.0], 8.0, 0);

    let cc = |v: i32| ChunkCoord::from_num(v);
    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, [cc(0), cc(0), cc(0), cc(0)]);
    assert_eq!(hit.scale_exp, 0);
    assert_eq!(targets.hit_block, Some(BlockData::simple(0, 7)));

    let place = targets.place.unwrap();
    assert_eq!(place.origin, [cc(0), cc(0), cc(-1), cc(0)]);
    assert_eq!(place.scale_exp, 0);
}

#[test]
fn block_edit_targets_scale1_hit_scale0_place() {
    // A 2x2x2x2 block at origin, hit from -Z direction.
    // Placement at scale 0 should be adjacent to the 2-unit face.
    let block = BlockData::simple(0, 7).at_scale(1);
    let mut scene = make_scene_with_blocks(&[]);
    // Fill the 2x2x2x2 region with this scale-1 block
    for dx in 0..2 {
        for dy in 0..2 {
            for dz in 0..2 {
                for dw in 0..2 {
                    scene.world_set_block(dx, dy, dz, dw, block.clone());
                }
            }
        }
    }

    let targets = scene.block_edit_targets(
        [0.5, 0.5, -2.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        8.0,
        0, // placement scale 0
    );

    let cc = |v: i32| ChunkCoord::from_num(v);
    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, [cc(0), cc(0), cc(0), cc(0)]);
    assert_eq!(hit.scale_exp, 1);
    assert_eq!(hit.size(), step_for_scale(1));

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, 0);
    // Should be placed at z=-1, adjacent to the 2-unit block's -Z face
    assert_eq!(place.origin[2], cc(-1));
}

#[test]
fn block_edit_targets_scale0_hit_scale1_place() {
    // A scale-0 block at origin, hit from -Z direction.
    // Placement at scale 1 should snap to 2-unit grid.
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets(
        [0.5, 0.5, -2.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        8.0,
        1, // placement scale 1
    );

    let cc = |v: i32| ChunkCoord::from_num(v);
    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, [cc(0), cc(0), cc(0), cc(0)]);
    assert_eq!(hit.scale_exp, 0);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, 1);
    // Should be placed adjacent to the -Z face, snapped to 2-unit grid
    assert_eq!(place.origin[2], cc(-2));
    assert_eq!(place.size(), step_for_scale(1));
}

#[test]
fn block_edit_targets_scale2_hit_origin_aligned() {
    // A 4x4x4x4 block, verify origin is 4-unit aligned.
    let block = BlockData::simple(0, 7).at_scale(2);
    let mut scene = make_scene_with_blocks(&[]);
    for dx in 0..4 {
        for dy in 0..4 {
            for dz in 0..4 {
                for dw in 0..4 {
                    scene.world_set_block(dx, dy, dz, dw, block.clone());
                }
            }
        }
    }

    let targets = scene.block_edit_targets([1.5, 1.5, -2.0, 1.5], [0.0, 0.0, 1.0, 0.0], 8.0, 0);

    let cc = |v: i32| ChunkCoord::from_num(v);
    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, [cc(0), cc(0), cc(0), cc(0)]);
    assert_eq!(hit.scale_exp, 2);
    assert_eq!(hit.size(), step_for_scale(2));
}

#[test]
fn scale_aware_block_target_helpers() {
    let cc = |v: i32| ChunkCoord::from_num(v);
    let target = ScaleAwareBlockTarget {
        origin: [cc(2), cc(4), cc(-6), cc(0)],
        scale_exp: 1,
    };
    assert_eq!(target.size(), step_for_scale(1));
    assert_eq!(target.world_min(), [2.0, 4.0, -6.0, 0.0]);
    assert_eq!(target.world_max(), [4.0, 6.0, -4.0, 2.0]);
}

#[test]
fn block_edit_targets_scale0_hit_scale_neg1_place() {
    // A scale-0 block at origin, hit from -Z direction.
    // Placement at scale -1 (half-size) should produce a fractional origin.
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets(
        [0.25, 0.25, -2.0, 0.25],
        [0.0, 0.0, 1.0, 0.0],
        8.0,
        -1, // placement scale -1
    );

    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, cc4([0, 0, 0, 0]));
    assert_eq!(hit.scale_exp, 0);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -1);
    let half = ChunkCoord::from_num(1) / 2; // 0.5
                                            // Face axis is Z (axis 2), face_sign = -1 (entered min face).
                                            // On face axis: origin[2] = hit.origin[2] - step = 0 - 0.5 = -0.5
    assert_eq!(place.origin[2], -half);
    // Non-face axes: hit_point snapped to 0.5 grid.
    // Ray at x=0.25 → snaps to 0.0 on 0.5 grid.
    assert_eq!(place.origin[0], cc(0));
    assert_eq!(place.origin[1], cc(0));
    assert_eq!(place.origin[3], cc(0));
    assert_eq!(place.size(), half);
}

#[test]
fn block_edit_targets_scale0_hit_scale_neg1_place_off_center() {
    // Same as above but ray hits at x=0.75 — should snap to 0.5 on the
    // half-grid for non-face axes.
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets([0.75, 0.75, -2.0, 0.25], [0.0, 0.0, 1.0, 0.0], 8.0, -1);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -1);
    let half = ChunkCoord::from_num(1) / 2;
    // On face axis (Z): -0.5
    assert_eq!(place.origin[2], -half);
    // x=0.75 snaps to 0.5 on the 0.5 grid
    assert_eq!(place.origin[0], half);
    // y=0.75 snaps to 0.5
    assert_eq!(place.origin[1], half);
    // w=0.25 snaps to 0.0
    assert_eq!(place.origin[3], cc(0));
}

#[test]
fn block_edit_targets_scale0_hit_scale_neg2_place() {
    // Place a quarter-size block (scale -2) against a scale-0 block.
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets([0.1, 0.1, -2.0, 0.1], [0.0, 0.0, 1.0, 0.0], 8.0, -2);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -2);
    let quarter = ChunkCoord::from_num(1) / 4; // 0.25
                                               // On face axis (Z): 0 - 0.25 = -0.25
    assert_eq!(place.origin[2], -quarter);
    // x=0.1 snaps to 0.0 on the 0.25 grid
    assert_eq!(place.origin[0], cc(0));
    assert_eq!(place.size(), quarter);
}

#[test]
fn block_edit_targets_scale_neg1_hit_scale_neg1_place() {
    // Place a scale -1 block against another scale -1 block.
    // The hit block is at [0,0,0,0] with size 0.5.
    let block = BlockData::simple(0, 7).at_scale(-1);
    let mut scene = make_scene_with_blocks(&[]);
    scene.world_set_block_at(cc4([0, 0, 0, 0]), block);

    let targets = scene.block_edit_targets([0.25, 0.25, -2.0, 0.25], [0.0, 0.0, 1.0, 0.0], 8.0, -1);

    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, cc4([0, 0, 0, 0]));
    assert_eq!(hit.scale_exp, -1);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -1);
    let half = ChunkCoord::from_num(1) / 2;
    // Adjacent to -Z face: 0 - 0.5 = -0.5
    assert_eq!(place.origin[2], -half);
}

#[test]
fn block_edit_targets_scale0_hit_scale_neg1_place_positive_face() {
    // Hit a scale-0 block from the +X direction.
    let scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let targets = scene.block_edit_targets([3.0, 0.25, 0.25, 0.25], [-1.0, 0.0, 0.0, 0.0], 8.0, -1);

    assert_eq!(targets.face_axis, 0); // X axis
    assert_eq!(targets.face_sign, 1); // entered through max face

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -1);
    // On face axis (X): hit.origin[0] + hit_size = 0 + 1 = 1
    assert_eq!(place.origin[0], cc(1));
    // Non-face axes snapped to 0.5 grid: 0.25 → 0.0
    assert_eq!(place.origin[1], cc(0));
    assert_eq!(place.origin[2], cc(0));
    assert_eq!(place.origin[3], cc(0));
}

#[test]
fn place_block_along_ray_sub_scale() {
    // Place a half-size block against a full-size block and verify it
    // lands at the correct fractional position.
    let mut scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 7))]);

    let material = BlockData::simple(0, 3).at_scale(-1);
    let placed = scene.place_block_along_ray(
        [0.25, 0.25, -2.0, 0.25],
        [0.0, 0.0, 1.0, 0.0],
        8.0,
        material,
    );

    let half = ChunkCoord::from_num(1) / 2;
    let expected_origin = [cc(0), cc(0), -half, cc(0)];
    assert_eq!(placed, Some(expected_origin));
    // The block should be readable at the placed position.
    assert!(!scene.get_block_data_at(expected_origin).is_air());
    // The original block should still be there.
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 7));
}

#[test]
fn remove_block_along_ray_sub_scale() {
    // Place a half-size block, then remove it via raycast.
    let block = BlockData::simple(0, 5).at_scale(-1);
    let mut scene = make_scene_with_blocks(&[]);
    scene.world_set_block_at(cc4([0, 0, 0, 0]), block.clone());

    let removed = scene.remove_block_along_ray([0.25, 0.25, -2.0, 0.25], [0.0, 0.0, 1.0, 0.0], 8.0);

    assert_eq!(removed, Some(cc4([0, 0, 0, 0])));
    assert!(scene.get_block_data_at(cc4([0, 0, 0, 0])).is_air());
}

#[test]
fn block_edit_targets_negative_coord_scale_neg1() {
    // Test that sub-scale placement works correctly with negative world
    // coordinates, ensuring div_euclid snaps properly.
    let scene = make_scene_with_blocks(&[([0, 0, -1, 0], BlockData::simple(0, 7))]);

    // Hit the block at [0,0,-1,0] from -Z direction (ray at z=-3)
    let targets = scene.block_edit_targets([0.25, 0.25, -3.0, 0.25], [0.0, 0.0, 1.0, 0.0], 8.0, -1);

    let hit = targets.hit.unwrap();
    assert_eq!(hit.origin, cc4([0, 0, -1, 0]));
    assert_eq!(hit.scale_exp, 0);

    let place = targets.place.unwrap();
    assert_eq!(place.scale_exp, -1);
    let half = ChunkCoord::from_num(1) / 2;
    // On face axis (Z): hit.origin[2] - step = -1 - 0.5 = -1.5
    let expected_z = cc(-1) - half;
    assert_eq!(place.origin[2], expected_z);
}

#[test]
fn resolve_player_collision_lands_on_voxel_surface() {
    let scene = make_scene_with_blocks(&[([0, -1, 0, 0], BlockData::simple(0, 3))]);

    let old_pos = [0.5, 2.4, 0.5, 0.5];
    let attempted = [0.5, 1.1, 0.5, 0.5];
    let mut velocity_y = -5.0;
    let (resolved, grounded) = scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

    assert!(grounded);
    assert!((resolved[1] - PLAYER_HEIGHT).abs() < 0.02);
    assert_eq!(velocity_y, 0.0);
}

#[test]
fn resolve_player_collision_blocks_horizontal_motion() {
    let scene = make_scene_with_blocks(&[([1, 0, 0, 0], BlockData::simple(0, 3))]);

    let old_pos = [0.35, 1.2, 0.5, 0.5];
    let attempted = [1.6, 1.2, 0.5, 0.5];
    let mut velocity_y = 0.0;
    let (resolved, _grounded) = scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

    // Player radius keeps center below x=0.7 when blocked by voxel slab at x>=1.
    assert!(resolved[0] < 0.72);
}

#[test]
fn resolve_player_collision_lands_on_hard_world_floor() {
    let mut scene = make_scene_with_blocks(&[]);
    scene.world_bounds = WorldBounds {
        min: [None, Some(-4.0), None, None],
        max: [None; 4],
    };

    let old_pos = [0.0, 0.4, 0.0, 0.0];
    let attempted = [0.0, -6.0, 0.0, 0.0];
    let mut velocity_y = -9.0;
    let (resolved, grounded) = scene.resolve_player_collision(old_pos, attempted, &mut velocity_y);

    assert!(grounded);
    assert!((resolved[1] - (-4.0 + PLAYER_HEIGHT)).abs() < 0.03);
    assert_eq!(velocity_y, 0.0);
}

#[test]
fn edit_back_to_air_removes_voxel_from_world_tree() {
    let mut scene = make_scene_with_blocks(&[([0, 0, 0, 0], BlockData::simple(0, 3))]);
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 3));

    scene.world_set_block(0, 0, 0, 0, BlockData::AIR);
    assert!(scene.get_block_data(0, 0, 0, 0).is_air());
    assert!(!scene.world_tree.has_chunk(chunk_key_i32(0, 0, 0, 0)));
}

#[test]
fn apply_region_patch_splices_chunk_and_queues_update() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key_i32(0, 0, 0, 0), 0);
    let mut patch_tree = RegionChunkTree::new();
    let changed = patch_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7))),
    );
    assert!(changed);
    let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

    let stats = scene.apply_region_patch(bounds, &patch_core);

    assert_eq!(stats.previous_non_empty, 0);
    assert_eq!(stats.desired_non_empty, 1);
    assert_eq!(stats.changed_chunks, 1);
    assert_eq!(stats.previous_total_chunks, 0);
    assert_eq!(stats.desired_total_chunks, 1);
    assert!(stats.collect_previous_ms >= 0.0);
    assert!(stats.splice_ms >= 0.0);
    assert!(stats.collect_desired_ms >= 0.0);
    assert!(stats.diff_ms >= 0.0);
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 7));
    assert_eq!(
        scene.world_drain_pending_chunk_updates(),
        Vec::<ChunkKey>::new()
    );
}

#[test]
fn apply_region_patch_removes_chunk_and_queues_update() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key_i32(0, 0, 0, 0), 0);
    let mut seed_tree = RegionChunkTree::new();
    let _ = seed_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 5))),
    );
    let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
    let _ = scene.apply_region_patch(bounds, &seed_core);
    let _ = scene.world_drain_pending_chunk_updates();

    let empty_core = RegionTreeCore {
        bounds,
        kind: polychora::shared::region_tree::RegionNodeKind::Empty,
        generator_version_hash: 0,
    };
    let stats = scene.apply_region_patch(bounds, &empty_core);

    assert_eq!(stats.previous_non_empty, 1);
    assert_eq!(stats.desired_non_empty, 0);
    assert_eq!(stats.changed_chunks, 1);
    assert_eq!(stats.previous_total_chunks, 1);
    assert_eq!(stats.desired_total_chunks, 0);
    assert!(stats.collect_previous_ms >= 0.0);
    assert!(stats.splice_ms >= 0.0);
    assert!(stats.collect_desired_ms >= 0.0);
    assert!(stats.diff_ms >= 0.0);
    assert!(scene.get_block_data(0, 0, 0, 0).is_air());
    assert_eq!(
        scene.world_drain_pending_chunk_updates(),
        Vec::<ChunkKey>::new()
    );
}

#[test]
fn apply_region_patch_identical_patch_is_noop() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::chunk_world_bounds(chunk_key_i32(0, 0, 0, 0), 0);
    let mut patch_tree = RegionChunkTree::new();
    let _ = patch_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7))),
    );
    let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

    let _ = scene.apply_region_patch(bounds, &patch_core);
    let _ = scene.world_drain_pending_chunk_updates();

    let stats = scene.apply_region_patch(bounds, &patch_core);
    assert_eq!(stats.previous_non_empty, 1);
    assert_eq!(stats.desired_non_empty, 1);
    assert_eq!(stats.upserts, 0);
    assert_eq!(stats.removals, 0);
    assert_eq!(stats.changed_chunks, 0);
    assert_eq!(stats.previous_total_chunks, 1);
    assert_eq!(stats.desired_total_chunks, 1);
    assert_eq!(stats.invalidated_cached_chunks, 0);
    assert_eq!(stats.queued_updates, 0);
    assert_eq!(
        scene.world_drain_pending_chunk_updates(),
        Vec::<ChunkKey>::new()
    );
}

#[test]
fn apply_region_patch_semantic_noop_skips_splice() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::from_i32([0, 0, 0, 0], [16, 8, 8, 8]);

    let mut seed_tree = RegionChunkTree::new();
    let _ = seed_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 3))),
    );
    let _ = seed_tree.set_chunk(
        chunk_key_i32(1, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4))),
    );
    let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
    let _ = scene.apply_region_patch(bounds, &seed_core);
    let _ = scene.world_drain_pending_chunk_updates();
    let before = scene.world_tree.root().cloned();

    let patch_block_palette = vec![
        BlockData::AIR,
        BlockData::simple(0, 3),
        BlockData::simple(0, 4),
    ];
    let patch_chunk_array =
        polychora::shared::chunk_payload::ChunkArrayData::from_dense_indices_with_block_palette(
            bounds,
            vec![ChunkPayload::Uniform(1), ChunkPayload::Uniform(2)],
            vec![0, 1],
            None,
            patch_block_palette,
            0,
        )
        .expect("chunk array");
    let patch_core = RegionTreeCore {
        bounds,
        kind: polychora::shared::region_tree::RegionNodeKind::ChunkArray(patch_chunk_array),
        generator_version_hash: 0,
    };

    let stats = scene.apply_region_patch(bounds, &patch_core);
    assert_eq!(stats.changed_chunks, 0);
    assert_eq!(stats.upserts, 0);
    assert_eq!(stats.removals, 0);
    assert_eq!(stats.queued_updates, 0);
    assert_eq!(stats.invalidated_cached_chunks, 0);
    assert!(stats.splice_ms >= 0.0);
    assert_eq!(scene.world_tree.root().cloned(), before);
    assert_eq!(
        scene.world_drain_pending_chunk_updates(),
        Vec::<ChunkKey>::new()
    );
}

#[test]
fn apply_region_patch_fast_matches_full_splice_state() {
    let bounds = Aabb4i::from_i32([0, 0, 0, 0], [16, 8, 8, 8]);
    let mut full_scene = Scene::new(ScenePreset::Empty);
    let mut fast_scene = Scene::new(ScenePreset::Empty);

    let mut seed_tree = RegionChunkTree::new();
    let _ = seed_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 3))),
    );
    let _ = seed_tree.set_chunk(
        chunk_key_i32(1, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 4))),
    );
    let seed_core = seed_tree.slice_non_empty_core_in_bounds(bounds);
    let _ = full_scene.apply_region_patch(bounds, &seed_core);
    let _ = fast_scene.apply_region_patch(bounds, &seed_core);
    let _ = full_scene.world_drain_pending_chunk_updates();
    let _ = fast_scene.world_drain_pending_chunk_updates();

    let mut patch_tree = RegionChunkTree::new();
    let _ = patch_tree.set_chunk(
        chunk_key_i32(0, 0, 0, 0),
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 7))),
    );
    let patch_core = patch_tree.slice_non_empty_core_in_bounds(bounds);

    let full_stats = full_scene.apply_region_patch(bounds, &patch_core);
    let fast_stats = fast_scene.apply_region_patch_fast(bounds, &patch_core);

    assert_eq!(full_stats.changed_chunks, 1);
    assert_eq!(fast_stats.changed_chunks, 1);
    assert_eq!(full_scene.world_tree.root(), fast_scene.world_tree.root());
    assert_eq!(
        full_scene.world_tree_revision,
        fast_scene.world_tree_revision
    );
    assert_eq!(
        full_scene.world_drain_pending_chunk_updates(),
        fast_scene.world_drain_pending_chunk_updates()
    );
    assert_eq!(
        full_scene.get_block_data(0, 0, 0, 0),
        BlockData::simple(0, 7)
    );
    assert_eq!(
        fast_scene.get_block_data(0, 0, 0, 0),
        BlockData::simple(0, 7)
    );
    assert!(full_scene
        .get_block_data(CHUNK_SIZE as i32, 0, 0, 0)
        .is_air());
    assert!(fast_scene
        .get_block_data(CHUNK_SIZE as i32, 0, 0, 0)
        .is_air());
}

#[test]
fn apply_region_patch_fast_updates_render_bvh_for_chunk_payload_changes() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let chunk_key = chunk_key_i32(0, 0, 0, 0);
    let chunk_bounds = Aabb4i::chunk_world_bounds(chunk_key, 0);

    // Seed one non-empty chunk.
    let mut initial_patch = RegionChunkTree::new();
    let _ = initial_patch.set_chunk(
        chunk_key,
        Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 11))),
    );
    let initial_core = initial_patch.slice_non_empty_core_in_bounds(chunk_bounds);
    let initial_stats = scene.apply_region_patch_fast(chunk_bounds, &initial_core);
    assert_eq!(initial_stats.changed_chunks, 1);
    // Verify through the world tree (block_palette-aware path).
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 11));

    // Update the same chunk with a dense payload that changes only one voxel.
    let mut edited_dense = vec![11u16; CHUNK_VOLUME];
    edited_dense[0] = 4;
    // Use a trivial identity palette (index N → BlockData::simple(0, N))
    // since this test checks patch mechanics, not content identity.
    let test_palette: Vec<BlockData> = (0..=11u32).map(|i| BlockData::simple(0, i)).collect();
    let edited_payload = ResolvedChunkPayload {
        payload: ChunkPayload::from_dense_materials_compact(&edited_dense)
            .expect("compact dense payload"),
        block_palette: test_palette,
    };
    let mut edited_patch = RegionChunkTree::new();
    let _ = edited_patch.set_chunk(chunk_key, Some(edited_payload));
    let edited_core = edited_patch.slice_non_empty_core_in_bounds(chunk_bounds);
    let edited_stats = scene.apply_region_patch_fast(chunk_bounds, &edited_core);
    assert_eq!(edited_stats.changed_chunks, 1);

    // Verify the edit is visible through the world tree.
    let (edited_world, _) = scene
        .debug_world_tree_chunk_payload(chunk_key)
        .expect("edited chunk");
    assert_eq!(edited_world.block_at(0), BlockData::simple(0, 4));
    assert_eq!(scene.get_block_data(0, 0, 0, 0), BlockData::simple(0, 4));
}

#[test]
fn voxel_scene_dirty_tracking_rebuild_clears_only_overlapping_chunks() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::from_i32([-16, -16, -16, -16], [24, 24, 24, 24]);

    // Prime cache.
    scene.prime_render_bvh_cache_for_bounds(bounds);
    assert_eq!(scene.active_config.render_bvh_cache_bounds, Some(bounds));
    assert!(scene.voxel_pending_scene_dirty_regions.is_empty());

    // Add one dirty chunk inside bounds and one far outside.
    scene.world_set_block(0, 0, 0, 0, BlockData::simple(0, 3));
    let far_world_x = (CHUNK_SIZE as i32) * 50;
    scene.world_set_block(far_world_x, 0, 0, 0, BlockData::simple(0, 4));

    let far_key = chunk_key_i32(50, 0, 0, 0);
    assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), 2);
    assert!(scene
        .voxel_pending_scene_dirty_regions
        .iter()
        .any(|region| region.contains_chunk_world_min(chunk_key_i32(0, 0, 0, 0))));
    assert!(scene
        .voxel_pending_scene_dirty_regions
        .iter()
        .any(|region| region.contains_chunk_world_min(far_key)));

    // Rebuild for local bounds should consume local dirty key only.
    scene.ensure_render_bvh_cache_for_bounds(bounds);
    assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), 1);
    assert!(scene
        .voxel_pending_scene_dirty_regions
        .iter()
        .any(|region| region.contains_chunk_world_min(far_key)));
}

#[test]
fn voxel_scene_dirty_tracking_offscreen_edits_do_not_invalidate_local_cache() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::from_i32([-16, -16, -16, -16], [24, 24, 24, 24]);

    // Prime cache once.
    scene.prime_render_bvh_cache_for_bounds(bounds);
    assert_eq!(scene.active_config.render_bvh_cache_bounds, Some(bounds));

    // Edit a far chunk; local bounds should remain cache-valid.
    let far_world_x = (CHUNK_SIZE as i32) * 60;
    scene.world_set_block(far_world_x, 0, 0, 0, BlockData::simple(0, 5));
    let far_key = chunk_key_i32(60, 0, 0, 0);
    assert!(scene
        .voxel_pending_scene_dirty_regions
        .iter()
        .any(|region| region.contains_chunk_world_min(far_key)));
    assert!(!scene.voxel_scene_bounds_has_pending_dirty_regions(bounds));

    // Re-requesting local cache should keep far dirty queued.
    scene.ensure_render_bvh_cache_for_bounds(bounds);
    assert!(scene
        .voxel_pending_scene_dirty_regions
        .iter()
        .any(|region| region.contains_chunk_world_min(far_key)));
    assert_eq!(scene.active_config.render_bvh_cache_bounds, Some(bounds));
}

#[test]
fn voxel_scene_dirty_budget_limits_chunks_per_rebuild() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let bounds = Aabb4i::from_i32([-16, -16, -16, -16], [1608, 24, 24, 24]);

    // Prime cache.
    scene.prime_render_bvh_cache_for_bounds(bounds);
    assert_eq!(scene.active_config.render_bvh_cache_bounds, Some(bounds));

    // Mark more dirty chunks than per-frame budget.
    let dirty_count = VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET + 5;
    for i in 0..dirty_count {
        let wx = (i as i32) * CHUNK_SIZE as i32;
        scene.world_set_block(wx, 0, 0, 0, BlockData::simple(0, 7));
    }
    assert_eq!(scene.voxel_pending_scene_dirty_regions.len(), dirty_count);

    // One rebuild pass should only consume the budgeted amount.
    scene.ensure_render_bvh_cache_for_bounds(bounds);
    let expected_remaining = dirty_count.saturating_sub(VOXEL_SCENE_DIRTY_REGION_UPDATE_BUDGET);
    assert_eq!(
        scene.voxel_pending_scene_dirty_regions.len(),
        expected_remaining
    );

    // Additional passes should eventually consume the remainder.
    while !scene.voxel_pending_scene_dirty_regions.is_empty() {
        scene.ensure_render_bvh_cache_for_bounds(bounds);
    }
    assert!(scene.voxel_pending_scene_dirty_regions.is_empty());
}

#[test]
fn voxel_frame_snapshot_path_clears_mutation_batch() {
    let mut scene = Scene::new(ScenePreset::Empty);
    scene.world_set_block(0, 0, 0, 0, BlockData::simple(0, 3));
    let _ = scene.build_voxel_frame_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        64.0,
        &test_resolver(),
    );
    scene.flush_voxel_background_rebuild();
    assert!(scene.active_config.frame_data.mutation_batch.is_none());
    assert!(scene
        .active_config
        .frame_data
        .mutation_base_generation
        .is_none());
}

#[test]
fn voxel_frame_delta_path_emits_mutation_batch() {
    let mut scene = Scene::new(ScenePreset::Empty);
    scene.world_set_block(0, 0, 0, 0, BlockData::simple(0, 3));
    let _ = scene.build_voxel_frame_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        64.0,
        &test_resolver(),
    );
    scene.flush_voxel_background_rebuild();
    assert!(scene.active_config.frame_data.mutation_batch.is_none());
    let base_generation = scene.active_config.frame_data.metadata_generation;

    scene.world_set_block(CHUNK_SIZE as i32, 0, 0, 0, BlockData::simple(0, 4));
    let _ = scene.build_voxel_frame_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        64.0,
        &test_resolver(),
    );

    let batch = scene.active_config.frame_data.mutation_batch.as_ref();
    assert!(batch.is_some());
    let batch = batch.unwrap();
    assert_eq!(
        scene.active_config.frame_data.mutation_base_generation,
        Some(base_generation)
    );
    assert!(
        !batch.chunk_header_writes.is_empty()
            || !batch.occupancy_word_writes.is_empty()
            || !batch.material_word_writes.is_empty()
            || !batch.macro_word_writes.is_empty()
            || !batch.region_bvh_node_writes.is_empty()
            || !batch.leaf_header_writes.is_empty()
            || !batch.leaf_chunk_entry_writes.is_empty()
    );
}

#[test]
fn voxel_frame_delta_root_mismatch_forces_snapshot_rebuild() {
    let mut scene = Scene::new(ScenePreset::Empty);
    scene.world_set_block(0, 0, 0, 0, BlockData::simple(0, 3));
    let _ = scene.build_voxel_frame_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        64.0,
        &test_resolver(),
    );
    scene.flush_voxel_background_rebuild();

    let frame_root = scene.active_config.frame_data.region_bvh_root_index;
    assert_ne!(frame_root, VTE_REGION_BVH_INVALID_NODE);
    assert!(scene.active_config.render_bvh_cache.is_some());

    scene.active_config.pending_render_bvh_rebuild = false;
    scene
        .active_config
        .pending_render_bvh_mutation_deltas
        .clear();
    scene
        .active_config
        .pending_render_bvh_mutation_deltas
        .push(RenderBvhChunkMutationDelta {
            key: chunk_key_i32(0, 0, 0, 0),
            expected_root: Some(frame_root.wrapping_add(1)),
            new_root: Some(frame_root),
            node_writes: Vec::new(),
            leaf_writes: Vec::new(),
            freed_node_ids: Vec::new(),
            freed_leaf_ids: Vec::new(),
        });
    scene.voxel_cached_visibility_bounds = None;

    let _ = scene.build_voxel_frame_data(
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        64.0,
        &test_resolver(),
    );
    scene.flush_voxel_background_rebuild();

    assert!(scene
        .active_config
        .pending_render_bvh_mutation_deltas
        .is_empty());
    assert!(scene.active_config.frame_data.mutation_batch.is_none());
    assert!(scene
        .active_config
        .frame_data
        .mutation_base_generation
        .is_none());
    let cpu_root = scene
        .active_config
        .render_bvh_cache
        .as_ref()
        .and_then(|bvh| bvh.root)
        .unwrap_or(VTE_REGION_BVH_INVALID_NODE);
    assert_eq!(
        scene.active_config.frame_data.region_bvh_root_index,
        cpu_root
    );
}

#[test]
fn voxel_frame_delta_updates_match_world_after_flat_floor_edit() {
    let mut scene = Scene::new(ScenePreset::Flat);
    let cam = [0.0, 2.0, 0.0, 0.0];

    let _ = scene.build_voxel_frame_data(cam, [0.0, 0.0, 1.0, 0.0], 96.0, &test_resolver());
    scene.flush_voxel_background_rebuild();

    let edit_voxel = [0, -1, 0, 0];
    let before_world =
        scene.get_block_data(edit_voxel[0], edit_voxel[1], edit_voxel[2], edit_voxel[3]);
    assert!(!before_world.is_air());
    let before_frame = sample_voxel_from_frame(
        &scene,
        edit_voxel[0],
        edit_voxel[1],
        edit_voxel[2],
        edit_voxel[3],
    );
    let reg = test_registry();
    let before_mat =
        reg.block_material_token(before_world.namespace, before_world.block_type) as u8;
    assert_eq!(before_frame, Some(before_mat));

    scene.world_set_block(
        edit_voxel[0],
        edit_voxel[1],
        edit_voxel[2],
        edit_voxel[3],
        BlockData::AIR,
    );
    let _ = scene.build_voxel_frame_data(cam, [0.0, 0.0, 1.0, 0.0], 96.0, &test_resolver());

    let after_world =
        scene.get_block_data(edit_voxel[0], edit_voxel[1], edit_voxel[2], edit_voxel[3]);
    assert!(after_world.is_air());
    let after_frame = sample_voxel_from_frame(
        &scene,
        edit_voxel[0],
        edit_voxel[1],
        edit_voxel[2],
        edit_voxel[3],
    );
    assert!(after_frame.is_none());
}

#[test]
fn voxel_frame_dense_cache_respects_block_palette_at_scale_neg3() {
    fn first_non_air_material(payloads: &[ChunkPayload]) -> Option<u16> {
        payloads.iter().find_map(|payload| match payload {
            ChunkPayload::Empty => None,
            ChunkPayload::Uniform(material) => (*material != 0).then_some(*material),
            _ => payload
                .dense_materials()
                .ok()
                .and_then(|dense| dense.into_iter().find(|material| *material != 0)),
        })
    }

    let mut scene = Scene::new(ScenePreset::Empty);
    let resolver = test_resolver();
    let cobblestone = polychora::content_registry::block_data_from_material_token(28).at_scale(-3);
    let red = polychora::content_registry::block_data_from_material_token(1).at_scale(-3);

    // Place two scale=-3 chunks with identical sparse payload shape but
    // different block palettes/material IDs.
    scene.world_set_block(0, 0, 0, 0, cobblestone.clone());
    scene.world_set_block(1, 0, 0, 0, red.clone());

    let _ =
        scene.build_voxel_frame_data([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], 64.0, &resolver);
    scene.flush_voxel_background_rebuild();

    let cobblestone_expected =
        resolver.resolve_block(cobblestone.namespace, cobblestone.block_type);
    let red_expected = resolver.resolve_block(red.namespace, red.block_type);
    assert_ne!(cobblestone_expected, red_expected);

    let cobblestone_frame = scene.debug_voxel_frame_chunk_payloads([0, 0, 0, 0]);
    let red_frame = scene.debug_voxel_frame_chunk_payloads([1, 0, 0, 0]);
    let cobblestone_material = first_non_air_material(&cobblestone_frame);
    let red_material = first_non_air_material(&red_frame);

    assert_eq!(
        cobblestone_material,
        Some(cobblestone_expected),
        "unexpected material in scale=-3 chunk [0,0,0,0], payloads={cobblestone_frame:?}",
    );
    assert_eq!(
        red_material,
        Some(red_expected),
        "unexpected material in scale=-3 chunk [1,0,0,0], payloads={red_frame:?}",
    );
}

/// Simulate the multiplayer FlatFloor scenario: server sends a Uniform(stone)
/// node covering many chunks at Y=-1 via apply_region_patch. Verify that
/// get_block_data, look-ray, and collision all work across a range of positions.
#[test]
fn uniform_floor_patch_query_collision_and_lookray_work_across_range() {
    use polychora::shared::spatial::ChunkCoord;

    let mut scene = Scene::new(ScenePreset::Empty);
    let stone = BlockData::simple(0, 7);
    let cs = ChunkCoord::from_num(CHUNK_SIZE as i32);
    let floor_bounds = Aabb4i::new(
        [
            ChunkCoord::from_num(-10i32) * cs,
            ChunkCoord::from_num(-1i32) * cs,
            ChunkCoord::from_num(-10i32) * cs,
            ChunkCoord::from_num(-10i32) * cs,
        ],
        [
            (ChunkCoord::from_num(10i32) + ChunkCoord::from_num(1)) * cs,
            (ChunkCoord::from_num(-1i32) + ChunkCoord::from_num(1)) * cs,
            (ChunkCoord::from_num(10i32) + ChunkCoord::from_num(1)) * cs,
            (ChunkCoord::from_num(10i32) + ChunkCoord::from_num(1)) * cs,
        ],
    );
    let floor_core = RegionTreeCore {
        bounds: floor_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };
    let stats = scene.apply_region_patch(floor_bounds, &floor_core);
    assert!(stats.changed_chunks > 0, "patch should have applied");

    // Verify get_block_data for voxels within the floor.
    // The floor covers chunks at Y=-1, which means world Y=-8 to Y=-1 (inclusive).
    let mut failures = Vec::new();
    for x in [-80, -40, -8, 0, 7, 40, 79] {
        for z in [-40, 0, 40] {
            for w in [0] {
                for y in [-8, -5, -1] {
                    let block = scene.get_block_data(x, y, z, w);
                    if block.is_air() {
                        failures.push(format!(
                            "get_block_data({x},{y},{z},{w}) returned AIR, expected stone"
                        ));
                    }
                }
                // Above the floor should be air.
                let block = scene.get_block_data(x, 0, z, w);
                if !block.is_air() {
                    failures.push(format!(
                        "get_block_data({x},0,{z},{w}) returned solid, expected AIR"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Block query failures:\n{}",
        failures.join("\n")
    );

    // Verify look-ray hits the floor when pointing downward from above.
    for x in [-40, 0, 40] {
        let origin = [x as f32 + 0.5, 2.0, 0.5, 0.5];
        let direction = [0.0, -1.0, 0.0, 0.0];
        let targets = scene.block_edit_targets(origin, direction, 20.0, 0);
        assert!(
            targets.hit.is_some(),
            "look-ray at x={x} should hit the floor, got None"
        );
        let hit = targets.hit.unwrap().origin_i32();
        assert_eq!(
            hit[1], -1,
            "look-ray at x={x} should hit at y=-1, got y={}",
            hit[1]
        );
    }

    // Verify collision detects the floor.
    for x in [-40, 0, 40] {
        let old_pos = [x as f32 + 0.5, 2.4, 0.5, 0.5];
        let attempted = [x as f32 + 0.5, -0.5, 0.5, 0.5];
        let mut vel = -5.0;
        let (resolved, grounded) = scene.resolve_player_collision(old_pos, attempted, &mut vel);
        assert!(
            grounded,
            "player at x={x} should be grounded, resolved_y={}",
            resolved[1]
        );
    }
}

/// Simulates the multiplayer flow: server sends a flat floor patch (subtree bounds
/// thinner than authoritative bounds), client applies via apply_region_patch_fast,
/// then verifies queries work.
#[test]
fn multiplayer_flat_floor_patch_fast_then_query() {
    let mut scene = Scene::new(ScenePreset::Empty);
    let stone = BlockData::simple(0, 7);

    // Simulate server sending a flat floor response:
    // - Client requests bounds [-5,-5,-5,-5] to [5,5,5,5] (the authoritative_bounds)
    // - Server responds with Uniform(stone) floor at Y=-1 only (subtree bounds thinner)
    let authoritative_bounds = Aabb4i::from_i32([-40, -40, -40, -40], [48, 48, 48, 48]);
    let floor_bounds = Aabb4i::from_i32([-40, -8, -40, -40], [48, 0, 48, 48]);
    let subtree = RegionTreeCore {
        bounds: floor_bounds,
        kind: RegionNodeKind::Uniform(stone.clone()),
        generator_version_hash: 0,
    };

    // Use apply_region_patch_fast (the default multiplayer path)
    let stats = scene.apply_region_patch_fast(authoritative_bounds, &subtree);
    assert!(
        stats.changed_chunks > 0,
        "patch should have applied, stats={:?}",
        stats
    );

    // Verify block queries at the floor level
    let mut query_failures = Vec::new();
    for x in [-40, -8, -1, 0, 1, 7, 39] {
        for z in [-40, 0, 39] {
            for w in [0] {
                for y in [-8, -5, -1] {
                    let block = scene.get_block_data(x, y, z, w);
                    if block.is_air() {
                        query_failures.push(format!(
                            "get_block_data({x},{y},{z},{w}) = AIR, expected stone"
                        ));
                    }
                }
                // Above floor should be air
                let block = scene.get_block_data(x, 0, z, w);
                if !block.is_air() {
                    query_failures.push(format!(
                        "get_block_data({x},0,{z},{w}) = solid, expected AIR"
                    ));
                }
            }
        }
    }
    assert!(
        query_failures.is_empty(),
        "Block query failures after multiplayer patch:\n{}",
        query_failures.join("\n")
    );

    // Verify look-ray hits floor
    let targets = scene.block_edit_targets([0.5, 2.0, 0.5, 0.5], [0.0, -1.0, 0.0, 0.0], 20.0, 0);
    assert!(targets.hit.is_some(), "look-ray should hit floor");
    assert_eq!(
        targets.hit.unwrap().origin_i32()[1],
        -1,
        "look-ray should hit at y=-1"
    );
}

fn dump_tree_structure(core: &RegionTreeCore, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    let b = &core.bounds;
    let bounds_str = format!(
        "[{},{},{},{}]->[{},{},{},{}]",
        b.min[0].to_num::<i32>(),
        b.min[1].to_num::<i32>(),
        b.min[2].to_num::<i32>(),
        b.min[3].to_num::<i32>(),
        b.max[0].to_num::<i32>(),
        b.max[1].to_num::<i32>(),
        b.max[2].to_num::<i32>(),
        b.max[3].to_num::<i32>(),
    );
    match &core.kind {
        RegionNodeKind::Empty => format!("{prefix}Empty {bounds_str}\n"),
        RegionNodeKind::Uniform(block) => {
            format!(
                "{prefix}Uniform(ns={},bt={}) {bounds_str}\n",
                block.namespace, block.block_type
            )
        }
        RegionNodeKind::ProceduralRef(_) => format!("{prefix}ProceduralRef {bounds_str}\n"),
        RegionNodeKind::ChunkArray(ca) => {
            let cells = ca
                .bounds
                .chunk_cell_count_at_scale(ca.scale_exp)
                .unwrap_or(0);
            format!(
                "{prefix}ChunkArray(cells={}, palette_len={}, scale={}) {bounds_str}\n",
                cells,
                ca.chunk_palette.len(),
                ca.scale_exp
            )
        }
        RegionNodeKind::Branch(children) => {
            let mut s = format!("{prefix}Branch({} children) {bounds_str}\n", children.len());
            for child in children {
                s += &dump_tree_structure(child, indent + 2);
            }
            s
        }
    }
}

/// Creates a MassivePlatforms world, queries it like the server does at
/// render distance ~200 blocks, applies the subtree to a Scene, then
/// verifies that every non-empty chunk in the subtree is queryable via
/// get_block_data.
#[test]
fn massive_platforms_world_query_roundtrip() {
    use polychora::server::world_field::{
        MassivePlatformsWorldGenerator, QueryDetail, QueryVolume,
    };
    use polychora::shared::region_tree::collect_non_empty_chunks_from_core_in_bounds;

    let material = BlockData::simple(0, 11);
    let generator = MassivePlatformsWorldGenerator::from_chunk_payloads(
        polychora::shared::voxel::BaseWorldKind::MassivePlatforms {
            material: material.clone(),
        },
        Vec::<([i32; 4], ChunkPayload)>::new(),
        1337,
        true, // procgen structures enabled
        std::collections::HashSet::new(),
    );

    // 200 world units radius
    let radius = 200;
    let authoritative_bounds = Aabb4i::from_i32(
        [-radius, -radius, -radius, -radius],
        [radius, radius, radius, radius],
    );

    let subtree = generator.query_region_core(
        QueryVolume {
            bounds: authoritative_bounds,
        },
        QueryDetail::Exact,
    );

    eprintln!("=== World tree from server ===");
    eprintln!("{}", dump_tree_structure(&subtree, 0));

    // Collect all non-empty chunks from the subtree
    let non_empty_chunks =
        collect_non_empty_chunks_from_core_in_bounds(&subtree, authoritative_bounds);
    eprintln!("Non-empty chunks in subtree: {}", non_empty_chunks.len());

    // Apply to scene like the multiplayer path does
    let mut scene = Scene::new(ScenePreset::Empty);
    let stats = scene.apply_region_patch_fast(authoritative_bounds, &subtree);
    eprintln!("Patch stats: changed_chunks={}", stats.changed_chunks);

    // Dump the resulting client tree
    if let Some(root) = scene.world_tree.root() {
        eprintln!("=== Client world tree after patch ===");
        eprintln!("{}", dump_tree_structure(root, 0));
    } else {
        eprintln!("Client world tree is EMPTY after patch!");
    }

    // For every non-empty chunk, verify that get_block_data returns
    // non-air for at least some voxels within that chunk.
    let mut missing_chunks = Vec::new();
    for (key, payload) in &non_empty_chunks {
        // Convert chunk key back to world coords: chunk_lattice * 8
        let cx = key[0].to_num::<i32>();
        let cy = key[1].to_num::<i32>();
        let cz = key[2].to_num::<i32>();
        let cw = key[3].to_num::<i32>();
        let wx_base = cx * 8;
        let wy_base = cy * 8;
        let wz_base = cz * 8;
        let ww_base = cw * 8;

        // Check if any voxel in this chunk is non-air according to get_block_data
        let mut found_any = false;
        for local_idx in 0..CHUNK_VOLUME {
            let lw = local_idx / (8 * 8 * 8);
            let lz = (local_idx / (8 * 8)) % 8;
            let ly = (local_idx / 8) % 8;
            let lx = local_idx % 8;
            let block = scene.get_block_data(
                wx_base + lx as i32,
                wy_base + ly as i32,
                wz_base + lz as i32,
                ww_base + lw as i32,
            );
            if !block.is_air() {
                found_any = true;
                break;
            }
        }

        // Also check what the payload says
        let payload_has_solid = (0..CHUNK_VOLUME).any(|i| !payload.block_at(i).is_air());

        if payload_has_solid && !found_any {
            missing_chunks.push(format!(
                "chunk [{},{},{},{}]: payload has solid blocks but get_block_data returns all AIR",
                cx, cy, cz, cw
            ));
        }
    }

    if !missing_chunks.is_empty() {
        eprintln!("=== MISSING CHUNKS ({}) ===", missing_chunks.len());
        for msg in &missing_chunks {
            eprintln!("  {}", msg);
        }
    }
    assert!(
        missing_chunks.is_empty(),
        "{} chunks have solid payloads but get_block_data returns AIR:\n{}",
        missing_chunks.len(),
        missing_chunks.join("\n")
    );
}
