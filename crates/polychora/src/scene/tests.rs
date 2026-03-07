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

    let _stats = scene.apply_region_patch(bounds, &patch_core);
    // Data must be preserved regardless of whether the splice detected a structural change.
    assert_eq!(
        scene
            .world_tree
            .chunk_payload(chunk_key_i32(0, 0, 0, 0))
            .unwrap()
            .0
            .uniform_block(),
        Some(&BlockData::simple(0, 3))
    );
    assert_eq!(
        scene
            .world_tree
            .chunk_payload(chunk_key_i32(1, 0, 0, 0))
            .unwrap()
            .0
            .uniform_block(),
        Some(&BlockData::simple(0, 4))
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
            ChunkPayload::Empty | ChunkPayload::Virgin => None,
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

/// Regression: place a block, break ground, replace ground. The replaced chunk
/// must not disappear from the world tree. Exercises the fast-path
/// apply_region_patch_fast with server-like subtrees (bounds larger than
/// authoritative_bounds).
///
/// Uses the EXACT server compose flow: splice base Uniform, then
/// overlay_core_in_bounds for overrides.
#[test]
fn place_break_replace_via_fast_patch_preserves_chunk() {
    use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
    use polychora::shared::region_tree::{RegionNodeKind, RegionTreeCore};
    use polychora::shared::voxel::CHUNK_VOLUME;

    let ground = BlockData::simple(0, 1);
    let placed = BlockData::simple(0, 99);

    /// Compose a server subtree: base Uniform + overrides via overlay.
    /// This mirrors PassthroughWorldOverlay::query_region_core.
    fn compose_server_subtree(
        platform: Aabb4i,
        base: &RegionTreeCore,
        overrides: &RegionChunkTree,
    ) -> RegionTreeCore {
        let compose_bounds = platform; // base_core.bounds for a Uniform platform
        let mut composed = RegionChunkTree::new();
        let _ = composed.splice_non_empty_core_in_bounds(platform, base);
        let override_core = overrides.slice_core_in_bounds(compose_bounds);
        let _ = composed.overlay_core_in_bounds(compose_bounds, &override_core);
        composed.root().cloned().unwrap_or(RegionTreeCore {
            bounds: compose_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        })
    }

    // Test many platform configurations to find the triggering geometry.
    let platform_configs: &[([i32; 4], [i32; 4], [i32; 4])] = &[
        // (platform_min_key, platform_max_key, chunk_a_key)
        // Start with the failing config to get fast feedback.
        ([0, 0, 0, 0], [3, 0, 3, 3], [1, 0, 1, 1]), // tiny platform
        ([0, 0, 0, 0], [15, 0, 15, 15], [4, 0, 6, 7]),
        ([0, 0, 0, 0], [15, 1, 15, 15], [4, 0, 6, 7]), // 2 thick
        ([-8, -1, -8, -8], [7, 0, 7, 7], [0, 0, 0, 0]), // centered at origin
        ([0, 0, 0, 0], [7, 0, 7, 7], [3, 0, 3, 3]),    // smaller platform
        ([0, 0, 0, 0], [19, 0, 19, 19], [10, 0, 10, 10]), // non-power-of-2
        ([5, 0, 5, 5], [14, 0, 14, 14], [8, 0, 8, 8]), // offset from origin
    ];

    let b_offsets: &[[i32; 4]] = &[
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [3, 0, 3, 3],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [-1, 0, -1, 0],
        [-1, 0, 0, -1],
        [0, 0, -1, -1],
    ];

    // Additional placement positions (simulates "place a few random blocks").
    let extra_offsets: &[[i32; 4]] = &[[-2, 0, -3, -4], [3, 0, 2, 1]];

    for &(pmin, pmax, akey) in platform_configs {
        let platform = Aabb4i::from_lattice_bounds(pmin, pmax, 0);
        let platform_core = RegionTreeCore {
            bounds: platform,
            kind: RegionNodeKind::Uniform(ground.clone()),
            generator_version_hash: 0,
        };
        let chunk_a_key = chunk_key_i32(akey[0], akey[1], akey[2], akey[3]);
        let chunk_a_bounds = Aabb4i::chunk_world_bounds(chunk_a_key, 0);

        for offset in b_offsets {
            let bx = akey[0] + offset[0];
            let by = akey[1] + offset[1];
            let bz = akey[2] + offset[2];
            let bw = akey[3] + offset[3];
            let chunk_b_key = chunk_key_i32(bx, by, bz, bw);
            let chunk_b_bounds = Aabb4i::chunk_world_bounds(chunk_b_key, 0);
            if !platform.contains_bounds(&chunk_b_bounds) || chunk_b_key == chunk_a_key {
                continue;
            }
            let extra_placements: Vec<[i32; 4]> = extra_offsets
                .iter()
                .map(|eo| {
                    [
                        akey[0] + eo[0],
                        akey[1] + eo[1],
                        akey[2] + eo[2],
                        akey[3] + eo[3],
                    ]
                })
                .collect();

            // Fresh scene for each position.
            let mut scene = Scene::new(ScenePreset::Empty);
            scene.apply_region_patch_fast(platform, &platform_core);

            // --- Build override trees as the server would ---

            // A override: ground with one placed block (realistic Dense16 — as if a
            // single block was placed in an otherwise-ground chunk).
            let a_block_palette = vec![BlockData::AIR, ground.clone(), placed.clone()];
            let mut a_dense = vec![1u16; CHUNK_VOLUME]; // all ground
            a_dense[0] = 2; // placed block at cell 0
            let a_ca = ChunkArrayData::from_dense_indices_with_block_palette(
                chunk_a_bounds,
                vec![ChunkPayload::Dense16 { materials: a_dense }],
                vec![0],
                None,
                a_block_palette,
                0,
            )
            .expect("chunk array A");
            let a_core = RegionTreeCore {
                bounds: chunk_a_bounds,
                kind: RegionNodeKind::ChunkArray(a_ca),
                generator_version_hash: 0,
            };

            // Build extra placement overrides (simulates "place a few random blocks").
            let mut extra_cores: Vec<(Aabb4i, RegionTreeCore)> = Vec::new();
            for ep in &extra_placements {
                let ek = chunk_key_i32(ep[0], ep[1], ep[2], ep[3]);
                let eb = Aabb4i::chunk_world_bounds(ek, 0);
                if !platform.contains_bounds(&eb) || ek == chunk_a_key || ek == chunk_b_key {
                    continue;
                }
                let ep_palette = vec![BlockData::AIR, placed.clone()];
                let ep_ca = ChunkArrayData::from_dense_indices_with_block_palette(
                    eb,
                    vec![ChunkPayload::Uniform(1)],
                    vec![0],
                    None,
                    ep_palette,
                    0,
                )
                .expect("extra placement");
                extra_cores.push((
                    eb,
                    RegionTreeCore {
                        bounds: eb,
                        kind: RegionNodeKind::ChunkArray(ep_ca),
                        generator_version_hash: 0,
                    },
                ));
            }

            // B override: ground with one hole.
            let b_block_palette = vec![BlockData::AIR, ground.clone()];
            let mut b_dense = vec![1u16; CHUNK_VOLUME];
            b_dense[0] = 0; // air at cell 0
            let b_ca = ChunkArrayData::from_dense_indices_with_block_palette(
                chunk_b_bounds,
                vec![ChunkPayload::Dense16 { materials: b_dense }],
                vec![0],
                None,
                b_block_palette,
                0,
            )
            .expect("chunk array B");
            let b_core = RegionTreeCore {
                bounds: chunk_b_bounds,
                kind: RegionNodeKind::ChunkArray(b_ca),
                generator_version_hash: 0,
            };

            // Helper: add all persistent overrides (A + extras) to an override tree.
            let add_persistent_overrides = |tree: &mut RegionChunkTree| {
                tree.splice_non_empty_core_in_bounds(chunk_a_bounds, &a_core);
                for (eb, ec) in &extra_cores {
                    tree.splice_non_empty_core_in_bounds(*eb, ec);
                }
            };

            // --- Step 0: apply extra placements to client scene ---
            for (eb, ec) in &extra_cores {
                let mut ov = RegionChunkTree::new();
                add_persistent_overrides(&mut ov);
                let st = compose_server_subtree(platform, &platform_core, &ov);
                scene.apply_region_patch_fast(*eb, &st);
            }

            // --- Step 1: place block at A ---
            let mut overrides1 = RegionChunkTree::new();
            add_persistent_overrides(&mut overrides1);
            let subtree1 = compose_server_subtree(platform, &platform_core, &overrides1);
            let stats1 = scene.apply_region_patch_fast(chunk_a_bounds, &subtree1);
            assert!(
                stats1.changed_chunks > 0,
                "step 1 no change for offset {:?}",
                offset
            );

            // --- Step 2: break ground at B ---
            let mut overrides2 = RegionChunkTree::new();
            add_persistent_overrides(&mut overrides2);
            overrides2.splice_non_empty_core_in_bounds(chunk_b_bounds, &b_core);
            let subtree2 = compose_server_subtree(platform, &platform_core, &overrides2);
            let stats2 = scene.apply_region_patch_fast(chunk_b_bounds, &subtree2);
            assert!(
                stats2.changed_chunks > 0,
                "step 2 no change for offset {:?}",
                offset
            );

            // --- Step 3: replace ground at B (restore virgin) ---
            let mut overrides3 = RegionChunkTree::new();
            add_persistent_overrides(&mut overrides3);
            let subtree3 = compose_server_subtree(platform, &platform_core, &overrides3);
            let stats3 = scene.apply_region_patch_fast(chunk_b_bounds, &subtree3);
            assert!(
            stats3.changed_chunks > 0,
            "step 3 no change for pmin={:?} pmax={:?} akey={:?} offset={:?} — splice missed the data change",
            pmin, pmax, akey, offset
        );

            // Verify chunk B has ground data (not air/empty).
            let b_block = scene.get_block_data(bx * 8, by * 8, bz * 8, bw * 8);
            assert_eq!(
                b_block, ground,
                "chunk B data wrong after replace! offset={:?} got={:?}",
                offset, b_block,
            );

            // Verify chunk A still has placed data.
            let a_block = scene.get_block_data(akey[0] * 8, akey[1] * 8, akey[2] * 8, akey[3] * 8);
            assert!(
                !a_block.is_air(),
                "chunk A data lost after B-replace! pmin={:?} pmax={:?} akey={:?} offset={:?}",
                pmin,
                pmax,
                akey,
                offset,
            );

            // Verify other platform chunks not involved are still ground.
            let extra_keys: Vec<_> = extra_cores
                .iter()
                .map(|(eb, _)| eb.chunk_key_from_world_bounds())
                .collect();
            // Use chunk A position as a reference for nearby check positions.
            let check_positions = [
                [pmin[0], pmin[1], pmin[2], pmin[3]],
                [pmax[0], pmax[1], pmax[2], pmax[3]],
                [
                    (pmin[0] + pmax[0]) / 2,
                    pmin[1],
                    (pmin[2] + pmax[2]) / 2,
                    (pmin[3] + pmax[3]) / 2,
                ],
            ];
            for check in check_positions {
                let ck = chunk_key_i32(check[0], check[1], check[2], check[3]);
                if ck == chunk_a_key || ck == chunk_b_key || extra_keys.contains(&ck) {
                    continue;
                }
                let cb = Aabb4i::chunk_world_bounds(ck, 0);
                if !platform.contains_bounds(&cb) {
                    continue;
                }
                let b =
                    scene.get_block_data(check[0] * 8, check[1] * 8, check[2] * 8, check[3] * 8);
                assert_eq!(
                    b, ground,
                    "platform chunk {:?} data wrong! pmin={:?} pmax={:?} offset={:?} got={:?}",
                    check, pmin, pmax, offset, b
                );
            }
        } // b_offsets
    } // platform_configs
}

/// Same as `place_break_replace_via_fast_patch_preserves_chunk` but loads
/// the initial world via multiple streaming patches (simulating the server's
/// `split_bounds_for_streaming` flow). The tree after initial loading may
/// have a different structure (nested Branches from root expansion) which
/// could interact differently with subsequent edit patches.
#[test]
fn place_break_replace_with_streamed_initial_world() {
    use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
    use polychora::shared::region_tree::{RegionNodeKind, RegionTreeCore};
    use polychora::shared::voxel::CHUNK_VOLUME;

    let ground = BlockData::simple(0, 1);
    let placed = BlockData::simple(0, 99);

    fn compose(
        platform: Aabb4i,
        base: &RegionTreeCore,
        overrides: &RegionChunkTree,
    ) -> RegionTreeCore {
        let compose_bounds = platform;
        let mut composed = RegionChunkTree::new();
        let _ = composed.splice_non_empty_core_in_bounds(platform, base);
        let ov = overrides.slice_core_in_bounds(compose_bounds);
        let _ = composed.overlay_core_in_bounds(compose_bounds, &ov);
        composed.root().cloned().unwrap_or(RegionTreeCore {
            bounds: compose_bounds,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        })
    }

    /// Split bounds into streaming slices (mirrors server's split_bounds_for_streaming).
    fn split_for_streaming(bounds: Aabb4i, max_cells: usize) -> Vec<Aabb4i> {
        let mut out = Vec::new();
        let mut stack = vec![bounds];
        while let Some(current) = stack.pop() {
            let cell_count = current.chunk_cell_count_at_scale(0).unwrap_or(0);
            if cell_count <= max_cells || out.len() + stack.len() + 1 >= 256 {
                out.push(current);
                continue;
            }
            let extents: Vec<_> = (0..4).map(|a| current.max[a] - current.min[a]).collect();
            let split_axis = (0..4).max_by_key(|&a| extents[a]).unwrap_or(0);
            if extents[split_axis] <= polychora::shared::spatial::ChunkCoord::ZERO {
                out.push(current);
                continue;
            }
            let two = polychora::shared::spatial::ChunkCoord::from_num(2);
            let half = (extents[split_axis] / two).floor();
            let mid = current.min[split_axis] + half;
            let mut left = current;
            let mut right = current;
            left.max[split_axis] = mid;
            right.min[split_axis] = mid;
            if left.is_valid() {
                stack.push(left);
            }
            if right.is_valid() {
                stack.push(right);
            }
        }
        out
    }

    // Test configs: (platform_min, platform_max, interest_bounds, chunk_a, chunk_b)
    let configs: &[([i32; 4], [i32; 4], [i32; 4], [i32; 4], [i32; 4], [i32; 4])] = &[
        // Small platform, interest larger than platform
        (
            [0, 0, 0, 0],
            [3, 0, 3, 3],
            [-2, -2, -2, -2],
            [5, 2, 5, 5],
            [1, 0, 1, 1],
            [2, 0, 1, 1],
        ),
        // Large platform that requires splitting
        (
            [0, 0, 0, 0],
            [15, 0, 15, 15],
            [-4, -4, -4, -4],
            [19, 4, 19, 19],
            [4, 0, 6, 7],
            [5, 0, 6, 7],
        ),
        // Platform offset from origin
        (
            [5, 0, 5, 5],
            [14, 0, 14, 14],
            [0, -4, 0, 0],
            [19, 4, 19, 19],
            [8, 0, 8, 8],
            [9, 0, 8, 8],
        ),
        // Tiny platform, big interest
        (
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [-5, -5, -5, -5],
            [6, 5, 6, 6],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ),
        // A and B far apart
        (
            [0, 0, 0, 0],
            [15, 0, 15, 15],
            [-2, -2, -2, -2],
            [17, 2, 17, 17],
            [1, 0, 1, 1],
            [14, 0, 14, 14],
        ),
    ];

    for &(pmin, pmax, imin, imax, akey, bkey) in configs {
        let platform = Aabb4i::from_lattice_bounds(pmin, pmax, 0);
        let interest = Aabb4i::from_lattice_bounds(imin, imax, 0);
        let platform_core = RegionTreeCore {
            bounds: platform,
            kind: RegionNodeKind::Uniform(ground.clone()),
            generator_version_hash: 0,
        };

        let chunk_a_key = chunk_key_i32(akey[0], akey[1], akey[2], akey[3]);
        let chunk_a_bounds = Aabb4i::chunk_world_bounds(chunk_a_key, 0);
        let chunk_b_key = chunk_key_i32(bkey[0], bkey[1], bkey[2], bkey[3]);
        let chunk_b_bounds = Aabb4i::chunk_world_bounds(chunk_b_key, 0);

        assert!(
            platform.contains_bounds(&chunk_a_bounds),
            "A outside platform"
        );
        assert!(
            platform.contains_bounds(&chunk_b_bounds),
            "B outside platform"
        );
        assert_ne!(chunk_a_key, chunk_b_key, "A == B");

        // --- Initial world loading via streaming patches ---
        let mut scene = Scene::new(ScenePreset::Empty);
        let slices = split_for_streaming(interest, 500); // Force many splits
        let no_overrides = RegionChunkTree::new();
        for slice_bounds in &slices {
            let subtree = compose(platform, &platform_core, &no_overrides);
            scene.apply_region_patch_fast(*slice_bounds, &subtree);
        }

        // Verify initial state: A and B should be ground
        let a0 = scene.get_block_data(akey[0] * 8, akey[1] * 8, akey[2] * 8, akey[3] * 8);
        assert_eq!(
            a0, ground,
            "A should be ground after streaming load, config pmin={pmin:?}"
        );
        let b0 = scene.get_block_data(bkey[0] * 8, bkey[1] * 8, bkey[2] * 8, bkey[3] * 8);
        assert_eq!(
            b0, ground,
            "B should be ground after streaming load, config pmin={pmin:?}"
        );

        // --- A override: ground with one placed block ---
        let a_bp = vec![BlockData::AIR, ground.clone(), placed.clone()];
        let mut a_dense = vec![1u16; CHUNK_VOLUME];
        a_dense[0] = 2;
        let a_core = RegionTreeCore {
            bounds: chunk_a_bounds,
            kind: RegionNodeKind::ChunkArray(
                ChunkArrayData::from_dense_indices_with_block_palette(
                    chunk_a_bounds,
                    vec![ChunkPayload::Dense16 { materials: a_dense }],
                    vec![0],
                    None,
                    a_bp,
                    0,
                )
                .unwrap(),
            ),
            generator_version_hash: 0,
        };

        // --- B override: ground with one hole ---
        let b_bp = vec![BlockData::AIR, ground.clone()];
        let mut b_dense = vec![1u16; CHUNK_VOLUME];
        b_dense[0] = 0;
        let b_core = RegionTreeCore {
            bounds: chunk_b_bounds,
            kind: RegionNodeKind::ChunkArray(
                ChunkArrayData::from_dense_indices_with_block_palette(
                    chunk_b_bounds,
                    vec![ChunkPayload::Dense16 { materials: b_dense }],
                    vec![0],
                    None,
                    b_bp,
                    0,
                )
                .unwrap(),
            ),
            generator_version_hash: 0,
        };

        // Step 1: place at A
        let mut ov1 = RegionChunkTree::new();
        ov1.splice_non_empty_core_in_bounds(chunk_a_bounds, &a_core);
        let st1 = compose(platform, &platform_core, &ov1);
        let stats1 = scene.apply_region_patch_fast(chunk_a_bounds, &st1);
        assert!(
            stats1.changed_chunks > 0,
            "step 1 no change, config pmin={pmin:?}"
        );
        let a1 = scene.get_block_data(akey[0] * 8, akey[1] * 8, akey[2] * 8, akey[3] * 8);
        assert_eq!(
            a1, placed,
            "A should be placed after step 1, config pmin={pmin:?}"
        );

        // Step 2: break at B
        let mut ov2 = RegionChunkTree::new();
        ov2.splice_non_empty_core_in_bounds(chunk_a_bounds, &a_core);
        ov2.splice_non_empty_core_in_bounds(chunk_b_bounds, &b_core);
        let st2 = compose(platform, &platform_core, &ov2);
        let stats2 = scene.apply_region_patch_fast(chunk_b_bounds, &st2);
        assert!(
            stats2.changed_chunks > 0,
            "step 2 no change, config pmin={pmin:?}"
        );

        // Step 3: replace at B (restore virgin)
        let mut ov3 = RegionChunkTree::new();
        ov3.splice_non_empty_core_in_bounds(chunk_a_bounds, &a_core);
        let st3 = compose(platform, &platform_core, &ov3);
        let stats3 = scene.apply_region_patch_fast(chunk_b_bounds, &st3);
        assert!(
            stats3.changed_chunks > 0,
            "step 3 no change — splice missed data change, config pmin={pmin:?}"
        );

        // Verify: B should be ground, A should still be placed
        let b3 = scene.get_block_data(bkey[0] * 8, bkey[1] * 8, bkey[2] * 8, bkey[3] * 8);
        assert_eq!(
            b3, ground,
            "B should be ground after restore, config pmin={pmin:?} got={b3:?}"
        );

        let a3 = scene.get_block_data(akey[0] * 8, akey[1] * 8, akey[2] * 8, akey[3] * 8);
        assert_eq!(
            a3, placed,
            "A data lost after B restore! config pmin={pmin:?} got={a3:?}"
        );

        // Spot-check other platform chunks
        for check in [
            pmin,
            pmax,
            [
                (pmin[0] + pmax[0]) / 2,
                pmin[1],
                (pmin[2] + pmax[2]) / 2,
                (pmin[3] + pmax[3]) / 2,
            ],
        ] {
            let ck = chunk_key_i32(check[0], check[1], check[2], check[3]);
            let cb = Aabb4i::chunk_world_bounds(ck, 0);
            if ck == chunk_a_key || ck == chunk_b_key || !platform.contains_bounds(&cb) {
                continue;
            }
            let b = scene.get_block_data(check[0] * 8, check[1] * 8, check[2] * 8, check[3] * 8);
            assert_eq!(
                b, ground,
                "platform chunk {check:?} wrong after restore, config pmin={pmin:?} got={b:?}"
            );
        }
    }
}

/// Minimal reproduction of the chunk A data loss bug.
/// Platform [0..3]×[0]×[0..3]×[0..3], A=[1,0,1,1], B=[2,0,1,1].
#[test]
fn place_break_replace_minimal_repro() {
    use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
    use polychora::shared::region_tree::{RegionNodeKind, RegionTreeCore};
    use polychora::shared::voxel::CHUNK_VOLUME;

    let ground = BlockData::simple(0, 1);
    let placed = BlockData::simple(0, 99);

    let platform = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [3, 0, 3, 3], 0);
    let platform_core = RegionTreeCore {
        bounds: platform,
        kind: RegionNodeKind::Uniform(ground.clone()),
        generator_version_hash: 0,
    };

    fn compose(
        platform: Aabb4i,
        base: &RegionTreeCore,
        overrides: &RegionChunkTree,
    ) -> RegionTreeCore {
        let mut composed = RegionChunkTree::new();
        let _ = composed.splice_non_empty_core_in_bounds(platform, base);
        let ov = overrides.slice_core_in_bounds(platform);
        let _ = composed.overlay_core_in_bounds(platform, &ov);
        composed.root().cloned().unwrap_or(RegionTreeCore {
            bounds: platform,
            kind: RegionNodeKind::Empty,
            generator_version_hash: 0,
        })
    }

    let chunk_a = chunk_key_i32(1, 0, 1, 1);
    let chunk_a_b = Aabb4i::chunk_world_bounds(chunk_a, 0);
    let chunk_b = chunk_key_i32(2, 0, 1, 1);
    let chunk_b_b = Aabb4i::chunk_world_bounds(chunk_b, 0);

    // A override: ground with one placed block (Dense16).
    let mut a_dense = vec![1u16; CHUNK_VOLUME];
    a_dense[0] = 2; // placed at cell 0
    let a_core = RegionTreeCore {
        bounds: chunk_a_b,
        kind: RegionNodeKind::ChunkArray(
            ChunkArrayData::from_dense_indices_with_block_palette(
                chunk_a_b,
                vec![ChunkPayload::Dense16 { materials: a_dense }],
                vec![0],
                None,
                vec![BlockData::AIR, ground.clone(), placed.clone()],
                0,
            )
            .unwrap(),
        ),
        generator_version_hash: 0,
    };

    // B override: ground with one hole.
    let mut b_dense = vec![1u16; CHUNK_VOLUME];
    b_dense[0] = 0;
    let b_core = RegionTreeCore {
        bounds: chunk_b_b,
        kind: RegionNodeKind::ChunkArray(
            ChunkArrayData::from_dense_indices_with_block_palette(
                chunk_b_b,
                vec![ChunkPayload::Dense16 { materials: b_dense }],
                vec![0],
                None,
                vec![BlockData::AIR, ground.clone()],
                0,
            )
            .unwrap(),
        ),
        generator_version_hash: 0,
    };

    let mut scene = Scene::new(ScenePreset::Empty);
    scene.apply_region_patch_fast(platform, &platform_core);

    // Check A is ground initially.
    let a0 = scene.get_block_data(1 * 8, 0, 1 * 8, 1 * 8);
    assert_eq!(a0, ground, "A should be ground initially");

    // Step 1: place at A.
    let mut ov1 = RegionChunkTree::new();
    ov1.splice_non_empty_core_in_bounds(chunk_a_b, &a_core);
    let st1 = compose(platform, &platform_core, &ov1);
    scene.apply_region_patch_fast(chunk_a_b, &st1);
    let a1 = scene.get_block_data(1 * 8, 0, 1 * 8, 1 * 8);
    assert_eq!(a1, placed, "A should be placed after step 1");

    // Step 2: break at B.
    let mut ov2 = RegionChunkTree::new();
    ov2.splice_non_empty_core_in_bounds(chunk_a_b, &a_core);
    ov2.splice_non_empty_core_in_bounds(chunk_b_b, &b_core);
    let st2 = compose(platform, &platform_core, &ov2);
    scene.apply_region_patch_fast(chunk_b_b, &st2);
    let a2 = scene.get_block_data(1 * 8, 0, 1 * 8, 1 * 8);
    assert_eq!(a2, placed, "A should still be placed after step 2");

    // Step 3: replace at B (restore virgin).
    let mut ov3 = RegionChunkTree::new();
    ov3.splice_non_empty_core_in_bounds(chunk_a_b, &a_core);
    let st3 = compose(platform, &platform_core, &ov3);

    // Dump tree state before step 3.
    eprintln!("=== Before step 3 ===");
    eprintln!(
        "  subtree3 root: kind={:?} bounds={:?}->{:?}",
        std::mem::discriminant(&st3.kind),
        st3.bounds.min,
        st3.bounds.max
    );
    let sliced = slice_non_empty_region_core_in_bounds(&st3, chunk_b_b);
    eprintln!(
        "  subtree3 sliced to B: kind={:?} bounds={:?}->{:?}",
        std::mem::discriminant(&sliced.kind),
        sliced.bounds.min,
        sliced.bounds.max
    );
    let prev = scene.world_tree.slice_non_empty_core_in_bounds(chunk_b_b);
    eprintln!(
        "  client prev at B: kind={:?} bounds={:?}->{:?}",
        std::mem::discriminant(&prev.kind),
        prev.bounds.min,
        prev.bounds.max
    );

    let stats3 = scene.apply_region_patch_fast(chunk_b_b, &st3);
    eprintln!("  step3 stats: changed_chunks={}", stats3.changed_chunks);

    // After step 3: B should be ground, A should still be placed.
    let b3 = scene.get_block_data(2 * 8, 0, 1 * 8, 1 * 8);
    assert_eq!(b3, ground, "B should be ground after step 3, got {:?}", b3);

    let a3 = scene.get_block_data(1 * 8, 0, 1 * 8, 1 * 8);
    assert_eq!(
        a3, placed,
        "A should still be placed after step 3, got {:?}",
        a3
    );
}

/// Regression test matching the EXACT runtime flow from the bug log:
/// - Welcome patch with subtree bounds larger than authoritative (platform + procgen)
/// - Green block placed above platform
/// - Break ground chunk
/// - Replace ground chunk
/// After replace, the ChunkArray at the broken chunk must be replaced with Uniform(ground).
#[test]
fn place_break_replace_runtime_exact_flow() {
    use polychora::shared::chunk_payload::{ChunkArrayData, ChunkPayload};
    use polychora::shared::region_tree::{RegionNodeKind, RegionTreeCore};
    use polychora::shared::voxel::CHUNK_VOLUME;

    let ground = BlockData::simple(1886350457, 3294548464);
    let green = BlockData::simple(0, 2968422830);

    // --- Server-side world definitions ---
    // Platform Uniform: [-144,-8,-144,-144]->[144,8,144,144]
    let platform = Aabb4i::new(
        [-144, -8, -144, -144].map(|v| ChunkCoord::from_num(v)),
        [144, 8, 144, 144].map(|v| ChunkCoord::from_num(v)),
    );
    let platform_core = RegionTreeCore {
        bounds: platform,
        kind: RegionNodeKind::Uniform(ground.clone()),
        generator_version_hash: 0,
    };

    // Procgen structures above the platform (simplified as Uniform blocks at those bounds)
    // The exact content doesn't matter — we need non-empty children above y=8 to create
    // the Branch structure matching the runtime tree.
    let procgen_block = BlockData::simple(0, 42);
    let procgen_bounds: &[([i32; 4], [i32; 4])] = &[
        ([-16, 8, -80, 64], [8, 16, -56, 88]),
        ([-16, 8, -8, 40], [-8, 24, 0, 56]),
        ([0, 8, -8, -8], [8, 16, 0, 0]),
        ([40, 8, -32, 72], [56, 16, -16, 88]),
    ];

    // Build the welcome subtree: Branch(platform + procgen)
    let mut welcome_children = vec![platform_core.clone()];
    for &(min, max) in procgen_bounds {
        let bounds = Aabb4i::new(
            min.map(|v| ChunkCoord::from_num(v)),
            max.map(|v| ChunkCoord::from_num(v)),
        );
        welcome_children.push(RegionTreeCore {
            bounds,
            kind: RegionNodeKind::Uniform(procgen_block.clone()),
            generator_version_hash: 0,
        });
    }
    let welcome_subtree = RegionTreeCore {
        bounds: Aabb4i::new(
            [-144, -8, -144, -144].map(|v| ChunkCoord::from_num(v)),
            [144, 24, 144, 144].map(|v| ChunkCoord::from_num(v)),
        ),
        kind: RegionNodeKind::Branch(welcome_children),
        generator_version_hash: 0,
    };

    // --- Client-side: apply patches in the exact order from the log ---
    let mut scene = Scene::new(ScenePreset::Empty);

    // Step 1: Welcome patch
    // authoritative = [-112,-112,-120,-120]->[112,112,104,104]
    // subtree = welcome_subtree (bounds [-144,-8,-144,-144]->[144,24,144,144])
    let welcome_auth = Aabb4i::new(
        [-112, -112, -120, -120].map(|v| ChunkCoord::from_num(v)),
        [112, 112, 104, 104].map(|v| ChunkCoord::from_num(v)),
    );
    let stats = scene.apply_region_patch_fast(welcome_auth, &welcome_subtree);
    eprintln!("Welcome: changed_chunks={}", stats.changed_chunks);

    // Verify ground is present at the break location
    let ground_check = scene.get_block_data(6, 7, -12, -4);
    assert_eq!(
        ground_check, ground,
        "ground should be present after welcome"
    );

    // Step 2: Place green block at (7,8,-8,-4)
    // Server sends: authoritative=[0,8,-8,-8]->[8,16,0,0] subtree=[0,8,-8,-8]->[8,16,0,0]
    let green_chunk_bounds = Aabb4i::new(
        [0, 8, -8, -8].map(|v| ChunkCoord::from_num(v)),
        [8, 16, 0, 0].map(|v| ChunkCoord::from_num(v)),
    );
    // Create a ChunkArray with the green block placed at (7,8,-8,-4)
    let mut green_dense = vec![0u16; CHUNK_VOLUME]; // mostly air (above platform)
                                                    // Voxel (7,8,-8,-4) in chunk [0,8,-8,-8] → local coords (7,0,0,4)
    let green_local_idx = 7 + 0 * 8 + 0 * 64 + 4 * 512;
    green_dense[green_local_idx] = 1;
    let green_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        green_chunk_bounds,
        vec![ChunkPayload::Dense16 {
            materials: green_dense,
        }],
        vec![0],
        None,
        vec![BlockData::AIR, green.clone()],
        0,
    )
    .unwrap();
    let green_subtree = RegionTreeCore {
        bounds: green_chunk_bounds,
        kind: RegionNodeKind::ChunkArray(green_ca),
        generator_version_hash: 0,
    };
    scene.apply_region_patch_fast(green_chunk_bounds, &green_subtree);

    // Step 3: Break ground at (6,7,-12,-4)
    // Server sends: authoritative=[0,0,-16,-8]->[8,8,-8,0] subtree=[-144,-8,-144,-144]->[144,8,144,144]
    // The subtree is the full platform WITH a break override (one chunk has a hole).
    let break_chunk_bounds = Aabb4i::new(
        [0, 0, -16, -8].map(|v| ChunkCoord::from_num(v)),
        [8, 8, -8, 0].map(|v| ChunkCoord::from_num(v)),
    );

    // Build break subtree: platform with one modified chunk (hole at (6,7,-12,-4))
    // Compose server-side: base Uniform + override with break
    let mut break_override_tree = RegionChunkTree::new();
    let mut break_dense = vec![1u16; CHUNK_VOLUME]; // all ground
                                                    // Voxel (6,7,-12,-4) in chunk [0,0,-16,-8] → local coords (6,7,4,4)
    let break_local_idx = 6 + 7 * 8 + 4 * 64 + 4 * 512;
    break_dense[break_local_idx] = 0; // air (broken)
    let break_ca = ChunkArrayData::from_dense_indices_with_block_palette(
        break_chunk_bounds,
        vec![ChunkPayload::Dense16 {
            materials: break_dense,
        }],
        vec![0],
        None,
        vec![BlockData::AIR, ground.clone()],
        0,
    )
    .unwrap();
    let break_chunk_core = RegionTreeCore {
        bounds: break_chunk_bounds,
        kind: RegionNodeKind::ChunkArray(break_ca),
        generator_version_hash: 0,
    };
    break_override_tree.splice_non_empty_core_in_bounds(break_chunk_bounds, &break_chunk_core);

    // Compose: platform + break override
    let mut break_composed = RegionChunkTree::new();
    let _ = break_composed.splice_non_empty_core_in_bounds(platform, &platform_core);
    let break_ov = break_override_tree.slice_core_in_bounds(platform);
    let _ = break_composed.overlay_core_in_bounds(platform, &break_ov);
    let break_subtree = break_composed.root().cloned().unwrap();

    eprintln!("\n=== Break subtree ===");
    eprintln!("{}", dump_tree_structure(&break_subtree, 0));

    let stats = scene.apply_region_patch_fast(break_chunk_bounds, &break_subtree);
    eprintln!("Break: changed_chunks={}", stats.changed_chunks);

    // Verify break worked
    let break_check = scene.get_block_data(6, 7, -12, -4);
    assert!(
        break_check.is_air(),
        "broken block should be air, got {:?}",
        break_check
    );

    // Dump tree after break
    if let Some(root) = scene.world_tree.root() {
        eprintln!("\n=== Client tree after break ===");
        eprintln!("{}", dump_tree_structure(root, 0));
    }

    // Step 4: Replace ground at (6,7,-12,-4)
    // Server sends: authoritative=[0,0,-16,-8]->[8,8,-8,0] subtree=[-144,-8,-144,-144]->[144,8,144,144]
    // The subtree is the full platform with NO override (block matches virgin world).
    let replace_subtree = platform_core.clone(); // Pure platform Uniform

    eprintln!("\n=== Replace subtree ===");
    eprintln!("{}", dump_tree_structure(&replace_subtree, 0));

    let stats = scene.apply_region_patch_fast(break_chunk_bounds, &replace_subtree);
    eprintln!("Replace: changed_chunks={}", stats.changed_chunks);

    // Dump tree after replace
    if let Some(root) = scene.world_tree.root() {
        eprintln!("\n=== Client tree after replace ===");
        eprintln!("{}", dump_tree_structure(root, 0));
    }

    // Verify replace worked: the broken chunk should be ground again
    let replace_check = scene.get_block_data(6, 7, -12, -4);
    assert_eq!(
        replace_check, ground,
        "replaced block should be ground, got {:?}",
        replace_check
    );

    // Verify the tree at break_chunk_bounds is Uniform, not ChunkArray
    let post_slice = scene
        .world_tree
        .slice_non_empty_core_in_bounds(break_chunk_bounds);
    assert!(
        matches!(post_slice.kind, RegionNodeKind::Uniform(_)),
        "chunk at break location should be Uniform after replace, got {:?}",
        std::mem::discriminant(&post_slice.kind)
    );
}

/// Full server-client integration test using the actual MassivePlatformsWorldGenerator
/// and PassthroughWorldOverlay. Reproduces the exact flow from the bug log.
#[test]
fn place_break_replace_with_real_generator() {
    use polychora::server::world_field::{
        QueryDetail, QueryVolume, ServerWorldOverlay, WorldField,
    };
    use polychora::shared::region_tree::RegionNodeKind;
    use std::collections::HashSet;

    let ground = BlockData::simple(
        polychora_plugin_api::content_ids::CONTENT_NS,
        polychora_plugin_api::content_ids::BLOCK_GRID_FLOOR,
    );
    let base_kind = polychora::shared::voxel::BaseWorldKind::MassivePlatforms {
        material: ground.clone(),
    };

    let mut overlay = ServerWorldOverlay::from_chunk_payloads(
        base_kind.clone(),
        Vec::<(
            [i32; 4],
            polychora::shared::chunk_payload::ResolvedChunkPayload,
        )>::new(),
        1337,
        true,
        HashSet::new(),
    );

    let mut scene = Scene::new(ScenePreset::Empty);

    // Step 1: Welcome patch
    let welcome_auth = Aabb4i::new(
        [-112, -112, -120, -120].map(|v| ChunkCoord::from_num(v)),
        [112, 112, 104, 104].map(|v| ChunkCoord::from_num(v)),
    );
    let welcome_subtree = overlay.query_region_core(
        QueryVolume {
            bounds: welcome_auth,
        },
        QueryDetail::Exact,
    );
    eprintln!(
        "Welcome subtree: bounds={:?}->{:?} kind={}",
        welcome_subtree.bounds.min,
        welcome_subtree.bounds.max,
        match &welcome_subtree.kind {
            RegionNodeKind::Empty => "Empty",
            RegionNodeKind::Uniform(_) => "Uniform",
            RegionNodeKind::ProceduralRef(_) => "ProceduralRef",
            RegionNodeKind::ChunkArray(_) => "ChunkArray",
            RegionNodeKind::Branch(c) => {
                eprintln!("  Branch({} children)", c.len());
                "Branch"
            }
        }
    );
    let stats = scene.apply_region_patch_fast(welcome_auth, &welcome_subtree);
    eprintln!("Welcome: changed_chunks={}", stats.changed_chunks);

    // Step 2: Place green block above platform at (7, 8, -8, -4)
    let green_block = BlockData::simple(0, 2968422830);
    let green_pos = [7i32, 8, -8, -4];
    // First break (to make air for placement collision check), then place
    overlay.apply_voxel_edit(cc4(green_pos), BlockData::AIR);
    overlay.apply_voxel_edit(cc4(green_pos), green_block.clone());
    let green_dirty = overlay.take_dirty_bounds();
    for db in &green_dirty {
        let green_sub = overlay.query_region_core(QueryVolume { bounds: *db }, QueryDetail::Exact);
        scene.apply_region_patch_fast(*db, &green_sub);
    }

    // Step 3: Break ground at (6, 7, -12, -4)
    let break_pos = [6i32, 7, -12, -4];
    overlay.apply_voxel_edit(cc4(break_pos), BlockData::AIR);
    let break_dirty = overlay.take_dirty_bounds();
    eprintln!("Break dirty bounds: {} entries", break_dirty.len());
    for db in &break_dirty {
        eprintln!("  dirty: {:?}->{:?}", db.min, db.max);
        let clip = db.intersection(&welcome_auth).unwrap_or(*db);
        let break_sub = overlay.query_region_core(QueryVolume { bounds: clip }, QueryDetail::Exact);
        eprintln!(
            "  subtree: bounds={:?}->{:?}",
            break_sub.bounds.min, break_sub.bounds.max
        );
        let stats = scene.apply_region_patch_fast(clip, &break_sub);
        eprintln!("  Break: changed_chunks={}", stats.changed_chunks);
    }

    // Verify break worked
    let break_check = scene.get_block_data(6, 7, -12, -4);
    assert!(
        break_check.is_air(),
        "broken block should be air, got {:?}",
        break_check
    );

    // Dump tree after break
    if let Some(root) = scene.world_tree.root() {
        eprintln!("\n=== Client tree after break ===");
        eprintln!("{}", dump_tree_structure(root, 0));
    }

    // Step 4: Replace ground at (6, 7, -12, -4)
    overlay.apply_voxel_edit(cc4(break_pos), ground.clone());

    // Debug: query the server for the FULL platform bounds and the chunk bounds
    let full_platform_query = overlay.query_region_core(
        QueryVolume {
            bounds: Aabb4i::new(
                [0, 0, -16, -8].map(|v| ChunkCoord::from_num(v)),
                [8, 8, -8, 0].map(|v| ChunkCoord::from_num(v)),
            ),
        },
        QueryDetail::Exact,
    );
    eprintln!(
        "\n=== Server query at break chunk after replace edit ===\n{}",
        dump_tree_structure(&full_platform_query, 0)
    );
    eprintln!(
        "Override non-empty chunks: {}",
        overlay.non_empty_chunk_count()
    );
    if let Some(ov_root) = overlay.debug_override_root() {
        eprintln!("Override tree:\n{}", dump_tree_structure(ov_root, 0));
    }

    let replace_dirty = overlay.take_dirty_bounds();
    eprintln!("Replace dirty bounds: {} entries", replace_dirty.len());
    for db in &replace_dirty {
        eprintln!("  dirty: {:?}->{:?}", db.min, db.max);
        let clip = db.intersection(&welcome_auth).unwrap_or(*db);
        let replace_sub =
            overlay.query_region_core(QueryVolume { bounds: clip }, QueryDetail::Exact);
        eprintln!(
            "  subtree: bounds={:?}->{:?} kind={}",
            replace_sub.bounds.min,
            replace_sub.bounds.max,
            match &replace_sub.kind {
                RegionNodeKind::Empty => "Empty".to_string(),
                RegionNodeKind::Uniform(b) =>
                    format!("Uniform(ns={},bt={})", b.namespace, b.block_type),
                RegionNodeKind::ProceduralRef(_) => "ProceduralRef".to_string(),
                RegionNodeKind::ChunkArray(_) => "ChunkArray".to_string(),
                RegionNodeKind::Branch(c) => format!("Branch({})", c.len()),
            }
        );
        let stats = scene.apply_region_patch_fast(clip, &replace_sub);
        eprintln!("  Replace: changed_chunks={}", stats.changed_chunks);
    }

    // Dump tree after replace
    if let Some(root) = scene.world_tree.root() {
        eprintln!("\n=== Client tree after replace ===");
        eprintln!("{}", dump_tree_structure(root, 0));
    }

    // Verify replace worked
    let replace_check = scene.get_block_data(6, 7, -12, -4);
    assert_eq!(
        replace_check, ground,
        "replaced block should be ground, got {:?}",
        replace_check
    );

    // Verify the tree at break chunk is Uniform
    let break_chunk_bounds = Aabb4i::new(
        [0, 0, -16, -8].map(|v| ChunkCoord::from_num(v)),
        [8, 8, -8, 0].map(|v| ChunkCoord::from_num(v)),
    );
    let post_slice = scene
        .world_tree
        .slice_non_empty_core_in_bounds(break_chunk_bounds);
    assert!(
        matches!(post_slice.kind, RegionNodeKind::Uniform(_)),
        "chunk at break location should be Uniform after replace, got {:?}",
        std::mem::discriminant(&post_slice.kind)
    );
}
