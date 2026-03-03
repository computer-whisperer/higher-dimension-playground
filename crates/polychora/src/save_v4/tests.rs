use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Helper to construct an IndexNode from world-space i32 bounds (half-open) for tests.
fn test_index_node(node_id: u32, min: [i32; 4], max: [i32; 4], kind: IndexNodeKind) -> IndexNode {
    IndexNode {
        node_id,
        bounds_min_fixed: min.map(|v| ChunkCoord::from_num(v).to_bits()),
        bounds_max_fixed: max.map(|v| ChunkCoord::from_num(v).to_bits()),
        scale_exp: 0,
        kind,
    }
}

static TEST_UNIQUIFIER: AtomicU64 = AtomicU64::new(0);

fn test_root(name: &str) -> PathBuf {
    let serial = TEST_UNIQUIFIER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "polychora-save-v4-{name}-{}-{}",
        std::process::id(),
        serial
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create test save root");
    path
}

fn collect_world_chunk_array_refs(index: &IndexPayload) -> Vec<BlobRef> {
    let node_by_id = build_node_lookup(index);
    let mut refs = Vec::new();
    collect_leaf_chunk_array_refs_recursive(index.root_node_id, &node_by_id, &mut refs)
        .expect("collect world refs");
    refs.sort_unstable_by_key(blob_ref_identity);
    refs
}

fn total_data_bytes(root: &Path) -> u64 {
    let data_root = root.join(DATA_DIR);
    let Ok(entries) = std::fs::read_dir(&data_root) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .filter(|meta| meta.is_file())
        .map(|meta| meta.len())
        .sum()
}

fn test_entity(entity_id: u64, position: [f32; 4], data: Vec<u8>) -> PersistedEntityRecord {
    use crate::shared::entity_types::ENTITY_TEST_ROTOR;
    PersistedEntityRecord {
        entity_id,
        entity: Entity {
            namespace: ENTITY_TEST_ROTOR.0,
            entity_type: ENTITY_TEST_ROTOR.1,
            pose: EntityPose {
                position,
                orientation: [0.0, 0.0, 1.0, 0.0],
                velocity: [0.0, 0.0, 0.0, 0.0],
                scale: 1.0,
            },
            data,
        },
        display_name: None,
        tags: vec!["persist".to_string()],
        last_saved_ms: now_unix_ms(),
    }
}

fn materialize_loaded_world(loaded: &LoadedState) -> RegionChunkWorld {
    chunk_payloads_to_voxel_world(
        loaded.global.base_world_kind.to_runtime(),
        &loaded.world_chunk_payloads,
    )
    .expect("materialize loaded world")
}

#[test]
fn save_and_load_roundtrip_preserves_world_entities_players() {
    let root = test_root("roundtrip");
    let now_ms = now_unix_ms();

    let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(3));
    world.set_voxel(40, 2, -5, 17, LegacyVoxel(7));

    let entities = vec![PersistedEntityRecord {
        entity_id: 42,
        entity: Entity {
            namespace: 0,
            entity_type: 2,
            pose: EntityPose {
                position: [2.0, 3.0, 4.0, 5.0],
                orientation: [0.0, 0.0, 1.0, 0.0],
                velocity: [0.1, 0.2, 0.3, 0.4],
                scale: 0.75,
            },
            data: vec![1, 2, 3, 4],
        },
        display_name: Some("spin".to_string()),
        tags: vec!["demo".to_string()],
        last_saved_ms: now_ms,
    }];
    let players = vec![PlayerRecord {
        player_id: 7,
        position: [8.0, 9.0, 10.0, 11.0],
        orientation: [0.0, 0.0, 1.0, 0.0],
        tags: vec!["tester".to_string()],
        inventory_payload: vec![99],
        last_saved_ms: now_ms,
    }];
    let empty_regions = HashSet::new();

    let save_result = save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &entities,
            players: &players,
            world_seed: 4242,
            next_entity_id: 1000,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");
    assert_eq!(save_result.generation, 1);

    let loaded = load_state(&root).expect("load state");
    assert_eq!(loaded.manifest.version, SAVE_FORMAT_VERSION);
    assert_eq!(loaded.global.world_seed, 4242);
    assert_eq!(loaded.global.next_entity_id, 1000);
    assert_eq!(loaded.players.players.len(), 1);
    assert_eq!(loaded.entities.len(), 1);
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), LegacyVoxel(3));
    assert_eq!(loaded_world.get_voxel(40, 2, -5, 17), LegacyVoxel(7));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_respects_dirty_block_regions_when_not_forced() {
    let root = test_root("dirty-block-regions");
    let now_ms = now_unix_ms();
    let chunk_size = CHUNK_SIZE as i32;
    let empty_regions = HashSet::new();

    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(2));
    world.set_voxel(chunk_size * 10 + 1, 1, 1, 1, LegacyVoxel(3));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");

    world.set_voxel(1, 1, 1, 1, LegacyVoxel(4));
    world.set_voxel(chunk_size * 10 + 1, 1, 1, 1, LegacyVoxel(9));

    let mut dirty_blocks = HashSet::new();
    dirty_blocks.insert(region_from_chunk([0, 0, 0, 0], DEFAULT_REGION_CHUNK_EDGE));
    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 1,
            dirty_block_regions: &dirty_blocks,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: false,
            force_full_entities: false,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("incremental save");

    let loaded = load_state(&root).expect("load state");
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), LegacyVoxel(4));
    assert_eq!(
        loaded_world.get_voxel(chunk_size * 10 + 1, 1, 1, 1),
        LegacyVoxel(3)
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_respects_dirty_entity_regions_when_not_forced() {
    let root = test_root("dirty-entity-regions");
    let now_ms = now_unix_ms();
    let chunk_size = CHUNK_SIZE as f32;
    let empty_regions = HashSet::new();
    let world = RegionChunkWorld::new();

    let entity_a_chunk = [0, 0, 0, 0];
    let entity_b_chunk = [20, 0, 0, 0];
    let entity_a_pos = [1.0, 1.0, 1.0, 1.0];
    let entity_b_pos = [chunk_size * 20.0 + 1.0, 1.0, 1.0, 1.0];
    let entities_initial = vec![
        test_entity(1, entity_a_pos, vec![1]),
        test_entity(2, entity_b_pos, vec![2]),
    ];

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &entities_initial,
            players: &[],
            world_seed: 1,
            next_entity_id: 3,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");

    let entities_next = vec![
        test_entity(1, entity_a_pos, vec![9]),
        test_entity(2, entity_b_pos, vec![8]),
    ];
    let mut dirty_entities = HashSet::new();
    dirty_entities.insert(region_from_chunk(entity_a_chunk, DEFAULT_REGION_CHUNK_EDGE));
    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &entities_next,
            players: &[],
            world_seed: 1,
            next_entity_id: 3,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &dirty_entities,
            force_full_blocks: false,
            force_full_entities: false,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("incremental save");

    let loaded = load_state(&root).expect("load state");
    let mut loaded_entities = loaded.entities;
    loaded_entities.sort_unstable_by_key(|entity| entity.entity_id);
    assert_eq!(loaded_entities.len(), 2);
    assert_eq!(loaded_entities[0].entity_id, 1);
    assert_eq!(loaded_entities[0].entity.data, vec![9]);
    assert_eq!(loaded_entities[1].entity_id, 2);
    assert_eq!(loaded_entities[1].entity.data, vec![2]);
    assert_eq!(
        region_from_chunk(
            chunk_from_world_position(loaded_entities[1].entity.pose.position),
            DEFAULT_REGION_CHUNK_EDGE,
        ),
        region_from_chunk(entity_b_chunk, DEFAULT_REGION_CHUNK_EDGE)
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_region_filtered_world_and_entities() {
    let root = test_root("region-filtered-load");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let chunk_size_i32 = CHUNK_SIZE as i32;
    let chunk_size_f32 = CHUNK_SIZE as f32;
    let chunk_a = [0, 0, 0, 0];
    let chunk_b = [DEFAULT_REGION_CHUNK_EDGE, 0, 0, 0];

    let mut world = RegionChunkWorld::new();
    world.set_voxel(chunk_a[0] * chunk_size_i32 + 1, 1, 1, 1, LegacyVoxel(2));
    world.set_voxel(chunk_b[0] * chunk_size_i32 + 1, 1, 1, 1, LegacyVoxel(5));

    let entities = vec![
        test_entity(1, [1.0, 2.0, 3.0, 4.0], vec![1]),
        test_entity(
            2,
            [chunk_b[0] as f32 * chunk_size_f32 + 1.0, 2.0, 3.0, 4.0],
            vec![2],
        ),
    ];

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &entities,
            players: &[],
            world_seed: 1,
            next_entity_id: 3,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let mut region_filter = HashSet::new();
    region_filter.insert(region_from_chunk(chunk_a, DEFAULT_REGION_CHUNK_EDGE));

    let loaded_chunk_payloads =
        load_world_chunk_payloads_for_regions(&root, &region_filter).expect("load chunks");
    assert_eq!(loaded_chunk_payloads.len(), 1);
    assert_eq!(loaded_chunk_payloads[0].0, chunk_key_from_i32(chunk_a));
    let reg = test_content_registry();
    let legacy_payload = loaded_chunk_payloads[0]
        .1
        .to_material_token_payload(|block| {
            reg.block_material_token(block.namespace, block.block_type) as u8
        });
    let chunk =
        legacy_chunk_from_field_chunk_payload(&legacy_payload).expect("decode chunk payload");
    assert_eq!(chunk.get(1, 1, 1, 1), LegacyVoxel(2));

    let loaded_entities = load_entities_for_regions(&root, &region_filter).expect("load ents");
    assert_eq!(loaded_entities.len(), 1);
    assert_eq!(loaded_entities[0].entity_id, 1);

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_bounds_filtered_world_chunks() {
    let root = test_root("bounds-filtered-load");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let chunk_size_i32 = CHUNK_SIZE as i32;
    let chunk_a = [0, 0, 0, 0];
    let chunk_b = [3, 0, 0, 0];

    let mut world = RegionChunkWorld::new();
    world.set_voxel(chunk_a[0] * chunk_size_i32 + 1, 1, 1, 1, LegacyVoxel(2));
    world.set_voxel(chunk_b[0] * chunk_size_i32 + 1, 1, 1, 1, LegacyVoxel(5));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let only_chunk_a = Aabb4i::from_lattice_bounds(chunk_a, chunk_a, 0);
    let loaded_a = load_world_chunk_payloads_for_bounds(&root, only_chunk_a).expect("load chunk a");
    assert_eq!(loaded_a.len(), 1);
    assert_eq!(loaded_a[0].0, chunk_key_from_i32(chunk_a));

    let both_bounds = Aabb4i::from_lattice_bounds(chunk_a, chunk_b, 0);
    let loaded_both =
        load_world_chunk_payloads_for_bounds(&root, both_bounds).expect("load both chunks");
    assert_eq!(loaded_both.len(), 2);
    assert_eq!(loaded_both[0].0, chunk_key_from_i32(chunk_a));
    assert_eq!(loaded_both[1].0, chunk_key_from_i32(chunk_b));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_world_subtree_core_for_bounds_keeps_uniform_leaf() {
    let root = test_root("bounds-core-uniform");
    let index = IndexPayload {
        generation: 1,
        root_node_id: 7,
        entity_root_node_id: None,
        nodes: vec![test_index_node(
            7,
            [-1024, -64, -1024, -1024],
            [1024, 64, 1024, 1024],
            IndexNodeKind::LeafUniform {
                block: BlockData::simple(0, 11),
            },
        )],
    };
    let bounds = Aabb4i::from_i32([-4, -2, -4, -4], [4, 2, 4, 4]);
    let core = load_world_subtree_core_for_bounds_from_index(&root, &index, bounds)
        .expect("load subtree core");
    assert_eq!(core.bounds, bounds);
    assert!(matches!(
        core.kind,
        RegionNodeKind::Uniform(ref block) if block.block_type == 11
    ));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_from_chunk_payloads_duplicate_chunk_latest_wins() {
    let root = test_root("chunk-payload-latest-wins");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let chunk_pos = [2, 0, 0, 0];
    let chunk_size = CHUNK_SIZE as i32;

    let save_result = save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: vec![
                (
                    chunk_key_from_i32(chunk_pos),
                    0i8,
                    ResolvedChunkPayload::uniform(BlockData::simple(0, 3)),
                ),
                (
                    chunk_key_from_i32(chunk_pos),
                    0i8,
                    ResolvedChunkPayload::uniform(BlockData::simple(0, 9)),
                ),
            ],
            entities: &[],
            players: &[],
            world_seed: 99,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");
    assert_eq!(save_result.generation, 1);

    let loaded = load_state(&root).expect("load state");
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(
        loaded_world.get_voxel(chunk_pos[0] * chunk_size + 1, 1, 1, 1),
        LegacyVoxel(9)
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_from_chunk_payloads_merges_dirty_region_without_full_snapshot() {
    let root = test_root("chunk-payload-dirty-region-merge");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let chunk_a = [0, 0, 0, 0];
    let chunk_b = [DEFAULT_REGION_CHUNK_EDGE, 0, 0, 0];
    let mut dirty_regions = HashSet::new();
    dirty_regions.insert(region_from_chunk(chunk_a, DEFAULT_REGION_CHUNK_EDGE));
    let chunk_size = CHUNK_SIZE as i32;

    save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: vec![
                (
                    chunk_key_from_i32(chunk_a),
                    0i8,
                    ResolvedChunkPayload::uniform(BlockData::simple(0, 3)),
                ),
                (
                    chunk_key_from_i32(chunk_b),
                    0i8,
                    ResolvedChunkPayload::uniform(BlockData::simple(0, 7)),
                ),
            ],
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 2,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial full save");

    save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: vec![(
                chunk_key_from_i32(chunk_a),
                0i8,
                ResolvedChunkPayload::uniform(BlockData::simple(0, 9)),
            )],
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 2,
            dirty_block_regions: &dirty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: false,
            force_full_entities: false,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("dirty region save");

    let loaded = load_state(&root).expect("load state");
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(
        loaded_world.get_voxel(chunk_a[0] * chunk_size + 1, 1, 1, 1),
        LegacyVoxel(9)
    );
    assert_eq!(
        loaded_world.get_voxel(chunk_b[0] * chunk_size + 1, 1, 1, 1),
        LegacyVoxel(7)
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_from_chunk_payloads_roundtrip_preserves_world() {
    let root = test_root("roundtrip-chunk-payloads");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(3));
    world.set_voxel(40, 2, -5, 17, LegacyVoxel(7));

    let chunk_payloads: Vec<(ChunkKey, i8, ResolvedChunkPayload)> = world
        .chunks
        .iter()
        .map(|(&chunk_pos, chunk)| {
            (
                chunk_key_from_i32(chunk_pos),
                0i8,
                ResolvedChunkPayload::from_payload_with_static_palette(
                    field_chunk_payload_from_legacy_chunk(chunk),
                ),
            )
        })
        .collect();

    let save_result = save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: world.base_kind(),
            chunk_payloads,
            entities: &[],
            players: &[],
            world_seed: 4242,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state from chunk payloads");
    assert_eq!(save_result.generation, 1);

    let loaded = load_state(&root).expect("load state");
    assert_eq!(loaded.global.world_seed, 4242);
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(1, 1, 1, 1), LegacyVoxel(3));
    assert_eq!(loaded_world.get_voxel(40, 2, -5, 17), LegacyVoxel(7));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_from_chunk_payload_patch_persists_without_overlapping_children() {
    let root = test_root("chunk-payload-patch-no-overlap");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut dense_blocks = vec![BlockData::simple(0, 1); CHUNK_VOLUME];
    dense_blocks[0] = BlockData::simple(0, 2);
    let dense_resolved =
        ResolvedChunkPayload::from_dense_blocks(&dense_blocks).expect("dense resolved");

    let mut initial_payloads = Vec::new();
    for x in 0..=3 {
        initial_payloads.push((
            chunk_key_from_i32([x, 0, 0, 0]),
            0i8,
            dense_resolved.clone(),
        ));
        initial_payloads.push((
            chunk_key_from_i32([x, 1, 0, 0]),
            0i8,
            dense_resolved.clone(),
        ));
    }

    let initial = save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: initial_payloads,
            entities: &[],
            players: &[],
            world_seed: 7,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");
    assert_eq!(initial.generation, 1);

    let patched = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: vec![(
                chunk_key_from_i32([1, 0, 0, 0]),
                0i8,
                Some(ResolvedChunkPayload::uniform(BlockData::simple(0, 9))),
            )],
            world_seed: 7,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("patch save")
    .expect("patch should write a generation");
    assert_eq!(patched.generation, 2);

    let loaded = load_state(&root).expect("load patched state");
    let loaded_world = materialize_loaded_world(&loaded);
    let chunk_size = CHUNK_SIZE as i32;
    assert_eq!(loaded_world.get_voxel(chunk_size, 0, 0, 0), LegacyVoxel(9));
    assert_eq!(loaded_world.get_voxel(0, chunk_size, 0, 0), LegacyVoxel(2));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn migrate_legacy_world_to_v4_roundtrip() {
    let root = test_root("migrate");
    let legacy_world = root.join("legacy.v4dw");
    let output_root = root.join("migrated-v4");

    let mut world = RegionChunkWorld::new();
    world.set_voxel(2, 2, 2, 2, LegacyVoxel(9));
    {
        let mut writer = BufWriter::new(File::create(&legacy_world).expect("create legacy"));
        crate::migration::legacy_world_io::save_world(&world, &mut writer).expect("save legacy");
        writer.flush().expect("flush legacy");
    }

    let save_result = crate::save_v4_migration::migrate_legacy_world_to_v4(
        &legacy_world,
        None,
        &output_root,
        777,
        now_unix_ms(),
    )
    .expect("migrate legacy to v4");
    assert_eq!(save_result.generation, 1);

    let loaded = load_state(&output_root).expect("load migrated");
    assert_eq!(loaded.global.world_seed, 777);
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(2, 2, 2, 2), LegacyVoxel(9));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn migrate_v3_save_to_v4_roundtrip() {
    let root = test_root("migrate-v3");
    let v3_root = root.join("input-v3");
    let output_root = root.join("output-v4");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let grid_floor = crate::content_registry::block_data_from_material_token(11);
    let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: grid_floor,
    });
    world.set_voxel(4, 1, 2, -3, LegacyVoxel(8));

    let entities = vec![crate::migration::save_v3::PersistedEntityRecord {
        entity_id: 99,
        class: crate::migration::save_v3::EntityClass::Accent,
        kind: crate::migration::save_v3::EntityKind::TestCube,
        position: [1.0, 2.0, 3.0, 4.0],
        orientation: [0.0, 0.0, 1.0, 0.0],
        velocity: [0.0, 0.0, 0.0, 0.0],
        scale: 1.0,
        material: 7,
        display_name: Some("marker".to_string()),
        tags: vec!["persist".to_string()],
        payload: vec![9, 8, 7],
        last_saved_ms: now_ms,
    }];
    let players = vec![crate::migration::save_v3::PlayerRecord {
        player_id: 123,
        position: [0.0, 1.0, 2.0, 3.0],
        orientation: [0.0, 0.0, 1.0, 0.0],
        tags: vec!["p".to_string()],
        inventory_payload: vec![5, 4, 3],
        last_saved_ms: now_ms,
    }];
    let custom_payload = vec![42, 24, 7];

    crate::migration::save_v3::save_state(
        &v3_root,
        crate::migration::save_v3::SaveRequest {
            world: &world,
            entities: &entities,
            players: &players,
            world_seed: 2026,
            next_entity_id: 1000,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            custom_global_payload: Some(custom_payload.clone()),
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("create v3 input");

    let save_result = crate::save_v4_migration::migrate_v3_save_to_v4(
        &v3_root,
        &output_root,
        false,
        now_ms.saturating_add(1),
    )
    .expect("migrate v3 -> v4");
    assert_eq!(save_result.generation, 1);

    let loaded = load_state(&output_root).expect("load v4 output");
    assert_eq!(loaded.manifest.version, SAVE_FORMAT_VERSION);
    assert_eq!(loaded.global.world_seed, 2026);
    assert_eq!(loaded.global.next_entity_id, 1000);
    assert_eq!(loaded.global.custom_global_payload, custom_payload);
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(4, 1, 2, -3), LegacyVoxel(8));
    assert_eq!(loaded.entities.len(), 1);
    assert_eq!(loaded.players.players.len(), 1);

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn validate_index_rejects_overlapping_siblings() {
    // Two children at the same world-space bounds → overlap
    let index = IndexPayload {
        generation: 1,
        root_node_id: 0,
        entity_root_node_id: None,
        nodes: vec![
            test_index_node(
                0,
                [0, 0, 0, 0],
                [16, 8, 8, 8],
                IndexNodeKind::Branch {
                    child_node_ids: vec![1, 2],
                },
            ),
            test_index_node(1, [0, 0, 0, 0], [8, 8, 8, 8], IndexNodeKind::LeafEmpty),
            test_index_node(2, [0, 0, 0, 0], [8, 8, 8, 8], IndexNodeKind::LeafEmpty),
        ],
    };

    let err = validate_index_payload(&index).expect_err("overlap should fail");
    assert!(err.to_string().contains("overlapping children"));
}

#[test]
fn build_temp_tree_from_leaves_emits_non_overlapping_branch_children() {
    // These leaves are disjoint, but BVH-style grouping can produce overlapping
    // sibling union bounds. The temp-tree builder must avoid that pattern.
    // World-space half-open bounds: disjoint in Y, overlapping in X.
    // If BVH grouped by X, the Y-union bounds would overlap — the builder
    // must avoid that by emitting flat children.
    let leaves = vec![
        LeafDescriptor {
            min: [0, 0, 0, 0].map(ChunkCoord::from_num),
            max: [800, 8, 8, 8].map(ChunkCoord::from_num),
            scale_exp: 0,
            kind: IndexNodeKind::LeafUniform {
                block: BlockData::simple(0, 1),
            },
        },
        LeafDescriptor {
            min: [0, 16, 0, 0].map(ChunkCoord::from_num),
            max: [800, 24, 8, 8].map(ChunkCoord::from_num),
            scale_exp: 0,
            kind: IndexNodeKind::LeafUniform {
                block: BlockData::simple(0, 1),
            },
        },
        LeafDescriptor {
            min: [400, 8, 0, 0].map(ChunkCoord::from_num),
            max: [1200, 16, 8, 8].map(ChunkCoord::from_num),
            scale_exp: 0,
            kind: IndexNodeKind::LeafUniform {
                block: BlockData::simple(0, 1),
            },
        },
        LeafDescriptor {
            min: [400, 24, 0, 0].map(ChunkCoord::from_num),
            max: [1200, 32, 8, 8].map(ChunkCoord::from_num),
            scale_exp: 0,
            kind: IndexNodeKind::LeafUniform {
                block: BlockData::simple(0, 1),
            },
        },
    ];

    let mut tree = build_temp_tree_from_leaves(&leaves).expect("temp tree");
    canonicalize_temp_tree(&mut tree);

    let mut nodes = Vec::<IndexNode>::new();
    let root_node_id = flatten_temp_tree(&tree, &mut nodes);
    let index = IndexPayload {
        generation: 1,
        root_node_id,
        entity_root_node_id: None,
        nodes,
    };

    validate_index_payload(&index).expect("leaf-based temp tree should validate");
}

#[test]
fn validate_index_rejects_non_canonical_child_order() {
    // Children ordered [2,1] but node 1 has lower bounds → non-canonical
    let index = IndexPayload {
        generation: 1,
        root_node_id: 0,
        entity_root_node_id: None,
        nodes: vec![
            test_index_node(
                0,
                [0, 0, 0, 0],
                [16, 8, 8, 8],
                IndexNodeKind::Branch {
                    child_node_ids: vec![2, 1],
                },
            ),
            test_index_node(1, [0, 0, 0, 0], [8, 8, 8, 8], IndexNodeKind::LeafEmpty),
            test_index_node(2, [8, 0, 0, 0], [16, 8, 8, 8], IndexNodeKind::LeafEmpty),
        ],
    };

    let err = validate_index_payload(&index).expect_err("order should fail");
    assert!(err.to_string().contains("non-canonical child ordering"));
}

#[test]
fn validate_index_rejects_wrong_world_leaf_blob_type() {
    let index = IndexPayload {
        generation: 1,
        root_node_id: 0,
        entity_root_node_id: None,
        nodes: vec![test_index_node(
            0,
            [0, 0, 0, 0],
            [8, 8, 8, 8],
            IndexNodeKind::LeafChunkArray {
                chunk_array_ref: BlobRef {
                    data_file_id: 1,
                    record_offset: 12,
                    record_len: 64,
                    record_crc32: 0,
                    blob_type: BLOB_KIND_ENTITY,
                    blob_version: ENTITY_BLOB_VERSION,
                },
            },
        )],
    };

    let err = validate_index_payload(&index).expect_err("wrong blob type should fail");
    assert!(err.to_string().contains("expected 2"));
}

#[test]
fn save_state_writes_deterministic_index_for_equivalent_worlds() {
    let root_a = test_root("deterministic-a");
    let root_b = test_root("deterministic-b");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut world_a = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world_a.set_voxel(1, 2, 3, 4, LegacyVoxel(5));
    world_a.set_voxel(-8, 0, 1, -3, LegacyVoxel(9));
    world_a.set_voxel(33, 7, -2, 12, LegacyVoxel(4));

    let mut world_b = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world_b.set_voxel(33, 7, -2, 12, LegacyVoxel(4));
    world_b.set_voxel(1, 2, 3, 4, LegacyVoxel(5));
    world_b.set_voxel(-8, 0, 1, -3, LegacyVoxel(9));

    save_state_from_world(
        &root_a,
        SaveWorldRequest {
            world: &world_a,
            entities: &[],
            players: &[],
            world_seed: 99,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save root a");
    save_state_from_world(
        &root_b,
        SaveWorldRequest {
            world: &world_b,
            entities: &[],
            players: &[],
            world_seed: 99,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save root b");

    let manifest_a = load_manifest(&root_a).expect("manifest a");
    let manifest_b = load_manifest(&root_b).expect("manifest b");
    let index_a = std::fs::read(root_a.join(&manifest_a.index_file)).expect("read index a");
    let index_b = std::fs::read(root_b.join(&manifest_b.index_file)).expect("read index b");
    assert_eq!(index_a, index_b);

    let _ = std::fs::remove_dir_all(root_a);
    let _ = std::fs::remove_dir_all(root_b);
}

#[test]
fn save_state_reuses_identical_chunk_payload_within_single_save() {
    let root = test_root("reuse-within-save");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(7));
    world.set_voxel(17, 1, 1, 1, LegacyVoxel(7));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let manifest = load_manifest(&root).expect("load manifest");
    let index = read_index_file(root.join(&manifest.index_file)).expect("load index");
    let chunk_array_refs = collect_world_chunk_array_refs(&index);
    assert_eq!(chunk_array_refs.len(), 2);

    let mut payload_refs = Vec::<BlobRef>::new();
    for chunk_array_ref in chunk_array_refs {
        let payload = read_blob_payload(&root, &chunk_array_ref).expect("read chunk array blob");
        let chunk_array_blob: ChunkArrayBlob =
            postcard::from_bytes(&payload).expect("decode chunk array blob");
        assert_eq!(chunk_array_blob.payload_palette.len(), 1);
        payload_refs.push(chunk_array_blob.payload_palette[0].clone());
    }

    assert_eq!(payload_refs[0], payload_refs[1]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_state_packs_contiguous_non_uniform_row_into_single_chunk_array() {
    let root = test_root("pack-contiguous-row");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let chunk_size = CHUNK_SIZE as i32;

    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(7));
    world.set_voxel(chunk_size + 1, 1, 1, 1, LegacyVoxel(9));
    world.set_voxel(chunk_size * 2 + 1, 1, 1, 1, LegacyVoxel(7));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 1,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let manifest = load_manifest(&root).expect("load manifest");
    let index = read_index_file(root.join(&manifest.index_file)).expect("load index");
    let chunk_array_refs = collect_world_chunk_array_refs(&index);
    assert_eq!(chunk_array_refs.len(), 1);

    let payload = read_blob_payload(&root, &chunk_array_refs[0]).expect("read chunk array blob");
    let chunk_array_blob: ChunkArrayBlob =
        postcard::from_bytes(&payload).expect("decode chunk array blob");
    assert_eq!(chunk_array_blob.volume_min_chunk, [0, 0, 0, 0]);
    assert_eq!(chunk_array_blob.volume_max_chunk, [2, 0, 0, 0]);
    assert_eq!(chunk_array_blob.scale_exp, 0);
    assert_eq!(chunk_array_blob.payload_palette.len(), 2);
    assert_eq!(chunk_array_blob.index_codec, ChunkArrayIndexCodec::DenseU16);

    let chunk_array_data = ChunkArrayData {
        bounds: Aabb4i::from_lattice_bounds(
            chunk_array_blob.volume_min_chunk,
            chunk_array_blob.volume_max_chunk,
            chunk_array_blob.scale_exp,
        ),
        scale_exp: chunk_array_blob.scale_exp,
        chunk_palette: vec![FieldChunkPayload::Empty; chunk_array_blob.payload_palette.len()],
        index_codec: chunk_array_blob.index_codec,
        index_data: chunk_array_blob.index_data,
        default_chunk_idx: chunk_array_blob.default_palette_index,
        block_palette: chunk_array_blob.block_palette,
    };
    let dense = chunk_array_data
        .decode_dense_indices()
        .expect("decode dense indices");
    assert_eq!(dense, vec![0, 1, 0]);

    let _ = std::fs::remove_dir_all(root);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyChunkArrayBlobV2 {
    volume_min_chunk: [i32; 4],
    volume_max_chunk: [i32; 4],
    payload_palette: Vec<BlobRef>,
    block_palette: Vec<BlockData>,
    index_codec: ChunkArrayIndexCodec,
    index_data: Vec<u8>,
    default_palette_index: Option<u16>,
}

#[test]
fn chunk_array_blob_without_scale_field_defaults_to_zero() {
    let legacy_blob = LegacyChunkArrayBlobV2 {
        volume_min_chunk: [0, 0, 0, 0],
        volume_max_chunk: [0, 0, 0, 0],
        payload_palette: Vec::new(),
        block_palette: vec![BlockData::AIR],
        index_codec: ChunkArrayIndexCodec::DenseU16,
        index_data: vec![0, 0],
        default_palette_index: None,
    };
    let bytes = postcard::to_stdvec(&legacy_blob).expect("encode v2 blob");
    let decoded = decode_chunk_array_blob(&bytes).expect("decode v2 blob with fallback");
    assert_eq!(decoded.scale_exp, 0);
}

#[test]
fn save_state_identical_resave_does_not_append_data_blobs() {
    let root = test_root("no-thrash-resave");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world.set_voxel(3, 2, 1, 0, LegacyVoxel(5));
    world.set_voxel(-13, 4, -2, 7, LegacyVoxel(9));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 123,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");
    let manifest_a = load_manifest(&root).expect("manifest a");
    let index_a = read_index_file(root.join(&manifest_a.index_file)).expect("index a");
    let refs_a = collect_world_chunk_array_refs(&index_a);
    let data_bytes_a = total_data_bytes(&root);

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 123,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("second save");
    let manifest_b = load_manifest(&root).expect("manifest b");
    let index_b = read_index_file(root.join(&manifest_b.index_file)).expect("index b");
    let refs_b = collect_world_chunk_array_refs(&index_b);
    let data_bytes_b = total_data_bytes(&root);

    assert_eq!(refs_a, refs_b);
    assert_eq!(data_bytes_a, data_bytes_b);

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_state_rejects_corrupt_index_payload_checksum() {
    let root = test_root("corrupt-index-checksum");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(7));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 7,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let manifest = load_manifest(&root).expect("manifest");
    let index_path = root.join(&manifest.index_file);
    let mut bytes = std::fs::read(&index_path).expect("read index");
    assert!(
        bytes.len() > 32,
        "index payload should contain at least one node"
    );
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&index_path, bytes).expect("rewrite tampered index");

    let err = load_state(&root).expect_err("corrupt index should fail load");
    assert!(
        err.to_string()
            .contains("index file payload checksum mismatch"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_state_rejects_corrupt_blob_payload_checksum() {
    let root = test_root("corrupt-blob-checksum");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(7));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 7,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let manifest = load_manifest(&root).expect("manifest");
    let index = read_index_file(root.join(&manifest.index_file)).expect("index");
    let chunk_array_refs = collect_world_chunk_array_refs(&index);
    assert!(!chunk_array_refs.is_empty(), "expected chunk-array refs");
    let target = &chunk_array_refs[0];

    let data_path = data_file_path(&root, target.data_file_id);
    let mut bytes = std::fs::read(&data_path).expect("read data file");
    let payload_offset = target
        .record_offset
        .saturating_add(RECORD_HEADER_LEN as u64) as usize;
    assert!(
        payload_offset < bytes.len(),
        "blob payload offset should be in range"
    );
    bytes[payload_offset] ^= 0x01;
    std::fs::write(&data_path, bytes).expect("rewrite tampered data file");

    let err = load_state(&root).expect_err("corrupt blob should fail load");
    assert!(
        err.to_string().contains("blob payload checksum mismatch"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn load_state_rejects_corrupt_data_file_header() {
    let root = test_root("corrupt-data-header");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();
    let mut world = RegionChunkWorld::new();
    world.set_voxel(1, 1, 1, 1, LegacyVoxel(7));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 7,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save state");

    let manifest = load_manifest(&root).expect("manifest");
    let index = read_index_file(root.join(&manifest.index_file)).expect("index");
    let chunk_array_refs = collect_world_chunk_array_refs(&index);
    assert!(!chunk_array_refs.is_empty(), "expected chunk-array refs");
    let target = &chunk_array_refs[0];

    let data_path = data_file_path(&root, target.data_file_id);
    let mut bytes = std::fs::read(&data_path).expect("read data file");
    assert!(
        bytes.len() >= 8,
        "data header should contain version bytes at offset 4..8"
    );
    bytes[4] ^= 0x01;
    std::fs::write(&data_path, bytes).expect("rewrite tampered data file header");

    let err = load_state(&root).expect_err("corrupt data header should fail load");
    assert!(
        err.to_string().contains("unsupported data file version"),
        "unexpected error: {err}"
    );

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn disable_block_persistence_clears_block_overrides_and_keeps_custom_payload() {
    let root = test_root("disable-block-persistence");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    let mut world = RegionChunkWorld::new_with_base(BaseWorldKind::FlatFloor {
        material: BlockData::simple(0, 11),
    });
    world.set_voxel(8, 8, 8, 8, LegacyVoxel(3));

    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &world,
            entities: &[],
            players: &[],
            world_seed: 123,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");

    let payload = vec![7, 6, 5, 4, 3, 2, 1];
    let empty_world = RegionChunkWorld::new_with_base(world.base_kind());
    save_state_from_world(
        &root,
        SaveWorldRequest {
            world: &empty_world,
            entities: &[],
            players: &[],
            world_seed: 123,
            next_entity_id: 2,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: false,
            force_full_entities: false,
            player_entity_hints: None,
            custom_global_payload: Some(payload.clone()),
            disable_block_persistence: true,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("save without block persistence");

    let loaded = load_state(&root).expect("load state");
    assert_eq!(loaded.global.custom_global_payload, payload);
    let loaded_world = materialize_loaded_world(&loaded);
    assert_eq!(loaded_world.get_voxel(8, 8, 8, 8), LegacyVoxel(0));

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn patch_save_cluster_edits_preserve_non_dirty_chunks() {
    // Reproduce: editing multiple chunks in a cluster, then reloading —
    // verify that non-dirty chunks survive sequential patch saves.
    let root = test_root("patch-cluster-edits");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    // Create 8 non-uniform chunks in a row (forces ChunkArray storage).
    let mut initial_payloads = Vec::new();
    for x in 0..8i32 {
        let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];
        blocks[0] = BlockData::simple(0, (x + 1) as u32);
        blocks[1] = BlockData::simple(0, (x + 10) as u32);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        initial_payloads.push((chunk_key_from_i32([x, 0, 0, 0]), 0i8, resolved));
    }

    let initial = save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: initial_payloads.clone(),
            entities: &[],
            players: &[],
            world_seed: 99,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("initial save");
    assert_eq!(initial.generation, 1);

    // Patch save 1: edit 3 chunks in the middle (cluster edit).
    let mut patch1_dirty = Vec::new();
    for x in 2..=4i32 {
        let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];
        blocks[0] = BlockData::simple(0, (x + 100) as u32);
        blocks[1] = BlockData::simple(0, (x + 200) as u32);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        patch1_dirty.push((chunk_key_from_i32([x, 0, 0, 0]), 0i8, Some(resolved)));
    }

    let patch1 = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: patch1_dirty,
            world_seed: 99,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: now_ms.saturating_add(1),
        },
    )
    .expect("patch save 1")
    .expect("patch 1 should write a generation");
    assert_eq!(patch1.generation, 2);

    // Verify ALL 8 chunks survive.
    let bounds = Aabb4i::from_lattice_bounds([0, 0, 0, 0], [7, 0, 0, 0], 0);
    let loaded_chunks =
        load_world_chunk_payloads_for_bounds(&root, bounds).expect("load after patch 1");
    let loaded_map: HashMap<ChunkKey, ResolvedChunkPayload> = loaded_chunks.into_iter().collect();
    for x in 0..8i32 {
        let key = chunk_key_from_i32([x, 0, 0, 0]);
        let resolved = loaded_map
            .get(&key)
            .unwrap_or_else(|| panic!("chunk {key:?} missing after patch 1"));
        let blocks = resolved.dense_blocks();
        if (2..=4).contains(&x) {
            // Edited chunks should have new data.
            assert_eq!(
                blocks[0],
                BlockData::simple(0, (x + 100) as u32),
                "chunk {key:?} block[0] after patch 1"
            );
        } else {
            // Non-edited chunks should retain original data.
            assert_eq!(
                blocks[0],
                BlockData::simple(0, (x + 1) as u32),
                "chunk {key:?} block[0] after patch 1"
            );
        }
    }

    // Patch save 2: edit 2 more chunks (x=1 and x=5).
    let mut patch2_dirty = Vec::new();
    for x in [1i32, 5] {
        let mut blocks = vec![BlockData::AIR; CHUNK_VOLUME];
        blocks[0] = BlockData::simple(0, (x + 300) as u32);
        let resolved = ResolvedChunkPayload::from_dense_blocks(&blocks).expect("from dense blocks");
        patch2_dirty.push((chunk_key_from_i32([x, 0, 0, 0]), 0i8, Some(resolved)));
    }

    let patch2 = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: patch2_dirty,
            world_seed: 99,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: now_ms.saturating_add(2),
        },
    )
    .expect("patch save 2")
    .expect("patch 2 should write a generation");
    assert_eq!(patch2.generation, 3);

    // Verify ALL 8 chunks survive after second patch.
    let loaded_chunks_2 =
        load_world_chunk_payloads_for_bounds(&root, bounds).expect("load after patch 2");
    let loaded_map_2: HashMap<ChunkKey, ResolvedChunkPayload> =
        loaded_chunks_2.into_iter().collect();
    for x in 0..8i32 {
        let key = chunk_key_from_i32([x, 0, 0, 0]);
        let resolved = loaded_map_2
            .get(&key)
            .unwrap_or_else(|| panic!("chunk {key:?} missing after patch 2"));
        let blocks = resolved.dense_blocks();
        if x == 1 || x == 5 {
            // Newly edited in patch 2.
            assert_eq!(
                blocks[0],
                BlockData::simple(0, (x + 300) as u32),
                "chunk {key:?} block[0] after patch 2"
            );
        } else if (2..=4).contains(&x) {
            // Edited in patch 1, should still be there.
            assert_eq!(
                blocks[0],
                BlockData::simple(0, (x + 100) as u32),
                "chunk {key:?} block[0] after patch 2"
            );
        } else {
            // Never edited (x=0,6,7), should retain original data.
            assert_eq!(
                blocks[0],
                BlockData::simple(0, (x + 1) as u32),
                "chunk {key:?} block[0] after patch 2"
            );
        }
    }

    // Patch save 3: send a None payload for x=3 (simulating undo to virgin).
    let patch3 = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: vec![(chunk_key_from_i32([3, 0, 0, 0]), 0i8, None)],
            world_seed: 99,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: now_ms.saturating_add(3),
        },
    )
    .expect("patch save 3")
    .expect("patch 3 should write a generation");
    assert_eq!(patch3.generation, 4);

    // Verify: chunk x=3 should be gone (None payload = deleted from index).
    // All OTHER chunks should survive.
    let loaded_chunks_3 =
        load_world_chunk_payloads_for_bounds(&root, bounds).expect("load after patch 3");
    let loaded_map_3: HashMap<ChunkKey, ResolvedChunkPayload> =
        loaded_chunks_3.into_iter().collect();
    assert!(
        !loaded_map_3.contains_key(&chunk_key_from_i32([3, 0, 0, 0])),
        "chunk [3,0,0,0] should be removed after None payload"
    );
    for x in [0i32, 1, 2, 4, 5, 6, 7] {
        let key = chunk_key_from_i32([x, 0, 0, 0]);
        assert!(
            loaded_map_3.contains_key(&key),
            "chunk {key:?} missing after patch 3 (should have survived)"
        );
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn save_load_roundtrip_preserves_multi_scale_chunks() {
    use crate::shared::spatial::chunk_key_from_lattice;

    let root = test_root("multi-scale-roundtrip");
    let now_ms = now_unix_ms();
    let empty_regions = HashSet::new();

    // Place one unique block per scale at lattice positions that produce
    // non-overlapping world-space bounds (the save format's tree requires
    // non-overlapping siblings). At lattice [1,0,0,0] per scale, chunks
    // tile perfectly along X: scale-2→[0,2), scale-1→[4,8), scale0→[8,16),
    // scale1→[16,32), scale2→[32,64).
    let scales: &[(i8, [i32; 4], u32)] = &[
        (-2, [0, 0, 0, 0], 10),
        (-1, [1, 0, 0, 0], 20),
        (0, [1, 0, 0, 0], 30),
        (1, [1, 0, 0, 0], 40),
        (2, [1, 0, 0, 0], 50),
    ];

    let mut payloads = Vec::new();
    for &(scale_exp, lattice, block_type) in scales {
        let key = chunk_key_from_lattice(lattice, scale_exp);
        let resolved = ResolvedChunkPayload::uniform(BlockData::simple(0, block_type));
        payloads.push((key, scale_exp, resolved));
    }

    save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: payloads,
            entities: &[],
            players: &[],
            world_seed: 42,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms,
        },
    )
    .expect("save multi-scale state");

    // Query with a wide bounds that covers all scales.
    let bounds = Aabb4i::from_lattice_bounds([-10, -10, -10, -10], [10, 10, 10, 10], 0);
    let loaded =
        load_world_chunk_payloads_for_bounds(&root, bounds).expect("load multi-scale state");
    let loaded_map: HashMap<ChunkKey, ResolvedChunkPayload> = loaded.into_iter().collect();

    for &(scale_exp, lattice, expected_block_type) in scales {
        let key = chunk_key_from_lattice(lattice, scale_exp);
        let resolved = loaded_map
                .get(&key)
                .unwrap_or_else(|| panic!(
                    "chunk at scale_exp={scale_exp} lattice={lattice:?} (key={key:?}) missing after load"
                ));
        let blocks = resolved.dense_blocks();
        assert_eq!(
            blocks[0],
            BlockData::simple(0, expected_block_type),
            "block mismatch at scale_exp={scale_exp} lattice={lattice:?}"
        );
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn coarsest_lattice_aligned_scale_exact_at_scale0() {
    // Bounds [0, 8) on all axes — exactly one scale-0 chunk.
    let bounds = Aabb4i::from_lattice_bounds([0; 4], [0; 4], 0);
    assert_eq!(coarsest_lattice_aligned_scale(&bounds, 0), 0);
}

#[test]
fn coarsest_lattice_aligned_scale_non_aligned() {
    // Bounds [2, 8) on axis 0 — produced by carving [0,2) out of [0,8).
    // Not aligned at scale 0 (lattice snaps to [0,8)).
    // At scale -2, chunk world size = 2, so [2,8) maps to lattice [1,3] → exact.
    let bounds = Aabb4i {
        min: [
            ChunkCoord::from_num(2),
            ChunkCoord::ZERO,
            ChunkCoord::ZERO,
            ChunkCoord::ZERO,
        ],
        max: [
            ChunkCoord::from_num(8),
            ChunkCoord::from_num(8),
            ChunkCoord::from_num(8),
            ChunkCoord::from_num(8),
        ],
    };
    let scale = coarsest_lattice_aligned_scale(&bounds, 0);
    assert!(
        scale < 0,
        "non-aligned bounds must use a finer scale, got {scale}"
    );
    // Verify the chosen scale round-trips correctly.
    let (lat_min, lat_max) = bounds.to_chunk_lattice_bounds(scale);
    let reconstructed = Aabb4i::from_lattice_bounds(lat_min, lat_max, scale);
    assert_eq!(
        reconstructed, bounds,
        "round-trip must be exact at scale {scale}"
    );
}

/// Regression test: a patch save that carves a Uniform leaf, followed by
/// bulk materialization of the saved index, must correctly represent the
/// carved Uniform pieces without expanding them via lattice snapping.
#[test]
fn patch_save_carved_uniform_materializes_correctly() {
    let root = test_root("carved-uniform-materialize");

    let block_a = BlockData::simple(0, 1);
    let block_b = BlockData::simple(0, 2);

    // Initial save: a single scale-0 chunk filled with block_a at [0,0,0,0].
    let payload_a = ResolvedChunkPayload::uniform(block_a.clone());
    let key_origin = chunk_key_from_lattice([0, 0, 0, 0], 0);
    let empty_regions = HashSet::new();
    let _initial = save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: vec![(key_origin, 0, payload_a)],
            entities: &[],
            players: &[],
            world_seed: 42,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: 1000,
        },
    )
    .expect("initial save");

    // Patch save: overwrite a sub-region with block_b at scale -2.
    // At scale -2, chunk world size = 2, so one chunk covers [0,2)^4.
    // This carves the original [0,8)^4 Uniform leaf.
    let key_fine = chunk_key_from_lattice([0, 0, 0, 0], -2);
    let payload_b = ResolvedChunkPayload::uniform(block_b.clone());
    let patch = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: vec![(key_fine, -2, Some(payload_b))],
            world_seed: 42,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: 2000,
        },
    )
    .expect("patch save");
    assert!(patch.is_some(), "patch save should produce a result");

    // Bulk-materialize the saved index. This exercises the code path
    // that was previously broken for non-lattice-aligned Uniform leaves.
    let metadata = load_state_metadata(&root).expect("load metadata");
    let materialized = materialize_world_chunk_payloads_from_index_filtered(
        &root,
        &metadata.index,
        None,
        true,
        None,
    )
    .expect("materialize");

    // The fine-scale edit at [0,2)^4 should be present.
    let fine_entry = materialized
        .iter()
        .find(|(k, se, _)| *k == key_fine && *se == -2);
    assert!(
        fine_entry.is_some(),
        "fine-scale edit should be present in materialized output"
    );

    // The carved Uniform pieces (block_a covering the rest of [0,8)^4)
    // must NOT expand back to the full [0,8)^4 lattice cell.
    // Check: no materialized chunk should claim to be at key_origin scale 0
    // (the original full chunk was carved and no longer exists as a whole).
    let has_original_full = materialized
        .iter()
        .any(|(k, se, _)| *k == key_origin && *se == 0);
    assert!(
        !has_original_full,
        "original full scale-0 chunk should not appear — it was carved"
    );

    // Verify block_a data IS present in the carved pieces.
    // The carved pieces may be Uniform or may have been re-encoded as
    // ChunkArrays containing block_a, so check both representations.
    let block_a_entries: Vec<_> = materialized
        .iter()
        .filter(|(_, _, p)| {
            if let Some(b) = p.uniform_block() {
                return b.block_type == block_a.block_type;
            }
            // Check dense blocks for non-uniform payloads
            let blocks = p.dense_blocks();
            blocks.iter().any(|b| b.block_type == block_a.block_type)
        })
        .collect();
    assert!(
        !block_a_entries.is_empty(),
        "block_a data should be present in materialized output; \
             got {} entries: {:?}",
        materialized.len(),
        materialized
            .iter()
            .map(|(k, se, p)| {
                let desc = if let Some(b) = p.uniform_block() {
                    format!("Uniform(type={})", b.block_type)
                } else {
                    format!("Chunks(blocks={})", p.dense_blocks().len())
                };
                format!("({k:?}, scale={se}, {desc})")
            })
            .collect::<Vec<_>>()
    );

    let _ = std::fs::remove_dir_all(root);
}

/// Regression: a Uniform chunk where block.scale_exp > chunk scale_exp
/// (e.g. scale-3 block stored in a scale-0 chunk) must survive a
/// two-phase save sequence: first save containing only this block, then
/// a second patch save adding blocks at a different scale.
#[test]
fn patch_save_preserves_uniform_with_mismatched_block_scale() {
    let root = test_root("uniform-mismatched-block-scale");
    let empty_regions = HashSet::new();

    // A scale-3 block placed in a scale-0 chunk (fills all 8^4 cells).
    let block_bottom = BlockData::simple(0, 1).at_scale(3);
    let payload_bottom = ResolvedChunkPayload::uniform(block_bottom.clone());
    let key_bottom = chunk_key_from_lattice([1, 1, 0, -1], 0);

    // Initial save: just the bottom block.
    save_state_from_chunk_payloads(
        &root,
        SaveChunkPayloadRequest {
            base_world_kind: BaseWorldKind::Empty,
            chunk_payloads: vec![(key_bottom, 0, payload_bottom.clone())],
            entities: &[],
            players: &[],
            world_seed: 42,
            next_entity_id: 1,
            dirty_block_regions: &empty_regions,
            dirty_entity_regions: &empty_regions,
            force_full_blocks: true,
            force_full_entities: true,
            player_entity_hints: None,
            custom_global_payload: None,
            disable_block_persistence: false,
            now_ms: 1000,
        },
    )
    .expect("initial save");

    // Patch save: add upper blocks at scale 1 (disjoint from bottom).
    let block_upper = BlockData::simple(0, 2).at_scale(3);
    let mut upper_blocks = vec![BlockData::AIR; CHUNK_VOLUME];
    // Fill a 4^4 sub-region (scale-3 block in scale-1 chunk).
    for w in 0..4 {
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let idx = x
                        + y * CHUNK_SIZE
                        + z * CHUNK_SIZE * CHUNK_SIZE
                        + w * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
                    upper_blocks[idx] = block_upper.clone();
                }
            }
        }
    }
    let payload_upper =
        ResolvedChunkPayload::from_dense_blocks(&upper_blocks).expect("upper payload");
    let key_upper = chunk_key_from_lattice([0, 1, -1, -1], 1);

    let patch = save_state_from_chunk_payload_patch(
        &root,
        SaveChunkPayloadPatchRequest {
            base_world_kind: BaseWorldKind::Empty,
            dirty_chunk_payloads: vec![(key_upper, 1, Some(payload_upper))],
            world_seed: 42,
            next_entity_id: 1,
            player_entity_hints: None,
            custom_global_payload: None,
            now_ms: 2000,
        },
    )
    .expect("patch save");
    assert!(patch.is_some());

    // Materialize and verify both blocks are present.
    let metadata = load_state_metadata(&root).expect("load metadata");
    let materialized = materialize_world_chunk_payloads_from_index_filtered(
        &root,
        &metadata.index,
        None,
        true,
        None,
    )
    .expect("materialize");

    let has_bottom = materialized.iter().any(|(k, se, p)| {
        *k == key_bottom && *se == 0 && p.uniform_block().map(|b| b.block_type) == Some(1)
    });
    let has_upper = materialized.iter().any(|(_, se, _)| *se == 1);

    assert!(
        has_bottom,
        "bottom scale-0 Uniform (block.scale_exp=3) must survive patch save; \
             got {} entries: {:?}",
        materialized.len(),
        materialized
            .iter()
            .map(|(k, se, p)| {
                let desc = if let Some(b) = p.uniform_block() {
                    format!(
                        "Uniform(type={}, block_scale={})",
                        b.block_type, b.scale_exp
                    )
                } else {
                    "ChunkArray".to_string()
                };
                format!("({k:?}, chunk_scale={se}, {desc})")
            })
            .collect::<Vec<_>>()
    );
    assert!(has_upper, "upper scale-1 chunk must be present");

    // Also verify production load path (streaming).
    let large_bounds = Aabb4i::from_i32([-200; 4], [200; 4]);
    let loaded_core =
        load_world_subtree_core_for_bounds_from_index(&root, &metadata.index, large_bounds)
            .expect("streaming load");

    // The loaded core should contain the bottom block somewhere.
    fn find_uniform_in_core(core: &RegionTreeCore, target_type: u32) -> bool {
        match &core.kind {
            RegionNodeKind::Uniform(b) => b.block_type == target_type,
            RegionNodeKind::Branch(children) => children
                .iter()
                .any(|c| find_uniform_in_core(c, target_type)),
            _ => false,
        }
    }
    assert!(
        find_uniform_in_core(&loaded_core, 1),
        "streaming load must include the bottom Uniform block"
    );

    let _ = std::fs::remove_dir_all(root);
}
