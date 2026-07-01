mod block_tick;
mod config;
mod core_state;
mod cpu_profile;
mod entities;
mod mob_sim;
mod procgen;
pub mod procgen_wasm;
mod runtime_net;
mod spawn_logic;
mod types;
mod world_cache;
pub mod world_field;

use self::block_tick::{run_block_ticks, BlockTickSpawnAction};
pub use self::config::{LocalConnection, RuntimeConfig, WorldGeneratorKind};
use self::core_state::{
    allocate_or_reserve_server_object_id, allocate_server_object_id, remove_entity_record,
    monotonic_ms, record_server_cpu_sample, upsert_entity_record, ServerState, SharedState,
};
use self::cpu_profile::ServerCpuProfile;
use self::entities::{EntityId, EntityStore};
use self::mob_sim::tick_entity_simulation_window;
use self::runtime_net::{
    handle_message, remove_client, spawn_client_thread, start_broadcast_thread,
};
use self::spawn_logic::{
    default_spawn_pose_for_client, entity_type_entry_for_token, env_flag_enabled, parse_spawn_vec4,
    phase_spider_next_phase_deadline, sanitize_player_name, spawn_usage_string,
};
use self::types::{
    ClientEntityReplicationBatch, CollisionChunkCacheEntry, EntityRecord, EntityRecordSummary,
    LiveReplicationFrame, MobNavPathResult, MobNavigationState, MobState, ReplicationEntry,
    PersistedMobEntry, PlayerState, QueuedExplosionEvent, QueuedPlayerMovementModifier,
};
use self::world_cache::ServerWorldCache;
use self::world_field::{QueryDetail, QueryVolume, ServerWorldOverlay, WorldField};
use crate::shared::entity_types::{EntityCategory, MobLocomotionMode, ENTITY_PLAYER_AVATAR};
use crate::shared::protocol::{
    ClientMessage, Entity, EntityPose, EntitySnapshot, EntityTransform, ServerMessage, WorldBounds,
    WorldSummary,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{self, BlockData, CHUNK_SIZE};
use std::cmp::Reverse;
use std::collections::{hash_map::Entry, BinaryHeap, HashMap, HashSet};
use std::io::{self, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

const STREAM_FAR_LOD_SCALE: i32 = 4;
const SERVER_CPU_PROFILE_INTERVAL: Duration = Duration::from_secs(2);
const ENTITY_INTEREST_RADIUS_PADDING_CHUNKS: i32 = 2;
const ENTITY_SIM_STEP_MAX_PER_BROADCAST: usize = 3;
// Unmoved entities are skipped in EntityTransforms; every Nth tick resends
// all visible poses as drift insurance (1 s at the default 10 Hz tick).
const TRANSFORM_KEEPALIVE_TICKS: u64 = 10;
const MOB_COLLISION_RADIUS_SCALE: f32 = 0.42;
const MOB_COLLISION_RADIUS_MIN: f32 = 0.20;
const MOB_COLLISION_RADIUS_MAX: f32 = 0.55;
const MOB_COLLISION_BINARY_STEPS: usize = 12;
const MOB_COLLISION_PUSHUP_STEP: f32 = 0.05;
const MOB_COLLISION_MAX_PUSHUP_STEPS: usize = 32;
const MOB_NAV_PATH_REPLAN_INTERVAL_MS: u64 = 450;
const MOB_NAV_PATH_GOAL_REPLAN_THRESHOLD_CELLS: i32 = 2;
const MOB_NAV_PATH_NODE_REACH_DISTANCE: f32 = 0.62;
const MOB_NAV_PATH_LOS_STEP: f32 = 0.30;
const MOB_NAV_PATH_MAX_SEARCH_STEPS: usize = 6144;
const MOB_NAV_PATH_MAX_SEARCH_RADIUS_CELLS: i32 = 42;
const MOB_NAV_PATH_GOAL_ADJUST_RADIUS_CELLS: i32 = 6;
const MOB_NAV_PATH_MAX_WAYPOINTS: usize = 96;
const MOB_NAV_DEBUG_MIN_INTERVAL_MS: u64 = 250;
const MOB_WALK_GROUND_STICK_STEP: f32 = 0.05;
const MOB_WALK_GROUND_STICK_MAX_DROP: f32 = 0.65;
const MOB_WALK_STEP_UP_MAX_HEIGHT: f32 = 1.10;
const MOB_WALK_STEP_UP_SAMPLE_STEP: f32 = 0.05;
const MOB_NAV_LOS_CACHE_MS: u64 = 200;
const MOB_NAV_CACHE_KEEP_RADIUS_CHUNKS: i32 = 16;
const MOB_NAV_CACHE_EVICT_BUDGET_PER_TICK: usize = 64;

const ITEM_PICKUP_RADIUS_SQ: f32 = 1.0; // 1 block
const ITEM_MAGNET_RADIUS_SQ: f32 = 9.0; // 3 blocks
const ITEM_MAGNET_LERP: f32 = 0.15; // 15% per sim step
const ITEM_SPAWN_DELAY_MS: u64 = 500; // Grace period before pickup

fn entity_snapshot_from_record(
    state: &ServerState,
    record: &EntityRecord,
) -> Option<(EntitySnapshot, [i32; 4])> {
    let mut snapshot = state.entity_store.snapshot(record.entity_id)?;
    snapshot.owner_client_id = record.owner_client_id;
    snapshot.display_name = record.display_name.clone();
    let chunk = world_chunk_from_position(snapshot.entity.pose.position);
    Some((snapshot, chunk))
}

fn collect_live_replication_frame(state: &ServerState) -> LiveReplicationFrame {
    let mut frame = LiveReplicationFrame::default();
    for record in state.entity_records.values() {
        let Some(entity_state) = state.entity_store.get(record.entity_id) else {
            continue;
        };
        let pose = entity_state.entity.pose.clone();
        let entry = ReplicationEntry {
            entity_id: record.entity_id,
            chunk: world_chunk_from_position(pose.position),
            pose,
            last_update_ms: entity_state.last_update_ms,
            pose_changed: false,
        };
        match record.category {
            EntityCategory::Player => {
                let Some(client_id) = record.owner_client_id else {
                    continue;
                };
                let Some(player) = state.players.get(&client_id) else {
                    continue;
                };
                if player.entity_id != record.entity_id {
                    continue;
                }
                frame.player_entries.push((client_id, entry));
            }
            EntityCategory::Accent | EntityCategory::Mob => {
                frame.non_player_entries.push(entry);
            }
        }
    }

    frame
        .player_entries
        .sort_unstable_by_key(|(_, entry)| entry.entity_id);
    frame
        .non_player_entries
        .sort_unstable_by_key(|entry| entry.entity_id);
    frame
}

/// Bitwise pose equality. Poses are sanitized on ingest, so identical bits
/// mean clients already hold exactly this pose; float tolerance would only
/// risk suppressing genuine drift.
fn pose_bits_eq(a: &EntityPose, b: &EntityPose) -> bool {
    fn bits(v: [f32; 4]) -> [u32; 4] {
        v.map(f32::to_bits)
    }
    bits(a.position) == bits(b.position)
        && bits(a.orientation) == bits(b.orientation)
        && bits(a.velocity) == bits(b.velocity)
        && a.scale.to_bits() == b.scale.to_bits()
}

fn world_chunk_from_position(position: [f32; 4]) -> [i32; 4] {
    let cs = CHUNK_SIZE as i32;
    [
        (position[0].floor() as i32).div_euclid(cs),
        (position[1].floor() as i32).div_euclid(cs),
        (position[2].floor() as i32).div_euclid(cs),
        (position[3].floor() as i32).div_euclid(cs),
    ]
}

fn chunk_distance2(a: [i32; 4], b: [i32; 4]) -> i64 {
    let dx = (a[0] - b[0]) as i64;
    let dy = (a[1] - b[1]) as i64;
    let dz = (a[2] - b[2]) as i64;
    let dw = (a[3] - b[3]) as i64;
    dx * dx + dy * dy + dz * dz + dw * dw
}

use crate::shared::normalize4_with_fallback as normalize4_or_default;

fn distance4_sq(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    let dw = a[3] - b[3];
    dx * dx + dy * dy + dz * dz + dw * dw
}

fn build_entity_replication_batches(
    state: &mut ServerState,
    entity_interest_radius_sq: i64,
    force_all_transforms: bool,
) -> Vec<ClientEntityReplicationBatch> {
    let mut frame = collect_live_replication_frame(state);

    // Mark entries whose pose changed since the last broadcast; unchanged
    // entities are omitted from `EntityTransforms` except on keepalive ticks.
    for entry in frame
        .player_entries
        .iter_mut()
        .map(|(_, entry)| entry)
        .chain(frame.non_player_entries.iter_mut())
    {
        match state.entity_last_broadcast_pose.entry(entry.entity_id) {
            Entry::Occupied(mut prev) => {
                if !pose_bits_eq(prev.get(), &entry.pose) {
                    prev.insert(entry.pose.clone());
                    entry.pose_changed = true;
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(entry.pose.clone());
                entry.pose_changed = true;
            }
        }
    }

    let mut player_chunk_by_client =
        HashMap::<u64, [i32; 4]>::with_capacity(frame.player_entries.len());
    for (client_id, entry) in &frame.player_entries {
        player_chunk_by_client.insert(*client_id, entry.chunk);
    }

    let connected_client_ids: HashSet<u64> = state.clients.keys().copied().collect();
    state
        .client_visible_entities
        .retain(|client_id, _| connected_client_ids.contains(client_id));
    let mut client_ids: Vec<u64> = state.clients.keys().copied().collect();
    client_ids.sort_unstable();

    let mut batches = Vec::with_capacity(client_ids.len());
    for client_id in client_ids {
        let Some(player_chunk) = player_chunk_by_client.get(&client_id).copied() else {
            state.client_visible_entities.remove(&client_id);
            continue;
        };

        let mut visible_entries: Vec<&ReplicationEntry> =
            Vec::with_capacity(frame.player_entries.len() + frame.non_player_entries.len());
        for (owner_client_id, entry) in &frame.player_entries {
            if *owner_client_id == client_id {
                continue;
            }
            if chunk_distance2(entry.chunk, player_chunk) <= entity_interest_radius_sq {
                visible_entries.push(entry);
            }
        }
        for entry in &frame.non_player_entries {
            if chunk_distance2(entry.chunk, player_chunk) <= entity_interest_radius_sq {
                visible_entries.push(entry);
            }
        }
        visible_entries.sort_unstable_by_key(|entry| entry.entity_id);
        let current_visible_ids: HashSet<u64> = visible_entries
            .iter()
            .map(|entry| entry.entity_id)
            .collect();
        let previous_visible_ids = state.client_visible_entities.entry(client_id).or_default();

        let mut despawned: Vec<u64> = previous_visible_ids
            .difference(&current_visible_ids)
            .copied()
            .collect();
        despawned.sort_unstable();

        let mut spawned_ids: Vec<u64> = current_visible_ids
            .difference(previous_visible_ids)
            .copied()
            .collect();
        spawned_ids.sort_unstable();

        *previous_visible_ids = current_visible_ids;

        let transforms: Vec<EntityTransform> = visible_entries
            .iter()
            .filter(|entry| force_all_transforms || entry.pose_changed)
            .map(|entry| EntityTransform {
                entity_id: entry.entity_id,
                pose: entry.pose.clone(),
                last_update_ms: entry.last_update_ms,
            })
            .collect();

        if spawned_ids.is_empty() && despawned.is_empty() && transforms.is_empty() {
            continue;
        }
        batches.push((client_id, spawned_ids, despawned, transforms));
    }

    // Full snapshots (entity `data`, display name) are cloned only for
    // entities newly entering a client's visible set.
    let mut result = Vec::with_capacity(batches.len());
    for (client_id, spawned_ids, despawned, transforms) in batches {
        let mut spawned = Vec::with_capacity(spawned_ids.len());
        for entity_id in &spawned_ids {
            let Some(record) = state.entity_records.get(entity_id) else {
                continue;
            };
            if let Some((snapshot, _)) = entity_snapshot_from_record(state, record) {
                spawned.push(snapshot);
            }
        }
        result.push(ClientEntityReplicationBatch {
            client_id,
            spawned,
            despawned,
            transforms,
        });
    }
    result
}

fn initialize_state(
    config: &mut RuntimeConfig,
    shutdown: Arc<AtomicBool>,
) -> io::Result<(SharedState, Instant)> {
    let start = Instant::now();
    let requested_world_seed = config.world_seed;
    let base_world_kind = config.world_generator.default_base_world_kind();
    let mut initial_world = ServerWorldOverlay::from_save_root(
        &config.world_file,
        base_world_kind,
        requested_world_seed,
        config.procgen_structures,
        HashSet::new(),
        crate::save_v4::now_unix_ms(),
        config.procgen_wasm.take(),
    )?;
    let runtime_world_seed = initial_world.world_seed();
    let next_object_id = initial_world.persisted_next_entity_id().max(1);
    initial_world.clear_dirty();
    let initial_chunks = initial_world.non_empty_chunk_count();
    eprintln!(
        "initialized runtime world (seed={}, {} non-empty chunks)",
        runtime_world_seed, initial_chunks,
    );
    eprintln!("v4 save streaming root={}", config.world_file.display());

    // Load persisted player records (for inventory restoration on reconnect).
    let persisted_players = match crate::save_v4::load_state_metadata(&config.world_file) {
        Ok(metadata) => {
            if !metadata.players.players.is_empty() {
                eprintln!(
                    "loaded {} persisted player record(s)",
                    metadata.players.players.len()
                );
            }
            metadata.players.players
        }
        Err(_) => Vec::new(),
    };

    let mob_nav_debug = env_flag_enabled("R4D_MOB_NAV_DEBUG");
    let mob_nav_simple_steer = env_flag_enabled("R4D_MOB_NAV_SIMPLE_STEER");
    if mob_nav_debug {
        eprintln!("mob nav debug logging enabled (R4D_MOB_NAV_DEBUG=1)");
    }
    if mob_nav_simple_steer {
        eprintln!("mob nav simple steering enabled (R4D_MOB_NAV_SIMPLE_STEER=1)");
    }

    let state = Arc::new(Mutex::new(ServerState::new(
        initial_world,
        next_object_id,
        mob_nav_debug,
        mob_nav_simple_steer,
        start,
        config.content_registry.clone(),
        persisted_players,
    )));

    let entity_interest_radius_chunks = config.procgen_far_chunk_radius.max(1)
        * STREAM_FAR_LOD_SCALE
        + ENTITY_INTEREST_RADIUS_PADDING_CHUNKS;
    start_broadcast_thread(
        state.clone(),
        config.tick_hz,
        config.entity_sim_hz,
        entity_interest_radius_chunks,
        config.save_interval_secs.max(1),
        start,
        shutdown.clone(),
        config.wasm_manager.take(),
    );

    Ok((state, start))
}

pub fn connect_local_client(config: &mut RuntimeConfig) -> io::Result<LocalConnection> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown.clone())?;
    let (client_to_server_tx, client_to_server_rx) = mpsc::channel::<ClientMessage>();
    let (server_to_client_tx, server_to_client_rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = allocate_server_object_id(&mut guard);
        guard.clients.insert(id, server_to_client_tx);
        id
    };

    let state_for_client = state.clone();
    let tick_hz = config.tick_hz;
    thread::spawn(move || {
        while let Ok(message) = client_to_server_rx.recv() {
            handle_message(
                &state_for_client,
                client_id,
                message,
                tick_hz.max(0.1),
                start,
            );
        }
        remove_client(&state_for_client, client_id);
        shutdown.store(true, Ordering::Relaxed);
    });

    Ok(LocalConnection {
        outgoing: client_to_server_tx,
        incoming: server_to_client_rx,
    })
}

pub fn run_tcp_server(config: &mut RuntimeConfig) -> io::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown)?;
    let runtime_world_seed = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.world_seed()
    };

    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, entity_sim {:.2} Hz, save_io=v4-streaming, save_interval={}s, procgen={}, seed={}, near_radius={} chunks, mid_radius={} chunks, far_radius={} chunks, keepout={}, keepout_padding={} chunks)",
        config.bind,
        config.tick_hz.max(0.1),
        config.entity_sim_hz.max(0.1),
        config.save_interval_secs.max(1),
        config.procgen_structures,
        runtime_world_seed,
        config.procgen_near_chunk_radius.max(0),
        config.procgen_mid_chunk_radius.max(1),
        config.procgen_far_chunk_radius.max(1),
        config.procgen_keepout_from_existing_world,
        config.procgen_keepout_padding_chunks.max(0),
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                spawn_client_thread(stream, state.clone(), config.tick_hz.max(0.1), start);
            }
            Err(error) => {
                eprintln!("accept failed: {}", error);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod replication_tests {
    use super::core_state::{remove_entity_record, upsert_entity_record};
    use super::types::PlayerState;
    use super::*;

    const BIG_INTEREST_RADIUS_SQ: i64 = 1_000_000;

    fn test_state() -> ServerState {
        let world = ServerWorldOverlay::from_chunk_payloads(
            crate::shared::voxel::BaseWorldKind::Empty,
            Vec::<([i32; 4], crate::shared::chunk_payload::ResolvedChunkPayload)>::new(),
            0,
            false,
            HashSet::new(),
        );
        let (registry, _pending) = crate::plugin_loader::create_full_registry();
        ServerState::new(
            world,
            1,
            false,
            false,
            Instant::now(),
            Arc::new(registry),
            Vec::new(),
        )
    }

    /// Returns the receiver so the client channel stays open for the test.
    fn add_player(
        state: &mut ServerState,
        client_id: u64,
        position: [f32; 4],
    ) -> mpsc::Receiver<ServerMessage> {
        let (tx, rx) = mpsc::channel();
        state.clients.insert(client_id, tx);
        let mut entity = Entity::simple(0, 0);
        entity.pose.position = position;
        state.entity_store.spawn(client_id, entity, 0);
        state.players.insert(
            client_id,
            PlayerState {
                entity_id: client_id,
                inventory_payload: Vec::new(),
            },
        );
        upsert_entity_record(
            state,
            client_id,
            EntityCategory::Player,
            Some(client_id),
            Some("player".to_string()),
            false,
            0,
        );
        rx
    }

    fn add_accent(state: &mut ServerState, entity_id: u64, position: [f32; 4]) {
        let mut entity = Entity::simple(7, 42);
        entity.pose.position = position;
        state.entity_store.spawn(entity_id, entity, 0);
        upsert_entity_record(state, entity_id, EntityCategory::Accent, None, None, false, 0);
    }

    #[test]
    fn unmoved_entities_are_gated_and_keepalive_resends() {
        let mut state = test_state();
        let _rx = add_player(&mut state, 1, [0.0; 4]);
        add_accent(&mut state, 100, [2.0, 0.0, 0.0, 0.0]);

        // First tick: newly visible -> full snapshot + transform.
        let batches = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, false);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].client_id, 1);
        assert_eq!(
            batches[0]
                .spawned
                .iter()
                .map(|s| s.entity_id)
                .collect::<Vec<_>>(),
            vec![100]
        );
        assert!(batches[0].transforms.iter().any(|t| t.entity_id == 100));

        // Second tick, nothing moved: no batch at all.
        let batches = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, false);
        assert!(batches.is_empty(), "unmoved entities should be gated");

        // Keepalive tick resends all visible poses.
        let batches = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, true);
        assert_eq!(batches.len(), 1);
        assert!(batches[0].spawned.is_empty());
        assert!(batches[0].transforms.iter().any(|t| t.entity_id == 100));

        // Moving the entity produces a transform on a normal tick again.
        state
            .entity_store
            .get_mut(100)
            .expect("accent exists")
            .entity
            .pose
            .position[0] = 5.0;
        let batches = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, false);
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0]
                .transforms
                .iter()
                .map(|t| t.entity_id)
                .collect::<Vec<_>>(),
            vec![100]
        );
    }

    #[test]
    fn despawn_emits_destroy_and_leaves_no_tombstone_state() {
        let mut state = test_state();
        let _rx = add_player(&mut state, 1, [0.0; 4]);
        add_accent(&mut state, 100, [2.0, 0.0, 0.0, 0.0]);

        let _ = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, false);

        state.entity_store.despawn(100);
        remove_entity_record(&mut state, 100);

        let batches = build_entity_replication_batches(&mut state, BIG_INTEREST_RADIUS_SQ, false);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].despawned, vec![100]);
        assert!(batches[0].spawned.is_empty());

        assert!(!state.entity_records.contains_key(&100));
        assert!(!state.entity_last_broadcast_pose.contains_key(&100));
    }

    #[test]
    fn interest_radius_drives_spawn_despawn_cycle() {
        let mut state = test_state();
        let _rx = add_player(&mut state, 1, [0.0; 4]);
        // ~50 chunks away on X (chunk edge 8): outside a radius^2 of 100.
        add_accent(&mut state, 100, [400.0, 0.0, 0.0, 0.0]);

        let batches = build_entity_replication_batches(&mut state, 100, false);
        assert!(batches.is_empty(), "far entity should not replicate");

        // Move it next to the player: spawned.
        state
            .entity_store
            .get_mut(100)
            .expect("accent exists")
            .entity
            .pose
            .position = [2.0, 0.0, 0.0, 0.0];
        let batches = build_entity_replication_batches(&mut state, 100, false);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].spawned.len(), 1);

        // Move it far away again: destroyed.
        state
            .entity_store
            .get_mut(100)
            .expect("accent exists")
            .entity
            .pose
            .position = [400.0, 0.0, 0.0, 0.0];
        let batches = build_entity_replication_batches(&mut state, 100, false);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].despawned, vec![100]);
    }
}
