mod config;
mod core_state;
mod cpu_profile;
mod entities;
mod mob_sim;
mod procgen;
mod runtime_net;
mod spawn_logic;
mod types;
pub mod world_field;

pub use self::config::{LocalConnection, RuntimeConfig};
use self::core_state::{
    allocate_or_reserve_server_object_id, allocate_server_object_id, mark_entity_record_despawned,
    monotonic_ms, record_server_cpu_sample, ServerState, SharedState, upsert_entity_record,
};
use self::cpu_profile::ServerCpuProfile;
use self::entities::{EntityId, EntityStore};
use self::mob_sim::tick_entity_simulation_window;
use self::runtime_net::{
    handle_message, remove_client, spawn_client_thread, start_broadcast_thread,
};
use self::spawn_logic::{
    default_spawn_pose_for_client, mob_archetype_defaults, parse_spawn_material_id,
    parse_spawn_vec4, phase_spider_next_phase_deadline, sanitize_player_name,
    spawn_usage_string, spawnable_entity_spec_for_kind, spawnable_entity_spec_for_token,
    env_flag_enabled,
};
use self::types::{
    ClientEntityReplicationBatch, CollisionChunkCacheEntry, EntityLifecycle, EntityRecord,
    EntityRecordSummary, LiveReplicationFrame, MobArchetype, MobNavCell, MobNavPathResult,
    MobNavigationState, MobState, PersistedMobEntry, PlayerState, QueuedExplosionEvent,
    QueuedPlayerMovementModifier, SpawnableEntitySpec,
    SPAWNABLE_ENTITY_SPECS,
};
use self::world_field::{QueryDetail, QueryVolume, ServerWorldOverlay, WorldField};
use crate::materials;
use crate::shared::protocol::{
    ClientMessage, EntityClass, EntityKind, EntitySnapshot, EntityTransform, ServerMessage,
    WorldSummary,
};
use crate::shared::spatial::Aabb4i;
use crate::shared::voxel::{self, BaseWorldKind, ChunkPos, VoxelType, CHUNK_SIZE};
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
const ENTITY_SIM_STEP_MAX_PER_BROADCAST: usize = 16;
const CREEPER_EXPLOSION_TRIGGER_DISTANCE: f32 = 1.55;
const CREEPER_EXPLOSION_RADIUS_VOXELS: i32 = 3;
const CREEPER_EXPLOSION_IMPULSE_RADIUS: f32 = 7.0;
const CREEPER_EXPLOSION_MAX_IMPULSE_DISTANCE: f32 = 5.0;
const CREEPER_POUNCE_TARGET_BELOW_PLAYER_Y: f32 = 1.05;
const MOB_HARD_WORLD_FLOOR_Y: f32 = -4.0;
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
const PHASE_SPIDER_PHASE_MIN_INTERVAL_MS: u64 = 720;
const PHASE_SPIDER_PHASE_MAX_INTERVAL_MS: u64 = 1520;
const PHASE_SPIDER_PHASE_DISTANCE: f32 = 2.8;
const PHASE_SPIDER_PHASE_MIN_DISTANCE: f32 = 1.0;
const PHASE_SPIDER_BLOCKED_PROGRESS_EPSILON: f32 = 0.08;

fn entity_snapshot_from_record(
    state: &ServerState,
    record: &EntityRecord,
) -> Option<(EntitySnapshot, [i32; 4])> {
    let mut snapshot = state.entity_store.snapshot(record.entity_id)?;
    snapshot.class = record.class;
    snapshot.owner_client_id = record.owner_client_id;
    snapshot.display_name = record.display_name.clone();
    let chunk = world_chunk_from_position(snapshot.position);
    Some((snapshot, chunk))
}

fn entity_transform_from_snapshot(snapshot: &EntitySnapshot) -> EntityTransform {
    EntityTransform {
        entity_id: snapshot.entity_id,
        position: snapshot.position,
        orientation: snapshot.orientation,
        velocity: snapshot.velocity,
        scale: snapshot.scale,
        material: snapshot.material,
        last_update_ms: snapshot.last_update_ms,
    }
}

fn collect_live_replication_frame(state: &ServerState) -> LiveReplicationFrame {
    let mut live_records: Vec<&EntityRecord> = state
        .entity_records
        .values()
        .filter(|record| record.lifecycle == EntityLifecycle::Live)
        .collect();
    live_records.sort_unstable_by_key(|record| record.entity_id);

    let mut frame = LiveReplicationFrame::default();
    for record in live_records {
        match record.class {
            EntityClass::Player => {
                let Some(client_id) = record.owner_client_id else {
                    continue;
                };
                let Some(player) = state.players.get(&client_id) else {
                    continue;
                };
                if player.entity_id != record.entity_id {
                    continue;
                }
                let Some((snapshot, chunk)) = entity_snapshot_from_record(state, record) else {
                    continue;
                };
                frame.player_entities.push(snapshot);
                frame.player_chunks.push((client_id, chunk));
            }
            EntityClass::Accent | EntityClass::Mob => {
                if let Some((snapshot, chunk)) = entity_snapshot_from_record(state, record) {
                    frame.non_player_entities.push((snapshot, chunk));
                }
            }
        }
    }

    frame
        .player_entities
        .sort_unstable_by_key(|snapshot| snapshot.entity_id);
    frame
        .player_chunks
        .sort_unstable_by_key(|(client_id, _)| *client_id);
    frame
        .non_player_entities
        .sort_unstable_by_key(|(snapshot, _)| snapshot.entity_id);
    frame
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

fn normalize4_or_default(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 || !len_sq.is_finite() {
        return fallback;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

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
) -> Vec<ClientEntityReplicationBatch> {
    let frame = collect_live_replication_frame(state);
    let mut player_chunk_by_client =
        HashMap::<u64, [i32; 4]>::with_capacity(frame.player_chunks.len());
    for (client_id, chunk) in frame.player_chunks {
        player_chunk_by_client.insert(client_id, chunk);
    }

    let mut player_entities_with_chunks = Vec::with_capacity(frame.player_entities.len());
    for entity in frame.player_entities {
        let Some(owner_client_id) = entity.owner_client_id else {
            continue;
        };
        let Some(owner_chunk) = player_chunk_by_client.get(&owner_client_id).copied() else {
            continue;
        };
        player_entities_with_chunks.push((entity, owner_chunk));
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

        let mut visible_entities =
            Vec::with_capacity(player_entities_with_chunks.len() + frame.non_player_entities.len());
        for (entity, owner_chunk) in &player_entities_with_chunks {
            if entity.owner_client_id == Some(client_id) {
                continue;
            }
            if chunk_distance2(*owner_chunk, player_chunk) <= entity_interest_radius_sq {
                visible_entities.push(entity.clone());
            }
        }
        for (entity, entity_chunk) in &frame.non_player_entities {
            if chunk_distance2(*entity_chunk, player_chunk) <= entity_interest_radius_sq {
                visible_entities.push(entity.clone());
            }
        }
        visible_entities.sort_unstable_by_key(|entity| entity.entity_id);
        let current_visible_ids: HashSet<u64> = visible_entities
            .iter()
            .map(|entity| entity.entity_id)
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

        let transforms: Vec<EntityTransform> = visible_entities
            .iter()
            .map(entity_transform_from_snapshot)
            .collect();
        let snapshot_by_id: HashMap<u64, EntitySnapshot> = visible_entities
            .into_iter()
            .map(|entity| (entity.entity_id, entity))
            .collect();
        let mut spawned = Vec::with_capacity(spawned_ids.len());
        for entity_id in spawned_ids {
            if let Some(entity) = snapshot_by_id.get(&entity_id) {
                spawned.push(entity.clone());
            }
        }

        if spawned.is_empty() && despawned.is_empty() && transforms.is_empty() {
            continue;
        }
        batches.push(ClientEntityReplicationBatch {
            client_id,
            spawned,
            despawned,
            transforms,
        });
    }
    batches
}

fn initialize_state(
    config: &RuntimeConfig,
    shutdown: Arc<AtomicBool>,
) -> io::Result<(SharedState, Instant)> {
    let start = Instant::now();
    procgen::clear_runtime_maze_layout_cache();
    let runtime_world_seed = config.world_seed;
    let base_world_kind = BaseWorldKind::FlatFloor {
        material: VoxelType(11),
    };
    let next_object_id = 1;
    let mut initial_world = ServerWorldOverlay::from_chunk_payloads(
        base_world_kind,
        Vec::new(),
        runtime_world_seed,
        config.procgen_structures,
        HashSet::new(),
    );
    let pruned = initial_world.prune_virgin_chunks();
    if pruned > 0 {
        eprintln!(
            "pruned {} virgin chunks from {}",
            pruned,
            config.world_file.display()
        );
    }
    if config.procgen_structures && config.procgen_keepout_from_existing_world {
        let blocked = initial_world
            .rebuild_procgen_keepout_from_chunks(config.procgen_keepout_padding_chunks);
        if blocked > 0 {
            eprintln!(
                "procgen keepout: {} blocked structure cells (padding={} chunks)",
                blocked,
                config.procgen_keepout_padding_chunks.max(0)
            );
        }
    } else {
        initial_world.set_procgen_blocked_cells(HashSet::new());
    }
    initial_world.clear_dirty();
    let initial_chunks = initial_world.non_empty_chunk_count();
    eprintln!(
        "initialized runtime world (seed={}, {} non-empty chunks)",
        runtime_world_seed,
        initial_chunks,
    );
    eprintln!(
        "server persistence is disabled in this rebuild path (world_file={} ignored at runtime)",
        config.world_file.display(),
    );
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
    )));

    let entity_interest_radius_chunks = config.procgen_far_chunk_radius.max(1)
        * STREAM_FAR_LOD_SCALE
        + ENTITY_INTEREST_RADIUS_PADDING_CHUNKS;
    start_broadcast_thread(
        state.clone(),
        config.tick_hz,
        config.entity_sim_hz,
        entity_interest_radius_chunks,
        start,
        shutdown.clone(),
    );

    Ok((state, start))
}

pub fn connect_local_client(config: &RuntimeConfig) -> io::Result<LocalConnection> {
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
    let cfg = config.clone();
    thread::spawn(move || {
        while let Ok(message) = client_to_server_rx.recv() {
            handle_message(
                &state_for_client,
                client_id,
                message,
                cfg.tick_hz.max(0.1),
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

pub fn run_tcp_server(config: &RuntimeConfig) -> io::Result<()> {
    let shutdown = Arc::new(AtomicBool::new(false));
    let (state, start) = initialize_state(config, shutdown)?;
    let runtime_world_seed = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.world_seed()
    };

    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, entity_sim {:.2} Hz, save_io=disabled, procgen={}, seed={}, near_radius={} chunks, mid_radius={} chunks, far_radius={} chunks, keepout={}, keepout_padding={} chunks)",
        config.bind,
        config.tick_hz.max(0.1),
        config.entity_sim_hz.max(0.1),
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
                spawn_client_thread(
                    stream,
                    state.clone(),
                    config.tick_hz.max(0.1),
                    start,
                );
            }
            Err(error) => {
                eprintln!("accept failed: {}", error);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Intentionally empty for now.
}
