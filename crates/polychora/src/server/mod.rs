mod entities;
mod procgen;
pub mod region_tree_cache;
pub mod world_field;

use self::entities::{EntityId, EntityStore};
use self::region_tree_cache::RegionTreeWorkingSet;
use self::world_field::ServerWorldField;
use crate::materials;
use crate::save_v3::{self, PersistedEntityRecord, PlayerRecord, SaveRequest};
use crate::shared::protocol::{
    ClientMessage, EntityClass, EntityKind, EntitySnapshot, EntityTransform, RegionClockPayload,
    ServerMessage, WorldChunkCoordPayload, WorldChunkPayload, WorldSnapshotPayload, WorldSummary,
    WORLD_CHUNK_LOD_NEAR,
};
use crate::shared::voxel::{self, BaseWorldKind, ChunkPos, VoxelType, CHUNK_SIZE};
use crate::shared::worldfield::{
    Aabb4i, ChunkPayload, QueryDetail, QueryVolume, RegionClockMap, RegionClockPrecondition,
    RegionClockUpdate, RegionId, RegionResyncEntry, RegionResyncRequest, RegionSubtreePatch,
    WorldField, MAX_PATCH_BYTES, REGION_TILE_EDGE_CHUNKS,
};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{hash_map::Entry, BinaryHeap, HashMap, HashSet};
use std::io::{self, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

const STREAM_FAR_LOD_SCALE: i32 = 4;
const STREAM_MESSAGE_CHUNK_LIMIT: usize = 256;
const SERVER_CPU_PROFILE_INTERVAL: Duration = Duration::from_secs(2);
const ENTITY_INTEREST_RADIUS_PADDING_CHUNKS: i32 = 2;
const ENTITY_SIM_STEP_MAX_PER_BROADCAST: usize = 16;
const PLAYER_PERSIST_INTERVAL_MS: u64 = 60_000;
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

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub bind: String,
    pub world_file: PathBuf,
    pub tick_hz: f32,
    pub entity_sim_hz: f32,
    pub save_interval_secs: u64,
    pub snapshot_on_join: bool,
    pub procgen_structures: bool,
    pub procgen_near_chunk_radius: i32,
    pub procgen_mid_chunk_radius: i32,
    pub procgen_far_chunk_radius: i32,
    pub procgen_keepout_from_existing_world: bool,
    pub procgen_keepout_padding_chunks: i32,
    pub world_seed: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:4000".to_string(),
            world_file: PathBuf::from("saves/world"),
            tick_hz: 10.0,
            entity_sim_hz: 30.0,
            save_interval_secs: 5,
            snapshot_on_join: true,
            procgen_structures: true,
            procgen_near_chunk_radius: 6,
            procgen_mid_chunk_radius: 10,
            procgen_far_chunk_radius: 6,
            procgen_keepout_from_existing_world: true,
            procgen_keepout_padding_chunks: 1,
            world_seed: 1337,
        }
    }
}

pub struct LocalConnection {
    pub outgoing: mpsc::Sender<ClientMessage>,
    pub incoming: mpsc::Receiver<ServerMessage>,
}

#[derive(Clone, Debug)]
struct PlayerState {
    entity_id: u64,
    working_set: RegionTreeWorkingSet,
    last_stream_center_chunk: Option<[i32; 4]>,
    last_stream_region_clocks: RegionClockMap,
    known_region_clocks: RegionClockMap,
    next_stream_patch_seq: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum MobArchetype {
    Seeker,
    Creeper4d,
    PhaseSpider,
}

type MobNavCell = [i32; 4];

#[derive(Clone, Debug, Default)]
struct MobNavigationState {
    goal_cell: Option<MobNavCell>,
    path_cells: Vec<MobNavCell>,
    path_cursor: usize,
    last_repath_ms: u64,
    last_debug_log_ms: u64,
}

#[derive(Clone, Debug)]
struct MobNavPathResult {
    path_cells: Vec<MobNavCell>,
    reached_goal: bool,
    expanded_steps: usize,
    best_cell: MobNavCell,
    best_goal_distance: i32,
}

#[derive(Clone, Debug)]
struct MobState {
    entity_id: u64,
    archetype: MobArchetype,
    phase_offset: f32,
    move_speed: f32,
    preferred_distance: f32,
    tangent_weight: f32,
    next_phase_ms: u64,
    navigation: MobNavigationState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedMobEntry {
    archetype: MobArchetype,
    phase_offset: f32,
    move_speed: f32,
    preferred_distance: f32,
    tangent_weight: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EntityLifecycle {
    Live,
    Despawned,
}

#[derive(Clone, Debug)]
struct EntityRecord {
    entity_id: u64,
    class: EntityClass,
    owner_client_id: Option<u64>,
    display_name: Option<String>,
    persistent: bool,
    spawned_at_ms: u64,
    lifecycle: EntityLifecycle,
    despawned_at_ms: Option<u64>,
}

#[derive(Copy, Clone, Debug, Default)]
struct EntityRecordSummary {
    live_total: usize,
    live_players: usize,
    live_accents: usize,
    live_mobs: usize,
    live_persistent: usize,
    live_owned: usize,
    tombstones: usize,
}

#[derive(Clone, Debug, Default)]
struct LiveReplicationFrame {
    player_entities: Vec<EntitySnapshot>,
    player_chunks: Vec<(u64, [i32; 4])>,
    non_player_entities: Vec<(EntitySnapshot, [i32; 4])>,
}

#[derive(Clone, Debug, Default)]
struct ClientEntityReplicationBatch {
    client_id: u64,
    spawned: Vec<EntitySnapshot>,
    despawned: Vec<u64>,
    transforms: Vec<EntityTransform>,
}

#[derive(Clone, Debug)]
struct QueuedWorldVoxelSet {
    position: [i32; 4],
    material: u8,
    source_client_id: Option<u64>,
    client_edit_id: Option<u64>,
    revision: u64,
    clock_updates: Option<Vec<RegionClockPayload>>,
}

#[derive(Clone, Debug)]
struct QueuedExplosionEvent {
    position: [f32; 4],
    radius: f32,
    source_entity_id: Option<u64>,
}

#[derive(Clone, Debug)]
struct QueuedPlayerMovementModifier {
    client_id: u64,
    delta_position: [f32; 4],
    delta_velocity_y: f32,
    source_entity_id: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct SpawnableEntitySpec {
    kind: EntityKind,
    class: EntityClass,
    mob_archetype: Option<MobArchetype>,
    canonical_name: &'static str,
    aliases: &'static [&'static str],
    default_scale: f32,
    default_material: u8,
}

const SPAWNABLE_ENTITY_SPECS: &[SpawnableEntitySpec] = &[
    SpawnableEntitySpec {
        kind: EntityKind::TestCube,
        class: EntityClass::Accent,
        mob_archetype: None,
        canonical_name: "cube",
        aliases: &["cube", "testcube"],
        default_scale: 0.50,
        default_material: 12,
    },
    SpawnableEntitySpec {
        kind: EntityKind::TestRotor,
        class: EntityClass::Accent,
        mob_archetype: None,
        canonical_name: "rotor",
        aliases: &["rotor", "testrotor"],
        default_scale: 0.54,
        default_material: 8,
    },
    SpawnableEntitySpec {
        kind: EntityKind::TestDrifter,
        class: EntityClass::Accent,
        mob_archetype: None,
        canonical_name: "drifter",
        aliases: &["drifter", "testdrifter"],
        default_scale: 0.48,
        default_material: 15,
    },
    SpawnableEntitySpec {
        kind: EntityKind::MobSeeker,
        class: EntityClass::Mob,
        mob_archetype: Some(MobArchetype::Seeker),
        canonical_name: "seeker",
        aliases: &["seeker", "mobseeker"],
        default_scale: 0.62,
        default_material: 24,
    },
    SpawnableEntitySpec {
        kind: EntityKind::MobCreeper4d,
        class: EntityClass::Mob,
        mob_archetype: Some(MobArchetype::Creeper4d),
        canonical_name: "creeper",
        aliases: &["creeper", "4dcreeper", "mobcreeper4d"],
        default_scale: 0.78,
        default_material: 19,
    },
    SpawnableEntitySpec {
        kind: EntityKind::MobPhaseSpider,
        class: EntityClass::Mob,
        mob_archetype: Some(MobArchetype::PhaseSpider),
        canonical_name: "phase_spider",
        aliases: &[
            "phase_spider",
            "phasespider",
            "phase-spider",
            "spider",
            "mobphasespider",
        ],
        default_scale: 0.86,
        default_material: 62,
    },
];

#[derive(Clone, Debug)]
struct ServerCpuProfile {
    window_start: Instant,
    message_samples: u64,
    message_cpu_ms_sum: f64,
    message_cpu_ms_max: f64,
    tick_samples: u64,
    tick_cpu_ms_sum: f64,
    tick_cpu_ms_max: f64,
    tick_player_snapshots_sum: u64,
    tick_player_snapshots_max: u64,
    tick_entity_snapshots_sum: u64,
    tick_entity_snapshots_max: u64,
}

#[derive(Copy, Clone, Debug)]
struct ServerCpuProfileReport {
    message_samples: u64,
    message_cpu_ms_sum: f64,
    message_cpu_ms_max: f64,
    tick_samples: u64,
    tick_cpu_ms_sum: f64,
    tick_cpu_ms_max: f64,
    tick_player_snapshots_sum: u64,
    tick_player_snapshots_max: u64,
    tick_entity_snapshots_sum: u64,
    tick_entity_snapshots_max: u64,
}

impl ServerCpuProfile {
    fn new(now: Instant) -> Self {
        Self {
            window_start: now,
            message_samples: 0,
            message_cpu_ms_sum: 0.0,
            message_cpu_ms_max: 0.0,
            tick_samples: 0,
            tick_cpu_ms_sum: 0.0,
            tick_cpu_ms_max: 0.0,
            tick_player_snapshots_sum: 0,
            tick_player_snapshots_max: 0,
            tick_entity_snapshots_sum: 0,
            tick_entity_snapshots_max: 0,
        }
    }

    fn record_message_sample(&mut self, elapsed: Duration) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.message_samples = self.message_samples.saturating_add(1);
        self.message_cpu_ms_sum += elapsed_ms;
        self.message_cpu_ms_max = self.message_cpu_ms_max.max(elapsed_ms);
    }

    fn record_tick_sample(
        &mut self,
        elapsed: Duration,
        player_snapshots: usize,
        entity_snapshots: usize,
    ) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.tick_samples = self.tick_samples.saturating_add(1);
        self.tick_cpu_ms_sum += elapsed_ms;
        self.tick_cpu_ms_max = self.tick_cpu_ms_max.max(elapsed_ms);
        let player_snapshots = player_snapshots as u64;
        let entity_snapshots = entity_snapshots as u64;
        self.tick_player_snapshots_sum = self
            .tick_player_snapshots_sum
            .saturating_add(player_snapshots);
        self.tick_player_snapshots_max = self.tick_player_snapshots_max.max(player_snapshots);
        self.tick_entity_snapshots_sum = self
            .tick_entity_snapshots_sum
            .saturating_add(entity_snapshots);
        self.tick_entity_snapshots_max = self.tick_entity_snapshots_max.max(entity_snapshots);
    }

    fn take_report_if_due(&mut self, now: Instant) -> Option<ServerCpuProfileReport> {
        if now.duration_since(self.window_start) < SERVER_CPU_PROFILE_INTERVAL {
            return None;
        }

        let report = ServerCpuProfileReport {
            message_samples: self.message_samples,
            message_cpu_ms_sum: self.message_cpu_ms_sum,
            message_cpu_ms_max: self.message_cpu_ms_max,
            tick_samples: self.tick_samples,
            tick_cpu_ms_sum: self.tick_cpu_ms_sum,
            tick_cpu_ms_max: self.tick_cpu_ms_max,
            tick_player_snapshots_sum: self.tick_player_snapshots_sum,
            tick_player_snapshots_max: self.tick_player_snapshots_max,
            tick_entity_snapshots_sum: self.tick_entity_snapshots_sum,
            tick_entity_snapshots_max: self.tick_entity_snapshots_max,
        };

        self.window_start = now;
        self.message_samples = 0;
        self.message_cpu_ms_sum = 0.0;
        self.message_cpu_ms_max = 0.0;
        self.tick_samples = 0;
        self.tick_cpu_ms_sum = 0.0;
        self.tick_cpu_ms_max = 0.0;
        self.tick_player_snapshots_sum = 0;
        self.tick_player_snapshots_max = 0;
        self.tick_entity_snapshots_sum = 0;
        self.tick_entity_snapshots_max = 0;

        if report.message_samples == 0 && report.tick_samples == 0 {
            None
        } else {
            Some(report)
        }
    }
}

#[derive(Debug)]
struct ServerState {
    next_object_id: u64,
    entity_store: EntityStore,
    entity_records: HashMap<u64, EntityRecord>,
    entities_dirty: bool,
    entity_revision: u64,
    dirty_block_regions: HashSet<[i32; 4]>,
    region_chunk_edge: i32,
    world: ServerWorldField,
    world_revision: u64,
    region_clocks: RegionClockMap,
    players: HashMap<u64, PlayerState>,
    last_players_persisted_ms: u64,
    mobs: HashMap<u64, MobState>,
    mob_nav_debug: bool,
    mob_nav_simple_steer: bool,
    clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
    client_visible_entities: HashMap<u64, HashSet<u64>>,
    cpu_profile: ServerCpuProfile,
}

type SharedState = Arc<Mutex<ServerState>>;

fn monotonic_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

fn allocate_server_object_id(state: &mut ServerState) -> u64 {
    let id = state.next_object_id;
    state.next_object_id = state.next_object_id.wrapping_add(1).max(1);
    id
}

fn allocate_or_reserve_server_object_id(
    state: &mut ServerState,
    persisted_id: Option<EntityId>,
) -> EntityId {
    match persisted_id {
        Some(entity_id) if entity_id > 0 => {
            state.next_object_id = state.next_object_id.max(entity_id.saturating_add(1).max(1));
            entity_id
        }
        _ => allocate_server_object_id(state),
    }
}

fn mark_entities_dirty(state: &mut ServerState) {
    state.entities_dirty = true;
    state.entity_revision = state.entity_revision.wrapping_add(1);
}

fn mark_block_region_dirty(state: &mut ServerState, chunk_pos: ChunkPos) {
    let region = save_v3::region_from_chunk_pos(chunk_pos, state.region_chunk_edge.max(1));
    state.dirty_block_regions.insert(region);
}

fn advance_world_revision(state: &mut ServerState) -> u64 {
    state.world_revision = state.world_revision.wrapping_add(1);
    state.world_revision
}

fn bump_region_clocks_for_chunks<I>(state: &mut ServerState, chunks: I) -> Vec<RegionClockPayload>
where
    I: IntoIterator<Item = ChunkPos>,
{
    let mut touched_regions = HashSet::<RegionId>::new();
    for chunk_pos in chunks {
        touched_regions.insert(RegionId::from_chunk_pos(chunk_pos));
    }
    let mut updates = Vec::with_capacity(touched_regions.len());
    for region_id in touched_regions {
        let clock = state.region_clocks.entry(region_id).or_insert(0);
        *clock = clock.wrapping_add(1);
        updates.push(RegionClockPayload {
            region_id: [region_id.x, region_id.y, region_id.z, region_id.w],
            clock: *clock,
        });
    }
    updates.sort_unstable_by_key(|update| {
        (
            update.region_id[3],
            update.region_id[2],
            update.region_id[1],
            update.region_id[0],
        )
    });
    updates
}

fn upsert_entity_record(
    state: &mut ServerState,
    entity_id: u64,
    class: EntityClass,
    owner_client_id: Option<u64>,
    display_name: Option<String>,
    persistent: bool,
    now_ms: u64,
) {
    let display_name_for_insert = display_name.clone();
    let record = state
        .entity_records
        .entry(entity_id)
        .or_insert(EntityRecord {
            entity_id,
            class,
            owner_client_id,
            display_name: display_name_for_insert,
            persistent,
            spawned_at_ms: now_ms,
            lifecycle: EntityLifecycle::Live,
            despawned_at_ms: None,
        });
    record.class = class;
    record.owner_client_id = owner_client_id;
    record.persistent = persistent;
    if display_name.is_some() || class != EntityClass::Player {
        record.display_name = display_name;
    }
    if record.lifecycle == EntityLifecycle::Despawned {
        record.spawned_at_ms = now_ms;
        record.lifecycle = EntityLifecycle::Live;
        record.despawned_at_ms = None;
    }
}

fn mark_entity_record_despawned(state: &mut ServerState, entity_id: u64, now_ms: Option<u64>) {
    let (was_live, was_persistent) = {
        let Some(record) = state.entity_records.get_mut(&entity_id) else {
            return;
        };
        let was_live = record.lifecycle == EntityLifecycle::Live;
        let was_persistent = record.persistent;
        record.lifecycle = EntityLifecycle::Despawned;
        if record.despawned_at_ms.is_none() {
            record.despawned_at_ms = now_ms;
        }
        (was_live, was_persistent)
    };
    if was_live && was_persistent {
        mark_entities_dirty(state);
    }
}

fn summarize_entity_records(state: &ServerState) -> EntityRecordSummary {
    let mut summary = EntityRecordSummary::default();
    for (&record_id, record) in &state.entity_records {
        debug_assert_eq!(record.entity_id, record_id);
        match record.lifecycle {
            EntityLifecycle::Live => {
                summary.live_total = summary.live_total.saturating_add(1);
                if record.persistent {
                    summary.live_persistent = summary.live_persistent.saturating_add(1);
                }
                if record.owner_client_id.is_some() {
                    summary.live_owned = summary.live_owned.saturating_add(1);
                }
                match record.class {
                    EntityClass::Player => {
                        summary.live_players = summary.live_players.saturating_add(1)
                    }
                    EntityClass::Accent => {
                        summary.live_accents = summary.live_accents.saturating_add(1)
                    }
                    EntityClass::Mob => summary.live_mobs = summary.live_mobs.saturating_add(1),
                }
            }
            EntityLifecycle::Despawned => {
                summary.tombstones = summary.tombstones.saturating_add(1);
            }
        }
    }
    summary
}

fn record_server_cpu_sample(
    state: &SharedState,
    message_elapsed: Option<Duration>,
    tick_sample: Option<(Duration, usize, usize)>,
) {
    let maybe_report = {
        let mut guard = state.lock().expect("server state lock poisoned");
        if let Some(elapsed) = message_elapsed {
            guard.cpu_profile.record_message_sample(elapsed);
        }
        if let Some((elapsed, player_snapshots, entity_snapshots)) = tick_sample {
            guard
                .cpu_profile
                .record_tick_sample(elapsed, player_snapshots, entity_snapshots);
        }
        guard
            .cpu_profile
            .take_report_if_due(Instant::now())
            .map(|report| {
                (
                    report,
                    guard.players.len(),
                    guard.entity_store.len(),
                    summarize_entity_records(&guard),
                )
            })
    };

    let Some((report, player_count, entity_count, entity_records)) = maybe_report else {
        return;
    };

    let msg_avg_ms = if report.message_samples > 0 {
        report.message_cpu_ms_sum / report.message_samples as f64
    } else {
        0.0
    };
    let tick_avg_ms = if report.tick_samples > 0 {
        report.tick_cpu_ms_sum / report.tick_samples as f64
    } else {
        0.0
    };
    let tick_players_avg = if report.tick_samples > 0 {
        report.tick_player_snapshots_sum as f64 / report.tick_samples as f64
    } else {
        0.0
    };
    let tick_entities_avg = if report.tick_samples > 0 {
        report.tick_entity_snapshots_sum as f64 / report.tick_samples as f64
    } else {
        0.0
    };

    eprintln!(
        "profile server-cpu msg_avg={:.3}ms msg_max={:.3}ms msg_samples={} tick_avg={:.3}ms tick_max={:.3}ms tick_samples={} tick_players_avg={:.1} tick_players_max={} tick_entities_avg={:.1} tick_entities_max={} players={} entities={} rec_live={} rec_players={} rec_accents={} rec_mobs={} rec_persistent={} rec_owned={} rec_tombstones={}",
        msg_avg_ms,
        report.message_cpu_ms_max,
        report.message_samples,
        tick_avg_ms,
        report.tick_cpu_ms_max,
        report.tick_samples,
        tick_players_avg,
        report.tick_player_snapshots_max,
        tick_entities_avg,
        report.tick_entity_snapshots_max,
        player_count,
        entity_count,
        entity_records.live_total,
        entity_records.live_players,
        entity_records.live_accents,
        entity_records.live_mobs,
        entity_records.live_persistent,
        entity_records.live_owned,
        entity_records.tombstones,
    );
}

fn sanitize_player_name(name: &str, client_id: u64) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("player-{client_id}");
    }
    trimmed.chars().take(32).collect()
}

fn normalize_spawn_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn parse_spawn_material_id(token: &str) -> Option<u8> {
    if let Ok(id) = token.parse::<u8>() {
        if (1..=materials::MAX_MATERIAL_ID).contains(&id) {
            return Some(id);
        }
    }
    let normalized = normalize_spawn_token(token);
    materials::MATERIALS.iter().find_map(|material| {
        (normalize_spawn_token(material.name) == normalized).then_some(material.id)
    })
}

fn parse_spawn_vec4(args: &[&str]) -> Option<[f32; 4]> {
    if args.len() != 4 {
        return None;
    }
    let x = args[0].parse::<f32>().ok()?;
    let y = args[1].parse::<f32>().ok()?;
    let z = args[2].parse::<f32>().ok()?;
    let w = args[3].parse::<f32>().ok()?;
    Some([x, y, z, w])
}

fn spawn_usage_string() -> String {
    let names = SPAWNABLE_ENTITY_SPECS
        .iter()
        .map(|spec| spec.canonical_name)
        .collect::<Vec<_>>()
        .join("|");
    format!("Usage: /spawn <{names}> [x y z w] [material-id|material-name]")
}

fn spawnable_entity_spec_for_token(token: &str) -> Option<SpawnableEntitySpec> {
    let normalized = normalize_spawn_token(token);
    SPAWNABLE_ENTITY_SPECS.iter().copied().find(|spec| {
        spec.aliases
            .iter()
            .any(|alias| normalize_spawn_token(alias) == normalized)
    })
}

fn spawnable_entity_spec_for_kind(kind: EntityKind) -> Option<SpawnableEntitySpec> {
    SPAWNABLE_ENTITY_SPECS
        .iter()
        .copied()
        .find(|spec| spec.kind == kind)
}

fn default_spawn_pose_for_client(state: &ServerState, client_id: u64) -> ([f32; 4], [f32; 4]) {
    let Some(player) = state.players.get(&client_id) else {
        return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
    };
    let Some(snapshot) = state.entity_store.snapshot(player.entity_id) else {
        return ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]);
    };
    let look = normalize4_or_default(snapshot.orientation, [0.0, 0.0, 1.0, 0.0]);
    let position = [
        snapshot.position[0] + look[0] * 3.0,
        snapshot.position[1] + look[1] * 3.0,
        snapshot.position[2] + look[2] * 3.0,
        snapshot.position[3] + look[3] * 3.0,
    ];
    (position, look)
}

fn mob_archetype_defaults(archetype: MobArchetype) -> (f32, f32, f32) {
    match archetype {
        MobArchetype::Seeker => (2.9, 2.6, 0.64),
        MobArchetype::Creeper4d => (2.4, 3.8, 1.15),
        MobArchetype::PhaseSpider => (3.1, 2.4, 0.95),
    }
}

fn phase_spider_next_phase_deadline(now_ms: u64, phase_offset: f32, phase_ticks: u32) -> u64 {
    let span = PHASE_SPIDER_PHASE_MAX_INTERVAL_MS
        .saturating_sub(PHASE_SPIDER_PHASE_MIN_INTERVAL_MS)
        .max(1);
    let wobble =
        ((phase_offset * 31.0 + phase_ticks as f32 * 1.73).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
    now_ms.saturating_add(
        PHASE_SPIDER_PHASE_MIN_INTERVAL_MS + ((span as f32 * wobble).round() as u64),
    )
}

fn mob_archetype_for_kind(kind: EntityKind) -> Option<MobArchetype> {
    match kind {
        EntityKind::MobSeeker => Some(MobArchetype::Seeker),
        EntityKind::MobCreeper4d => Some(MobArchetype::Creeper4d),
        EntityKind::MobPhaseSpider => Some(MobArchetype::PhaseSpider),
        EntityKind::PlayerAvatar
        | EntityKind::TestCube
        | EntityKind::TestRotor
        | EntityKind::TestDrifter => None,
    }
}

fn encode_persisted_mob_payload(state: &ServerState, entity_id: u64) -> Vec<u8> {
    let Some(mob) = state.mobs.get(&entity_id) else {
        return Vec::new();
    };
    let payload = PersistedMobEntry {
        archetype: mob.archetype,
        phase_offset: mob.phase_offset,
        move_speed: mob.move_speed,
        preferred_distance: mob.preferred_distance,
        tangent_weight: mob.tangent_weight,
    };
    postcard::to_stdvec(&payload).unwrap_or_default()
}

fn decode_persisted_mob_payload(bytes: &[u8]) -> Option<PersistedMobEntry> {
    if bytes.is_empty() {
        return None;
    }
    postcard::from_bytes(bytes).ok()
}

fn collect_persisted_entities(state: &ServerState, now_ms: u64) -> Vec<PersistedEntityRecord> {
    let mut records: Vec<&EntityRecord> = state
        .entity_records
        .values()
        .filter(|record| {
            record.lifecycle == EntityLifecycle::Live
                && record.persistent
                && record.class != EntityClass::Player
        })
        .collect();
    records.sort_unstable_by_key(|record| record.entity_id);

    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let Some(snapshot) = state.entity_store.snapshot(record.entity_id) else {
            continue;
        };
        let payload = if record.class == EntityClass::Mob {
            encode_persisted_mob_payload(state, record.entity_id)
        } else {
            Vec::new()
        };
        out.push(PersistedEntityRecord {
            entity_id: record.entity_id,
            class: record.class,
            kind: snapshot.kind,
            position: snapshot.position,
            orientation: snapshot.orientation,
            velocity: snapshot.velocity,
            scale: snapshot.scale,
            material: snapshot.material,
            display_name: record.display_name.clone(),
            tags: Vec::new(),
            payload,
            last_saved_ms: now_ms,
        });
    }
    out
}

fn collect_player_records(state: &ServerState, now_ms: u64) -> Vec<PlayerRecord> {
    let mut player_ids: Vec<u64> = state.players.keys().copied().collect();
    player_ids.sort_unstable();
    let mut out = Vec::with_capacity(player_ids.len());
    for player_id in player_ids {
        let Some(player) = state.players.get(&player_id) else {
            continue;
        };
        let Some(snapshot) = state.entity_store.snapshot(player.entity_id) else {
            continue;
        };
        out.push(PlayerRecord {
            player_id,
            position: snapshot.position,
            orientation: snapshot.orientation,
            tags: Vec::new(),
            inventory_payload: Vec::new(),
            last_saved_ms: now_ms,
        });
    }
    out
}

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

fn mob_collision_radius_for_scale(scale: f32) -> f32 {
    let clamped_scale = if scale.is_finite() { scale } else { 0.5 };
    (clamped_scale * MOB_COLLISION_RADIUS_SCALE)
        .clamp(MOB_COLLISION_RADIUS_MIN, MOB_COLLISION_RADIUS_MAX)
}

#[derive(Clone, Debug)]
enum CollisionChunkCacheEntry {
    Explicit(voxel::Chunk),
    ExplicitEmpty,
    Effective(Option<voxel::Chunk>),
}

fn sample_effective_voxel_for_collision(
    state: &ServerState,
    cache: &mut HashMap<ChunkPos, CollisionChunkCacheEntry>,
    wx: i32,
    wy: i32,
    wz: i32,
    ww: i32,
) -> VoxelType {
    let (chunk_pos, voxel_index) = voxel::world_to_chunk(wx, wy, wz, ww);
    if let Some(entry) = cache.get(&chunk_pos) {
        return match entry {
            CollisionChunkCacheEntry::Explicit(chunk) => chunk.voxels[voxel_index],
            CollisionChunkCacheEntry::ExplicitEmpty => VoxelType::AIR,
            CollisionChunkCacheEntry::Effective(chunk) => chunk
                .as_ref()
                .map(|chunk| chunk.voxels[voxel_index])
                .unwrap_or(VoxelType::AIR),
        };
    }

    if let Some(chunk_data) = state.world.chunk_at(chunk_pos) {
        if chunk_data.is_empty() {
            cache.insert(chunk_pos, CollisionChunkCacheEntry::ExplicitEmpty);
            return VoxelType::AIR;
        }
        let voxel = chunk_data.voxels[voxel_index];
        cache.insert(chunk_pos, CollisionChunkCacheEntry::Explicit(chunk_data));
        return voxel;
    }

    let effective_chunk = state.world.effective_chunk(chunk_pos, false);
    let voxel = effective_chunk
        .as_ref()
        .map(|chunk| chunk.voxels[voxel_index])
        .unwrap_or(VoxelType::AIR);
    cache.insert(
        chunk_pos,
        CollisionChunkCacheEntry::Effective(effective_chunk),
    );
    voxel
}

fn mob_collides_at(
    state: &ServerState,
    cache: &mut HashMap<ChunkPos, CollisionChunkCacheEntry>,
    position: [f32; 4],
    radius: f32,
) -> bool {
    if position[1] - radius < MOB_HARD_WORLD_FLOOR_Y {
        return true;
    }

    let min = [
        position[0] - radius,
        position[1] - radius,
        position[2] - radius,
        position[3] - radius,
    ];
    let max = [
        position[0] + radius,
        position[1] + radius,
        position[2] + radius,
        position[3] + radius,
    ];
    let max_epsilon = 1e-4f32;
    let lo = [
        min[0].floor() as i32,
        min[1].floor() as i32,
        min[2].floor() as i32,
        min[3].floor() as i32,
    ];
    let hi = [
        (max[0] - max_epsilon).floor() as i32,
        (max[1] - max_epsilon).floor() as i32,
        (max[2] - max_epsilon).floor() as i32,
        (max[3] - max_epsilon).floor() as i32,
    ];
    if hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2] || hi[3] < lo[3] {
        return false;
    }

    for x in lo[0]..=hi[0] {
        for y in lo[1]..=hi[1] {
            for z in lo[2]..=hi[2] {
                for w in lo[3]..=hi[3] {
                    if sample_effective_voxel_for_collision(state, cache, x, y, z, w).is_solid() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn resolve_mob_collision(
    state: &ServerState,
    old_pos: [f32; 4],
    attempted_pos: [f32; 4],
    scale: f32,
) -> [f32; 4] {
    let radius = mob_collision_radius_for_scale(scale);
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    let mut pos = old_pos;
    if mob_collides_at(state, &mut cache, pos, radius) {
        for _ in 0..MOB_COLLISION_MAX_PUSHUP_STEPS {
            pos[1] += MOB_COLLISION_PUSHUP_STEP;
            if !mob_collides_at(state, &mut cache, pos, radius) {
                break;
            }
        }
    }

    for axis in [0usize, 2, 3, 1] {
        let target = attempted_pos[axis];
        if (target - pos[axis]).abs() <= 1e-6 {
            continue;
        }

        let mut candidate = pos;
        candidate[axis] = target;
        if !mob_collides_at(state, &mut cache, candidate, radius) {
            pos = candidate;
            continue;
        }

        let mut feasible = pos[axis];
        let mut blocked = target;
        for _ in 0..MOB_COLLISION_BINARY_STEPS {
            let mid = 0.5 * (feasible + blocked);
            let mut probe = pos;
            probe[axis] = mid;
            if mob_collides_at(state, &mut cache, probe, radius) {
                blocked = mid;
            } else {
                feasible = mid;
            }
        }
        pos[axis] = feasible;
    }

    pos
}

fn nearest_position_to(position: [f32; 4], candidates: &[[f32; 4]]) -> Option<[f32; 4]> {
    candidates.iter().copied().min_by(|a, b| {
        distance4_sq(*a, position)
            .partial_cmp(&distance4_sq(*b, position))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn mob_nav_base_cell_from_position(position: [f32; 4]) -> MobNavCell {
    [
        position[0].round() as i32,
        position[1].ceil() as i32,
        position[2].round() as i32,
        position[3].round() as i32,
    ]
}

fn mob_nav_position_from_cell(cell: MobNavCell) -> [f32; 4] {
    [
        cell[0] as f32,
        cell[1] as f32,
        cell[2] as f32,
        cell[3] as f32,
    ]
}

fn mob_nav_manhattan_distance(a: MobNavCell, b: MobNavCell) -> i32 {
    let dx = a[0].abs_diff(b[0]) as i32;
    let dy = a[1].abs_diff(b[1]) as i32;
    let dz = a[2].abs_diff(b[2]) as i32;
    let dw = a[3].abs_diff(b[3]) as i32;
    dx.saturating_add(dy).saturating_add(dz).saturating_add(dw)
}

fn mob_nav_has_line_of_sight(
    state: &ServerState,
    from: [f32; 4],
    to: [f32; 4],
    collision_radius: f32,
) -> bool {
    let delta = [
        to[0] - from[0],
        to[1] - from[1],
        to[2] - from[2],
        to[3] - from[3],
    ];
    let dist_sq = distance4_sq(from, to);
    if dist_sq <= 1e-6 {
        return true;
    }
    let dist = dist_sq.sqrt();
    let steps = (dist / MOB_NAV_PATH_LOS_STEP).ceil().max(1.0).min(256.0) as usize;
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    for idx in 1..=steps {
        let t = idx as f32 / steps as f32;
        let probe = [
            from[0] + delta[0] * t,
            from[1] + delta[1] * t,
            from[2] + delta[2] * t,
            from[3] + delta[3] * t,
        ];
        if mob_collides_at(state, &mut cache, probe, collision_radius) {
            return false;
        }
    }
    true
}

fn mob_nav_find_walkable_goal_cell(
    state: &ServerState,
    desired_goal: MobNavCell,
    origin: MobNavCell,
    collision_radius: f32,
) -> Option<MobNavCell> {
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    if !mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(desired_goal),
        collision_radius,
    ) {
        return Some(desired_goal);
    }

    let max_radius = MOB_NAV_PATH_GOAL_ADJUST_RADIUS_CELLS.max(0);
    let mut best: Option<(i32, MobNavCell)> = None;
    for dx in -max_radius..=max_radius {
        for dy in -max_radius..=max_radius {
            for dz in -max_radius..=max_radius {
                for dw in -max_radius..=max_radius {
                    let ring = dx.abs() + dy.abs() + dz.abs() + dw.abs();
                    if ring == 0 || ring > max_radius {
                        continue;
                    }
                    let candidate = [
                        desired_goal[0] + dx,
                        desired_goal[1] + dy,
                        desired_goal[2] + dz,
                        desired_goal[3] + dw,
                    ];
                    if mob_collides_at(
                        state,
                        &mut cache,
                        mob_nav_position_from_cell(candidate),
                        collision_radius,
                    ) {
                        continue;
                    }
                    let score = ring
                        .saturating_mul(16)
                        .saturating_add(mob_nav_manhattan_distance(candidate, origin));
                    match best {
                        Some((best_score, _)) if score >= best_score => {}
                        _ => best = Some((score, candidate)),
                    }
                }
            }
        }
    }
    best.map(|(_, cell)| cell)
}

fn mob_nav_reconstruct_path(
    start: MobNavCell,
    goal: MobNavCell,
    came_from: &HashMap<MobNavCell, MobNavCell>,
) -> Option<Vec<MobNavCell>> {
    if start == goal {
        return Some(Vec::new());
    }
    let mut reverse = Vec::new();
    let mut cursor = goal;
    let mut guards = 0usize;
    while cursor != start {
        reverse.push(cursor);
        cursor = *came_from.get(&cursor)?;
        guards = guards.saturating_add(1);
        if guards > MOB_NAV_PATH_MAX_SEARCH_STEPS {
            return None;
        }
    }
    reverse.reverse();
    if reverse.len() > MOB_NAV_PATH_MAX_WAYPOINTS {
        reverse.truncate(MOB_NAV_PATH_MAX_WAYPOINTS);
    }
    Some(reverse)
}

fn mob_nav_find_path(
    state: &ServerState,
    start: MobNavCell,
    goal: MobNavCell,
    collision_radius: f32,
) -> Option<MobNavPathResult> {
    if start == goal {
        return Some(MobNavPathResult {
            path_cells: Vec::new(),
            reached_goal: true,
            expanded_steps: 0,
            best_cell: start,
            best_goal_distance: 0,
        });
    }
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    if mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(start),
        collision_radius,
    ) {
        return None;
    }
    if mob_collides_at(
        state,
        &mut cache,
        mob_nav_position_from_cell(goal),
        collision_radius,
    ) {
        return None;
    }

    let mut open = BinaryHeap::<(Reverse<i32>, Reverse<i32>, MobNavCell)>::new();
    let mut g_scores = HashMap::<MobNavCell, i32>::new();
    let mut came_from = HashMap::<MobNavCell, MobNavCell>::new();
    let mut best_cell = start;
    let mut best_h = mob_nav_manhattan_distance(start, goal);
    let mut best_g = 0i32;

    g_scores.insert(start, 0);
    open.push((Reverse(best_h), Reverse(0), start));

    let mut visited_steps = 0usize;
    while let Some((_f_score, Reverse(g_cost), cell)) = open.pop() {
        let current_best = g_scores.get(&cell).copied().unwrap_or(i32::MAX);
        if g_cost > current_best {
            continue;
        }
        let h_cost = mob_nav_manhattan_distance(cell, goal);
        if h_cost < best_h || (h_cost == best_h && g_cost < best_g) {
            best_cell = cell;
            best_h = h_cost;
            best_g = g_cost;
        }
        if cell == goal {
            let path_cells = mob_nav_reconstruct_path(start, goal, &came_from)?;
            return Some(MobNavPathResult {
                path_cells,
                reached_goal: true,
                expanded_steps: visited_steps,
                best_cell: goal,
                best_goal_distance: 0,
            });
        }

        visited_steps = visited_steps.saturating_add(1);
        if visited_steps > MOB_NAV_PATH_MAX_SEARCH_STEPS {
            break;
        }

        for step in [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
        ] {
            let next = [
                cell[0] + step[0],
                cell[1] + step[1],
                cell[2] + step[2],
                cell[3] + step[3],
            ];
            if mob_nav_manhattan_distance(start, next) > MOB_NAV_PATH_MAX_SEARCH_RADIUS_CELLS {
                continue;
            }

            let tentative_g = g_cost.saturating_add(1);
            let known_next_g = g_scores.get(&next).copied().unwrap_or(i32::MAX);
            if tentative_g >= known_next_g {
                continue;
            }
            if mob_collides_at(
                state,
                &mut cache,
                mob_nav_position_from_cell(next),
                collision_radius,
            ) {
                continue;
            }

            came_from.insert(next, cell);
            g_scores.insert(next, tentative_g);
            let h_cost = mob_nav_manhattan_distance(next, goal);
            open.push((
                Reverse(tentative_g.saturating_add(h_cost)),
                Reverse(tentative_g),
                next,
            ));
        }
    }

    if best_cell != start {
        let path_cells = mob_nav_reconstruct_path(start, best_cell, &came_from)?;
        return Some(MobNavPathResult {
            path_cells,
            reached_goal: false,
            expanded_steps: visited_steps,
            best_cell,
            best_goal_distance: best_h,
        });
    }
    None
}

fn mob_nav_debug_log(
    navigation: &mut MobNavigationState,
    debug_enabled: bool,
    now_ms: u64,
    mob_entity_id: u64,
    archetype: MobArchetype,
    message: &str,
) {
    if !debug_enabled {
        return;
    }
    if now_ms.saturating_sub(navigation.last_debug_log_ms) < MOB_NAV_DEBUG_MIN_INTERVAL_MS {
        return;
    }
    navigation.last_debug_log_ms = now_ms;
    eprintln!(
        "[mob-nav][server] t={} entity={} archetype={:?} {}",
        now_ms, mob_entity_id, archetype, message
    );
}

fn update_mob_navigation_state(
    state: &ServerState,
    mut navigation: MobNavigationState,
    mob_entity_id: u64,
    archetype: MobArchetype,
    debug_enabled: bool,
    position: [f32; 4],
    scale: f32,
    target_position: Option<[f32; 4]>,
    now_ms: u64,
) -> (Option<[f32; 4]>, bool, MobNavigationState) {
    let Some(target) = target_position else {
        navigation.goal_cell = None;
        navigation.path_cells.clear();
        navigation.path_cursor = 0;
        return (None, false, navigation);
    };

    let collision_radius = mob_collision_radius_for_scale(scale);
    let desired_start_cell = mob_nav_base_cell_from_position(position);
    let start_cell = mob_nav_find_walkable_goal_cell(
        state,
        desired_start_cell,
        desired_start_cell,
        collision_radius,
    )
    .unwrap_or(desired_start_cell);
    let desired_goal_cell = mob_nav_base_cell_from_position(target);
    let goal_changed = navigation
        .goal_cell
        .map(|goal| {
            mob_nav_manhattan_distance(goal, desired_goal_cell)
                > MOB_NAV_PATH_GOAL_REPLAN_THRESHOLD_CELLS
        })
        .unwrap_or(true);
    let path_exhausted = navigation.path_cursor >= navigation.path_cells.len();
    let repath_due =
        now_ms.saturating_sub(navigation.last_repath_ms) >= MOB_NAV_PATH_REPLAN_INTERVAL_MS;

    let has_los = mob_nav_has_line_of_sight(state, position, target, collision_radius);
    if has_los {
        navigation.goal_cell = Some(desired_goal_cell);
        navigation.path_cells.clear();
        navigation.path_cursor = 0;
        if goal_changed || path_exhausted || repath_due {
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
                &format!(
                    "mode=los start={:?} goal={:?} path_exhausted={} repath_due={}",
                    start_cell, desired_goal_cell, path_exhausted, repath_due
                ),
            );
        }
        return (Some(target), false, navigation);
    }

    if goal_changed || path_exhausted || repath_due {
        let goal_cell =
            mob_nav_find_walkable_goal_cell(state, desired_goal_cell, start_cell, collision_radius)
                .unwrap_or(desired_goal_cell);
        navigation.goal_cell = Some(goal_cell);
        navigation.last_repath_ms = now_ms;

        if let Some(path_result) = mob_nav_find_path(state, start_cell, goal_cell, collision_radius)
        {
            let path_len = path_result.path_cells.len();
            let reached_goal = path_result.reached_goal;
            let expanded_steps = path_result.expanded_steps;
            let best_goal_distance = path_result.best_goal_distance;
            let best_cell = path_result.best_cell;
            navigation.path_cells = path_result.path_cells;
            navigation.path_cursor = 0;
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
                &format!(
                    "mode=path start={:?} goal={:?} reached_goal={} path_len={} expanded={} best_cell={:?} best_goal_dist={}",
                    start_cell,
                    goal_cell,
                    reached_goal,
                    path_len,
                    expanded_steps,
                    best_cell,
                    best_goal_distance
                ),
            );
        } else if goal_changed || path_exhausted {
            navigation.path_cells.clear();
            navigation.path_cursor = 0;
            mob_nav_debug_log(
                &mut navigation,
                debug_enabled,
                now_ms,
                mob_entity_id,
                archetype,
                &format!(
                    "mode=path-fail start={:?} goal={:?} (no reachable cell found)",
                    start_cell, goal_cell
                ),
            );
        }
    }

    while navigation.path_cursor < navigation.path_cells.len() {
        let waypoint = mob_nav_position_from_cell(navigation.path_cells[navigation.path_cursor]);
        if distance4_sq(position, waypoint).sqrt() <= MOB_NAV_PATH_NODE_REACH_DISTANCE {
            navigation.path_cursor = navigation.path_cursor.saturating_add(1);
        } else {
            break;
        }
    }

    if navigation.path_cursor < navigation.path_cells.len() {
        let waypoint = mob_nav_position_from_cell(navigation.path_cells[navigation.path_cursor]);
        (Some(waypoint), true, navigation)
    } else {
        (Some(target), false, navigation)
    }
}

fn simulate_seeker_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, slow_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let slow = if distance < MOB_NAV_PATH_NODE_REACH_DISTANCE * 0.9 {
                0.62
            } else {
                1.05
            };
            (direct, slow)
        } else if simple_steer {
            let slow = if distance < mob.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            (direct, slow)
        } else {
            // A 4D tangent that rotates [x,z] and [y,w] planes together.
            let tangent = normalize4_or_default(
                [-direct[2], -direct[3], direct[0], direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let weave_phase = t_s * 1.7 + mob.phase_offset;
            let weave = [
                0.10 * weave_phase.sin(),
                0.08 * (weave_phase * 0.7).cos(),
                -0.10 * weave_phase.sin(),
                0.08 * (weave_phase * 1.3).sin(),
            ];
            let pursuit = if distance > mob.preferred_distance {
                1.0
            } else {
                -0.45
            };
            let slow = if distance < mob.preferred_distance * 0.6 {
                0.42
            } else {
                1.0
            };
            let desired = normalize4_or_default(
                [
                    direct[0] * pursuit + tangent[0] * mob.tangent_weight + weave[0],
                    direct[1] * pursuit + tangent[1] * mob.tangent_weight + weave[1],
                    direct[2] * pursuit + tangent[2] * mob.tangent_weight + weave[2],
                    direct[3] * pursuit + tangent[3] * mob.tangent_weight + weave[3],
                ],
                direct,
            );
            (desired, slow)
        }
    } else {
        let phase = t_s * 0.65 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    phase.cos(),
                    0.45 * (phase * 0.7).sin(),
                    phase.sin(),
                    (phase * 1.1).cos(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.35,
        )
    };

    let step = mob.move_speed.max(0.1) * slow_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn simulate_creeper_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let speed = if distance > MOB_NAV_PATH_NODE_REACH_DISTANCE * 1.2 {
                1.18
            } else {
                0.66
            };
            (direct, speed)
        } else if simple_steer {
            let speed = if distance > mob.preferred_distance * 0.9 {
                1.15
            } else {
                0.70
            };
            (direct, speed)
        } else {
            // Creeper keeps pressure by orbiting in two coupled 4D planes, then lunges.
            let orbit_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let orbit_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let phase = t_s * 2.35 + mob.phase_offset;
            let orbit_mix = [
                orbit_a[0] * phase.sin() + orbit_b[0] * phase.cos(),
                orbit_a[1] * phase.sin() + orbit_b[1] * phase.cos(),
                orbit_a[2] * phase.sin() + orbit_b[2] * phase.cos(),
                orbit_a[3] * phase.sin() + orbit_b[3] * phase.cos(),
            ];
            let too_close = distance < mob.preferred_distance * 0.55;
            let in_lunge_band = distance > mob.preferred_distance * 0.80
                && distance < mob.preferred_distance * 1.95;
            let lunge = in_lunge_band && phase.sin() > 0.58;
            let pressure = if too_close {
                -0.72
            } else if lunge {
                1.85
            } else {
                0.68
            };
            let speed = if lunge {
                1.65
            } else if too_close {
                0.48
            } else {
                0.95
            };

            let desired = normalize4_or_default(
                [
                    direct[0] * pressure + orbit_mix[0] * mob.tangent_weight,
                    direct[1] * pressure + orbit_mix[1] * mob.tangent_weight,
                    direct[2] * pressure + orbit_mix[2] * mob.tangent_weight,
                    direct[3] * pressure + orbit_mix[3] * mob.tangent_weight,
                ],
                direct,
            );
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.72 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    0.7 * phase.sin(),
                    (phase * 0.9).cos(),
                    0.7 * phase.cos(),
                    (phase * 1.4).sin(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.38,
        )
    };

    let step = mob.move_speed.max(0.1) * speed_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn simulate_phase_spider_step(
    mob: &MobState,
    position: [f32; 4],
    target_position: Option<[f32; 4]>,
    path_following: bool,
    simple_steer: bool,
    now_ms: u64,
    previous_update_ms: u64,
) -> ([f32; 4], [f32; 4]) {
    let dt_s = (now_ms.saturating_sub(previous_update_ms)).max(1) as f32 * 0.001;
    let t_s = now_ms as f32 * 0.001;
    let (desired_dir, speed_factor) = if let Some(target) = target_position {
        let to_target = [
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
            target[3] - position[3],
        ];
        let distance = distance4_sq(target, position).sqrt();
        let direct = normalize4_or_default(to_target, [0.0, 0.0, 1.0, 0.0]);
        if path_following {
            let phase = t_s * 6.8 + mob.phase_offset;
            let tangent = normalize4_or_default(
                [direct[3], 0.0, -direct[0], direct[2]],
                [-direct[2], direct[3], direct[0], -direct[1]],
            );
            let lateral_scale = if simple_steer { 0.0 } else { 0.18 };
            let desired = normalize4_or_default(
                [
                    direct[0] + tangent[0] * lateral_scale * phase.sin(),
                    direct[1] + tangent[1] * lateral_scale * (phase * 0.7).sin(),
                    direct[2] + tangent[2] * lateral_scale * phase.sin(),
                    direct[3] + tangent[3] * lateral_scale * (phase * 0.9).cos(),
                ],
                direct,
            );
            let speed = if distance > MOB_NAV_PATH_NODE_REACH_DISTANCE * 1.4 {
                1.32
            } else {
                0.78
            };
            (desired, speed)
        } else if simple_steer {
            let speed = if distance > mob.preferred_distance {
                1.20
            } else {
                0.74
            };
            (direct, speed)
        } else {
            let phase = t_s * 5.2 + mob.phase_offset;
            let strafe_a = normalize4_or_default(
                [-direct[2], direct[3], direct[0], -direct[1]],
                [direct[3], 0.0, -direct[0], direct[1]],
            );
            let strafe_b = normalize4_or_default(
                [direct[1], -direct[0], direct[3], -direct[2]],
                [0.0, direct[2], -direct[1], direct[0]],
            );
            let strafe_mix = [
                strafe_a[0] * phase.sin() + strafe_b[0] * phase.cos(),
                strafe_a[1] * phase.sin() + strafe_b[1] * phase.cos(),
                strafe_a[2] * phase.sin() + strafe_b[2] * phase.cos(),
                strafe_a[3] * phase.sin() + strafe_b[3] * phase.cos(),
            ];
            let stalk = if distance > mob.preferred_distance * 1.35 {
                1.0
            } else if distance < mob.preferred_distance * 0.65 {
                -0.52
            } else {
                0.24
            };
            let vertical = 0.11 * (phase * 1.4).sin();
            let desired = normalize4_or_default(
                [
                    direct[0] * stalk + strafe_mix[0] * mob.tangent_weight * 1.30,
                    direct[1] * stalk + strafe_mix[1] * mob.tangent_weight * 1.30 + vertical,
                    direct[2] * stalk + strafe_mix[2] * mob.tangent_weight * 1.30,
                    direct[3] * stalk + strafe_mix[3] * mob.tangent_weight * 1.30,
                ],
                direct,
            );
            let speed = if distance > mob.preferred_distance {
                1.15
            } else {
                0.70
            };
            (desired, speed)
        }
    } else {
        let phase = t_s * 0.93 + mob.phase_offset;
        (
            normalize4_or_default(
                [
                    0.9 * phase.sin(),
                    0.28 * (phase * 1.1).cos(),
                    0.9 * phase.cos(),
                    (phase * 1.6).sin(),
                ],
                [0.0, 0.0, 1.0, 0.0],
            ),
            0.44,
        )
    };

    let step = mob.move_speed.max(0.1) * speed_factor * dt_s;
    let next_position = [
        position[0] + desired_dir[0] * step,
        position[1] + desired_dir[1] * step,
        position[2] + desired_dir[2] * step,
        position[3] + desired_dir[3] * step,
    ];
    (next_position, desired_dir)
}

fn attempt_phase_spider_blink(
    state: &ServerState,
    position: [f32; 4],
    forward: [f32; 4],
    scale: f32,
    phase_offset: f32,
    now_ms: u64,
) -> Option<[f32; 4]> {
    let forward = normalize4_or_default(forward, [0.0, 0.0, 1.0, 0.0]);
    let phase = now_ms as f32 * 0.0047 + phase_offset * 1.3;
    let strafe_a = normalize4_or_default(
        [-forward[2], forward[3], forward[0], -forward[1]],
        [forward[3], 0.0, -forward[0], forward[1]],
    );
    let strafe_b = normalize4_or_default(
        [forward[1], -forward[0], forward[3], -forward[2]],
        [0.0, forward[2], -forward[1], forward[0]],
    );
    let drift = normalize4_or_default(
        [
            strafe_a[0] * phase.sin() + strafe_b[0] * phase.cos(),
            strafe_a[1] * phase.sin() + strafe_b[1] * phase.cos(),
            strafe_a[2] * phase.sin() + strafe_b[2] * phase.cos(),
            strafe_a[3] * phase.sin() + strafe_b[3] * phase.cos(),
        ],
        strafe_a,
    );
    let blink_dir = normalize4_or_default(
        [
            forward[0] * 1.0 + drift[0] * 0.42,
            forward[1] * 1.0 + drift[1] * 0.42,
            forward[2] * 1.0 + drift[2] * 0.42,
            forward[3] * 1.0 + drift[3] * 0.42,
        ],
        forward,
    );

    let radius = mob_collision_radius_for_scale(scale);
    let mut cache = HashMap::<ChunkPos, CollisionChunkCacheEntry>::new();
    let lift_primary = 0.16 + 0.12 * (phase * 0.8).sin().abs();
    let lift_options = [lift_primary, 0.05, -0.08];

    let mut distance = PHASE_SPIDER_PHASE_DISTANCE;
    while distance >= PHASE_SPIDER_PHASE_MIN_DISTANCE {
        for &lift in &lift_options {
            let candidate = [
                position[0] + blink_dir[0] * distance,
                position[1] + blink_dir[1] * distance + lift,
                position[2] + blink_dir[2] * distance,
                position[3] + blink_dir[3] * distance,
            ];
            if !mob_collides_at(state, &mut cache, candidate, radius) {
                return Some(candidate);
            }
        }
        distance -= 0.55;
    }
    None
}

fn simulate_mobs(
    state: &mut ServerState,
    now_ms: u64,
) -> (
    Vec<QueuedWorldVoxelSet>,
    Vec<QueuedExplosionEvent>,
    Vec<QueuedPlayerMovementModifier>,
) {
    let player_positions: Vec<[f32; 4]> = state
        .players
        .values()
        .filter_map(|player| state.entity_store.snapshot(player.entity_id))
        .map(|snapshot| snapshot.position)
        .collect();

    let mut stale = Vec::new();
    let mut detonations = Vec::new();
    let mut navigation_updates = Vec::with_capacity(state.mobs.len());
    let mut phase_deadline_updates = Vec::with_capacity(state.mobs.len());
    let mut updates = Vec::with_capacity(state.mobs.len());
    let mob_entity_ids: Vec<u64> = state.mobs.keys().copied().collect();
    for entity_id in mob_entity_ids {
        let Some(mob) = state.mobs.get(&entity_id).cloned() else {
            continue;
        };
        let Some(snapshot) = state.entity_store.snapshot(mob.entity_id) else {
            stale.push(mob.entity_id);
            continue;
        };
        let nearest_target = nearest_position_to(snapshot.position, &player_positions);
        let navigation_target = match mob.archetype {
            MobArchetype::Seeker => nearest_target,
            MobArchetype::Creeper4d => nearest_target.map(|target| {
                [
                    target[0],
                    target[1] - CREEPER_POUNCE_TARGET_BELOW_PLAYER_Y,
                    target[2],
                    target[3],
                ]
            }),
            MobArchetype::PhaseSpider => nearest_target,
        };
        if mob.archetype == MobArchetype::Creeper4d {
            let should_detonate = player_positions.iter().any(|player_position| {
                distance4_sq(snapshot.position, *player_position).sqrt()
                    <= CREEPER_EXPLOSION_TRIGGER_DISTANCE
            });
            if should_detonate {
                detonations.push((mob.entity_id, snapshot.position));
                continue;
            }
        }
        let (target_position, path_following, mut navigation) = update_mob_navigation_state(
            state,
            mob.navigation.clone(),
            mob.entity_id,
            mob.archetype,
            state.mob_nav_debug,
            snapshot.position,
            snapshot.scale,
            navigation_target,
            now_ms,
        );
        let (next_position, next_forward) = match mob.archetype {
            MobArchetype::Seeker => simulate_seeker_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
            MobArchetype::Creeper4d => simulate_creeper_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
            MobArchetype::PhaseSpider => simulate_phase_spider_step(
                &mob,
                snapshot.position,
                target_position,
                path_following,
                state.mob_nav_simple_steer,
                now_ms,
                snapshot.last_update_ms,
            ),
        };
        let mut final_position =
            resolve_mob_collision(state, snapshot.position, next_position, snapshot.scale);
        let mut final_forward = next_forward;
        if mob.archetype == MobArchetype::PhaseSpider && now_ms >= mob.next_phase_ms {
            let attempted = distance4_sq(snapshot.position, next_position).sqrt();
            let resolved = distance4_sq(snapshot.position, final_position).sqrt();
            let blocked =
                attempted > 0.12 && resolved + PHASE_SPIDER_BLOCKED_PROGRESS_EPSILON < attempted;
            let should_phase = nearest_target.is_some() && (path_following || blocked);
            if should_phase {
                if let Some(phase_position) = attempt_phase_spider_blink(
                    state,
                    snapshot.position,
                    next_forward,
                    snapshot.scale,
                    mob.phase_offset,
                    now_ms,
                ) {
                    final_forward = normalize4_or_default(
                        [
                            phase_position[0] - snapshot.position[0],
                            phase_position[1] - snapshot.position[1],
                            phase_position[2] - snapshot.position[2],
                            phase_position[3] - snapshot.position[3],
                        ],
                        next_forward,
                    );
                    final_position = phase_position;
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.archetype,
                        &format!("mode=phase-blink success pos={:?}", phase_position),
                    );
                } else {
                    mob_nav_debug_log(
                        &mut navigation,
                        state.mob_nav_debug,
                        now_ms,
                        mob.entity_id,
                        mob.archetype,
                        "mode=phase-blink fail (no valid destination)",
                    );
                }
            }
            let next_deadline = phase_spider_next_phase_deadline(
                now_ms,
                mob.phase_offset,
                (mob.entity_id as u32) ^ (now_ms as u32),
            );
            phase_deadline_updates.push((mob.entity_id, next_deadline));
        }
        navigation_updates.push((mob.entity_id, navigation));
        updates.push((mob.entity_id, final_position, final_forward));
    }

    for (entity_id, navigation) in navigation_updates {
        if let Some(mob) = state.mobs.get_mut(&entity_id) {
            mob.navigation = navigation;
        }
    }
    for (entity_id, next_phase_ms) in phase_deadline_updates {
        if let Some(mob) = state.mobs.get_mut(&entity_id) {
            mob.next_phase_ms = next_phase_ms;
        }
    }

    let mut persistent_motion = false;
    for (entity_id, next_position, next_forward) in updates {
        if !state
            .entity_store
            .set_motion_state(entity_id, next_position, next_forward, now_ms)
        {
            stale.push(entity_id);
            continue;
        }
        if state
            .entity_records
            .get(&entity_id)
            .map(|record| record.persistent)
            .unwrap_or(false)
        {
            persistent_motion = true;
        }
    }
    if persistent_motion {
        mark_entities_dirty(state);
    }

    stale.sort_unstable();
    stale.dedup();
    for entity_id in stale {
        state.mobs.remove(&entity_id);
        mark_entity_record_despawned(state, entity_id, Some(now_ms));
        let _ = state.entity_store.despawn(entity_id);
    }

    let mut queued_voxel_sets = Vec::new();
    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    for (entity_id, center) in detonations {
        if !state.mobs.contains_key(&entity_id) {
            continue;
        }
        let (mut voxel_sets, explosion) =
            apply_creeper_explosion(state, entity_id, center, CREEPER_EXPLOSION_RADIUS_VOXELS);
        let voxel_update_count = voxel_sets.len();
        queued_voxel_sets.append(&mut voxel_sets);
        queued_explosions.push(explosion);
        let (persistent_motion, mut player_modifiers) = apply_explosion_impulse(
            state,
            entity_id,
            center,
            CREEPER_EXPLOSION_IMPULSE_RADIUS,
            CREEPER_EXPLOSION_MAX_IMPULSE_DISTANCE,
            now_ms,
        );
        eprintln!(
            "[impulse-debug][server] creeper_explosion source_entity={} center={:?} voxel_updates={} player_modifiers={} persistent_motion={}",
            entity_id,
            center,
            voxel_update_count,
            player_modifiers.len(),
            persistent_motion
        );
        for modifier in &player_modifiers {
            let delta = modifier.delta_position;
            let delta_len = (delta[0] * delta[0]
                + delta[1] * delta[1]
                + delta[2] * delta[2]
                + delta[3] * delta[3])
                .sqrt();
            eprintln!(
                "[impulse-debug][server] queued_modifier client_id={} source_entity={:?} delta={:?} delta_len={:.3} delta_velocity_y={:.3}",
                modifier.client_id,
                modifier.source_entity_id,
                modifier.delta_position,
                delta_len,
                modifier.delta_velocity_y
            );
        }
        queued_player_modifiers.append(&mut player_modifiers);
        if persistent_motion {
            mark_entities_dirty(state);
        }
        state.mobs.remove(&entity_id);
        mark_entity_record_despawned(state, entity_id, Some(now_ms));
        let _ = state.entity_store.despawn(entity_id);
    }

    (
        queued_voxel_sets,
        queued_explosions,
        queued_player_modifiers,
    )
}

fn tick_entity_simulation_window(
    state: &mut ServerState,
    now_ms: u64,
    next_sim_ms: &mut u64,
    sim_step_ms: u64,
) -> (
    Vec<QueuedWorldVoxelSet>,
    Vec<QueuedExplosionEvent>,
    Vec<QueuedPlayerMovementModifier>,
) {
    let mut queued_voxel_sets = Vec::new();
    let mut queued_explosions = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    let mut sim_steps = 0usize;
    while *next_sim_ms <= now_ms && sim_steps < ENTITY_SIM_STEP_MAX_PER_BROADCAST {
        let moved_entities = state.entity_store.simulate(*next_sim_ms);
        let persistent_accent_motion = moved_entities.iter().any(|entity_id| {
            state
                .entity_records
                .get(entity_id)
                .map(|record| {
                    record.lifecycle == EntityLifecycle::Live
                        && record.persistent
                        && record.class == EntityClass::Accent
                })
                .unwrap_or(false)
        });
        if persistent_accent_motion {
            mark_entities_dirty(state);
        }
        let (mut step_voxel_sets, mut step_explosions, mut step_player_modifiers) =
            simulate_mobs(state, *next_sim_ms);
        queued_voxel_sets.append(&mut step_voxel_sets);
        queued_explosions.append(&mut step_explosions);
        queued_player_modifiers.append(&mut step_player_modifiers);
        *next_sim_ms = (*next_sim_ms).saturating_add(sim_step_ms);
        sim_steps += 1;
    }
    if sim_steps == ENTITY_SIM_STEP_MAX_PER_BROADCAST && *next_sim_ms <= now_ms {
        *next_sim_ms = now_ms.saturating_add(sim_step_ms);
    }
    (
        queued_voxel_sets,
        queued_explosions,
        queued_player_modifiers,
    )
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

fn encode_world_chunk_payload_from_realized(
    lod_level: u8,
    chunk_pos: ChunkPos,
    payload: &ChunkPayload,
) -> Option<WorldChunkPayload> {
    if matches!(payload, ChunkPayload::Empty) {
        return None;
    }
    let dense = payload.dense_materials().ok()?;
    if dense.len() != voxel::CHUNK_VOLUME {
        return None;
    }
    Some(WorldChunkPayload {
        lod_level,
        chunk_pos: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        // Current wire format is u8-only. Clamp out-of-range materials to u8::MAX
        // until the transport is upgraded to support larger material IDs.
        voxels: dense
            .into_iter()
            .map(|material| u8::try_from(material).unwrap_or(u8::MAX))
            .collect(),
    })
}

fn stream_chunk_distance2(chunk_pos: ChunkPos, center_chunk: [i32; 4]) -> i64 {
    let dx = (chunk_pos.x - center_chunk[0]) as i64;
    let dy = (chunk_pos.y - center_chunk[1]) as i64;
    let dz = (chunk_pos.z - center_chunk[2]) as i64;
    let dw = (chunk_pos.w - center_chunk[3]) as i64;
    dx * dx + dy * dy + dz * dz + dw * dw
}

fn stream_near_bounds(center_chunk: [i32; 4], near_chunk_radius: i32) -> Aabb4i {
    let near_radius = near_chunk_radius.max(0);
    let min_chunk = [
        center_chunk[0] - near_radius,
        center_chunk[1] - near_radius,
        center_chunk[2] - near_radius,
        center_chunk[3] - near_radius,
    ];
    let max_chunk = [
        center_chunk[0] + near_radius,
        center_chunk[1] + near_radius,
        center_chunk[2] + near_radius,
        center_chunk[3] + near_radius,
    ];
    Aabb4i::new(min_chunk, max_chunk)
}

fn region_bounds_for_chunk_bounds(bounds: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            bounds.min[0].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.min[1].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.min[2].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.min[3].div_euclid(REGION_TILE_EDGE_CHUNKS),
        ],
        [
            bounds.max[0].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.max[1].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.max[2].div_euclid(REGION_TILE_EDGE_CHUNKS),
            bounds.max[3].div_euclid(REGION_TILE_EDGE_CHUNKS),
        ],
    )
}

fn collect_region_clocks_in_chunk_bounds(
    region_clocks: &RegionClockMap,
    chunk_bounds: Aabb4i,
) -> RegionClockMap {
    if !chunk_bounds.is_valid() {
        return HashMap::new();
    }
    let region_bounds = region_bounds_for_chunk_bounds(chunk_bounds);
    let mut out = HashMap::new();
    for rw in region_bounds.min[3]..=region_bounds.max[3] {
        for rz in region_bounds.min[2]..=region_bounds.max[2] {
            for ry in region_bounds.min[1]..=region_bounds.max[1] {
                for rx in region_bounds.min[0]..=region_bounds.max[0] {
                    let region_id = RegionId {
                        x: rx,
                        y: ry,
                        z: rz,
                        w: rw,
                    };
                    let clock = *region_clocks.get(&region_id).unwrap_or(&0);
                    if clock != 0 {
                        out.insert(region_id, clock);
                    }
                }
            }
        }
    }
    out
}

fn region_clock_payloads_from_map(clocks: &RegionClockMap) -> Vec<RegionClockPayload> {
    let mut out: Vec<RegionClockPayload> = clocks
        .iter()
        .map(|(region_id, clock)| RegionClockPayload {
            region_id: [region_id.x, region_id.y, region_id.z, region_id.w],
            clock: *clock,
        })
        .collect();
    out.sort_unstable_by_key(|update| {
        (
            update.region_id[3],
            update.region_id[2],
            update.region_id[1],
            update.region_id[0],
        )
    });
    out
}

fn region_clock_updates_from_map(clocks: &RegionClockMap) -> Vec<RegionClockUpdate> {
    let mut out: Vec<RegionClockUpdate> = clocks
        .iter()
        .map(|(region_id, clock)| RegionClockUpdate {
            region_id: *region_id,
            new_clock: *clock,
        })
        .collect();
    out.sort_unstable_by_key(|update| {
        (
            update.region_id.w,
            update.region_id.z,
            update.region_id.y,
            update.region_id.x,
        )
    });
    out
}

fn region_clock_preconditions_for_patch(
    previous_clocks: &RegionClockMap,
    current_clocks: &RegionClockMap,
) -> Vec<RegionClockPrecondition> {
    let mut out: Vec<RegionClockPrecondition> = current_clocks
        .keys()
        .map(|region_id| RegionClockPrecondition {
            region_id: *region_id,
            expected_clock: *previous_clocks.get(region_id).unwrap_or(&0),
        })
        .collect();
    out.sort_unstable_by_key(|precondition| {
        (
            precondition.region_id.w,
            precondition.region_id.z,
            precondition.region_id.y,
            precondition.region_id.x,
        )
    });
    out
}

fn next_stream_patch_seq(next_seq: &mut u64) -> u64 {
    let seq = (*next_seq).max(1);
    *next_seq = seq.wrapping_add(1);
    if *next_seq == 0 {
        *next_seq = 1;
    }
    seq
}

fn sanitize_region_resync_entries(mut entries: Vec<RegionResyncEntry>) -> Vec<RegionResyncEntry> {
    const MAX_REGION_RESYNC_ENTRIES: usize = 512;
    entries.sort_unstable_by_key(|entry| {
        (
            entry.region_id.w,
            entry.region_id.z,
            entry.region_id.y,
            entry.region_id.x,
        )
    });
    entries.dedup_by_key(|entry| entry.region_id);
    if entries.len() > MAX_REGION_RESYNC_ENTRIES {
        entries.truncate(MAX_REGION_RESYNC_ENTRIES);
    }
    entries
}

fn chunk_bounds_for_region_id(region_id: RegionId) -> Aabb4i {
    let min = [
        region_id.x.saturating_mul(REGION_TILE_EDGE_CHUNKS),
        region_id.y.saturating_mul(REGION_TILE_EDGE_CHUNKS),
        region_id.z.saturating_mul(REGION_TILE_EDGE_CHUNKS),
        region_id.w.saturating_mul(REGION_TILE_EDGE_CHUNKS),
    ];
    let max = [
        min[0].saturating_add(REGION_TILE_EDGE_CHUNKS.saturating_sub(1)),
        min[1].saturating_add(REGION_TILE_EDGE_CHUNKS.saturating_sub(1)),
        min[2].saturating_add(REGION_TILE_EDGE_CHUNKS.saturating_sub(1)),
        min[3].saturating_add(REGION_TILE_EDGE_CHUNKS.saturating_sub(1)),
    ];
    Aabb4i::new(min, max)
}

fn merge_aabb(a: Aabb4i, b: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
            a.min[3].min(b.min[3]),
        ],
        [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
            a.max[3].max(b.max[3]),
        ],
    )
}

fn chunk_bounds_for_resync_entries(entries: &[RegionResyncEntry]) -> Option<Aabb4i> {
    let mut iter = entries.iter();
    let first = iter.next()?;
    let mut bounds = chunk_bounds_for_region_id(first.region_id);
    for entry in iter {
        bounds = merge_aabb(bounds, chunk_bounds_for_region_id(entry.region_id));
    }
    Some(bounds)
}

fn region_patch_wire_size_bytes(patch: &RegionSubtreePatch) -> Option<usize> {
    postcard::to_stdvec(&ServerMessage::WorldRegionPatch {
        patch: patch.clone(),
    })
    .ok()
    .map(|bytes| bytes.len())
}

fn split_chunk_bounds_longest_axis(bounds: Aabb4i) -> Option<(Aabb4i, Aabb4i)> {
    if !bounds.is_valid() {
        return None;
    }
    let spans = [
        i64::from(bounds.max[0]) - i64::from(bounds.min[0]) + 1,
        i64::from(bounds.max[1]) - i64::from(bounds.min[1]) + 1,
        i64::from(bounds.max[2]) - i64::from(bounds.min[2]) + 1,
        i64::from(bounds.max[3]) - i64::from(bounds.min[3]) + 1,
    ];
    let mut axis = 0usize;
    for candidate in 1..4 {
        if spans[candidate] > spans[axis] {
            axis = candidate;
        }
    }
    if spans[axis] <= 1 {
        return None;
    }

    let left_len = spans[axis] / 2;
    let left_max_axis = i64::from(bounds.min[axis]) + left_len - 1;
    let right_min_axis = left_max_axis + 1;
    let left_max_axis = i32::try_from(left_max_axis).ok()?;
    let right_min_axis = i32::try_from(right_min_axis).ok()?;

    let mut left_max = bounds.max;
    left_max[axis] = left_max_axis;
    let mut right_min = bounds.min;
    right_min[axis] = right_min_axis;
    Some((
        Aabb4i::new(bounds.min, left_max),
        Aabb4i::new(right_min, bounds.max),
    ))
}

fn build_region_patches_recursive<W: WorldField + ?Sized>(
    world: &W,
    bounds: Aabb4i,
    full_region_clocks: &RegionClockMap,
    working_known_clocks: &mut RegionClockMap,
    next_patch_seq: &mut u64,
    out: &mut Vec<RegionSubtreePatch>,
) -> bool {
    if !bounds.is_valid() {
        return true;
    }

    let current_region_clocks = collect_region_clocks_in_chunk_bounds(full_region_clocks, bounds);
    let preconditions =
        region_clock_preconditions_for_patch(working_known_clocks, &current_region_clocks);
    let clock_updates = region_clock_updates_from_map(&current_region_clocks);
    let subtree = world
        .query_region_core(QueryVolume { bounds }, QueryDetail::Exact)
        .as_ref()
        .clone();
    let estimated_patch = RegionSubtreePatch {
        patch_seq: 1,
        bounds,
        preconditions: preconditions.clone(),
        clock_updates: clock_updates.clone(),
        subtree: subtree.clone(),
    };
    let estimated_size = region_patch_wire_size_bytes(&estimated_patch).unwrap_or(usize::MAX);

    if estimated_size > MAX_PATCH_BYTES {
        let Some((left_bounds, right_bounds)) = split_chunk_bounds_longest_axis(bounds) else {
            eprintln!(
                "Unable to split oversized region patch bounds={:?}->{:?} bytes={} limit={}",
                bounds.min, bounds.max, estimated_size, MAX_PATCH_BYTES
            );
            return false;
        };
        return build_region_patches_recursive(
            world,
            left_bounds,
            full_region_clocks,
            working_known_clocks,
            next_patch_seq,
            out,
        ) && build_region_patches_recursive(
            world,
            right_bounds,
            full_region_clocks,
            working_known_clocks,
            next_patch_seq,
            out,
        );
    }

    let patch = RegionSubtreePatch {
        patch_seq: next_stream_patch_seq(next_patch_seq),
        bounds,
        preconditions,
        clock_updates,
        subtree,
    };
    let wire_size = region_patch_wire_size_bytes(&patch).unwrap_or(usize::MAX);
    if wire_size > MAX_PATCH_BYTES {
        eprintln!(
            "Region patch became oversized after sequencing bounds={:?}->{:?} bytes={} limit={}",
            bounds.min, bounds.max, wire_size, MAX_PATCH_BYTES
        );
        return false;
    }

    for update in &patch.clock_updates {
        working_known_clocks.insert(update.region_id, update.new_clock);
    }
    out.push(patch);
    true
}

fn build_region_patches_for_bounds<W: WorldField + ?Sized>(
    world: &W,
    bounds: Aabb4i,
    full_region_clocks: &RegionClockMap,
    initial_known_clocks: &RegionClockMap,
    next_patch_seq: &mut u64,
) -> Option<(Vec<RegionSubtreePatch>, RegionClockMap)> {
    let mut working_known_clocks = initial_known_clocks.clone();
    let mut patches = Vec::new();
    if !build_region_patches_recursive(
        world,
        bounds,
        full_region_clocks,
        &mut working_known_clocks,
        next_patch_seq,
        &mut patches,
    ) {
        return None;
    }
    Some((patches, working_known_clocks))
}

fn plan_stream_window_update(
    state: &mut ServerState,
    client_id: u64,
    center_chunk: [i32; 4],
    near_chunk_radius: i32,
    _mid_chunk_radius: i32,
    _far_chunk_radius: i32,
) -> Option<(u64, Vec<WorldChunkPayload>, Vec<WorldChunkCoordPayload>)> {
    let near_bounds = stream_near_bounds(center_chunk, near_chunk_radius);
    let query_core = state.world.query_region_core(
        QueryVolume {
            bounds: near_bounds,
        },
        QueryDetail::Exact,
    );

    let (mut desired_streamed, mut load_chunk_positions, unload_positions) = {
        let player = state.players.get_mut(&client_id)?;
        let refresh = player
            .working_set
            .refresh_from_core(near_bounds, query_core.as_ref());
        (
            refresh.desired_chunks,
            refresh.load_positions,
            refresh.unload_positions,
        )
    };

    load_chunk_positions.sort_unstable_by_key(|chunk_pos| {
        (
            stream_chunk_distance2(*chunk_pos, center_chunk),
            chunk_pos.w,
            chunk_pos.z,
            chunk_pos.y,
            chunk_pos.x,
        )
    });

    let mut load_payloads = Vec::with_capacity(load_chunk_positions.len());
    let mut invalid_positions = Vec::new();
    for chunk_pos in load_chunk_positions {
        let Some(payload) = desired_streamed.get(&chunk_pos) else {
            continue;
        };
        if let Some(encoded) =
            encode_world_chunk_payload_from_realized(WORLD_CHUNK_LOD_NEAR, chunk_pos, payload)
        {
            load_payloads.push(encoded);
        } else {
            invalid_positions.push(chunk_pos);
        }
    }
    for chunk_pos in invalid_positions {
        desired_streamed.remove(&chunk_pos);
        let player = state.players.get_mut(&client_id)?;
        let _ = player.working_set.remove_chunk(chunk_pos);
    }

    let mut unload_chunks: Vec<WorldChunkCoordPayload> = unload_positions
        .into_iter()
        .map(|chunk_pos| WorldChunkCoordPayload {
            lod_level: WORLD_CHUNK_LOD_NEAR,
            chunk_pos: [chunk_pos.x, chunk_pos.y, chunk_pos.z, chunk_pos.w],
        })
        .collect();
    unload_chunks.sort_unstable_by_key(|payload| {
        (
            payload.lod_level,
            payload.chunk_pos[3],
            payload.chunk_pos[2],
            payload.chunk_pos[1],
            payload.chunk_pos[0],
        )
    });

    Some((state.world_revision, load_payloads, unload_chunks))
}

fn send_to_client(state: &SharedState, client_id: u64, message: ServerMessage) {
    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    if let Some(tx) = sender {
        let _ = tx.send(message);
    }
}

fn prune_stale_clients(state: &SharedState, stale: Vec<u64>, notify_entity_destroyed: bool) {
    if stale.is_empty() {
        return;
    }
    let mut disconnected_entity_ids = Vec::new();
    {
        let mut guard = state.lock().expect("server state lock poisoned");
        for client_id in stale {
            guard.clients.remove(&client_id);
            guard.client_visible_entities.remove(&client_id);
            if let Some(player) = guard.players.remove(&client_id) {
                mark_entity_record_despawned(&mut guard, player.entity_id, None);
                let _ = guard.entity_store.despawn(player.entity_id);
                disconnected_entity_ids.push(player.entity_id);
            }
        }
    }
    if notify_entity_destroyed {
        for entity_id in disconnected_entity_ids {
            broadcast(state, ServerMessage::EntityDestroyed { entity_id });
        }
    }
}

fn broadcast(state: &SharedState, message: ServerMessage) {
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .map(|(&client_id, tx)| (client_id, tx.clone()))
            .collect()
    };

    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    prune_stale_clients(state, stale, true);
}

fn send_world_voxel_set_to_streamed_clients(
    state: &SharedState,
    position: [i32; 4],
    material: u8,
    source_client_id: Option<u64>,
    client_edit_id: Option<u64>,
    revision: u64,
) {
    let (chunk_pos, _) = voxel::world_to_chunk(position[0], position[1], position[2], position[3]);
    let clients: Vec<_> = {
        let guard = state.lock().expect("server state lock poisoned");
        guard
            .clients
            .iter()
            .filter_map(|(&client_id, tx)| {
                let is_source = source_client_id == Some(client_id);
                let is_streaming_chunk = guard
                    .players
                    .get(&client_id)
                    .map(|player| player.working_set.contains_chunk(chunk_pos))
                    .unwrap_or(false);
                if is_source || is_streaming_chunk {
                    Some((client_id, tx.clone()))
                } else {
                    None
                }
            })
            .collect()
    };

    let message = ServerMessage::WorldVoxelSet {
        position,
        material,
        source_client_id,
        client_edit_id,
        revision,
    };
    let mut stale = Vec::new();
    for (client_id, tx) in clients {
        if tx.send(message.clone()).is_err() {
            stale.push(client_id);
        }
    }

    prune_stale_clients(state, stale, true);
}

fn send_region_clock_updates_to_clients(state: &SharedState, updates: Vec<RegionClockPayload>) {
    if updates.is_empty() {
        return;
    }
    {
        let mut guard = state.lock().expect("server state lock poisoned");
        for player in guard.players.values_mut() {
            for update in &updates {
                let region_id = RegionId {
                    x: update.region_id[0],
                    y: update.region_id[1],
                    z: update.region_id[2],
                    w: update.region_id[3],
                };
                let entry = player.known_region_clocks.entry(region_id).or_insert(0);
                if update.clock > *entry {
                    *entry = update.clock;
                }
            }
        }
    }
    broadcast(state, ServerMessage::WorldRegionClockUpdate { updates });
}

fn sync_streamed_chunks_for_client(
    state: &SharedState,
    client_id: u64,
    center_chunk: [i32; 4],
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    force: bool,
) {
    let emit_region_patch_only = env_flag_enabled("R4D_STREAM_REGION_PATCH_ONLY");
    let emit_region_patch = emit_region_patch_only || env_flag_enabled("R4D_STREAM_REGION_PATCH");
    let update = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let near_bounds = stream_near_bounds(center_chunk, near_chunk_radius);
        let current_region_clocks =
            collect_region_clocks_in_chunk_bounds(&guard.region_clocks, near_bounds);
        let mut should_emit_stream_patch = false;
        let should_sync = match guard.players.get_mut(&client_id) {
            Some(player) => {
                let changed = player.last_stream_center_chunk != Some(center_chunk);
                let world_changed = player.last_stream_region_clocks != current_region_clocks;
                if force || changed || world_changed {
                    should_emit_stream_patch = emit_region_patch;
                    player.last_stream_center_chunk = Some(center_chunk);
                    player.last_stream_region_clocks = current_region_clocks.clone();
                    true
                } else {
                    false
                }
            }
            None => false,
        };

        if should_sync {
            let update = plan_stream_window_update(
                &mut guard,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
            );
            update.map(|(revision, loads, unloads)| {
                let mut region_patches = Vec::new();
                let mut patch_fallback_to_chunks = false;

                if should_emit_stream_patch {
                    let (known_before, mut next_patch_seq) = {
                        let Some(player) = guard.players.get(&client_id) else {
                            return (
                                revision,
                                loads,
                                unloads,
                                region_clock_payloads_from_map(&current_region_clocks),
                                region_patches,
                                true,
                            );
                        };
                        (
                            player.known_region_clocks.clone(),
                            player.next_stream_patch_seq,
                        )
                    };

                    match build_region_patches_for_bounds(
                        &guard.world,
                        near_bounds,
                        &current_region_clocks,
                        &known_before,
                        &mut next_patch_seq,
                    ) {
                        Some((patches, known_after)) => {
                            region_patches = patches;
                            if let Some(player) = guard.players.get_mut(&client_id) {
                                player.known_region_clocks = known_after;
                                player.next_stream_patch_seq = next_patch_seq;
                            }
                        }
                        None => {
                            patch_fallback_to_chunks = true;
                            if let Some(player) = guard.players.get_mut(&client_id) {
                                for (region_id, clock) in &current_region_clocks {
                                    player.known_region_clocks.insert(*region_id, *clock);
                                }
                            }
                        }
                    }
                } else if let Some(player) = guard.players.get_mut(&client_id) {
                    for (region_id, clock) in &current_region_clocks {
                        player.known_region_clocks.insert(*region_id, *clock);
                    }
                }
                (
                    revision,
                    loads,
                    unloads,
                    region_clock_payloads_from_map(&current_region_clocks),
                    region_patches,
                    patch_fallback_to_chunks,
                )
            })
        } else {
            None
        }
    };

    let Some((
        revision,
        chunk_loads,
        chunk_unloads,
        region_clock_snapshot,
        region_patches,
        patch_fallback_to_chunks,
    )) = update
    else {
        return;
    };

    let sender = {
        let guard = state.lock().expect("server state lock poisoned");
        guard.clients.get(&client_id).cloned()
    };
    let Some(sender) = sender else {
        return;
    };

    let should_send_chunk_batches = !emit_region_patch_only || patch_fallback_to_chunks;
    let mut should_send_clock_snapshot =
        patch_fallback_to_chunks && !region_clock_snapshot.is_empty();

    if !region_patches.is_empty() {
        for patch in region_patches {
            let _ = sender.send(ServerMessage::WorldRegionPatch { patch });
        }
    } else if !patch_fallback_to_chunks {
        should_send_clock_snapshot = !region_clock_snapshot.is_empty();
    }

    if should_send_clock_snapshot {
        let _ = sender.send(ServerMessage::WorldRegionClockUpdate {
            updates: region_clock_snapshot,
        });
    }

    if should_send_chunk_batches && !chunk_loads.is_empty() {
        let mut batch = Vec::with_capacity(STREAM_MESSAGE_CHUNK_LIMIT);
        for chunk in chunk_loads {
            batch.push(chunk);
            if batch.len() == STREAM_MESSAGE_CHUNK_LIMIT {
                let _ = sender.send(ServerMessage::WorldChunkBatch {
                    revision,
                    chunks: std::mem::take(&mut batch),
                });
            }
        }
        if !batch.is_empty() {
            let _ = sender.send(ServerMessage::WorldChunkBatch {
                revision,
                chunks: batch,
            });
        }
    }

    if should_send_chunk_batches && !chunk_unloads.is_empty() {
        let mut batch = Vec::with_capacity(STREAM_MESSAGE_CHUNK_LIMIT);
        for chunk in chunk_unloads {
            batch.push(chunk);
            if batch.len() == STREAM_MESSAGE_CHUNK_LIMIT {
                let _ = sender.send(ServerMessage::WorldChunkUnloadBatch {
                    revision,
                    chunks: std::mem::take(&mut batch),
                });
            }
        }
        if !batch.is_empty() {
            let _ = sender.send(ServerMessage::WorldChunkUnloadBatch {
                revision,
                chunks: batch,
            });
        }
    }
}

fn send_region_resync_patch_to_client(
    state: &SharedState,
    client_id: u64,
    request: RegionResyncRequest,
) {
    let response = {
        let mut guard = state.lock().expect("server state lock poisoned");
        if !guard.players.contains_key(&client_id) {
            None
        } else {
            let entries = sanitize_region_resync_entries(request.regions);
            let Some(bounds) = chunk_bounds_for_resync_entries(&entries) else {
                return;
            };

            let (known_before, mut next_patch_seq) = {
                let Some(player) = guard.players.get_mut(&client_id) else {
                    return;
                };
                for entry in &entries {
                    player
                        .known_region_clocks
                        .insert(entry.region_id, entry.client_clock);
                }
                (
                    player.known_region_clocks.clone(),
                    player.next_stream_patch_seq,
                )
            };

            let current_region_clocks =
                collect_region_clocks_in_chunk_bounds(&guard.region_clocks, bounds);
            match build_region_patches_for_bounds(
                &guard.world,
                bounds,
                &current_region_clocks,
                &known_before,
                &mut next_patch_seq,
            ) {
                Some((patches, known_after)) => {
                    if let Some(player) = guard.players.get_mut(&client_id) {
                        player.known_region_clocks = known_after;
                        player.next_stream_patch_seq = next_patch_seq;
                    }
                    Some(Ok(patches))
                }
                None => Some(Err(format!(
                    "region resync patch exceeded wire budget after spatial splitting in bounds {:?}->{:?}",
                    bounds.min, bounds.max
                ))),
            }
        }
    };

    match response {
        Some(Ok(patches)) => {
            for patch in patches {
                send_to_client(state, client_id, ServerMessage::WorldRegionPatch { patch });
            }
        }
        Some(Err(message)) => {
            send_to_client(state, client_id, ServerMessage::Error { message });
        }
        None => {}
    }
}

fn apply_authoritative_voxel_edit(
    state: &mut ServerState,
    position: [i32; 4],
    material: VoxelType,
) -> Option<ChunkPos> {
    let Some(chunk_pos) = state.world.apply_voxel_edit(position, material) else {
        return None;
    };
    mark_block_region_dirty(state, chunk_pos);
    Some(chunk_pos)
}

fn apply_creeper_explosion(
    state: &mut ServerState,
    source_entity_id: u64,
    center: [f32; 4],
    radius_voxels: i32,
) -> (Vec<QueuedWorldVoxelSet>, QueuedExplosionEvent) {
    let radius = radius_voxels.max(1);
    let radius_sq = radius * radius;
    let center_voxel = [
        center[0].floor() as i32,
        center[1].floor() as i32,
        center[2].floor() as i32,
        center[3].floor() as i32,
    ];
    let blast_centers = [
        center_voxel,
        [
            center_voxel[0],
            center_voxel[1] - 1,
            center_voxel[2],
            center_voxel[3],
        ],
    ];

    let mut changed_positions = HashSet::new();
    let mut changed_chunks = HashSet::new();
    for blast_center in blast_centers {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    for dw in -radius..=radius {
                        let dist_sq = dx * dx + dy * dy + dz * dz + dw * dw;
                        if dist_sq > radius_sq {
                            continue;
                        }
                        let pos = [
                            blast_center[0] + dx,
                            blast_center[1] + dy,
                            blast_center[2] + dz,
                            blast_center[3] + dw,
                        ];
                        if let Some(chunk_pos) =
                            apply_authoritative_voxel_edit(state, pos, VoxelType::AIR)
                        {
                            changed_positions.insert(pos);
                            changed_chunks.insert(chunk_pos);
                        }
                    }
                }
            }
        }
    }

    let mut voxel_sets = Vec::new();
    if !changed_positions.is_empty() {
        let clock_updates = bump_region_clocks_for_chunks(state, changed_chunks);
        let revision = advance_world_revision(state);
        let mut sorted_positions: Vec<[i32; 4]> = changed_positions.into_iter().collect();
        sorted_positions.sort_unstable_by_key(|pos| (pos[3], pos[2], pos[1], pos[0]));
        let mut pending_clock_updates = Some(clock_updates);
        voxel_sets.reserve(sorted_positions.len());
        for pos in sorted_positions {
            voxel_sets.push(QueuedWorldVoxelSet {
                position: pos,
                material: VoxelType::AIR.0,
                source_client_id: None,
                client_edit_id: None,
                revision,
                clock_updates: pending_clock_updates
                    .take()
                    .filter(|updates| !updates.is_empty()),
            });
        }
    }

    (
        voxel_sets,
        QueuedExplosionEvent {
            position: center,
            radius: radius as f32,
            source_entity_id: Some(source_entity_id),
        },
    )
}

fn apply_explosion_impulse(
    state: &mut ServerState,
    source_entity_id: u64,
    center: [f32; 4],
    impulse_radius: f32,
    max_push_distance: f32,
    now_ms: u64,
) -> (bool, Vec<QueuedPlayerMovementModifier>) {
    if !impulse_radius.is_finite()
        || !max_push_distance.is_finite()
        || impulse_radius <= 0.0
        || max_push_distance <= 0.0
    {
        return (false, Vec::new());
    }
    let impulse_radius_sq = impulse_radius * impulse_radius;
    let mut pending_impulses = Vec::new();
    let mut queued_player_modifiers = Vec::new();
    for record in state.entity_records.values() {
        if record.lifecycle != EntityLifecycle::Live || record.entity_id == source_entity_id {
            continue;
        }
        let Some(snapshot) = state.entity_store.snapshot(record.entity_id) else {
            continue;
        };
        let offset = [
            snapshot.position[0] - center[0],
            snapshot.position[1] - center[1],
            snapshot.position[2] - center[2],
            snapshot.position[3] - center[3],
        ];
        let distance_sq = offset[0] * offset[0]
            + offset[1] * offset[1]
            + offset[2] * offset[2]
            + offset[3] * offset[3];
        if !distance_sq.is_finite() || distance_sq > impulse_radius_sq {
            continue;
        }
        let distance = distance_sq.sqrt();
        let falloff = (1.0 - distance / impulse_radius).clamp(0.0, 1.0);
        if falloff <= 0.0 {
            continue;
        }
        let push_distance = max_push_distance * falloff * falloff;
        if push_distance <= 1e-3 {
            continue;
        }
        let outward = if distance > 1e-4 {
            [
                offset[0] / distance,
                offset[1] / distance,
                offset[2] / distance,
                offset[3] / distance,
            ]
        } else {
            normalize4_or_default(snapshot.orientation, [1.0, 0.0, 0.0, 0.0])
        };
        let next_position = [
            snapshot.position[0] + outward[0] * push_distance,
            snapshot.position[1] + outward[1] * push_distance,
            snapshot.position[2] + outward[2] * push_distance,
            snapshot.position[3] + outward[3] * push_distance,
        ];
        let next_orientation = normalize4_or_default(outward, snapshot.orientation);
        pending_impulses.push((
            record.entity_id,
            next_position,
            next_orientation,
            record.persistent,
        ));
    }
    for (&client_id, player) in &state.players {
        if player.entity_id == source_entity_id {
            continue;
        }
        let Some(snapshot) = state.entity_store.snapshot(player.entity_id) else {
            continue;
        };
        let offset = [
            snapshot.position[0] - center[0],
            snapshot.position[1] - center[1],
            snapshot.position[2] - center[2],
            snapshot.position[3] - center[3],
        ];
        let distance_sq = offset[0] * offset[0]
            + offset[1] * offset[1]
            + offset[2] * offset[2]
            + offset[3] * offset[3];
        if !distance_sq.is_finite() || distance_sq > impulse_radius_sq {
            continue;
        }
        let distance = distance_sq.sqrt();
        let falloff = (1.0 - distance / impulse_radius).clamp(0.0, 1.0);
        if falloff <= 0.0 {
            continue;
        }
        let push_distance = max_push_distance * falloff * falloff;
        if push_distance <= 1e-3 {
            continue;
        }
        let outward = if distance > 1e-4 {
            [
                offset[0] / distance,
                offset[1] / distance,
                offset[2] / distance,
                offset[3] / distance,
            ]
        } else {
            normalize4_or_default(snapshot.orientation, [1.0, 0.0, 0.0, 0.0])
        };
        let delta_position = [
            outward[0] * push_distance,
            outward[1] * push_distance,
            outward[2] * push_distance,
            outward[3] * push_distance,
        ];
        let delta_velocity_y = (outward[1] * push_distance * 6.0).clamp(-18.0, 18.0);
        queued_player_modifiers.push(QueuedPlayerMovementModifier {
            client_id,
            delta_position,
            delta_velocity_y,
            source_entity_id: Some(source_entity_id),
        });
    }
    if queued_player_modifiers.is_empty() && !state.players.is_empty() {
        eprintln!(
            "[impulse-debug][server] explosion produced no player modifiers source_entity={} center={:?} players_connected={}",
            source_entity_id,
            center,
            state.players.len()
        );
    }

    let mut persistent_motion = false;
    for (entity_id, next_position, next_orientation, persistent) in pending_impulses {
        if !state
            .entity_store
            .set_motion_state(entity_id, next_position, next_orientation, now_ms)
        {
            continue;
        }
        if persistent {
            persistent_motion = true;
        }
    }
    (persistent_motion, queued_player_modifiers)
}

fn directory_is_empty(path: &std::path::Path) -> io::Result<bool> {
    Ok(std::fs::read_dir(path)?.next().is_none())
}

fn load_or_init_world_state(
    root: &std::path::Path,
    world_seed: u64,
) -> io::Result<save_v3::LoadedState> {
    if root.exists() {
        if root.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "legacy .v4dw world file '{}' is unsupported; use worldgen-cli migration to v3",
                    root.display()
                ),
            ));
        }
        if !save_v3::is_v3_save_root(root) && !directory_is_empty(root)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "save root '{}' is not a v3 world directory (missing manifest.json)",
                    root.display()
                ),
            ));
        }
    }

    save_v3::load_or_init_state(
        root,
        BaseWorldKind::FlatFloor {
            material: VoxelType(11),
        },
        world_seed,
        save_v3::now_unix_ms(),
    )
}

fn build_world_snapshot_payload(state: &ServerState) -> io::Result<WorldSnapshotPayload> {
    let mut bytes = Vec::new();
    let legacy_world = state.world.to_legacy_world();
    voxel::save_world(&legacy_world, &mut bytes)?;
    Ok(WorldSnapshotPayload {
        format: "v4dw".to_string(),
        non_empty_chunks: state.world.non_empty_chunk_count(),
        revision: state.world_revision,
        bytes,
    })
}

fn start_broadcast_thread(
    state: SharedState,
    tick_hz: f32,
    entity_sim_hz: f32,
    entity_interest_radius_chunks: i32,
    start: Instant,
    shutdown: Arc<AtomicBool>,
) {
    let interval = Duration::from_secs_f64(1.0 / tick_hz.max(0.1) as f64);
    let entity_sim_step_ms = (1000.0 / entity_sim_hz.max(0.1) as f64).round().max(1.0) as u64;
    let entity_interest_radius_sq = {
        let radius = entity_interest_radius_chunks.max(0) as i64;
        radius * radius
    };
    let mut next_entity_sim_ms = 0u64;
    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let tick_cpu_start = Instant::now();
            let now = monotonic_ms(start);
            let (entity_batches, world_voxel_updates, explosion_events, player_movement_modifiers) = {
                let mut guard = state.lock().expect("server state lock poisoned");
                let (world_voxel_updates, explosion_events, player_movement_modifiers) =
                    tick_entity_simulation_window(
                        &mut guard,
                        now,
                        &mut next_entity_sim_ms,
                        entity_sim_step_ms,
                    );
                let entity_batches =
                    build_entity_replication_batches(&mut guard, entity_interest_radius_sq);
                (
                    entity_batches,
                    world_voxel_updates,
                    explosion_events,
                    player_movement_modifiers,
                )
            };
            let spawned_count: usize = entity_batches.iter().map(|batch| batch.spawned.len()).sum();
            let transform_count: usize = entity_batches
                .iter()
                .map(|batch| batch.transforms.len())
                .sum();
            let world_voxel_update_count = world_voxel_updates.len();
            let explosion_count = explosion_events.len();
            let player_modifier_count = player_movement_modifiers.len();
            let did_broadcast = entity_batches.iter().any(|batch| {
                !batch.spawned.is_empty()
                    || !batch.despawned.is_empty()
                    || !batch.transforms.is_empty()
            }) || world_voxel_update_count > 0
                || explosion_count > 0
                || player_modifier_count > 0;

            for update in world_voxel_updates {
                if let Some(clock_updates) = update.clock_updates {
                    send_region_clock_updates_to_clients(&state, clock_updates);
                }
                send_world_voxel_set_to_streamed_clients(
                    &state,
                    update.position,
                    update.material,
                    update.source_client_id,
                    update.client_edit_id,
                    update.revision,
                );
            }
            for explosion in explosion_events {
                broadcast(
                    &state,
                    ServerMessage::Explosion {
                        position: explosion.position,
                        radius: explosion.radius,
                        source_entity_id: explosion.source_entity_id,
                    },
                );
            }
            for modifier in player_movement_modifiers {
                let delta = modifier.delta_position;
                let delta_len = (delta[0] * delta[0]
                    + delta[1] * delta[1]
                    + delta[2] * delta[2]
                    + delta[3] * delta[3])
                    .sqrt();
                eprintln!(
                    "[impulse-debug][server] send_modifier client_id={} source_entity={:?} delta={:?} delta_len={:.3} delta_velocity_y={:.3}",
                    modifier.client_id,
                    modifier.source_entity_id,
                    modifier.delta_position,
                    delta_len,
                    modifier.delta_velocity_y
                );
                send_to_client(
                    &state,
                    modifier.client_id,
                    ServerMessage::PlayerMovementModifier {
                        delta_position: modifier.delta_position,
                        delta_velocity_y: modifier.delta_velocity_y,
                        source_entity_id: modifier.source_entity_id,
                    },
                );
            }

            for batch in entity_batches {
                for entity_id in batch.despawned {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntityDestroyed { entity_id },
                    );
                }
                for entity in batch.spawned {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntitySpawned { entity },
                    );
                }
                if !batch.transforms.is_empty() {
                    send_to_client(
                        &state,
                        batch.client_id,
                        ServerMessage::EntityTransforms {
                            server_time_ms: now,
                            entities: batch.transforms,
                        },
                    );
                }
            }
            if did_broadcast {
                record_server_cpu_sample(
                    &state,
                    None,
                    Some((
                        tick_cpu_start.elapsed(),
                        spawned_count.saturating_add(explosion_count),
                        transform_count
                            .saturating_add(world_voxel_update_count)
                            .saturating_add(player_modifier_count),
                    )),
                );
            }
        }
    });
}

fn start_autosave_thread(
    state: SharedState,
    world_file: PathBuf,
    save_interval_secs: u64,
    shutdown: Arc<AtomicBool>,
) {
    if save_interval_secs == 0 {
        return;
    }

    thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(save_interval_secs));
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            let save_result = {
                let mut guard = state.lock().expect("server state lock poisoned");
                let now_ms = save_v3::now_unix_ms();
                let players = collect_player_records(&guard, now_ms);
                let should_persist_players = !players.is_empty()
                    && now_ms.saturating_sub(guard.last_players_persisted_ms)
                        >= PLAYER_PERSIST_INTERVAL_MS;
                let should_save =
                    guard.world.any_dirty() || guard.entities_dirty || should_persist_players;
                if !should_save {
                    None
                } else {
                    let legacy_world = guard.world.to_legacy_world();
                    let persisted_entities = collect_persisted_entities(&guard, now_ms);
                    let dirty_block_regions = guard.dirty_block_regions.clone();
                    let dirty_entity_regions = HashSet::new();
                    let force_full_blocks =
                        guard.world.any_dirty() && dirty_block_regions.is_empty();
                    let force_full_entities = guard.entities_dirty;
                    Some(
                        save_v3::save_state(
                            &world_file,
                            SaveRequest {
                                world: &legacy_world,
                                entities: &persisted_entities,
                                players: &players,
                                world_seed: guard.world.world_seed(),
                                next_entity_id: guard.next_object_id,
                                dirty_block_regions: &dirty_block_regions,
                                dirty_entity_regions: &dirty_entity_regions,
                                force_full_blocks,
                                force_full_entities,
                                now_ms,
                            },
                        )
                        .map(|save_result| {
                            let chunk_count = guard.world.non_empty_chunk_count();
                            guard.entities_dirty = false;
                            guard.dirty_block_regions.clear();
                            guard.world.clear_dirty();
                            if !players.is_empty() {
                                guard.last_players_persisted_ms = now_ms;
                            }
                            (
                                save_result,
                                chunk_count,
                                persisted_entities.len(),
                                players.len(),
                                guard.world_revision,
                                guard.entity_revision,
                            )
                        }),
                    )
                }
            };

            match save_result {
                Some(Ok((
                    save_result,
                    chunk_count,
                    entity_count,
                    player_count,
                    world_revision,
                    entity_revision,
                ))) => {
                    eprintln!(
                        "autosave: generation {} world_rev={} entity_rev={} regions(block={}, entity={}) world_chunks={} entities={} players={} -> {}",
                        save_result.generation,
                        world_revision,
                        entity_revision,
                        save_result.saved_block_regions,
                        save_result.saved_entity_regions,
                        chunk_count,
                        entity_count,
                        player_count,
                        world_file.display()
                    );
                }
                Some(Err(error)) => {
                    eprintln!("autosave failed ({}): {}", world_file.display(), error);
                }
                None => {}
            }
        }
    });
}

fn remove_client(state: &SharedState, client_id: u64) {
    let removed_entity_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let _ = guard.clients.remove(&client_id);
        guard.client_visible_entities.remove(&client_id);
        match guard.players.remove(&client_id) {
            Some(player) => {
                mark_entity_record_despawned(&mut guard, player.entity_id, None);
                let _ = guard.entity_store.despawn(player.entity_id);
                Some(player.entity_id)
            }
            None => None,
        }
    };
    if let Some(entity_id) = removed_entity_id {
        broadcast(state, ServerMessage::EntityDestroyed { entity_id });
    }
}

fn install_or_update_player(
    state: &SharedState,
    client_id: u64,
    name: Option<String>,
    position: Option<[f32; 4]>,
    look: Option<[f32; 4]>,
    start: Instant,
) -> (EntitySnapshot, bool) {
    let now = monotonic_ms(start);
    let mut guard = state.lock().expect("server state lock poisoned");
    let default_orientation = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let mut spawned_now = false;
    let entity_id = match guard.players.entry(client_id) {
        Entry::Occupied(entry) => entry.get().entity_id,
        Entry::Vacant(entry) => {
            spawned_now = true;
            let entity_id = client_id;
            entry.insert(PlayerState {
                entity_id,
                working_set: RegionTreeWorkingSet::new(),
                last_stream_center_chunk: None,
                last_stream_region_clocks: HashMap::new(),
                known_region_clocks: HashMap::new(),
                next_stream_patch_seq: 1,
            });
            guard.entity_store.spawn(
                entity_id,
                EntityClass::Player,
                EntityKind::PlayerAvatar,
                position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
                look.unwrap_or(default_orientation),
                1.0,
                0,
                now,
            );
            entity_id
        }
    };

    if guard.entity_store.snapshot(entity_id).is_none() {
        spawned_now = true;
        guard.entity_store.spawn(
            entity_id,
            EntityClass::Player,
            EntityKind::PlayerAvatar,
            position.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            look.unwrap_or(default_orientation),
            1.0,
            0,
            now,
        );
    }

    let current = guard
        .entity_store
        .snapshot(entity_id)
        .expect("player entity should exist in store");
    let target_position = position.unwrap_or(current.position);
    let target_orientation = look.unwrap_or(current.orientation);
    let _ =
        guard
            .entity_store
            .set_motion_state(entity_id, target_position, target_orientation, now);

    let player_name = name
        .map(|n| sanitize_player_name(&n, client_id))
        .or_else(|| {
            guard
                .entity_records
                .get(&entity_id)
                .and_then(|record| record.display_name.clone())
        })
        .unwrap_or_else(|| format!("player-{client_id}"));

    let mut snapshot = guard
        .entity_store
        .snapshot(entity_id)
        .expect("updated player entity should exist in store");
    snapshot.class = EntityClass::Player;
    snapshot.kind = EntityKind::PlayerAvatar;
    snapshot.scale = 1.0;
    snapshot.material = 0;
    snapshot.owner_client_id = Some(client_id);
    snapshot.display_name = Some(player_name.clone());
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityClass::Player,
        Some(client_id),
        Some(player_name),
        false,
        now,
    );
    (snapshot, spawned_now)
}

fn spawn_entity_from_request(
    state: &SharedState,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    start: Instant,
) -> Result<EntitySnapshot, String> {
    if kind == EntityKind::PlayerAvatar {
        return Err("player entities are server-managed and cannot be spawned".to_string());
    }
    let Some(spec) = spawnable_entity_spec_for_kind(kind) else {
        return Err(format!("entity kind {:?} is not spawnable", kind));
    };

    let spawn_position = [
        if position[0].is_finite() {
            position[0]
        } else {
            0.0
        },
        if position[1].is_finite() {
            position[1]
        } else {
            0.0
        },
        if position[2].is_finite() {
            position[2]
        } else {
            0.0
        },
        if position[3].is_finite() {
            position[3]
        } else {
            0.0
        },
    ];
    let spawn_orientation = if orientation.iter().all(|axis| axis.is_finite()) {
        normalize4_or_default(orientation, [0.0, 0.0, 1.0, 0.0])
    } else {
        [0.0, 0.0, 1.0, 0.0]
    };
    let spawn_scale = if scale.is_finite() {
        scale.clamp(0.10, 8.0)
    } else {
        0.5
    };
    let spawn_material = material.clamp(1, materials::MAX_MATERIAL_ID);

    let entity = match (spec.class, spec.mob_archetype) {
        (EntityClass::Mob, Some(archetype)) => spawn_mob_entity(
            state,
            kind,
            archetype,
            spawn_position,
            spawn_orientation,
            spawn_scale,
            spawn_material,
            None,
            true,
            None,
            None,
            start,
        ),
        (EntityClass::Accent, None) => spawn_entity(
            state,
            kind,
            spawn_position,
            spawn_orientation,
            spawn_scale,
            spawn_material,
            None,
            true,
            None,
            start,
        ),
        (class, archetype) => {
            return Err(format!(
                "spawn spec for {:?} is invalid (class={:?}, archetype={:?})",
                kind, class, archetype
            ));
        }
    };
    Ok(entity)
}

fn handle_console_spawn_command(
    state: &SharedState,
    client_id: u64,
    args: &[&str],
    start: Instant,
) -> Result<(), String> {
    if args.is_empty() {
        return Err(spawn_usage_string());
    }
    let Some(spec) = spawnable_entity_spec_for_token(args[0]) else {
        let available = SPAWNABLE_ENTITY_SPECS
            .iter()
            .map(|spec| spec.canonical_name)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "unknown spawn kind '{}'; available kinds: {available}",
            args[0]
        ));
    };

    let (default_position, default_orientation) = {
        let guard = state.lock().expect("server state lock poisoned");
        default_spawn_pose_for_client(&guard, client_id)
    };
    let parse_usage = || spawn_usage_string();
    let (position, material_id) = match args.len() {
        1 => (default_position, spec.default_material),
        2 => {
            let Some(material_id) = parse_spawn_material_id(args[1]) else {
                return Err(format!("unknown material '{}'", args[1]));
            };
            (default_position, material_id)
        }
        5 => {
            let Some(position) = parse_spawn_vec4(&args[1..5]) else {
                return Err(parse_usage());
            };
            (position, spec.default_material)
        }
        6 => {
            let Some(position) = parse_spawn_vec4(&args[1..5]) else {
                return Err(parse_usage());
            };
            let Some(material_id) = parse_spawn_material_id(args[5]) else {
                return Err(format!("unknown material '{}'", args[5]));
            };
            (position, material_id)
        }
        _ => return Err(parse_usage()),
    };

    let _ = spawn_entity_from_request(
        state,
        spec.kind,
        position,
        default_orientation,
        spec.default_scale,
        material_id,
        start,
    )?;
    Ok(())
}

fn run_server_console_command(
    state: &SharedState,
    client_id: u64,
    command: &str,
    start: Instant,
) -> Result<(), String> {
    let mut command = command.trim();
    if command.is_empty() {
        return Ok(());
    }
    if let Some(stripped) = command.strip_prefix('/') {
        command = stripped;
    }
    let mut parts = command.split_whitespace();
    let Some(command_name) = parts.next() else {
        return Ok(());
    };
    let args: Vec<&str> = parts.collect();

    if command_name.eq_ignore_ascii_case("spawn") {
        return handle_console_spawn_command(state, client_id, &args, start);
    }
    Err(format!(
        "unknown server command '{}'; supported: /spawn",
        command_name
    ))
}

fn handle_message(
    state: &SharedState,
    client_id: u64,
    message: ClientMessage,
    snapshot_on_join: bool,
    tick_hz: f32,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    start: Instant,
) {
    let message_cpu_start = Instant::now();
    match message {
        ClientMessage::Hello { name } => {
            let (snapshot, _spawned_now) =
                install_or_update_player(state, client_id, Some(name), None, None, start);
            send_to_client(
                state,
                client_id,
                ServerMessage::Welcome {
                    client_id,
                    server_time_ms: monotonic_ms(start),
                    tick_hz,
                    world: {
                        let guard = state.lock().expect("server state lock poisoned");
                        WorldSummary {
                            non_empty_chunks: guard.world.non_empty_chunk_count(),
                            revision: guard.world_revision,
                        }
                    },
                },
            );
            if snapshot_on_join {
                let snapshot_message = {
                    let guard = state.lock().expect("server state lock poisoned");
                    build_world_snapshot_payload(&guard)
                };
                match snapshot_message {
                    Ok(world) => {
                        send_to_client(state, client_id, ServerMessage::WorldSnapshot { world });
                    }
                    Err(error) => {
                        send_to_client(
                            state,
                            client_id,
                            ServerMessage::Error {
                                message: format!("failed to build world snapshot: {error}"),
                            },
                        );
                    }
                }
            }
            let center_chunk = world_chunk_from_position(snapshot.position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                true,
            );
        }
        ClientMessage::UpdatePlayer { position, look } => {
            let safe_position = if position.iter().all(|axis| axis.is_finite()) {
                position
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            let safe_look = normalize4_or_default(
                look,
                [
                    0.0,
                    0.0,
                    std::f32::consts::FRAC_1_SQRT_2,
                    std::f32::consts::FRAC_1_SQRT_2,
                ],
            );
            let (_snapshot, _spawned_now) = install_or_update_player(
                state,
                client_id,
                None,
                Some(safe_position),
                Some(safe_look),
                start,
            );
            let center_chunk = world_chunk_from_position(safe_position);
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                false,
            );
        }
        ClientMessage::SetVoxel {
            position,
            material,
            client_edit_id,
        } => {
            let requested_voxel = VoxelType(material);
            let maybe_update = {
                let mut guard = state.lock().expect("server state lock poisoned");
                if let Some(chunk_pos) =
                    apply_authoritative_voxel_edit(&mut guard, position, requested_voxel)
                {
                    let clock_updates =
                        bump_region_clocks_for_chunks(&mut guard, std::iter::once(chunk_pos));
                    let revision = advance_world_revision(&mut guard);
                    Some((revision, clock_updates))
                } else {
                    None
                }
            };

            if let Some((revision, clock_updates)) = maybe_update {
                send_region_clock_updates_to_clients(state, clock_updates);
                send_world_voxel_set_to_streamed_clients(
                    state,
                    position,
                    material,
                    Some(client_id),
                    client_edit_id,
                    revision,
                );
            }
        }
        ClientMessage::SpawnEntity {
            kind,
            position,
            orientation,
            scale,
            material,
        } => {
            if let Err(message) = spawn_entity_from_request(
                state,
                kind,
                position,
                orientation,
                scale,
                material,
                start,
            ) {
                send_to_client(state, client_id, ServerMessage::Error { message });
            }
        }
        ClientMessage::ConsoleCommand { command } => {
            if let Err(message) = run_server_console_command(state, client_id, &command, start) {
                send_to_client(state, client_id, ServerMessage::Error { message });
            }
        }
        ClientMessage::WorldRegionResyncRequest { request } => {
            send_region_resync_patch_to_client(state, client_id, request);
        }
        ClientMessage::RequestWorldSnapshot => {
            let snapshot_message = {
                let guard = state.lock().expect("server state lock poisoned");
                build_world_snapshot_payload(&guard)
            };
            match snapshot_message {
                Ok(world) => {
                    send_to_client(state, client_id, ServerMessage::WorldSnapshot { world })
                }
                Err(error) => send_to_client(
                    state,
                    client_id,
                    ServerMessage::Error {
                        message: format!("failed to build world snapshot: {error}"),
                    },
                ),
            }
            let center_chunk = {
                let guard = state.lock().expect("server state lock poisoned");
                guard
                    .players
                    .get(&client_id)
                    .and_then(|player| guard.entity_store.snapshot(player.entity_id))
                    .map(|snapshot| world_chunk_from_position(snapshot.position))
                    .unwrap_or([0, 0, 0, 0])
            };
            sync_streamed_chunks_for_client(
                state,
                client_id,
                center_chunk,
                near_chunk_radius,
                mid_chunk_radius,
                far_chunk_radius,
                true,
            );
        }
        ClientMessage::Ping { nonce } => {
            send_to_client(state, client_id, ServerMessage::Pong { nonce });
        }
    }
    record_server_cpu_sample(state, Some(message_cpu_start.elapsed()), None);
}

fn spawn_client_thread(
    stream: TcpStream,
    state: SharedState,
    snapshot_on_join: bool,
    tick_hz: f32,
    near_chunk_radius: i32,
    mid_chunk_radius: i32,
    far_chunk_radius: i32,
    start: Instant,
) {
    let peer_label = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let (tx, rx) = mpsc::channel::<ServerMessage>();

    let client_id = {
        let mut guard = state.lock().expect("server state lock poisoned");
        let id = allocate_server_object_id(&mut guard);
        guard.clients.insert(id, tx.clone());
        id
    };

    let writer_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(error) => {
            eprintln!("failed to clone stream for {}: {}", peer_label, error);
            remove_client(&state, client_id);
            return;
        }
    };

    thread::spawn(move || {
        let mut writer = BufWriter::new(writer_stream);
        while let Ok(message) = rx.recv() {
            let Ok(encoded) = postcard::to_stdvec(&message) else {
                continue;
            };
            let len = (encoded.len() as u32).to_le_bytes();
            if writer.write_all(&len).is_err() {
                break;
            }
            if writer.write_all(&encoded).is_err() {
                break;
            }
            if writer.flush().is_err() {
                break;
            }
        }
    });

    thread::spawn(move || {
        eprintln!("client {} connected from {}", client_id, peer_label);
        let mut reader = stream;
        let mut len_buf = [0u8; 4];

        loop {
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                    break;
                }
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let len = u32::from_le_bytes(len_buf) as usize;
            if len > 100_000_000 {
                eprintln!(
                    "client {} sent oversized message ({} bytes)",
                    client_id, len
                );
                break;
            }

            let mut msg_buf = vec![0u8; len];
            match reader.read_exact(&mut msg_buf) {
                Ok(()) => {}
                Err(error) => {
                    eprintln!("read error for client {}: {}", client_id, error);
                    break;
                }
            }

            let parsed = postcard::from_bytes::<ClientMessage>(&msg_buf);
            match parsed {
                Ok(message) => {
                    handle_message(
                        &state,
                        client_id,
                        message,
                        snapshot_on_join,
                        tick_hz,
                        near_chunk_radius,
                        mid_chunk_radius,
                        far_chunk_radius,
                        start,
                    );
                }
                Err(error) => {
                    send_to_client(
                        &state,
                        client_id,
                        ServerMessage::Error {
                            message: format!("invalid message: {error}"),
                        },
                    );
                }
            }
        }

        remove_client(&state, client_id);
        eprintln!("client {} disconnected", client_id);
    });
}

fn spawn_entity(
    state: &SharedState,
    kind: EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    persistent: bool,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    let entity_id = guard.entity_store.spawn(
        allocated_id,
        EntityClass::Accent,
        kind,
        position,
        orientation,
        scale,
        material,
        now_ms,
    );
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityClass::Accent,
        None,
        display_name,
        persistent,
        now_ms,
    );
    if persistent {
        mark_entities_dirty(&mut guard);
    }
    guard
        .entity_store
        .snapshot(entity_id)
        .expect("spawned entity should exist in store")
}

fn spawn_mob_entity(
    state: &SharedState,
    kind: EntityKind,
    archetype: MobArchetype,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    display_name: Option<String>,
    persistent: bool,
    persisted_mob: Option<PersistedMobEntry>,
    persisted_entity_id: Option<EntityId>,
    start: Instant,
) -> EntitySnapshot {
    let mut guard = state.lock().expect("server state lock poisoned");
    let allocated_id = allocate_or_reserve_server_object_id(&mut guard, persisted_entity_id);
    let now_ms = monotonic_ms(start);
    let entity_id = guard.entity_store.spawn(
        allocated_id,
        EntityClass::Mob,
        kind,
        position,
        orientation,
        scale,
        material,
        now_ms,
    );
    let (default_speed, default_distance, default_tangent) = mob_archetype_defaults(archetype);
    let phase_offset = persisted_mob
        .as_ref()
        .map(|mob| mob.phase_offset)
        .unwrap_or_else(|| ((entity_id as f32) * 0.73).rem_euclid(std::f32::consts::TAU));
    let move_speed = persisted_mob
        .as_ref()
        .map(|mob| mob.move_speed)
        .unwrap_or(default_speed)
        .max(0.1);
    let preferred_distance = persisted_mob
        .as_ref()
        .map(|mob| mob.preferred_distance)
        .unwrap_or(default_distance)
        .max(0.1);
    let tangent_weight = persisted_mob
        .as_ref()
        .map(|mob| mob.tangent_weight)
        .unwrap_or(default_tangent)
        .clamp(0.0, 2.0);
    let initial_phase_tick_seed = (entity_id as u32).wrapping_mul(7477);
    let next_phase_ms =
        phase_spider_next_phase_deadline(now_ms, phase_offset, initial_phase_tick_seed);
    guard.mobs.insert(
        entity_id,
        MobState {
            entity_id,
            archetype,
            phase_offset,
            move_speed,
            preferred_distance,
            tangent_weight,
            next_phase_ms,
            navigation: MobNavigationState::default(),
        },
    );
    upsert_entity_record(
        &mut guard,
        entity_id,
        EntityClass::Mob,
        None,
        display_name,
        persistent,
        now_ms,
    );
    if persistent {
        mark_entities_dirty(&mut guard);
    }
    guard
        .entity_store
        .snapshot(entity_id)
        .expect("spawned mob entity should exist in store")
}

fn restore_persisted_entities(
    state: &SharedState,
    entries: Vec<PersistedEntityRecord>,
    start: Instant,
) -> usize {
    let mut restored = 0usize;
    for entry in entries {
        match entry.class {
            EntityClass::Player => {}
            EntityClass::Accent => match entry.kind {
                EntityKind::TestCube | EntityKind::TestRotor | EntityKind::TestDrifter => {
                    let _ = spawn_entity(
                        state,
                        entry.kind,
                        entry.position,
                        entry.orientation,
                        entry.scale,
                        entry.material,
                        entry.display_name,
                        true,
                        Some(entry.entity_id),
                        start,
                    );
                    restored = restored.saturating_add(1);
                }
                EntityKind::PlayerAvatar
                | EntityKind::MobSeeker
                | EntityKind::MobCreeper4d
                | EntityKind::MobPhaseSpider => {}
            },
            EntityClass::Mob => {
                let Some(default_archetype) = mob_archetype_for_kind(entry.kind) else {
                    continue;
                };
                let persisted_mob = decode_persisted_mob_payload(&entry.payload);
                let archetype = persisted_mob
                    .as_ref()
                    .map(|mob| mob.archetype)
                    .unwrap_or(default_archetype);
                let persisted_mob = persisted_mob.map(|mut mob| {
                    mob.archetype = archetype;
                    mob
                });
                let _ = spawn_mob_entity(
                    state,
                    entry.kind,
                    archetype,
                    entry.position,
                    entry.orientation,
                    entry.scale,
                    entry.material,
                    entry.display_name,
                    true,
                    persisted_mob,
                    Some(entry.entity_id),
                    start,
                );
                restored = restored.saturating_add(1);
            }
        }
    }
    restored
}

fn initialize_state(
    config: &RuntimeConfig,
    shutdown: Arc<AtomicBool>,
) -> io::Result<(SharedState, Instant)> {
    let start = Instant::now();
    procgen::clear_runtime_maze_layout_cache();
    let loaded = load_or_init_world_state(&config.world_file, config.world_seed)?;
    let save_generation = loaded.manifest.current_generation;
    let last_persisted_ms = loaded.manifest.last_modified_ms;
    let region_chunk_edge = loaded.manifest.limits.region_chunk_edge.max(1);
    let runtime_world_seed = loaded.global.world_seed;
    let persisted_entities = loaded.entities;
    let persisted_players = loaded.players.players;
    let mut initial_world = ServerWorldField::from_legacy_world(
        loaded.world,
        runtime_world_seed,
        config.procgen_structures,
        HashSet::new(),
    );
    let next_object_id = loaded.global.next_entity_id.max(1);
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
        "loaded v3 world {} (generation {}, {} non-empty chunks, {} persisted entities, {} player records)",
        config.world_file.display(),
        save_generation,
        initial_chunks,
        persisted_entities.len(),
        persisted_players.len(),
    );
    let mob_nav_debug = env_flag_enabled("R4D_MOB_NAV_DEBUG");
    let mob_nav_simple_steer = env_flag_enabled("R4D_MOB_NAV_SIMPLE_STEER");
    if mob_nav_debug {
        eprintln!("mob nav debug logging enabled (R4D_MOB_NAV_DEBUG=1)");
    }
    if mob_nav_simple_steer {
        eprintln!("mob nav simple steering enabled (R4D_MOB_NAV_SIMPLE_STEER=1)");
    }

    let state = Arc::new(Mutex::new(ServerState {
        next_object_id,
        entity_store: EntityStore::new(),
        entity_records: HashMap::new(),
        entities_dirty: false,
        entity_revision: 0,
        dirty_block_regions: HashSet::new(),
        region_chunk_edge,
        world: initial_world,
        world_revision: 0,
        region_clocks: HashMap::new(),
        players: HashMap::new(),
        last_players_persisted_ms: last_persisted_ms,
        mobs: HashMap::new(),
        mob_nav_debug,
        mob_nav_simple_steer,
        clients: HashMap::new(),
        client_visible_entities: HashMap::new(),
        cpu_profile: ServerCpuProfile::new(start),
    }));

    let entity_interest_radius_chunks = config.procgen_far_chunk_radius.max(1)
        * STREAM_FAR_LOD_SCALE
        + ENTITY_INTEREST_RADIUS_PADDING_CHUNKS;
    let restored = restore_persisted_entities(&state, persisted_entities, start);
    eprintln!(
        "loaded {} persisted entities from {}",
        restored,
        config.world_file.display()
    );
    {
        let mut guard = state.lock().expect("server state lock poisoned");
        guard.entities_dirty = false;
        guard.dirty_block_regions.clear();
    }
    start_broadcast_thread(
        state.clone(),
        config.tick_hz,
        config.entity_sim_hz,
        entity_interest_radius_chunks,
        start,
        shutdown.clone(),
    );
    start_autosave_thread(
        state.clone(),
        config.world_file.clone(),
        config.save_interval_secs,
        shutdown,
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
                cfg.snapshot_on_join,
                cfg.tick_hz.max(0.1),
                cfg.procgen_near_chunk_radius.max(0),
                cfg.procgen_mid_chunk_radius.max(1),
                cfg.procgen_far_chunk_radius.max(1),
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
        guard.world.world_seed()
    };

    let listener = TcpListener::bind(&config.bind)?;
    eprintln!(
        "polychora-server listening on {} (tick {:.2} Hz, entity_sim {:.2} Hz, autosave {}s, procgen={}, seed={}, near_radius={} chunks, mid_radius={} chunks, far_radius={} chunks, keepout={}, keepout_padding={} chunks)",
        config.bind,
        config.tick_hz.max(0.1),
        config.entity_sim_hz.max(0.1),
        config.save_interval_secs,
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
                    config.snapshot_on_join,
                    config.tick_hz.max(0.1),
                    config.procgen_near_chunk_radius.max(0),
                    config.procgen_mid_chunk_radius.max(1),
                    config.procgen_far_chunk_radius.max(1),
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
    use super::*;

    #[test]
    fn region_bounds_use_floor_division_for_negative_chunks() {
        let bounds = Aabb4i::new([-5, -1, 0, 7], [3, 2, 0, 7]);
        let region_bounds = region_bounds_for_chunk_bounds(bounds);
        assert_eq!(region_bounds.min, [-2, -1, 0, 1]);
        assert_eq!(region_bounds.max, [0, 0, 0, 1]);
    }

    #[test]
    fn collect_region_clocks_only_returns_intersecting_regions() {
        let mut clocks = RegionClockMap::new();
        clocks.insert(
            RegionId {
                x: -1,
                y: 0,
                z: 0,
                w: 0,
            },
            5,
        );
        clocks.insert(
            RegionId {
                x: 0,
                y: 0,
                z: 0,
                w: 0,
            },
            3,
        );
        clocks.insert(
            RegionId {
                x: 2,
                y: 0,
                z: 0,
                w: 0,
            },
            9,
        );

        let bounds = Aabb4i::new([0, 0, 0, 0], [3, 0, 0, 0]);
        let collected = collect_region_clocks_in_chunk_bounds(&clocks, bounds);
        assert_eq!(collected.len(), 1);
        assert_eq!(
            collected.get(&RegionId {
                x: 0,
                y: 0,
                z: 0,
                w: 0,
            }),
            Some(&3)
        );
    }

    #[test]
    fn stream_near_bounds_expands_symmetrically() {
        let center = [10, -2, 7, 1];
        let bounds = stream_near_bounds(center, 3);
        assert_eq!(bounds.min, [7, -5, 4, -2]);
        assert_eq!(bounds.max, [13, 1, 10, 4]);
    }

    #[test]
    fn split_chunk_bounds_longest_axis_prefers_widest_axis() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [7, 1, 1, 1]);
        let (left, right) = split_chunk_bounds_longest_axis(bounds).expect("split bounds");
        assert_eq!(left.min, [0, 0, 0, 0]);
        assert_eq!(left.max, [3, 1, 1, 1]);
        assert_eq!(right.min, [4, 0, 0, 0]);
        assert_eq!(right.max, [7, 1, 1, 1]);
    }

    #[test]
    fn split_chunk_bounds_longest_axis_rejects_single_cell_bounds() {
        let bounds = Aabb4i::new([2, -1, 5, 0], [2, -1, 5, 0]);
        assert!(split_chunk_bounds_longest_axis(bounds).is_none());
    }

    #[test]
    fn region_clock_preconditions_default_to_zero_for_new_regions() {
        let mut previous = RegionClockMap::new();
        previous.insert(
            RegionId {
                x: -1,
                y: 0,
                z: 0,
                w: 0,
            },
            2,
        );

        let mut current = previous.clone();
        current.insert(
            RegionId {
                x: 0,
                y: 0,
                z: 0,
                w: 0,
            },
            7,
        );

        let preconditions = region_clock_preconditions_for_patch(&previous, &current);
        assert_eq!(preconditions.len(), 2);
        assert_eq!(
            preconditions[0],
            RegionClockPrecondition {
                region_id: RegionId {
                    x: -1,
                    y: 0,
                    z: 0,
                    w: 0
                },
                expected_clock: 2,
            }
        );
        assert_eq!(
            preconditions[1],
            RegionClockPrecondition {
                region_id: RegionId {
                    x: 0,
                    y: 0,
                    z: 0,
                    w: 0
                },
                expected_clock: 0,
            }
        );
    }

    #[test]
    fn next_stream_patch_seq_is_monotonic_and_non_zero() {
        let mut next = 1u64;
        assert_eq!(next_stream_patch_seq(&mut next), 1);
        assert_eq!(next_stream_patch_seq(&mut next), 2);

        next = u64::MAX;
        assert_eq!(next_stream_patch_seq(&mut next), u64::MAX);
        assert_eq!(next, 1);
    }

    #[test]
    fn chunk_bounds_for_resync_entries_unions_region_bounds() {
        let entries = vec![
            RegionResyncEntry {
                region_id: RegionId {
                    x: -1,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                client_clock: 0,
            },
            RegionResyncEntry {
                region_id: RegionId {
                    x: 1,
                    y: 0,
                    z: 0,
                    w: 0,
                },
                client_clock: 2,
            },
        ];
        let bounds = chunk_bounds_for_resync_entries(&entries).expect("bounds");
        assert_eq!(bounds.min, [-4, 0, 0, 0]);
        assert_eq!(bounds.max, [7, 3, 3, 3]);
    }

    #[test]
    fn region_patch_wire_size_bytes_serializes_patch_message() {
        let bounds = Aabb4i::new([0, 0, 0, 0], [0, 0, 0, 0]);
        let patch = RegionSubtreePatch {
            patch_seq: 1,
            bounds,
            preconditions: Vec::new(),
            clock_updates: Vec::new(),
            subtree: crate::shared::worldfield::RegionTreeCore {
                bounds,
                kind: crate::shared::worldfield::RegionNodeKind::Uniform(3),
                generator_version_hash: 0,
            },
        };
        let size = region_patch_wire_size_bytes(&patch).expect("serialized size");
        assert!(size > 0);
        assert!(size <= MAX_PATCH_BYTES);
    }
}
