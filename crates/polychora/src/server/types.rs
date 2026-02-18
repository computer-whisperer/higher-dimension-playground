use crate::shared::protocol::{EntityClass, EntityKind, EntitySnapshot, EntityTransform};
use crate::shared::voxel::{ChunkPos, VoxelType, CHUNK_VOLUME};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub(super) struct PlayerState {
    pub(super) entity_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum MobArchetype {
    Seeker,
    Creeper4d,
    PhaseSpider,
}

pub(super) type MobNavCell = [i32; 4];

#[derive(Clone, Debug, Default)]
pub(super) struct MobNavigationState {
    pub(super) goal_cell: Option<MobNavCell>,
    pub(super) path_cells: Vec<MobNavCell>,
    pub(super) path_cursor: usize,
    pub(super) last_repath_ms: u64,
    pub(super) last_debug_log_ms: u64,
}

#[derive(Clone, Debug)]
pub(super) struct MobNavPathResult {
    pub(super) path_cells: Vec<MobNavCell>,
    pub(super) reached_goal: bool,
    pub(super) expanded_steps: usize,
    pub(super) best_cell: MobNavCell,
    pub(super) best_goal_distance: i32,
}

#[derive(Clone, Debug)]
pub(super) struct MobState {
    pub(super) entity_id: u64,
    pub(super) archetype: MobArchetype,
    pub(super) phase_offset: f32,
    pub(super) move_speed: f32,
    pub(super) preferred_distance: f32,
    pub(super) tangent_weight: f32,
    pub(super) next_phase_ms: u64,
    pub(super) navigation: MobNavigationState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PersistedMobEntry {
    pub(super) archetype: MobArchetype,
    pub(super) phase_offset: f32,
    pub(super) move_speed: f32,
    pub(super) preferred_distance: f32,
    pub(super) tangent_weight: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum EntityLifecycle {
    Live,
    Despawned,
}

#[derive(Clone, Debug)]
pub(super) struct EntityRecord {
    pub(super) entity_id: u64,
    pub(super) class: EntityClass,
    pub(super) owner_client_id: Option<u64>,
    pub(super) display_name: Option<String>,
    pub(super) persistent: bool,
    pub(super) spawned_at_ms: u64,
    pub(super) lifecycle: EntityLifecycle,
    pub(super) despawned_at_ms: Option<u64>,
}

#[derive(Copy, Clone, Debug, Default)]
pub(super) struct EntityRecordSummary {
    pub(super) live_total: usize,
    pub(super) live_players: usize,
    pub(super) live_accents: usize,
    pub(super) live_mobs: usize,
    pub(super) live_persistent: usize,
    pub(super) live_owned: usize,
    pub(super) tombstones: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct LiveReplicationFrame {
    pub(super) player_entities: Vec<EntitySnapshot>,
    pub(super) player_chunks: Vec<(u64, [i32; 4])>,
    pub(super) non_player_entities: Vec<(EntitySnapshot, [i32; 4])>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ClientEntityReplicationBatch {
    pub(super) client_id: u64,
    pub(super) spawned: Vec<EntitySnapshot>,
    pub(super) despawned: Vec<u64>,
    pub(super) transforms: Vec<EntityTransform>,
}

#[derive(Clone, Debug)]
pub(super) struct QueuedWorldChunkUpdate {
    pub(super) changed_chunks: Vec<ChunkPos>,
}

#[derive(Clone, Debug)]
pub(super) struct QueuedExplosionEvent {
    pub(super) position: [f32; 4],
    pub(super) radius: f32,
    pub(super) source_entity_id: Option<u64>,
}

#[derive(Clone, Debug)]
pub(super) struct QueuedPlayerMovementModifier {
    pub(super) client_id: u64,
    pub(super) delta_position: [f32; 4],
    pub(super) delta_velocity_y: f32,
    pub(super) source_entity_id: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SpawnableEntitySpec {
    pub(super) kind: EntityKind,
    pub(super) class: EntityClass,
    pub(super) mob_archetype: Option<MobArchetype>,
    pub(super) canonical_name: &'static str,
    pub(super) aliases: &'static [&'static str],
    pub(super) default_scale: f32,
    pub(super) default_material: u8,
}

pub(super) const SPAWNABLE_ENTITY_SPECS: &[SpawnableEntitySpec] = &[
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
pub(super) enum CollisionChunkCacheEntry {
    Explicit([VoxelType; CHUNK_VOLUME]),
    ExplicitEmpty,
    Effective(Option<[VoxelType; CHUNK_VOLUME]>),
}
