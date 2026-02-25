use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::shared::entity_types::EntityCategory;
use crate::shared::protocol::{EntitySnapshot, EntityTransform};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub(super) struct PlayerState {
    pub(super) entity_id: u64,
}

pub(super) type MobNavCell = [i32; 4];

#[derive(Clone, Debug, Default)]
pub(super) struct MobNavigationState {
    pub(super) goal_cell: Option<MobNavCell>,
    pub(super) path_cells: Vec<MobNavCell>,
    pub(super) path_cursor: usize,
    pub(super) blocked_without_path: bool,
    pub(super) last_repath_ms: u64,
    pub(super) last_debug_log_ms: u64,
    pub(super) last_los_result: bool,
    pub(super) last_los_check_ms: u64,
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
    pub(super) entity_ns: u32,
    pub(super) entity_type: u32,
    pub(super) phase_offset: f32,
    pub(super) move_speed: f32,
    pub(super) preferred_distance: f32,
    pub(super) tangent_weight: f32,
    pub(super) next_phase_ms: u64,
    pub(super) navigation: MobNavigationState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PersistedMobEntry {
    pub(super) entity_ns: u32,
    pub(super) entity_type: u32,
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
    pub(super) category: EntityCategory,
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

#[derive(Clone, Debug)]
pub(super) enum CollisionChunkCacheEntry {
    Explicit(ResolvedChunkPayload),
    ExplicitEmpty,
    Effective(Option<ResolvedChunkPayload>),
}
