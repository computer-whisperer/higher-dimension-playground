use crate::entity::MobLocomotionMode;
use serde::{Deserialize, Serialize};

/// Input to OP_MOB_STEERING — host sends this per mob per tick.
#[derive(Serialize, Deserialize, Default)]
pub struct MobSteeringInput {
    pub entity_ns: u32,
    pub entity_type: u32,
    #[serde(default)]
    pub position: [f32; 4],
    #[serde(default)]
    pub target_position: Option<[f32; 4]>,
    #[serde(default)]
    pub path_following: bool,
    #[serde(default)]
    pub simple_steer: bool,
    #[serde(default)]
    pub now_ms: u64,
    #[serde(default)]
    pub phase_offset: f32,
    #[serde(default)]
    pub preferred_distance: f32,
    #[serde(default)]
    pub tangent_weight: f32,
    #[serde(default)]
    pub locomotion: MobLocomotionMode,
}

/// Output from OP_MOB_STEERING.
#[derive(Serialize, Deserialize, Default)]
pub struct MobSteeringOutput {
    #[serde(default)]
    pub desired_direction: [f32; 4],
    #[serde(default)]
    pub speed_factor: f32,
}

/// Input to OP_MOB_SPECIAL_ABILITY — one variant per ability kind.
///
/// Each variant contains only the fields relevant to that ability, avoiding
/// a flat union where callers must ignore inapplicable fields.
#[derive(Serialize, Deserialize)]
pub enum MobAbilityCheck {
    /// Should this mob detonate (e.g. creeper)?
    Detonate {
        entity_ns: u32,
        entity_type: u32,
        #[serde(default)]
        nearest_player_distance: f32,
        #[serde(default)]
        trigger_distance: f32,
    },
    /// Should this mob blink/teleport (e.g. phase spider)?
    Blink {
        entity_ns: u32,
        entity_type: u32,
        #[serde(default)]
        has_target: bool,
        #[serde(default)]
        path_following: bool,
        #[serde(default)]
        now_ms: u64,
        #[serde(default)]
        next_phase_ms: u64,
        #[serde(default)]
        blocked_progress_epsilon: f32,
        #[serde(default)]
        attempted_move_distance: f32,
        #[serde(default)]
        resolved_move_distance: f32,
    },
}

/// Output from OP_MOB_SPECIAL_ABILITY.
#[derive(Serialize, Deserialize, Default)]
pub struct MobAbilityResult {
    #[serde(default)]
    pub should_trigger: bool,
}
