use crate::entity::MobLocomotionMode;
use serde::{Deserialize, Serialize};

/// Input to OP_ENTITY_TICK — host sends this per entity per tick.
#[derive(Serialize, Deserialize, Default)]
pub struct EntityTickInput {
    pub entity_ns: u32,
    pub entity_type: u32,
    #[serde(default)]
    pub entity_id: u64,
    #[serde(default)]
    pub position: [f32; 4],
    #[serde(default)]
    pub home_position: [f32; 4],
    #[serde(default)]
    pub scale: f32,
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

/// Output from OP_ENTITY_TICK.
#[derive(Serialize, Deserialize)]
pub enum EntityTickOutput {
    /// Engine applies collision, gravity, locomotion from the desired direction.
    Steer {
        #[serde(default)]
        desired_direction: [f32; 4],
        #[serde(default)]
        speed_factor: f32,
    },
    /// Engine directly sets pose — no physics.
    SetPose {
        #[serde(default)]
        position: [f32; 4],
        #[serde(default)]
        orientation: [f32; 4],
        #[serde(default)]
        scale: f32,
    },
}

impl Default for EntityTickOutput {
    fn default() -> Self {
        Self::Steer {
            desired_direction: [0.0; 4],
            speed_factor: 0.0,
        }
    }
}

/// Input to OP_ENTITY_ABILITY — one variant per ability kind.
///
/// Each variant contains only the fields relevant to that ability, avoiding
/// a flat union where callers must ignore inapplicable fields.
#[derive(Serialize, Deserialize)]
pub enum EntityAbilityCheck {
    /// Should this entity detonate (e.g. creeper)?
    Detonate {
        entity_ns: u32,
        entity_type: u32,
        #[serde(default)]
        nearest_player_distance: f32,
        #[serde(default)]
        trigger_distance: f32,
    },
    /// Should this entity blink/teleport (e.g. phase spider)?
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

/// Output from OP_ENTITY_ABILITY.
#[derive(Serialize, Deserialize, Default)]
pub struct EntityAbilityResult {
    #[serde(default)]
    pub should_trigger: bool,
}
