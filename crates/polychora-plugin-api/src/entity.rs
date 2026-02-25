use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// High-level entity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityCategory {
    Player,
    Accent,
    Mob,
}

/// Behavioural archetype tag for mob AI dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobArchetype {
    Seeker,
    Creeper4d,
    PhaseSpider,
}

/// How the engine simulates an entity each tick.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimulationMode {
    /// Engine applies collision, gravity, locomotion from a desired direction.
    #[default]
    PhysicsDriven,
    /// Engine directly sets pose — no physics.
    Parametric,
}

/// How a mob moves through the world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobLocomotionMode {
    #[default]
    Walking,
    Flying,
}

/// Default movement parameters for a mob archetype.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MobArchetypeDefaults {
    pub move_speed: f32,
    pub preferred_distance: f32,
    pub tangent_weight: f32,
    pub locomotion: MobLocomotionMode,
}

/// Data-driven entity simulation configuration declared in the WASM manifest.
///
/// This is the single source of truth for entity simulation parameters — the
/// host reads these values instead of hardcoding them per entity type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntitySimConfig {
    pub mode: SimulationMode,
    pub locomotion: MobLocomotionMode,
    pub move_speed: f32,
    pub preferred_distance: f32,
    pub tangent_weight: f32,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub nav_target_y_offset: f32,
    #[serde(default)]
    pub ability_params: Option<MobAbilityParams>,
}

/// Ability-specific parameters for mobs (detonation, blink, etc.).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MobAbilityParams {
    /// Creeper: distance to player that triggers detonation.
    #[serde(default)]
    pub detonate_trigger_distance: f32,
    #[serde(default)]
    pub detonate_radius_voxels: i32,
    #[serde(default)]
    pub detonate_impulse_radius: f32,
    #[serde(default)]
    pub detonate_max_impulse_distance: f32,
    /// PhaseSpider: minimum blink interval in milliseconds.
    #[serde(default)]
    pub blink_min_interval_ms: u64,
    #[serde(default)]
    pub blink_max_interval_ms: u64,
    #[serde(default)]
    pub blink_distance: f32,
    #[serde(default)]
    pub blink_min_distance: f32,
    #[serde(default)]
    pub blink_blocked_progress_epsilon: f32,
}
