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

/// How a mob moves through the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobLocomotionMode {
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
