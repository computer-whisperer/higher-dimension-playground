// Re-export core entity types from the plugin API crate.
pub use polychora_plugin_api::entity::{
    EntityCategory, MobConfig, MobLocomotionMode,
};

// ---------------------------------------------------------------------------
// Well-known entity type constants: (namespace, entity_type)
// ---------------------------------------------------------------------------

use polychora_plugin_api::content_ids;

pub const ENTITY_PLAYER_AVATAR: (u32, u32) = (0, 0);
pub const ENTITY_TEST_CUBE: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_CUBE);
pub const ENTITY_TEST_ROTOR: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_ROTOR);
pub const ENTITY_TEST_DRIFTER: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_DRIFTER);
pub const ENTITY_MOB_SEEKER: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_SEEKER);
pub const ENTITY_MOB_CREEPER4D: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_CREEPER);
pub const ENTITY_MOB_PHASE_SPIDER: (u32, u32) = (content_ids::CONTENT_NS, content_ids::ENTITY_PHASE_SPIDER);
