// Re-export core entity types from the plugin API crate.
pub use polychora_plugin_api::entity::{
    EntityCategory, MobArchetype, MobArchetypeDefaults, MobLocomotionMode,
};

// ---------------------------------------------------------------------------
// Entity type registry entry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EntityTypeEntry {
    pub namespace: u32,
    pub entity_type: u32,
    pub category: EntityCategory,
    pub canonical_name: &'static str,
    pub aliases: &'static [&'static str],
    pub default_scale: f32,
    pub base_material: u8,
    pub mob_archetype: Option<MobArchetype>,
    pub mob_defaults: Option<MobArchetypeDefaults>,
}

impl EntityTypeEntry {
    pub fn type_key(&self) -> (u32, u32) {
        (self.namespace, self.entity_type)
    }

    pub fn is_spawnable(&self) -> bool {
        self.category != EntityCategory::Player
    }
}

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

// ---------------------------------------------------------------------------
// Static registry
// ---------------------------------------------------------------------------

pub const ENTITY_TYPES: &[EntityTypeEntry] = &[
    EntityTypeEntry {
        namespace: 0,
        entity_type: 0,
        category: EntityCategory::Player,
        canonical_name: "player",
        aliases: &["player", "avatar"],
        default_scale: 1.0,
        base_material: 0,
        mob_archetype: None,
        mob_defaults: None,
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0x776b1b69,
        category: EntityCategory::Accent,
        canonical_name: "cube",
        aliases: &["cube", "testcube"],
        default_scale: 0.50,
        base_material: 7,
        mob_archetype: None,
        mob_defaults: None,
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0x71790134,
        category: EntityCategory::Accent,
        canonical_name: "rotor",
        aliases: &["rotor", "testrotor"],
        default_scale: 0.54,
        base_material: 12,
        mob_archetype: None,
        mob_defaults: None,
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0x433824fe,
        category: EntityCategory::Accent,
        canonical_name: "drifter",
        aliases: &["drifter", "testdrifter"],
        default_scale: 0.48,
        base_material: 17,
        mob_archetype: None,
        mob_defaults: None,
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0xa974d75b,
        category: EntityCategory::Mob,
        canonical_name: "seeker",
        aliases: &["seeker", "mobseeker"],
        default_scale: 0.62,
        base_material: 3,
        mob_archetype: Some(MobArchetype::Seeker),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 3.0,
            preferred_distance: 2.6,
            tangent_weight: 0.72,
            locomotion: MobLocomotionMode::Walking,
        }),
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0x3dc5fd3d,
        category: EntityCategory::Mob,
        canonical_name: "creeper",
        aliases: &["creeper", "4dcreeper", "mobcreeper4d"],
        default_scale: 0.78,
        base_material: 6,
        mob_archetype: Some(MobArchetype::Creeper4d),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 2.55,
            preferred_distance: 3.8,
            tangent_weight: 0.92,
            locomotion: MobLocomotionMode::Walking,
        }),
    },
    EntityTypeEntry {
        namespace: content_ids::CONTENT_NS,
        entity_type: 0x4af27f80,
        category: EntityCategory::Mob,
        canonical_name: "phase_spider",
        aliases: &[
            "phase_spider",
            "phasespider",
            "phase-spider",
            "spider",
            "mobphasespider",
        ],
        default_scale: 0.86,
        base_material: 9,
        mob_archetype: Some(MobArchetype::PhaseSpider),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 3.1,
            preferred_distance: 2.4,
            tangent_weight: 0.95,
            locomotion: MobLocomotionMode::Flying,
        }),
    },
];

// ---------------------------------------------------------------------------
// Lookup helpers
// ---------------------------------------------------------------------------

fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect()
}

#[deprecated(note = "use ContentRegistry::entity_lookup() instead")]
pub fn lookup(namespace: u32, entity_type: u32) -> Option<&'static EntityTypeEntry> {
    ENTITY_TYPES
        .iter()
        .find(|entry| entry.namespace == namespace && entry.entity_type == entity_type)
}

#[deprecated(note = "use ContentRegistry::entity_lookup_by_name() instead")]
pub fn lookup_by_name(name: &str) -> Option<&'static EntityTypeEntry> {
    let normalized = normalize_token(name);
    ENTITY_TYPES.iter().find(|entry| {
        entry
            .aliases
            .iter()
            .any(|alias| normalize_token(alias) == normalized)
    })
}

#[deprecated(note = "use ContentRegistry::entity_category() instead")]
#[allow(deprecated)]
pub fn category_for(namespace: u32, entity_type: u32) -> EntityCategory {
    lookup(namespace, entity_type)
        .map(|entry| entry.category)
        .unwrap_or(EntityCategory::Accent) // unknown types default to Accent
}

#[deprecated(note = "use ContentRegistry::mob_archetype_defaults() instead")]
pub fn mob_archetype_defaults(archetype: MobArchetype) -> MobArchetypeDefaults {
    for entry in ENTITY_TYPES {
        if entry.mob_archetype == Some(archetype) {
            if let Some(defaults) = entry.mob_defaults {
                return defaults;
            }
        }
    }
    // Fallback defaults
    MobArchetypeDefaults {
        move_speed: 2.0,
        preferred_distance: 3.0,
        tangent_weight: 0.5,
        locomotion: MobLocomotionMode::Walking,
    }
}

#[deprecated(note = "use ContentRegistry::entity_base_material_token() instead")]
#[allow(deprecated)]
pub fn base_material_for(namespace: u32, entity_type: u32) -> u8 {
    lookup(namespace, entity_type)
        .map(|entry| entry.base_material)
        .unwrap_or(7)
}

#[deprecated(note = "use ContentRegistry::spawnable_entity_names() instead")]
pub fn spawnable_names() -> Vec<&'static str> {
    ENTITY_TYPES
        .iter()
        .filter(|e| e.is_spawnable())
        .map(|e| e.canonical_name)
        .collect()
}
