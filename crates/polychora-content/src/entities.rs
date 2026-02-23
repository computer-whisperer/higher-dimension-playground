use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::entity::EntityCategory;
use polychora_plugin_api::manifest::EntityDeclaration;

// Random entity type IDs for the polychora-content namespace.
pub const ENTITY_CUBE: u32         = 0x776b1b69;
pub const ENTITY_ROTOR: u32        = 0x71790134;
pub const ENTITY_DRIFTER: u32      = 0x433824fe;
pub const ENTITY_SEEKER: u32       = 0xa974d75b;
pub const ENTITY_CREEPER: u32      = 0x3dc5fd3d;
pub const ENTITY_PHASE_SPIDER: u32 = 0x4af27f80;

/// 6 entity declarations (all non-player entities).
/// Player entity stays in namespace 0 as an engine internal.
pub fn entity_declarations() -> Vec<EntityDeclaration> {
    alloc::vec![
        EntityDeclaration {
            type_id: ENTITY_CUBE,
            name: String::from("cube"),
            category: EntityCategory::Accent,
            default_scale: 0.50,
            base_material_color: [128, 0, 255], // Purple
        },
        EntityDeclaration {
            type_id: ENTITY_ROTOR,
            name: String::from("rotor"),
            category: EntityCategory::Accent,
            default_scale: 0.54,
            base_material_color: [255, 255, 255], // White
        },
        EntityDeclaration {
            type_id: ENTITY_DRIFTER,
            name: String::from("drifter"),
            category: EntityCategory::Accent,
            default_scale: 0.48,
            base_material_color: [210, 214, 224], // Marble
        },
        EntityDeclaration {
            type_id: ENTITY_SEEKER,
            name: String::from("seeker"),
            category: EntityCategory::Mob,
            default_scale: 0.62,
            base_material_color: [128, 255, 0], // Yellow-Green
        },
        EntityDeclaration {
            type_id: ENTITY_CREEPER,
            name: String::from("creeper"),
            category: EntityCategory::Mob,
            default_scale: 0.78,
            base_material_color: [0, 51, 255], // Blue
        },
        EntityDeclaration {
            type_id: ENTITY_PHASE_SPIDER,
            name: String::from("phase_spider"),
            category: EntityCategory::Mob,
            default_scale: 0.86,
            base_material_color: [180, 180, 180], // Rainbow
        },
    ]
}
