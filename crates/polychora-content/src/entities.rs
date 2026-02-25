use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::content_ids::*;
use polychora_plugin_api::entity::{
    EntityCategory, EntitySimConfig, MobAbilityParams, MobLocomotionMode, SimulationMode,
};
use polychora_plugin_api::manifest::EntityDeclaration;
use polychora_plugin_api::texture::builtin_textures::*;
use polychora_plugin_api::texture::TextureRef;

/// Helper to create a TextureRef for a builtin (namespace 0) texture.
fn tex(texture_id: u32) -> TextureRef {
    TextureRef { namespace: 0, texture_id }
}

/// 6 entity declarations (all non-player entities).
/// Player entity stays in namespace 0 as an engine internal.
///
/// `model_textures` provides an explicit texture palette for each entity.
/// Rendering code indexes into this palette (slot 0, 1, 2, ...) instead of
/// relying on `base_material_token + saturating_add(N)`.
pub fn entity_declarations() -> Vec<EntityDeclaration> {
    alloc::vec![
        EntityDeclaration {
            type_id: ENTITY_CUBE,
            name: String::from("cube"),
            category: EntityCategory::Accent,
            default_scale: 0.50,
            base_material_color: [128, 0, 255], // Purple
            model_textures: alloc::vec![
                tex(TEX_PURPLE),        // slot 0
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::Parametric,
                locomotion: MobLocomotionMode::default(),
                move_speed: 0.0,
                preferred_distance: 0.0,
                tangent_weight: 0.0,
                aliases: alloc::vec![String::from("testcube")],
                nav_target_y_offset: 0.0,
                ability_params: None,
            }),
        },
        EntityDeclaration {
            type_id: ENTITY_ROTOR,
            name: String::from("rotor"),
            category: EntityCategory::Accent,
            default_scale: 0.54,
            base_material_color: [255, 255, 255], // White
            model_textures: alloc::vec![
                tex(TEX_WHITE),         // slot 0
                tex(TEX_LIGHT),         // slot 1
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::Parametric,
                locomotion: MobLocomotionMode::default(),
                move_speed: 0.0,
                preferred_distance: 0.0,
                tangent_weight: 0.0,
                aliases: alloc::vec![String::from("testrotor")],
                nav_target_y_offset: 0.0,
                ability_params: None,
            }),
        },
        EntityDeclaration {
            type_id: ENTITY_DRIFTER,
            name: String::from("drifter"),
            category: EntityCategory::Accent,
            default_scale: 0.48,
            base_material_color: [210, 214, 224], // Marble
            model_textures: alloc::vec![
                tex(TEX_MARBLE),        // slot 0
                tex(TEX_OXIDIZED_METAL),// slot 1 (unused but keeps alignment)
                tex(TEX_BIO_SPORE_MOSS),// slot 2
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::Parametric,
                locomotion: MobLocomotionMode::default(),
                move_speed: 0.0,
                preferred_distance: 0.0,
                tangent_weight: 0.0,
                aliases: alloc::vec![String::from("testdrifter")],
                nav_target_y_offset: 0.0,
                ability_params: None,
            }),
        },
        EntityDeclaration {
            type_id: ENTITY_SEEKER,
            name: String::from("seeker"),
            category: EntityCategory::Mob,
            default_scale: 0.62,
            base_material_color: [128, 255, 0], // Yellow-Green
            model_textures: alloc::vec![
                tex(TEX_YELLOW_GREEN),  // slot 0
                tex(TEX_GREEN),         // slot 1
                tex(TEX_CYAN),          // slot 2
                tex(TEX_BLUE),          // slot 3
                tex(TEX_PURPLE),        // slot 4
                tex(TEX_MAGENTA),       // slot 5
                tex(TEX_RAINBOW),       // slot 6
                tex(TEX_BROWN),         // slot 7 (unused)
                tex(TEX_GRID_FLOOR),    // slot 8
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::PhysicsDriven,
                locomotion: MobLocomotionMode::Walking,
                move_speed: 3.0,
                preferred_distance: 2.6,
                tangent_weight: 0.72,
                aliases: alloc::vec![String::from("mobseeker")],
                nav_target_y_offset: 0.0,
                ability_params: None,
            }),
        },
        EntityDeclaration {
            type_id: ENTITY_CREEPER,
            name: String::from("creeper"),
            category: EntityCategory::Mob,
            default_scale: 0.78,
            base_material_color: [0, 51, 255], // Blue
            model_textures: alloc::vec![
                tex(TEX_BLUE),          // slot 0
                tex(TEX_PURPLE),        // slot 1
                tex(TEX_MAGENTA),       // slot 2
                tex(TEX_RAINBOW),       // slot 3
                tex(TEX_BROWN),         // slot 4
                tex(TEX_GRID_FLOOR),    // slot 5
                tex(TEX_WHITE),         // slot 6
                tex(TEX_LIGHT),         // slot 7
                tex(TEX_MIRROR),        // slot 8 (unused)
                tex(TEX_LAVA_VEINED_BASALT), // slot 9
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::PhysicsDriven,
                locomotion: MobLocomotionMode::Walking,
                move_speed: 2.55,
                preferred_distance: 3.8,
                tangent_weight: 0.92,
                aliases: alloc::vec![
                    String::from("4dcreeper"),
                    String::from("mobcreeper4d"),
                ],
                nav_target_y_offset: 1.05,
                ability_params: Some(MobAbilityParams {
                    detonate_trigger_distance: 1.55,
                    detonate_radius_voxels: 3,
                    detonate_impulse_radius: 7.0,
                    detonate_max_impulse_distance: 5.0,
                    blink_min_interval_ms: 0,
                    blink_max_interval_ms: 0,
                    blink_distance: 0.0,
                    blink_min_distance: 0.0,
                    blink_blocked_progress_epsilon: 0.0,
                }),
            }),
        },
        EntityDeclaration {
            type_id: ENTITY_PHASE_SPIDER,
            name: String::from("phase_spider"),
            category: EntityCategory::Mob,
            default_scale: 0.86,
            base_material_color: [180, 180, 180], // Rainbow
            model_textures: alloc::vec![
                tex(TEX_RAINBOW),       // slot 0
                tex(TEX_BROWN),         // slot 1
                tex(TEX_GRID_FLOOR),    // slot 2
                tex(TEX_WHITE),         // slot 3
                tex(TEX_LIGHT),         // slot 4
                tex(TEX_MIRROR),        // slot 5
                tex(TEX_LAVA_VEINED_BASALT), // slot 6 (unused)
                tex(TEX_CRYSTAL_LATTICE), // slot 7
                tex(TEX_MARBLE),        // slot 8
                tex(TEX_OXIDIZED_METAL),// slot 9
            ],
            sim_config: Some(EntitySimConfig {
                mode: SimulationMode::PhysicsDriven,
                locomotion: MobLocomotionMode::Flying,
                move_speed: 3.1,
                preferred_distance: 2.4,
                tangent_weight: 0.95,
                aliases: alloc::vec![
                    String::from("phase-spider"),
                    String::from("spider"),
                    String::from("mobphasespider"),
                ],
                nav_target_y_offset: 0.0,
                ability_params: Some(MobAbilityParams {
                    detonate_trigger_distance: 0.0,
                    detonate_radius_voxels: 0,
                    detonate_impulse_radius: 0.0,
                    detonate_max_impulse_distance: 0.0,
                    blink_min_interval_ms: 720,
                    blink_max_interval_ms: 1520,
                    blink_distance: 2.8,
                    blink_min_distance: 1.0,
                    blink_blocked_progress_epsilon: 0.08,
                }),
            }),
        },
    ]
}
