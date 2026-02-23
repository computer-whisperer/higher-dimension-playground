use crate::content_registry::{ContentRegistry, EntityEntry};
use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::entity::{
    EntityCategory, MobArchetype, MobArchetypeDefaults, MobLocomotionMode,
};

/// Register all built-in blocks, entities, and legacy remap tables.
///
/// Phase 1: keeps namespace 0 and legacy block_type IDs (1-68) identical to
/// the old `MATERIALS` array.  Material tokens are forced to match the old
/// procedural shader IDs so the GPU pipeline is unchanged.
///
/// Phase 2 will migrate first-party content to a random-ID namespace.
pub fn register_builtin_content(registry: &mut ContentRegistry) {
    register_builtin_blocks(registry);
    register_builtin_entities(registry);
}

fn register_builtin_blocks(registry: &mut ContentRegistry) {
    // Macro to reduce repetition.  Each call registers a block at
    // namespace 0 with the legacy block_type == forced material token.
    macro_rules! block {
        ($bt:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr]) => {
            registry.register_block_with_token(0, $bt, $name, $cat, [$r, $g, $b], $bt as u16);
        };
    }

    use BlockCategory::*;

    // Basic colored blocks (1-8)
    block!(1,  "Red",           Basic,   [255, 0, 0]);
    block!(2,  "Orange",        Basic,   [255, 200, 0]);
    block!(3,  "Yellow-Green",  Basic,   [128, 255, 0]);
    block!(4,  "Green",         Basic,   [0, 255, 51]);
    block!(5,  "Cyan",          Basic,   [0, 255, 255]);
    block!(6,  "Blue",          Basic,   [0, 51, 255]);
    block!(7,  "Purple",        Basic,   [128, 0, 255]);
    block!(8,  "Magenta",       Basic,   [255, 0, 204]);

    // Special materials (9-14)
    block!(9,  "Rainbow",       Special, [180, 180, 180]);
    block!(10, "Brown",         Basic,   [139, 69, 20]);
    block!(11, "Grid Floor",    Special, [115, 120, 125]);
    block!(12, "White",         Basic,   [255, 255, 255]);
    block!(13, "Light",         Light,   [255, 255, 220]);
    block!(14, "Mirror",        Special, [220, 220, 230]);

    // Animated/special materials (15-26)
    block!(15, "Lava-Veined Basalt",      Special,  [140, 55, 20]);
    block!(16, "Crystal Lattice",         Special,  [130, 180, 220]);
    block!(17, "Marble",                  Building, [210, 214, 224]);
    block!(18, "Oxidized Metal",          Special,  [130, 100, 80]);
    block!(19, "Bio-Spore Moss",          Special,  [30, 80, 35]);
    block!(20, "Void Mirror",             Special,  [20, 30, 55]);
    block!(21, "Avatar Marker",           Special,  [50, 58, 85]);
    block!(22, "Holographic Laminate",    Special,  [40, 100, 170]);
    block!(23, "Tidal Glass",             Glass,    [25, 75, 140]);
    block!(24, "Circuit Weave",           Special,  [50, 130, 80]);
    block!(25, "Aurora Stone",            Special,  [65, 40, 100]);
    block!(26, "Hazard Chevrons",         Special,  [180, 120, 20]);

    // Natural materials (27-30)
    block!(27, "Stone",         Natural, [125, 128, 133]);
    block!(28, "Cobblestone",   Natural, [115, 117, 120]);
    block!(29, "Dirt",          Natural, [110, 77, 46]);
    block!(30, "Coarse Dirt",   Natural, [105, 74, 43]);

    // Wood materials (31-34)
    block!(31, "Oak Planks",     Wood, [153, 117, 61]);
    block!(32, "Spruce Planks",  Wood, [97, 68, 36]);
    block!(33, "Log Bark",       Wood, [77, 53, 28]);
    block!(34, "Log End Rings",  Wood, [145, 112, 62]);

    // New natural materials (35-40)
    block!(35, "Sand",          Natural, [237, 201, 175]);
    block!(36, "Gravel",        Natural, [131, 126, 126]);
    block!(37, "Clay",          Natural, [160, 166, 179]);
    block!(38, "Grass Block",   Natural, [115, 162, 75]);
    block!(39, "Snow",          Natural, [248, 248, 255]);
    block!(40, "Ice",           Glass,   [145, 180, 240]);

    // Ore materials (41-45)
    block!(41, "Coal Ore",      Ore, [85, 85, 85]);
    block!(42, "Iron Ore",      Ore, [200, 155, 140]);
    block!(43, "Gold Ore",      Ore, [255, 215, 0]);
    block!(44, "Diamond Ore",   Ore, [90, 220, 220]);
    block!(45, "Redstone Ore",  Ore, [200, 50, 50]);

    // Additional wood and building materials (46-48)
    block!(46, "Birch Planks",  Wood,     [216, 205, 163]);
    block!(47, "Bricks",        Building, [150, 97, 83]);
    block!(48, "Sandstone",     Building, [228, 208, 168]);

    // Glass and light materials (49-51)
    block!(49, "Glass",         Glass,   [200, 220, 230]);
    block!(50, "Glowstone",     Light,   [255, 200, 100]);
    block!(51, "Obsidian",      Natural, [20, 18, 29]);

    // Special decorative materials (52-54)
    block!(52, "Prismarine",    Building, [99, 171, 158]);
    block!(53, "Terracotta",    Building, [152, 94, 67]);
    block!(54, "Wool (White)",  Building, [233, 236, 236]);

    // Advanced structural materials (55-62)
    block!(55, "Basalt Tiles",  Building, [68, 70, 76]);
    block!(56, "Copper Weave",  Special,  [168, 105, 72]);
    block!(57, "Nebula Strata", Special,  [69, 78, 126]);
    block!(58, "Starforged Core", Light,  [255, 236, 168]);
    block!(59, "Cryo Circuit",  Special,  [118, 188, 206]);
    block!(60, "Smoked Glass",  Glass,    [78, 98, 115]);
    block!(61, "Ivory Marble",  Building, [226, 228, 232]);
    block!(62, "Runic Alloy",   Special,  [140, 146, 158]);

    // Volumetric animated materials (63-68)
    block!(63, "Hyperphase Gel",    Glass,   [102, 208, 255]);
    block!(64, "Singularity Core",  Light,   [255, 168, 110]);
    block!(65, "Chrono Bloom",      Special, [146, 255, 164]);
    block!(66, "Tesseract Weave",   Special, [170, 132, 255]);
    block!(67, "Eventide Alloy",    Special, [112, 130, 168]);
    block!(68, "Beacon Matrix",     Light,   [255, 248, 196]);
}

fn register_builtin_entities(registry: &mut ContentRegistry) {
    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 0,
        category: EntityCategory::Player,
        canonical_name: "player",
        aliases: &["player", "avatar"],
        default_scale: 1.0,
        base_material_token: 0,
        mob_archetype: None,
        mob_defaults: None,
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 1,
        category: EntityCategory::Accent,
        canonical_name: "cube",
        aliases: &["cube", "testcube"],
        default_scale: 0.50,
        base_material_token: 7,
        mob_archetype: None,
        mob_defaults: None,
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 2,
        category: EntityCategory::Accent,
        canonical_name: "rotor",
        aliases: &["rotor", "testrotor"],
        default_scale: 0.54,
        base_material_token: 12,
        mob_archetype: None,
        mob_defaults: None,
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 3,
        category: EntityCategory::Accent,
        canonical_name: "drifter",
        aliases: &["drifter", "testdrifter"],
        default_scale: 0.48,
        base_material_token: 17,
        mob_archetype: None,
        mob_defaults: None,
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 10,
        category: EntityCategory::Mob,
        canonical_name: "seeker",
        aliases: &["seeker", "mobseeker"],
        default_scale: 0.62,
        base_material_token: 3,
        mob_archetype: Some(MobArchetype::Seeker),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 3.0,
            preferred_distance: 2.6,
            tangent_weight: 0.72,
            locomotion: MobLocomotionMode::Walking,
        }),
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 11,
        category: EntityCategory::Mob,
        canonical_name: "creeper",
        aliases: &["creeper", "4dcreeper", "mobcreeper4d"],
        default_scale: 0.78,
        base_material_token: 6,
        mob_archetype: Some(MobArchetype::Creeper4d),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 2.55,
            preferred_distance: 3.8,
            tangent_weight: 0.92,
            locomotion: MobLocomotionMode::Walking,
        }),
    });

    registry.register_entity(EntityEntry {
        namespace: 0,
        entity_type: 12,
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
        base_material_token: 9,
        mob_archetype: Some(MobArchetype::PhaseSpider),
        mob_defaults: Some(MobArchetypeDefaults {
            move_speed: 3.1,
            preferred_distance: 2.4,
            tangent_weight: 0.95,
            locomotion: MobLocomotionMode::Flying,
        }),
    });
}
