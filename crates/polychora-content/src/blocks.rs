use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::content_ids::*;
use polychora_plugin_api::manifest::{BlockDeclaration, BlockTickConfig};
use polychora_plugin_api::texture::builtin_textures::*;
use polychora_plugin_api::texture::TextureRef;

/// All block declarations for the polychora-content plugin.
///
/// Each block references a namespace 0 procedural texture via `TextureRef`,
/// which the host resolves to the corresponding GPU shader case.
pub fn block_declarations() -> Vec<BlockDeclaration> {
    use BlockCategory::*;

    macro_rules! block {
        ($id:expr, $ns:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr]) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: $ns, texture_id: $tex },
                transparent: false,
                light_emission: 0,
                interactable: false,
                tick_config: None,
            }
        };
        ($id:expr, $ns:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr], transparent) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: $ns, texture_id: $tex },
                transparent: true,
                light_emission: 0,
                interactable: false,
                tick_config: None,
            }
        };
        ($id:expr, $ns:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr], light: $em:expr) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: $ns, texture_id: $tex },
                transparent: false,
                light_emission: $em,
                interactable: false,
                tick_config: None,
            }
        };
    }

    let ns = crate::NAMESPACE;
    alloc::vec![
        // Basic colored blocks (1-8) — texture pool via plugin namespace
        block!(BLOCK_RED,          ns, TEX_RED,          "Red",           Basic,   [255, 0, 0]),
        block!(BLOCK_ORANGE,       ns, TEX_ORANGE,       "Orange",        Basic,   [255, 200, 0]),
        block!(BLOCK_YELLOW_GREEN, ns, TEX_YELLOW_GREEN, "Yellow-Green",  Basic,   [128, 255, 0]),
        block!(BLOCK_GREEN,        ns, TEX_GREEN,        "Green",         Basic,   [0, 255, 51]),
        block!(BLOCK_CYAN,         ns, TEX_CYAN,         "Cyan",          Basic,   [0, 255, 255]),
        block!(BLOCK_BLUE,         ns, TEX_BLUE,         "Blue",          Basic,   [0, 51, 255]),
        block!(BLOCK_PURPLE,       ns, TEX_PURPLE,       "Purple",        Basic,   [128, 0, 255]),
        block!(BLOCK_MAGENTA,      ns, TEX_MAGENTA,      "Magenta",       Basic,   [255, 0, 204]),

        // Special materials (9-14)
        block!(BLOCK_RAINBOW,      0, TEX_RAINBOW,      "Rainbow",       Special, [180, 180, 180]),
        block!(BLOCK_BROWN,        ns, TEX_BROWN,        "Brown",         Basic,   [139, 69, 20]),
        block!(BLOCK_GRID_FLOOR,   0, TEX_GRID_FLOOR,   "Grid Floor",    Special, [115, 120, 125]),
        block!(BLOCK_WHITE,        ns, TEX_WHITE,        "White",         Basic,   [255, 255, 255]),
        block!(BLOCK_LIGHT,        0, TEX_LIGHT,        "Light",         Light,   [255, 255, 220], light: 15),
        block!(BLOCK_MIRROR,       0, TEX_MIRROR,       "Mirror",        Special, [220, 220, 230]),

        // Animated/special materials (15-26) — procedural (namespace 0)
        block!(BLOCK_LAVA_VEINED_BASALT,    0, TEX_LAVA_VEINED_BASALT,    "Lava-Veined Basalt",     Special,  [140, 55, 20]),
        block!(BLOCK_CRYSTAL_LATTICE,       0, TEX_CRYSTAL_LATTICE,       "Crystal Lattice",        Special,  [130, 180, 220]),
        block!(BLOCK_MARBLE,                0, TEX_MARBLE,                "Marble",                 Building, [210, 214, 224]),
        block!(BLOCK_OXIDIZED_METAL,        0, TEX_OXIDIZED_METAL,        "Oxidized Metal",         Special,  [130, 100, 80]),
        block!(BLOCK_BIO_SPORE_MOSS,        0, TEX_BIO_SPORE_MOSS,       "Bio-Spore Moss",         Special,  [30, 80, 35]),
        block!(BLOCK_VOID_MIRROR,           0, TEX_VOID_MIRROR,           "Void Mirror",            Special,  [20, 30, 55]),
        block!(BLOCK_AVATAR_MARKER,         0, TEX_AVATAR_MARKER,         "Avatar Marker",          Special,  [50, 58, 85]),
        block!(BLOCK_HOLOGRAPHIC_LAMINATE,  0, TEX_HOLOGRAPHIC_LAMINATE,  "Holographic Laminate",   Special,  [40, 100, 170]),
        block!(BLOCK_TIDAL_GLASS,           0, TEX_TIDAL_GLASS,           "Tidal Glass",            Glass,    [25, 75, 140], transparent),
        block!(BLOCK_CIRCUIT_WEAVE,         0, TEX_CIRCUIT_WEAVE,         "Circuit Weave",          Special,  [50, 130, 80]),
        block!(BLOCK_AURORA_STONE,          0, TEX_AURORA_STONE,          "Aurora Stone",            Special,  [65, 40, 100]),
        block!(BLOCK_HAZARD_CHEVRONS,       0, TEX_HAZARD_CHEVRONS,       "Hazard Chevrons",        Special,  [180, 120, 20]),

        // Natural materials (27-30)
        block!(BLOCK_STONE,        0, TEX_STONE,        "Stone",         Natural, [125, 128, 133]),
        block!(BLOCK_COBBLESTONE,  0, TEX_COBBLESTONE,  "Cobblestone",   Natural, [115, 117, 120]),
        block!(BLOCK_DIRT,         0, TEX_DIRT,         "Dirt",          Natural, [110, 77, 46]),
        block!(BLOCK_COARSE_DIRT,  0, TEX_COARSE_DIRT,  "Coarse Dirt",   Natural, [105, 74, 43]),

        // Wood materials (31-34)
        block!(BLOCK_OAK_PLANKS,    0, TEX_OAK_PLANKS,    "Oak Planks",     Wood, [153, 117, 61]),
        block!(BLOCK_SPRUCE_PLANKS, 0, TEX_SPRUCE_PLANKS, "Spruce Planks",  Wood, [97, 68, 36]),
        block!(BLOCK_LOG_BARK,      0, TEX_LOG_BARK,      "Log Bark",       Wood, [77, 53, 28]),
        block!(BLOCK_LOG_END_RINGS, 0, TEX_LOG_END_RINGS, "Log End Rings",  Wood, [145, 112, 62]),

        // New natural materials (35-40)
        block!(BLOCK_SAND,         0, TEX_SAND,         "Sand",          Natural, [237, 201, 175]),
        block!(BLOCK_GRAVEL,       0, TEX_GRAVEL,       "Gravel",        Natural, [131, 126, 126]),
        block!(BLOCK_CLAY,         0, TEX_CLAY,         "Clay",          Natural, [160, 166, 179]),
        block!(BLOCK_GRASS_BLOCK,  0, TEX_GRASS_BLOCK,  "Grass Block",   Natural, [115, 162, 75]),
        block!(BLOCK_SNOW,         0, TEX_SNOW,         "Snow",          Natural, [248, 248, 255]),
        block!(BLOCK_ICE,          0, TEX_ICE,          "Ice",           Glass,   [145, 180, 240], transparent),

        // Ore materials (41-45)
        block!(BLOCK_COAL_ORE,     0, TEX_COAL_ORE,     "Coal Ore",      Ore, [85, 85, 85]),
        block!(BLOCK_IRON_ORE,     0, TEX_IRON_ORE,     "Iron Ore",      Ore, [200, 155, 140]),
        block!(BLOCK_GOLD_ORE,     0, TEX_GOLD_ORE,     "Gold Ore",      Ore, [255, 215, 0]),
        block!(BLOCK_DIAMOND_ORE,  0, TEX_DIAMOND_ORE,  "Diamond Ore",   Ore, [90, 220, 220]),
        block!(BLOCK_REDSTONE_ORE, 0, TEX_REDSTONE_ORE, "Redstone Ore",  Ore, [200, 50, 50]),

        // Additional wood and building materials (46-48)
        block!(BLOCK_BIRCH_PLANKS, 0, TEX_BIRCH_PLANKS, "Birch Planks",  Wood,     [216, 205, 163]),
        block!(BLOCK_BRICKS,       0, TEX_BRICKS,       "Bricks",        Building, [150, 97, 83]),
        block!(BLOCK_SANDSTONE,    0, TEX_SANDSTONE,    "Sandstone",     Building, [228, 208, 168]),

        // Glass and light materials (49-51)
        block!(BLOCK_GLASS,        0, TEX_GLASS,        "Glass",         Glass,   [200, 220, 230], transparent),
        block!(BLOCK_GLOWSTONE,    0, TEX_GLOWSTONE,    "Glowstone",     Light,   [255, 200, 100], light: 15),
        block!(BLOCK_OBSIDIAN,     0, TEX_OBSIDIAN,     "Obsidian",      Natural, [20, 18, 29]),

        // Special decorative materials (52-54)
        block!(BLOCK_PRISMARINE,   0, TEX_PRISMARINE,   "Prismarine",    Building, [99, 171, 158]),
        block!(BLOCK_TERRACOTTA,   0, TEX_TERRACOTTA,   "Terracotta",    Building, [152, 94, 67]),
        block!(BLOCK_WOOL_WHITE,   0, TEX_WOOL_WHITE,   "Wool (White)",  Building, [233, 236, 236]),

        // Advanced structural materials (55-62)
        block!(BLOCK_BASALT_TILES,    0, TEX_BASALT_TILES,    "Basalt Tiles",    Building, [68, 70, 76]),
        block!(BLOCK_COPPER_WEAVE,    0, TEX_COPPER_WEAVE,    "Copper Weave",    Special,  [168, 105, 72]),
        block!(BLOCK_NEBULA_STRATA,   0, TEX_NEBULA_STRATA,   "Nebula Strata",   Special,  [69, 78, 126]),
        block!(BLOCK_STARFORGED_CORE, 0, TEX_STARFORGED_CORE, "Starforged Core", Light,    [255, 236, 168], light: 12),
        block!(BLOCK_CRYO_CIRCUIT,    0, TEX_CRYO_CIRCUIT,    "Cryo Circuit",    Special,  [118, 188, 206]),
        block!(BLOCK_SMOKED_GLASS,    0, TEX_SMOKED_GLASS,    "Smoked Glass",    Glass,    [78, 98, 115], transparent),
        block!(BLOCK_IVORY_MARBLE,    0, TEX_IVORY_MARBLE,    "Ivory Marble",    Building, [226, 228, 232]),
        block!(BLOCK_RUNIC_ALLOY,     0, TEX_RUNIC_ALLOY,     "Runic Alloy",     Special,  [140, 146, 158]),

        // Volumetric animated materials (63-68)
        block!(BLOCK_HYPERPHASE_GEL,    0, TEX_HYPERPHASE_GEL,    "Hyperphase Gel",    Glass,   [102, 208, 255], transparent),
        block!(BLOCK_SINGULARITY_CORE,  0, TEX_SINGULARITY_CORE,  "Singularity Core",  Light,   [255, 168, 110], light: 15),
        block!(BLOCK_CHRONO_BLOOM,      0, TEX_CHRONO_BLOOM,      "Chrono Bloom",      Special, [146, 255, 164]),
        block!(BLOCK_TESSERACT_WEAVE,   0, TEX_TESSERACT_WEAVE,   "Tesseract Weave",   Special, [170, 132, 255]),
        block!(BLOCK_EVENTIDE_ALLOY,    0, TEX_EVENTIDE_ALLOY,    "Eventide Alloy",    Special, [112, 130, 168]),
        block!(BLOCK_BEACON_MATRIX,     0, TEX_BEACON_MATRIX,     "Beacon Matrix",     Light,   [255, 248, 196], light: 15),

        // Chest (70) — interactive block with inventory
        BlockDeclaration {
            type_id: BLOCK_CHEST,
            name: String::from("Chest"),
            category: Special,
            color_hint: [153, 117, 61],
            texture: TextureRef { namespace: CONTENT_NS, texture_id: TEX_CHEST },
            transparent: false,
            light_emission: 0,
            interactable: true,
            tick_config: None,
        },

        // Spawner (71) — ticking block that spawns entities
        BlockDeclaration {
            type_id: BLOCK_SPAWNER,
            name: String::from("Spawner"),
            category: Special,
            color_hint: [80, 40, 120],
            texture: TextureRef { namespace: 0, texture_id: TEX_SINGULARITY_CORE },
            transparent: false,
            light_emission: 8,
            interactable: true,
            tick_config: Some(BlockTickConfig {
                interval_ms: 5000,
                activation_radius: 24.0,
            }),
        },
    ]
}
