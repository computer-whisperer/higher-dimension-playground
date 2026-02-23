use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::block::BlockCategory;
use polychora_plugin_api::content_ids::*;
use polychora_plugin_api::manifest::BlockDeclaration;
use polychora_plugin_api::texture::builtin_textures::*;
use polychora_plugin_api::texture::TextureRef;

/// All 68 block declarations for the polychora-content plugin.
///
/// Each block references a namespace 0 procedural texture via `TextureRef`,
/// which the host resolves to the corresponding GPU shader case.
pub fn block_declarations() -> Vec<BlockDeclaration> {
    use BlockCategory::*;

    macro_rules! block {
        ($id:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr]) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: 0, texture_id: $tex },
                transparent: false,
                light_emission: 0,
            }
        };
        ($id:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr], transparent) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: 0, texture_id: $tex },
                transparent: true,
                light_emission: 0,
            }
        };
        ($id:expr, $tex:expr, $name:expr, $cat:expr, [$r:expr, $g:expr, $b:expr], light: $em:expr) => {
            BlockDeclaration {
                type_id: $id,
                name: String::from($name),
                category: $cat,
                color_hint: [$r, $g, $b],
                texture: TextureRef { namespace: 0, texture_id: $tex },
                transparent: false,
                light_emission: $em,
            }
        };
    }

    alloc::vec![
        // Basic colored blocks (1-8)
        block!(BLOCK_RED,          TEX_RED,          "Red",           Basic,   [255, 0, 0]),
        block!(BLOCK_ORANGE,       TEX_ORANGE,       "Orange",        Basic,   [255, 200, 0]),
        block!(BLOCK_YELLOW_GREEN, TEX_YELLOW_GREEN, "Yellow-Green",  Basic,   [128, 255, 0]),
        block!(BLOCK_GREEN,        TEX_GREEN,        "Green",         Basic,   [0, 255, 51]),
        block!(BLOCK_CYAN,         TEX_CYAN,         "Cyan",          Basic,   [0, 255, 255]),
        block!(BLOCK_BLUE,         TEX_BLUE,         "Blue",          Basic,   [0, 51, 255]),
        block!(BLOCK_PURPLE,       TEX_PURPLE,       "Purple",        Basic,   [128, 0, 255]),
        block!(BLOCK_MAGENTA,      TEX_MAGENTA,      "Magenta",       Basic,   [255, 0, 204]),

        // Special materials (9-14)
        block!(BLOCK_RAINBOW,      TEX_RAINBOW,      "Rainbow",       Special, [180, 180, 180]),
        block!(BLOCK_BROWN,        TEX_BROWN,        "Brown",         Basic,   [139, 69, 20]),
        block!(BLOCK_GRID_FLOOR,   TEX_GRID_FLOOR,   "Grid Floor",    Special, [115, 120, 125]),
        block!(BLOCK_WHITE,        TEX_WHITE,        "White",         Basic,   [255, 255, 255]),
        block!(BLOCK_LIGHT,        TEX_LIGHT,        "Light",         Light,   [255, 255, 220], light: 15),
        block!(BLOCK_MIRROR,       TEX_MIRROR,       "Mirror",        Special, [220, 220, 230]),

        // Animated/special materials (15-26)
        block!(BLOCK_LAVA_VEINED_BASALT,    TEX_LAVA_VEINED_BASALT,    "Lava-Veined Basalt",     Special,  [140, 55, 20]),
        block!(BLOCK_CRYSTAL_LATTICE,       TEX_CRYSTAL_LATTICE,       "Crystal Lattice",        Special,  [130, 180, 220]),
        block!(BLOCK_MARBLE,                TEX_MARBLE,                "Marble",                 Building, [210, 214, 224]),
        block!(BLOCK_OXIDIZED_METAL,        TEX_OXIDIZED_METAL,        "Oxidized Metal",         Special,  [130, 100, 80]),
        block!(BLOCK_BIO_SPORE_MOSS,        TEX_BIO_SPORE_MOSS,       "Bio-Spore Moss",         Special,  [30, 80, 35]),
        block!(BLOCK_VOID_MIRROR,           TEX_VOID_MIRROR,           "Void Mirror",            Special,  [20, 30, 55]),
        block!(BLOCK_AVATAR_MARKER,         TEX_AVATAR_MARKER,         "Avatar Marker",          Special,  [50, 58, 85]),
        block!(BLOCK_HOLOGRAPHIC_LAMINATE,  TEX_HOLOGRAPHIC_LAMINATE,  "Holographic Laminate",   Special,  [40, 100, 170]),
        block!(BLOCK_TIDAL_GLASS,           TEX_TIDAL_GLASS,           "Tidal Glass",            Glass,    [25, 75, 140], transparent),
        block!(BLOCK_CIRCUIT_WEAVE,         TEX_CIRCUIT_WEAVE,         "Circuit Weave",          Special,  [50, 130, 80]),
        block!(BLOCK_AURORA_STONE,          TEX_AURORA_STONE,          "Aurora Stone",            Special,  [65, 40, 100]),
        block!(BLOCK_HAZARD_CHEVRONS,       TEX_HAZARD_CHEVRONS,       "Hazard Chevrons",        Special,  [180, 120, 20]),

        // Natural materials (27-30)
        block!(BLOCK_STONE,        TEX_STONE,        "Stone",         Natural, [125, 128, 133]),
        block!(BLOCK_COBBLESTONE,  TEX_COBBLESTONE,  "Cobblestone",   Natural, [115, 117, 120]),
        block!(BLOCK_DIRT,         TEX_DIRT,         "Dirt",          Natural, [110, 77, 46]),
        block!(BLOCK_COARSE_DIRT,  TEX_COARSE_DIRT,  "Coarse Dirt",   Natural, [105, 74, 43]),

        // Wood materials (31-34)
        block!(BLOCK_OAK_PLANKS,    TEX_OAK_PLANKS,    "Oak Planks",     Wood, [153, 117, 61]),
        block!(BLOCK_SPRUCE_PLANKS, TEX_SPRUCE_PLANKS, "Spruce Planks",  Wood, [97, 68, 36]),
        block!(BLOCK_LOG_BARK,      TEX_LOG_BARK,      "Log Bark",       Wood, [77, 53, 28]),
        block!(BLOCK_LOG_END_RINGS, TEX_LOG_END_RINGS, "Log End Rings",  Wood, [145, 112, 62]),

        // New natural materials (35-40)
        block!(BLOCK_SAND,         TEX_SAND,         "Sand",          Natural, [237, 201, 175]),
        block!(BLOCK_GRAVEL,       TEX_GRAVEL,       "Gravel",        Natural, [131, 126, 126]),
        block!(BLOCK_CLAY,         TEX_CLAY,         "Clay",          Natural, [160, 166, 179]),
        block!(BLOCK_GRASS_BLOCK,  TEX_GRASS_BLOCK,  "Grass Block",   Natural, [115, 162, 75]),
        block!(BLOCK_SNOW,         TEX_SNOW,         "Snow",          Natural, [248, 248, 255]),
        block!(BLOCK_ICE,          TEX_ICE,          "Ice",           Glass,   [145, 180, 240], transparent),

        // Ore materials (41-45)
        block!(BLOCK_COAL_ORE,     TEX_COAL_ORE,     "Coal Ore",      Ore, [85, 85, 85]),
        block!(BLOCK_IRON_ORE,     TEX_IRON_ORE,     "Iron Ore",      Ore, [200, 155, 140]),
        block!(BLOCK_GOLD_ORE,     TEX_GOLD_ORE,     "Gold Ore",      Ore, [255, 215, 0]),
        block!(BLOCK_DIAMOND_ORE,  TEX_DIAMOND_ORE,  "Diamond Ore",   Ore, [90, 220, 220]),
        block!(BLOCK_REDSTONE_ORE, TEX_REDSTONE_ORE, "Redstone Ore",  Ore, [200, 50, 50]),

        // Additional wood and building materials (46-48)
        block!(BLOCK_BIRCH_PLANKS, TEX_BIRCH_PLANKS, "Birch Planks",  Wood,     [216, 205, 163]),
        block!(BLOCK_BRICKS,       TEX_BRICKS,       "Bricks",        Building, [150, 97, 83]),
        block!(BLOCK_SANDSTONE,    TEX_SANDSTONE,    "Sandstone",     Building, [228, 208, 168]),

        // Glass and light materials (49-51)
        block!(BLOCK_GLASS,        TEX_GLASS,        "Glass",         Glass,   [200, 220, 230], transparent),
        block!(BLOCK_GLOWSTONE,    TEX_GLOWSTONE,    "Glowstone",     Light,   [255, 200, 100], light: 15),
        block!(BLOCK_OBSIDIAN,     TEX_OBSIDIAN,     "Obsidian",      Natural, [20, 18, 29]),

        // Special decorative materials (52-54)
        block!(BLOCK_PRISMARINE,   TEX_PRISMARINE,   "Prismarine",    Building, [99, 171, 158]),
        block!(BLOCK_TERRACOTTA,   TEX_TERRACOTTA,   "Terracotta",    Building, [152, 94, 67]),
        block!(BLOCK_WOOL_WHITE,   TEX_WOOL_WHITE,   "Wool (White)",  Building, [233, 236, 236]),

        // Advanced structural materials (55-62)
        block!(BLOCK_BASALT_TILES,    TEX_BASALT_TILES,    "Basalt Tiles",    Building, [68, 70, 76]),
        block!(BLOCK_COPPER_WEAVE,    TEX_COPPER_WEAVE,    "Copper Weave",    Special,  [168, 105, 72]),
        block!(BLOCK_NEBULA_STRATA,   TEX_NEBULA_STRATA,   "Nebula Strata",   Special,  [69, 78, 126]),
        block!(BLOCK_STARFORGED_CORE, TEX_STARFORGED_CORE, "Starforged Core", Light,    [255, 236, 168], light: 12),
        block!(BLOCK_CRYO_CIRCUIT,    TEX_CRYO_CIRCUIT,    "Cryo Circuit",    Special,  [118, 188, 206]),
        block!(BLOCK_SMOKED_GLASS,    TEX_SMOKED_GLASS,    "Smoked Glass",    Glass,    [78, 98, 115], transparent),
        block!(BLOCK_IVORY_MARBLE,    TEX_IVORY_MARBLE,    "Ivory Marble",    Building, [226, 228, 232]),
        block!(BLOCK_RUNIC_ALLOY,     TEX_RUNIC_ALLOY,     "Runic Alloy",     Special,  [140, 146, 158]),

        // Volumetric animated materials (63-68)
        block!(BLOCK_HYPERPHASE_GEL,    TEX_HYPERPHASE_GEL,    "Hyperphase Gel",    Glass,   [102, 208, 255], transparent),
        block!(BLOCK_SINGULARITY_CORE,  TEX_SINGULARITY_CORE,  "Singularity Core",  Light,   [255, 168, 110], light: 15),
        block!(BLOCK_CHRONO_BLOOM,      TEX_CHRONO_BLOOM,      "Chrono Bloom",      Special, [146, 255, 164]),
        block!(BLOCK_TESSERACT_WEAVE,   TEX_TESSERACT_WEAVE,   "Tesseract Weave",   Special, [170, 132, 255]),
        block!(BLOCK_EVENTIDE_ALLOY,    TEX_EVENTIDE_ALLOY,    "Eventide Alloy",    Special, [112, 130, 168]),
        block!(BLOCK_BEACON_MATRIX,     TEX_BEACON_MATRIX,     "Beacon Matrix",     Light,   [255, 248, 196], light: 15),
    ]
}
