// Material metadata system for Polychora 4D voxel game

// Namespace IDs
pub const NAMESPACE_POLYCHORA: u32 = 0;

// Block type IDs (polychora namespace, matching legacy VoxelType u8 values)
pub const BLOCK_AIR: u32 = 0;
pub const BLOCK_RED: u32 = 1;
pub const BLOCK_ORANGE: u32 = 2;
pub const BLOCK_YELLOW_GREEN: u32 = 3;
pub const BLOCK_GREEN: u32 = 4;
pub const BLOCK_CYAN: u32 = 5;
pub const BLOCK_BLUE: u32 = 6;
pub const BLOCK_PURPLE: u32 = 7;
pub const BLOCK_MAGENTA: u32 = 8;
pub const BLOCK_RAINBOW: u32 = 9;
pub const BLOCK_BROWN: u32 = 10;
pub const BLOCK_GRID_FLOOR: u32 = 11;
pub const BLOCK_WHITE: u32 = 12;
pub const BLOCK_LIGHT: u32 = 13;
pub const BLOCK_MIRROR: u32 = 14;
pub const BLOCK_LAVA_VEINED_BASALT: u32 = 15;
pub const BLOCK_CRYSTAL_LATTICE: u32 = 16;
pub const BLOCK_MARBLE: u32 = 17;
pub const BLOCK_OXIDIZED_METAL: u32 = 18;
pub const BLOCK_BIO_SPORE_MOSS: u32 = 19;
pub const BLOCK_VOID_MIRROR: u32 = 20;
pub const BLOCK_AVATAR_MARKER: u32 = 21;
pub const BLOCK_HOLOGRAPHIC_LAMINATE: u32 = 22;
pub const BLOCK_TIDAL_GLASS: u32 = 23;
pub const BLOCK_CIRCUIT_WEAVE: u32 = 24;
pub const BLOCK_AURORA_STONE: u32 = 25;
pub const BLOCK_HAZARD_CHEVRONS: u32 = 26;
pub const BLOCK_STONE: u32 = 27;
pub const BLOCK_COBBLESTONE: u32 = 28;
pub const BLOCK_DIRT: u32 = 29;
pub const BLOCK_COARSE_DIRT: u32 = 30;
pub const BLOCK_OAK_PLANKS: u32 = 31;
pub const BLOCK_SPRUCE_PLANKS: u32 = 32;
pub const BLOCK_LOG_BARK: u32 = 33;
pub const BLOCK_LOG_END_RINGS: u32 = 34;
pub const BLOCK_SAND: u32 = 35;
pub const BLOCK_GRAVEL: u32 = 36;
pub const BLOCK_CLAY: u32 = 37;
pub const BLOCK_GRASS_BLOCK: u32 = 38;
pub const BLOCK_SNOW: u32 = 39;
pub const BLOCK_ICE: u32 = 40;
pub const BLOCK_COAL_ORE: u32 = 41;
pub const BLOCK_IRON_ORE: u32 = 42;
pub const BLOCK_GOLD_ORE: u32 = 43;
pub const BLOCK_DIAMOND_ORE: u32 = 44;
pub const BLOCK_REDSTONE_ORE: u32 = 45;
pub const BLOCK_BIRCH_PLANKS: u32 = 46;
pub const BLOCK_BRICKS: u32 = 47;
pub const BLOCK_SANDSTONE: u32 = 48;
pub const BLOCK_GLASS: u32 = 49;
pub const BLOCK_GLOWSTONE: u32 = 50;
pub const BLOCK_OBSIDIAN: u32 = 51;
pub const BLOCK_PRISMARINE: u32 = 52;
pub const BLOCK_TERRACOTTA: u32 = 53;
pub const BLOCK_WOOL_WHITE: u32 = 54;
pub const BLOCK_BASALT_TILES: u32 = 55;
pub const BLOCK_COPPER_WEAVE: u32 = 56;
pub const BLOCK_NEBULA_STRATA: u32 = 57;
pub const BLOCK_STARFORGED_CORE: u32 = 58;
pub const BLOCK_CRYO_CIRCUIT: u32 = 59;
pub const BLOCK_SMOKED_GLASS: u32 = 60;
pub const BLOCK_IVORY_MARBLE: u32 = 61;
pub const BLOCK_RUNIC_ALLOY: u32 = 62;
pub const BLOCK_HYPERPHASE_GEL: u32 = 63;
pub const BLOCK_SINGULARITY_CORE: u32 = 64;
pub const BLOCK_CHRONO_BLOOM: u32 = 65;
pub const BLOCK_TESSERACT_WEAVE: u32 = 66;
pub const BLOCK_EVENTIDE_ALLOY: u32 = 67;
pub const BLOCK_BEACON_MATRIX: u32 = 68;

/// Resolve (namespace, block_type) to a GPU material appearance index (u8).
///
/// For the core polychora namespace, block_type maps 1:1 to the legacy material ID.
/// Unknown blocks fall back to material 1 (Red).
pub fn block_to_material_appearance(namespace: u32, block_type: u32) -> u8 {
    if namespace == NAMESPACE_POLYCHORA && block_type <= MAX_MATERIAL_ID as u32 {
        block_type as u8
    } else {
        1 // fallback to Red
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MaterialCategory {
    Basic,    // Simple colored blocks
    Natural,  // Dirt, stone, sand, gravel
    Wood,     // Planks, logs
    Ore,      // Ores and minerals
    Building, // Bricks, concrete, sandstone
    Glass,    // Transparent/translucent
    Light,    // Emissive/light sources
    Special,  // Mirrors, animated, effects
}

impl MaterialCategory {
    pub fn label(self) -> &'static str {
        match self {
            MaterialCategory::Basic => "Basic",
            MaterialCategory::Natural => "Natural",
            MaterialCategory::Wood => "Wood",
            MaterialCategory::Ore => "Ore",
            MaterialCategory::Building => "Building",
            MaterialCategory::Glass => "Glass",
            MaterialCategory::Light => "Light",
            MaterialCategory::Special => "Special",
        }
    }

    pub const ALL: [MaterialCategory; 8] = [
        MaterialCategory::Basic,
        MaterialCategory::Natural,
        MaterialCategory::Wood,
        MaterialCategory::Ore,
        MaterialCategory::Building,
        MaterialCategory::Glass,
        MaterialCategory::Light,
        MaterialCategory::Special,
    ];
}

pub struct MaterialInfo {
    pub id: u8,
    pub name: &'static str,
    pub category: MaterialCategory,
    pub color: [u8; 3],
}

pub const MAX_MATERIAL_ID: u8 = 68;

// All material definitions (IDs 1-68)
pub const MATERIALS: &[MaterialInfo] = &[
    // Basic colored blocks (1-8)
    MaterialInfo {
        id: 1,
        name: "Red",
        category: MaterialCategory::Basic,
        color: [255, 0, 0],
    },
    MaterialInfo {
        id: 2,
        name: "Orange",
        category: MaterialCategory::Basic,
        color: [255, 200, 0],
    },
    MaterialInfo {
        id: 3,
        name: "Yellow-Green",
        category: MaterialCategory::Basic,
        color: [128, 255, 0],
    },
    MaterialInfo {
        id: 4,
        name: "Green",
        category: MaterialCategory::Basic,
        color: [0, 255, 51],
    },
    MaterialInfo {
        id: 5,
        name: "Cyan",
        category: MaterialCategory::Basic,
        color: [0, 255, 255],
    },
    MaterialInfo {
        id: 6,
        name: "Blue",
        category: MaterialCategory::Basic,
        color: [0, 51, 255],
    },
    MaterialInfo {
        id: 7,
        name: "Purple",
        category: MaterialCategory::Basic,
        color: [128, 0, 255],
    },
    MaterialInfo {
        id: 8,
        name: "Magenta",
        category: MaterialCategory::Basic,
        color: [255, 0, 204],
    },
    // Special materials (9-14)
    MaterialInfo {
        id: 9,
        name: "Rainbow",
        category: MaterialCategory::Special,
        color: [180, 180, 180],
    },
    MaterialInfo {
        id: 10,
        name: "Brown",
        category: MaterialCategory::Basic,
        color: [139, 69, 20],
    },
    MaterialInfo {
        id: 11,
        name: "Grid Floor",
        category: MaterialCategory::Special,
        color: [115, 120, 125],
    },
    MaterialInfo {
        id: 12,
        name: "White",
        category: MaterialCategory::Basic,
        color: [255, 255, 255],
    },
    MaterialInfo {
        id: 13,
        name: "Light",
        category: MaterialCategory::Light,
        color: [255, 255, 220],
    },
    MaterialInfo {
        id: 14,
        name: "Mirror",
        category: MaterialCategory::Special,
        color: [220, 220, 230],
    },
    // Animated/special materials (15-26)
    MaterialInfo {
        id: 15,
        name: "Lava-Veined Basalt",
        category: MaterialCategory::Special,
        color: [140, 55, 20],
    },
    MaterialInfo {
        id: 16,
        name: "Crystal Lattice",
        category: MaterialCategory::Special,
        color: [130, 180, 220],
    },
    MaterialInfo {
        id: 17,
        name: "Marble",
        category: MaterialCategory::Building,
        color: [210, 214, 224],
    },
    MaterialInfo {
        id: 18,
        name: "Oxidized Metal",
        category: MaterialCategory::Special,
        color: [130, 100, 80],
    },
    MaterialInfo {
        id: 19,
        name: "Bio-Spore Moss",
        category: MaterialCategory::Special,
        color: [30, 80, 35],
    },
    MaterialInfo {
        id: 20,
        name: "Void Mirror",
        category: MaterialCategory::Special,
        color: [20, 30, 55],
    },
    MaterialInfo {
        id: 21,
        name: "Avatar Marker",
        category: MaterialCategory::Special,
        color: [50, 58, 85],
    },
    MaterialInfo {
        id: 22,
        name: "Holographic Laminate",
        category: MaterialCategory::Special,
        color: [40, 100, 170],
    },
    MaterialInfo {
        id: 23,
        name: "Tidal Glass",
        category: MaterialCategory::Glass,
        color: [25, 75, 140],
    },
    MaterialInfo {
        id: 24,
        name: "Circuit Weave",
        category: MaterialCategory::Special,
        color: [50, 130, 80],
    },
    MaterialInfo {
        id: 25,
        name: "Aurora Stone",
        category: MaterialCategory::Special,
        color: [65, 40, 100],
    },
    MaterialInfo {
        id: 26,
        name: "Hazard Chevrons",
        category: MaterialCategory::Special,
        color: [180, 120, 20],
    },
    // Natural materials (27-30)
    MaterialInfo {
        id: 27,
        name: "Stone",
        category: MaterialCategory::Natural,
        color: [125, 128, 133],
    },
    MaterialInfo {
        id: 28,
        name: "Cobblestone",
        category: MaterialCategory::Natural,
        color: [115, 117, 120],
    },
    MaterialInfo {
        id: 29,
        name: "Dirt",
        category: MaterialCategory::Natural,
        color: [110, 77, 46],
    },
    MaterialInfo {
        id: 30,
        name: "Coarse Dirt",
        category: MaterialCategory::Natural,
        color: [105, 74, 43],
    },
    // Wood materials (31-34)
    MaterialInfo {
        id: 31,
        name: "Oak Planks",
        category: MaterialCategory::Wood,
        color: [153, 117, 61],
    },
    MaterialInfo {
        id: 32,
        name: "Spruce Planks",
        category: MaterialCategory::Wood,
        color: [97, 68, 36],
    },
    MaterialInfo {
        id: 33,
        name: "Log Bark",
        category: MaterialCategory::Wood,
        color: [77, 53, 28],
    },
    MaterialInfo {
        id: 34,
        name: "Log End Rings",
        category: MaterialCategory::Wood,
        color: [145, 112, 62],
    },
    // New natural materials (35-40)
    MaterialInfo {
        id: 35,
        name: "Sand",
        category: MaterialCategory::Natural,
        color: [237, 201, 175],
    },
    MaterialInfo {
        id: 36,
        name: "Gravel",
        category: MaterialCategory::Natural,
        color: [131, 126, 126],
    },
    MaterialInfo {
        id: 37,
        name: "Clay",
        category: MaterialCategory::Natural,
        color: [160, 166, 179],
    },
    MaterialInfo {
        id: 38,
        name: "Grass Block",
        category: MaterialCategory::Natural,
        color: [115, 162, 75],
    },
    MaterialInfo {
        id: 39,
        name: "Snow",
        category: MaterialCategory::Natural,
        color: [248, 248, 255],
    },
    MaterialInfo {
        id: 40,
        name: "Ice",
        category: MaterialCategory::Glass,
        color: [145, 180, 240],
    },
    // Ore materials (41-45)
    MaterialInfo {
        id: 41,
        name: "Coal Ore",
        category: MaterialCategory::Ore,
        color: [85, 85, 85],
    },
    MaterialInfo {
        id: 42,
        name: "Iron Ore",
        category: MaterialCategory::Ore,
        color: [200, 155, 140],
    },
    MaterialInfo {
        id: 43,
        name: "Gold Ore",
        category: MaterialCategory::Ore,
        color: [255, 215, 0],
    },
    MaterialInfo {
        id: 44,
        name: "Diamond Ore",
        category: MaterialCategory::Ore,
        color: [90, 220, 220],
    },
    MaterialInfo {
        id: 45,
        name: "Redstone Ore",
        category: MaterialCategory::Ore,
        color: [200, 50, 50],
    },
    // Additional wood and building materials (46-48)
    MaterialInfo {
        id: 46,
        name: "Birch Planks",
        category: MaterialCategory::Wood,
        color: [216, 205, 163],
    },
    MaterialInfo {
        id: 47,
        name: "Bricks",
        category: MaterialCategory::Building,
        color: [150, 97, 83],
    },
    MaterialInfo {
        id: 48,
        name: "Sandstone",
        category: MaterialCategory::Building,
        color: [228, 208, 168],
    },
    // Glass and light materials (49-51)
    MaterialInfo {
        id: 49,
        name: "Glass",
        category: MaterialCategory::Glass,
        color: [200, 220, 230],
    },
    MaterialInfo {
        id: 50,
        name: "Glowstone",
        category: MaterialCategory::Light,
        color: [255, 200, 100],
    },
    MaterialInfo {
        id: 51,
        name: "Obsidian",
        category: MaterialCategory::Natural,
        color: [20, 18, 29],
    },
    // Special decorative materials (52-54)
    MaterialInfo {
        id: 52,
        name: "Prismarine",
        category: MaterialCategory::Building,
        color: [99, 171, 158],
    },
    MaterialInfo {
        id: 53,
        name: "Terracotta",
        category: MaterialCategory::Building,
        color: [152, 94, 67],
    },
    MaterialInfo {
        id: 54,
        name: "Wool (White)",
        category: MaterialCategory::Building,
        color: [233, 236, 236],
    },
    // Advanced structural materials (55-62)
    MaterialInfo {
        id: 55,
        name: "Basalt Tiles",
        category: MaterialCategory::Building,
        color: [68, 70, 76],
    },
    MaterialInfo {
        id: 56,
        name: "Copper Weave",
        category: MaterialCategory::Special,
        color: [168, 105, 72],
    },
    MaterialInfo {
        id: 57,
        name: "Nebula Strata",
        category: MaterialCategory::Special,
        color: [69, 78, 126],
    },
    MaterialInfo {
        id: 58,
        name: "Starforged Core",
        category: MaterialCategory::Light,
        color: [255, 236, 168],
    },
    MaterialInfo {
        id: 59,
        name: "Cryo Circuit",
        category: MaterialCategory::Special,
        color: [118, 188, 206],
    },
    MaterialInfo {
        id: 60,
        name: "Smoked Glass",
        category: MaterialCategory::Glass,
        color: [78, 98, 115],
    },
    MaterialInfo {
        id: 61,
        name: "Ivory Marble",
        category: MaterialCategory::Building,
        color: [226, 228, 232],
    },
    MaterialInfo {
        id: 62,
        name: "Runic Alloy",
        category: MaterialCategory::Special,
        color: [140, 146, 158],
    },
    // Volumetric animated materials (63-68)
    MaterialInfo {
        id: 63,
        name: "Hyperphase Gel",
        category: MaterialCategory::Glass,
        color: [102, 208, 255],
    },
    MaterialInfo {
        id: 64,
        name: "Singularity Core",
        category: MaterialCategory::Light,
        color: [255, 168, 110],
    },
    MaterialInfo {
        id: 65,
        name: "Chrono Bloom",
        category: MaterialCategory::Special,
        color: [146, 255, 164],
    },
    MaterialInfo {
        id: 66,
        name: "Tesseract Weave",
        category: MaterialCategory::Special,
        color: [170, 132, 255],
    },
    MaterialInfo {
        id: 67,
        name: "Eventide Alloy",
        category: MaterialCategory::Special,
        color: [112, 130, 168],
    },
    MaterialInfo {
        id: 68,
        name: "Beacon Matrix",
        category: MaterialCategory::Light,
        color: [255, 248, 196],
    },
];

/// Get the name of a material by ID
pub fn material_name(id: u8) -> &'static str {
    MATERIALS
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.name)
        .unwrap_or("Unknown")
}

/// Get the color of a material by ID (RGB)
pub fn material_color(id: u8) -> [u8; 3] {
    MATERIALS
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.color)
        .unwrap_or([128, 128, 128])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_lookup() {
        assert_eq!(material_name(1), "Red");
        assert_eq!(material_name(27), "Stone");
        assert_eq!(material_name(35), "Sand");
        assert_eq!(material_name(68), "Beacon Matrix");
    }

    #[test]
    fn test_max_material_id() {
        assert_eq!(MAX_MATERIAL_ID, 68);
        assert_eq!(
            MATERIALS.iter().map(|m| m.id).max().unwrap_or(0),
            MAX_MATERIAL_ID
        );
    }

    #[test]
    fn test_categories() {
        let natural: Vec<_> = MATERIALS
            .iter()
            .filter(|m| m.category == MaterialCategory::Natural)
            .collect();
        assert!(natural.len() > 0);

        let ores: Vec<_> = MATERIALS
            .iter()
            .filter(|m| m.category == MaterialCategory::Ore)
            .collect();
        assert_eq!(ores.len(), 5);
    }

    #[test]
    fn test_material_colors() {
        assert_eq!(material_color(1), [255, 0, 0]); // Red
        assert_eq!(material_color(12), [255, 255, 255]); // White
        assert_eq!(material_color(35), [237, 201, 175]); // Sand
        assert_eq!(material_color(58), [255, 236, 168]); // Starforged Core
        assert_eq!(material_color(68), [255, 248, 196]); // Beacon Matrix
    }
}
