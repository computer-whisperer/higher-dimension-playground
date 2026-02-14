// Material metadata system for Polychora 4D voxel game

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MaterialCategory {
    Basic,      // Simple colored blocks
    Natural,    // Dirt, stone, sand, gravel
    Wood,       // Planks, logs
    Ore,        // Ores and minerals
    Building,   // Bricks, concrete, sandstone
    Glass,      // Transparent/translucent
    Light,      // Emissive/light sources
    Special,    // Mirrors, animated, effects
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

// All material definitions (IDs 1-54)
pub const MATERIALS: &[MaterialInfo] = &[
    // Basic colored blocks (1-8)
    MaterialInfo { id: 1, name: "Red", category: MaterialCategory::Basic, color: [255, 0, 0] },
    MaterialInfo { id: 2, name: "Orange", category: MaterialCategory::Basic, color: [255, 200, 0] },
    MaterialInfo { id: 3, name: "Yellow-Green", category: MaterialCategory::Basic, color: [128, 255, 0] },
    MaterialInfo { id: 4, name: "Green", category: MaterialCategory::Basic, color: [0, 255, 51] },
    MaterialInfo { id: 5, name: "Cyan", category: MaterialCategory::Basic, color: [0, 255, 255] },
    MaterialInfo { id: 6, name: "Blue", category: MaterialCategory::Basic, color: [0, 51, 255] },
    MaterialInfo { id: 7, name: "Purple", category: MaterialCategory::Basic, color: [128, 0, 255] },
    MaterialInfo { id: 8, name: "Magenta", category: MaterialCategory::Basic, color: [255, 0, 204] },

    // Special materials (9-14)
    MaterialInfo { id: 9, name: "Rainbow", category: MaterialCategory::Special, color: [180, 180, 180] },
    MaterialInfo { id: 10, name: "Brown", category: MaterialCategory::Basic, color: [139, 69, 20] },
    MaterialInfo { id: 11, name: "Grid Floor", category: MaterialCategory::Special, color: [115, 120, 125] },
    MaterialInfo { id: 12, name: "White", category: MaterialCategory::Basic, color: [255, 255, 255] },
    MaterialInfo { id: 13, name: "Light", category: MaterialCategory::Light, color: [255, 255, 220] },
    MaterialInfo { id: 14, name: "Mirror", category: MaterialCategory::Special, color: [220, 220, 230] },

    // Animated/special materials (15-26)
    MaterialInfo { id: 15, name: "Lava-Veined Basalt", category: MaterialCategory::Special, color: [140, 55, 20] },
    MaterialInfo { id: 16, name: "Crystal Lattice", category: MaterialCategory::Special, color: [130, 180, 220] },
    MaterialInfo { id: 17, name: "Marble", category: MaterialCategory::Building, color: [210, 214, 224] },
    MaterialInfo { id: 18, name: "Oxidized Metal", category: MaterialCategory::Special, color: [130, 100, 80] },
    MaterialInfo { id: 19, name: "Bio-Spore Moss", category: MaterialCategory::Special, color: [30, 80, 35] },
    MaterialInfo { id: 20, name: "Void Mirror", category: MaterialCategory::Special, color: [20, 30, 55] },
    MaterialInfo { id: 21, name: "Avatar Marker", category: MaterialCategory::Special, color: [50, 58, 85] },
    MaterialInfo { id: 22, name: "Holographic Laminate", category: MaterialCategory::Special, color: [40, 100, 170] },
    MaterialInfo { id: 23, name: "Tidal Glass", category: MaterialCategory::Glass, color: [25, 75, 140] },
    MaterialInfo { id: 24, name: "Circuit Weave", category: MaterialCategory::Special, color: [50, 130, 80] },
    MaterialInfo { id: 25, name: "Aurora Stone", category: MaterialCategory::Special, color: [65, 40, 100] },
    MaterialInfo { id: 26, name: "Hazard Chevrons", category: MaterialCategory::Special, color: [180, 120, 20] },

    // Natural materials (27-30)
    MaterialInfo { id: 27, name: "Stone", category: MaterialCategory::Natural, color: [125, 128, 133] },
    MaterialInfo { id: 28, name: "Cobblestone", category: MaterialCategory::Natural, color: [115, 117, 120] },
    MaterialInfo { id: 29, name: "Dirt", category: MaterialCategory::Natural, color: [110, 77, 46] },
    MaterialInfo { id: 30, name: "Coarse Dirt", category: MaterialCategory::Natural, color: [105, 74, 43] },

    // Wood materials (31-34)
    MaterialInfo { id: 31, name: "Oak Planks", category: MaterialCategory::Wood, color: [153, 117, 61] },
    MaterialInfo { id: 32, name: "Spruce Planks", category: MaterialCategory::Wood, color: [97, 68, 36] },
    MaterialInfo { id: 33, name: "Log Bark", category: MaterialCategory::Wood, color: [77, 53, 28] },
    MaterialInfo { id: 34, name: "Log End Rings", category: MaterialCategory::Wood, color: [145, 112, 62] },

    // New natural materials (35-40)
    MaterialInfo { id: 35, name: "Sand", category: MaterialCategory::Natural, color: [237, 201, 175] },
    MaterialInfo { id: 36, name: "Gravel", category: MaterialCategory::Natural, color: [131, 126, 126] },
    MaterialInfo { id: 37, name: "Clay", category: MaterialCategory::Natural, color: [160, 166, 179] },
    MaterialInfo { id: 38, name: "Grass Block", category: MaterialCategory::Natural, color: [115, 162, 75] },
    MaterialInfo { id: 39, name: "Snow", category: MaterialCategory::Natural, color: [248, 248, 255] },
    MaterialInfo { id: 40, name: "Ice", category: MaterialCategory::Glass, color: [145, 180, 240] },

    // Ore materials (41-45)
    MaterialInfo { id: 41, name: "Coal Ore", category: MaterialCategory::Ore, color: [85, 85, 85] },
    MaterialInfo { id: 42, name: "Iron Ore", category: MaterialCategory::Ore, color: [200, 155, 140] },
    MaterialInfo { id: 43, name: "Gold Ore", category: MaterialCategory::Ore, color: [255, 215, 0] },
    MaterialInfo { id: 44, name: "Diamond Ore", category: MaterialCategory::Ore, color: [90, 220, 220] },
    MaterialInfo { id: 45, name: "Redstone Ore", category: MaterialCategory::Ore, color: [200, 50, 50] },

    // Additional wood and building materials (46-48)
    MaterialInfo { id: 46, name: "Birch Planks", category: MaterialCategory::Wood, color: [216, 205, 163] },
    MaterialInfo { id: 47, name: "Bricks", category: MaterialCategory::Building, color: [150, 97, 83] },
    MaterialInfo { id: 48, name: "Sandstone", category: MaterialCategory::Building, color: [228, 208, 168] },

    // Glass and light materials (49-51)
    MaterialInfo { id: 49, name: "Glass", category: MaterialCategory::Glass, color: [200, 220, 230] },
    MaterialInfo { id: 50, name: "Glowstone", category: MaterialCategory::Light, color: [255, 200, 100] },
    MaterialInfo { id: 51, name: "Obsidian", category: MaterialCategory::Natural, color: [20, 18, 29] },

    // Special decorative materials (52-54)
    MaterialInfo { id: 52, name: "Prismarine", category: MaterialCategory::Building, color: [99, 171, 158] },
    MaterialInfo { id: 53, name: "Terracotta", category: MaterialCategory::Building, color: [152, 94, 67] },
    MaterialInfo { id: 54, name: "Wool (White)", category: MaterialCategory::Building, color: [233, 236, 236] },
];

/// Get material info by ID
pub fn material_info(id: u8) -> Option<&'static MaterialInfo> {
    MATERIALS.iter().find(|m| m.id == id)
}

/// Get the name of a material by ID
pub fn material_name(id: u8) -> &'static str {
    MATERIALS
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.name)
        .unwrap_or("Unknown")
}

/// Get the category of a material by ID
pub fn material_category(id: u8) -> MaterialCategory {
    MATERIALS
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.category)
        .unwrap_or(MaterialCategory::Basic)
}

/// Get the color of a material by ID (RGB)
pub fn material_color(id: u8) -> [u8; 3] {
    MATERIALS
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.color)
        .unwrap_or([128, 128, 128])
}

/// Get all materials in a specific category
pub fn materials_in_category(cat: MaterialCategory) -> Vec<&'static MaterialInfo> {
    MATERIALS
        .iter()
        .filter(|m| m.category == cat)
        .collect()
}

/// Get all material info
pub fn all_materials() -> &'static [MaterialInfo] {
    MATERIALS
}

/// Get the maximum material ID
pub fn max_material_id() -> u8 {
    MATERIALS
        .iter()
        .map(|m| m.id)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_lookup() {
        assert_eq!(material_name(1), "Red");
        assert_eq!(material_name(27), "Stone");
        assert_eq!(material_name(35), "Sand");
        assert_eq!(material_name(54), "Wool (White)");
    }

    #[test]
    fn test_max_material_id() {
        assert_eq!(max_material_id(), 54);
    }

    #[test]
    fn test_categories() {
        let natural = materials_in_category(MaterialCategory::Natural);
        assert!(natural.len() > 0);

        let ores = materials_in_category(MaterialCategory::Ore);
        assert_eq!(ores.len(), 5);
    }

    #[test]
    fn test_material_colors() {
        assert_eq!(material_color(1), [255, 0, 0]); // Red
        assert_eq!(material_color(12), [255, 255, 255]); // White
        assert_eq!(material_color(35), [237, 201, 175]); // Sand
    }
}
