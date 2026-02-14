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

pub struct MaterialInfo {
    pub id: u8,
    pub name: &'static str,
    pub category: MaterialCategory,
}

// All material definitions (IDs 1-54)
static MATERIALS: &[MaterialInfo] = &[
    // Basic colored blocks (1-8)
    MaterialInfo { id: 1, name: "Red", category: MaterialCategory::Basic },
    MaterialInfo { id: 2, name: "Orange", category: MaterialCategory::Basic },
    MaterialInfo { id: 3, name: "Yellow-Green", category: MaterialCategory::Basic },
    MaterialInfo { id: 4, name: "Green", category: MaterialCategory::Basic },
    MaterialInfo { id: 5, name: "Cyan", category: MaterialCategory::Basic },
    MaterialInfo { id: 6, name: "Blue", category: MaterialCategory::Basic },
    MaterialInfo { id: 7, name: "Purple", category: MaterialCategory::Basic },
    MaterialInfo { id: 8, name: "Magenta", category: MaterialCategory::Basic },

    // Special materials (9-14)
    MaterialInfo { id: 9, name: "Rainbow", category: MaterialCategory::Special },
    MaterialInfo { id: 10, name: "Brown", category: MaterialCategory::Basic },
    MaterialInfo { id: 11, name: "Grid Floor", category: MaterialCategory::Special },
    MaterialInfo { id: 12, name: "White", category: MaterialCategory::Basic },
    MaterialInfo { id: 13, name: "Light", category: MaterialCategory::Light },
    MaterialInfo { id: 14, name: "Mirror", category: MaterialCategory::Special },

    // Animated/special materials (15-26)
    MaterialInfo { id: 15, name: "Lava-Veined Basalt", category: MaterialCategory::Special },
    MaterialInfo { id: 16, name: "Crystal Lattice", category: MaterialCategory::Special },
    MaterialInfo { id: 17, name: "Marble", category: MaterialCategory::Building },
    MaterialInfo { id: 18, name: "Oxidized Metal", category: MaterialCategory::Special },
    MaterialInfo { id: 19, name: "Bio-Spore Moss", category: MaterialCategory::Special },
    MaterialInfo { id: 20, name: "Void Mirror", category: MaterialCategory::Special },
    MaterialInfo { id: 21, name: "Avatar Marker", category: MaterialCategory::Special },
    MaterialInfo { id: 22, name: "Holographic Laminate", category: MaterialCategory::Special },
    MaterialInfo { id: 23, name: "Tidal Glass", category: MaterialCategory::Glass },
    MaterialInfo { id: 24, name: "Circuit Weave", category: MaterialCategory::Special },
    MaterialInfo { id: 25, name: "Aurora Stone", category: MaterialCategory::Special },
    MaterialInfo { id: 26, name: "Hazard Chevrons", category: MaterialCategory::Special },

    // Natural materials (27-30)
    MaterialInfo { id: 27, name: "Stone", category: MaterialCategory::Natural },
    MaterialInfo { id: 28, name: "Cobblestone", category: MaterialCategory::Natural },
    MaterialInfo { id: 29, name: "Dirt", category: MaterialCategory::Natural },
    MaterialInfo { id: 30, name: "Coarse Dirt", category: MaterialCategory::Natural },

    // Wood materials (31-34)
    MaterialInfo { id: 31, name: "Oak Planks", category: MaterialCategory::Wood },
    MaterialInfo { id: 32, name: "Spruce Planks", category: MaterialCategory::Wood },
    MaterialInfo { id: 33, name: "Log Bark", category: MaterialCategory::Wood },
    MaterialInfo { id: 34, name: "Log End Rings", category: MaterialCategory::Wood },

    // New natural materials (35-40)
    MaterialInfo { id: 35, name: "Sand", category: MaterialCategory::Natural },
    MaterialInfo { id: 36, name: "Gravel", category: MaterialCategory::Natural },
    MaterialInfo { id: 37, name: "Clay", category: MaterialCategory::Natural },
    MaterialInfo { id: 38, name: "Grass Block", category: MaterialCategory::Natural },
    MaterialInfo { id: 39, name: "Snow", category: MaterialCategory::Natural },
    MaterialInfo { id: 40, name: "Ice", category: MaterialCategory::Glass },

    // Ore materials (41-45)
    MaterialInfo { id: 41, name: "Coal Ore", category: MaterialCategory::Ore },
    MaterialInfo { id: 42, name: "Iron Ore", category: MaterialCategory::Ore },
    MaterialInfo { id: 43, name: "Gold Ore", category: MaterialCategory::Ore },
    MaterialInfo { id: 44, name: "Diamond Ore", category: MaterialCategory::Ore },
    MaterialInfo { id: 45, name: "Redstone Ore", category: MaterialCategory::Ore },

    // Additional wood and building materials (46-48)
    MaterialInfo { id: 46, name: "Birch Planks", category: MaterialCategory::Wood },
    MaterialInfo { id: 47, name: "Bricks", category: MaterialCategory::Building },
    MaterialInfo { id: 48, name: "Sandstone", category: MaterialCategory::Building },

    // Glass and light materials (49-51)
    MaterialInfo { id: 49, name: "Glass", category: MaterialCategory::Glass },
    MaterialInfo { id: 50, name: "Glowstone", category: MaterialCategory::Light },
    MaterialInfo { id: 51, name: "Obsidian", category: MaterialCategory::Natural },

    // Special decorative materials (52-54)
    MaterialInfo { id: 52, name: "Prismarine", category: MaterialCategory::Building },
    MaterialInfo { id: 53, name: "Terracotta", category: MaterialCategory::Building },
    MaterialInfo { id: 54, name: "Wool (White)", category: MaterialCategory::Building },
];

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
}
