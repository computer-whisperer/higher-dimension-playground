#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MaterialCategory {
    Basic,
    Natural,
    Wood,
    Building,
    Light,
    Special,
}

impl MaterialCategory {
    pub fn label(self) -> &'static str {
        match self {
            MaterialCategory::Basic => "Basic",
            MaterialCategory::Natural => "Natural",
            MaterialCategory::Wood => "Wood",
            MaterialCategory::Building => "Building",
            MaterialCategory::Light => "Light",
            MaterialCategory::Special => "Special",
        }
    }

    pub const ALL: [MaterialCategory; 6] = [
        MaterialCategory::Basic,
        MaterialCategory::Natural,
        MaterialCategory::Wood,
        MaterialCategory::Building,
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

pub const MATERIALS: &[MaterialInfo] = &[
    MaterialInfo { id: 1,  name: "Red",             category: MaterialCategory::Basic,    color: [255, 0, 0] },
    MaterialInfo { id: 2,  name: "Orange",          category: MaterialCategory::Basic,    color: [255, 200, 0] },
    MaterialInfo { id: 3,  name: "Yellow-Green",    category: MaterialCategory::Basic,    color: [128, 255, 0] },
    MaterialInfo { id: 4,  name: "Green",           category: MaterialCategory::Basic,    color: [0, 255, 51] },
    MaterialInfo { id: 5,  name: "Cyan",            category: MaterialCategory::Basic,    color: [0, 255, 255] },
    MaterialInfo { id: 6,  name: "Blue",            category: MaterialCategory::Basic,    color: [0, 51, 255] },
    MaterialInfo { id: 7,  name: "Purple",          category: MaterialCategory::Basic,    color: [128, 0, 255] },
    MaterialInfo { id: 8,  name: "Magenta",         category: MaterialCategory::Basic,    color: [255, 0, 204] },
    MaterialInfo { id: 9,  name: "Rainbow",         category: MaterialCategory::Special,  color: [180, 180, 180] },
    MaterialInfo { id: 10, name: "Brown",           category: MaterialCategory::Natural,  color: [39, 69, 20] },
    MaterialInfo { id: 11, name: "Grid Floor",      category: MaterialCategory::Building, color: [115, 120, 125] },
    MaterialInfo { id: 12, name: "White",           category: MaterialCategory::Basic,    color: [255, 255, 255] },
    MaterialInfo { id: 13, name: "Light",           category: MaterialCategory::Light,    color: [255, 255, 220] },
    MaterialInfo { id: 14, name: "Mirror",          category: MaterialCategory::Special,  color: [220, 220, 230] },
    MaterialInfo { id: 15, name: "Lava Basalt",     category: MaterialCategory::Natural,  color: [140, 55, 20] },
    MaterialInfo { id: 16, name: "Crystal",         category: MaterialCategory::Special,  color: [30, 120, 170] },
    MaterialInfo { id: 17, name: "Marble",          category: MaterialCategory::Building, color: [210, 214, 224] },
    MaterialInfo { id: 18, name: "Oxidized Metal",  category: MaterialCategory::Building, color: [130, 100, 80] },
    MaterialInfo { id: 19, name: "Bio-Spore Moss",  category: MaterialCategory::Natural,  color: [30, 80, 35] },
    MaterialInfo { id: 20, name: "Void Mirror",     category: MaterialCategory::Special,  color: [20, 30, 55] },
    MaterialInfo { id: 21, name: "Avatar",          category: MaterialCategory::Special,  color: [50, 58, 85] },
    MaterialInfo { id: 22, name: "Holographic",     category: MaterialCategory::Special,  color: [40, 100, 170] },
    MaterialInfo { id: 23, name: "Tidal Glass",     category: MaterialCategory::Special,  color: [25, 75, 140] },
    MaterialInfo { id: 24, name: "Circuit Weave",   category: MaterialCategory::Special,  color: [50, 130, 80] },
    MaterialInfo { id: 25, name: "Aurora Stone",    category: MaterialCategory::Special,  color: [65, 40, 100] },
    MaterialInfo { id: 26, name: "Hazard Chevrons", category: MaterialCategory::Building, color: [180, 120, 20] },
    MaterialInfo { id: 27, name: "Stone",           category: MaterialCategory::Natural,  color: [125, 128, 133] },
    MaterialInfo { id: 28, name: "Cobblestone",     category: MaterialCategory::Natural,  color: [115, 117, 120] },
    MaterialInfo { id: 29, name: "Dirt",            category: MaterialCategory::Natural,  color: [110, 77, 46] },
    MaterialInfo { id: 30, name: "Coarse Dirt",     category: MaterialCategory::Natural,  color: [105, 74, 43] },
    MaterialInfo { id: 31, name: "Oak Planks",      category: MaterialCategory::Wood,     color: [153, 117, 61] },
    MaterialInfo { id: 32, name: "Spruce Planks",   category: MaterialCategory::Wood,     color: [97, 68, 36] },
    MaterialInfo { id: 33, name: "Log Bark",        category: MaterialCategory::Wood,     color: [77, 53, 28] },
    MaterialInfo { id: 34, name: "Log End",         category: MaterialCategory::Wood,     color: [145, 112, 62] },
];

pub fn material_info(id: u8) -> Option<&'static MaterialInfo> {
    MATERIALS.iter().find(|m| m.id == id)
}

pub fn material_name(id: u8) -> &'static str {
    material_info(id).map(|m| m.name).unwrap_or("Unknown")
}

pub fn material_color(id: u8) -> [u8; 3] {
    material_info(id).map(|m| m.color).unwrap_or([128, 128, 128])
}

pub fn max_material_id() -> u8 {
    34
}
