use serde::{Deserialize, Serialize};

/// Categorisation for blocks.  Identical semantics to the former
/// `MaterialCategory` in `materials.rs`; the rename reflects the shift
/// from "material" (GPU procedural ID) to "block" (gameplay object).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlockCategory {
    Basic,
    Natural,
    Wood,
    Ore,
    Building,
    Glass,
    Light,
    Special,
}

impl BlockCategory {
    pub fn label(self) -> &'static str {
        match self {
            BlockCategory::Basic => "Basic",
            BlockCategory::Natural => "Natural",
            BlockCategory::Wood => "Wood",
            BlockCategory::Ore => "Ore",
            BlockCategory::Building => "Building",
            BlockCategory::Glass => "Glass",
            BlockCategory::Light => "Light",
            BlockCategory::Special => "Special",
        }
    }

    pub const ALL: [BlockCategory; 8] = [
        BlockCategory::Basic,
        BlockCategory::Natural,
        BlockCategory::Wood,
        BlockCategory::Ore,
        BlockCategory::Building,
        BlockCategory::Glass,
        BlockCategory::Light,
        BlockCategory::Special,
    ];
}
