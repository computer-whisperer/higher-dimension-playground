use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Input to OP_ENTITY_MODEL — host sends this per entity per frame.
#[derive(Serialize, Deserialize, Default)]
pub struct EntityModelInput {
    pub entity_ns: u32,
    pub entity_type: u32,
    #[serde(default)]
    pub entity_id: u64,
    #[serde(default)]
    pub elapsed_s: f32,
    #[serde(default)]
    pub speed_xzw: f32,
    #[serde(default)]
    pub scale: f32,
}

/// Output from OP_ENTITY_MODEL.
#[derive(Serialize, Deserialize, Default)]
pub struct EntityModelOutput {
    pub parts: Vec<EntityModelPart>,
}

/// A single tesseract part in entity-local space.
#[derive(Serialize, Deserialize, Default)]
pub struct EntityModelPart {
    /// Position offset in entity-local space (Y = up).
    #[serde(default)]
    pub offset: [f32; 4],
    /// Half-size per axis (already scaled by entity.scale in the plugin).
    #[serde(default)]
    pub half_extents: [f32; 4],
    /// Texture palette indices (0-9) for the 8 cells of a tesseract.
    #[serde(default)]
    pub cell_materials: [u8; 8],
}
