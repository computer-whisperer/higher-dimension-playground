use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::block::BlockCategory;
use crate::entity::{EntityCategory, EntitySimConfig};
use crate::texture::{TextureFormat, TextureRef};

/// Declares a plugin's content to the host at load time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PluginManifest {
    pub namespace_id: u32,
    pub name: String,
    pub version: [u32; 3],
    pub blocks: Vec<BlockDeclaration>,
    pub entities: Vec<EntityDeclaration>,
    pub items: Vec<ItemDeclaration>,
    pub textures: Vec<TextureDeclaration>,
}

/// A block type declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockDeclaration {
    pub type_id: u32,
    pub name: String,
    pub category: BlockCategory,
    pub color_hint: [u8; 3],
    pub texture: TextureRef,
    pub transparent: bool,
    pub light_emission: u8,
    /// If set, the server will periodically tick instances of this block type
    /// via `OP_BLOCK_TICK`.
    #[serde(default)]
    pub tick_config: Option<BlockTickConfig>,
}

/// Configuration for server-side block ticking, declared per block type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockTickConfig {
    /// Minimum interval between ticks for each instance, in milliseconds.
    pub interval_ms: u64,
    /// Instances are only ticked when a player is within this distance (in voxels).
    pub activation_radius: f32,
}

/// An entity type declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityDeclaration {
    pub type_id: u32,
    pub name: String,
    pub category: EntityCategory,
    pub default_scale: f32,
    pub base_material_color: [u8; 3],
    /// Explicit texture palette for entity model parts.
    /// Each entry maps a palette slot to a specific texture.
    /// Rendering code indexes into this palette instead of using
    /// base_material_token + saturating_add(N).
    #[serde(default)]
    pub model_textures: Vec<TextureRef>,
    /// Texture ID (namespace 0) for the spawn egg icon in the material icon sheet.
    /// Zero means no dedicated spawn egg icon (falls back to base_material_color).
    #[serde(default)]
    pub spawn_egg_texture_id: u32,
    /// Data-driven simulation configuration. Present for entities with WASM-driven ticks.
    #[serde(default)]
    pub sim_config: Option<EntitySimConfig>,
}

/// How an item renders as a dropped entity in the world.
///
/// Contains a texture palette (like entity `model_textures`) used for the
/// in-world floating item model.  If `textures` is empty, the renderer falls
/// back to a single cube colored by `color_hint`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ItemWorldModel {
    pub textures: Vec<TextureRef>,
}

/// How an item renders as a thumbnail in the inventory UI.
///
/// If `texture` is set, the UI can render a GPU-rendered cube icon using that
/// texture (like block icons in the material icon sheet).  If `None`, the UI
/// falls back to a flat color swatch from `color_hint`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ItemThumbnail {
    pub texture: Option<TextureRef>,
}

/// An item type declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemDeclaration {
    pub type_id: u32,
    pub name: String,
    #[serde(default = "default_max_stack_size")]
    pub max_stack_size: u32,
    #[serde(default)]
    pub color_hint: [u8; 3],
    /// In-world model configuration for dropped item entities.
    #[serde(default)]
    pub world_model: ItemWorldModel,
    /// Inventory thumbnail configuration.
    #[serde(default)]
    pub thumbnail: ItemThumbnail,
}

fn default_max_stack_size() -> u32 {
    64
}

/// A texture asset declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureDeclaration {
    pub texture_id: u32,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: TextureFormat,
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::entity::{MobAbilityParams, MobLocomotionMode, SimulationMode};
    use alloc::string::String;
    use alloc::vec;
    use super::BlockTickConfig;

    #[test]
    fn manifest_postcard_round_trip() {
        let manifest = PluginManifest {
            namespace_id: 0x706f6c79,
            name: String::from("test-plugin"),
            version: [1, 2, 3],
            blocks: vec![BlockDeclaration {
                type_id: 0xdeadbeef,
                name: String::from("TestBlock"),
                category: BlockCategory::Basic,
                color_hint: [255, 0, 128],
                texture: TextureRef {
                    namespace: 0,
                    texture_id: 0xb3b1799d,
                },
                transparent: true,
                light_emission: 15,
                tick_config: None,
            }],
            entities: vec![EntityDeclaration {
                type_id: 0xcafebabe,
                name: String::from("TestEntity"),
                category: EntityCategory::Mob,
                default_scale: 1.5,
                base_material_color: [10, 20, 30],
                model_textures: vec![],
                spawn_egg_texture_id: 0xdeadcafe,
                sim_config: Some(EntitySimConfig {
                    mode: SimulationMode::PhysicsDriven,
                    locomotion: MobLocomotionMode::Walking,
                    move_speed: 3.0,
                    preferred_distance: 2.6,
                    tangent_weight: 0.72,
                    aliases: vec![String::from("testalias")],
                    nav_target_y_offset: 0.0,
                    ability_params: Some(MobAbilityParams {
                        detonate_trigger_distance: 1.55,
                        detonate_radius_voxels: 3,
                        detonate_impulse_radius: 7.0,
                        detonate_max_impulse_distance: 5.0,
                        blink_min_interval_ms: 720,
                        blink_max_interval_ms: 1520,
                        blink_distance: 2.8,
                        blink_min_distance: 1.0,
                        blink_blocked_progress_epsilon: 0.08,
                    }),
                }),
            }],
            items: vec![ItemDeclaration {
                type_id: 0x12345678,
                name: String::from("TestItem"),
                max_stack_size: 16,
                color_hint: [255, 128, 0],
                world_model: ItemWorldModel {
                    textures: vec![TextureRef {
                        namespace: 0,
                        texture_id: 0xb3b1799d,
                    }],
                },
                thumbnail: ItemThumbnail {
                    texture: Some(TextureRef {
                        namespace: 0,
                        texture_id: 0xb3b1799d,
                    }),
                },
            }],
            textures: vec![TextureDeclaration {
                texture_id: 0xaabbccdd,
                name: String::from("TestTexture"),
                width: 64,
                height: 64,
                depth: 16,
                format: crate::texture::TextureFormat::Rgba8Srgb,
            }],
        };

        let bytes = postcard::to_allocvec(&manifest).expect("serialize");
        let deserialized: PluginManifest = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(manifest, deserialized);
    }
}
