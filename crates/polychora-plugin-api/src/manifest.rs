use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::block::BlockCategory;
use crate::entity::{EntityCategory, EntitySimConfig};
use crate::texture::TextureRef;

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
    /// Data-driven simulation configuration. Present for entities with WASM-driven ticks.
    #[serde(default)]
    pub sim_config: Option<EntitySimConfig>,
}

/// An item type declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemDeclaration {
    pub type_id: u32,
    pub name: String,
}

/// A texture asset declared by a plugin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureDeclaration {
    pub texture_id: u32,
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::entity::{MobAbilityParams, MobLocomotionMode, SimulationMode};
    use alloc::string::String;
    use alloc::vec;

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
            }],
            entities: vec![EntityDeclaration {
                type_id: 0xcafebabe,
                name: String::from("TestEntity"),
                category: EntityCategory::Mob,
                default_scale: 1.5,
                base_material_color: [10, 20, 30],
                model_textures: vec![],
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
            }],
            textures: vec![TextureDeclaration {
                texture_id: 0xaabbccdd,
                name: String::from("TestTexture"),
                width: 64,
                height: 64,
            }],
        };

        let bytes = postcard::to_allocvec(&manifest).expect("serialize");
        let deserialized: PluginManifest =
            postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(manifest, deserialized);
    }
}
