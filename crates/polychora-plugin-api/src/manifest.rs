use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::block::BlockCategory;
use crate::entity::EntityCategory;
use crate::texture::TextureRef;

/// Declares a plugin's content to the host at load time.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityDeclaration {
    pub type_id: u32,
    pub name: String,
    pub category: EntityCategory,
    pub default_scale: f32,
    pub base_material_color: [u8; 3],
}

/// An item type declared by a plugin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ItemDeclaration {
    pub type_id: u32,
    pub name: String,
}

/// A texture asset declared by a plugin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextureDeclaration {
    pub texture_id: u32,
    pub name: String,
    pub width: u32,
    pub height: u32,
}
