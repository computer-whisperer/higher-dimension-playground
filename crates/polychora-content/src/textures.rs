use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::manifest::TextureDeclaration;
use polychora_plugin_api::texture::builtin_textures::*;
use polychora_plugin_api::texture::{TextureFormat, TexturePayload};

/// Texture declaration + sRGB u8 color for each solid-color block.
const SOLID_COLOR_TEXTURES: &[(u32, &str, [u8; 4])] = &[
    (TEX_RED, "Red", [255, 0, 0, 255]),
    (TEX_ORANGE, "Orange", [255, 204, 0, 255]),
    (TEX_YELLOW_GREEN, "Yellow-Green", [128, 255, 0, 255]),
    (TEX_GREEN, "Green", [0, 255, 51, 255]),
    (TEX_CYAN, "Cyan", [0, 255, 255, 255]),
    (TEX_BLUE, "Blue", [0, 51, 255, 255]),
    (TEX_PURPLE, "Purple", [128, 0, 255, 255]),
    (TEX_MAGENTA, "Magenta", [255, 0, 204, 255]),
    (TEX_BROWN, "Brown", [39, 69, 20, 255]),
    (TEX_WHITE, "White", [255, 255, 255, 255]),
];

/// 1x1x1 3D texture declarations for the 10 solid-color blocks.
pub fn texture_declarations() -> Vec<TextureDeclaration> {
    SOLID_COLOR_TEXTURES
        .iter()
        .map(|&(texture_id, name, _)| TextureDeclaration {
            texture_id,
            name: String::from(name),
            width: 1,
            height: 1,
            depth: 1,
            format: TextureFormat::Rgba8Srgb,
        })
        .collect()
}

/// 1x1x1 3D texture payloads (4 bytes each) for the 10 solid-color blocks.
pub fn texture_payloads() -> Vec<TexturePayload> {
    SOLID_COLOR_TEXTURES
        .iter()
        .map(|&(texture_id, _, rgba)| TexturePayload {
            texture_id,
            width: 1,
            height: 1,
            depth: 1,
            format: TextureFormat::Rgba8Srgb,
            data: rgba.to_vec(),
        })
        .collect()
}
