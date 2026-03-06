use alloc::string::String;
use alloc::vec::Vec;
use polychora_plugin_api::manifest::TextureDeclaration;
use polychora_plugin_api::content_ids::{self, *};
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

/// Size for entity 3D textures (4×4×4).
const ENTITY_TEX_SIZE: u32 = 4;

/// Simple hash for deterministic pseudo-random per-voxel variation.
fn hash3(x: u32, y: u32, z: u32) -> u32 {
    let mut h = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(z.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h ^ (h >> 16)
}

fn lerp_u8(a: u8, b: u8, t: u32, scale: u32) -> u8 {
    let a = a as u32;
    let b = b as u32;
    ((a * (scale - t) + b * t) / scale) as u8
}

fn gen_creeper_hide() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x, y, z) % 100;
                // Mottled dark green — base [30,51,38] with variation
                let r = if h < 20 { 22 } else if h < 60 { 30 } else { 36 };
                let g = if h < 20 { 42 } else if h < 60 { 51 } else { 58 };
                let b = if h < 20 { 30 } else if h < 60 { 38 } else { 44 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_creeper_dark() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x.wrapping_add(7), y, z) % 100;
                // Very dark green-black with faint variation
                let r = if h < 30 { 8 } else { 14 };
                let g = if h < 30 { 20 } else { 30 };
                let b = if h < 30 { 10 } else { 16 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_creeper_belly() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x, y.wrapping_add(3), z) % 100;
                // Lighter muted green with mottling
                let r = if h < 25 { 38 } else if h < 70 { 46 } else { 54 };
                let g = if h < 25 { 68 } else if h < 70 { 84 } else { 92 };
                let b = if h < 25 { 44 } else if h < 70 { 56 } else { 62 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_creeper_eyes() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Radial glow from center — bright yellow-green core, dimmer edges
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(200, 100, t, 100);
                let g = lerp_u8(230, 170, t, 100);
                let b = lerp_u8(50, 10, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_creeper_core() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Hot orange-red with bright center
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(255, 180, t, 100);
                let g = lerp_u8(120, 20, t, 100);
                let b = lerp_u8(40, 0, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

// --- Seeker textures (4×4×4 3D) ---

fn gen_seeker_chitin() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x, y, z) % 100;
                // Olive-green chitinous shell with mottled variation
                let r = if h < 25 { 48 } else if h < 65 { 58 } else { 66 };
                let g = if h < 25 { 62 } else if h < 65 { 74 } else { 82 };
                let b = if h < 25 { 22 } else if h < 65 { 28 } else { 34 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_seeker_glow() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Bioluminescent yellow-green glow, brighter at center
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(180, 90, t, 100);
                let g = lerp_u8(240, 160, t, 100);
                let b = lerp_u8(40, 10, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_seeker_joint() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x.wrapping_add(5), y, z) % 100;
                // Dark metallic joint segments
                let r = if h < 40 { 28 } else { 36 };
                let g = if h < 40 { 32 } else { 40 };
                let b = if h < 40 { 24 } else { 30 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_seeker_sensor() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Bright cyan sensory organ with radial glow
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(60, 20, t, 100);
                let g = lerp_u8(220, 140, t, 100);
                let b = lerp_u8(240, 160, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_seeker_belly() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x, y.wrapping_add(7), z) % 100;
                // Lighter tan-green underbelly
                let r = if h < 30 { 78 } else if h < 70 { 88 } else { 96 };
                let g = if h < 30 { 86 } else if h < 70 { 98 } else { 106 };
                let b = if h < 30 { 48 } else if h < 70 { 56 } else { 62 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

// --- Phase Spider textures (4×4×4 3D) ---

fn gen_spider_carapace() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x, y, z) % 100;
                // Deep indigo-purple crystalline shell
                let r = if h < 20 { 38 } else if h < 60 { 48 } else { 58 };
                let g = if h < 20 { 22 } else if h < 60 { 28 } else { 36 };
                let b = if h < 20 { 72 } else if h < 60 { 88 } else { 100 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_spider_phase() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Shifting blue-white phase energy, radial from center
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(180, 80, t, 100);
                let g = lerp_u8(200, 110, t, 100);
                let b = lerp_u8(255, 180, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_spider_core() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Bright white-violet dimensional core
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(230, 140, t, 100);
                let g = lerp_u8(200, 100, t, 100);
                let b = lerp_u8(255, 200, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_spider_web() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let h = hash3(x.wrapping_add(3), y.wrapping_add(5), z) % 100;
                // Translucent silver-grey webbing
                let r = if h < 30 { 140 } else if h < 70 { 160 } else { 180 };
                let g = if h < 30 { 138 } else if h < 70 { 156 } else { 174 };
                let b = if h < 30 { 150 } else if h < 70 { 170 } else { 190 };
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

fn gen_spider_eye() -> Vec<u8> {
    let n = ENTITY_TEX_SIZE;
    let mut data = Vec::with_capacity((n * n * n * 4) as usize);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                // Glowing magenta-pink sensor eye
                let cx = (x * 2 + 1) as i32 - n as i32;
                let cy = (y * 2 + 1) as i32 - n as i32;
                let cz = (z * 2 + 1) as i32 - n as i32;
                let dist_sq = (cx * cx + cy * cy + cz * cz) as u32;
                let max_sq = n * n * 3;
                let t = (dist_sq * 100 / max_sq).min(100);
                let r = lerp_u8(240, 160, t, 100);
                let g = lerp_u8(60, 20, t, 100);
                let b = lerp_u8(200, 120, t, 100);
                data.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    data
}

struct EntityTex {
    id: u32,
    name: &'static str,
    gen: fn() -> Vec<u8>,
}

const CREEPER_TEXTURES: &[EntityTex] = &[
    EntityTex { id: TEX_CREEPER_HIDE,  name: "Creeper Hide",  gen: gen_creeper_hide },
    EntityTex { id: TEX_CREEPER_DARK,  name: "Creeper Dark",  gen: gen_creeper_dark },
    EntityTex { id: TEX_CREEPER_BELLY, name: "Creeper Belly", gen: gen_creeper_belly },
    EntityTex { id: TEX_CREEPER_EYES,  name: "Creeper Eyes",  gen: gen_creeper_eyes },
    EntityTex { id: TEX_CREEPER_CORE,  name: "Creeper Core",  gen: gen_creeper_core },
];

const SEEKER_TEXTURES: &[EntityTex] = &[
    EntityTex { id: content_ids::TEX_SEEKER_CHITIN, name: "Seeker Chitin",  gen: gen_seeker_chitin },
    EntityTex { id: content_ids::TEX_SEEKER_GLOW,   name: "Seeker Glow",    gen: gen_seeker_glow },
    EntityTex { id: content_ids::TEX_SEEKER_JOINT,  name: "Seeker Joint",   gen: gen_seeker_joint },
    EntityTex { id: content_ids::TEX_SEEKER_SENSOR, name: "Seeker Sensor",  gen: gen_seeker_sensor },
    EntityTex { id: content_ids::TEX_SEEKER_BELLY,  name: "Seeker Belly",   gen: gen_seeker_belly },
];

const SPIDER_TEXTURES: &[EntityTex] = &[
    EntityTex { id: content_ids::TEX_SPIDER_CARAPACE, name: "Spider Carapace", gen: gen_spider_carapace },
    EntityTex { id: content_ids::TEX_SPIDER_PHASE,    name: "Spider Phase",    gen: gen_spider_phase },
    EntityTex { id: content_ids::TEX_SPIDER_CORE,     name: "Spider Core",     gen: gen_spider_core },
    EntityTex { id: content_ids::TEX_SPIDER_WEB,      name: "Spider Web",      gen: gen_spider_web },
    EntityTex { id: content_ids::TEX_SPIDER_EYE,      name: "Spider Eye",      gen: gen_spider_eye },
];

const ALL_ENTITY_TEXTURES: &[&[EntityTex]] = &[
    CREEPER_TEXTURES,
    SEEKER_TEXTURES,
    SPIDER_TEXTURES,
];

pub fn texture_declarations() -> Vec<TextureDeclaration> {
    let mut decls: Vec<TextureDeclaration> = SOLID_COLOR_TEXTURES
        .iter()
        .map(|&(texture_id, name, _)| TextureDeclaration {
            texture_id,
            name: String::from(name),
            width: 1,
            height: 1,
            depth: 1,
            format: TextureFormat::Rgba8Srgb,
        })
        .collect();
    for set in ALL_ENTITY_TEXTURES {
        for et in *set {
            decls.push(TextureDeclaration {
                texture_id: et.id,
                name: String::from(et.name),
                width: ENTITY_TEX_SIZE,
                height: ENTITY_TEX_SIZE,
                depth: ENTITY_TEX_SIZE,
                format: TextureFormat::Rgba8Srgb,
            });
        }
    }
    decls
}

pub fn texture_payloads() -> Vec<TexturePayload> {
    let mut payloads: Vec<TexturePayload> = SOLID_COLOR_TEXTURES
        .iter()
        .map(|&(texture_id, _, rgba)| TexturePayload {
            texture_id,
            width: 1,
            height: 1,
            depth: 1,
            format: TextureFormat::Rgba8Srgb,
            data: rgba.to_vec(),
        })
        .collect();
    for set in ALL_ENTITY_TEXTURES {
        for et in *set {
            payloads.push(TexturePayload {
                texture_id: et.id,
                width: ENTITY_TEX_SIZE,
                height: ENTITY_TEX_SIZE,
                depth: ENTITY_TEX_SIZE,
                format: TextureFormat::Rgba8Srgb,
                data: (et.gen)(),
            });
        }
    }
    payloads
}
