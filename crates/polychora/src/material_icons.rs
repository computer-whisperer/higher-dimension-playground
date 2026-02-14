use crate::cpu_render::{cpu_render, CpuRenderParams};
use crate::materials;
use common::{MatN, ModelInstance, ModelTetrahedron};
use std::collections::HashMap;

pub const ICON_SIZE: u32 = 64;
const SHEET_COLUMNS: u32 = 10;

/// Build a 5x5 homogeneous rotation matrix in the XZ plane (angle in radians).
fn rot_xz(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[0, 0]] = c;
    m[[0, 2]] = -s;
    m[[2, 0]] = s;
    m[[2, 2]] = c;
    m
}

/// Build a 5x5 homogeneous rotation matrix in the YZ plane.
fn rot_yz(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[1, 1]] = c;
    m[[1, 2]] = -s;
    m[[2, 1]] = s;
    m[[2, 2]] = c;
    m
}

/// Build a 5x5 homogeneous rotation matrix in the XW plane.
fn rot_xw(a: f32) -> MatN<5> {
    let (s, c) = a.sin_cos();
    let mut m = MatN::<5>::identity();
    m[[0, 0]] = c;
    m[[0, 3]] = -s;
    m[[3, 0]] = s;
    m[[3, 3]] = c;
    m
}

/// Build a 5x5 homogeneous translation matrix.
fn translate(dx: f32, dy: f32, dz: f32, dw: f32) -> MatN<5> {
    let mut m = MatN::<5>::identity();
    m[[0, 4]] = dx;
    m[[1, 4]] = dy;
    m[[2, 4]] = dz;
    m[[3, 4]] = dw;
    m
}

/// A sprite sheet containing all material icons packed into a single texture.
pub struct MaterialIconSheet {
    /// RGBA pixel data for the entire sprite sheet
    pub pixels: Vec<u8>,
    /// Width of the sprite sheet in pixels
    pub width: u32,
    /// Height of the sprite sheet in pixels
    pub height: u32,
    /// Map from material ID to its UV rectangle [u_min, v_min, u_max, v_max]
    uv_rects: HashMap<u8, [f32; 4]>,
}

impl MaterialIconSheet {
    /// Get the UV rectangle for a material ID, or None if not found.
    pub fn uv_rect(&self, material_id: u8) -> Option<[f32; 4]> {
        self.uv_rects.get(&material_id).copied()
    }
}

/// Generate a sprite sheet containing all material icons packed into a grid.
/// Returns the sheet with pixel data and UV lookup.
pub fn generate_material_icon_sheet(model_tets: &[ModelTetrahedron]) -> MaterialIconSheet {
    let num_materials = materials::MATERIALS.len() as u32;
    let rows = (num_materials + SHEET_COLUMNS - 1) / SHEET_COLUMNS;
    let sheet_w = SHEET_COLUMNS * ICON_SIZE;
    let sheet_h = rows * ICON_SIZE;

    let mut pixels = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    let mut uv_rects = HashMap::new();

    let center = translate(-0.5, -0.5, -0.5, -0.5);
    let r1 = rot_xz(0.35);
    let r2 = rot_yz(-0.30);
    let r3 = rot_xw(0.25);
    let push_back = translate(0.0, 0.0, 2.0, 2.0);
    let view_matrix = push_back * r3 * r2 * r1 * center;

    let params = CpuRenderParams {
        view_matrix,
        focal_length_xy: 1.8,
        focal_length_zw: 3.0,  // 3x narrower ZW FOV (30° instead of 90°) to fill frame
        width: ICON_SIZE,
        height: ICON_SIZE,
        ..Default::default()
    };

    for (idx, mat) in materials::MATERIALS.iter().enumerate() {
        let col = (idx as u32) % SHEET_COLUMNS;
        let row = (idx as u32) / SHEET_COLUMNS;

        let instance = ModelInstance {
            model_transform: MatN::<5>::identity(),
            cell_material_ids: [mat.id as u32; 8],
        };

        let img = cpu_render(&[instance], model_tets, &params);
        let raw = img.into_raw();

        // Copy icon pixels into the sprite sheet
        let dst_x = col * ICON_SIZE;
        let dst_y = row * ICON_SIZE;
        for py in 0..ICON_SIZE {
            let src_offset = (py * ICON_SIZE * 4) as usize;
            let dst_offset = ((dst_y + py) * sheet_w + dst_x) as usize * 4;
            pixels[dst_offset..dst_offset + (ICON_SIZE * 4) as usize]
                .copy_from_slice(&raw[src_offset..src_offset + (ICON_SIZE * 4) as usize]);
        }

        // Compute UV coordinates
        let u_min = dst_x as f32 / sheet_w as f32;
        let v_min = dst_y as f32 / sheet_h as f32;
        let u_max = (dst_x + ICON_SIZE) as f32 / sheet_w as f32;
        let v_max = (dst_y + ICON_SIZE) as f32 / sheet_h as f32;
        uv_rects.insert(mat.id, [u_min, v_min, u_max, v_max]);
    }

    MaterialIconSheet {
        pixels,
        width: sheet_w,
        height: sheet_h,
        uv_rects,
    }
}
