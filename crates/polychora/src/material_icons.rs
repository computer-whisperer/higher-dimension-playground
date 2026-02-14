use crate::cpu_render::{cpu_render, CpuRenderParams};
use crate::materials;
use common::{MatN, ModelInstance, ModelTetrahedron};
use std::collections::HashMap;

const ICON_SIZE: u32 = 64;

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

/// Generate 64x64 egui `ColorImage` icons for each material by CPU-rendering a
/// tesseract with that material applied to all 8 cells.
pub fn generate_material_icons(
    model_tets: &[ModelTetrahedron],
) -> HashMap<u8, egui::ColorImage> {
    let mut icons = HashMap::new();

    // View matrix: move the tesseract center (0.5,0.5,0.5,0.5) to in front of
    // the camera, then apply a slight rotation to reveal 4D structure.
    //
    // 1. Translate so the tesseract center is at origin.
    // 2. Apply rotations for a nice viewing angle.
    // 3. Translate along +Z/+W so the object is in front of the camera.
    let center = translate(-0.5, -0.5, -0.5, -0.5);
    let r1 = rot_xz(0.35);
    let r2 = rot_yz(-0.30);
    let r3 = rot_xw(0.25); // slight 4D rotation
    let push_back = translate(0.0, 0.0, 2.0, 2.0);
    // View matrix = push_back * rotations * center
    let view_matrix = push_back * r3 * r2 * r1 * center;

    let params = CpuRenderParams {
        view_matrix,
        focal_length_xy: 1.8,
        focal_length_zw: 1.0,
        width: ICON_SIZE,
        height: ICON_SIZE,
        ..Default::default()
    };

    for mat in materials::MATERIALS {
        let material_id = mat.id as u32;

        let instance = ModelInstance {
            model_transform: MatN::<5>::identity(),
            cell_material_ids: [material_id; 8],
        };

        let img = cpu_render(&[instance], model_tets, &params);

        // Convert image::RgbaImage -> egui::ColorImage
        let size = [img.width() as usize, img.height() as usize];
        let rgba_flat: Vec<u8> = img.into_raw();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &rgba_flat);

        icons.insert(mat.id, color_image);
    }

    icons
}
