use crate::materials;
use common::{MatN, ModelInstance};
use higher_dimension_playground::render::{
    FrameParams, RenderBackend, RenderContext, RenderOptions, TetraFrameInput,
};
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;

pub const ICON_SIZE: u32 = 64;
const SHEET_COLUMNS: u32 = 10;
const ICON_FOCAL_LENGTH_XY: f32 = 4.0;
const ICON_FOCAL_LENGTH_ZW: f32 = 6.0;

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

fn icon_view_matrix() -> MatN<5> {
    let center = translate(-0.5, -0.5, -0.5, -0.5);
    let r1 = rot_xz(0.50);
    let r2 = rot_yz(-0.40);
    // Keep a subtle hidden-dimension tilt, but reduce it so icons stay crisp.
    let r3 = rot_xw(0.22);
    let push_back = translate(0.0, 0.15, 2.8, 2.8);
    push_back * r3 * r2 * r1 * center
}

fn icon_render_options() -> RenderOptions {
    RenderOptions {
        do_frame_clear: true,
        do_raster: true,
        do_raytrace: false,
        render_backend: RenderBackend::TetraRaster,
        do_edges: false,
        do_tetrahedron_edges: false,
        do_navigation_hud: false,
        zw_angle_color_shift_enabled: false,
        // Keep padding flags clear so rasterizer stays in non-overlay mode.
        zw_angle_color_shift_strength: 0.0,
        prepare_render_screenshot: true,
        ..Default::default()
    }
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

/// Generate a sprite sheet containing all material icons packed into a grid
/// by rendering the tetrahedron pipeline offscreen on GPU.
pub fn generate_material_icon_sheet_gpu(
    device: Arc<Device>,
    queue: Arc<Queue>,
    instance: Arc<Instance>,
) -> Option<MaterialIconSheet> {
    let num_materials = materials::MATERIALS.len() as u32;
    let rows = (num_materials + SHEET_COLUMNS - 1) / SHEET_COLUMNS;
    let sheet_w = SHEET_COLUMNS * ICON_SIZE;
    let sheet_h = rows * ICON_SIZE;

    let mut pixels = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    let mut uv_rects = HashMap::new();
    let icon_pixel_len = (ICON_SIZE * ICON_SIZE * 4) as usize;

    let mut offscreen = RenderContext::new(
        device.clone(),
        queue.clone(),
        instance,
        None,
        [ICON_SIZE, ICON_SIZE, 1],
    );
    let view_matrix: ndarray::Array2<f32> = icon_view_matrix().into();

    for (idx, mat) in materials::MATERIALS.iter().enumerate() {
        let col = (idx as u32) % SHEET_COLUMNS;
        let row = (idx as u32) / SHEET_COLUMNS;

        let model_instance = [ModelInstance {
            model_transform: MatN::<5>::identity(),
            cell_material_ids: [mat.id as u32; 8],
        }];

        offscreen.render_tetra_frame(
            device.clone(),
            queue.clone(),
            FrameParams {
                view_matrix: view_matrix.clone(),
                time_ticks_ms: 0,
                focal_length_xy: ICON_FOCAL_LENGTH_XY,
                focal_length_zw: ICON_FOCAL_LENGTH_ZW,
                render_options: icon_render_options(),
            },
            TetraFrameInput {
                model_instances: &model_instance,
            },
        );
        let Some((icon_w, icon_h, raw)) = offscreen.capture_rendered_frame_rgba8(true) else {
            eprintln!(
                "Failed to capture offscreen material icon render for material {}",
                mat.id
            );
            return None;
        };
        if icon_w != ICON_SIZE || icon_h != ICON_SIZE || raw.len() != icon_pixel_len {
            eprintln!(
                "Unexpected offscreen icon size for material {}: {}x{} ({} bytes)",
                mat.id,
                icon_w,
                icon_h,
                raw.len()
            );
            return None;
        }

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

    Some(MaterialIconSheet {
        pixels,
        width: sheet_w,
        height: sheet_h,
        uv_rects,
    })
}
