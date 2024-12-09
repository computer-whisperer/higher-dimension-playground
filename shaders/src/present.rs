use spirv_std::spirv;

use glam::{vec4, Vec4, Vec2, UVec2};
use common::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[spirv(vertex)]
pub fn main_line_vs(
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] output_line_vertices: &mut [Vec2],) {
    // Do nothing for now
    *out_pos = vec4(
        output_line_vertices[vert_id as usize].x,
        output_line_vertices[vert_id as usize].y,
        0.0,
        1.0,
    );
}

#[spirv(vertex)]
pub fn main_buffer_vs(
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(vertex_index)] vert_id: i32) {
    let vertices = [
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new( 1.0, -1.0, 0.0, 1.0),
        Vec4::new( 1.0,  1.0, 0.0, 1.0),
        Vec4::new(-1.0, -1.0, 0.0, 1.0),
        Vec4::new(-1.0,  1.0, 0.0, 1.0),
        Vec4::new( 1.0,  1.0, 0.0, 1.0),
    ];
    *out_pos = vertices[vert_id as usize];
}

#[spirv(fragment)]
pub fn main_line_fs(output: &mut glam::Vec4) {
    *output = glam::vec4(0.0, 0.0, 1.0, 1.0);
}

fn linear_to_gamma(value: f32) -> f32 {
    // Apply gamma correction
    let gamma = 2.2;
    value.powf(1.0/gamma)
}

#[spirv(fragment)]
pub fn main_buffer_fs(
    #[spirv(frag_coord)] in_frag_coord: Vec4,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &[Vec4],
    output: &mut glam::Vec4,
) {
    *output = Vec4::new(0.0, 0.0, 0.0, 0.0); // Default color
    let normal_pos = Vec2::new(
        in_frag_coord.x/(working_data.present_dimensions.x as f32),
        in_frag_coord.y/(working_data.present_dimensions.y as f32)); // Normalized to 0..1
    let pixel_pos = Vec2::new(
        working_data.render_dimensions.x as f32*normal_pos.x,
        working_data.render_dimensions.y as f32*normal_pos.y
    );
    let u_pixel_pos = UVec2::new(pixel_pos.x as u32, pixel_pos.y as u32);
    let mut cs_result = Vec4::ZERO;
    for z in 0..working_data.render_dimensions.z {
        cs_result += pixel_buffer[(z*working_data.render_dimensions.x*working_data.render_dimensions.y + u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize];
    }

    let cs_result_adjusted = Vec4::new(
        linear_to_gamma(cs_result[0]/cs_result[3]),
        linear_to_gamma(cs_result[1]/cs_result[3]),
        linear_to_gamma(cs_result[2]/cs_result[3]),
        1.0
    );
    
    if cs_result[3] > 0.0 {
        *output += cs_result_adjusted;
    }
}