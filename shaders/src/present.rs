use spirv_std::spirv;

use glam::{vec4, Vec4, Vec2, UVec2};
use common::*;

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


#[spirv(fragment)]
pub fn main_buffer_fs(
    #[spirv(frag_coord)] in_frag_coord: Vec4,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &[Vec4],
    output: &mut glam::Vec4,
) {
    *output = Vec4::new(0.0, 0.0, 0.0, 1.0); // Default color
    let normal_pos = Vec2::new(
        in_frag_coord.x/(working_data.present_dimensions.x as f32),
        in_frag_coord.y/(working_data.present_dimensions.y as f32)); // Normalized to 0..1
    let pixel_pos = Vec2::new(
        working_data.render_dimensions.x as f32*normal_pos.x,
        working_data.render_dimensions.y as f32*normal_pos.y
    );
    let u_pixel_pos = UVec2::new(pixel_pos.x as u32, pixel_pos.y as u32);
    *output += pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize];
}