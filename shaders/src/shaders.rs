#![cfg_attr(target_arch = "spirv", no_std)]

use spirv_std::spirv;

pub use spirv_std::glam;
use glam::{vec4, Vec4, Vec3, Vec2, UVec2, UVec3, vec2};

use common::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use bytemuck::{Pod, Zeroable};

#[spirv(fragment)]
pub fn main_fs(output: &mut glam::Vec4) {
    *output = glam::vec4(0.0, 0.0, 1.0, 1.0);
}

#[spirv(vertex)]
pub fn main_vs(
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

struct CSVertexShaderOutput {
    vertex_position: Vec5GPU,
    texture_position: Vec4
}

fn cs_vertex_shader(
    vertex_position: Vec4,
    texture_position: Vec4,
    instance: &ModelInstance,
    working_data: &WorkingData
) -> CSVertexShaderOutput {

    let vertex_position = Vec5GPU::from_4_and_1(vertex_position, 1.0);

    let view_position = working_data.view_matrix * instance.model_transform * vertex_position;

    let focal_length = 1.0;
    //var projection_divisor = sqrt(view_vector[2]*view_vector[2] + view_vector[3]*view_vector[3])/focal_length;
    //var projection_divisor = (view_vector[2] + view_vector[3])/focal_length;
    //var projection_divisor = max(view_vector[2], view_vector[3])/focal_length;
    //var projection_divisor = view_vector[2]/focal_length;
    let aspect_ratio = working_data.render_dimensions.x as f32 / working_data.render_dimensions.y as f32;

    let theta = (f32::atan2(view_position.z(), view_position.w())/3.14159)/2.0;
    let depth = f32::sqrt(view_position.z()*view_position.z() + view_position.w()*view_position.w());
    //let depth = view_position.z() + view_position.w();
    let projection_divisor = depth/focal_length;
    //let projection_divisor = 1.0;

    let vertex_position = Vec5GPU::new(
        view_position.x(),
        aspect_ratio * (-view_position.y()),
        theta,
        depth,
        projection_divisor
    );

    CSVertexShaderOutput {
        vertex_position,
        texture_position
    }
}

fn project_4d_vertex(v: Vec5GPU) -> Vec4 {
    Vec4::new(v.x()/v.v(), v.y()/v.v(), v.z(), v.w())
    //Vec4::new(v.x(), v.y(), v.z(), v.w())
}

#[spirv(compute(threads(1)))]
pub fn main_tetrahedron_cs(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] model_tetrahedrons: &[ModelTetrahedron],
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] instances: &[ModelInstance],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] output_line_vertices: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] output_tetrahedrons: &mut [Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData,
) {
    let output_index = global_invocation_id.x as usize;
    let model_index = output_index%model_tetrahedrons.len();
    let instance_index = output_index/model_tetrahedrons.len();
    
    let input_instance = &instances[instance_index];
    let input_tetrahedron = &model_tetrahedrons[model_index];

    let output_tetrahedron = &mut output_tetrahedrons[output_index];

    let mut output_vertex_positions = [
        Vec5GPU::zero(),
        Vec5GPU::zero(),
        Vec5GPU::zero(),
        Vec5GPU::zero()
    ];
    let mut output_texture_positions = [
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO];
    
    for i in 0..4 {
        let vertex_result = cs_vertex_shader(
            input_tetrahedron.vertex_positions[i],
            input_tetrahedron.texture_positions[i],
            &input_instance,
            &working_data);
        output_vertex_positions[i] = vertex_result.vertex_position;
        output_texture_positions[i] = vertex_result.texture_position;
    }
    
    // Perspective scaling
    let output_vertex_positions_projected = [
        project_4d_vertex(output_vertex_positions[0]),
        project_4d_vertex(output_vertex_positions[1]),
        project_4d_vertex(output_vertex_positions[2]),
        project_4d_vertex(output_vertex_positions[3]),
                          ];

    // Verify tetrahedron is completely in-frame
    for i in 0..output_vertex_positions_projected.len() {
        let v = output_vertex_positions_projected[i];
        if v.x < -1.0 || v.x > 1.0 || v.y < -1.0 || v.y > 1.0 || v.z < -1.0 || v.z > 1.0  {
            output_tetrahedron.texture_id = 0;
            for i in 0..2 {
                output_line_vertices[output_index*2 + i] = Vec2::new(0.0, 0.0);
            }
            return;
        }
    }

    *output_tetrahedron = Tetrahedron{
        vertex_positions: output_vertex_positions_projected,
        texture_positions: output_texture_positions,
        texture_id: input_instance.cell_texture_ids[input_tetrahedron.cell_id as usize],
        padding: input_tetrahedron.padding,
    };

    let line_vertices = [
        Vec2::new(output_vertex_positions_projected[0].x, output_vertex_positions_projected[0].y),
        Vec2::new(output_vertex_positions_projected[1].x, output_vertex_positions_projected[1].y),
        Vec2::new(output_vertex_positions_projected[2].x, output_vertex_positions_projected[2].y),
        Vec2::new(output_vertex_positions_projected[3].x, output_vertex_positions_projected[3].y),
    ];


    output_line_vertices[(output_index*12) + 0] = line_vertices[0];
    output_line_vertices[(output_index*12) + 1] = line_vertices[1];

    output_line_vertices[(output_index*12) + 2] = line_vertices[0];
    output_line_vertices[(output_index*12) + 3] = line_vertices[2];

    output_line_vertices[(output_index*12) + 4] = line_vertices[0];
    output_line_vertices[(output_index*12) + 5] = line_vertices[3];

    output_line_vertices[(output_index*12) + 6] = line_vertices[1];
    output_line_vertices[(output_index*12) + 7] = line_vertices[2];

    output_line_vertices[(output_index*12) + 8] = line_vertices[1];
    output_line_vertices[(output_index*12) + 9] = line_vertices[3];

    output_line_vertices[(output_index*12) + 10] = line_vertices[2];
    output_line_vertices[(output_index*12) + 11] = line_vertices[3];
}

#[spirv(compute(threads(1)))]
pub fn main_edge_cs(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] model_edges: &[ModelEdge],
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] instances: &[ModelInstance],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] output_line_vertices: &mut [Vec2],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData,
) {
    let output_index = global_invocation_id.x as usize;
    let model_index = output_index%model_edges.len();
    let instance_index = output_index/model_edges.len();

    let input_instance = &instances[instance_index];
    let input_edge = &model_edges[model_index];

    let mut output_vertex_positions = [
        Vec5GPU::zero(),
        Vec5GPU::zero()
    ];

    for i in 0..input_edge.vertex_positions.len() {
        let vertex_result = cs_vertex_shader(
            input_edge.vertex_positions[i],
            Vec4::ZERO,
            &input_instance,
            &working_data);
        output_vertex_positions[i] = vertex_result.vertex_position;
    }


    // Perspective scaling
    let output_vertex_positions_projected = [
        project_4d_vertex(output_vertex_positions[0]),
        project_4d_vertex(output_vertex_positions[1])
    ];
    
    // Verify tetrahedron is completely in-frame
    for i in 0..output_vertex_positions_projected.len() {
        let v = output_vertex_positions_projected[i];
        if v.x < -1.0 || v.x > 1.0 || v.y < -1.0 || v.y > 1.0 || v.z < -1.0 || v.z > 1.0  {
            for i in 0..2 {
                output_line_vertices[output_index*2 + i] = Vec2::new(0.0, 0.0);
            }
            return;
        }
    }
    
    
    let line_vertices = [
        Vec2::new(output_vertex_positions_projected[0].x, output_vertex_positions_projected[0].y),
        Vec2::new(output_vertex_positions_projected[1].x, output_vertex_positions_projected[1].y),
    ];


    output_line_vertices[(output_index*2) + 0] = line_vertices[0];
    output_line_vertices[(output_index*2) + 1] = line_vertices[1];
}