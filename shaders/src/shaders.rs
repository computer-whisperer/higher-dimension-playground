#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use spirv_std::spirv;

pub use spirv_std::glam;
use glam::{vec4, Vec4, Vec2, UVec3, Vec3Swizzles, Vec3, Vec4Swizzles, UVec2, vec3};

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
    let aspect_ratio = working_data.present_dimensions.x as f32 / working_data.present_dimensions.y as f32;

    let theta = (f32::atan2(view_position.z(), view_position.w())/3.14159)/2.0;
    let depth = f32::sqrt(view_position.z()*view_position.z() + view_position.w()*view_position.w());
    //let depth = view_position.z() + view_position.w();
    let projection_divisor = depth/focal_length;
    //let projection_divisor = 1.0;

    let vertex_position = Vec5GPU::new(
        view_position.x(),
        aspect_ratio * (-view_position.y()),
        view_position.z(),
        view_position.w(),
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
    /*
    for i in 0..output_vertex_positions_projected.len() {
        let v = output_vertex_positions_projected[i];
        if v.x < -1.0 || v.x > 1.0 || v.y < -1.0 || v.y > 1.0  {
            output_tetrahedron.texture_id = 0;
            for i in 0..2 {
                output_line_vertices[output_index*2 + i] = Vec2::new(0.0, 0.0);
            }
            return;
        }
    }*/

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
        if v.x < -1.0 || v.x > 1.0 || v.y < -1.0 || v.y > 1.0  {
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

fn get_barycentric(vertices: &[Vec2; 3], point: Vec2) -> Vec3 {
    let u = Vec3::new(
        vertices[2].x - vertices[0].x,
        vertices[1].x - vertices[0].x,
        vertices[0].x - point.x).cross(
        Vec3::new(
            vertices[2].y - vertices[0].y,
            vertices[1].y - vertices[0].y,
            vertices[0].y - point.y
        )
    );

    if u.z.abs() < 0.000001 {
        Vec3::new(-1.0, 1.0, 1.0)
    }
    else {
        Vec3::new(
            1.0 - (u.x + u.y)/u.z,
            u.y/u.z,
            u.x/u.z
        )
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct ZWLine {
    zw_positions: [Vec2; 2],
    texture_positions: [Vec3; 2],
    texture_id: u32,
    start: f32,
    stop: f32,
}

impl Default for ZWLine {
    fn default() -> Self {
        Self {
            zw_positions: [Vec2::ZERO, Vec2::ZERO],
            texture_positions: [Vec3::ZERO, Vec3::ZERO],
            texture_id: 0,
            start: 0.0,
            stop: 1.0
        }
    }
}

fn get_intersection(line_a: &[Vec2; 2], line_b: &[Vec2; 2]) -> Vec2
{
    //https://gamedev.stackexchange.com/questions/116422/best-way-to-find-line-segment-intersection
    let r_num = (line_a[0].y - line_b[0].y)*(line_b[1].x - line_b[0].x) - (line_a[0].x - line_b[0].x)*(line_b[1].y - line_b[0].y);
    let r_den = (line_a[1].x - line_a[0].x)*(line_b[1].y - line_b[0].y) - (line_a[1].y - line_a[0].y)*(line_b[1].x - line_b[0].x);
    let s_num = (line_a[0].y - line_b[0].y)*(line_a[1].x - line_a[0].x) - (line_a[0].x - line_b[0].x)*(line_a[1].y - line_a[0].y);
    let s_den = r_den;
    Vec2::new(
        r_num/r_den,
        s_num/s_den
    )
}

fn sample_texture(texture_id: u32, texture_pos: Vec3) -> Vec3 {
    match texture_id {
        1 => {vec3(1.0, 0.0, 0.0)}
        2 => {vec3(1.0, 0.8, 0.0)}
        3 => {vec3(0.5, 1.0, 0.0)}
        4 => {vec3(0.0, 1.0, 0.2)}
        5 => {vec3(0.0, 1.0, 1.0)}
        6 => {vec3(0.0, 0.2, 1.0)}
        7 => {vec3(0.5, 0.0, 1.0)}
        8 => {vec3(1.0, 0.0, 0.8)}
        9 => {(texture_pos+vec3(1.0, 1.0, 1.0))/2.0}
        10 => {vec3(139.0, 69.0, 19.8)/256.0}
        11 => {vec3(34.0, 139.0, 34.0)/256.0}
        12=> {vec3(0.0, 0.0, 1.0)}
        _ => {vec3(0.0, 0.0, 0.0)}
    }
    
    /*
            vec3<f32>(1.0, 0.0, 0.0), //0
        vec3<f32>(1.0, 0.0, 0.0), //1
        vec3<f32>(1.0, 0.8, 0.0), //2
        vec3<f32>(0.5, 1.0, 0.0), //3
        vec3<f32>(0.0, 1.0, 0.2), //4
        vec3<f32>(0.0, 1.0, 1.0), //5
        vec3<f32>(0.0, 0.2, 1.0), //6
        vec3<f32>(0.5, 0.0, 1.0), //7
        vec3<f32>(1.0, 0.0, 0.8), //8
        (texture_pos+vec3<f32>(1.0, 1.0, 1.0))/2.0,  // 9
        vec3<f32>(139.0, 69.0, 19.8)/256.0, // 10
        vec3<f32>(34.0, 139.0, 34.0)/256.0, // 11
        vec3<f32>(0.0, 0.0, 1.0), // 12
    */
}

fn render_zw_lines_simple(lines: &[ZWLine; 96], num_lines: usize) -> Vec4 {
    const DEPTH_FACTOR: u32 = 256;

    let mut output_accumulation = Vec3::ZERO;

    for i in 0..DEPTH_FACTOR {
        let pi = 3.14159;
        let angle_ratio = (((i as f32)/(DEPTH_FACTOR as f32)) - 0.5);
        let theta = angle_ratio*(pi/4.0) + (pi/4.0);
        let sample_ray = Vec2::new(theta.cos(), theta.sin());

        let mut closest_line: Option<usize> = None;
        let mut closest_dist: Option<f32> = None;
        let mut closest_val: Option<f32> = None;

        for j in 0..num_lines {
            let intersection = get_intersection(
                &[Vec2::ZERO, sample_ray],
                &lines[j].zw_positions
            );
            let dist = intersection.x;
            if intersection.y > 1.0 || intersection.y < 0.0 {
                continue;
            }
            if dist < 0.0 {
                continue;
            }

            if closest_line.is_none() || dist < closest_dist.unwrap() {
                closest_line = Some(j);
                closest_dist = Some(dist);
                closest_val = Some(intersection.y);
            }
        }

        if let Some(closest_line_index) = closest_line {
            let val = closest_val.unwrap();
            let texture_pos = lines[closest_line_index].texture_positions[0]*val + lines[closest_line_index].texture_positions[0]*(1.0-val);
            let sample = sample_texture(lines[closest_line_index].texture_id, texture_pos);
            output_accumulation += sample/DEPTH_FACTOR as f32;
        }
    }

    if true
    {
        Vec4::new(output_accumulation.x, output_accumulation.y, output_accumulation.z, 1.0)
    }
    else
    {
        if num_lines != 0 {
            Vec4::new(0.0, 0.0, 1.0, 1.0)
        }
        else {
            Vec4::ZERO
        }
    }

}

#[spirv(compute(threads(8, 8)))]
pub fn main_tetrahedron_pixel_cs(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] input_tetrahedrons: &[Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &mut [Vec4]
)
{
    let mut num_zw_lines = 0;
    let mut current_zw_lines = [
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
        ZWLine::default(),
    ];

    let u_pixel_pos = global_invocation_id.xy();
    let pixel_pos = Vec2::new(
        (u_pixel_pos.x as f32)/(working_data.render_dimensions.x as f32)*2.0 - 1.0,
        (u_pixel_pos.y as f32)/(working_data.render_dimensions.y as f32)*2.0 - 1.0);
    for i in 0..working_data.total_num_tetrahedrons as usize {
        let tetrahedron = &input_tetrahedrons[i];
        let min_bound = tetrahedron.vertex_positions[0]
            .min(tetrahedron.vertex_positions[1])
            .min(tetrahedron.vertex_positions[2])
            .min(tetrahedron.vertex_positions[3]);
        let max_bound = tetrahedron.vertex_positions[0]
            .max(tetrahedron.vertex_positions[1])
            .max(tetrahedron.vertex_positions[2])
            .max(tetrahedron.vertex_positions[3]);

        if tetrahedron.texture_id == 0 {
            continue;
        }

        if pixel_pos.x < min_bound.x || pixel_pos.x > max_bound.x ||
            pixel_pos.y < min_bound.y || pixel_pos.y > max_bound.y {
            continue;
        }

        let combinations_4_3 = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
        //let combinations_4_2 = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

        let mut barycentrics = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];

        let mut any_in = false;

        let mut in_triangle = [false; 4];
        for j in 0..4 {
            let vertices = [
                tetrahedron.vertex_positions[combinations_4_3[j][0]].xy(),
                tetrahedron.vertex_positions[combinations_4_3[j][1]].xy(),
                tetrahedron.vertex_positions[combinations_4_3[j][2]].xy(),
            ];

            barycentrics[j] = get_barycentric(&vertices, pixel_pos);

            in_triangle[j] = (barycentrics[j].x >= 0.0) && (barycentrics[j].y >= 0.0) && (barycentrics[j].z >= 0.0);

            any_in = any_in || in_triangle[j];
        }

        if !any_in {
            continue;
        }

        // Identify which triangle ids we intersect with. Assume we intersect 2 by this point

        let tri_indexes = if in_triangle[0] {
            if in_triangle[1] {
                [0, 1]
            }
            else if in_triangle[2] {
                [0, 2]
            }
            else {
                [0, 3]
            }
        }
        else {
            if in_triangle[1] {
                if in_triangle[2] {
                    [1, 2]
                }
                else {
                    [1, 3]
                }
            }
            else {
                [2, 3]
            }
        };

        if num_zw_lines >= current_zw_lines.len() {
            continue;
        }

        current_zw_lines[num_zw_lines].texture_id = tetrahedron.texture_id;
        for j in 0..2 {
            let bary = barycentrics[tri_indexes[j]];

            let zw_vertices = [
                tetrahedron.vertex_positions[combinations_4_3[tri_indexes[j]][0]].zw(),
                tetrahedron.vertex_positions[combinations_4_3[tri_indexes[j]][1]].zw(),
                tetrahedron.vertex_positions[combinations_4_3[tri_indexes[j]][2]].zw(),
            ];

            let tex_vertices = [
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][0]].xyz(),
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][1]].xyz(),
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][2]].xyz(),
            ];

            let zw_value = bary.x*zw_vertices[0] + bary.y*zw_vertices[1] + bary.z*zw_vertices[2];
            let tex_value = bary.x*tex_vertices[0] + bary.y*tex_vertices[1] + bary.z*tex_vertices[2];

            current_zw_lines[num_zw_lines].zw_positions[j] = zw_value;
            current_zw_lines[num_zw_lines].texture_positions[j] = tex_value;
        }
        num_zw_lines += 1;
    }

    let output = render_zw_lines_simple(&current_zw_lines, num_zw_lines);
    if u_pixel_pos.x < working_data.render_dimensions.x && u_pixel_pos.y < working_data.render_dimensions.y {

        pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] = output;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_barycentric() {
        let vertices = [
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
        ];
        let point = Vec2::new(0.5, 0.5);
        let bary = get_barycentric(&vertices, point);
        assert_eq!(bary, Vec3::new(0.0, 0.5, 0.5));

        let point = Vec2::new(0.2, 0.2);
        let bary = get_barycentric(&vertices, point);
        assert_eq!(bary, Vec3::new(0.6, 0.2, 0.2));

        let point = Vec2::new(1.0, 1.0);
        let bary = get_barycentric(&vertices, point);
        assert_eq!(bary, Vec3::new(-1.0, 1.0, 1.0));
    }

    #[test]
    fn test_get_intersection() {
        let line_0 = [Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0)];
        let line_1 = [Vec2::new(-1.0, 4.0), Vec2::new(1.0, 4.0)];
        let point = get_intersection(&line_0, &line_1);
        assert_eq!(point, Vec2::new(1.0, 0.5));

        let line_0 = [Vec2::new(0.0, 0.0), Vec2::new(4.0, 4.0)];
        let line_1 = [Vec2::new(0.0, 10.0), Vec2::new(10.0, 0.0)];
        let point = get_intersection(&line_0, &line_1);
        assert_eq!(point, Vec2::new(1.25, 0.5));

        let line_0 = [Vec2::new(0.0, 0.0), Vec2::new(4.0, -4.0)];
        let line_1 = [Vec2::new(10.0, 10.0), Vec2::new(10.0, 0.0)];
        let point = get_intersection(&line_0, &line_1);
        assert_eq!(point, Vec2::new(2.5, 2.0));
    }

}