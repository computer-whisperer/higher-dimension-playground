use glam::{UVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use common::{ModelEdge, ModelInstance, ModelTetrahedron, Tetrahedron, Vec5GPU, WorkingData};
use spirv_std::spirv;
use bytemuck::{Pod, Zeroable};
use crate::textures::sample_texture;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

fn sample_texture_integral(line: ZWLine) -> Vec3 {
    const STEP_SIZE: f32 = VIEW_ANGLE/128.0;
    let mut output_accumulation = Vec3::ZERO;

    let angle_a = f32::atan2(line.zw_positions[0].x, line.zw_positions[0].y).max(ANGLE_MIN).min(ANGLE_MAX);
    let angle_b = f32::atan2(line.zw_positions[1].x, line.zw_positions[1].y).max(ANGLE_MIN).min(ANGLE_MAX);

    let (start_angle, end_angle) = if angle_a > angle_b {
        (angle_b, angle_a)
    }
    else {
        (angle_a, angle_b)
    };

    let mut angle = start_angle;
    loop {
        if angle > end_angle {
            break;
        }

        let intersection = get_intersection(&[Vec2::ZERO, Vec2::new(angle.cos(), angle.sin())], &line.zw_positions);
        let texture_pos = line.texture_positions[0]*intersection.y + line.texture_positions[1]*(1.0-intersection.y);
        let texture_color = sample_texture(line.texture_id, texture_pos.xyz());
        output_accumulation += texture_color*STEP_SIZE/VIEW_ANGLE;

        angle += STEP_SIZE;
    }

    output_accumulation
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

    //let theta = (f32::atan2(view_position.z(), view_position.w())/3.14159)/2.0;
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

#[spirv(compute(threads(64)))]
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

#[spirv(compute(threads(64)))]
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

pub fn get_barycentric(vertices: &[Vec2; 3], point: Vec2) -> Vec3 {
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

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ZWLine {
    zw_positions: [Vec2; 2],
    texture_positions: [Vec4; 2],
    texture_id: u32,
    padding: [u32; 3],
}

impl Default for ZWLine {
    fn default() -> Self {
        Self {
            zw_positions: [Vec2::ZERO, Vec2::ZERO],
            texture_positions: [Vec4::ZERO, Vec4::ZERO],
            texture_id: 0,
            padding: [0; 3]
        }
    }
}

impl ZWLine {
    fn get_segment(&self, start: f32, end: f32) -> Self {
        Self {
            zw_positions: [
                self.zw_positions[0]*(1.0 - start) + self.zw_positions[1]*(start),
                self.zw_positions[0]*(1.0 - end) + self.zw_positions[1]*(end),
            ],
            texture_positions: [
                self.texture_positions[0]*(1.0 - start) + self.texture_positions[1]*(start),
                self.texture_positions[0]*(1.0 - end) + self.texture_positions[1]*(end),
            ],
            texture_id: self.texture_id,
            padding: [0; 3]
        }
    }
}

pub fn get_intersection(line_a: &[Vec2; 2], line_b: &[Vec2; 2]) -> Vec2
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

fn test_intersection(line_a: &[Vec2; 2], line_b: &[Vec2; 2]) -> bool
{
    let res = get_intersection(line_a, line_b);

    res.x > 0.0 && res.x < 1.0 && res.y > 0.0 && res.y < 1.0
}



const PI: f32 = 3.14159;
pub const VIEW_ANGLE: f32 = PI/2.0;
pub const ANGLE_MIN: f32 = PI/4.0 - VIEW_ANGLE/2.0;
pub const ANGLE_MAX: f32 = PI/4.0 + VIEW_ANGLE/2.0;

fn render_zw_lines_simple(lines: &[ZWLine; 96], num_lines: usize) -> Vec4 {
    const DEPTH_FACTOR: u32 = 512;

    let mut output_accumulation = Vec3::ZERO;

    for i in 0..DEPTH_FACTOR {
        let angle_ratio = (i as f32)/(DEPTH_FACTOR as f32);
        let theta = angle_ratio*(ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN;
        let sample_ray = Vec2::new(theta.cos(), theta.sin());

        let mut closest_line: Option<usize> = None;
        let mut closest_dist: Option<f32> = None;
        let mut closest_val: Option<f32> = None;

        for j in 0..num_lines {
            if lines[j].texture_id == 0 {
                continue;
            }

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
            let sample = sample_texture(lines[closest_line_index].texture_id, texture_pos.xyz());
            output_accumulation += sample/DEPTH_FACTOR as f32;
        }
    }

    Vec4::new(output_accumulation.x, output_accumulation.y, output_accumulation.z, 1.0)
}

fn condense_zw_lines(lines: &mut [ZWLine; 96], num_lines: usize) -> usize {
    // Remove lines with texture_id == 0, returning the new count of good lines
    let mut count = 0;
    for i in 0..num_lines {
        if lines[i].texture_id!= 0 {
            lines[count] = lines[i];
            count += 1;
        }
    }
    count
}

fn render_zw_lines_2(lines: &mut [ZWLine; 96], num_lines: usize) -> Vec4 {

    // Scan for complete occlusions
    for i in 0..num_lines {

        if lines[i].texture_id == 0 {
            continue;
        }
        let mut complete_occlusion = false;

        for j in 0..num_lines {
            if j == i {
                continue;
            }
            if lines[j].texture_id == 0 {
                continue;
            }
            let mut local_full_occlusion = true;
            for k in 0..2 {
                let intersection = get_intersection(&[Vec2::ZERO, lines[i].zw_positions[k]], &lines[j].zw_positions);
                let does_occlude = intersection.x >= 0.0 && intersection.x <= 1.0 && intersection.y >= 0.0 && intersection.y <= 1.0;
                if !does_occlude {
                    local_full_occlusion = false;
                    break;
                }
            }
            if local_full_occlusion {
                complete_occlusion = true;
                break;
            }
        }
        if complete_occlusion {
            lines[i].texture_id = 0;
            continue;
        }
    }

    let num_lines = condense_zw_lines(lines, num_lines);

    // Scan for full line intersections
    let mut full_intersection = false;
    for i in 0..num_lines {
        if lines[i].texture_id == 0 {
            continue;
        }
        for j in 0..num_lines {
            if j == i {
                continue;
            }
            if lines[j].texture_id == 0 {
                continue;
            }
            if test_intersection(&lines[i].zw_positions, &lines[j].zw_positions)
            {
                full_intersection = true;
                break;
            }
        }
        if full_intersection {
            break;
        }
    }
    if full_intersection {
        //working_data.shader_fault = 1;
    }

    // Scan for intermittent occlusions

    let mut intermittent_occlusion = false;
    if !full_intersection {
        for i in 0..num_lines {
            if lines[i].texture_id == 0 {
                continue;
            }
            for j in 0..num_lines {
                if j == i {
                    continue;
                }
                if lines[j].texture_id == 0 {
                    continue;
                }
                let mut local_intermittent_occlusion = true;
                for k in 0..2 {
                    let intersection = get_intersection(&[Vec2::ZERO, lines[j].zw_positions[k]], &lines[i].zw_positions);
                    let does_occlude = intersection.x > 1.0 && intersection.y > 0.0 && intersection.y < 1.0;
                    if !does_occlude {
                        local_intermittent_occlusion = false;
                        break;
                    }
                }
                if local_intermittent_occlusion {
                    intermittent_occlusion = true;
                    break;
                }
            }
            if intermittent_occlusion {
                break;
            }
        }
    }


    if intermittent_occlusion || full_intersection {
        // Must revert to simple algorithm
        render_zw_lines_simple(&lines, num_lines)
    }
    else
    {
        let mut output_accumulation = Vec3::ZERO;

        let mut late_fail = false;

        // Only simple occlusions or no occlusions, so render quickly
        for i in 0..num_lines {
            if lines[i].texture_id == 0 {
                continue;
            }

            // Scan for simple occlusions and trim appropriately
            for j in 0..num_lines {
                if j == i {
                    continue;
                }
                if lines[j].texture_id == 0 {
                    continue;
                }
                let intersection_a = get_intersection(&[Vec2::ZERO, lines[j].zw_positions[0]], &lines[i].zw_positions);
                let intersection_b = get_intersection(&[Vec2::ZERO, lines[j].zw_positions[1]], &lines[i].zw_positions);
                let intersection_c = get_intersection(&[Vec2::ZERO, lines[i].zw_positions[0]], &lines[j].zw_positions);
                let does_a_clip = intersection_a.x > 1.0 && intersection_a.y > 0.0 && intersection_a.y < 1.0;
                let does_b_clip = intersection_b.x > 1.0 && intersection_b.y > 0.0 && intersection_b.y < 1.0;
                let does_c_clip = intersection_c.x > 0.0 && intersection_c.x < 1.0 && intersection_c.y > 0.0 && intersection_c.y < 1.0;

                if does_a_clip || does_b_clip {

                    if does_a_clip && does_b_clip {
                        late_fail = true;
                        break;
                    }

                    let split = if does_a_clip {intersection_a.y} else {intersection_b.y};

                    // Something needs to be trimmed, but we need to figure out if it's the start or end of the line. c_clip should tell us
                    if does_c_clip {
                        lines[i] = lines[i].get_segment(split, 1.0);
                    }
                    else {
                        lines[i] = lines[i].get_segment(0.0, split);
                    }
                }
            }

            if late_fail {
                break;
            }

            // All occlusions should have been resolved
            output_accumulation += sample_texture_integral(lines[i]);
        }
        if late_fail {
            render_zw_lines_simple(&lines, num_lines)
        }
        else {
            Vec4::new(output_accumulation.x, output_accumulation.y, output_accumulation.z, 1.0)
        }
    }
}

#[spirv(compute(threads(8, 8)))]
pub fn main_tetrahedron_pixel_cs(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] input_tetrahedrons: &[Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &mut WorkingData,
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
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][0]],
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][1]],
                tetrahedron.texture_positions[combinations_4_3[tri_indexes[j]][2]],
            ];

            let zw_value = bary.x*zw_vertices[0] + bary.y*zw_vertices[1] + bary.z*zw_vertices[2];
            let tex_value = bary.x*tex_vertices[0] + bary.y*tex_vertices[1] + bary.z*tex_vertices[2];

            current_zw_lines[num_zw_lines].zw_positions[j] = zw_value;
            current_zw_lines[num_zw_lines].texture_positions[j] = tex_value;
        }
        num_zw_lines += 1;
    }

    let output = if false {
        render_zw_lines_2(&mut current_zw_lines, num_zw_lines)
    }
    else {
        render_zw_lines_simple(&mut current_zw_lines, num_zw_lines)
    };

    if u_pixel_pos.x < working_data.render_dimensions.x && u_pixel_pos.y < working_data.render_dimensions.y {

        pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] = output;
    }
}

#[cfg(test)]
mod tests {
    use crate::rasterizer::{get_barycentric, get_intersection};
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