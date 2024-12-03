
use spirv_std::spirv;
use common::{Tetrahedron, WorkingData, ModelInstance, ModelTetrahedron, Vec5GPU, VecN, get_normal, get_pseudo_barycentric};
use glam::{Vec4, UVec3, Vec2, Vec3Swizzles, Vec4Swizzles};
use bytemuck::{Pod, Zeroable};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use crate::textures::sample_texture;

struct CSVertexShaderOutput {
    vertex_position: Vec4,
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

    let vertex_position = Vec4::new(
        view_position.x(),
        view_position.y(),
        view_position.z(),
        view_position.w(),
    );

    CSVertexShaderOutput {
        vertex_position,
        texture_position
    }
}

#[spirv(compute(threads(64)))]
pub fn main_raytracer_tetrahedron_preprocessor_cs(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] model_tetrahedrons: &[ModelTetrahedron],
    #[spirv(storage_buffer, descriptor_set = 2, binding = 0)] instances: &[ModelInstance],
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
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO,
        Vec4::ZERO
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
    
    let normal: Vec4 = get_normal(&[
        (output_vertex_positions[1] - output_vertex_positions[0]).into(),
        (output_vertex_positions[2] - output_vertex_positions[0]).into(),
        (output_vertex_positions[3] - output_vertex_positions[0]).into()
    ]).into();
    
    let normal = normal.normalize();

    *output_tetrahedron = Tetrahedron{
        vertex_positions: output_vertex_positions,
        texture_positions: output_texture_positions,
        texture_id: input_instance.cell_texture_ids[input_tetrahedron.cell_id as usize],
        padding: input_tetrahedron.padding,
        normal
    };
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct RayHit {
    hit_position: Vec4,
    barycentric_coordinates: Vec4,
    hit_normal: Vec4,
    tetrahedron_index: u32,
    hit_distance: f32,
    padding: [u32; 2]
}



fn raycast(start: Vec4, direction: Vec4, tetrahedrons: &[Tetrahedron], num_tetrahedrons: usize) -> RayHit {
    let mut closest_hit = RayHit {
        hit_position: Vec4::ZERO,
        barycentric_coordinates: Vec4::ZERO,
        hit_normal: Vec4::ZERO,
        tetrahedron_index: num_tetrahedrons as u32,
        hit_distance: 0.0,
        padding: [0; 2]
    };
    
    for i in 0..num_tetrahedrons {
        if tetrahedrons[i].texture_id == 0 {
            continue;
        }
        
        //Moller-Trumbore expanded for 4-dimensions:
        
        let value = tetrahedrons[i].normal.dot(direction);
        
        if value.abs() < 0.00001 {
            continue; // Only handle tets that are facing the right way
        }
        
        let intercept_distance = tetrahedrons[i].normal.dot(tetrahedrons[i].vertex_positions[0] - start)/value;
        //let intercept_distance = 10f32; 
        
        if closest_hit.tetrahedron_index < num_tetrahedrons as u32 && closest_hit.hit_distance < intercept_distance {
            continue;
        }
        
        let vecn_positions: [VecN::<4>; 4] = [
            tetrahedrons[i].vertex_positions[0].into(),
            tetrahedrons[i].vertex_positions[1].into(),
            tetrahedrons[i].vertex_positions[2].into(),
            tetrahedrons[i].vertex_positions[3].into()
        ];
        
        let intercept_point = intercept_distance*direction + start;
        
        let barycentric = get_pseudo_barycentric(&vecn_positions, intercept_point.into());
        
        if Vec4::from(barycentric) != Vec4::ZERO {
            // We already filtered for distance, so go straight to processing
            closest_hit = RayHit {
                tetrahedron_index: i as u32,
                hit_position: intercept_point,
                hit_distance: intercept_distance,
                barycentric_coordinates: barycentric.into(),
                hit_normal: tetrahedrons[i].normal,
                padding: [0; 2]
            };
        }
    }

    closest_hit
}

#[spirv(compute(threads(8, 8)))]
pub fn main_raytracer_pixel_cs(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] input_tetrahedrons: &[Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &mut WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &mut [Vec4]
) {
    let u_pixel_pos = global_invocation_id.xy();
    let pixel_pos = Vec2::new(
        (u_pixel_pos.x as f32)/(working_data.render_dimensions.x as f32)*2.0 - 1.0,
        (u_pixel_pos.y as f32)/(working_data.render_dimensions.y as f32)*2.0 - 1.0);

    let mut output_pixel = Vec4::new(0.0, 0.0, 0.0, 1.0);
    
    let view_origin = Vec4::ZERO;
    let view_direction = Vec4::new(pixel_pos.x*1.0, pixel_pos.y*1.0, 1.0, 0.0).normalize();

    
    let ray_hit = raycast(view_origin, view_direction, &input_tetrahedrons, working_data.total_num_tetrahedrons as usize);

    
    if ray_hit.tetrahedron_index < working_data.total_num_tetrahedrons {
        let tetrahedron = input_tetrahedrons[ray_hit.tetrahedron_index as usize];
        let mut texture_coordinates = Vec4::ZERO;
        
        for i in 0..4 {
            //texture_coordinates += ray_hit.barycentric_coordinates[i]*tetrahedron.texture_positions[i];
        }
        
        let color = sample_texture(tetrahedron.texture_id, texture_coordinates.xyz());
        //output_pixel = Vec4::new(color.x, color.y, color.z, 1.0);
        output_pixel = Vec4::new(0.0, 0.0, 1.0, 1.0);
    }
     
    if u_pixel_pos.x < working_data.render_dimensions.x && u_pixel_pos.y < working_data.render_dimensions.y {
        pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] = output_pixel;
    }
}

#[cfg(test)]
mod tests {
    use glam::UVec2;
    use common::Mat5GPU;
    use super::*;

    #[test]
    fn test_raycast() {
        let model_tetrahedrons = [
            ModelTetrahedron {
                vertex_positions: [
                    Vec4::new(1.0, 0.0, 2.0, 0.0),
                    Vec4::new(0.0, 0.0, 2.0, 0.0),
                    Vec4::new(0.0, 1.0, 2.0, 0.0),
                    Vec4::new(0.0, 0.0, 2.0, 1.0),
                ],
                texture_positions: [Vec4::ZERO; 4],
                cell_id: 0,
                padding: [0; 3],
            }
        ];
        let instances = [
            ModelInstance {
                model_transform: Mat5GPU::identity(),
                cell_texture_ids: [1, 1, 1, 1, 1, 1, 1, 1],
            }
        ];
        const WIDTH: u32 = 512;
        const HEIGHT: u32 = 512;
        let mut working_data = WorkingData {
            view_matrix: Mat5GPU::identity(),
            total_num_tetrahedrons: (model_tetrahedrons.len()*instances.len()) as u32,
            shader_fault: 0,
            render_dimensions: UVec2::new(WIDTH, HEIGHT),
            present_dimensions: UVec2::new(WIDTH, HEIGHT),
            padding: Default::default(),
        };
        
        let mut output_tetrahedrons = [Tetrahedron {
            vertex_positions: [Vec4::ZERO; 4],
            texture_positions: [Vec4::ZERO; 4],
            texture_id: 0,
            normal: Vec4::new(0.0, 0.0, 1.0, 0.0),
            padding: [0; 3],
        }; 10];
        
        main_raytracer_tetrahedron_preprocessor_cs(
            &model_tetrahedrons,
            &instances,
            &mut output_tetrahedrons,
            UVec3::new(0, 0, 0),
            &mut working_data
        );
        
        assert_eq!(output_tetrahedrons[0].texture_id, 1);
        assert_eq!(output_tetrahedrons[0].normal, Vec4::new(0.0, 0.0, -1.0, 0.0));
        
        let mut pixel_buffer = vec![Vec4::ZERO; (WIDTH*HEIGHT) as usize];
        
        let sample = UVec2::new(WIDTH/2, HEIGHT/2);
        
        // Test dead center
        main_raytracer_pixel_cs(
            &output_tetrahedrons,
            UVec3::new(sample.x, sample.y, 0),
            &mut working_data,
            &mut pixel_buffer
        );
        let color = pixel_buffer[(sample.y*working_data.render_dimensions.x + sample.x) as usize];
        
        assert_eq!(color, Vec4::new(0.0, 0.0, 1.0, 1.0));

        let sample = UVec2::new(WIDTH/2, HEIGHT/8);

        // Test dead center
        main_raytracer_pixel_cs(
            &output_tetrahedrons,
            UVec3::new(sample.x, sample.y, 0),
            &mut working_data,
            &mut pixel_buffer
        );
        let color = pixel_buffer[(sample.y*working_data.render_dimensions.x + sample.x) as usize];

        assert_eq!(color, Vec4::new(0.0, 0.0, 0.0, 1.0));

        let sample = UVec2::new(10, 0);

        // Test dead center
        main_raytracer_pixel_cs(
            &output_tetrahedrons,
            UVec3::new(sample.x, sample.y, 0),
            &mut working_data,
            &mut pixel_buffer
        );
        let color = pixel_buffer[(sample.y*working_data.render_dimensions.x + sample.x) as usize];

        assert_eq!(color, Vec4::new(0.0, 0.0, 0.0, 1.0));
    }
}