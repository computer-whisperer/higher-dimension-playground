
use spirv_std::spirv;
use common::{Tetrahedron, WorkingData, ModelInstance, ModelTetrahedron, Vec5GPU, VecN, get_normal, get_pseudo_barycentric, basic_rand, basic_rand_f32};
use glam::{Vec4, UVec3, Vec2, Vec3Swizzles, Vec4Swizzles, Mat4, Vec3};
use bytemuck::{Pod, Zeroable};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use crate::materials::sample_material;

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

    let view_position = instance.model_transform * vertex_position;

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
        material_id: input_instance.cell_material_ids[input_tetrahedron.cell_id as usize],
        padding: [0; 3],
        normal
    };
}

#[spirv(compute(threads(8, 8)))]
pub fn main_raytracer_clear_cs(
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &WorkingData
)
{
    let u_pixel_pos = global_invocation_id.xy();
    pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] = Vec4::ZERO;
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct RayHit {
    hit_position: Vec4,
    barycentric: Vec4,
    hit_normal: Vec4,
    texture_coordinates: Vec4,
    tetrahedron_index: u32,
    hit_distance: f32,
    did_hit: u32,
    padding: [u32; 1]
}

impl RayHit {
    const NONE: Self = Self {
        hit_position: Vec4::ZERO,
        barycentric: Vec4::ZERO,
        hit_normal: Vec4::ZERO,
        tetrahedron_index: 0,
        texture_coordinates: Vec4::ZERO,
        hit_distance: 0.0,
        did_hit: 0,
        padding: [0; 1]
    };
}

pub fn rand_unit_vec4(working_value: &mut u64) -> Vec4 {
    Vec4::new(
        basic_rand_f32(working_value)-0.5,
        basic_rand_f32(working_value)-0.5,
        basic_rand_f32(working_value)-0.5,
        basic_rand_f32(working_value)-0.5
    ).normalize()
}


fn raycast(start: Vec4, direction: Vec4, tetrahedrons: &[Tetrahedron], num_tetrahedrons: usize) -> RayHit {
    
    let direction = direction.normalize();
    let mut closest_hit = RayHit::NONE;
    
    for i in 0..num_tetrahedrons {
        if tetrahedrons[i].material_id == 0 {
            continue;
        }
        
        //Moller-Trumbore expanded for 4-dimensions:
        
        let value = tetrahedrons[i].normal.dot(direction);
        
        if value.abs() < 0.001 {
            continue; // Only handle tets that are facing the right way
        }

        /*
        let intercept_distance = tetrahedrons[i].normal.dot(tetrahedrons[i].vertex_positions[0] - start)/value;
        
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
        */

        let important_mat = Mat4::from_cols(
            direction,
            tetrahedrons[i].vertex_positions[1] - tetrahedrons[i].vertex_positions[0],
            tetrahedrons[i].vertex_positions[2] - tetrahedrons[i].vertex_positions[0],
            tetrahedrons[i].vertex_positions[3] - tetrahedrons[i].vertex_positions[0]
        );

        let even_more_important_mat = important_mat.inverse();

        let result = even_more_important_mat * (start - tetrahedrons[i].vertex_positions[0]);

        let barycentric = Vec4::new(1.0 - (result[1] + result[2] + result[3]), result[1], result[2], result[3]);

        let hit_distance = -result[0];

        let hit_position = hit_distance*direction + start;

        if hit_distance > 0.0 && barycentric.x >= 0.0 && barycentric.y >= 0.0 && barycentric.z >= 0.0 && barycentric.w >= 0.0 {
            let hit_normal = if tetrahedrons[i].normal.dot(direction) < 0.0 {
                tetrahedrons[i].normal
            }
            else {
                -tetrahedrons[i].normal
            };

            // We already filtered for distance, so go straight to processing
            closest_hit = RayHit {
                tetrahedron_index: i as u32,
                hit_position,
                hit_distance,
                barycentric,
                hit_normal,
                texture_coordinates: Vec4::ZERO,
                did_hit: 1,
                padding: [0; 1]
            };
        }
    }

    if closest_hit.tetrahedron_index != num_tetrahedrons as u32 {
        let tetrahedron = tetrahedrons[closest_hit.tetrahedron_index as usize];
        closest_hit.texture_coordinates =
            closest_hit.barycentric.x*tetrahedron.texture_positions[0] +
                closest_hit.barycentric.y*tetrahedron.texture_positions[1] +
                closest_hit.barycentric.z*tetrahedron.texture_positions[2] +
                closest_hit.barycentric.w*tetrahedron.texture_positions[3];
    }

    closest_hit
}

fn raycast_sample(mut start: Vec4, mut direction: Vec4, tetrahedrons: &[Tetrahedron], num_tetrahedrons: usize, rng_state: &mut u64) -> Vec4 {

    direction = direction.normalize();
    
    let mut hit_stack = [
        RayHit::NONE,
        RayHit::NONE,
        RayHit::NONE,
        RayHit::NONE,
        RayHit::NONE,
        RayHit::NONE,
    ];
    let mut num_hits = 0usize;

    while num_hits < hit_stack.len() {
        let ray_hit = raycast(start, direction, tetrahedrons, num_tetrahedrons);

        if ray_hit.did_hit == 1 {
            let material = sample_material(tetrahedrons[ray_hit.tetrahedron_index as usize].material_id, ray_hit.texture_coordinates.xyz());
            
            let reflection_vector = direction - 2.0*direction.dot(ray_hit.hit_normal)*ray_hit.hit_normal;
            
            if basic_rand_f32(rng_state) < material.metallic {
                direction = reflection_vector;
            }
            else {
                direction = (rand_unit_vec4(rng_state) + ray_hit.hit_normal).normalize();
            }
            start = ray_hit.hit_position + 0.001*direction + rand_unit_vec4(rng_state)*material.roughness*0.5;

            hit_stack[num_hits] = ray_hit;
            num_hits += 1;
        }
        else {
            break;
        }
    }

    let mut light_value = Vec3::ZERO;
    let background_light_direction = Vec4::new(0.0, 1.0, -0.3, 0.0).normalize();
    if num_hits < 4 {
        // We must have hit the sky
        if direction.dot(background_light_direction) > 0.95 {
            light_value = Vec3::new(1.0, 1.0, 1.0)*0.0;
        }
        else {
            let a = 0.5*(direction.y + 1.0);
            light_value = ((1.0 - a)*Vec3::new(1.0, 1.0, 1.0) + a*Vec3::new(0.5, 0.7, 1.0))*0.002;
        }
    }
    
    if num_hits == 0 {
        //light_value = light_value*0.8;
    }

    for i in 0..num_hits {
        let hit = hit_stack[num_hits - i - 1];
        let material = sample_material(tetrahedrons[hit.tetrahedron_index as usize].material_id, hit.texture_coordinates.xyz());

        light_value = light_value*0.9*material.albedo.xyz() + (material.luminance + 0.00)*material.albedo.xyz();
    }

    Vec4::new(light_value.x, light_value.y, light_value.z, 1.0)
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


    let mut rng_state: u64 =
        (working_data.raytrace_seed*(working_data.render_dimensions.x as u64)*(working_data.render_dimensions.y as u64) +
            (global_invocation_id.x as u64) + (global_invocation_id.y as u64*working_data.render_dimensions.x as u64));

    // RNG prime
    for _ in 0..40 {
        //basic_rand(&mut rng_state);
    }
    
    //let view_origin = Vec4::ZERO;
    //let view_direction = Vec4::new(pixel_pos.x*1.0, pixel_pos.y*1.0, 1.0, 0.0).normalize();
    

    //pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] = Vec4::ZERO;
    let aspect_ratio = working_data.present_dimensions.x as f32 / working_data.present_dimensions.y as f32;

    let aa_noise = if false {
        Vec4::new(
            (basic_rand_f32(&mut rng_state)-0.5)*0.01,
            (basic_rand_f32(&mut rng_state)-0.5)*0.01,
            (basic_rand_f32(&mut rng_state)-0.5)*0.01,
            (basic_rand_f32(&mut rng_state)-0.5)*0.01
        )
    }
    else {
        Vec4::ZERO
    };
    
    for i in 0..1 {
        let pi = 3.14159;
        let view_angle = (pi/2.0)/working_data.focal_length;
        let zw_angle = (basic_rand_f32(&mut rng_state)-0.5) * view_angle;

        let view_origin = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let view_direction = aa_noise + Vec4::new(
            pixel_pos.x/working_data.focal_length,
            (pixel_pos.y/aspect_ratio)/working_data.focal_length,
            zw_angle.cos(),
            zw_angle.sin()).normalize();

        let new_view_origin = working_data.view_matrix_inverse * Vec5GPU::from_4_and_1(view_origin, 1.0);
        let new_view_direction = working_data.view_matrix_inverse * Vec5GPU::from_4_and_1(view_origin + view_direction, 1.0);
        
        let view_origin = Vec4::new(new_view_origin.x(), new_view_origin.y(), new_view_origin.z(), new_view_origin.w());
        let mut view_direction = Vec4::new(new_view_direction.x(), new_view_direction.y(), new_view_direction.z(), new_view_direction.w()) - view_origin;
        
        view_direction.y = -view_direction.y;

        let output_pixel = raycast_sample(view_origin, view_direction, &input_tetrahedrons, working_data.total_num_tetrahedrons as usize, &mut rng_state);

        if u_pixel_pos.x < working_data.render_dimensions.x && u_pixel_pos.y < working_data.render_dimensions.y {
            pixel_buffer[(u_pixel_pos.y*working_data.render_dimensions.x + u_pixel_pos.x) as usize] += output_pixel;
        }
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
                cell_material_ids: [1, 1, 1, 1, 1, 1, 1, 1],
            }
        ];
        const WIDTH: u32 = 512;
        const HEIGHT: u32 = 512;
        let mut working_data = WorkingData {
            view_matrix: Mat5GPU::identity(),
            view_matrix_inverse: Mat5GPU::identity(),
            total_num_tetrahedrons: (model_tetrahedrons.len()*instances.len()) as u32,
            shader_fault: 0,
            render_dimensions: UVec2::new(WIDTH, HEIGHT),
            present_dimensions: UVec2::new(WIDTH, HEIGHT),
            raytrace_seed: 0,
            focal_length: 1.0,
            padding: Default::default(),
        };
        
        let mut output_tetrahedrons = [Tetrahedron {
            vertex_positions: [Vec4::ZERO; 4],
            texture_positions: [Vec4::ZERO; 4],
            material_id: 0,
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
        
        assert_eq!(output_tetrahedrons[0].material_id, 1);
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