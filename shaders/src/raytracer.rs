
use spirv_std::spirv;
use common::{Tetrahedron, WorkingData};
use glam::{Vec4, UVec3};

fn raycast(start: Vec4, direction: Vec4, tetrahedrons: &[Tetrahedron], num_tetrahedrons: usize) {
    
}

#[spirv(compute(threads(64)))]
pub fn main_raytracer_tetrahedron_preprocessor_cs(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] input_tetrahedrons: &[Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &mut WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &mut [Vec4]
) {

}

#[spirv(compute(threads(8, 8)))]
pub fn main_raytracer_pixel_cs(
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] input_tetrahedrons: &[Tetrahedron],
    #[spirv(global_invocation_id)] global_invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 2, binding = 1)] working_data: &mut WorkingData,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] pixel_buffer: &mut [Vec4]
) {
    
}