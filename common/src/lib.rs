#![cfg_attr(target_arch = "spirv", no_std)]

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_for)]
#![feature(effects)]
#![feature(const_mut_refs)]

mod utils;
mod linalg_n;
mod vec_n;
mod mat_n;

use glam::{Vec4, UVec2, UVec4};
pub use vec_n::VecN;
pub use mat_n::MatN;
pub use utils::{factorial, binomial, generate_combinations, generate_permutations};
pub use linalg_n::*;
pub use utils::{BasicRNG};
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Tetrahedron {
    pub vertex_positions: [Vec4; 4],
    pub texture_positions: [Vec4; 4],
    pub normal: Vec4,
    pub material_id: u32,
    pub padding: [u32; 3]
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelTetrahedron {
    pub vertex_positions: [Vec4; 4],
    pub texture_positions: [Vec4; 4],
    pub cell_id: u32,
    pub padding: [u32; 3]
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelEdge {
    pub vertex_positions: [Vec4; 2]
}


#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelInstance {
    pub model_transform: MatN::<5>,
    pub cell_material_ids: [u32; 8],
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WorkingData {
    pub render_dimensions: UVec4,
    pub present_dimensions: UVec2,
    pub raytrace_seed: u64,
    pub view_matrix: MatN::<5>,
    pub view_matrix_inverse: MatN::<5>,
    pub total_num_tetrahedrons: u32,
    pub shader_fault: u32,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub padding: [u32; 2]
}