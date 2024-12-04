#![cfg_attr(target_arch = "spirv", no_std)]

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_for)]
#![feature(effects)]
#![feature(const_mut_refs)]

mod matmul_5_gpu;
mod utils;
mod linalg_n;

use glam::{Vec4, UVec2};
pub use matmul_5_gpu::{Mat5GPU, Vec5GPU};
pub use utils::{factorial, binomial, generate_combinations, generate_permutations};
pub use linalg_n::*;
pub use utils::{basic_rand, basic_rand_f32};
use bytemuck::{Pod, Zeroable};

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Tetrahedron {
    pub vertex_positions: [Vec4; 4],
    pub texture_positions: [Vec4; 4],
    pub normal: Vec4,
    pub texture_id: u32,
    pub luminance: f32,
    pub padding: [u32; 2]
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
    pub model_transform: Mat5GPU,
    pub cell_texture_ids: [u32; 8],
    pub luminance: f32,
    pub padding: [u32; 3]
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WorkingData {
    pub view_matrix: Mat5GPU,
    pub render_dimensions: UVec2,
    pub present_dimensions: UVec2,
    pub total_num_tetrahedrons: u32,
    pub raytrace_seed: u32,
    pub shader_fault: u32,
    pub padding: [u32; 1]
}