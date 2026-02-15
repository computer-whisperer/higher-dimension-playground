mod layout_verify;
mod linalg_n;
mod mat_n;
mod utils;
mod vec_n;

use bytemuck::{Pod, Zeroable};
use glam::{UVec2, UVec4, Vec4};
pub use linalg_n::*;
pub use mat_n::MatN;
pub use utils::BasicRNG;
pub use utils::{binomial, factorial, generate_combinations, generate_permutations};
pub use vec_n::VecN;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Tetrahedron {
    pub vertex_positions: [Vec4; 4],
    pub texture_positions: [Vec4; 4],
    pub normal: Vec4,
    pub inv_projection_divisors: Vec4,
    pub material_id: u32,
    pub padding: [u32; 3],
}

/// Morton code for spatial sorting in LBVH construction
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct MortonCode {
    pub code: u64,
    pub tetrahedron_index: u32,
    pub padding: u32,
}

/// Scene bounding box for Morton code normalization
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct SceneBounds {
    pub min_bounds: Vec4,
    pub max_bounds: Vec4,
}

/// BVH node for GPU-based tree traversal
/// Supports both internal nodes and leaf nodes
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct BVHNode {
    pub min_bounds: Vec4,
    pub max_bounds: Vec4,
    pub left_child: u32,
    pub right_child: u32,
    pub parent: u32,
    pub is_leaf: u32,
    pub tetrahedron_index: u32,
    pub atomic_visit_count: u32,
    pub padding: [u32; 2],
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelTetrahedron {
    pub vertex_positions: [Vec4; 4],
    pub texture_positions: [Vec4; 4],
    pub cell_id: u32,
    pub padding: [u32; 3],
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelEdge {
    pub vertex_positions: [Vec4; 2],
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ModelInstance {
    pub model_transform: MatN<5>,
    pub cell_material_ids: [u32; 8],
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct WorkingData {
    pub render_dimensions: UVec4,
    pub present_dimensions: UVec2,
    pub raytrace_seed: u64,
    pub view_matrix: MatN<5>,
    pub view_matrix_inverse: MatN<5>,
    pub total_num_tetrahedrons: u32,
    pub time_ticks_ms: u32,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub padding: [u32; 2],
    pub world_origin: Vec4,
    pub world_dir_x: Vec4,
    pub world_dir_y: Vec4,
    pub world_dir_z: Vec4,
    pub world_dir_w: Vec4,
}
