#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod raytracer;
mod present;
mod rasterizer;
mod materials;

pub use spirv_std::glam;

pub use rasterizer::{main_edge_cs, main_tetrahedron_cs, main_tetrahedron_pixel_cs};
pub use present::{main_line_fs, main_line_vs, main_buffer_fs, main_buffer_vs};
pub use raytracer::{main_raytracer_tetrahedron_preprocessor_cs, main_raytracer_pixel_cs, main_raytracer_clear_cs};
