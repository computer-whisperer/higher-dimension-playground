use crate::hypercube::{generate_simplexes_for_k_face_3, Hypercube};
use ab_glyph::{point, Font, FontArc, ScaleFont};
use bytemuck::{Pod, Zeroable};
use common::{get_normal, ModelTetrahedron};
use exr::prelude::{ImageAttributes, WritableImage};
use glam::{UVec3, Vec2, Vec4, Vec4Swizzles};
use image::{ImageBuffer, Rgb, Rgba};
use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyBufferToImageInfo,
    CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::{
    DescriptorSetAllocator, StandardDescriptorSetAllocator,
    StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::sys::ImageCreateInfo;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageType, ImageUsage};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::PipelineStage;
use vulkano::{sync, Validated, VulkanError};
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// Collection of all shader modules loaded from Slang-compiled SPIR-V
struct ShaderModules {
    // Present shaders
    line_vs: Arc<ShaderModule>,
    line_fs: Arc<ShaderModule>,
    buffer_vs: Arc<ShaderModule>,
    buffer_fs: Arc<ShaderModule>,
    // HUD shaders
    hud_vs: Arc<ShaderModule>,
    hud_fs: Arc<ShaderModule>,
    // Compute shaders
    tetrahedron_cs: Arc<ShaderModule>,
    edge_cs: Arc<ShaderModule>,
    tetrahedron_pixel_cs: Arc<ShaderModule>,
    raytrace_preprocess: Arc<ShaderModule>,
    raytrace_pixel: Arc<ShaderModule>,
    raytrace_clear: Arc<ShaderModule>,
    // Voxel traversal engine (VTE) compute shaders
    voxel_clear: Arc<ShaderModule>,
    voxel_trace_stage_a: Arc<ShaderModule>,
    voxel_display_stage_b: Arc<ShaderModule>,
    // Tile binning
    bin_tets_cs: Arc<ShaderModule>,
    // BVH compute shaders
    bvh_scene_bounds: Arc<ShaderModule>,
    bvh_morton_codes: Arc<ShaderModule>,
    bvh_bitonic_sort_local: Arc<ShaderModule>,
    bvh_bitonic_sort: Arc<ShaderModule>,
    bvh_bitonic_sort_local_merge: Arc<ShaderModule>,
    bvh_init_leaves: Arc<ShaderModule>,
    bvh_build_tree: Arc<ShaderModule>,
    bvh_compute_leaf_aabbs: Arc<ShaderModule>,
    bvh_propagate_aabbs: Arc<ShaderModule>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RenderBackend {
    /// Legacy behavior: derive backend from existing booleans.
    Auto,
    /// Existing tetrahedron tile raster path.
    TetraRaster,
    /// Existing tetrahedron raytrace path.
    TetraRaytrace,
    /// New voxel traversal engine path (currently placeholder).
    VoxelTraversal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VteDisplayMode {
    Integral,
    Slice,
    ThickSlice,
    DebugCompare,
    DebugIntegral,
}

impl Default for VteDisplayMode {
    fn default() -> Self {
        Self::Integral
    }
}

impl VteDisplayMode {
    fn as_u32(self) -> u32 {
        match self {
            Self::Integral => 0,
            Self::Slice => 1,
            Self::ThickSlice => 2,
            Self::DebugCompare => 3,
            Self::DebugIntegral => 4,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Integral => "integral",
            Self::Slice => "slice",
            Self::ThickSlice => "thick_slice",
            Self::DebugCompare => "debug_compare",
            Self::DebugIntegral => "debug_integral",
        }
    }
}

impl Default for RenderBackend {
    fn default() -> Self {
        Self::Auto
    }
}

impl RenderBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::TetraRaster => "tetra_raster",
            Self::TetraRaytrace => "tetra_raytrace",
            Self::VoxelTraversal => "voxel_traversal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CustomOverlayLine {
    pub start_ndc: [f32; 2],
    pub end_ndc: [f32; 2],
    pub color: [f32; 4],
}

pub struct RenderOptions {
    pub do_frame_clear: bool,
    pub do_raster: bool,
    pub do_raytrace: bool,
    pub render_backend: RenderBackend,
    pub vte_max_trace_steps: u32,
    pub vte_max_trace_distance: f32,
    pub vte_display_mode: VteDisplayMode,
    pub vte_slice_layer: Option<u32>,
    pub vte_thick_half_width: u32,
    pub vte_reference_compare: bool,
    pub vte_reference_mismatch_only: bool,
    pub vte_compare_slice_only: bool,
    pub vte_highlight_hit_voxel: Option<[i32; 4]>,
    pub vte_highlight_place_voxel: Option<[i32; 4]>,
    pub do_edges: bool,
    pub do_tetrahedron_edges: bool,
    pub do_navigation_hud: bool,
    pub custom_overlay_lines: Vec<CustomOverlayLine>,
    pub take_framebuffer_screenshot: bool,
    pub prepare_render_screenshot: bool,
    pub hud_rotation_label: Option<String>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            do_frame_clear: false,
            do_raster: true,
            do_raytrace: false,
            render_backend: RenderBackend::Auto,
            vte_max_trace_steps: 320,
            vte_max_trace_distance: 160.0,
            vte_display_mode: VteDisplayMode::Integral,
            vte_slice_layer: None,
            vte_thick_half_width: 2,
            vte_reference_compare: false,
            vte_reference_mismatch_only: false,
            vte_compare_slice_only: false,
            vte_highlight_hit_voxel: None,
            vte_highlight_place_voxel: None,
            do_edges: false,
            do_tetrahedron_edges: false,
            do_navigation_hud: false,
            custom_overlay_lines: Vec::new(),
            take_framebuffer_screenshot: false,
            prepare_render_screenshot: false,
            hud_rotation_label: None,
        }
    }
}

pub struct FrameParams {
    pub view_matrix: ndarray::Array2<f32>,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub render_options: RenderOptions,
}

pub struct TetraFrameInput<'a> {
    pub model_instances: &'a [common::ModelInstance],
}

#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelChunkHeader {
    pub chunk_coord: [i32; 4],
    pub occupancy_word_offset: u32,
    pub material_word_offset: u32,
    pub flags: u32,
    pub _padding: u32,
}

impl GpuVoxelChunkHeader {
    pub const FLAG_EMPTY: u32 = 1 << 0;
    pub const FLAG_FULL: u32 = 1 << 1;
}

pub struct VoxelFrameInput<'a> {
    pub chunk_headers: &'a [GpuVoxelChunkHeader],
    pub occupancy_words: &'a [u32],
    pub material_words: &'a [u32],
    pub visible_chunk_indices: &'a [u32],
}

#[derive(Copy, Clone, Default, Pod, Zeroable)]
#[repr(C)]
struct GpuVoxelFrameMeta {
    chunk_count: u32,
    visible_chunk_count: u32,
    occupancy_word_count: u32,
    material_word_count: u32,
    max_trace_steps: u32,
    max_trace_distance: f32,
    chunk_lookup_capacity: u32,
    stage_b_mode: u32,
    stage_b_slice_layer: u32,
    stage_b_thick_half_width: u32,
    debug_flags: u32,
    visible_chunk_min_x: i32,
    visible_chunk_min_y: i32,
    visible_chunk_min_z: i32,
    visible_chunk_min_w: i32,
    visible_chunk_max_x: i32,
    visible_chunk_max_y: i32,
    visible_chunk_max_z: i32,
    visible_chunk_max_w: i32,
    highlight_flags: u32,
    _highlight_padding: [u32; 3],
    highlight_hit_voxel: [i32; 4],
    highlight_place_voxel: [i32; 4],
}

#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct GpuVoxelChunkLookupEntry {
    chunk_coord: [i32; 4],
    chunk_index: u32,
    _padding: [u32; 3],
}

impl GpuVoxelChunkLookupEntry {
    const INVALID_INDEX: u32 = u32::MAX;

    fn empty() -> Self {
        Self {
            chunk_coord: [0; 4],
            chunk_index: Self::INVALID_INDEX,
            _padding: [0; 3],
        }
    }
}

impl Default for GpuVoxelChunkLookupEntry {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Copy, Clone, Default)]
pub struct VteDebugCounters {
    pub candidate_chunks: u32,
    pub frustum_culled_chunks: u32,
    pub empty_chunks_skipped: u32,
    pub macro_cells_skipped: u32,
    pub chunk_steps: u64,
    pub voxel_steps: u64,
    pub primary_hits: u64,
    pub s_samples: u64,
    pub visible_set_hash_valid: bool,
    pub visible_set_hash: u32,
}

#[derive(Copy, Clone, Default)]
struct VteCompareStats {
    compared: u32,
    matches: u32,
    mismatches: u32,
    hit_state_mismatches: u32,
    chunk_material_mismatches: u32,
    fast_miss_ref_hit: u32,
    fast_hit_ref_miss: u32,
    miss_reason_counts: [u32; 6],
    zero_interval_flags: u32,
    tie_stepped_flags: u32,
    lookup_fallback_flags: u32,
}

#[derive(Copy, Clone, Default)]
struct VteFirstMismatch {
    valid: bool,
    pixel_x: u32,
    pixel_y: u32,
    layer: u32,
    mismatch_kind: u32,
    miss_reason: u32,
    debug_flags: u32,
    fast_hit: bool,
    ref_hit: bool,
    fast_chunk: [i32; 4],
    ref_chunk: [i32; 4],
    fast_material: u32,
    ref_material: u32,
    fast_hit_t: f32,
    ref_hit_t: f32,
    chunk_steps_taken: u32,
    remaining_voxel_steps: u32,
    final_t: f32,
    last_chunk: [i32; 4],
}

const LINE_VERTEX_CAPACITY: usize = 100_000;
const HUD_BREADCRUMB_CAPACITY: usize = 128;
const HUD_BREADCRUMB_MIN_STEP: f32 = 0.2;
const FRAMES_IN_FLIGHT: usize = 2;
const VTE_MAX_CHUNKS: usize = 8_192;
const VTE_OCCUPANCY_WORDS_PER_CHUNK: usize = 128; // 8^4 / 32
const VTE_MATERIAL_WORDS_PER_CHUNK: usize = 1_024; // 8^4 / 4 packed u8
const VTE_CHUNK_LOOKUP_CAPACITY: usize = 32_768; // Must be power of two.
const VTE_DEBUG_FLAG_REFERENCE_COMPARE: u32 = 1 << 0;
const VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY: u32 = 1 << 1;
const VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY: u32 = 1 << 2;
const VTE_HIGHLIGHT_FLAG_HIT_VOXEL: u32 = 1 << 0;
const VTE_HIGHLIGHT_FLAG_PLACE_VOXEL: u32 = 1 << 1;
const VTE_COMPARE_STATS_WORD_COUNT: usize = 16;
const VTE_COMPARE_STAT_COMPARED: usize = 0;
const VTE_COMPARE_STAT_MATCHES: usize = 1;
const VTE_COMPARE_STAT_MISMATCHES: usize = 2;
const VTE_COMPARE_STAT_HIT_STATE_MISMATCHES: usize = 3;
const VTE_COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES: usize = 4;
const VTE_COMPARE_STAT_FAST_MISS_REF_HIT: usize = 5;
const VTE_COMPARE_STAT_FAST_HIT_REF_MISS: usize = 6;
const VTE_COMPARE_STAT_REASON_NONE: usize = 7;
const VTE_COMPARE_STAT_REASON_TOUCHED_VISIBLE: usize = 8;
const VTE_COMPARE_STAT_REASON_VOXEL_BUDGET: usize = 9;
const VTE_COMPARE_STAT_REASON_CHUNK_BUDGET: usize = 10;
const VTE_COMPARE_STAT_REASON_MAX_DISTANCE: usize = 11;
const VTE_COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE: usize = 12;
const VTE_COMPARE_STAT_ZERO_INTERVAL_FLAG: usize = 13;
const VTE_COMPARE_STAT_TIE_STEPPED_FLAG: usize = 14;
const VTE_COMPARE_STAT_LOOKUP_FALLBACK_FLAG: usize = 15;
const VTE_FIRST_MISMATCH_WORD_COUNT: usize = 27;
const VTE_FIRST_MISMATCH_VALID: usize = 0;
const VTE_FIRST_MISMATCH_PIXEL_X: usize = 1;
const VTE_FIRST_MISMATCH_PIXEL_Y: usize = 2;
const VTE_FIRST_MISMATCH_LAYER: usize = 3;
const VTE_FIRST_MISMATCH_KIND: usize = 4;
const VTE_FIRST_MISMATCH_MISS_REASON: usize = 5;
const VTE_FIRST_MISMATCH_DEBUG_FLAGS: usize = 6;
const VTE_FIRST_MISMATCH_HIT_MASK: usize = 7;
const VTE_FIRST_MISMATCH_FAST_CHUNK_X: usize = 8;
const VTE_FIRST_MISMATCH_FAST_CHUNK_Y: usize = 9;
const VTE_FIRST_MISMATCH_FAST_CHUNK_Z: usize = 10;
const VTE_FIRST_MISMATCH_FAST_CHUNK_W: usize = 11;
const VTE_FIRST_MISMATCH_REF_CHUNK_X: usize = 12;
const VTE_FIRST_MISMATCH_REF_CHUNK_Y: usize = 13;
const VTE_FIRST_MISMATCH_REF_CHUNK_Z: usize = 14;
const VTE_FIRST_MISMATCH_REF_CHUNK_W: usize = 15;
const VTE_FIRST_MISMATCH_FAST_MATERIAL: usize = 16;
const VTE_FIRST_MISMATCH_REF_MATERIAL: usize = 17;
const VTE_FIRST_MISMATCH_FAST_HIT_T: usize = 18;
const VTE_FIRST_MISMATCH_REF_HIT_T: usize = 19;
const VTE_FIRST_MISMATCH_CHUNK_STEPS: usize = 20;
const VTE_FIRST_MISMATCH_REMAINING_VOXELS: usize = 21;
const VTE_FIRST_MISMATCH_FINAL_T: usize = 22;
const VTE_FIRST_MISMATCH_LAST_CHUNK_X: usize = 23;
const VTE_FIRST_MISMATCH_LAST_CHUNK_Y: usize = 24;
const VTE_FIRST_MISMATCH_LAST_CHUNK_Z: usize = 25;
const VTE_FIRST_MISMATCH_LAST_CHUNK_W: usize = 26;

#[inline]
fn vte_hash_chunk_coord(chunk_coord: [i32; 4]) -> u32 {
    let x = chunk_coord[0] as u32;
    let y = chunk_coord[1] as u32;
    let z = chunk_coord[2] as u32;
    let w = chunk_coord[3] as u32;
    x.wrapping_mul(0x8DA6_B343)
        ^ y.wrapping_mul(0xD816_3841)
        ^ z.wrapping_mul(0xCB1A_B31F)
        ^ w.wrapping_mul(0x1656_67B1)
}

struct FrameInFlight {
    live_buffers: LiveBuffers,
    line_vertexes_buffer: Subbuffer<[LineVertex]>,
    hud_vertex_buffer: Option<Subbuffer<[HudVertex]>>,
    hud_descriptor_set: Option<Arc<DescriptorSet>>,
    sized_descriptor_set: Arc<DescriptorSet>,
    cpu_clipped_tet_count_buffer: Subbuffer<[u32]>,
    query_pool: Arc<QueryPool>,
    fence: Option<Box<dyn GpuFuture>>,
    vte_compare_enabled: bool,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    position: Vec2,
    _padding: [f32; 2],
    color: Vec4,
}

impl LineVertex {
    fn new(position: Vec2, color: Vec4) -> Self {
        Self {
            position,
            _padding: [0.0; 2],
            color,
        }
    }
}

#[derive(Copy, Clone)]
struct OverlayLine {
    start: Vec2,
    end: Vec2,
    color: Vec4,
}

const HUD_VERTEX_CAPACITY: usize = 64_000;

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct HudVertex {
    position: Vec2,
    texcoord: Vec2,
    color: Vec4,
}

#[derive(Clone)]
struct GlyphInfo {
    uv_min: Vec2,
    uv_max: Vec2,
    size_px: Vec2,
    bearing_px: Vec2,
    advance_px: f32,
}

struct FontAtlas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    glyphs: [Option<GlyphInfo>; 128],
    white_uv: Vec2,
    line_height: f32,
    ascent: f32,
}

fn build_font_atlas(font: &FontArc, pixel_size: f32) -> FontAtlas {
    let atlas_w: u32 = 512;
    let atlas_h: u32 = 256;
    let pixels = vec![0u8; (atlas_w * atlas_h) as usize];
    let glyphs: [Option<GlyphInfo>; 128] = std::array::from_fn(|_| None);
    let mut atlas = FontAtlas {
        width: atlas_w,
        height: atlas_h,
        pixels,
        glyphs,
        white_uv: Vec2::ZERO,
        line_height: 0.0,
        ascent: 0.0,
    };

    let scaled = font.as_scaled(pixel_size);
    atlas.line_height = scaled.height() + scaled.line_gap();
    atlas.ascent = scaled.ascent();

    // Shelf packing
    let mut cursor_x: u32 = 0;
    let mut cursor_y: u32 = 0;
    let mut row_height: u32 = 0;
    let pad = 2u32; // padding between glyphs

    for ch in 32u8..=126 {
        let glyph_id = scaled.glyph_id(ch as char);
        let glyph = glyph_id.with_scale_and_position(pixel_size, point(0.0, scaled.ascent()));
        let advance = scaled.h_advance(glyph_id);

        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            let gw = bounds.width().ceil() as u32;
            let gh = bounds.height().ceil() as u32;

            if gw == 0 || gh == 0 {
                atlas.glyphs[ch as usize] = Some(GlyphInfo {
                    uv_min: Vec2::ZERO,
                    uv_max: Vec2::ZERO,
                    size_px: Vec2::ZERO,
                    bearing_px: Vec2::new(bounds.min.x, bounds.min.y - scaled.ascent()),
                    advance_px: advance,
                });
                continue;
            }

            // Advance to next row if needed
            if cursor_x + gw + pad > atlas_w {
                cursor_x = 0;
                cursor_y += row_height + pad;
                row_height = 0;
            }
            if cursor_y + gh + pad > atlas_h {
                // Out of atlas space, skip remaining glyphs
                break;
            }

            // Rasterize glyph into atlas
            outline.draw(|x, y, c| {
                let px = cursor_x + x;
                let py = cursor_y + y;
                if px < atlas_w && py < atlas_h {
                    let idx = (py * atlas_w + px) as usize;
                    let val = (c * 255.0).clamp(0.0, 255.0) as u8;
                    if val > atlas.pixels[idx] {
                        atlas.pixels[idx] = val;
                    }
                }
            });

            atlas.glyphs[ch as usize] = Some(GlyphInfo {
                uv_min: Vec2::new(
                    cursor_x as f32 / atlas_w as f32,
                    cursor_y as f32 / atlas_h as f32,
                ),
                uv_max: Vec2::new(
                    (cursor_x + gw) as f32 / atlas_w as f32,
                    (cursor_y + gh) as f32 / atlas_h as f32,
                ),
                size_px: Vec2::new(gw as f32, gh as f32),
                bearing_px: Vec2::new(bounds.min.x, bounds.min.y - scaled.ascent()),
                advance_px: advance,
            });

            cursor_x += gw + pad;
            row_height = row_height.max(gh);
        } else {
            // No outline (e.g. space character)
            atlas.glyphs[ch as usize] = Some(GlyphInfo {
                uv_min: Vec2::ZERO,
                uv_max: Vec2::ZERO,
                size_px: Vec2::ZERO,
                bearing_px: Vec2::ZERO,
                advance_px: advance,
            });
        }
    }

    // Reserve a 4x4 white block for solid rectangles
    if cursor_x + 4 + pad > atlas_w {
        cursor_x = 0;
        cursor_y += row_height + pad;
    }
    for dy in 0..4u32 {
        for dx in 0..4u32 {
            let px = cursor_x + dx;
            let py = cursor_y + dy;
            if px < atlas_w && py < atlas_h {
                atlas.pixels[(py * atlas_w + px) as usize] = 255;
            }
        }
    }
    atlas.white_uv = Vec2::new(
        (cursor_x as f32 + 2.0) / atlas_w as f32,
        (cursor_y as f32 + 2.0) / atlas_h as f32,
    );

    atlas
}

struct HudResources {
    font_atlas: FontAtlas,
    atlas_view: Arc<ImageView>,
    atlas_sampler: Arc<Sampler>,
    hud_descriptor_set_layout: Arc<DescriptorSetLayout>,
}

impl HudResources {
    fn create_per_frame_hud(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    ) -> (Subbuffer<[HudVertex]>, Arc<DescriptorSet>) {
        let hud_vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![HudVertex::zeroed(); HUD_VERTEX_CAPACITY],
        )
        .unwrap();

        let hud_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            self.hud_descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, hud_vertex_buffer.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.atlas_view.clone(),
                    self.atlas_sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        (hud_vertex_buffer, hud_descriptor_set)
    }
}

fn push_text_quads(
    quads: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    text: &str,
    top_left_px: Vec2,
    pixel_size: f32,
    color: Vec4,
    present_size: [u32; 2],
) {
    let scale = pixel_size / 32.0; // atlas was rasterized at 32px
    let line_advance = atlas.line_height * scale;
    let mut caret_x = top_left_px.x;
    // ndc_to_pixels produces Y-up coords (pixel 0 = screen bottom, pixel h = screen top).
    // Baseline sits below the top of the text box.
    let mut baseline_y = top_left_px.y - atlas.ascent * scale;

    for ch in text.chars() {
        if ch == '\n' {
            caret_x = top_left_px.x;
            baseline_y -= line_advance; // next line goes down in Y-up
            continue;
        }

        let idx = ch as usize;
        if idx >= 128 {
            continue;
        }
        let glyph = match atlas.glyphs[idx].as_ref() {
            Some(g) => g,
            None => continue,
        };

        if glyph.size_px.x > 0.0 && glyph.size_px.y > 0.0 {
            let gw = glyph.size_px.x * scale;
            let gh = glyph.size_px.y * scale;
            let x0 = caret_x + glyph.bearing_px.x * scale;
            // bearing_px.y is Y-down offset from baseline to glyph top (negative).
            // In Y-up, negate it to get glyph top, then subtract gh for glyph bottom.
            let y0 = baseline_y - glyph.bearing_px.y * scale - gh;

            let p_min = pixels_to_ndc(Vec2::new(x0, y0), present_size);
            let p_max = pixels_to_ndc(Vec2::new(x0 + gw, y0 + gh), present_size);

            let uv_min = glyph.uv_min;
            let uv_max = glyph.uv_max;

            // Two triangles forming a quad
            // Note: pixels_to_ndc flips Y (pixel Y=0 → NDC Y=+1 = Vulkan bottom),
            // so p_min.y (larger NDC) is the screen bottom and p_max.y (smaller NDC)
            // is the screen top. We must swap UV Y to compensate: atlas top (uv_min.y)
            // maps to screen top (p_max.y), atlas bottom (uv_max.y) maps to screen
            // bottom (p_min.y).
            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_min.y),
                texcoord: Vec2::new(uv_min.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_min.y),
                texcoord: Vec2::new(uv_max.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_max.y),
                texcoord: Vec2::new(uv_max.x, uv_min.y),
                color,
            });

            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_min.y),
                texcoord: Vec2::new(uv_min.x, uv_max.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_max.x, p_max.y),
                texcoord: Vec2::new(uv_max.x, uv_min.y),
                color,
            });
            quads.push(HudVertex {
                position: Vec2::new(p_min.x, p_max.y),
                texcoord: Vec2::new(uv_min.x, uv_min.y),
                color,
            });
        }

        caret_x += glyph.advance_px * scale;
    }
}

fn push_filled_rect_quads(
    quads: &mut Vec<HudVertex>,
    atlas: &FontAtlas,
    min_ndc: Vec2,
    max_ndc: Vec2,
    color: Vec4,
) {
    let uv = atlas.white_uv;
    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });

    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, min_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(max_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });
    quads.push(HudVertex {
        position: Vec2::new(min_ndc.x, max_ndc.y),
        texcoord: uv,
        color,
    });
}

pub fn generate_tesseract_tetrahedrons() -> Vec<ModelTetrahedron> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let cube_vertexes = Hypercube::<3, usize>::generate_vertices();
    let tetrahedron_cells = Hypercube::<4, usize>::generate_k_faces_3();

    let mut output_tetrahedrons = Vec::new();
    let texture_position_simplexes =
        generate_simplexes_for_k_face_3::<3>([0b000, 0b001, 0b010, 0b100]);

    for cell_id in 0..tetrahedron_cells.len() {
        //for cell_id in 0..1 {
        let position_simplexes = generate_simplexes_for_k_face_3::<4>(tetrahedron_cells[cell_id]);

        for simplex_id in 0..position_simplexes.len() {
            let texture_simplex = texture_position_simplexes[simplex_id];

            // Convert arrays to vec4
            let mut vertex_positions = position_simplexes[simplex_id].map(|i| {
                let vertex = tesseract_vertexes[i];
                glam::Vec4::new(
                    vertex[0] as f32,
                    vertex[1] as f32,
                    vertex[2] as f32,
                    vertex[3] as f32,
                )
            });

            let mut texture_positions = texture_simplex.map(|i| {
                let vertex = cube_vertexes[i];
                glam::Vec4::new(vertex[0] as f32, vertex[1] as f32, vertex[2] as f32, 1.0)
            });

            // HACK!!! Fix this later with better math in the face/simplex generation
            // Determine if we need to re-order to flip normal
            let normal: Vec4 = get_normal(&[
                (vertex_positions[1] - vertex_positions[0]).into(),
                (vertex_positions[2] - vertex_positions[0]).into(),
                (vertex_positions[3] - vertex_positions[0]).into(),
            ])
            .into();
            let test_vector = Vec4::new(1.0, 1.0, 1.0, 1.0);
            let is_normal_flipped = test_vector.dot(normal.into()) < 0.0;
            let should_be_flipped = tetrahedron_cells[cell_id].contains(&0);
            if should_be_flipped != is_normal_flipped {
                let temp = vertex_positions[1];
                vertex_positions[1] = vertex_positions[2];
                vertex_positions[2] = temp;
                let temp = texture_positions[1];
                texture_positions[1] = texture_positions[2];
                texture_positions[2] = temp;
            }

            output_tetrahedrons.push(common::ModelTetrahedron {
                vertex_positions,
                texture_positions,
                cell_id: cell_id as u32,
                padding: [0; 3],
            })
        }
    }

    output_tetrahedrons
}

pub fn generate_tesseract_edges() -> Vec<common::ModelEdge> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let tesseract_edges = Hypercube::<4, usize>::generate_k_faces_1();

    let mut output_edges = Vec::new();

    for edge in tesseract_edges {
        let vertex_positions = edge.map(|i| {
            let vertex = tesseract_vertexes[i];
            glam::Vec4::new(
                vertex[0] as f32,
                vertex[1] as f32,
                vertex[2] as f32,
                vertex[3] as f32,
            )
        });

        output_edges.push(common::ModelEdge { vertex_positions });
    }

    output_edges
}

fn mat5_mul_vec5(mat: &nalgebra::Matrix5<f32>, v: [f32; 5]) -> [f32; 5] {
    let mut out = [0.0; 5];
    for r in 0..5 {
        for c in 0..5 {
            out[r] += mat[(r, c)] * v[c];
        }
    }
    out
}

fn transform_model_point(mat: &common::MatN<5>, point: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 5];
    let in_p = [point[0], point[1], point[2], point[3], 1.0];

    for r in 0..5 {
        for c in 0..5 {
            out[r] += mat[[r, c]] * in_p[c];
        }
    }

    let inv_w = if out[4].abs() > 1e-6 {
        1.0 / out[4]
    } else {
        1.0
    };
    [
        out[0] * inv_w,
        out[1] * inv_w,
        out[2] * inv_w,
        out[3] * inv_w,
    ]
}

fn project_view_point_to_ndc(
    view_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<Vec2> {
    let depth = (view_point[2] * view_point[2] + view_point[3] * view_point[3]).sqrt();
    if depth < 1e-4 {
        return None;
    }

    let projection_divisor = depth / focal_length_xy.max(1e-4);
    Some(Vec2::new(
        view_point[0] / projection_divisor,
        aspect * (-view_point[1]) / projection_divisor,
    ))
}

fn push_line(lines: &mut Vec<OverlayLine>, a: Vec2, b: Vec2, color: Vec4) {
    lines.push(OverlayLine {
        start: a,
        end: b,
        color,
    });
}

fn push_rect(lines: &mut Vec<OverlayLine>, min: Vec2, max: Vec2, color: Vec4) {
    let a = Vec2::new(min.x, min.y);
    let b = Vec2::new(max.x, min.y);
    let c = Vec2::new(max.x, max.y);
    let d = Vec2::new(min.x, max.y);
    push_line(lines, a, b, color);
    push_line(lines, b, c, color);
    push_line(lines, c, d, color);
    push_line(lines, d, a, color);
}

fn push_cross(lines: &mut Vec<OverlayLine>, center: Vec2, radius: f32, color: Vec4) {
    push_line(
        lines,
        center + Vec2::new(-radius, 0.0),
        center + Vec2::new(radius, 0.0),
        color,
    );
    push_line(
        lines,
        center + Vec2::new(0.0, -radius),
        center + Vec2::new(0.0, radius),
        color,
    );
}

fn map_to_panel(center: Vec2, half_size: Vec2, map_range: f32, a: f32, b: f32) -> Vec2 {
    let x = (a / map_range).clamp(-1.0, 1.0) * half_size.x;
    let y = (-b / map_range).clamp(-1.0, 1.0) * half_size.y; // negate for Vulkan +Y=down
    center + Vec2::new(x, y)
}

fn ndc_to_pixels(p: Vec2, present_size: [u32; 2]) -> Vec2 {
    let w = present_size[0] as f32;
    let h = present_size[1] as f32;
    Vec2::new((p.x * 0.5 + 0.5) * w, (1.0 - (p.y * 0.5 + 0.5)) * h)
}

fn pixels_to_ndc(p: Vec2, present_size: [u32; 2]) -> Vec2 {
    let w = present_size[0] as f32;
    let h = present_size[1] as f32;
    Vec2::new((p.x / w) * 2.0 - 1.0, 1.0 - (p.y / h) * 2.0)
}

fn load_hud_font() -> Option<FontArc> {
    const FONT_CANDIDATES: &[&str] = &[
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    ];

    if let Ok(custom_path) = std::env::var("HUD_FONT_PATH") {
        if let Ok(data) = std::fs::read(&custom_path) {
            if let Ok(font) = FontArc::try_from_vec(data) {
                return Some(font);
            }
        }
    }

    for &path in FONT_CANDIDATES {
        if let Ok(data) = std::fs::read(path) {
            if let Ok(font) = FontArc::try_from_vec(data) {
                return Some(font);
            }
        }
    }

    None
}

fn push_text_lines(
    lines: &mut Vec<OverlayLine>,
    font: &FontArc,
    text: &str,
    top_left_px: Vec2,
    pixel_size: f32,
    color: Vec4,
    present_size: [u32; 2],
) {
    let scaled = font.as_scaled(pixel_size);
    let h = present_size[1] as f32;
    let mut caret_x = top_left_px.x;
    // top_left_px is in Y-up pixel space (from ndc_to_pixels), but ab_glyph uses Y-down.
    // Convert: Y-down baseline = (h - top_left_y_up) + ascent
    let mut baseline_y = (h - top_left_px.y) + scaled.ascent();
    let line_advance = (scaled.height() + scaled.line_gap()).max(pixel_size * 1.2);
    let mut prev_glyph = None;

    for ch in text.chars() {
        if ch == '\n' {
            caret_x = top_left_px.x;
            baseline_y += line_advance; // Y-down: next line increases y
            prev_glyph = None;
            continue;
        }

        let glyph_id = scaled.glyph_id(ch);
        if let Some(prev) = prev_glyph {
            caret_x += scaled.kern(prev, glyph_id);
        }

        let glyph = glyph_id.with_scale_and_position(pixel_size, point(caret_x, baseline_y));
        if let Some(outline) = font.outline_glyph(glyph) {
            let bounds = outline.px_bounds();
            let width = bounds.width().max(0.0).ceil() as usize;
            let height = bounds.height().max(0.0).ceil() as usize;

            if width > 0 && height > 0 {
                let mut coverage = vec![0.0f32; width * height];
                outline.draw(|x, y, c| {
                    let idx = y as usize * width + x as usize;
                    if c > coverage[idx] {
                        coverage[idx] = c;
                    }
                });

                for y in 0..height {
                    let mut x = 0usize;
                    while x < width {
                        while x < width && coverage[y * width + x] < 0.35 {
                            x += 1;
                        }
                        if x >= width {
                            break;
                        }

                        let run_start = x;
                        while x < width && coverage[y * width + x] >= 0.35 {
                            x += 1;
                        }

                        // bounds are in Y-down pixels; convert to Y-up for pixels_to_ndc
                        let px_y_down = bounds.min.y + y as f32 + 0.5;
                        let px_y = h - px_y_down;
                        let px_start = Vec2::new(bounds.min.x + run_start as f32, px_y);
                        let px_end = Vec2::new(bounds.min.x + x as f32, px_y);
                        push_line(
                            lines,
                            pixels_to_ndc(px_start, present_size),
                            pixels_to_ndc(px_end, present_size),
                            color,
                        );
                    }
                }
            }
        }

        caret_x += scaled.h_advance(glyph_id);
        prev_glyph = Some(glyph_id);
    }
}

pub struct OneTimeBuffers {
    model_tetrahedron_count: usize,
    model_tetrahedron_buffer: Subbuffer<[common::ModelTetrahedron]>,
    model_edge_count: usize,
    model_edge_buffer: Subbuffer<[common::ModelEdge]>,
    descriptor_set: Arc<DescriptorSet>,
}

impl OneTimeBuffers {
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        let model_tetrahedrons = generate_tesseract_tetrahedrons();
        let model_tetrahedron_count = model_tetrahedrons.len();
        let model_tetrahedron_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model_tetrahedrons,
        )
        .unwrap();

        let model_edges = generate_tesseract_edges();
        let model_edge_count = model_edges.len();
        let model_edge_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model_edges,
        )
        .unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, model_tetrahedron_buffer.clone()),
                WriteDescriptorSet::buffer(1, model_edge_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        Self {
            model_tetrahedron_buffer,
            model_tetrahedron_count,
            model_edge_buffer,
            model_edge_count,
            descriptor_set,
        }
    }

    pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )
        .unwrap()
    }
}

const MAX_TETS_PER_TILE: usize = 8192;

pub struct SizedBuffers {
    render_dimensions: [u32; 3],
    pixel_storage_layers: u32,
    max_tetrahedrons: usize,
    output_tetrahedron_buffer: Subbuffer<[common::Tetrahedron]>,
    output_pixel_buffer: Subbuffer<[Vec4]>,
    morton_codes_buffer: Subbuffer<[common::MortonCode]>,
    bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    scene_bounds_buffer: Subbuffer<[common::SceneBounds]>,
    atomic_counter_buffer: Subbuffer<[u32]>,
    cpu_atomic_counter_buffer: Subbuffer<[u32]>,
    output_cpu_pixel_buffer: Subbuffer<[Vec4]>,
    // Tile binning buffers
    tile_tet_counts_buffer: Subbuffer<[u32]>,
    tile_tet_indices_buffer: Subbuffer<[u32]>,
    tile_count: u32,
    // Debug readback buffers
    cpu_bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    cpu_morton_codes_buffer: Subbuffer<[common::MortonCode]>,
}

impl SizedBuffers {
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        render_dimensions: [u32; 3],
        pixel_storage_layers: Option<u32>,
    ) -> Self {
        // Raytracing path currently expands voxel surface instances into full tesseract tet sets.
        // Keep this high enough for dense game scenes used in quality comparisons.
        let max_tetrahedrons: usize = 700_000;
        let logical_layers = render_dimensions[2].max(1);
        let storage_layers = pixel_storage_layers
            .unwrap_or(logical_layers)
            .clamp(1, logical_layers);
        let pixel_count = (render_dimensions[0] as usize)
            .saturating_mul(render_dimensions[1] as usize)
            .saturating_mul(storage_layers as usize);

        let output_tetrahedron_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::Tetrahedron::zeroed(); max_tetrahedrons],
        )
        .unwrap();

        let output_pixel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![Vec4::ZERO; pixel_count],
        )
        .unwrap();

        let output_cpu_pixel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![Vec4::ZERO; pixel_count],
        )
        .unwrap();

        // BVH buffers - GPU-only, initialized by compute shaders
        // Morton codes buffer padded to next power of 2 for bitonic sort
        let morton_codes_padded_count = max_tetrahedrons.next_power_of_two();
        let morton_codes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![
                common::MortonCode {
                    code: u64::MAX,
                    tetrahedron_index: 0xFFFFFFFF,
                    padding: 0
                };
                morton_codes_padded_count
            ],
        )
        .unwrap();

        // BVH needs 2N-1 nodes (N leaves + N-1 internal nodes)
        let bvh_node_count = 2 * max_tetrahedrons;
        let bvh_nodes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::BVHNode::zeroed(); bvh_node_count],
        )
        .unwrap();

        let scene_bounds_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::SceneBounds::zeroed(); 1],
        )
        .unwrap();

        // Atomic counter for tetrahedron clipping output
        let atomic_counter_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; 1],
        )
        .unwrap();

        let cpu_atomic_counter_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![0u32; 1],
        )
        .unwrap();

        // Tile binning buffers
        let tiles_x = (render_dimensions[0] + 7) / 8;
        let tiles_y = (render_dimensions[1] + 7) / 8;
        let tile_count = tiles_x * tiles_y;

        let tile_tet_counts_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; tile_count as usize],
        )
        .unwrap();

        let tile_tet_indices_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; tile_count as usize * MAX_TETS_PER_TILE],
        )
        .unwrap();

        // CPU readback buffers for BVH debugging
        let cpu_bvh_nodes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![common::BVHNode::zeroed(); bvh_node_count],
        )
        .unwrap();

        let cpu_morton_codes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![common::MortonCode::zeroed(); morton_codes_padded_count],
        )
        .unwrap();

        Self {
            render_dimensions,
            pixel_storage_layers: storage_layers,
            max_tetrahedrons,
            output_tetrahedron_buffer,
            output_pixel_buffer,
            morton_codes_buffer,
            bvh_nodes_buffer,
            scene_bounds_buffer,
            atomic_counter_buffer,
            cpu_atomic_counter_buffer,
            output_cpu_pixel_buffer,
            tile_tet_counts_buffer,
            tile_tet_indices_buffer,
            tile_count,
            cpu_bvh_nodes_buffer,
            cpu_morton_codes_buffer,
        }
    }

    fn create_sized_descriptor_set(
        &self,
        line_vertexes_buffer: &Subbuffer<[LineVertex]>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, line_vertexes_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.output_tetrahedron_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.output_pixel_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.morton_codes_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.bvh_nodes_buffer.clone()),
                WriteDescriptorSet::buffer(5, self.scene_bounds_buffer.clone()),
                WriteDescriptorSet::buffer(6, self.atomic_counter_buffer.clone()),
                WriteDescriptorSet::buffer(7, self.tile_tet_counts_buffer.clone()),
                WriteDescriptorSet::buffer(8, self.tile_tet_indices_buffer.clone()),
            ],
            [],
        )
        .unwrap()
    }

    pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            2,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        // BVH buffers
        bindings.insert(
            3,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            4,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            5,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        // Atomic counter for tetrahedron clipping
        bindings.insert(
            6,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        // Tile binning buffers
        bindings.insert(
            7,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            8,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )
        .unwrap()
    }
}

pub struct LiveBuffers {
    model_instance_buffer: Subbuffer<[common::ModelInstance]>,
    working_data_buffer: Subbuffer<common::WorkingData>,
    voxel_frame_meta_buffer: Subbuffer<GpuVoxelFrameMeta>,
    voxel_chunk_headers_buffer: Subbuffer<[GpuVoxelChunkHeader]>,
    voxel_occupancy_words_buffer: Subbuffer<[u32]>,
    voxel_material_words_buffer: Subbuffer<[u32]>,
    voxel_visible_chunk_indices_buffer: Subbuffer<[u32]>,
    voxel_chunk_lookup_buffer: Subbuffer<[GpuVoxelChunkLookupEntry]>,
    vte_compare_stats_buffer: Subbuffer<[u32]>,
    vte_first_mismatch_buffer: Subbuffer<[u32]>,
    descriptor_set: Arc<DescriptorSet>,
}

impl LiveBuffers {
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        let model_instance_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::ModelInstance::zeroed(); 100000],
        )
        .unwrap();

        let working_data_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            common::WorkingData::zeroed(),
        )
        .unwrap();

        let voxel_frame_meta_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            GpuVoxelFrameMeta::zeroed(),
        )
        .unwrap();

        let voxel_chunk_headers_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![GpuVoxelChunkHeader::zeroed(); VTE_MAX_CHUNKS],
        )
        .unwrap();

        let voxel_occupancy_words_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; VTE_MAX_CHUNKS * VTE_OCCUPANCY_WORDS_PER_CHUNK],
        )
        .unwrap();

        let voxel_material_words_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; VTE_MAX_CHUNKS * VTE_MATERIAL_WORDS_PER_CHUNK],
        )
        .unwrap();

        let voxel_visible_chunk_indices_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; VTE_MAX_CHUNKS],
        )
        .unwrap();

        let voxel_chunk_lookup_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![GpuVoxelChunkLookupEntry::empty(); VTE_CHUNK_LOOKUP_CAPACITY],
        )
        .unwrap();

        let vte_compare_stats_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; VTE_COMPARE_STATS_WORD_COUNT],
        )
        .unwrap();

        let vte_first_mismatch_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u32; VTE_FIRST_MISMATCH_WORD_COUNT],
        )
        .unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, model_instance_buffer.clone()),
                WriteDescriptorSet::buffer(1, working_data_buffer.clone()),
                WriteDescriptorSet::buffer(2, voxel_frame_meta_buffer.clone()),
                WriteDescriptorSet::buffer(3, voxel_chunk_headers_buffer.clone()),
                WriteDescriptorSet::buffer(4, voxel_occupancy_words_buffer.clone()),
                WriteDescriptorSet::buffer(5, voxel_material_words_buffer.clone()),
                WriteDescriptorSet::buffer(6, voxel_visible_chunk_indices_buffer.clone()),
                WriteDescriptorSet::buffer(7, voxel_chunk_lookup_buffer.clone()),
                WriteDescriptorSet::buffer(8, vte_compare_stats_buffer.clone()),
                WriteDescriptorSet::buffer(9, vte_first_mismatch_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        Self {
            model_instance_buffer,
            working_data_buffer,
            voxel_frame_meta_buffer,
            voxel_chunk_headers_buffer,
            voxel_occupancy_words_buffer,
            voxel_material_words_buffer,
            voxel_visible_chunk_indices_buffer,
            voxel_chunk_lookup_buffer,
            vte_compare_stats_buffer,
            vte_first_mismatch_buffer,
            descriptor_set,
        }
    }

    pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            },
        );
        for binding in 2..=9u32 {
            bindings.insert(
                binding,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            );
        }
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )
        .unwrap()
    }
}

struct PresentPipelineContext {
    line_pipeline: Arc<GraphicsPipeline>,
    buffer_pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    hud_pipeline: Arc<GraphicsPipeline>,
    hud_pipeline_layout: Arc<PipelineLayout>,
}

impl PresentPipelineContext {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        shaders: &ShaderModules,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Self {
        let line_pipeline = {
            // Load vertex and fragment shaders for line rendering
            let vs = shaders.line_vs.entry_point("mainLineVS").unwrap();
            let fs = shaders.line_fs.entry_point("mainLineFS").unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            // We have to indicate which subpass of which render pass this pipeline is going to be
            // used in. The pipeline will only be usable from this particular subpass.
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(VertexInputState::new()),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::LineList,
                        primitive_restart_enable: false,
                        ..Default::default()
                    }),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(RasterizationState {
                        depth_clamp_enable: false,
                        rasterizer_discard_enable: false,
                        polygon_mode: PolygonMode::Line,
                        cull_mode: Default::default(),
                        front_face: Default::default(),
                        depth_bias: None,
                        line_width: 1.0,
                        line_rasterization_mode: Default::default(),
                        line_stipple: None,
                        conservative: None,
                        ..Default::default()
                    }),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        let buffer_pipeline = {
            // Load vertex and fragment shaders for buffer display
            let vs = shaders.buffer_vs.entry_point("mainBufferVS").unwrap();
            let fs = shaders.buffer_fs.entry_point("mainBufferFS").unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            // We have to indicate which subpass of which render pass this pipeline is going to be
            // used in. The pipeline will only be usable from this particular subpass.
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(VertexInputState::new()),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(RasterizationState {
                        depth_clamp_enable: false,
                        rasterizer_discard_enable: false,
                        cull_mode: Default::default(),
                        front_face: Default::default(),
                        depth_bias: None,
                        line_width: 1.0,
                        line_rasterization_mode: Default::default(),
                        line_stipple: None,
                        conservative: None,
                        ..Default::default()
                    }),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        // HUD pipeline: separate descriptor set layout and pipeline layout
        let hud_descriptor_set_layout = {
            let mut bindings = BTreeMap::new();
            // Binding 0: StorageBuffer for HudVertex data
            bindings.insert(
                0,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::VERTEX,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            );
            // Binding 1: CombinedImageSampler for font atlas
            bindings.insert(
                1,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::FRAGMENT,
                    ..DescriptorSetLayoutBinding::descriptor_type(
                        DescriptorType::CombinedImageSampler,
                    )
                },
            );
            DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let hud_pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![hud_descriptor_set_layout],
                ..Default::default()
            },
        )
        .unwrap();

        let hud_pipeline = {
            let vs = shaders.hud_vs.entry_point("mainHudVS").unwrap();
            let fs = shaders.hud_fs.entry_point("mainHudFS").unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(VertexInputState::new()),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(hud_pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        Self {
            line_pipeline,
            buffer_pipeline,
            pipeline_layout,
            hud_pipeline,
            hud_pipeline_layout,
        }
    }
}

struct ComputePipelineContext {
    tetrahedron_pipeline: Arc<ComputePipeline>,
    edge_pipeline: Arc<ComputePipeline>,
    tetrahedron_pixel_pipeline: Arc<ComputePipeline>,
    bin_tets_pipeline: Arc<ComputePipeline>,
    raytrace_pre_pipeline: Arc<ComputePipeline>,
    raytrace_pixel_pipeline: Arc<ComputePipeline>,
    raytrace_clear_pipeline: Arc<ComputePipeline>,
    // Voxel traversal engine (VTE) pipelines
    voxel_clear_pipeline: Arc<ComputePipeline>,
    voxel_trace_stage_a_pipeline: Arc<ComputePipeline>,
    voxel_display_stage_b_pipeline: Arc<ComputePipeline>,
    // BVH pipelines
    bvh_scene_bounds_pipeline: Arc<ComputePipeline>,
    bvh_morton_codes_pipeline: Arc<ComputePipeline>,
    bvh_bitonic_sort_local_pipeline: Arc<ComputePipeline>,
    bvh_bitonic_sort_pipeline: Arc<ComputePipeline>,
    bvh_bitonic_sort_local_merge_pipeline: Arc<ComputePipeline>,
    bvh_init_leaves_pipeline: Arc<ComputePipeline>,
    bvh_build_tree_pipeline: Arc<ComputePipeline>,
    bvh_compute_leaf_aabbs_pipeline: Arc<ComputePipeline>,
    bvh_propagate_aabbs_pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
}

impl ComputePipelineContext {
    pub fn new(
        device: Arc<Device>,
        shaders: &ShaderModules,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Self {
        Self {
            tetrahedron_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .tetrahedron_cs
                            .entry_point("mainTetrahedronCS")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            edge_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders.edge_cs.entry_point("mainEdgeCS").unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            tetrahedron_pixel_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .tetrahedron_pixel_cs
                            .entry_point("mainTetrahedronPixelCS")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bin_tets_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders.bin_tets_cs.entry_point("mainBinTetsCS").unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            raytrace_pre_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .raytrace_preprocess
                            .entry_point("mainRaytracerTetrahedronPreprocessor")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            raytrace_pixel_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .raytrace_pixel
                            .entry_point("mainRaytracerPixel")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            raytrace_clear_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .raytrace_clear
                            .entry_point("mainRaytracerClear")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            voxel_clear_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders.voxel_clear.entry_point("mainVoxelClear").unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            voxel_trace_stage_a_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .voxel_trace_stage_a
                            .entry_point("mainVoxelTraceStageA")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            voxel_display_stage_b_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .voxel_display_stage_b
                            .entry_point("mainVoxelDisplayStageB")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            // BVH pipelines
            bvh_scene_bounds_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_scene_bounds
                            .entry_point("mainBVHSceneBounds")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_morton_codes_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_morton_codes
                            .entry_point("mainBVHMortonCodes")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_bitonic_sort_local_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_bitonic_sort_local
                            .entry_point("mainBVHBitonicSortLocal")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_bitonic_sort_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_bitonic_sort
                            .entry_point("mainBVHBitonicSort")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_bitonic_sort_local_merge_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_bitonic_sort_local_merge
                            .entry_point("mainBVHBitonicSortLocalMerge")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_init_leaves_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_init_leaves
                            .entry_point("mainBVHInitLeaves")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_build_tree_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_build_tree
                            .entry_point("mainBVHBuildTree")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_compute_leaf_aabbs_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_compute_leaf_aabbs
                            .entry_point("mainBVHComputeLeafAABBs")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            bvh_propagate_aabbs_pipeline: ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(
                    PipelineShaderStageCreateInfo::new(
                        shaders
                            .bvh_propagate_aabbs
                            .entry_point("mainBVHPropagateAABBs")
                            .unwrap(),
                    ),
                    pipeline_layout.clone(),
                ),
            )
            .unwrap(),
            pipeline_layout,
        }
    }
}

const PROFILER_MAX_TIMESTAMPS: u32 = 16;
const PROFILER_REPORT_INTERVAL: usize = 100;
const PROFILER_SLOW_FRAME_THRESHOLD_MS: f64 = 100.0;
const PROFILER_SLOW_FRAME_REPORT_INTERVAL: usize = 60;

struct GpuProfiler {
    timestamp_period_ns: f32,
    next_query: u32,
    /// Phase name for each timestamp (interval is from previous to this timestamp)
    phase_names: Vec<&'static str>,
    /// Accumulated (total_ms, count) per phase name
    accum: Vec<(&'static str, f64, usize)>,
    total_frames: usize,
    last_slow_report_frame: Option<usize>,
    /// Last frame's GPU total in ms (for HUD display)
    last_gpu_total_ms: f32,
    /// Last frame's per-phase breakdown (for HUD display)
    last_frame_phases: Vec<(&'static str, f32)>,
}

impl GpuProfiler {
    fn new(device: Arc<Device>) -> Self {
        let timestamp_period_ns = device.physical_device().properties().timestamp_period;

        GpuProfiler {
            timestamp_period_ns,
            next_query: 0,
            phase_names: Vec::new(),
            accum: Vec::new(),
            total_frames: 0,
            last_slow_report_frame: None,
            last_gpu_total_ms: 0.0,
            last_frame_phases: Vec::new(),
        }
    }

    fn create_query_pool(device: &Arc<Device>) -> Arc<QueryPool> {
        QueryPool::new(
            device.clone(),
            QueryPoolCreateInfo {
                query_count: PROFILER_MAX_TIMESTAMPS,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap()
    }

    fn begin_frame(&mut self) {
        self.next_query = 0;
        self.phase_names.clear();
    }

    fn next_query_index(&mut self, name: &'static str) -> u32 {
        let idx = self.next_query;
        self.phase_names.push(name);
        self.next_query += 1;
        idx
    }

    fn read_results_and_accumulate(&mut self, query_pool: &Arc<QueryPool>, clipped_tet_count: u32) {
        let n = self.next_query;
        if n < 2 {
            return;
        }

        let mut results = vec![0u64; n as usize];
        match query_pool.get_results::<u64>(0..n, &mut results, QueryResultFlags::empty()) {
            Ok(true) => {}
            _ => return,
        }

        let period_ms = self.timestamp_period_ns as f64 / 1_000_000.0;

        self.last_frame_phases.clear();
        let mut total_ms = 0.0f64;

        // Compute intervals between consecutive timestamps
        for i in 1..n as usize {
            let interval_ms = (results[i] - results[i - 1]) as f64 * period_ms;
            let name = self.phase_names[i];
            total_ms += interval_ms;
            self.last_frame_phases.push((name, interval_ms as f32));

            // Accumulate into named buckets
            if let Some(entry) = self.accum.iter_mut().find(|(n, _, _)| *n == name) {
                entry.1 += interval_ms;
                entry.2 += 1;
            } else {
                self.accum.push((name, interval_ms, 1));
            }
        }
        self.last_gpu_total_ms = total_ms as f32;

        self.total_frames += 1;

        if total_ms > PROFILER_SLOW_FRAME_THRESHOLD_MS {
            let should_report = match self.last_slow_report_frame {
                None => true,
                Some(last_frame) => {
                    self.total_frames.saturating_sub(last_frame)
                        >= PROFILER_SLOW_FRAME_REPORT_INTERVAL
                }
            };
            if should_report {
                println!(
                    "!!! SLOW FRAME ({:.1} ms) — clipped_tets={} — per-phase:",
                    total_ms, clipped_tet_count
                );
                for (name, ms) in &self.last_frame_phases {
                    println!("  {}: {:.3} ms", name, ms);
                }
                self.last_slow_report_frame = Some(self.total_frames);
            }
        }

        if self.total_frames > 0 && self.total_frames % PROFILER_REPORT_INTERVAL == 0 {
            self.print_report();
        }
    }

    fn print_report(&mut self) {
        println!("=== GPU Profile ({} frames) ===", self.total_frames);

        let mut total_avg = 0.0f64;
        for (name, total_ms, count) in &self.accum {
            if *count > 0 {
                let avg = total_ms / *count as f64;
                total_avg += avg;
                println!("  {}: {:.3} ms (avg over {} samples)", name, avg, count);
            }
        }
        println!(
            "  Total avg: {:.3} ms ({:.0} FPS)",
            total_avg,
            1000.0 / total_avg.max(0.001)
        );

        println!("================================");

        // Reset accumulators
        self.accum.clear();
    }
}

pub struct RenderContext {
    pub window: Option<Arc<Window>>,
    swapchain: Option<Arc<Swapchain>>,
    render_pass: Option<Arc<RenderPass>>,
    framebuffers: Option<Vec<Arc<Framebuffer>>>,
    present_pipeline: Option<PresentPipelineContext>,
    compute_pipeline: ComputePipelineContext,
    viewport: Viewport,
    recreate_swapchain: bool,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    one_time_buffers: OneTimeBuffers,
    sized_buffers: SizedBuffers,
    frames_in_flight: Vec<FrameInFlight>,
    cpu_screen_capture_buffer: Subbuffer<[u8]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    frames_rendered: usize,
    bvh_scene_hash: u64,
    last_clipped_tet_count: u32,
    profiler: GpuProfiler,
    hud_font: Option<FontArc>,
    hud_resources: Option<HudResources>,
    hud_breadcrumbs: VecDeque<[f32; 4]>,
    hud_previous_camera: Option<[f32; 4]>,
    hud_previous_sample_time: Option<Instant>,
    hud_w_velocity: f32,
    frame_time_ms: f32,
    last_render_start: Option<Instant>,
    stall_trace: bool,
    last_backend: RenderBackend,
    vte_debug_counters: VteDebugCounters,
    vte_compare_stats: VteCompareStats,
    vte_first_mismatch: VteFirstMismatch,
    vte_backend_notice_printed: bool,
}

impl RenderContext {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
        window: Option<Arc<Window>>,
        render_dimensions: [u32; 3],
    ) -> RenderContext {
        Self::new_with_pixel_storage_layers(
            device,
            queue,
            instance,
            window,
            render_dimensions,
            None,
        )
    }

    pub fn new_with_pixel_storage_layers(
        device: Arc<Device>,
        queue: Arc<Queue>,
        instance: Arc<Instance>,
        window: Option<Arc<Window>>,
        render_dimensions: [u32; 3],
        pixel_storage_layers: Option<u32>,
    ) -> RenderContext {
        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let (surface, window_size) = match window.clone() {
            Some(window) => (
                Some(Surface::from_window(instance.clone(), window.clone()).unwrap()),
                window.inner_size(),
            ),
            None => (
                None,
                PhysicalSize {
                    width: render_dimensions[0],
                    height: render_dimensions[1],
                },
            ),
        };

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = match surface {
            Some(surface) => {
                // Querying the capabilities of the surface. When we create the swapchain we can only
                // pass values that are allowed by the capabilities.
                let surface_capabilities = device
                    .physical_device()
                    .surface_capabilities(&surface, Default::default())
                    .unwrap();

                // Choosing the internal format that the images will have.
                let image_formats = device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap();

                let preferred_swapchain_formats = [
                    Format::B8G8R8A8_SRGB,
                    Format::B8G8R8A8_UNORM,
                    Format::R8G8B8A8_SRGB,
                    Format::R8G8B8A8_UNORM,
                ];
                let (image_format, image_color_space) = preferred_swapchain_formats
                    .iter()
                    .find_map(|preferred| {
                        image_formats
                            .iter()
                            .copied()
                            .find(|(fmt, _)| *fmt == *preferred)
                    })
                    .or_else(|| {
                        image_formats
                            .iter()
                            .copied()
                            .find(|(fmt, _)| fmt.block_size() == 4)
                    })
                    .unwrap_or(image_formats[0]);
                //let image_format = R8G8B8A8_UNORM;
                // Please take a look at the docs for the meaning of the parameters we didn't mention.
                let (swapchain, images) = Swapchain::new(
                    device.clone(),
                    surface,
                    SwapchainCreateInfo {
                        // Some drivers report an `min_image_count` of 1, but fullscreen mode requires
                        // at least 2. Therefore we must ensure the count is at least 2, otherwise the
                        // program would crash when entering fullscreen mode on those drivers.
                        min_image_count: surface_capabilities.min_image_count.max(2),

                        image_format,
                        image_color_space,

                        // The size of the window, only used to initially setup the swapchain.
                        //
                        // NOTE:
                        // On some drivers the swapchain extent is specified by
                        // `surface_capabilities.current_extent` and the swapchain size must use this
                        // extent. This extent is always the same as the window size.
                        //
                        // However, other drivers don't specify a value, i.e.
                        // `surface_capabilities.current_extent` is `None`. These drivers will allow
                        // anything, but the only sensible value is the window size.
                        //
                        // Both of these cases need the swapchain to use the window size, so we just
                        // use that.
                        image_extent: window_size.into(),

                        image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,

                        // The alpha mode indicates how the alpha value of the final image will behave.
                        // For example, you can choose whether the window will be
                        // opaque or transparent.
                        composite_alpha: surface_capabilities
                            .supported_composite_alpha
                            .into_iter()
                            .next()
                            .unwrap(),

                        ..Default::default()
                    },
                )
                .unwrap();
                (Some(swapchain), Some(images))
            }
            None => (None, None),
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Load shaders at runtime from embedded SPIR-V bytes
        // (vulkano_shaders macro can't parse Slang's SPIR-V 1.4 output)
        fn load_shader(device: Arc<Device>, spirv: &[u8]) -> Arc<ShaderModule> {
            unsafe {
                ShaderModule::from_bytes(device, spirv).expect("Failed to load shader module")
            }
        }

        let spirv_dir = std::path::Path::new(env!("SPIRV_OUT_DIR"));

        let raytrace_pixel = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerPixel.spv"))
                .expect("Failed to read shader"),
        );
        let raytrace_preprocess = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerTetrahedronPreprocessor.spv"))
                .expect("Failed to read shader"),
        );
        let raytrace_clear = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerClear.spv"))
                .expect("Failed to read shader"),
        );
        let voxel_clear = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainVoxelClear.spv")).expect("Failed to read shader"),
        );
        let voxel_trace_stage_a = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainVoxelTraceStageA.spv"))
                .expect("Failed to read shader"),
        );
        let voxel_display_stage_b = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainVoxelDisplayStageB.spv"))
                .expect("Failed to read shader"),
        );
        let raster_tet = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainTetrahedronCS.spv")).expect("Failed to read shader"),
        );
        let raster_edge = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainEdgeCS.spv")).expect("Failed to read shader"),
        );
        let raster_pixel = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainTetrahedronPixelCS.spv"))
                .expect("Failed to read shader"),
        );
        let bin_tets_cs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBinTetsCS.spv")).expect("Failed to read shader"),
        );
        let present_line_vs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainLineVS.spv")).expect("Failed to read shader"),
        );
        let present_line_fs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainLineFS.spv")).expect("Failed to read shader"),
        );
        let present_buffer_vs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBufferVS.spv")).expect("Failed to read shader"),
        );
        let present_buffer_fs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBufferFS.spv")).expect("Failed to read shader"),
        );
        // HUD shaders
        let hud_vs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainHudVS.spv")).expect("Failed to read shader"),
        );
        let hud_fs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainHudFS.spv")).expect("Failed to read shader"),
        );
        // BVH shaders
        let bvh_scene_bounds = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHSceneBounds.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_morton_codes = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHMortonCodes.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_bitonic_sort_local = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBitonicSortLocal.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_bitonic_sort = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBitonicSort.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_bitonic_sort_local_merge = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBitonicSortLocalMerge.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_init_leaves = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHInitLeaves.spv")).expect("Failed to read shader"),
        );
        let bvh_build_tree = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBuildTree.spv")).expect("Failed to read shader"),
        );
        let bvh_compute_leaf_aabbs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHComputeLeafAABBs.spv"))
                .expect("Failed to read shader"),
        );
        let bvh_propagate_aabbs = load_shader(
            device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHPropagateAABBs.spv"))
                .expect("Failed to read shader"),
        );

        let render_pass = match swapchain.clone() {
            Some(swapchain) => {
                // The next step is to create a *render pass*, which is an object that describes where the
                // output of the graphics pipeline will go. It describes the layout of the images where the
                // colors, depth and/or stencil information will be written.
                Some(
                    vulkano::single_pass_renderpass!(
                        device.clone(),
                        attachments: {
                            // `color` is a custom name we give to the first and only attachment.
                            color: {
                                // `format: <ty>` indicates the type of the format of the image. This has to be
                                // one of the types of the `vulkano::format` module (or alternatively one of
                                // your structs that implements the `FormatDesc` trait). Here we use the same
                                // format as the swapchain.
                                format: swapchain.image_format(),
                                // `samples: 1` means that we ask the GPU to use one sample to determine the
                                // value of each pixel in the color attachment. We could use a larger value
                                // (multisampling) for antialiasing. An example of this can be found in
                                // msaa-renderpass.rs.
                                samples: 1,
                                // `load_op: Clear` means that we ask the GPU to clear the content of this
                                // attachment at the start of the drawing.
                                load_op: Clear,
                                // `store_op: Store` means that we ask the GPU to store the output of the draw
                                // in the actual image. We could also ask it to discard the result.
                                store_op: Store,
                            },
                        },
                        pass: {
                            // We use the attachment named `color` as the one and only color attachment.
                            color: [color],
                            // No depth-stencil attachment is indicated with empty brackets.
                            depth_stencil: {},
                        },
                    )
                    .unwrap(),
                )
            }
            None => None,
        };

        let framebuffers = match render_pass.clone() {
            Some(render_pass) => match images {
                Some(images) => Some(window_size_dependent_setup(&images, &render_pass)),
                None => None,
            },
            None => None,
        };

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let one_time_descriptor_set_layout =
            OneTimeBuffers::create_descriptor_set_layout(device.clone());
        let one_time_buffers = OneTimeBuffers::new(
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            one_time_descriptor_set_layout.clone(),
        );

        let sized_descriptor_set_layout =
            SizedBuffers::create_descriptor_set_layout(device.clone());
        let sized_buffers = SizedBuffers::new(
            memory_allocator.clone(),
            render_dimensions,
            pixel_storage_layers,
        );

        let live_descriptor_set_layout = LiveBuffers::create_descriptor_set_layout(device.clone());

        // We must now create a **pipeline layout** object, which describes the locations and
        // types of descriptor sets and push constants used by the shaders in the pipeline.
        //
        // Multiple pipelines can share a common layout object, which is more efficient. The
        // shaders in a pipeline must use a subset of the resources described in its pipeline
        // layout, but the pipeline layout is allowed to contain resources that are not present
        // in the shaders; they can be used by shaders in other pipelines that share the same
        // layout. Thus, it is a good idea to design shaders so that many pipelines have common
        // resource locations, which allows them to share pipeline layouts.
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: Vec::from([
                    one_time_descriptor_set_layout.clone(),
                    sized_descriptor_set_layout.clone(),
                    live_descriptor_set_layout.clone(),
                ]),
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: 16, // 4 u32s: stage, step, count, padding
                }],
                ..Default::default()
            },
        )
        .unwrap();

        // Create ShaderModules struct from individually loaded shaders
        let shaders = ShaderModules {
            line_vs: present_line_vs,
            line_fs: present_line_fs,
            buffer_vs: present_buffer_vs,
            buffer_fs: present_buffer_fs,
            hud_vs,
            hud_fs,
            tetrahedron_cs: raster_tet,
            edge_cs: raster_edge,
            tetrahedron_pixel_cs: raster_pixel,
            bin_tets_cs,
            raytrace_preprocess: raytrace_preprocess,
            raytrace_pixel: raytrace_pixel,
            raytrace_clear: raytrace_clear,
            voxel_clear,
            voxel_trace_stage_a,
            voxel_display_stage_b,
            bvh_scene_bounds,
            bvh_morton_codes,
            bvh_bitonic_sort_local,
            bvh_bitonic_sort,
            bvh_bitonic_sort_local_merge,
            bvh_init_leaves,
            bvh_build_tree,
            bvh_compute_leaf_aabbs,
            bvh_propagate_aabbs,
        };

        let present_pipeline = match render_pass.clone() {
            Some(render_pass) => Some(PresentPipelineContext::new(
                device.clone(),
                render_pass.clone(),
                &shaders,
                pipeline_layout.clone(),
            )),
            None => None,
        };
        let compute_pipeline =
            ComputePipelineContext::new(device.clone(), &shaders, pipeline_layout.clone());

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let cpu_screen_capture_buffer = match swapchain.as_ref() {
            Some(swapchain) => {
                let [width, height] = swapchain.image_extent();
                create_cpu_screencapture_buffer(
                    memory_allocator.clone(),
                    width,
                    height,
                    swapchain.image_format(),
                )
            }
            None => create_cpu_screencapture_buffer(
                memory_allocator.clone(),
                window_size.width,
                window_size.height,
                Format::R8G8B8A8_UNORM,
            ),
        };

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's) or, on Android, when the application went to the background and goes back
        // to the foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        let profiler = GpuProfiler::new(device.clone());
        let hud_font = load_hud_font();

        // Build HUD resources (font atlas, GPU texture, vertex buffer, descriptor set)
        let hud_resources = match (&hud_font, &present_pipeline) {
            (Some(font), Some(present_ctx)) => {
                let font_atlas = build_font_atlas(font, 32.0);

                // Create staging buffer with atlas pixel data
                let staging_buffer = Buffer::from_iter(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_HOST
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    font_atlas.pixels.iter().copied(),
                )
                .unwrap();

                // Create GPU image for font atlas
                let atlas_image = Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R8_UNORM,
                        extent: [font_atlas.width, font_atlas.height, 1],
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                )
                .unwrap();

                // Upload atlas via one-shot command buffer
                {
                    let mut upload_builder = AutoCommandBufferBuilder::primary(
                        command_buffer_allocator.clone(),
                        queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    upload_builder
                        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                            staging_buffer,
                            atlas_image.clone(),
                        ))
                        .unwrap();
                    let upload_cmd = upload_builder.build().unwrap();
                    let upload_future = sync::now(device.clone())
                        .then_execute(queue.clone(), upload_cmd)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    upload_future.wait(None).unwrap();
                }

                let atlas_view = ImageView::new_default(atlas_image).unwrap();
                let atlas_sampler = Sampler::new(
                    device.clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        address_mode: [SamplerAddressMode::ClampToEdge; 3],
                        ..Default::default()
                    },
                )
                .unwrap();

                let hud_descriptor_set_layout = present_ctx
                    .hud_pipeline_layout
                    .set_layouts()
                    .first()
                    .unwrap()
                    .clone();

                Some(HudResources {
                    font_atlas,
                    atlas_view,
                    atlas_sampler,
                    hud_descriptor_set_layout,
                })
            }
            _ => None,
        };

        // Create per-frame resources
        let mut frames_in_flight = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let live_buffers = LiveBuffers::new(
                memory_allocator.clone(),
                descriptor_set_allocator.clone(),
                live_descriptor_set_layout.clone(),
            );

            let line_vertexes_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![LineVertex::zeroed(); LINE_VERTEX_CAPACITY],
            )
            .unwrap();

            let sized_descriptor_set = sized_buffers.create_sized_descriptor_set(
                &line_vertexes_buffer,
                descriptor_set_allocator.clone(),
                sized_descriptor_set_layout.clone(),
            );

            let cpu_clipped_tet_count_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
                },
                vec![0u32; 1],
            )
            .unwrap();

            let (hud_vertex_buffer, hud_descriptor_set) = match &hud_resources {
                Some(hud_res) => {
                    let (buf, ds) = hud_res.create_per_frame_hud(
                        memory_allocator.clone(),
                        descriptor_set_allocator.clone(),
                    );
                    (Some(buf), Some(ds))
                }
                None => (None, None),
            };

            let query_pool = GpuProfiler::create_query_pool(&device);

            frames_in_flight.push(FrameInFlight {
                live_buffers,
                line_vertexes_buffer,
                hud_vertex_buffer,
                hud_descriptor_set,
                sized_descriptor_set,
                cpu_clipped_tet_count_buffer,
                query_pool,
                fence: None,
                vte_compare_enabled: false,
            });
        }

        RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            present_pipeline,
            compute_pipeline,
            viewport,
            recreate_swapchain,
            command_buffer_allocator,
            descriptor_set_allocator,
            memory_allocator,
            one_time_buffers,
            sized_buffers,
            frames_in_flight,
            cpu_screen_capture_buffer,
            frames_rendered: 0,
            bvh_scene_hash: 0,
            last_clipped_tet_count: 0,
            profiler,
            hud_font,
            hud_resources,
            hud_breadcrumbs: VecDeque::new(),
            hud_previous_camera: None,
            hud_previous_sample_time: None,
            hud_w_velocity: 0.0,
            frame_time_ms: 0.0,
            last_render_start: None,
            stall_trace: std::env::var_os("R4D_TRACE_STALLS").is_some(),
            last_backend: RenderBackend::Auto,
            vte_debug_counters: VteDebugCounters::default(),
            vte_compare_stats: VteCompareStats::default(),
            vte_first_mismatch: VteFirstMismatch::default(),
            vte_backend_notice_printed: false,
        }
    }

    fn reset_vte_compare_buffers(&mut self, frame_idx: usize) {
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .vte_compare_stats_buffer
                .write()
                .unwrap();
            writer.fill(0u32);
        }
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .vte_first_mismatch_buffer
                .write()
                .unwrap();
            writer.fill(0u32);
        }
    }

    fn clear_vte_compare_diagnostics(&mut self) {
        self.vte_compare_stats = VteCompareStats::default();
        self.vte_first_mismatch = VteFirstMismatch::default();
    }

    fn refresh_vte_compare_diagnostics(&mut self, frame_idx: usize) {
        let stats_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_compare_stats_buffer
            .read()
            .unwrap();
        if stats_words.len() >= VTE_COMPARE_STATS_WORD_COUNT {
            self.vte_compare_stats = VteCompareStats {
                compared: stats_words[VTE_COMPARE_STAT_COMPARED],
                matches: stats_words[VTE_COMPARE_STAT_MATCHES],
                mismatches: stats_words[VTE_COMPARE_STAT_MISMATCHES],
                hit_state_mismatches: stats_words[VTE_COMPARE_STAT_HIT_STATE_MISMATCHES],
                chunk_material_mismatches: stats_words[VTE_COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES],
                fast_miss_ref_hit: stats_words[VTE_COMPARE_STAT_FAST_MISS_REF_HIT],
                fast_hit_ref_miss: stats_words[VTE_COMPARE_STAT_FAST_HIT_REF_MISS],
                miss_reason_counts: [
                    stats_words[VTE_COMPARE_STAT_REASON_NONE],
                    stats_words[VTE_COMPARE_STAT_REASON_TOUCHED_VISIBLE],
                    stats_words[VTE_COMPARE_STAT_REASON_VOXEL_BUDGET],
                    stats_words[VTE_COMPARE_STAT_REASON_CHUNK_BUDGET],
                    stats_words[VTE_COMPARE_STAT_REASON_MAX_DISTANCE],
                    stats_words[VTE_COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE],
                ],
                zero_interval_flags: stats_words[VTE_COMPARE_STAT_ZERO_INTERVAL_FLAG],
                tie_stepped_flags: stats_words[VTE_COMPARE_STAT_TIE_STEPPED_FLAG],
                lookup_fallback_flags: stats_words[VTE_COMPARE_STAT_LOOKUP_FALLBACK_FLAG],
            };
        } else {
            self.vte_compare_stats = VteCompareStats::default();
        }

        let first_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_first_mismatch_buffer
            .read()
            .unwrap();
        if first_words.len() >= VTE_FIRST_MISMATCH_WORD_COUNT
            && first_words[VTE_FIRST_MISMATCH_VALID] != 0
        {
            let hit_mask = first_words[VTE_FIRST_MISMATCH_HIT_MASK];
            self.vte_first_mismatch = VteFirstMismatch {
                valid: true,
                pixel_x: first_words[VTE_FIRST_MISMATCH_PIXEL_X],
                pixel_y: first_words[VTE_FIRST_MISMATCH_PIXEL_Y],
                layer: first_words[VTE_FIRST_MISMATCH_LAYER],
                mismatch_kind: first_words[VTE_FIRST_MISMATCH_KIND],
                miss_reason: first_words[VTE_FIRST_MISMATCH_MISS_REASON],
                debug_flags: first_words[VTE_FIRST_MISMATCH_DEBUG_FLAGS],
                fast_hit: (hit_mask & 0x1) != 0,
                ref_hit: (hit_mask & 0x2) != 0,
                fast_chunk: [
                    first_words[VTE_FIRST_MISMATCH_FAST_CHUNK_X] as i32,
                    first_words[VTE_FIRST_MISMATCH_FAST_CHUNK_Y] as i32,
                    first_words[VTE_FIRST_MISMATCH_FAST_CHUNK_Z] as i32,
                    first_words[VTE_FIRST_MISMATCH_FAST_CHUNK_W] as i32,
                ],
                ref_chunk: [
                    first_words[VTE_FIRST_MISMATCH_REF_CHUNK_X] as i32,
                    first_words[VTE_FIRST_MISMATCH_REF_CHUNK_Y] as i32,
                    first_words[VTE_FIRST_MISMATCH_REF_CHUNK_Z] as i32,
                    first_words[VTE_FIRST_MISMATCH_REF_CHUNK_W] as i32,
                ],
                fast_material: first_words[VTE_FIRST_MISMATCH_FAST_MATERIAL],
                ref_material: first_words[VTE_FIRST_MISMATCH_REF_MATERIAL],
                fast_hit_t: f32::from_bits(first_words[VTE_FIRST_MISMATCH_FAST_HIT_T]),
                ref_hit_t: f32::from_bits(first_words[VTE_FIRST_MISMATCH_REF_HIT_T]),
                chunk_steps_taken: first_words[VTE_FIRST_MISMATCH_CHUNK_STEPS],
                remaining_voxel_steps: first_words[VTE_FIRST_MISMATCH_REMAINING_VOXELS],
                final_t: f32::from_bits(first_words[VTE_FIRST_MISMATCH_FINAL_T]),
                last_chunk: [
                    first_words[VTE_FIRST_MISMATCH_LAST_CHUNK_X] as i32,
                    first_words[VTE_FIRST_MISMATCH_LAST_CHUNK_Y] as i32,
                    first_words[VTE_FIRST_MISMATCH_LAST_CHUNK_Z] as i32,
                    first_words[VTE_FIRST_MISMATCH_LAST_CHUNK_W] as i32,
                ],
            };
        } else {
            self.vte_first_mismatch = VteFirstMismatch::default();
        }
    }

    /// Returns (line_count, hud_vertex_count)
    fn write_navigation_hud_overlay(
        &mut self,
        frame_idx: usize,
        base_line_count: usize,
        view_matrix: &nalgebra::Matrix5<f32>,
        view_matrix_inverse: &nalgebra::Matrix5<f32>,
        focal_length_xy: f32,
        model_instances: &[common::ModelInstance],
        rotation_label: Option<&str>,
    ) -> (usize, usize) {
        let max_lines = LINE_VERTEX_CAPACITY / 2;
        if base_line_count >= max_lines {
            return (0, 0);
        }

        let axis_colors = [
            Vec4::new(1.00, 0.30, 0.30, 1.0), // X
            Vec4::new(0.30, 1.00, 0.45, 1.0), // Y
            Vec4::new(0.35, 0.65, 1.00, 1.0), // Z
            Vec4::new(1.00, 0.75, 0.25, 1.0), // W
        ];
        let rose_frame_color = Vec4::new(0.85, 0.85, 0.90, 1.0);
        let xy_frame_color = Vec4::new(0.25, 0.92, 0.92, 1.0);
        let zw_frame_color = Vec4::new(0.95, 0.55, 0.95, 1.0);
        let map_axis_color = Vec4::new(0.65, 0.65, 0.70, 1.0);
        let marker_xy_color = Vec4::new(0.70, 1.00, 1.00, 1.0);
        let marker_zw_color = Vec4::new(1.00, 0.85, 1.00, 1.0);
        let breadcrumb_xy_start = Vec4::new(0.18, 0.42, 0.50, 1.0);
        let breadcrumb_xy_end = Vec4::new(0.30, 0.95, 0.95, 1.0);
        let breadcrumb_zw_start = Vec4::new(0.35, 0.22, 0.42, 1.0);
        let breadcrumb_zw_end = Vec4::new(1.00, 0.50, 0.95, 1.0);
        let altimeter_color = Vec4::new(1.00, 0.75, 0.25, 1.0);
        let drift_color = Vec4::new(0.95, 0.95, 0.35, 1.0);

        let (present_size, text_scale) = match self.window.as_ref() {
            Some(window) => {
                let s = window.inner_size();
                let sf = window.scale_factor() as f32;
                ([s.width.max(1), s.height.max(1)], sf * 1.5)
            }
            None => (
                [
                    self.sized_buffers.render_dimensions[0].max(1),
                    self.sized_buffers.render_dimensions[1].max(1),
                ],
                2.0,
            ),
        };
        let aspect = present_size[0] as f32 / present_size[1] as f32;

        let camera_h = mat5_mul_vec5(view_matrix_inverse, [0.0, 0.0, 0.0, 0.0, 1.0]);
        let inv_w = if camera_h[4].abs() > 1e-6 {
            1.0 / camera_h[4]
        } else {
            1.0
        };
        let camera_position = [
            camera_h[0] * inv_w,
            camera_h[1] * inv_w,
            camera_h[2] * inv_w,
            camera_h[3] * inv_w,
        ];
        let look_view = [
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let look_world_h = mat5_mul_vec5(view_matrix_inverse, look_view);
        let mut look_world = [
            look_world_h[0],
            look_world_h[1],
            look_world_h[2],
            look_world_h[3],
        ];
        let look_len = (look_world[0] * look_world[0]
            + look_world[1] * look_world[1]
            + look_world[2] * look_world[2]
            + look_world[3] * look_world[3])
            .sqrt();
        if look_len > 1e-6 {
            for v in &mut look_world {
                *v /= look_len;
            }
        }

        let now = Instant::now();
        if let (Some(prev_camera), Some(prev_time)) =
            (self.hud_previous_camera, self.hud_previous_sample_time)
        {
            let dt = (now - prev_time).as_secs_f32();
            if dt > 1e-4 {
                self.hud_w_velocity = (camera_position[3] - prev_camera[3]) / dt;
            }
        }
        self.hud_previous_camera = Some(camera_position);
        self.hud_previous_sample_time = Some(now);

        let should_append_breadcrumb = match self.hud_breadcrumbs.back() {
            Some(last) => {
                let dx = camera_position[0] - last[0];
                let dy = camera_position[1] - last[1];
                let dz = camera_position[2] - last[2];
                let dw = camera_position[3] - last[3];
                (dx * dx + dy * dy + dz * dz + dw * dw).sqrt() >= HUD_BREADCRUMB_MIN_STEP
            }
            None => true,
        };
        if should_append_breadcrumb {
            self.hud_breadcrumbs.push_back(camera_position);
            while self.hud_breadcrumbs.len() > HUD_BREADCRUMB_CAPACITY {
                self.hud_breadcrumbs.pop_front();
            }
        }

        let mut instance_centers = Vec::<[f32; 4]>::with_capacity(model_instances.len());
        for instance in model_instances {
            instance_centers.push(transform_model_point(
                &instance.model_transform,
                [0.5, 0.5, 0.5, 0.5],
            ));
        }

        let mut lines = Vec::<OverlayLine>::with_capacity(2048);

        // Axis rose: orientation widget with labeled arrows and circle boundary.
        let rose_origin = Vec2::new(-0.78, 0.72);
        let rose_radius: f32 = 0.09;

        // Circle boundary (~32 segments)
        let circle_segments = 32;
        for i in 0..circle_segments {
            let a0 = (i as f32 / circle_segments as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / circle_segments as f32) * std::f32::consts::TAU;
            push_line(
                &mut lines,
                rose_origin + Vec2::new(a0.cos(), a0.sin()) * rose_radius,
                rose_origin + Vec2::new(a1.cos(), a1.sin()) * rose_radius,
                rose_frame_color,
            );
        }

        let axis_directions = [
            [1.0, 0.0, 0.0, 0.0], // X
            [0.0, 1.0, 0.0, 0.0], // Y
            [0.0, 0.0, 1.0, 0.0], // Z
            [0.0, 0.0, 0.0, 1.0], // W
        ];
        let fallback_dirs = [
            Vec2::new(1.0, 0.0),  // X: right
            Vec2::new(0.0, -1.0), // Y: up (Vulkan +Y=down)
            Vec2::new(-1.0, 0.0), // Z: left
            Vec2::new(0.0, 1.0),  // W: down (Vulkan +Y=down)
        ];
        let axis_labels = ["X", "Y", "Z", "W"];
        let arrow_len = rose_radius * 0.70;
        let arrowhead_back = rose_radius * 0.12;
        let arrowhead_side = rose_radius * 0.07;

        // Compute arrow data for all axes, then draw lines and labels
        struct ArrowData {
            ray_dir: Vec2,
            color: Vec4,
            label_pos: Vec2,
        }
        let mut arrows = Vec::with_capacity(4);

        for axis_id in 0..4 {
            let dir = axis_directions[axis_id];
            let view_dir = mat5_mul_vec5(view_matrix, [dir[0], dir[1], dir[2], dir[3], 0.0]);
            let projected = project_view_point_to_ndc(
                [view_dir[0], view_dir[1], view_dir[2], view_dir[3]],
                focal_length_xy,
                aspect,
            );

            let mut ray_dir = projected.unwrap_or(fallback_dirs[axis_id]);
            if ray_dir.length_squared() > 1e-8 {
                ray_dir = ray_dir.normalize();
            } else {
                ray_dir = fallback_dirs[axis_id];
            }

            // Depth dimming: arrows pointing away from camera are dimmed
            // Positive Z or W in view space means pointing away
            let depth_component = view_dir[2] + view_dir[3];
            let brightness = if depth_component > 0.0 { 0.45 } else { 1.0 };
            let axis_color = axis_colors[axis_id] * brightness;

            let tip = rose_origin + ray_dir * arrow_len;
            push_line(&mut lines, rose_origin, tip, axis_color);

            // Arrowhead
            let side = Vec2::new(-ray_dir.y, ray_dir.x);
            push_line(
                &mut lines,
                tip,
                tip - ray_dir * arrowhead_back + side * arrowhead_side,
                axis_color,
            );
            push_line(
                &mut lines,
                tip,
                tip - ray_dir * arrowhead_back - side * arrowhead_side,
                axis_color,
            );

            // Label position: just beyond arrowhead
            let label_pos = rose_origin + ray_dir * (arrow_len + rose_radius * 0.28);
            arrows.push(ArrowData {
                ray_dir,
                color: axis_color,
                label_pos,
            });
        }

        let mut hud_quads = Vec::<HudVertex>::with_capacity(2048);

        let rose_label_text_size = 13.0 * text_scale;
        let readout_text_size = 12.0 * text_scale;

        // Rose background
        if let Some(hud_res) = self.hud_resources.as_ref() {
            let panel_bg = Vec4::new(0.0, 0.0, 0.0, 0.45);
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                rose_origin - Vec2::splat(rose_radius * 1.15),
                rose_origin + Vec2::splat(rose_radius * 1.15),
                panel_bg,
            );
        }

        // Render axis labels at arrow tips
        for (axis_id, arrow) in arrows.iter().enumerate() {
            // Offset label position so text is centered on the label point
            let label_offset = Vec2::new(-arrow.ray_dir.x.abs() * 0.012, arrow.ray_dir.y * 0.012);
            let label_ndc = arrow.label_pos + label_offset;
            let label_px = ndc_to_pixels(label_ndc, present_size);
            if let Some(hud_res) = self.hud_resources.as_ref() {
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    axis_labels[axis_id],
                    label_px,
                    rose_label_text_size,
                    arrow.color,
                    present_size,
                );
            } else if let Some(font) = self.hud_font.as_ref() {
                push_text_lines(
                    &mut lines,
                    font,
                    axis_labels[axis_id],
                    label_px,
                    rose_label_text_size,
                    arrow.color,
                    present_size,
                );
            }
        }

        // Rotation mode label below compass rose
        if let Some(label) = rotation_label {
            let label_ndc = Vec2::new(rose_origin.x, rose_origin.y + rose_radius + 0.04);
            let label_text_size = 11.0 * text_scale;
            let label_color = Vec4::new(0.90, 0.90, 0.55, 1.0);
            if let Some(hud_res) = self.hud_resources.as_ref() {
                let label_px = ndc_to_pixels(label_ndc, present_size);
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    label,
                    label_px,
                    label_text_size,
                    label_color,
                    present_size,
                );
            } else if let Some(font) = self.hud_font.as_ref() {
                let label_px = ndc_to_pixels(label_ndc, present_size);
                push_text_lines(
                    &mut lines,
                    font,
                    label,
                    label_px,
                    label_text_size,
                    label_color,
                    present_size,
                );
            }
        }

        // Dual minimap: XZ on top (ground plane), YW below (height + 4th dim).
        let panel_size = Vec2::new(0.34, 0.20);
        let panel_half = panel_size * 0.5;
        let panel_gap = 0.04;
        let margin = Vec2::new(0.05, 0.06);
        let right = 1.0 - margin.x;
        let left = right - panel_size.x;
        let lower_bottom = -1.0 + margin.y;
        let lower_top = lower_bottom + panel_size.y;
        let upper_bottom = lower_top + panel_gap;
        let upper_top = upper_bottom + panel_size.y;
        // In Vulkan clip space +Y=down, so lower clip Y = higher on screen.
        // XZ on top (lower clip Y), YW below (higher clip Y).
        let xz_center = Vec2::new((left + right) * 0.5, (lower_bottom + lower_top) * 0.5);
        let yw_center = Vec2::new((left + right) * 0.5, (upper_bottom + upper_top) * 0.5);
        let xz_min = Vec2::new(left, lower_bottom);
        let xz_max = Vec2::new(right, lower_top);
        let yw_min = Vec2::new(left, upper_bottom);
        let yw_max = Vec2::new(right, upper_top);
        let map_range = 12.0;
        let xz_frame_color = xy_frame_color;
        let yw_frame_color = zw_frame_color;
        let marker_xz_color = marker_xy_color;
        let marker_yw_color = marker_zw_color;
        let breadcrumb_xz_start = breadcrumb_xy_start;
        let breadcrumb_xz_end = breadcrumb_xy_end;
        let breadcrumb_yw_start = breadcrumb_zw_start;
        let breadcrumb_yw_end = breadcrumb_zw_end;

        // Panel frames
        push_rect(&mut lines, xz_min, xz_max, xz_frame_color);
        push_rect(&mut lines, yw_min, yw_max, yw_frame_color);

        // Crosshair axes
        push_line(
            &mut lines,
            Vec2::new(xz_min.x, xz_center.y),
            Vec2::new(xz_max.x, xz_center.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(xz_center.x, xz_min.y),
            Vec2::new(xz_center.x, xz_max.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(yw_min.x, yw_center.y),
            Vec2::new(yw_max.x, yw_center.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(yw_center.x, yw_min.y),
            Vec2::new(yw_center.x, yw_max.y),
            map_axis_color,
        );

        // Scale tick marks every 4 world units along crosshairs (skip origin)
        let tick_size = 0.008;
        let tick_interval = 4.0;
        let max_ticks = (map_range / tick_interval) as i32;
        for panel_data in [(xz_center, xz_min, xz_max), (yw_center, yw_min, yw_max)] {
            let (center, pmin, pmax) = panel_data;
            for i in 1..=max_ticks {
                let frac = (i as f32 * tick_interval) / map_range;
                // Horizontal axis ticks (vertical dashes)
                for sign in [-1.0f32, 1.0] {
                    let tx = center.x + sign * frac * panel_half.x * 0.9;
                    if tx > pmin.x && tx < pmax.x {
                        push_line(
                            &mut lines,
                            Vec2::new(tx, center.y - tick_size),
                            Vec2::new(tx, center.y + tick_size),
                            map_axis_color * 0.6,
                        );
                    }
                    // Vertical axis ticks (horizontal dashes)
                    let ty = center.y + sign * frac * panel_half.y * 0.9;
                    if ty > pmin.y && ty < pmax.y {
                        push_line(
                            &mut lines,
                            Vec2::new(center.x - tick_size, ty),
                            Vec2::new(center.x + tick_size, ty),
                            map_axis_color * 0.6,
                        );
                    }
                }
            }
        }

        // Direction wedge on XZ panel: small triangle showing camera yaw
        {
            let yaw_x = look_world[0]; // X component of look direction
            let yaw_z = look_world[2]; // Z component of look direction
            let yaw_len = (yaw_x * yaw_x + yaw_z * yaw_z).sqrt();
            if yaw_len > 1e-4 {
                let dx = yaw_x / yaw_len;
                let dz = yaw_z / yaw_len;
                // Map to panel space: X horizontal, Z vertical (up on screen)
                // Negate dz because Vulkan +Y=down but +Z should point up on minimap
                let panel_aspect = panel_half.x / panel_half.y;
                let wedge_dir = Vec2::new(dx, -dz * panel_aspect).normalize();
                let wedge_side = Vec2::new(-wedge_dir.y, wedge_dir.x);
                let wedge_len = 0.025;
                let wedge_width = 0.012;
                let wedge_tip = xz_center + wedge_dir * wedge_len;
                let wedge_left = xz_center - wedge_dir * wedge_len * 0.3 + wedge_side * wedge_width;
                let wedge_right =
                    xz_center - wedge_dir * wedge_len * 0.3 - wedge_side * wedge_width;
                let wedge_color = Vec4::new(1.0, 1.0, 1.0, 0.9);
                push_line(&mut lines, wedge_tip, wedge_left, wedge_color);
                push_line(&mut lines, wedge_tip, wedge_right, wedge_color);
                push_line(&mut lines, wedge_left, wedge_right, wedge_color);
            }
        }

        // Direction wedge on YW panel: small triangle showing camera look in Y/W plane
        {
            let look_y = look_world[1]; // Y component of look direction
            let look_w = look_world[3]; // W component of look direction
            let yw_len = (look_y * look_y + look_w * look_w).sqrt();
            if yw_len > 1e-4 {
                let dy = look_y / yw_len;
                let dw = look_w / yw_len;
                // Map to panel space: Y horizontal, W vertical (up on screen)
                // Negate dw because Vulkan +Y=down but +W should point up on minimap
                let panel_aspect = panel_half.x / panel_half.y;
                let wedge_dir = Vec2::new(dy, -dw * panel_aspect).normalize();
                let wedge_side = Vec2::new(-wedge_dir.y, wedge_dir.x);
                let wedge_len = 0.025;
                let wedge_width = 0.012;
                let wedge_tip = yw_center + wedge_dir * wedge_len;
                let wedge_left = yw_center - wedge_dir * wedge_len * 0.3 + wedge_side * wedge_width;
                let wedge_right =
                    yw_center - wedge_dir * wedge_len * 0.3 - wedge_side * wedge_width;
                let wedge_color = Vec4::new(1.0, 1.0, 1.0, 0.9);
                push_line(&mut lines, wedge_tip, wedge_left, wedge_color);
                push_line(&mut lines, wedge_tip, wedge_right, wedge_color);
                push_line(&mut lines, wedge_left, wedge_right, wedge_color);
            }
        }

        // Panel title labels and axis labels
        let minimap_label_size = 11.0 * text_scale;
        if let Some(hud_res) = self.hud_resources.as_ref() {
            // Panel titles inside top-left corner (Vulkan: lower clip Y = higher on screen)
            let xz_title_px =
                ndc_to_pixels(Vec2::new(xz_min.x + 0.015, xz_min.y + 0.01), present_size);
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "XZ",
                xz_title_px,
                minimap_label_size,
                xz_frame_color,
                present_size,
            );
            let yw_title_px =
                ndc_to_pixels(Vec2::new(yw_min.x + 0.015, yw_min.y + 0.01), present_size);
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "YW",
                yw_title_px,
                minimap_label_size,
                yw_frame_color,
                present_size,
            );

            // XZ panel axis labels: X at right edge, Z at top edge (lower clip Y)
            let xz_x_label_px = ndc_to_pixels(
                Vec2::new(xz_max.x - 0.025, xz_center.y + 0.02),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "X",
                xz_x_label_px,
                minimap_label_size,
                axis_colors[0],
                present_size,
            );
            let xz_z_label_px = ndc_to_pixels(
                Vec2::new(xz_center.x + 0.015, xz_min.y + 0.01),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "Z",
                xz_z_label_px,
                minimap_label_size,
                axis_colors[2],
                present_size,
            );

            // YW panel axis labels: Y at right edge, W at top edge (lower clip Y)
            let yw_y_label_px = ndc_to_pixels(
                Vec2::new(yw_max.x - 0.025, yw_center.y + 0.02),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "Y",
                yw_y_label_px,
                minimap_label_size,
                axis_colors[1],
                present_size,
            );
            let yw_w_label_px = ndc_to_pixels(
                Vec2::new(yw_center.x + 0.015, yw_min.y + 0.01),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "W",
                yw_w_label_px,
                minimap_label_size,
                axis_colors[3],
                present_size,
            );
        } else if let Some(font) = self.hud_font.as_ref() {
            let xz_title_px =
                ndc_to_pixels(Vec2::new(xz_min.x + 0.015, xz_min.y + 0.01), present_size);
            push_text_lines(
                &mut lines,
                font,
                "XZ",
                xz_title_px,
                minimap_label_size,
                xz_frame_color,
                present_size,
            );
            let yw_title_px =
                ndc_to_pixels(Vec2::new(yw_min.x + 0.015, yw_min.y + 0.01), present_size);
            push_text_lines(
                &mut lines,
                font,
                "YW",
                yw_title_px,
                minimap_label_size,
                yw_frame_color,
                present_size,
            );
        }

        // Instance markers: XZ panel uses indices [0]=X, [2]=Z; YW panel uses [1]=Y, [3]=W
        for center in &instance_centers {
            let xz_point = map_to_panel(
                xz_center,
                panel_half * 0.9,
                map_range,
                center[0] - camera_position[0],
                center[2] - camera_position[2],
            );
            let yw_point = map_to_panel(
                yw_center,
                panel_half * 0.9,
                map_range,
                center[1] - camera_position[1],
                center[3] - camera_position[3],
            );
            push_cross(&mut lines, xz_point, 0.005, marker_xz_color);
            push_cross(&mut lines, yw_point, 0.005, marker_yw_color);
        }

        // Breadcrumb trails: XZ uses [0],[2]; YW uses [1],[3]
        if self.hud_breadcrumbs.len() >= 2 {
            let denom = (self.hud_breadcrumbs.len() - 1) as f32;
            for i in 1..self.hud_breadcrumbs.len() {
                let prev = self.hud_breadcrumbs[i - 1];
                let next = self.hud_breadcrumbs[i];
                let t = i as f32 / denom;
                let breadcrumb_xz_color = breadcrumb_xz_start.lerp(breadcrumb_xz_end, t);
                let breadcrumb_yw_color = breadcrumb_yw_start.lerp(breadcrumb_yw_end, t);

                let xz_prev = map_to_panel(
                    xz_center,
                    panel_half * 0.9,
                    map_range,
                    prev[0] - camera_position[0],
                    prev[2] - camera_position[2],
                );
                let xz_next = map_to_panel(
                    xz_center,
                    panel_half * 0.9,
                    map_range,
                    next[0] - camera_position[0],
                    next[2] - camera_position[2],
                );
                push_line(&mut lines, xz_prev, xz_next, breadcrumb_xz_color);

                let yw_prev = map_to_panel(
                    yw_center,
                    panel_half * 0.9,
                    map_range,
                    prev[1] - camera_position[1],
                    prev[3] - camera_position[3],
                );
                let yw_next = map_to_panel(
                    yw_center,
                    panel_half * 0.9,
                    map_range,
                    next[1] - camera_position[1],
                    next[3] - camera_position[3],
                );
                push_line(&mut lines, yw_prev, yw_next, breadcrumb_yw_color);
            }
        }

        // W altimeter and drift gauge to visualize W position and velocity.
        let altimeter_x = left - 0.06;
        let altimeter_min_y = lower_bottom;
        let altimeter_max_y = upper_top;
        let altimeter_range = 12.0;
        let zero_ratio = 0.5;
        // In Vulkan clip space, altimeter_min_y (lower value) is higher on screen.
        // Higher W should be higher on screen (lower clip Y), so invert the mapping.
        let zero_y = altimeter_max_y + (altimeter_min_y - altimeter_max_y) * zero_ratio;
        let w_ratio = ((camera_position[3] / altimeter_range) * 0.5 + 0.5).clamp(0.0, 1.0);
        let w_y = altimeter_max_y + (altimeter_min_y - altimeter_max_y) * w_ratio;

        push_line(
            &mut lines,
            Vec2::new(altimeter_x, altimeter_min_y),
            Vec2::new(altimeter_x, altimeter_max_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - 0.012, altimeter_min_y),
            Vec2::new(altimeter_x + 0.012, altimeter_min_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - 0.012, altimeter_max_y),
            Vec2::new(altimeter_x + 0.012, altimeter_max_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - 0.018, zero_y),
            Vec2::new(altimeter_x + 0.018, zero_y),
            altimeter_color * 0.8,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - 0.02, w_y),
            Vec2::new(altimeter_x + 0.02, w_y),
            altimeter_color,
        );

        let drift_center = Vec2::new(altimeter_x, lower_bottom - 0.08);
        let drift_half_width = 0.065;
        let drift_ratio = (self.hud_w_velocity / 6.0).clamp(-1.0, 1.0);
        let drift_x = drift_center.x + drift_ratio * drift_half_width;

        push_line(
            &mut lines,
            Vec2::new(drift_center.x - drift_half_width, drift_center.y),
            Vec2::new(drift_center.x + drift_half_width, drift_center.y),
            drift_color * 0.7,
        );
        push_line(
            &mut lines,
            Vec2::new(drift_center.x, drift_center.y - 0.01),
            Vec2::new(drift_center.x, drift_center.y + 0.01),
            drift_color * 0.8,
        );
        push_line(
            &mut lines,
            Vec2::new(drift_x, drift_center.y - 0.015),
            Vec2::new(drift_x, drift_center.y + 0.015),
            drift_color,
        );

        let fps = if self.frame_time_ms > 0.1 {
            1000.0 / self.frame_time_ms
        } else {
            0.0
        };
        let mut readout_text = format!(
            "{:.1} ms ({:.0} fps) GPU {:.1} ms [{}]\nPOS {:+.2} {:+.2} {:+.2} {:+.2}\nLOOK {:+.2} {:+.2} {:+.2} {:+.2}",
            self.frame_time_ms,
            fps,
            self.profiler.last_gpu_total_ms,
            self.last_backend.label(),
            camera_position[0],
            camera_position[1],
            camera_position[2],
            camera_position[3],
            look_world[0],
            look_world[1],
            look_world[2],
            look_world[3]
        );
        if self.last_backend == RenderBackend::VoxelTraversal {
            readout_text.push_str(&format!(
                "\nVTE c:{} f:{} e:{} m:{}",
                self.vte_debug_counters.candidate_chunks,
                self.vte_debug_counters.frustum_culled_chunks,
                self.vte_debug_counters.empty_chunks_skipped,
                self.vte_debug_counters.macro_cells_skipped,
            ));
            readout_text.push_str(&format!(
                "\n    cs:{} vs:{} h:{} s:{}",
                self.vte_debug_counters.chunk_steps,
                self.vte_debug_counters.voxel_steps,
                self.vte_debug_counters.primary_hits,
                self.vte_debug_counters.s_samples
            ));
            if self.vte_debug_counters.visible_set_hash_valid {
                readout_text.push_str(&format!(
                    "\n    vh:{:08x}",
                    self.vte_debug_counters.visible_set_hash
                ));
            }
            if self.vte_compare_stats.compared > 0 || self.vte_compare_stats.mismatches > 0 {
                readout_text.push_str(&format!(
                    "\n    cmp:{}/{} mm:{} hs:{} cm:{}",
                    self.vte_compare_stats.matches,
                    self.vte_compare_stats.compared,
                    self.vte_compare_stats.mismatches,
                    self.vte_compare_stats.hit_state_mismatches,
                    self.vte_compare_stats.chunk_material_mismatches
                ));
            }
        }
        if !self.profiler.last_frame_phases.is_empty() {
            readout_text.push('\n');
            for (name, ms) in &self.profiler.last_frame_phases {
                readout_text.push_str(&format!(" {}:{:.1}", name, ms));
            }
        }
        // Position readout below the minimaps on the actual screen.
        // In Vulkan NDC +Y is down. upper_top is the panel's lowest screen edge.
        let readout_anchor_ndc = Vec2::new(left, upper_top + 0.06);

        if let Some(hud_res) = self.hud_resources.as_ref() {
            let panel_bg = Vec4::new(0.0, 0.0, 0.0, 0.45);

            // Semi-transparent backgrounds behind minimap panels
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                xz_min,
                xz_max,
                panel_bg,
            );
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                yw_min,
                yw_max,
                panel_bg,
            );

            // Semi-transparent background behind text readout
            let readout_bg_min = Vec2::new(left, upper_top + 0.02);
            let readout_bg_max = Vec2::new(right, upper_top + 0.30);
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                readout_bg_min,
                readout_bg_max,
                panel_bg,
            );

            let readout_anchor_px = ndc_to_pixels(readout_anchor_ndc, present_size);
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                &readout_text,
                readout_anchor_px,
                readout_text_size,
                map_axis_color,
                present_size,
            );
        } else if let Some(font) = self.hud_font.as_ref() {
            let readout_anchor_px = ndc_to_pixels(readout_anchor_ndc, present_size);
            push_text_lines(
                &mut lines,
                font,
                &readout_text,
                readout_anchor_px,
                readout_text_size,
                map_axis_color,
                present_size,
            );
        }

        // Write line data to GPU buffer
        let lines_to_write = lines.len().min(max_lines - base_line_count);
        if lines_to_write > 0 {
            let mut writer = self.frames_in_flight[frame_idx]
                .line_vertexes_buffer
                .write()
                .unwrap();
            for (line_id, segment) in lines.iter().take(lines_to_write).enumerate() {
                let vertex_id = (base_line_count + line_id) * 2;
                writer[vertex_id] = LineVertex::new(segment.start, segment.color);
                writer[vertex_id + 1] = LineVertex::new(segment.end, segment.color);
            }
        }

        // Write HUD quad data to GPU buffer
        let hud_verts_to_write = hud_quads.len().min(HUD_VERTEX_CAPACITY);
        if hud_verts_to_write > 0 {
            if let Some(hud_buf) = self.frames_in_flight[frame_idx].hud_vertex_buffer.as_ref() {
                let mut writer = hud_buf.write().unwrap();
                for (i, v) in hud_quads.iter().take(hud_verts_to_write).enumerate() {
                    writer[i] = *v;
                }
            }
        }

        (lines_to_write, hud_verts_to_write)
    }

    fn write_custom_overlay_lines(
        &mut self,
        frame_idx: usize,
        base_line_count: usize,
        custom_lines: &[CustomOverlayLine],
    ) -> usize {
        let max_lines = LINE_VERTEX_CAPACITY / 2;
        if base_line_count >= max_lines || custom_lines.is_empty() {
            return 0;
        }

        let lines_to_write = custom_lines.len().min(max_lines - base_line_count);
        let mut writer = self.frames_in_flight[frame_idx]
            .line_vertexes_buffer
            .write()
            .unwrap();
        for (line_id, line) in custom_lines.iter().take(lines_to_write).enumerate() {
            let vertex_id = (base_line_count + line_id) * 2;
            let start = Vec2::from_array(line.start_ndc);
            let end = Vec2::from_array(line.end_ndc);
            let color = Vec4::from_array(line.color);
            writer[vertex_id] = LineVertex::new(start, color);
            writer[vertex_id + 1] = LineVertex::new(end, color);
        }

        lines_to_write
    }

    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    fn wait_for_all_frames(&mut self) {
        for frame in &mut self.frames_in_flight {
            if let Some(future) = frame.fence.take() {
                if self.stall_trace {
                    eprintln!("[stall] wait_for_all_frames: begin");
                }
                let wait_start = Instant::now();
                let f = future.then_signal_fence_and_flush().unwrap();
                f.wait(None).unwrap();
                if self.stall_trace {
                    eprintln!(
                        "[stall] wait_for_all_frames: end ({:.2} ms)",
                        wait_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
            }
        }
    }

    pub fn render(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        view_matrix: ndarray::Array2<f32>,
        focal_length_xy: f32,
        focal_length_zw: f32,
        model_instances: &[common::ModelInstance],
        render_options: RenderOptions,
    ) {
        // Backward-compatible shim while call sites migrate to explicit tetra/voxel contracts.
        self.render_tetra_frame(
            device,
            queue,
            FrameParams {
                view_matrix,
                focal_length_xy,
                focal_length_zw,
                render_options,
            },
            TetraFrameInput { model_instances },
        );
    }

    pub fn render_tetra_frame(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        mut frame_params: FrameParams,
        tetra_input: TetraFrameInput<'_>,
    ) {
        if frame_params.render_options.render_backend == RenderBackend::VoxelTraversal {
            eprintln!(
                "render_tetra_frame called with '{}' backend; forcing '{}'.",
                RenderBackend::VoxelTraversal.label(),
                RenderBackend::TetraRaster.label()
            );
            frame_params.render_options.render_backend = RenderBackend::TetraRaster;
        }
        self.render_internal(
            device,
            queue,
            frame_params,
            tetra_input.model_instances,
            None,
        );
    }

    pub fn render_voxel_frame(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        mut frame_params: FrameParams,
        voxel_input: VoxelFrameInput<'_>,
        tetra_overlay_instances: &[common::ModelInstance],
    ) {
        frame_params.render_options.render_backend = RenderBackend::VoxelTraversal;
        self.render_internal(
            device,
            queue,
            frame_params,
            tetra_overlay_instances,
            Some(&voxel_input),
        );
    }

    fn render_internal(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        frame_params: FrameParams,
        model_instances: &[common::ModelInstance],
        voxel_input: Option<&VoxelFrameInput<'_>>,
    ) {
        let FrameParams {
            view_matrix,
            focal_length_xy,
            focal_length_zw,
            render_options,
        } = frame_params;
        let view_matrix_view = view_matrix.into_owned();

        let slice = view_matrix_view.view().to_slice().unwrap();
        let view_matrix_nalgebra: nalgebra::OMatrix<f32, nalgebra::U5, nalgebra::U5> =
            nalgebra::Matrix5::from_column_slice(slice).transpose();
        let view_matrix_nalgebra_inv = view_matrix_nalgebra.try_inverse().unwrap();

        match self.window.clone() {
            Some(window) => {
                let window_size = window.inner_size();
                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }
            }
            None => {}
        };

        // CPU frame time tracking
        let render_start = Instant::now();
        if let Some(prev_start) = self.last_render_start {
            self.frame_time_ms = (render_start - prev_start).as_secs_f32() * 1000.0;
        }
        self.last_render_start = Some(render_start);

        let frame_idx = self.frames_rendered % FRAMES_IN_FLIGHT;

        // Wait for this frame slot's previous GPU work to complete before writing its buffers.
        // This protects the per-frame LiveBuffers/line vertex buffer from being overwritten
        // while still in use by the GPU.
        if let Some(prev_fence) = self.frames_in_flight[frame_idx].fence.take() {
            if self.stall_trace {
                eprintln!(
                    "[stall] frame={} slot_wait begin (slot={})",
                    self.frames_rendered, frame_idx
                );
            }
            let wait_start = Instant::now();
            let f = prev_fence.then_signal_fence_and_flush().unwrap();
            f.wait(None).unwrap();
            if self.stall_trace {
                eprintln!(
                    "[stall] frame={} slot_wait end ({:.2} ms)",
                    self.frames_rendered,
                    wait_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        if self.frames_in_flight[frame_idx].vte_compare_enabled {
            self.refresh_vte_compare_diagnostics(frame_idx);
        } else {
            self.clear_vte_compare_diagnostics();
        }

        let force_clear = false;

        if let Some(swapchain) = self.swapchain.clone() {
            if let Some(window) = self.window.clone() {
                if let Some(render_pass) = self.render_pass.clone() {
                    let window_size = window.inner_size();
                    // Whenever the window resizes we need to recreate everything dependent on the
                    // window size. In this example that includes the swapchain, the framebuffers and
                    // the dynamic state viewport.
                    if self.recreate_swapchain {
                        // Use the new dimensions of the window.

                        //force_clear = true;

                        let (new_swapchain, new_images) = swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent: window_size.into(),
                                ..swapchain.create_info()
                            })
                            .expect("failed to recreate swapchain");

                        self.swapchain = Some(new_swapchain);

                        // Because framebuffers contains a reference to the old swapchain, we need to
                        // recreate framebuffers as well.
                        self.framebuffers =
                            Some(window_size_dependent_setup(&new_images, &render_pass));

                        self.viewport.extent = window_size.into();

                        self.recreate_swapchain = false;

                        let [capture_w, capture_h] = self
                            .swapchain
                            .as_ref()
                            .map(|s| s.image_extent())
                            .unwrap_or([window_size.width, window_size.height]);
                        let capture_format = self
                            .swapchain
                            .as_ref()
                            .map(|s| s.image_format())
                            .unwrap_or(Format::R8G8B8A8_UNORM);
                        self.cpu_screen_capture_buffer = create_cpu_screencapture_buffer(
                            self.memory_allocator.clone(),
                            capture_w,
                            capture_h,
                            capture_format,
                        );
                    }
                }
            }
        }

        let model_instance_capacity = usize::try_from(
            self.frames_in_flight[frame_idx]
                .live_buffers
                .model_instance_buffer
                .len(),
        )
        .unwrap_or(usize::MAX);
        let used_instance_count = model_instances.len().min(model_instance_capacity);
        let requested_tetrahedron_count =
            self.one_time_buffers.model_tetrahedron_count * used_instance_count;
        let total_tetrahedron_count =
            requested_tetrahedron_count.min(self.sized_buffers.max_tetrahedrons);

        // Debug: print scene info on first frame only
        if self.frames_rendered == 0 {
            if model_instances.len() > used_instance_count {
                eprintln!(
                    "Model instance input truncated to buffer capacity: {} -> {}",
                    model_instances.len(),
                    used_instance_count
                );
            }
            if requested_tetrahedron_count > total_tetrahedron_count {
                eprintln!(
                    "Tetrahedron input truncated to buffer capacity: {} -> {}",
                    requested_tetrahedron_count, total_tetrahedron_count
                );
            }
            println!(
                "Scene: {} tetrahedrons ({} per instance × {} instances)",
                total_tetrahedron_count,
                self.one_time_buffers.model_tetrahedron_count,
                used_instance_count
            );
            println!(
                "BVH: {} internal nodes, {} total nodes",
                total_tetrahedron_count.saturating_sub(1),
                2 * total_tetrahedron_count.saturating_sub(1) + 1
            );
        }
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .working_data_buffer
                .write()
                .unwrap();
            writer.view_matrix = view_matrix_nalgebra.into();
            writer.view_matrix_inverse = view_matrix_nalgebra_inv.into();
            writer.render_dimensions = glam::UVec4::new(
                self.sized_buffers.render_dimensions[0],
                self.sized_buffers.render_dimensions[1],
                self.sized_buffers.render_dimensions[2],
                0,
            );
            writer.present_dimensions = match self.window.clone() {
                None => writer.render_dimensions.xy(),
                Some(window) => {
                    let window_size = window.inner_size();
                    glam::UVec2::new(window_size.width, window_size.height)
                }
            };
            writer.total_num_tetrahedrons = total_tetrahedron_count as u32;
            writer.raytrace_seed = 6364136223846793005u64
                .wrapping_mul(self.frames_rendered as u64)
                .wrapping_add(1442695040888963407);
            writer.focal_length_xy = focal_length_xy;
            writer.focal_length_zw = focal_length_zw;
            // writer.total_num_tetrahedrons = 1;
            writer.shader_fault = 0;
            // Flag used by present shader:
            // 0 = legacy per-layer accumulation, 1 = VTE Stage-B-collapsed output in layer 0.
            // padding[1] carries VTE stage_b_mode so present shader can conditionally
            // bypass tone mapping for debug compare output.
            writer.padding = [
                u32::from(voxel_input.is_some()),
                render_options.vte_display_mode.as_u32(),
            ];
        }

        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .model_instance_buffer
                .write()
                .unwrap();
            for i in 0..used_instance_count {
                writer[i] = model_instances[i];
            }
        }

        let mut vte_chunk_count: usize = 0;
        let mut vte_visible_chunk_count: usize = 0;
        let mut vte_occupancy_word_count: usize = 0;
        let mut vte_material_word_count: usize = 0;
        let mut vte_visible_chunk_min = [i32::MAX; 4];
        let mut vte_visible_chunk_max = [i32::MIN; 4];
        if let Some(input) = voxel_input {
            vte_chunk_count = input.chunk_headers.len().min(VTE_MAX_CHUNKS);
            vte_visible_chunk_count = input
                .visible_chunk_indices
                .len()
                .min(VTE_MAX_CHUNKS)
                .min(vte_chunk_count);
            vte_occupancy_word_count = input
                .occupancy_words
                .len()
                .min(VTE_MAX_CHUNKS * VTE_OCCUPANCY_WORDS_PER_CHUNK);
            vte_material_word_count = input
                .material_words
                .len()
                .min(VTE_MAX_CHUNKS * VTE_MATERIAL_WORDS_PER_CHUNK);

            if self.frames_rendered == 0
                && (input.chunk_headers.len() > vte_chunk_count
                    || input.visible_chunk_indices.len() > vte_visible_chunk_count
                    || input.occupancy_words.len() > vte_occupancy_word_count
                    || input.material_words.len() > vte_material_word_count)
            {
                eprintln!(
                    "VTE input truncated to capacities: chunks {}->{}, visible {}->{}, occupancy {}->{}, materials {}->{}",
                    input.chunk_headers.len(),
                    vte_chunk_count,
                    input.visible_chunk_indices.len(),
                    vte_visible_chunk_count,
                    input.occupancy_words.len(),
                    vte_occupancy_word_count,
                    input.material_words.len(),
                    vte_material_word_count
                );
            }

            {
                let mut writer = self.frames_in_flight[frame_idx]
                    .live_buffers
                    .voxel_chunk_headers_buffer
                    .write()
                    .unwrap();
                for i in 0..vte_chunk_count {
                    writer[i] = input.chunk_headers[i];
                }
            }
            {
                let mut writer = self.frames_in_flight[frame_idx]
                    .live_buffers
                    .voxel_occupancy_words_buffer
                    .write()
                    .unwrap();
                for i in 0..vte_occupancy_word_count {
                    writer[i] = input.occupancy_words[i];
                }
            }
            {
                let mut writer = self.frames_in_flight[frame_idx]
                    .live_buffers
                    .voxel_material_words_buffer
                    .write()
                    .unwrap();
                for i in 0..vte_material_word_count {
                    writer[i] = input.material_words[i];
                }
            }
            {
                let mut writer = self.frames_in_flight[frame_idx]
                    .live_buffers
                    .voxel_visible_chunk_indices_buffer
                    .write()
                    .unwrap();
                for i in 0..vte_visible_chunk_count {
                    let chunk_index = input.visible_chunk_indices[i]
                        .min(vte_chunk_count.saturating_sub(1) as u32);
                    writer[i] = chunk_index;
                    let chunk_coord = input.chunk_headers[chunk_index as usize].chunk_coord;
                    for axis in 0..4 {
                        vte_visible_chunk_min[axis] =
                            vte_visible_chunk_min[axis].min(chunk_coord[axis]);
                        vte_visible_chunk_max[axis] =
                            vte_visible_chunk_max[axis].max(chunk_coord[axis]);
                    }
                }
            }
            {
                debug_assert!(VTE_CHUNK_LOOKUP_CAPACITY.is_power_of_two());
                let mut writer = self.frames_in_flight[frame_idx]
                    .live_buffers
                    .voxel_chunk_lookup_buffer
                    .write()
                    .unwrap();

                for entry in writer.iter_mut() {
                    *entry = GpuVoxelChunkLookupEntry::empty();
                }

                if vte_chunk_count > 0 {
                    let hash_mask = (VTE_CHUNK_LOOKUP_CAPACITY as u32) - 1;
                    for i in 0..vte_visible_chunk_count {
                        let chunk_index = input.visible_chunk_indices[i]
                            .min(vte_chunk_count.saturating_sub(1) as u32);
                        let chunk_coord = input.chunk_headers[chunk_index as usize].chunk_coord;
                        let mut slot = (vte_hash_chunk_coord(chunk_coord) & hash_mask) as usize;

                        for _ in 0..VTE_CHUNK_LOOKUP_CAPACITY {
                            let entry = &mut writer[slot];
                            if entry.chunk_index == GpuVoxelChunkLookupEntry::INVALID_INDEX
                                || entry.chunk_coord == chunk_coord
                            {
                                *entry = GpuVoxelChunkLookupEntry {
                                    chunk_coord,
                                    chunk_index,
                                    _padding: [0; 3],
                                };
                                break;
                            }
                            slot = (slot + 1) & (VTE_CHUNK_LOOKUP_CAPACITY - 1);
                        }
                    }
                }
            }
        }
        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .voxel_frame_meta_buffer
                .write()
                .unwrap();
            let layer_count = self.sized_buffers.render_dimensions[2].max(1);
            let default_slice_layer = (layer_count - 1) / 2;
            let stage_b_slice_layer = render_options
                .vte_slice_layer
                .unwrap_or(default_slice_layer)
                .min(layer_count - 1);
            let mut highlight_flags = 0u32;
            let mut highlight_hit_voxel = [0; 4];
            let mut highlight_place_voxel = [0; 4];
            let highlight_mode_supported = matches!(
                render_options.vte_display_mode,
                VteDisplayMode::Integral | VteDisplayMode::Slice | VteDisplayMode::ThickSlice
            );
            if highlight_mode_supported {
                if let Some(hit_voxel) = render_options.vte_highlight_hit_voxel {
                    highlight_flags |= VTE_HIGHLIGHT_FLAG_HIT_VOXEL;
                    highlight_hit_voxel = hit_voxel;
                }
                if let Some(place_voxel) = render_options.vte_highlight_place_voxel {
                    highlight_flags |= VTE_HIGHLIGHT_FLAG_PLACE_VOXEL;
                    highlight_place_voxel = place_voxel;
                }
            }
            *writer = GpuVoxelFrameMeta {
                chunk_count: vte_chunk_count as u32,
                visible_chunk_count: vte_visible_chunk_count as u32,
                occupancy_word_count: vte_occupancy_word_count as u32,
                material_word_count: vte_material_word_count as u32,
                max_trace_steps: render_options.vte_max_trace_steps.max(1),
                max_trace_distance: render_options.vte_max_trace_distance.max(1.0),
                chunk_lookup_capacity: VTE_CHUNK_LOOKUP_CAPACITY as u32,
                stage_b_mode: render_options.vte_display_mode.as_u32(),
                stage_b_slice_layer,
                stage_b_thick_half_width: render_options.vte_thick_half_width,
                debug_flags: {
                    let mut flags = 0;
                    if render_options.vte_reference_compare {
                        flags |= VTE_DEBUG_FLAG_REFERENCE_COMPARE;
                    }
                    if render_options.vte_reference_mismatch_only {
                        flags |= VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY;
                    }
                    if render_options.vte_compare_slice_only {
                        flags |= VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY;
                    }
                    flags
                },
                visible_chunk_min_x: vte_visible_chunk_min[0],
                visible_chunk_min_y: vte_visible_chunk_min[1],
                visible_chunk_min_z: vte_visible_chunk_min[2],
                visible_chunk_min_w: vte_visible_chunk_min[3],
                visible_chunk_max_x: vte_visible_chunk_max[0],
                visible_chunk_max_y: vte_visible_chunk_max[1],
                visible_chunk_max_z: vte_visible_chunk_max[2],
                visible_chunk_max_w: vte_visible_chunk_max[3],
                highlight_flags,
                _highlight_padding: [0; 3],
                highlight_hit_voxel,
                highlight_place_voxel,
            };
        }

        let mut line_render_count = 0;

        // Do compute stage

        // In order to draw, we have to record a *command buffer*. The command buffer
        // object holds the list of commands that are going to be executed.
        //
        // Recording a command buffer is an expensive operation (usually a few hundred
        // microseconds), but it is known to be a hot path in the driver and is expected to
        // be optimized.
        //
        // Note that we have to pass a queue family when we create the command buffer. The
        // command buffer will only be executable on that given queue family.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (image_index, acquire_future) = match self.swapchain.clone() {
            Some(swapchain) => {
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} acquire_next_image begin",
                        self.frames_rendered
                    );
                }
                let acquire_start = Instant::now();
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            self.recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} acquire_next_image end ({:.2} ms)",
                        self.frames_rendered,
                        acquire_start.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    self.recreate_swapchain = true;
                }

                (Some(image_index), Some(acquire_future))
            }
            None => (None, None),
        };

        let mut do_raster = render_options.do_raster;
        let mut do_raytrace = render_options.do_raytrace;
        let mut do_edges = render_options.do_edges;
        let mut do_tetrahedron_edges = render_options.do_tetrahedron_edges;
        let mut do_voxel_vte = false;
        let mut vte_compare_diagnostics_enabled = false;
        let logical_layers = self.sized_buffers.render_dimensions[2].max(1);
        let storage_layers = self.sized_buffers.pixel_storage_layers.max(1);

        match render_options.render_backend {
            RenderBackend::Auto => {}
            RenderBackend::TetraRaster => {
                do_raster = true;
                do_raytrace = false;
            }
            RenderBackend::TetraRaytrace => {
                do_raster = false;
                do_raytrace = true;
            }
            RenderBackend::VoxelTraversal => {
                // VTE is the primary pass, but optional tetra overlays (e.g. held-item previews)
                // can be rasterized on top in the same frame.
                do_raster = !model_instances.is_empty();
                do_raytrace = false;
                do_edges = false;
                do_tetrahedron_edges = false;
                do_voxel_vte = true;
            }
        }

        if do_voxel_vte {
            vte_compare_diagnostics_enabled = render_options.vte_reference_compare
                && matches!(
                    render_options.vte_display_mode,
                    VteDisplayMode::DebugCompare | VteDisplayMode::DebugIntegral
                );
            if vte_compare_diagnostics_enabled {
                self.reset_vte_compare_buffers(frame_idx);
            } else {
                self.clear_vte_compare_diagnostics();
            }
            let (
                candidate_chunks,
                visible_chunks,
                empty_chunks,
                full_chunks,
                occupancy_words,
                material_words,
                visible_set_hash_valid,
                visible_set_hash,
            ) = if let Some(input) = voxel_input {
                let empty = input
                    .chunk_headers
                    .iter()
                    .filter(|h| (h.flags & GpuVoxelChunkHeader::FLAG_EMPTY) != 0)
                    .count() as u32;
                let full = input
                    .chunk_headers
                    .iter()
                    .filter(|h| (h.flags & GpuVoxelChunkHeader::FLAG_FULL) != 0)
                    .count() as u32;
                let collect_visible_set_hash = render_options.vte_reference_compare;
                let hash = if collect_visible_set_hash {
                    let mut hash = 0x811C_9DC5u32;
                    for &chunk_index in input.visible_chunk_indices.iter() {
                        let Some(header) = input.chunk_headers.get(chunk_index as usize) else {
                            continue;
                        };
                        let h = vte_hash_chunk_coord(header.chunk_coord);
                        hash ^= h;
                        hash = hash.wrapping_mul(0x0100_0193);
                    }
                    hash
                } else {
                    0
                };
                (
                    input.chunk_headers.len() as u32,
                    input.visible_chunk_indices.len() as u32,
                    empty,
                    full,
                    input.occupancy_words.len() as u64,
                    input.material_words.len() as u64,
                    collect_visible_set_hash,
                    hash,
                )
            } else {
                (0, 0, 0, 0, 0, 0, false, 0)
            };

            self.vte_debug_counters = VteDebugCounters {
                candidate_chunks,
                frustum_culled_chunks: candidate_chunks.saturating_sub(visible_chunks),
                empty_chunks_skipped: empty_chunks,
                macro_cells_skipped: 0,
                chunk_steps: 0,
                voxel_steps: 0,
                primary_hits: full_chunks as u64,
                s_samples: self.sized_buffers.render_dimensions[0] as u64
                    * self.sized_buffers.render_dimensions[1] as u64
                    * self.sized_buffers.render_dimensions[2] as u64,
                visible_set_hash_valid,
                visible_set_hash,
            };
            if !self.vte_backend_notice_printed {
                println!(
                    "Render backend '{}' selected: VTE active (chunks={}, visible={}, occupancy_words={}, material_words={}, max_trace_steps={}, max_trace_distance={:.1}, stage_b={}, slice_layer={:?}, thick_half_width={}, reference_compare={}, mismatch_only={}, compare_slice_only={}, storage_layers={}/{}).",
                    RenderBackend::VoxelTraversal.label(),
                    candidate_chunks,
                    visible_chunks,
                    occupancy_words,
                    material_words,
                    render_options.vte_max_trace_steps.max(1),
                    render_options.vte_max_trace_distance.max(1.0),
                    render_options.vte_display_mode.label(),
                    render_options.vte_slice_layer,
                    render_options.vte_thick_half_width,
                    render_options.vte_reference_compare,
                    render_options.vte_reference_mismatch_only,
                    render_options.vte_compare_slice_only,
                    storage_layers,
                    logical_layers,
                );
                self.vte_backend_notice_printed = true;
            }
        } else {
            self.clear_vte_compare_diagnostics();
        }

        let reduced_storage_supported =
            do_voxel_vte && render_options.vte_display_mode == VteDisplayMode::Integral;
        if storage_layers < logical_layers && !reduced_storage_supported {
            panic!(
                "pixel storage layers ({storage_layers}) are less than logical render layers ({logical_layers}); \
this reduced-storage configuration currently supports only '--backend voxel-traversal --vte-display-mode integral'."
            );
        }

        self.frames_in_flight[frame_idx].vte_compare_enabled =
            do_voxel_vte && vte_compare_diagnostics_enabled;

        self.last_backend = if do_voxel_vte {
            RenderBackend::VoxelTraversal
        } else if do_raytrace {
            RenderBackend::TetraRaytrace
        } else if do_raster || do_tetrahedron_edges || do_edges {
            RenderBackend::TetraRaster
        } else {
            RenderBackend::Auto
        };

        let cpu_mode = false;

        if cpu_mode {
            self.wait_for_all_frames();
        }

        if cpu_mode {
            if do_raster || do_tetrahedron_edges {
                // Tetrahedron pre-raster
                unimplemented!();
            }

            if do_raster {
                unimplemented!();
            }

            if do_edges {
                unimplemented!();
            }

            if do_raytrace {
                // CPU shader fallback not available with Slang shaders
                // (Slang compiles to SPIR-V only, not CPU-executable code)
                unimplemented!("CPU raytracing not available - use GPU mode");
            }
            if do_voxel_vte {
                unimplemented!("CPU voxel traversal backend not implemented");
            }
        } else {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.compute_pipeline.pipeline_layout.clone(),
                    0,
                    vec![
                        self.one_time_buffers.descriptor_set.clone(),
                        self.frames_in_flight[frame_idx]
                            .sized_descriptor_set
                            .clone(),
                        self.frames_in_flight[frame_idx]
                            .live_buffers
                            .descriptor_set
                            .clone(),
                    ],
                )
                .unwrap();

            // Set default push constants (required by pipeline layout even for shaders that don't use them)
            let dummy_push_data: [u32; 4] = [0, 0, 0, 0];
            builder
                .push_constants(
                    self.compute_pipeline.pipeline_layout.clone(),
                    0,
                    dummy_push_data,
                )
                .unwrap();

            // GPU profiling: reset query pool and write start timestamp
            self.profiler.begin_frame();
            unsafe {
                builder.reset_query_pool(
                    self.frames_in_flight[frame_idx].query_pool.clone(),
                    0..PROFILER_MAX_TIMESTAMPS,
                )
            }
            .unwrap();
            {
                let q = self.profiler.next_query_index("start");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            if do_voxel_vte {
                let fuse_integral_in_stage_a =
                    render_options.vte_display_mode == VteDisplayMode::Integral;
                builder
                    .bind_pipeline_compute(
                        self.compute_pipeline.voxel_trace_stage_a_pipeline.clone(),
                    )
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        (self.sized_buffers.render_dimensions[0] + 7) / 8,
                        (self.sized_buffers.render_dimensions[1] + 7) / 8,
                        if fuse_integral_in_stage_a {
                            1
                        } else {
                            self.sized_buffers.render_dimensions[2].max(1)
                        },
                    ])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("vte_stage_a");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                if !fuse_integral_in_stage_a {
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.voxel_display_stage_b_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([
                            (self.sized_buffers.render_dimensions[0] + 7) / 8,
                            (self.sized_buffers.render_dimensions[1] + 7) / 8,
                            1,
                        ])
                    }
                    .unwrap();
                }
                {
                    let q = self.profiler.next_query_index("vte_stage_b");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
            } else if (render_options.do_frame_clear || force_clear) && !do_raytrace {
                builder
                    .bind_pipeline_compute(self.compute_pipeline.raytrace_clear_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        self.sized_buffers.render_dimensions[0] / 8,
                        self.sized_buffers.render_dimensions[1] / 8,
                        1,
                    ])
                }
                .unwrap();
                let q = self.profiler.next_query_index("clear");
                unsafe {
                    builder.write_timestamp(
                        self.frames_in_flight[frame_idx].query_pool.clone(),
                        q,
                        PipelineStage::AllCommands,
                    )
                }
                .unwrap();
            }

            if do_raster || do_tetrahedron_edges {
                // Tetrahedron pre-raster
                // Reset atomic counter to 0 before clipping dispatch
                builder
                    .fill_buffer(self.sized_buffers.atomic_counter_buffer.clone(), 0u32)
                    .unwrap();

                builder
                    .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pipeline.clone())
                    .unwrap();
                unsafe { builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1]) }
                    .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_clip");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                // Copy atomic counter for CPU readback (clipped tet count diagnostic)
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.atomic_counter_buffer.clone(),
                        self.frames_in_flight[frame_idx]
                            .cpu_clipped_tet_count_buffer
                            .clone(),
                    ))
                    .unwrap();

                if do_tetrahedron_edges {
                    line_render_count = total_tetrahedron_count * 6;
                }
            }

            if do_raster {
                // Zero tile counts
                builder
                    .fill_buffer(self.sized_buffers.tile_tet_counts_buffer.clone(), 0u32)
                    .unwrap();

                // Bin tetrahedra into tiles
                builder
                    .bind_pipeline_compute(self.compute_pipeline.bin_tets_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([(self.sized_buffers.max_tetrahedrons as u32 + 63) / 64, 1, 1])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_bin");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                // Tetrahedron pixel raster (tile-based)
                builder
                    .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pixel_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        (self.sized_buffers.render_dimensions[0] + 7) / 8,
                        (self.sized_buffers.render_dimensions[1] + 7) / 8,
                        1,
                    ])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_raster");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
            }

            if do_edges {
                // Tetrahedron edge pre-raster
                builder
                    .bind_pipeline_compute(self.compute_pipeline.edge_pipeline.clone())
                    .unwrap();
                let total_edge_count =
                    self.one_time_buffers.model_edge_count * model_instances.len();
                unsafe { builder.dispatch([(total_edge_count as u32 + 63) / 64u32, 1, 1]) }
                    .unwrap();
                {
                    let q = self.profiler.next_query_index("edges");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                line_render_count = total_edge_count;
            }

            if do_raytrace {
                // Compute a hash of the scene inputs that affect the BVH.
                // If unchanged, skip tetrahedron preprocessing and BVH construction.
                let scene_hash = {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    for &val in view_matrix_nalgebra.as_slice() {
                        val.to_bits().hash(&mut hasher);
                    }
                    focal_length_xy.to_bits().hash(&mut hasher);
                    focal_length_zw.to_bits().hash(&mut hasher);
                    model_instances.len().hash(&mut hasher);
                    bytemuck::cast_slice::<_, u8>(model_instances).hash(&mut hasher);
                    hasher.finish()
                };
                let bvh_needs_rebuild = scene_hash != self.bvh_scene_hash;
                let raytrace_should_clear =
                    render_options.do_frame_clear || force_clear || bvh_needs_rebuild;

                if raytrace_should_clear {
                    builder
                        .bind_pipeline_compute(
                            self.compute_pipeline.raytrace_clear_pipeline.clone(),
                        )
                        .unwrap();
                    unsafe {
                        builder.dispatch([
                            self.sized_buffers.render_dimensions[0] / 8,
                            self.sized_buffers.render_dimensions[1] / 8,
                            1,
                        ])
                    }
                    .unwrap();
                    let q = self.profiler.next_query_index("clear");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                if bvh_needs_rebuild {
                    // 1. Tetrahedron preprocessing (transform to view space)
                    builder
                        .bind_pipeline_compute(self.compute_pipeline.raytrace_pre_pipeline.clone())
                        .unwrap();
                    unsafe {
                        builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1])
                    }
                    .unwrap();
                    {
                        let q = self.profiler.next_query_index("rt_preprocess");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // 2. BVH Construction
                    if total_tetrahedron_count > 0 {
                        // 2a. Compute scene bounds
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_scene_bounds_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

                        // 2b. Compute Morton codes (dispatch n_pow2 threads to fill sentinels for padding)
                        let n = total_tetrahedron_count as u32;
                        let n_pow2 = n.next_power_of_two();
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_morton_codes_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe { builder.dispatch([(n_pow2 + 63) / 64u32, 1, 1]) }.unwrap();

                        // 2c. Bitonic sort using shared memory optimization
                        // Sort all n_pow2 elements (including sentinel-padded entries)
                        let num_stages = n_pow2.trailing_zeros(); // log2(n_pow2)
                        let local_stages = 6u32.min(num_stages); // stages 0-5 fit in 64-element workgroups
                        let workgroups = (n_pow2 + 63) / 64;

                        // Phase 1: Sort each 64-element block in shared memory (stages 0-5)
                        let push_data: [u32; 4] = [0, 0, n_pow2, 0];
                        builder
                            .push_constants(
                                self.compute_pipeline.pipeline_layout.clone(),
                                0,
                                push_data,
                            )
                            .unwrap();
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline
                                    .bvh_bitonic_sort_local_pipeline
                                    .clone(),
                            )
                            .unwrap();
                        unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();

                        // Phase 2: Global merge stages (stages 6+)
                        for stage in local_stages..num_stages {
                            // Global steps: stepSize >= 64 (step >= 6)
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_bitonic_sort_pipeline.clone(),
                                )
                                .unwrap();
                            for step in (local_stages..=stage).rev() {
                                let push_data: [u32; 4] = [stage, step, n_pow2, 0];
                                builder
                                    .push_constants(
                                        self.compute_pipeline.pipeline_layout.clone(),
                                        0,
                                        push_data,
                                    )
                                    .unwrap();
                                unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                            }
                            // Local merge: steps 5-0 in shared memory (1 dispatch)
                            let push_data: [u32; 4] = [stage, 0, n_pow2, 0];
                            builder
                                .push_constants(
                                    self.compute_pipeline.pipeline_layout.clone(),
                                    0,
                                    push_data,
                                )
                                .unwrap();
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline
                                        .bvh_bitonic_sort_local_merge_pipeline
                                        .clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([workgroups, 1, 1]) }.unwrap();
                        }

                        // 2d. Initialize leaf nodes
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_init_leaves_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1])
                        }
                        .unwrap();

                        // 2e. Build internal nodes (Karras algorithm)
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_build_tree_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1])
                        }
                        .unwrap();

                        // 2f. Compute leaf AABBs
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline
                                    .bvh_compute_leaf_aabbs_pipeline
                                    .clone(),
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1])
                        }
                        .unwrap();

                        // 2g. Propagate AABBs from leaves to root (multi-pass)
                        let num_internal_nodes = total_tetrahedron_count.saturating_sub(1) as u32;
                        if num_internal_nodes > 0 {
                            let num_passes = 2 * (32 - num_internal_nodes.leading_zeros()).max(1);
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                                )
                                .unwrap();
                            for _ in 0..num_passes {
                                unsafe { builder.dispatch([(num_internal_nodes + 63) / 64, 1, 1]) }
                                    .unwrap();
                            }
                        }
                    }

                    self.bvh_scene_hash = scene_hash;

                    // Debug: copy BVH data to CPU on first rebuild
                    if self.frames_rendered == 0 {
                        builder
                            .copy_buffer(CopyBufferInfo::buffers(
                                self.sized_buffers.bvh_nodes_buffer.clone(),
                                self.sized_buffers.cpu_bvh_nodes_buffer.clone(),
                            ))
                            .unwrap();
                        builder
                            .copy_buffer(CopyBufferInfo::buffers(
                                self.sized_buffers.morton_codes_buffer.clone(),
                                self.sized_buffers.cpu_morton_codes_buffer.clone(),
                            ))
                            .unwrap();
                    }

                    {
                        let q = self.profiler.next_query_index("bvh_build");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }
                }

                // 3. Raytrace pixels (using BVH) - always runs, only seed changes
                builder
                    .bind_pipeline_compute(self.compute_pipeline.raytrace_pixel_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        (self.sized_buffers.render_dimensions[0] + 7) / 8,
                        (self.sized_buffers.render_dimensions[1] + 7) / 8,
                        1,
                    ])
                }
                .unwrap();
                {
                    let q = self.profiler.next_query_index("raytrace");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
            }
        }

        let mut hud_vertex_count = 0usize;
        line_render_count += self.write_custom_overlay_lines(
            frame_idx,
            line_render_count,
            &render_options.custom_overlay_lines,
        );
        if render_options.do_navigation_hud {
            let (hud_line_count, hud_quad_count) = self.write_navigation_hud_overlay(
                frame_idx,
                line_render_count,
                &view_matrix_nalgebra,
                &view_matrix_nalgebra_inv,
                focal_length_xy,
                model_instances,
                render_options.hud_rotation_label.as_deref(),
            );
            line_render_count += hud_line_count;
            hud_vertex_count = hud_quad_count;
        }

        if let Some(image_index) = image_index {
            if let Some(framebuffers) = self.framebuffers.clone() {
                if let Some(present_pipeline) = self.present_pipeline.as_ref() {
                    // Begin render pass
                    builder
                        // Before we can draw, we have to *enter a render pass*.
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                // A list of values to clear the attachments with. This list contains
                                // one item for each attachment in the render pass. In this case, there
                                // is only one attachment, and we clear it with a blue color.
                                //
                                // Only attachments that have `AttachmentLoadOp::Clear` are provided
                                // with clear values, any others should use `None` as the clear value.
                                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],

                                ..RenderPassBeginInfo::framebuffer(
                                    framebuffers[image_index as usize].clone(),
                                )
                            },
                            SubpassBeginInfo {
                                // The contents of the first (and only) subpass. This can be either
                                // `Inline` or `SecondaryCommandBuffers`. The latter is a bit more
                                // advanced and is not covered here.
                                contents: SubpassContents::Inline,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    builder
                        .set_viewport(0, [self.viewport.clone()].into_iter().collect())
                        .unwrap();
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            present_pipeline.pipeline_layout.clone(),
                            0,
                            vec![
                                self.one_time_buffers.descriptor_set.clone(),
                                self.frames_in_flight[frame_idx]
                                    .sized_descriptor_set
                                    .clone(),
                                self.frames_in_flight[frame_idx]
                                    .live_buffers
                                    .descriptor_set
                                    .clone(),
                            ],
                        )
                        .unwrap();

                    // Render from compute shader buffer
                    {
                        builder
                            .bind_pipeline_graphics(present_pipeline.buffer_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(6, 1, 0, 0) }.unwrap();
                    }

                    // Render the edge lines
                    if line_render_count > 0 {
                        builder
                            .bind_pipeline_graphics(present_pipeline.line_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(line_render_count as u32 * 2, 1, 0, 0) }.unwrap();
                    }

                    // Render HUD quads (text + panels, alpha-blended on top)
                    if hud_vertex_count > 0 {
                        if let Some(hud_ds) =
                            self.frames_in_flight[frame_idx].hud_descriptor_set.as_ref()
                        {
                            builder
                                .bind_pipeline_graphics(present_pipeline.hud_pipeline.clone())
                                .unwrap();
                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    present_pipeline.hud_pipeline_layout.clone(),
                                    0,
                                    vec![hud_ds.clone()],
                                )
                                .unwrap();
                            unsafe { builder.draw(hud_vertex_count as u32, 1, 0, 0) }.unwrap();
                        }
                    }

                    // End render pass
                    builder
                        // We leave the render pass. Note that if we had multiple subpasses we could
                        // have called `next_subpass` to jump to the next subpass.
                        .end_render_pass(Default::default())
                        .unwrap();
                    if render_options.take_framebuffer_screenshot {
                        builder
                            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                                framebuffers[image_index as usize].attachments()[0]
                                    .image()
                                    .clone(),
                                self.cpu_screen_capture_buffer.clone(),
                            ))
                            .unwrap();
                    }
                }
            }
        }

        if render_options.prepare_render_screenshot {
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.sized_buffers.output_pixel_buffer.clone(),
                    self.sized_buffers.output_cpu_pixel_buffer.clone(),
                ))
                .unwrap();
        }

        // Final frame marker so profiling includes graphics/present-tail work
        // after the last compute phase timestamp.
        {
            let q = self.profiler.next_query_index("frame_end");
            unsafe {
                builder.write_timestamp(
                    self.frames_in_flight[frame_idx].query_pool.clone(),
                    q,
                    PipelineStage::AllCommands,
                )
            }
            .unwrap();
        }

        // Finish recording the command buffer by calling `end`.
        let command_buffer = builder.build().unwrap();

        // Wait for the most recently submitted frame to complete before submitting.
        // This protects shared SizedBuffers (pixel buffer, BVH, tetrahedra) and
        // ensures GPU ordering is maintained.
        if self.frames_rendered > 0 {
            let prev_idx = (self.frames_rendered - 1) % FRAMES_IN_FLIGHT;
            if let Some(prev_fence) = self.frames_in_flight[prev_idx].fence.take() {
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} prev_submit_wait begin (slot={})",
                        self.frames_rendered, prev_idx
                    );
                }
                let wait_start = Instant::now();
                let f = prev_fence.then_signal_fence_and_flush().unwrap();
                f.wait(None).unwrap();
                if self.stall_trace {
                    eprintln!(
                        "[stall] frame={} prev_submit_wait end ({:.2} ms)",
                        self.frames_rendered,
                        wait_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                // Read back clipped tetrahedron count for diagnostics
                let clipped_count = self.frames_in_flight[prev_idx]
                    .cpu_clipped_tet_count_buffer
                    .read()
                    .map(|data| data[0])
                    .unwrap_or(0);
                self.last_clipped_tet_count = clipped_count;
                self.profiler.read_results_and_accumulate(
                    &self.frames_in_flight[prev_idx].query_pool,
                    clipped_count,
                );
            }
        }

        // Submit the command buffer
        let base_future = sync::now(device.clone());
        match acquire_future {
            Some(acquire_future) => {
                let future = base_future
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            self.swapchain.clone().unwrap().clone(),
                            image_index.unwrap(),
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.frames_in_flight[frame_idx].fence = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
            None => {
                let future = base_future
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.frames_in_flight[frame_idx].fence = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                    }
                }
            }
        };

        if let Some(window) = self.window.clone() {
            let window_size = window.inner_size();
            // Save frame
            if self.frames_rendered > 3 && render_options.take_framebuffer_screenshot {
                self.wait_for_all_frames();

                let result = self.cpu_screen_capture_buffer.read();
                match result {
                    Ok(buffer_content) => {
                        let screenshot_index = self.frames_rendered - 3;
                        let screenshot_webp_path =
                            format!("frames/framebuffer_{}.webp", screenshot_index);
                        let screenshot_png_path =
                            format!("frames/framebuffer_{}.png", screenshot_index);
                        let (capture_w, capture_h, capture_format) = self
                            .swapchain
                            .as_ref()
                            .map(|swapchain| {
                                let [w, h] = swapchain.image_extent();
                                (w, h, swapchain.image_format())
                            })
                            .unwrap_or((
                                window_size.width,
                                window_size.height,
                                Format::R8G8B8A8_UNORM,
                            ));
                        let expected_bytes = (capture_w as usize)
                            .saturating_mul(capture_h as usize)
                            .saturating_mul(4);

                        let rgba_bytes = match capture_format {
                            Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB => {
                                if buffer_content.len() < expected_bytes {
                                    eprintln!(
                                        "Framebuffer screenshot buffer too small: have {}, need {}",
                                        buffer_content.len(),
                                        expected_bytes
                                    );
                                    None
                                } else {
                                    Some(buffer_content[..expected_bytes].to_vec())
                                }
                            }
                            Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => {
                                if buffer_content.len() < expected_bytes {
                                    eprintln!(
                                        "Framebuffer screenshot buffer too small: have {}, need {}",
                                        buffer_content.len(),
                                        expected_bytes
                                    );
                                    None
                                } else {
                                    let mut bytes = vec![0u8; expected_bytes];
                                    for (src, dst) in buffer_content[..expected_bytes]
                                        .chunks_exact(4)
                                        .zip(bytes.chunks_exact_mut(4))
                                    {
                                        dst[0] = src[2];
                                        dst[1] = src[1];
                                        dst[2] = src[0];
                                        dst[3] = src[3];
                                    }
                                    Some(bytes)
                                }
                            }
                            _ => {
                                eprintln!(
                                    "Framebuffer screenshot not supported for swapchain format {:?}",
                                    capture_format
                                );
                                None
                            }
                        };

                        if let Some(bytes) = rgba_bytes {
                            if let Some(image) =
                                ImageBuffer::<Rgba<u8>, _>::from_raw(capture_w, capture_h, bytes)
                            {
                                if let Err(err) = image.save(screenshot_webp_path.clone()) {
                                    eprintln!(
                                        "Failed to save screenshot to {}: {}",
                                        screenshot_webp_path, err
                                    );
                                } else {
                                    println!("Saved screenshot to {}", screenshot_webp_path);
                                }
                                if let Err(err) = image.save(screenshot_png_path.clone()) {
                                    eprintln!(
                                        "Failed to save screenshot to {}: {}",
                                        screenshot_png_path, err
                                    );
                                } else {
                                    println!("Saved screenshot to {}", screenshot_png_path);
                                }

                                let camera_h = mat5_mul_vec5(
                                    &view_matrix_nalgebra_inv,
                                    [0.0, 0.0, 0.0, 0.0, 1.0],
                                );
                                let inv_w = if camera_h[4].abs() > 1e-6 {
                                    1.0 / camera_h[4]
                                } else {
                                    1.0
                                };
                                let camera_pos = [
                                    camera_h[0] * inv_w,
                                    camera_h[1] * inv_w,
                                    camera_h[2] * inv_w,
                                    camera_h[3] * inv_w,
                                ];
                                let look_h = mat5_mul_vec5(
                                    &view_matrix_nalgebra_inv,
                                    [
                                        0.0,
                                        0.0,
                                        std::f32::consts::FRAC_1_SQRT_2,
                                        std::f32::consts::FRAC_1_SQRT_2,
                                        0.0,
                                    ],
                                );
                                let mut look = [look_h[0], look_h[1], look_h[2], look_h[3]];
                                let look_len = (look[0] * look[0]
                                    + look[1] * look[1]
                                    + look[2] * look[2]
                                    + look[3] * look[3])
                                    .sqrt();
                                if look_len > 1e-6 {
                                    for c in &mut look {
                                        *c /= look_len;
                                    }
                                }
                                println!(
                                    "Screenshot meta frame={} backend={} size={}x{} layers={} focal_xy={:.3} focal_zw={:.3}",
                                    screenshot_index,
                                    self.last_backend.label(),
                                    capture_w,
                                    capture_h,
                                    self.sized_buffers.render_dimensions[2],
                                    focal_length_xy,
                                    focal_length_zw
                                );
                                if self.last_backend == RenderBackend::VoxelTraversal {
                                    println!(
                                        "  VTE mode={} slice_layer={:?} thick_half_width={} max_steps={} max_distance={:.1} reference_compare={} mismatch_only={} compare_slice_only={}",
                                        render_options.vte_display_mode.label(),
                                        render_options.vte_slice_layer,
                                        render_options.vte_thick_half_width,
                                        render_options.vte_max_trace_steps,
                                        render_options.vte_max_trace_distance,
                                        render_options.vte_reference_compare,
                                        render_options.vte_reference_mismatch_only,
                                        render_options.vte_compare_slice_only,
                                    );
                                    if self.vte_compare_stats.compared > 0
                                        || self.vte_compare_stats.mismatches > 0
                                    {
                                        println!(
                                            "  VTE compare compared={} match={} mismatch={} hit_state={} chunk_material={} fm_ref={} fh_ref={} reason=[none:{} touched:{} voxel:{} chunk:{} dist:{} lookup:{}] flags=[zero:{} tie:{} fallback:{}]",
                                            self.vte_compare_stats.compared,
                                            self.vte_compare_stats.matches,
                                            self.vte_compare_stats.mismatches,
                                            self.vte_compare_stats.hit_state_mismatches,
                                            self.vte_compare_stats.chunk_material_mismatches,
                                            self.vte_compare_stats.fast_miss_ref_hit,
                                            self.vte_compare_stats.fast_hit_ref_miss,
                                            self.vte_compare_stats.miss_reason_counts[0],
                                            self.vte_compare_stats.miss_reason_counts[1],
                                            self.vte_compare_stats.miss_reason_counts[2],
                                            self.vte_compare_stats.miss_reason_counts[3],
                                            self.vte_compare_stats.miss_reason_counts[4],
                                            self.vte_compare_stats.miss_reason_counts[5],
                                            self.vte_compare_stats.zero_interval_flags,
                                            self.vte_compare_stats.tie_stepped_flags,
                                            self.vte_compare_stats.lookup_fallback_flags,
                                        );
                                        if self.vte_first_mismatch.valid {
                                            println!(
                                                "  VTE first_mismatch px=({}, {}, l={}) kind={} miss_reason={} debug=0x{:x} hit=({}/{}) fast_chunk=({},{},{},{}) ref_chunk=({},{},{},{}) mat=({}/{}) t=({:.5}/{:.5}) steps={} rem_vox={} final_t={:.5} last_chunk=({},{},{},{})",
                                                self.vte_first_mismatch.pixel_x,
                                                self.vte_first_mismatch.pixel_y,
                                                self.vte_first_mismatch.layer,
                                                self.vte_first_mismatch.mismatch_kind,
                                                self.vte_first_mismatch.miss_reason,
                                                self.vte_first_mismatch.debug_flags,
                                                self.vte_first_mismatch.fast_hit as u32,
                                                self.vte_first_mismatch.ref_hit as u32,
                                                self.vte_first_mismatch.fast_chunk[0],
                                                self.vte_first_mismatch.fast_chunk[1],
                                                self.vte_first_mismatch.fast_chunk[2],
                                                self.vte_first_mismatch.fast_chunk[3],
                                                self.vte_first_mismatch.ref_chunk[0],
                                                self.vte_first_mismatch.ref_chunk[1],
                                                self.vte_first_mismatch.ref_chunk[2],
                                                self.vte_first_mismatch.ref_chunk[3],
                                                self.vte_first_mismatch.fast_material,
                                                self.vte_first_mismatch.ref_material,
                                                self.vte_first_mismatch.fast_hit_t,
                                                self.vte_first_mismatch.ref_hit_t,
                                                self.vte_first_mismatch.chunk_steps_taken,
                                                self.vte_first_mismatch.remaining_voxel_steps,
                                                self.vte_first_mismatch.final_t,
                                                self.vte_first_mismatch.last_chunk[0],
                                                self.vte_first_mismatch.last_chunk[1],
                                                self.vte_first_mismatch.last_chunk[2],
                                                self.vte_first_mismatch.last_chunk[3],
                                            );
                                        }
                                    }
                                }
                                println!(
                                    "  POS {:+.4} {:+.4} {:+.4} {:+.4}",
                                    camera_pos[0], camera_pos[1], camera_pos[2], camera_pos[3]
                                );
                                println!(
                                    "  LOOK {:+.4} {:+.4} {:+.4} {:+.4}",
                                    look[0], look[1], look[2], look[3]
                                );
                            } else {
                                eprintln!(
                                    "Failed to build screenshot image buffer ({}x{})",
                                    capture_w, capture_h
                                );
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("Error saving screenshot: {:?}", error);
                    }
                };
            }
        }

        // Debug: print BVH diagnostics on first frame
        if self.frames_rendered == 0 && do_raytrace {
            // Ensure GPU work is done
            self.wait_for_all_frames();

            let bvh_nodes = self.sized_buffers.cpu_bvh_nodes_buffer.read().unwrap();
            let morton_codes = self.sized_buffers.cpu_morton_codes_buffer.read().unwrap();
            let num_leaves = total_tetrahedron_count;
            let num_internal = num_leaves.saturating_sub(1);
            let total_nodes = num_leaves.saturating_mul(2).saturating_sub(1);

            // Check Morton code sorting
            let mut sorted = true;
            for i in 1..num_leaves {
                if morton_codes[i].code < morton_codes[i - 1].code {
                    println!(
                        "  SORT ERROR at {}: code[{}]={} > code[{}]={}",
                        i,
                        i - 1,
                        morton_codes[i - 1].code,
                        i,
                        morton_codes[i].code
                    );
                    sorted = false;
                }
            }
            println!("Morton codes sorted: {}", sorted);

            // Check root node
            let root = &bvh_nodes[0];
            println!(
                "Root node: left={}, right={}, isLeaf={}, visitCount={}",
                root.left_child, root.right_child, root.is_leaf, root.atomic_visit_count
            );
            println!(
                "Root AABB: min=({:.2},{:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2},{:.2})",
                root.min_bounds.x,
                root.min_bounds.y,
                root.min_bounds.z,
                root.min_bounds.w,
                root.max_bounds.x,
                root.max_bounds.y,
                root.max_bounds.z,
                root.max_bounds.w
            );

            // Count valid internal nodes
            let mut valid_internal = 0;
            let mut invalid_children = 0;
            let mut zero_aabb_internal = 0;
            for i in 0..num_internal {
                if bvh_nodes[i].atomic_visit_count >= 2 {
                    valid_internal += 1;
                }
                if bvh_nodes[i].left_child >= total_nodes as u32
                    || bvh_nodes[i].right_child >= total_nodes as u32
                {
                    if bvh_nodes[i].left_child != 0xFFFFFFFF
                        && bvh_nodes[i].right_child != 0xFFFFFFFF
                    {
                        invalid_children += 1;
                    }
                }
                let aabb_size = (bvh_nodes[i].max_bounds - bvh_nodes[i].min_bounds).length();
                if aabb_size < 0.001 && bvh_nodes[i].atomic_visit_count >= 2 {
                    zero_aabb_internal += 1;
                }
            }
            println!(
                "Internal nodes: {}/{} valid (visitCount>=2), {} invalid children, {} zero-AABB",
                valid_internal, num_internal, invalid_children, zero_aabb_internal
            );

            // Count valid leaves
            let mut zero_aabb_leaves = 0;
            for i in 0..num_leaves {
                let leaf_idx = num_internal + i;
                let aabb_size =
                    (bvh_nodes[leaf_idx].max_bounds - bvh_nodes[leaf_idx].min_bounds).length();
                if aabb_size < 0.001 {
                    zero_aabb_leaves += 1;
                }
            }
            println!("Leaves: {}/{} with zero AABB", zero_aabb_leaves, num_leaves);

            // Print first few internal nodes for inspection
            println!("First 5 internal nodes:");
            for i in 0..5.min(num_internal) {
                let n = &bvh_nodes[i];
                let aabb_size = (n.max_bounds - n.min_bounds).length();
                println!("  [{}] L={} R={} leaf={} visit={} aabb_size={:.2} min=({:.1},{:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1},{:.1})",
                         i, n.left_child, n.right_child, n.is_leaf, n.atomic_visit_count,
                         aabb_size,
                         n.min_bounds.x, n.min_bounds.y, n.min_bounds.z, n.min_bounds.w,
                         n.max_bounds.x, n.max_bounds.y, n.max_bounds.z, n.max_bounds.w);
            }

            // Print a few leaves
            println!("First 5 leaf nodes:");
            for i in 0..5.min(num_leaves) {
                let leaf_idx = num_internal + i;
                let n = &bvh_nodes[leaf_idx];
                let aabb_size = (n.max_bounds - n.min_bounds).length();
                println!("  [{}] tetIdx={} leaf={} aabb_size={:.2} min=({:.1},{:.1},{:.1},{:.1}) max=({:.1},{:.1},{:.1},{:.1})",
                         leaf_idx, n.tetrahedron_index, n.is_leaf, aabb_size,
                         n.min_bounds.x, n.min_bounds.y, n.min_bounds.z, n.min_bounds.w,
                         n.max_bounds.x, n.max_bounds.y, n.max_bounds.z, n.max_bounds.w);
            }
        }

        self.frames_rendered += 1;
    }

    pub fn save_rendered_frame(&mut self, path: &str) {
        self.wait_for_all_frames();

        let result = self.sized_buffers.output_cpu_pixel_buffer.read();
        match result {
            Ok(buffer_content) => {
                let mut buffer_stored = Vec::new();
                buffer_stored.extend_from_slice(&buffer_content[..]);
                let mut buffer_arc = Arc::new(buffer_stored);
                let mut buffer_dimensions = self.sized_buffers.render_dimensions;
                let logical_depth = buffer_dimensions[2].max(1);
                let storage_depth = self.sized_buffers.pixel_storage_layers.max(1);
                buffer_dimensions[2] = if self.last_backend == RenderBackend::VoxelTraversal {
                    1
                } else {
                    logical_depth.min(storage_depth)
                };
                let mut layers = Vec::new();
                struct HereGetPixel {
                    buffer_arc: Arc<Vec<Vec4>>,
                    dimensions: [u32; 3],
                    z: Option<u32>,
                }
                impl exr::prelude::GetPixel for HereGetPixel {
                    type Pixel = (f32, f32, f32, f32);

                    fn get_pixel(&self, position: exr::math::Vec2<usize>) -> Self::Pixel {
                        // Get cumulative pixel value
                        let mut full_pixel = Vec4::ZERO;
                        for z in 0..self.dimensions[2] {
                            full_pixel += self.buffer_arc[position.x()
                                + position.y() * self.dimensions[0] as usize
                                + (z * self.dimensions[0] * self.dimensions[1]) as usize];
                        }
                        if let Some(z) = self.z {
                            let local_pixel = self.buffer_arc[position.x()
                                + position.y() * self.dimensions[0] as usize
                                + (z * self.dimensions[0] * self.dimensions[1]) as usize];
                            (
                                local_pixel.x / local_pixel.w,
                                local_pixel.y / local_pixel.w,
                                local_pixel.z / local_pixel.w,
                                1.0,
                            )
                        } else {
                            (
                                full_pixel.x / full_pixel.w,
                                full_pixel.y / full_pixel.w,
                                full_pixel.z / full_pixel.w,
                                1.0,
                            )
                        }
                    }
                }
                // Render the full thing in layer 0 (because apparently support for openexr is rather poor)
                let pixel_getter = HereGetPixel {
                    buffer_arc: buffer_arc.clone(),
                    dimensions: buffer_dimensions,
                    z: None,
                };
                layers.push(exr::prelude::Layer::new(
                    (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                    exr::prelude::LayerAttributes {
                        layer_name: Some(exr::prelude::Text::new_or_panic(format!("Full Render"))),
                        ..Default::default()
                    },
                    exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                    exr::prelude::SpecificChannels::rgba(pixel_getter),
                ));
                for z in 0..buffer_dimensions[2] {
                    let pixel_getter = HereGetPixel {
                        buffer_arc: buffer_arc.clone(),
                        dimensions: buffer_dimensions,
                        z: Some(z),
                    };
                    let layer = exr::prelude::Layer::new(
                        (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                        exr::prelude::LayerAttributes {
                            layer_name: Some(exr::prelude::Text::new_or_panic(format!(
                                "ZW Slice {}/{}",
                                z, buffer_dimensions[2]
                            ))),
                            ..Default::default()
                        },
                        exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                        exr::prelude::SpecificChannels::rgba(pixel_getter),
                    );
                    layers.push(layer);
                }

                let image = exr::image::Image::from_layers(
                    exr::prelude::ImageAttributes {
                        display_window: exr::prelude::IntegerBounds::new(
                            (0, 0),
                            (
                                self.sized_buffers.render_dimensions[0] as usize,
                                self.sized_buffers.render_dimensions[1] as usize,
                            ),
                        ),
                        pixel_aspect: 1.0,
                        chromaticities: None,
                        time_code: None,
                        other: Default::default(),
                    },
                    layers,
                );
                image.write().to_file(path).unwrap();
                println!("Saved screenshot to {}", path);
            }
            Err(error) => {
                eprintln!("Error saving screenshot: {:?}", error);
            }
        };
    }

    pub fn save_rendered_frame_png(&mut self, path: &str) {
        self.wait_for_all_frames();

        let result = self.sized_buffers.output_cpu_pixel_buffer.read();
        match result {
            Ok(buffer_content) => {
                fn aces_tone_map(x: f32) -> f32 {
                    let a = 2.51f32;
                    let b = 0.03f32;
                    let c = 2.43f32;
                    let d = 0.59f32;
                    let e = 0.14f32;
                    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
                }

                let w = self.sized_buffers.render_dimensions[0] as u32;
                let h = self.sized_buffers.render_dimensions[1] as u32;
                let vte_collapsed = self.last_backend == RenderBackend::VoxelTraversal;
                let depth = if vte_collapsed {
                    1
                } else {
                    self.sized_buffers.render_dimensions[2]
                        .max(1)
                        .min(self.sized_buffers.pixel_storage_layers.max(1))
                };

                let mut pixels = Vec::with_capacity((w * h * 4) as usize);
                for y in 0..h {
                    for x in 0..w {
                        // Match present shader behavior:
                        // - VTE uses Stage-B-collapsed layer 0.
                        // - Legacy tetra paths accumulate all Z slices.
                        let accum_pixel = if vte_collapsed {
                            let idx = (y * w + x) as usize;
                            buffer_content[idx]
                        } else {
                            let mut full_pixel = Vec4::ZERO;
                            for z in 0..depth {
                                let idx = (z * w * h + y * w + x) as usize;
                                full_pixel += buffer_content[idx];
                            }
                            full_pixel
                        };

                        // Normalize by alpha (denominator) if non-zero.
                        let mut r = 0.0f32;
                        let mut g = 0.0f32;
                        let mut b = 0.0f32;
                        if accum_pixel.w > 0.0 {
                            r = accum_pixel.x / accum_pixel.w;
                            g = accum_pixel.y / accum_pixel.w;
                            b = accum_pixel.z / accum_pixel.w;
                        }

                        // Match present pass tone-map + gamma.
                        r = aces_tone_map(r);
                        g = aces_tone_map(g);
                        b = aces_tone_map(b);
                        let gamma = 1.0 / 2.2;
                        pixels.push((r.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push((g.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push((b.clamp(0.0, 1.0).powf(gamma) * 255.0) as u8);
                        pixels.push(255u8);
                    }
                }

                let image = ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, pixels).unwrap();
                image.save(path).unwrap();
                println!("Saved PNG to {}", path);
            }
            Err(error) => {
                eprintln!("Error saving PNG: {:?}", error);
            }
        };
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_cpu_screencapture_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    width: u32,
    height: u32,
    format: Format,
) -> Subbuffer<[u8]> {
    let block_extent = format.block_extent();
    let blocks_x = width.div_ceil(block_extent[0]) as u64;
    let blocks_y = height.div_ceil(block_extent[1]) as u64;
    let byte_len = blocks_x
        .saturating_mul(blocks_y)
        .saturating_mul(format.block_size()) as usize;

    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0; byte_len],
    )
    .unwrap()
}
