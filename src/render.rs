mod buffers;
#[cfg(test)]
mod bvh_topology_tests;
mod capture;
mod context_init;
mod geometry;
mod hud;
mod overlay;
mod pipelines;
mod profiler;
mod types;
mod vte;

use self::buffers::{LiveBuffers, OneTimeBuffers, SizedBuffers};
use self::geometry::{mat5_mul_vec5, project_view_point_to_ndc, transform_model_point};
use self::hud::{
    build_font_atlas, load_hud_font, map_to_panel, ndc_to_pixels, pixels_to_ndc, push_cross,
    push_filled_rect_quads, push_line, push_minecraft_crosshair, push_rect, push_text_lines,
    push_text_quads, text_width_px, HudResources, HudVertex, LineVertex, OverlayLine,
    HUD_VERTEX_CAPACITY,
};
use self::pipelines::{ComputePipelineContext, PresentPipelineContext};
use self::profiler::{GpuProfiler, PROFILER_MAX_TIMESTAMPS};
pub use self::types::*;
pub use self::vte::{
    GpuVoxelChunkHeader, GpuVoxelYSliceBounds, VoxelFrameInput, VteDebugCounters, VTE_MAX_CHUNKS,
};
use ab_glyph::FontArc;
use bytemuck::Zeroable;
use common::ModelTetrahedron;
use exr::prelude::WritableImage;
use glam::{Vec2, Vec4, Vec4Swizzles};
use image::{ImageBuffer, Rgba};
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
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
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::PipelineStage;
use vulkano::{sync, Validated, VulkanError};
use winit::dpi::PhysicalSize;
use winit::window::Window;

const VTE_LOD_TINT_ENV: &str = "R4D_VTE_LOD_TINT";
const VTE_ENTITY_LINEAR_ONLY_ENV: &str = "R4D_VTE_ENTITY_LINEAR_ONLY";
const VTE_ENTITY_BVH_COMPARE_ENV: &str = "R4D_VTE_ENTITY_BVH_COMPARE";
const VTE_ENTITY_DIAG_ENV: &str = "R4D_VTE_ENTITY_DIAG";
const VTE_ENTITY_DIAG_VERBOSE_ENV: &str = "R4D_VTE_ENTITY_DIAG_VERBOSE";
const VTE_ENTITY_DIAG_BVH_READBACK_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_READBACK";
const VTE_ENTITY_DIAG_BVH_TOPOLOGY_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_TOPOLOGY";
const VTE_ENTITY_DIAG_BVH_INTERVAL_ENV: &str = "R4D_VTE_ENTITY_DIAG_BVH_INTERVAL";
const VTE_ENTITY_DIAG_DEFAULT_INTERVAL: usize = 120;
const VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN: f32 = 16_384.0;
// Keep in sync with OVERLAY_RASTER_SCALE in `slang-shaders/src/rasterizer.slang`.
const VTE_OVERLAY_RASTER_SCALE: u32 = 3;
const WORKING_FLAG_VTE_COLLAPSED: u32 = 1u32 << 0u32;
const WORKING_FLAG_ZW_ANGLE_COLOR_SHIFT: u32 = 1u32 << 1u32;
const WORKING_ZW_SHIFT_STRENGTH_SHIFT: u32 = 8u32;

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

fn vte_lod_tint_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_LOD_TINT_ENV))
}

fn vte_entity_linear_only_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_ENTITY_LINEAR_ONLY_ENV))
}

fn vte_entity_bvh_compare_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled(VTE_ENTITY_BVH_COMPARE_ENV))
}

fn env_usize(name: &str, default_value: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|v| *v > 0)
            .unwrap_or(default_value),
        Err(_) => default_value,
    }
}

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
    bvh_link_parents: Arc<ShaderModule>,
    bvh_propagate_aabbs: Arc<ShaderModule>,
}

const LINE_VERTEX_CAPACITY: usize = 100_000;
const HUD_BREADCRUMB_CAPACITY: usize = 128;
const HUD_BREADCRUMB_MIN_STEP: f32 = 0.2;
const FRAMES_IN_FLIGHT: usize = 2;

struct FrameInFlight {
    live_buffers: LiveBuffers,
    line_vertexes_buffer: Subbuffer<[LineVertex]>,
    hud_vertex_buffer: Option<Subbuffer<[HudVertex]>>,
    hud_descriptor_set: Option<Arc<DescriptorSet>>,
    egui_descriptor_set: Option<Arc<DescriptorSet>>,
    material_icons_descriptor_set: Option<Arc<DescriptorSet>>,
    sized_descriptor_set: Arc<DescriptorSet>,
    cpu_clipped_tet_count_buffer: Subbuffer<[u32]>,
    query_pool: Arc<QueryPool>,
    fence: Option<Box<dyn GpuFuture>>,
    vte_compare_enabled: bool,
    pending_voxel_payload_slots: Vec<u32>,
    pending_voxel_payload_slot_set: HashSet<u32>,
    last_voxel_metadata_generation: Option<u64>,
    vte_entity_diag_copy_scheduled: bool,
    vte_entity_diag_non_voxel_tet_count: usize,
}

struct EguiResources {
    atlas_view: Arc<ImageView>,
    atlas_sampler: Arc<Sampler>,
    texture_size: [u32; 2],
    texture_pixels: Vec<u8>,
    retired_atlas_views: Vec<(Arc<ImageView>, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HudTextureSlot {
    Hud,
    EguiAtlas,
    MaterialIcons,
}

#[derive(Clone, Copy)]
struct HudDrawBatch {
    first_vertex: u32,
    vertex_count: u32,
    scissor: Scissor,
    texture_slot: HudTextureSlot,
}

fn create_rgba8_srgb_texture_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Arc<ImageView> {
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
        pixels.iter().copied(),
    )
    .unwrap();

    let atlas_image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent: [width.max(1), height.max(1), 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let mut upload_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
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
    let upload_future = sync::now(queue.device().clone())
        .then_execute(queue.clone(), upload_cmd)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    upload_future.wait(None).unwrap();

    ImageView::new_default(atlas_image).unwrap()
}

fn create_hud_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    hud_vertex_buffer: Subbuffer<[HudVertex]>,
    atlas_view: Arc<ImageView>,
    atlas_sampler: Arc<Sampler>,
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator,
        descriptor_set_layout,
        [
            WriteDescriptorSet::buffer(0, hud_vertex_buffer),
            WriteDescriptorSet::image_view_sampler(1, atlas_view, atlas_sampler),
        ],
        [],
    )
    .unwrap()
}

fn model_instance_is_finite(instance: &common::ModelInstance) -> bool {
    for row in 0..5 {
        for col in 0..5 {
            if !instance.model_transform[[row, col]].is_finite() {
                return false;
            }
        }
    }
    true
}

fn model_instance_transform_extrema(instances: &[common::ModelInstance]) -> (f32, f32, usize) {
    let mut translation_abs_max = 0.0f32;
    let mut basis_abs_max = 0.0f32;
    let mut outlier_count = 0usize;

    for instance in instances {
        let mut instance_translation_abs_max = 0.0f32;
        let mut instance_basis_abs_max = 0.0f32;
        for axis in 0..4 {
            instance_translation_abs_max =
                instance_translation_abs_max.max(instance.model_transform[[axis, 4]].abs());
            for basis_axis in 0..4 {
                instance_basis_abs_max =
                    instance_basis_abs_max.max(instance.model_transform[[axis, basis_axis]].abs());
            }
        }
        translation_abs_max = translation_abs_max.max(instance_translation_abs_max);
        basis_abs_max = basis_abs_max.max(instance_basis_abs_max);
        if instance_translation_abs_max > VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN
            || instance_basis_abs_max > VTE_ENTITY_DIAG_TRANSFORM_ABS_WARN
        {
            outlier_count += 1;
        }
    }

    (translation_abs_max, basis_abs_max, outlier_count)
}

struct BvhTopologySummary {
    total_nodes: usize,
    internal_nodes: usize,
    internal_ready: usize,
    invalid_child_edges: usize,
    self_child_edges: usize,
    nodes_without_parent_excluding_root: usize,
    nodes_with_multiple_parents: usize,
    unreachable_internal_nodes: usize,
    unreachable_leaf_nodes: usize,
    leaf_invalid_tetra_indices: usize,
    leaf_duplicate_tetra_indices: usize,
    leaf_missing_tetra_indices: usize,
}

fn summarize_bvh_topology(
    bvh_nodes: &[common::BVHNode],
    num_tetrahedrons: usize,
) -> Option<BvhTopologySummary> {
    if num_tetrahedrons == 0 {
        return None;
    }
    let total_nodes = num_tetrahedrons.checked_mul(2)?.checked_sub(1)?;
    if total_nodes == 0 || total_nodes > bvh_nodes.len() {
        return None;
    }
    let internal_nodes = num_tetrahedrons.saturating_sub(1);
    let mut parent_ref_counts = vec![0u32; total_nodes];
    let mut invalid_child_edges = 0usize;
    let mut self_child_edges = 0usize;
    let mut internal_ready = 0usize;

    for idx in 0..internal_nodes {
        let node = &bvh_nodes[idx];
        if node.atomic_visit_count >= 2 {
            internal_ready += 1;
        }
        for child in [node.left_child, node.right_child] {
            if child == u32::MAX {
                invalid_child_edges += 1;
                continue;
            }
            let child_idx = child as usize;
            if child_idx >= total_nodes {
                invalid_child_edges += 1;
                continue;
            }
            if child_idx == idx {
                self_child_edges += 1;
            }
            parent_ref_counts[child_idx] = parent_ref_counts[child_idx].saturating_add(1);
        }
    }

    let mut nodes_without_parent_excluding_root = 0usize;
    let mut nodes_with_multiple_parents = 0usize;
    for (idx, &count) in parent_ref_counts.iter().enumerate() {
        if idx == 0 {
            continue;
        }
        if count == 0 {
            nodes_without_parent_excluding_root += 1;
        } else if count > 1 {
            nodes_with_multiple_parents += 1;
        }
    }

    let mut visited = vec![false; total_nodes];
    let mut stack = Vec::with_capacity(total_nodes.min(256));
    stack.push(0usize);
    while let Some(idx) = stack.pop() {
        if idx >= total_nodes || visited[idx] {
            continue;
        }
        visited[idx] = true;
        let node = &bvh_nodes[idx];
        if node.is_leaf == 0 {
            for child in [node.left_child, node.right_child] {
                if child != u32::MAX {
                    let child_idx = child as usize;
                    if child_idx < total_nodes {
                        stack.push(child_idx);
                    }
                }
            }
        }
    }

    let unreachable_internal_nodes = visited[..internal_nodes]
        .iter()
        .filter(|&&seen| !seen)
        .count();
    let unreachable_leaf_nodes = visited[internal_nodes..total_nodes]
        .iter()
        .filter(|&&seen| !seen)
        .count();
    let mut leaf_seen = vec![0u8; num_tetrahedrons];
    let mut leaf_invalid_tetra_indices = 0usize;
    let mut leaf_duplicate_tetra_indices = 0usize;
    for idx in internal_nodes..total_nodes {
        let tet_idx = bvh_nodes[idx].tetrahedron_index as usize;
        if tet_idx >= num_tetrahedrons {
            leaf_invalid_tetra_indices += 1;
            continue;
        }
        if leaf_seen[tet_idx] != 0 {
            leaf_duplicate_tetra_indices += 1;
        } else {
            leaf_seen[tet_idx] = 1;
        }
    }
    let leaf_missing_tetra_indices = leaf_seen.iter().filter(|&&count| count == 0).count();

    Some(BvhTopologySummary {
        total_nodes,
        internal_nodes,
        internal_ready,
        invalid_child_edges,
        self_child_edges,
        nodes_without_parent_excluding_root,
        nodes_with_multiple_parents,
        unreachable_internal_nodes,
        unreachable_leaf_nodes,
        leaf_invalid_tetra_indices,
        leaf_duplicate_tetra_indices,
        leaf_missing_tetra_indices,
    })
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
    vte_non_voxel_scene_hash: u64,
    last_clipped_tet_count: u32,
    profiler: GpuProfiler,
    hud_font: Option<FontArc>,
    hud_resources: Option<HudResources>,
    egui_resources: Option<EguiResources>,
    material_icons_view: Option<Arc<ImageView>>,
    material_icons_sampler: Option<Arc<Sampler>>,
    hud_breadcrumbs: VecDeque<[f32; 4]>,
    hud_previous_camera: Option<[f32; 4]>,
    hud_previous_sample_time: Option<Instant>,
    hud_w_velocity: f32,
    frame_time_ms: f32,
    last_render_start: Option<Instant>,
    stall_trace: bool,
    last_backend: RenderBackend,
    vte_debug_counters: VteDebugCounters,
    vte_compare_stats: vte::VteCompareStats,
    vte_first_mismatch: vte::VteFirstMismatch,
    vte_backend_notice_printed: bool,
    vte_entity_diag_enabled: bool,
    vte_entity_diag_verbose: bool,
    vte_entity_diag_bvh_readback: bool,
    vte_entity_diag_bvh_topology: bool,
    vte_entity_diag_interval: usize,
    vte_entity_diag_last_log_frame: Option<usize>,
    vte_entity_diag_prev_used_non_voxel: Option<usize>,
    vte_entity_diag_prev_tets_non_voxel: Option<usize>,
    drop_next_profile_sample: bool,
    voxel_payload_cache_occupancy_words: Vec<u32>,
    voxel_payload_cache_material_words: Vec<u32>,
    voxel_payload_cache_macro_words: Vec<u32>,
}

impl RenderContext {
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
        self.vte_compare_stats = vte::VteCompareStats::default();
        self.vte_first_mismatch = vte::VteFirstMismatch::default();
    }

    fn refresh_vte_compare_diagnostics(&mut self, frame_idx: usize) {
        let stats_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_compare_stats_buffer
            .read()
            .unwrap();
        if stats_words.len() >= vte::VTE_COMPARE_STATS_WORD_COUNT {
            self.vte_compare_stats = vte::VteCompareStats {
                compared: stats_words[vte::VTE_COMPARE_STAT_COMPARED],
                matches: stats_words[vte::VTE_COMPARE_STAT_MATCHES],
                mismatches: stats_words[vte::VTE_COMPARE_STAT_MISMATCHES],
                hit_state_mismatches: stats_words[vte::VTE_COMPARE_STAT_HIT_STATE_MISMATCHES],
                chunk_material_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_CHUNK_MATERIAL_MISMATCHES],
                fast_miss_ref_hit: stats_words[vte::VTE_COMPARE_STAT_FAST_MISS_REF_HIT],
                fast_hit_ref_miss: stats_words[vte::VTE_COMPARE_STAT_FAST_HIT_REF_MISS],
                miss_reason_counts: [
                    stats_words[vte::VTE_COMPARE_STAT_REASON_NONE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_TOUCHED_VISIBLE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_VOXEL_BUDGET],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_CHUNK_BUDGET],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_MAX_DISTANCE],
                    stats_words[vte::VTE_COMPARE_STAT_REASON_LOOKUP_FALSE_NEGATIVE],
                ],
                zero_interval_flags: stats_words[vte::VTE_COMPARE_STAT_ZERO_INTERVAL_FLAG],
                tie_stepped_flags: stats_words[vte::VTE_COMPARE_STAT_TIE_STEPPED_FLAG],
                lookup_fallback_flags: stats_words[vte::VTE_COMPARE_STAT_LOOKUP_FALLBACK_FLAG],
                entity_bvh_samples: stats_words[vte::VTE_COMPARE_STAT_ENTITY_BVH_SAMPLE],
                entity_bvh_mismatches: stats_words[vte::VTE_COMPARE_STAT_ENTITY_BVH_MISMATCH],
                entity_bvh_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_HIT_STATE_MISMATCH],
                entity_bvh_material_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_MATERIAL_MISMATCH],
                entity_bvh_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_DISTANCE_MISMATCH],
                entity_bvh_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_TETRA_MISMATCH],
                entity_bvh_miss_linear_hit: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_MISS_LINEAR_HIT],
                entity_bvh_hit_linear_miss: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_HIT_LINEAR_MISS],
                entity_bvh_noprune_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_MISMATCH],
                entity_bvh_noprune_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_HIT_STATE_MISMATCH],
                entity_bvh_noprune_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_DISTANCE_MISMATCH],
                entity_bvh_noprune_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOPRUNE_TETRA_MISMATCH],
                entity_bvh_noaabb_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_MISMATCH],
                entity_bvh_noaabb_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_HIT_STATE_MISMATCH],
                entity_bvh_noaabb_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_DISTANCE_MISMATCH],
                entity_bvh_noaabb_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_NOAABB_TETRA_MISMATCH],
                entity_linear_order_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_MISMATCH],
                entity_linear_order_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_HIT_STATE_MISMATCH],
                entity_linear_order_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_DISTANCE_MISMATCH],
                entity_linear_order_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_LINEAR_ORDER_TETRA_MISMATCH],
                entity_bvh_leafarray_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_MISMATCH],
                entity_bvh_leafarray_hit_state_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_HIT_STATE_MISMATCH],
                entity_bvh_leafarray_distance_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_DISTANCE_MISMATCH],
                entity_bvh_leafarray_tetra_mismatches: stats_words
                    [vte::VTE_COMPARE_STAT_ENTITY_BVH_LEAFARRAY_TETRA_MISMATCH],
            };
        } else {
            self.vte_compare_stats = vte::VteCompareStats::default();
        }

        let first_words = self.frames_in_flight[frame_idx]
            .live_buffers
            .vte_first_mismatch_buffer
            .read()
            .unwrap();
        if first_words.len() >= vte::VTE_FIRST_MISMATCH_WORD_COUNT
            && first_words[vte::VTE_FIRST_MISMATCH_VALID] != 0
        {
            let hit_mask = first_words[vte::VTE_FIRST_MISMATCH_HIT_MASK];
            self.vte_first_mismatch = vte::VteFirstMismatch {
                valid: true,
                pixel_x: first_words[vte::VTE_FIRST_MISMATCH_PIXEL_X],
                pixel_y: first_words[vte::VTE_FIRST_MISMATCH_PIXEL_Y],
                layer: first_words[vte::VTE_FIRST_MISMATCH_LAYER],
                mismatch_kind: first_words[vte::VTE_FIRST_MISMATCH_KIND],
                miss_reason: first_words[vte::VTE_FIRST_MISMATCH_MISS_REASON],
                debug_flags: first_words[vte::VTE_FIRST_MISMATCH_DEBUG_FLAGS],
                fast_hit: (hit_mask & 0x1) != 0,
                ref_hit: (hit_mask & 0x2) != 0,
                fast_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_FAST_CHUNK_W] as i32,
                ],
                ref_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_REF_CHUNK_W] as i32,
                ],
                fast_material: first_words[vte::VTE_FIRST_MISMATCH_FAST_MATERIAL],
                ref_material: first_words[vte::VTE_FIRST_MISMATCH_REF_MATERIAL],
                fast_hit_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_FAST_HIT_T]),
                ref_hit_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_REF_HIT_T]),
                chunk_steps_taken: first_words[vte::VTE_FIRST_MISMATCH_CHUNK_STEPS],
                remaining_voxel_steps: first_words[vte::VTE_FIRST_MISMATCH_REMAINING_VOXELS],
                final_t: f32::from_bits(first_words[vte::VTE_FIRST_MISMATCH_FINAL_T]),
                last_chunk: [
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_X] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_Y] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_Z] as i32,
                    first_words[vte::VTE_FIRST_MISMATCH_LAST_CHUNK_W] as i32,
                ],
            };
        } else {
            self.vte_first_mismatch = vte::VteFirstMismatch::default();
        }
    }

    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }

    /// Recreate all resolution-dependent GPU buffers at a new render size.
    /// Waits for in-flight GPU work to complete, then rebuilds sized buffers
    /// and all per-frame descriptor sets that reference them.
    pub fn recreate_sized_buffers(
        &mut self,
        new_dimensions: [u32; 3],
        pixel_storage_layers: Option<u32>,
    ) {
        self.wait_for_all_frames();

        let new_sized = SizedBuffers::new(
            self.memory_allocator.clone(),
            new_dimensions,
            pixel_storage_layers,
        );

        // The sized descriptor set layout is set_layouts[1] in the compute pipeline layout.
        let sized_ds_layout = self.compute_pipeline.pipeline_layout.set_layouts()[1].clone();

        for frame in &mut self.frames_in_flight {
            frame.sized_descriptor_set = new_sized.create_sized_descriptor_set(
                &frame.line_vertexes_buffer,
                self.descriptor_set_allocator.clone(),
                sized_ds_layout.clone(),
            );
        }

        self.sized_buffers = new_sized;

        // Reset profiler to avoid stale timing data from the old resolution.
        self.profiler.next_query = 0;
        self.profiler.phase_names.clear();
        self.profiler.accum.clear();
        self.profiler.total_frames = 0;
        self.profiler.last_frame_phases.clear();
        self.profiler.last_gpu_total_ms = 0.0;
        self.profiler.last_slow_report_frame = None;
        self.drop_next_profile_sample = true;

        eprintln!(
            "[render] Resized buffers to {}x{}x{}",
            new_dimensions[0], new_dimensions[1], new_dimensions[2]
        );
    }

    pub fn reset_gpu_profile_window(&mut self) {
        self.profiler.next_query = 0;
        self.profiler.phase_names.clear();
        self.profiler.accum.clear();
        self.profiler.total_frames = 0;
        self.profiler.last_frame_phases.clear();
        self.profiler.last_gpu_total_ms = 0.0;
        self.profiler.last_slow_report_frame = None;
        self.drop_next_profile_sample = true;
    }

    pub fn flush_gpu_profile_report_now(&mut self) {
        if self.profiler.total_frames == 0 && self.profiler.accum.is_empty() {
            return;
        }
        self.profiler.print_report();
        self.drop_next_profile_sample = true;
    }

    pub fn last_gpu_frame_ms(&self) -> f32 {
        self.profiler.last_gpu_total_ms
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
                time_ticks_ms: (self.frames_rendered as u64).wrapping_mul(16) as u32,
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
            &[],
            None,
        );
    }

    pub fn render_voxel_frame(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        mut frame_params: FrameParams,
        voxel_input: VoxelFrameInput<'_>,
        tetra_entity_instances: &[common::ModelInstance],
        tetra_overlay_instances: &[common::ModelInstance],
    ) {
        frame_params.render_options.render_backend = RenderBackend::VoxelTraversal;
        self.render_internal(
            device,
            queue,
            frame_params,
            tetra_entity_instances,
            tetra_overlay_instances,
            Some(&voxel_input),
        );
    }

    fn stage_voxel_payload_updates(&mut self, input: &VoxelFrameInput<'_>) -> usize {
        let mut touched_slots = Vec::new();
        let max_updates = vte::stage_voxel_payload_updates(
            input,
            &mut self.voxel_payload_cache_occupancy_words,
            &mut self.voxel_payload_cache_material_words,
            &mut self.voxel_payload_cache_macro_words,
            &mut touched_slots,
        );

        for slot_u32 in touched_slots {
            for frame in &mut self.frames_in_flight {
                if frame.pending_voxel_payload_slot_set.insert(slot_u32) {
                    frame.pending_voxel_payload_slots.push(slot_u32);
                }
            }
        }

        max_updates
    }

    fn apply_pending_voxel_payload_updates_for_frame(&mut self, frame_idx: usize) -> usize {
        let frame = &mut self.frames_in_flight[frame_idx];
        vte::apply_pending_voxel_payload_updates_for_frame(
            &mut frame.pending_voxel_payload_slots,
            &mut frame.pending_voxel_payload_slot_set,
            &frame.live_buffers.voxel_occupancy_words_buffer,
            &frame.live_buffers.voxel_material_words_buffer,
            &frame.live_buffers.voxel_macro_words_buffer,
            &self.voxel_payload_cache_occupancy_words,
            &self.voxel_payload_cache_material_words,
            &self.voxel_payload_cache_macro_words,
        )
    }

    fn render_internal(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        frame_params: FrameParams,
        model_instances_input: &[common::ModelInstance],
        raster_overlay_instances_input: &[common::ModelInstance],
        voxel_input: Option<&VoxelFrameInput<'_>>,
    ) {
        let FrameParams {
            view_matrix,
            time_ticks_ms,
            focal_length_xy,
            focal_length_zw,
            render_options,
        } = frame_params;
        let view_matrix_view = view_matrix.into_owned();

        // Guard against non-finite transforms/material data poisoning shared
        // non-voxel preprocess/BVH buffers for the entire frame.
        let mut filtered_model_instances: Vec<common::ModelInstance> = Vec::new();
        let mut dropped_model_instance_count = 0usize;
        let model_instances: &[common::ModelInstance] =
            if model_instances_input.iter().all(model_instance_is_finite) {
                model_instances_input
            } else {
                filtered_model_instances.reserve(model_instances_input.len());
                for instance in model_instances_input.iter().copied() {
                    if model_instance_is_finite(&instance) {
                        filtered_model_instances.push(instance);
                    } else {
                        dropped_model_instance_count += 1;
                    }
                }
                filtered_model_instances.as_slice()
            };

        let mut filtered_overlay_instances: Vec<common::ModelInstance> = Vec::new();
        let mut dropped_overlay_instance_count = 0usize;
        let raster_overlay_instances: &[common::ModelInstance] = if raster_overlay_instances_input
            .iter()
            .all(model_instance_is_finite)
        {
            raster_overlay_instances_input
        } else {
            filtered_overlay_instances.reserve(raster_overlay_instances_input.len());
            for instance in raster_overlay_instances_input.iter().copied() {
                if model_instance_is_finite(&instance) {
                    filtered_overlay_instances.push(instance);
                } else {
                    dropped_overlay_instance_count += 1;
                }
            }
            filtered_overlay_instances.as_slice()
        };

        let mut filtered_custom_overlay_edge_instances: Vec<common::ModelInstance> = Vec::new();
        let mut dropped_custom_overlay_edge_instance_count = 0usize;
        let custom_overlay_edge_instances: &[common::ModelInstance] = if render_options
            .custom_overlay_edge_instances
            .iter()
            .all(model_instance_is_finite)
        {
            &render_options.custom_overlay_edge_instances
        } else {
            filtered_custom_overlay_edge_instances
                .reserve(render_options.custom_overlay_edge_instances.len());
            for instance in render_options.custom_overlay_edge_instances.iter().copied() {
                if model_instance_is_finite(&instance) {
                    filtered_custom_overlay_edge_instances.push(instance);
                } else {
                    dropped_custom_overlay_edge_instance_count += 1;
                }
            }
            filtered_custom_overlay_edge_instances.as_slice()
        };

        if dropped_model_instance_count > 0
            || dropped_overlay_instance_count > 0
            || dropped_custom_overlay_edge_instance_count > 0
        {
            eprintln!(
                "Dropped non-finite model instances before render: non-voxel {} overlay {} edge_overlay {} (frame {}).",
                dropped_model_instance_count,
                dropped_overlay_instance_count,
                dropped_custom_overlay_edge_instance_count,
                self.frames_rendered
            );
        }

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
        let use_split_voxel_instances = voxel_input.is_some();
        let non_voxel_used_instance_count = model_instances.len().min(model_instance_capacity);
        let overlay_instance_capacity =
            model_instance_capacity.saturating_sub(non_voxel_used_instance_count);
        let overlay_used_instance_count = if use_split_voxel_instances {
            raster_overlay_instances
                .len()
                .min(overlay_instance_capacity)
        } else {
            0
        };
        let used_instance_count = non_voxel_used_instance_count + overlay_used_instance_count;
        let custom_overlay_edge_instance_base = used_instance_count;
        let custom_overlay_edge_instance_capacity =
            model_instance_capacity.saturating_sub(custom_overlay_edge_instance_base);
        let custom_overlay_edge_used_instance_count = custom_overlay_edge_instances
            .len()
            .min(custom_overlay_edge_instance_capacity);

        let requested_tetrahedron_count =
            self.one_time_buffers.model_tetrahedron_count * non_voxel_used_instance_count;
        let total_tetrahedron_count =
            requested_tetrahedron_count.min(self.sized_buffers.max_tetrahedrons);
        let requested_raster_overlay_tetrahedron_count =
            self.one_time_buffers.model_tetrahedron_count * overlay_used_instance_count;
        let raster_overlay_tetrahedron_count =
            requested_raster_overlay_tetrahedron_count.min(self.sized_buffers.max_tetrahedrons);
        let raster_instance_base = if use_split_voxel_instances {
            non_voxel_used_instance_count
        } else {
            0
        };
        let raster_tetrahedron_count = if use_split_voxel_instances {
            raster_overlay_tetrahedron_count
        } else {
            total_tetrahedron_count
        };
        let (non_voxel_translation_abs_max, non_voxel_basis_abs_max, non_voxel_outlier_count) =
            model_instance_transform_extrema(&model_instances[..non_voxel_used_instance_count]);

        // Debug: print scene info on first frame only
        if self.frames_rendered == 0 {
            if use_split_voxel_instances {
                if model_instances.len() > non_voxel_used_instance_count {
                    eprintln!(
                        "VTE non-voxel input truncated to buffer capacity: {} -> {}",
                        model_instances.len(),
                        non_voxel_used_instance_count
                    );
                }
                if raster_overlay_instances.len() > overlay_used_instance_count {
                    eprintln!(
                        "VTE raster overlay input truncated to buffer capacity: {} -> {}",
                        raster_overlay_instances.len(),
                        overlay_used_instance_count
                    );
                }
                if requested_tetrahedron_count > total_tetrahedron_count {
                    eprintln!(
                        "VTE non-voxel tetrahedrons truncated to buffer capacity: {} -> {}",
                        requested_tetrahedron_count, total_tetrahedron_count
                    );
                }
                if requested_raster_overlay_tetrahedron_count > raster_overlay_tetrahedron_count {
                    eprintln!(
                        "VTE raster overlay tetrahedrons truncated to buffer capacity: {} -> {}",
                        requested_raster_overlay_tetrahedron_count,
                        raster_overlay_tetrahedron_count
                    );
                }
                println!(
                    "VTE scene: {} non-voxel tetrahedrons, {} raster overlay tetrahedrons ({} per instance × non_voxel_instances={}, overlays={})",
                    total_tetrahedron_count,
                    raster_overlay_tetrahedron_count,
                    self.one_time_buffers.model_tetrahedron_count,
                    non_voxel_used_instance_count,
                    overlay_used_instance_count
                );
                println!(
                    "VTE non-voxel BVH: {} internal nodes, {} total nodes",
                    total_tetrahedron_count.saturating_sub(1),
                    2 * total_tetrahedron_count.saturating_sub(1) + 1
                );
            } else {
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
            if custom_overlay_edge_instances.len() > custom_overlay_edge_used_instance_count {
                eprintln!(
                    "Custom edge-overlay instances truncated to buffer capacity: {} -> {}",
                    custom_overlay_edge_instances.len(),
                    custom_overlay_edge_used_instance_count
                );
            }
        }
        {
            let world_origin_h =
                mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 0.0, 0.0, 1.0]);
            let world_origin_inv_w = if world_origin_h[4].abs() > 1e-6 {
                1.0 / world_origin_h[4]
            } else {
                1.0
            };
            let world_origin = glam::Vec4::new(
                world_origin_h[0] * world_origin_inv_w,
                world_origin_h[1] * world_origin_inv_w,
                world_origin_h[2] * world_origin_inv_w,
                world_origin_h[3] * world_origin_inv_w,
            );
            let world_dir_x_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [1.0, 0.0, 0.0, 0.0, 0.0]);
            let world_dir_y_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 1.0, 0.0, 0.0, 0.0]);
            let world_dir_z_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 1.0, 0.0, 0.0]);
            let world_dir_w_h = mat5_mul_vec5(&view_matrix_nalgebra_inv, [0.0, 0.0, 0.0, 1.0, 0.0]);
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
            writer.time_ticks_ms = time_ticks_ms;
            writer.focal_length_xy = focal_length_xy;
            writer.focal_length_zw = focal_length_zw;
            let mut working_flags = 0u32;
            if voxel_input.is_some() {
                working_flags |= WORKING_FLAG_VTE_COLLAPSED;
            }
            if render_options.zw_angle_color_shift_enabled {
                working_flags |= WORKING_FLAG_ZW_ANGLE_COLOR_SHIFT;
            }
            let zw_shift_strength_q = (render_options.zw_angle_color_shift_strength.clamp(0.0, 1.0)
                * 255.0)
                .round() as u32;
            working_flags |= zw_shift_strength_q << WORKING_ZW_SHIFT_STRENGTH_SHIFT;
            // Flag used by present shader:
            // - padding[0] bit0: 0 = legacy per-layer accumulation, 1 = VTE Stage-B-collapsed output in layer 0.
            // - padding[0] bit1: ZW angle color shift enabled.
            // - padding[0] bits8..15: quantized ZW angle color shift strength [0, 255].
            // padding[1] carries VTE stage_b_mode so present shader can conditionally
            // bypass tone mapping for debug compare output.
            writer.padding = [working_flags, render_options.vte_display_mode.as_u32()];
            writer.world_origin = world_origin;
            writer.world_dir_x = glam::Vec4::new(
                world_dir_x_h[0],
                world_dir_x_h[1],
                world_dir_x_h[2],
                world_dir_x_h[3],
            );
            writer.world_dir_y = glam::Vec4::new(
                world_dir_y_h[0],
                world_dir_y_h[1],
                world_dir_y_h[2],
                world_dir_y_h[3],
            );
            writer.world_dir_z = glam::Vec4::new(
                world_dir_z_h[0],
                world_dir_z_h[1],
                world_dir_z_h[2],
                world_dir_z_h[3],
            );
            writer.world_dir_w = glam::Vec4::new(
                world_dir_w_h[0],
                world_dir_w_h[1],
                world_dir_w_h[2],
                world_dir_w_h[3],
            );
        }

        // Compute non-voxel scene hash BEFORE writing to the GPU buffer so the
        // BVH rebuild decision later in this frame reflects what is actually in
        // the buffer.  Using a per-frame hash avoids stale comparisons when
        // multiple frames are in flight.
        let non_voxel_scene_hash = if total_tetrahedron_count == 0 {
            0u64
        } else {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            non_voxel_used_instance_count.hash(&mut hasher);
            total_tetrahedron_count.hash(&mut hasher);
            bytemuck::cast_slice::<_, u8>(&model_instances[..non_voxel_used_instance_count])
                .hash(&mut hasher);
            hasher.finish()
        };

        {
            let mut writer = self.frames_in_flight[frame_idx]
                .live_buffers
                .model_instance_buffer
                .write()
                .unwrap();
            for i in 0..non_voxel_used_instance_count {
                writer[i] = model_instances[i];
            }
            for i in 0..overlay_used_instance_count {
                writer[non_voxel_used_instance_count + i] = raster_overlay_instances[i];
            }
            for i in 0..custom_overlay_edge_used_instance_count {
                writer[custom_overlay_edge_instance_base + i] = custom_overlay_edge_instances[i];
            }
        }

        let mut vte_chunk_count: usize = 0;
        let mut vte_visible_chunk_count: usize = 0;
        let mut vte_chunk_lookup_capacity: usize = 0;
        let mut vte_payload_update_count: usize = 0;
        let mut vte_payload_updates_applied: usize = 0;
        let mut vte_occupancy_word_count: usize = 0;
        let mut vte_material_word_count: usize = 0;
        let mut vte_macro_word_count: usize = 0;
        let mut vte_y_slice_count: usize = 0;
        let mut vte_y_slice_lookup_entry_count: usize = 0;
        let mut vte_visible_lod_counts = [0u32; 3];
        let mut vte_visible_chunk_min = [i32::MAX; 4];
        let mut vte_visible_chunk_max = [i32::MIN; 4];
        if let Some(input) = voxel_input {
            vte_payload_update_count = self.stage_voxel_payload_updates(input);
            vte_chunk_count = input.chunk_headers.len().min(vte::VTE_MAX_CHUNKS);
            vte_visible_chunk_count = input
                .visible_chunk_indices
                .len()
                .min(vte::VTE_MAX_CHUNKS)
                .min(vte_chunk_count);
            vte_occupancy_word_count = vte::VTE_MAX_CHUNKS * vte::VTE_OCCUPANCY_WORDS_PER_CHUNK;
            vte_material_word_count = vte::VTE_MAX_CHUNKS * vte::VTE_MATERIAL_WORDS_PER_CHUNK;
            vte_macro_word_count = vte::VTE_MAX_CHUNKS * vte::VTE_MACRO_WORDS_PER_CHUNK;
            vte_y_slice_count = input.y_slice_bounds.len().min(vte::VTE_MAX_Y_SLICES);
            vte_y_slice_lookup_entry_count = input
                .y_slice_lookup_entries
                .len()
                .min(vte::VTE_MAX_Y_SLICE_LOOKUP_ENTRIES);

            let max_payload_updates = input
                .payload_update_slots
                .len()
                .min(input.occupancy_words.len() / vte::VTE_OCCUPANCY_WORDS_PER_CHUNK)
                .min(input.material_words.len() / vte::VTE_MATERIAL_WORDS_PER_CHUNK)
                .min(input.macro_words.len() / vte::VTE_MACRO_WORDS_PER_CHUNK);
            if self.frames_rendered == 0
                && (input.chunk_headers.len() > vte_chunk_count
                    || input.visible_chunk_indices.len() > vte_visible_chunk_count
                    || input.y_slice_bounds.len() > vte_y_slice_count
                    || input.y_slice_lookup_entries.len() > vte_y_slice_lookup_entry_count
                    || max_payload_updates < input.payload_update_slots.len())
            {
                eprintln!(
                    "VTE input truncated to capacities: chunks {}->{}, visible {}->{}, payload_updates {}->{}, y_slices {}->{}, y_slice_lookup_entries {}->{}",
                    input.chunk_headers.len(),
                    vte_chunk_count,
                    input.visible_chunk_indices.len(),
                    vte_visible_chunk_count,
                    input.payload_update_slots.len(),
                    max_payload_updates,
                    input.y_slice_bounds.len(),
                    vte_y_slice_count,
                    input.y_slice_lookup_entries.len(),
                    vte_y_slice_lookup_entry_count
                );
            }

            let metadata_dirty = self.frames_in_flight[frame_idx].last_voxel_metadata_generation
                != Some(input.metadata_generation);
            vte_chunk_lookup_capacity = if vte_visible_chunk_count == 0 {
                0
            } else {
                let requested = (vte_visible_chunk_count.saturating_mul(2)).next_power_of_two();
                requested.clamp(1, vte::VTE_CHUNK_LOOKUP_CAPACITY)
            };

            if metadata_dirty {
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
                        .voxel_y_slice_bounds_buffer
                        .write()
                        .unwrap();
                    for i in 0..vte_y_slice_count {
                        writer[i] = input.y_slice_bounds[i];
                    }
                }
                {
                    let mut writer = self.frames_in_flight[frame_idx]
                        .live_buffers
                        .voxel_y_slice_lookup_entries_buffer
                        .write()
                        .unwrap();
                    for i in 0..vte_y_slice_lookup_entry_count {
                        writer[i] = input.y_slice_lookup_entries[i];
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
                        let header = &input.chunk_headers[chunk_index as usize];
                        let chunk_coord = header.chunk_coord;
                        if let Some(slot) =
                            vte_visible_lod_counts.get_mut(header.lod_level as usize)
                        {
                            *slot = slot.saturating_add(1);
                        }
                        for axis in 0..4 {
                            vte_visible_chunk_min[axis] =
                                vte_visible_chunk_min[axis].min(chunk_coord[axis]);
                            vte_visible_chunk_max[axis] =
                                vte_visible_chunk_max[axis].max(chunk_coord[axis]);
                        }
                    }
                }
                {
                    debug_assert!(vte::VTE_CHUNK_LOOKUP_CAPACITY.is_power_of_two());
                    let mut writer = self.frames_in_flight[frame_idx]
                        .live_buffers
                        .voxel_chunk_lookup_buffer
                        .write()
                        .unwrap();

                    for entry in writer.iter_mut().take(vte_chunk_lookup_capacity) {
                        *entry = vte::GpuVoxelChunkLookupEntry::empty();
                    }

                    if vte_chunk_count > 0 && vte_chunk_lookup_capacity > 0 {
                        let hash_mask = (vte_chunk_lookup_capacity as u32) - 1;
                        for i in 0..vte_visible_chunk_count {
                            let chunk_index = input.visible_chunk_indices[i]
                                .min(vte_chunk_count.saturating_sub(1) as u32);
                            let chunk_coord = input.chunk_headers[chunk_index as usize].chunk_coord;
                            let lod_level = input.chunk_headers[chunk_index as usize].lod_level;
                            let mut slot =
                                (vte::vte_hash_chunk_coord_with_lod(chunk_coord, lod_level)
                                    & hash_mask) as usize;

                            for _ in 0..vte_chunk_lookup_capacity {
                                let entry = &mut writer[slot];
                                if entry.chunk_index == vte::GpuVoxelChunkLookupEntry::INVALID_INDEX
                                    || entry.chunk_coord == chunk_coord
                                {
                                    *entry = vte::GpuVoxelChunkLookupEntry {
                                        chunk_coord,
                                        chunk_index,
                                        lod_level,
                                        _padding: [0; 2],
                                    };
                                    break;
                                }
                                slot = (slot + 1) & (vte_chunk_lookup_capacity - 1);
                            }
                        }
                    }
                }
                self.frames_in_flight[frame_idx].last_voxel_metadata_generation =
                    Some(input.metadata_generation);
            } else {
                for i in 0..vte_visible_chunk_count {
                    let chunk_index = input.visible_chunk_indices[i]
                        .min(vte_chunk_count.saturating_sub(1) as u32);
                    let header = &input.chunk_headers[chunk_index as usize];
                    let chunk_coord = header.chunk_coord;
                    if let Some(slot) = vte_visible_lod_counts.get_mut(header.lod_level as usize) {
                        *slot = slot.saturating_add(1);
                    }
                    for axis in 0..4 {
                        vte_visible_chunk_min[axis] =
                            vte_visible_chunk_min[axis].min(chunk_coord[axis]);
                        vte_visible_chunk_max[axis] =
                            vte_visible_chunk_max[axis].max(chunk_coord[axis]);
                    }
                }
            }

            vte_payload_updates_applied =
                self.apply_pending_voxel_payload_updates_for_frame(frame_idx);
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
                    highlight_flags |= vte::VTE_HIGHLIGHT_FLAG_HIT_VOXEL;
                    highlight_hit_voxel = hit_voxel;
                }
                if let Some(place_voxel) = render_options.vte_highlight_place_voxel {
                    highlight_flags |= vte::VTE_HIGHLIGHT_FLAG_PLACE_VOXEL;
                    highlight_place_voxel = place_voxel;
                }
            }
            // Reuse frame-meta padding words for fused-integral controls.
            // Values are consumed in shader via asfloat().
            let integral_sky_scale = if render_options.vte_integral_sky_emissive_tweak {
                render_options.vte_integral_sky_scale.max(0.0)
            } else {
                1.0
            };
            let integral_hit_emissive_boost = if render_options.vte_integral_sky_emissive_tweak {
                render_options.vte_integral_hit_emissive_boost.max(0.0)
            } else {
                0.0
            };
            let integral_log_merge_k = if render_options.vte_integral_log_merge_tweak {
                render_options.vte_integral_log_merge_k.max(0.0)
            } else {
                0.0
            };
            let max_trace_distance = render_options.vte_max_trace_distance.max(1.0);
            let lod_near_max_distance = render_options
                .vte_lod_near_max_distance
                .max(1.0)
                .min(max_trace_distance);
            let lod_mid_max_distance = render_options
                .vte_lod_mid_max_distance
                .max(lod_near_max_distance)
                .min(max_trace_distance);
            *writer = vte::GpuVoxelFrameMeta {
                chunk_count: vte_chunk_count as u32,
                visible_chunk_count: vte_visible_chunk_count as u32,
                occupancy_word_count: vte_occupancy_word_count as u32,
                material_word_count: vte_material_word_count as u32,
                macro_word_count: vte_macro_word_count as u32,
                max_trace_steps: render_options.vte_max_trace_steps.max(1),
                max_trace_distance,
                lod_near_max_distance,
                lod_mid_max_distance,
                chunk_lookup_capacity: vte_chunk_lookup_capacity as u32,
                y_slice_count: vte_y_slice_count as u32,
                y_slice_lookup_entry_count: vte_y_slice_lookup_entry_count as u32,
                stage_b_mode: render_options.vte_display_mode.as_u32(),
                stage_b_slice_layer,
                stage_b_thick_half_width: render_options.vte_thick_half_width,
                debug_flags: {
                    let mut flags = 0;
                    if render_options.vte_reference_compare {
                        flags |= vte::VTE_DEBUG_FLAG_REFERENCE_COMPARE;
                    }
                    if render_options.vte_reference_mismatch_only {
                        flags |= vte::VTE_DEBUG_FLAG_REFERENCE_MISMATCH_ONLY;
                    }
                    if render_options.vte_compare_slice_only {
                        flags |= vte::VTE_DEBUG_FLAG_COMPARE_SLICE_ONLY;
                    }
                    if render_options.vte_y_slice_lookup_cache {
                        flags |= vte::VTE_DEBUG_FLAG_YSLICE_LOOKUP_CACHE;
                    }
                    if vte_lod_tint_enabled() {
                        flags |= vte::VTE_DEBUG_FLAG_LOD_TINT;
                    }
                    if vte_entity_linear_only_enabled() {
                        flags |= vte::VTE_DEBUG_FLAG_ENTITY_LINEAR_ONLY;
                    }
                    if vte_entity_bvh_compare_enabled() {
                        flags |= vte::VTE_DEBUG_FLAG_ENTITY_BVH_COMPARE;
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
                _highlight_padding: [
                    integral_sky_scale.to_bits(),
                    integral_hit_emissive_boost.to_bits(),
                    integral_log_merge_k.to_bits(),
                ],
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
                // VTE resolves non-voxel tetrahedra directly in Stage A for depth correctness.
                // Optional post-raster overlays (e.g. held block preview) are rasterized from
                // a separate instance range to avoid double-rendering depth-tested entities.
                do_raster = raster_tetrahedron_count > 0;
                do_raytrace = false;
                do_edges = false;
                do_tetrahedron_edges = false;
                do_voxel_vte = true;
            }
        }

        let previous_vte_non_voxel_scene_hash = self.vte_non_voxel_scene_hash;
        let mut vte_non_voxel_rebuild_needed = false;
        let mut vte_non_voxel_rebuild_executed = false;
        let mut vte_non_voxel_rebuild_reason = "non_vte";

        if do_voxel_vte {
            vte_non_voxel_rebuild_reason = "empty";
            vte_compare_diagnostics_enabled = render_options.vte_reference_compare
                && matches!(
                    render_options.vte_display_mode,
                    VteDisplayMode::DebugCompare | VteDisplayMode::DebugIntegral
                );
            let vte_entity_bvh_compare = vte_entity_bvh_compare_enabled();
            if vte_compare_diagnostics_enabled || vte_entity_bvh_compare {
                self.reset_vte_compare_buffers(frame_idx);
            } else {
                self.clear_vte_compare_diagnostics();
            }
            let (
                candidate_chunks,
                visible_chunks,
                empty_chunks,
                full_chunks,
                payload_updates_received,
                payload_updates_applied,
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
                        let h = vte::vte_hash_chunk_coord_with_lod(
                            header.chunk_coord,
                            header.lod_level,
                        );
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
                    vte_payload_update_count as u64,
                    vte_payload_updates_applied as u64,
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
                    "Render backend '{}' selected: VTE active (chunks={}, visible={}, payload_updates={} applied={}, max_trace_steps={}, max_trace_distance={:.1}, lod_near={:.1}, lod_mid={:.1}, stage_b={}, slice_layer={:?}, thick_half_width={}, reference_compare={}, mismatch_only={}, compare_slice_only={}, yslice_fastpath=true, chunk_solid_clip=true, yslice_lookup_cache={}, lod_tint={}, entity_linear_only={}, entity_bvh_compare={}, storage_layers={}/{}).",
                    RenderBackend::VoxelTraversal.label(),
                    candidate_chunks,
                    visible_chunks,
                    payload_updates_received,
                    payload_updates_applied,
                    render_options.vte_max_trace_steps.max(1),
                    render_options.vte_max_trace_distance.max(1.0),
                    render_options.vte_lod_near_max_distance,
                    render_options.vte_lod_mid_max_distance,
                    render_options.vte_display_mode.label(),
                    render_options.vte_slice_layer,
                    render_options.vte_thick_half_width,
                    render_options.vte_reference_compare,
                    render_options.vte_reference_mismatch_only,
                    render_options.vte_compare_slice_only,
                    render_options.vte_y_slice_lookup_cache,
                    vte_lod_tint_enabled(),
                    vte_entity_linear_only_enabled(),
                    vte_entity_bvh_compare_enabled(),
                    storage_layers,
                    logical_layers,
                );
                self.vte_backend_notice_printed = true;
            }
        } else {
            // Non-VTE passes reuse the same tetra/BVH buffers for other pipelines.
            // Force non-voxel reprovisioning when VTE is re-enabled.
            self.vte_non_voxel_scene_hash = 0;
            self.clear_vte_compare_diagnostics();
        }

        let reduced_storage_supported =
            do_voxel_vte && render_options.vte_display_mode == VteDisplayMode::Integral;
        self.profiler.record_scene_stats(
            do_voxel_vte,
            vte_chunk_count,
            vte_visible_chunk_count,
            vte_visible_lod_counts,
            vte_y_slice_count,
            vte_y_slice_lookup_entry_count,
            raster_tetrahedron_count,
            total_tetrahedron_count,
        );
        if storage_layers < logical_layers && !reduced_storage_supported {
            panic!(
                "pixel storage layers ({storage_layers}) are less than logical render layers ({logical_layers}); \
this reduced-storage configuration currently supports only '--backend voxel-traversal --vte-display-mode integral'."
            );
        }

        self.frames_in_flight[frame_idx].vte_compare_enabled =
            do_voxel_vte && (vte_compare_diagnostics_enabled || vte_entity_bvh_compare_enabled());

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
                let non_voxel_tetrahedron_count = total_tetrahedron_count;
                if non_voxel_tetrahedron_count == 0 {
                    vte_non_voxel_rebuild_needed = false;
                    vte_non_voxel_rebuild_reason = "empty";
                    self.vte_non_voxel_scene_hash = 0;
                } else {
                    // In overlay-raster mode, the shared tetra output buffer is reused by the
                    // overlay tet preprocess pass later in the frame. Force non-voxel
                    // reprovision every frame in that mode so Stage A never consumes stale
                    // non-voxel tetra/BVH data on the next frame.
                    vte_non_voxel_rebuild_needed =
                        do_raster || non_voxel_scene_hash != self.vte_non_voxel_scene_hash;
                    vte_non_voxel_rebuild_reason = match (
                        do_raster,
                        non_voxel_scene_hash != self.vte_non_voxel_scene_hash,
                    ) {
                        (true, true) => "overlay_raster+scene_hash",
                        (true, false) => "overlay_raster",
                        (false, true) => "scene_hash",
                        (false, false) => "unchanged",
                    };
                    if vte_non_voxel_rebuild_needed {
                        vte_non_voxel_rebuild_executed = true;
                        // Preprocess tetra non-voxel instances into world-space tetrahedra.
                        let vte_preprocess_push_data: [u32; 4] =
                            [0, non_voxel_tetrahedron_count as u32, 0, 0];
                        builder
                            .push_constants(
                                self.compute_pipeline.pipeline_layout.clone(),
                                0,
                                vte_preprocess_push_data,
                            )
                            .unwrap();
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.raytrace_pre_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([
                                (non_voxel_tetrahedron_count as u32 + 63) / 64u32,
                                1,
                                1,
                            ])
                        }
                        .unwrap();
                        {
                            let q = self.profiler.next_query_index("vte_non_voxel_preprocess");
                            unsafe {
                                builder.write_timestamp(
                                    self.frames_in_flight[frame_idx].query_pool.clone(),
                                    q,
                                    PipelineStage::AllCommands,
                                )
                            }
                            .unwrap();
                        }

                        if non_voxel_tetrahedron_count > vte::VTE_ENTITY_LINEAR_THRESHOLD_TETS {
                            // Build a non-voxel-only BVH used by the VTE Stage A tetra pass.
                            let n = non_voxel_tetrahedron_count as u32;
                            let n_pow2 = n.next_power_of_two();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_scene_bounds_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([1, 1, 1]) }.unwrap();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_morton_codes_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([(n_pow2 + 63) / 64u32, 1, 1]) }.unwrap();

                            let num_stages = n_pow2.trailing_zeros();
                            let local_stages = 6u32.min(num_stages);
                            let workgroups = (n_pow2 + 63) / 64;

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

                            for stage in local_stages..num_stages {
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

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_init_leaves_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([(n + 63) / 64u32, 1, 1]) }.unwrap();

                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_build_tree_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([(n + 63) / 64u32, 1, 1]) }.unwrap();

                            let num_internal_nodes =
                                non_voxel_tetrahedron_count.saturating_sub(1) as u32;
                            if num_internal_nodes > 0 {
                                builder
                                    .bind_pipeline_compute(
                                        self.compute_pipeline.bvh_link_parents_pipeline.clone(),
                                    )
                                    .unwrap();
                                unsafe { builder.dispatch([(num_internal_nodes + 63) / 64, 1, 1]) }
                                    .unwrap();
                            }
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([(n + 63) / 64u32, 1, 1]) }.unwrap();

                            {
                                let q = self.profiler.next_query_index("vte_non_voxel_bvh");
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

                        self.vte_non_voxel_scene_hash = non_voxel_scene_hash;
                    }
                }

                let fuse_integral_in_stage_a =
                    render_options.vte_display_mode == VteDisplayMode::Integral;
                {
                    let q = self.profiler.next_query_index("vte_stage_a_setup");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }
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
                    {
                        let q = self.profiler.next_query_index("vte_stage_b_setup");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }
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
                        (self.sized_buffers.render_dimensions[0] + 7) / 8,
                        (self.sized_buffers.render_dimensions[1] + 7) / 8,
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
                let raster_preprocess_tetrahedron_count = if do_voxel_vte {
                    raster_tetrahedron_count
                } else {
                    total_tetrahedron_count
                };
                let raster_preprocess_push_data: [u32; 4] = [
                    raster_instance_base as u32,
                    raster_preprocess_tetrahedron_count as u32,
                    0,
                    0,
                ];

                // Tetrahedron pre-raster
                // Reset atomic counter to 0 before clipping dispatch
                builder
                    .fill_buffer(self.sized_buffers.atomic_counter_buffer.clone(), 0u32)
                    .unwrap();
                {
                    let q = self.profiler.next_query_index("tet_counter_clear");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                builder
                    .push_constants(
                        self.compute_pipeline.pipeline_layout.clone(),
                        0,
                        raster_preprocess_push_data,
                    )
                    .unwrap();
                builder
                    .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pipeline.clone())
                    .unwrap();
                unsafe {
                    builder.dispatch([
                        (raster_preprocess_tetrahedron_count as u32 + 63) / 64u32,
                        1,
                        1,
                    ])
                }
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
                {
                    let q = self.profiler.next_query_index("tet_counter_copy");
                    unsafe {
                        builder.write_timestamp(
                            self.frames_in_flight[frame_idx].query_pool.clone(),
                            q,
                            PipelineStage::AllCommands,
                        )
                    }
                    .unwrap();
                }

                if do_tetrahedron_edges {
                    line_render_count = raster_preprocess_tetrahedron_count * 6;
                }
            }

            if do_raster {
                if !do_voxel_vte {
                    // Zero tile counts
                    builder
                        .fill_buffer(self.sized_buffers.tile_tet_counts_buffer.clone(), 0u32)
                        .unwrap();
                    {
                        let q = self.profiler.next_query_index("tet_bin_clear");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Bin tetrahedra into tiles
                    builder
                        .bind_pipeline_compute(self.compute_pipeline.bin_tets_pipeline.clone())
                        .unwrap();
                    unsafe {
                        builder.dispatch([
                            (self.sized_buffers.max_tetrahedrons as u32 + 63) / 64,
                            1,
                            1,
                        ])
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
                }

                // Tetrahedron pixel raster (tile-based)
                builder
                    .bind_pipeline_compute(self.compute_pipeline.tetrahedron_pixel_pipeline.clone())
                    .unwrap();
                let (raster_dispatch_push, raster_dispatch_dims) = if do_voxel_vte {
                    let region = self.vte_overlay_raster_region();
                    let overlay_work_w =
                        (region[2] + VTE_OVERLAY_RASTER_SCALE - 1) / VTE_OVERLAY_RASTER_SCALE;
                    let overlay_work_h =
                        (region[3] + VTE_OVERLAY_RASTER_SCALE - 1) / VTE_OVERLAY_RASTER_SCALE;
                    if self.frames_rendered == 0 {
                        println!(
                            "VTE overlay raster region: origin=({}, {}) size={}x{} (work {}x{} @{}x upsample)",
                            region[0], region[1], region[2], region[3], overlay_work_w, overlay_work_h, VTE_OVERLAY_RASTER_SCALE
                        );
                    }
                    (
                        region,
                        [(overlay_work_w + 7) / 8, (overlay_work_h + 7) / 8, 1],
                    )
                } else {
                    (
                        [0, 0, 0, 0],
                        [
                            (self.sized_buffers.render_dimensions[0] + 7) / 8,
                            (self.sized_buffers.render_dimensions[1] + 7) / 8,
                            1,
                        ],
                    )
                };
                builder
                    .push_constants(
                        self.compute_pipeline.pipeline_layout.clone(),
                        0,
                        raster_dispatch_push,
                    )
                    .unwrap();
                unsafe { builder.dispatch(raster_dispatch_dims) }.unwrap();
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

            if do_edges || custom_overlay_edge_used_instance_count > 0 {
                // Edge pre-raster supports dedicated instance ranges (for debug edges and
                // GPU-only custom overlay boxes) and appends into the shared line buffer.
                let model_edge_count = self.one_time_buffers.model_edge_count;
                let max_lines = LINE_VERTEX_CAPACITY / 2;
                let mut dispatched_any_edges = false;

                builder
                    .bind_pipeline_compute(self.compute_pipeline.edge_pipeline.clone())
                    .unwrap();

                if model_edge_count > 0 {
                    if do_edges {
                        let available_lines = max_lines.saturating_sub(line_render_count);
                        let max_instances = available_lines / model_edge_count;
                        let edge_instance_count = used_instance_count.min(max_instances);
                        if edge_instance_count > 0 {
                            let edge_line_count = edge_instance_count * model_edge_count;
                            let edge_push_data: [u32; 4] = [
                                0,
                                edge_instance_count.min(u32::MAX as usize) as u32,
                                line_render_count.min(u32::MAX as usize) as u32,
                                0,
                            ];
                            builder
                                .push_constants(
                                    self.compute_pipeline.pipeline_layout.clone(),
                                    0,
                                    edge_push_data,
                                )
                                .unwrap();
                            unsafe {
                                builder.dispatch([((edge_line_count as u32) + 63) / 64u32, 1, 1])
                            }
                            .unwrap();
                            line_render_count += edge_line_count;
                            dispatched_any_edges = true;
                        }
                    }

                    if custom_overlay_edge_used_instance_count > 0 {
                        let available_lines = max_lines.saturating_sub(line_render_count);
                        let max_instances = available_lines / model_edge_count;
                        let edge_overlay_instance_count =
                            custom_overlay_edge_used_instance_count.min(max_instances);
                        if edge_overlay_instance_count > 0 {
                            let edge_line_count = edge_overlay_instance_count * model_edge_count;
                            let edge_push_data: [u32; 4] = [
                                custom_overlay_edge_instance_base.min(u32::MAX as usize) as u32,
                                edge_overlay_instance_count.min(u32::MAX as usize) as u32,
                                line_render_count.min(u32::MAX as usize) as u32,
                                0,
                            ];
                            builder
                                .push_constants(
                                    self.compute_pipeline.pipeline_layout.clone(),
                                    0,
                                    edge_push_data,
                                )
                                .unwrap();
                            unsafe {
                                builder.dispatch([((edge_line_count as u32) + 63) / 64u32, 1, 1])
                            }
                            .unwrap();
                            line_render_count += edge_line_count;
                            dispatched_any_edges = true;
                        }
                    }
                }

                if dispatched_any_edges {
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
                            (self.sized_buffers.render_dimensions[0] + 7) / 8,
                            (self.sized_buffers.render_dimensions[1] + 7) / 8,
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
                    let raytrace_preprocess_push_data: [u32; 4] =
                        [0, total_tetrahedron_count as u32, 0, 0];
                    builder
                        .push_constants(
                            self.compute_pipeline.pipeline_layout.clone(),
                            0,
                            raytrace_preprocess_push_data,
                        )
                        .unwrap();
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

                        // 2f. Link parent pointers for leaf-to-root propagation.
                        let num_internal_nodes = total_tetrahedron_count.saturating_sub(1) as u32;
                        if num_internal_nodes > 0 {
                            builder
                                .bind_pipeline_compute(
                                    self.compute_pipeline.bvh_link_parents_pipeline.clone(),
                                )
                                .unwrap();
                            unsafe { builder.dispatch([(num_internal_nodes + 63) / 64, 1, 1]) }
                                .unwrap();
                        }

                        // 2g. Compute leaf AABBs and propagate all parent bounds in one pass.
                        builder
                            .bind_pipeline_compute(
                                self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone(),
                            )
                            .unwrap();
                        unsafe {
                            builder.dispatch([(total_tetrahedron_count as u32 + 63) / 64u32, 1, 1])
                        }
                        .unwrap();
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

        let prev_used_non_voxel = self.vte_entity_diag_prev_used_non_voxel;
        let prev_tets_non_voxel = self.vte_entity_diag_prev_tets_non_voxel;
        let used_non_voxel_went_zero = do_voxel_vte
            && prev_used_non_voxel
                .map(|prev| prev > 0 && non_voxel_used_instance_count == 0)
                .unwrap_or(false);
        let tets_non_voxel_went_zero = do_voxel_vte
            && prev_tets_non_voxel
                .map(|prev| prev > 0 && total_tetrahedron_count == 0)
                .unwrap_or(false);
        if do_voxel_vte {
            self.vte_entity_diag_prev_used_non_voxel = Some(non_voxel_used_instance_count);
            self.vte_entity_diag_prev_tets_non_voxel = Some(total_tetrahedron_count);
        } else {
            self.vte_entity_diag_prev_used_non_voxel = None;
            self.vte_entity_diag_prev_tets_non_voxel = None;
        }

        let vte_entity_diag_anomaly = do_voxel_vte
            && (dropped_model_instance_count > 0
                || non_voxel_outlier_count > 0
                || used_non_voxel_went_zero
                || tets_non_voxel_went_zero
                || (model_instances_input.len() > 0 && non_voxel_used_instance_count == 0)
                || (non_voxel_used_instance_count > 0 && total_tetrahedron_count == 0));
        let vte_entity_diag_periodic_due = self
            .vte_entity_diag_last_log_frame
            .map(|last| {
                self.frames_rendered.saturating_sub(last) >= self.vte_entity_diag_interval.max(1)
            })
            .unwrap_or(true);
        let vte_entity_diag_copy_requested = self.vte_entity_diag_enabled
            && self.vte_entity_diag_bvh_readback
            && do_voxel_vte
            && total_tetrahedron_count > 0
            && (self.vte_entity_diag_verbose
                || vte_entity_diag_anomaly
                || self.frames_rendered % self.vte_entity_diag_interval.max(1) == 0);
        self.frames_in_flight[frame_idx].vte_entity_diag_copy_scheduled =
            vte_entity_diag_copy_requested;
        self.frames_in_flight[frame_idx].vte_entity_diag_non_voxel_tet_count = if do_voxel_vte {
            total_tetrahedron_count
        } else {
            0
        };
        if vte_entity_diag_copy_requested {
            if self.vte_entity_diag_bvh_topology {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.bvh_nodes_buffer.clone(),
                        self.sized_buffers.cpu_bvh_nodes_buffer.clone(),
                    ))
                    .unwrap();
            } else {
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.bvh_nodes_buffer.clone(),
                        self.sized_buffers.cpu_bvh_root_buffer.clone(),
                    ))
                    .unwrap();
            }
        }

        if self.vte_entity_diag_enabled
            && (self.vte_entity_diag_verbose
                || vte_entity_diag_anomaly
                || (do_voxel_vte && vte_entity_diag_periodic_due))
        {
            eprintln!(
                "[vte-entity-diag] frame={} backend={} mode={} input_non_voxel={} input_overlay={} used_non_voxel={} dropped_non_finite={} tets_non_voxel={} tets_overlay={} do_raster={} prev_hash=0x{:016x} hash=0x{:016x} rebuild_needed={} rebuild_executed={} rebuild_reason={} max_abs_translation={:.2} max_abs_basis={:.2} outlier_instances={}",
                self.frames_rendered,
                self.last_backend.label(),
                render_options.vte_display_mode.label(),
                model_instances_input.len(),
                raster_overlay_instances_input.len(),
                non_voxel_used_instance_count,
                dropped_model_instance_count,
                total_tetrahedron_count,
                raster_overlay_tetrahedron_count,
                do_raster,
                previous_vte_non_voxel_scene_hash,
                non_voxel_scene_hash,
                vte_non_voxel_rebuild_needed,
                vte_non_voxel_rebuild_executed,
                vte_non_voxel_rebuild_reason,
                non_voxel_translation_abs_max,
                non_voxel_basis_abs_max,
                non_voxel_outlier_count
            );
            if self.vte_compare_stats.entity_bvh_samples > 0 {
                eprintln!(
                    "[vte-entity-diag][bvh-compare] frame={} samples={} mismatches={} hit_state={} material={} distance={} tetra={} bvh_miss_linear_hit={} bvh_hit_linear_miss={} noprune_mismatches={} noprune_hit_state={} noprune_distance={} noprune_tetra={} noaabb_mismatches={} noaabb_hit_state={} noaabb_distance={} noaabb_tetra={} linear_order_mismatches={} linear_order_hit_state={} linear_order_distance={} linear_order_tetra={} leafarray_mismatches={} leafarray_hit_state={} leafarray_distance={} leafarray_tetra={}",
                    self.frames_rendered,
                    self.vte_compare_stats.entity_bvh_samples,
                    self.vte_compare_stats.entity_bvh_mismatches,
                    self.vte_compare_stats.entity_bvh_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_material_mismatches,
                    self.vte_compare_stats.entity_bvh_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_miss_linear_hit,
                    self.vte_compare_stats.entity_bvh_hit_linear_miss,
                    self.vte_compare_stats.entity_bvh_noprune_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_noprune_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_noaabb_tetra_mismatches,
                    self.vte_compare_stats.entity_linear_order_mismatches,
                    self.vte_compare_stats.entity_linear_order_hit_state_mismatches,
                    self.vte_compare_stats.entity_linear_order_distance_mismatches,
                    self.vte_compare_stats.entity_linear_order_tetra_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_hit_state_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_distance_mismatches,
                    self.vte_compare_stats.entity_bvh_leafarray_tetra_mismatches,
                );
            }
            self.vte_entity_diag_last_log_frame = Some(self.frames_rendered);
        }
        if self.vte_entity_diag_enabled && (used_non_voxel_went_zero || tets_non_voxel_went_zero) {
            eprintln!(
                "[vte-entity-diag][transition-zero] frame={} backend={} mode={} used_non_voxel:{}->{} tets_non_voxel:{}->{} input_non_voxel={} input_overlay={} dropped_non_finite={} do_raster={} prev_hash=0x{:016x} hash=0x{:016x} rebuild_needed={} rebuild_executed={} rebuild_reason={}",
                self.frames_rendered,
                self.last_backend.label(),
                render_options.vte_display_mode.label(),
                prev_used_non_voxel.unwrap_or(0),
                non_voxel_used_instance_count,
                prev_tets_non_voxel.unwrap_or(0),
                total_tetrahedron_count,
                model_instances_input.len(),
                raster_overlay_instances_input.len(),
                dropped_model_instance_count,
                do_raster,
                previous_vte_non_voxel_scene_hash,
                non_voxel_scene_hash,
                vte_non_voxel_rebuild_needed,
                vte_non_voxel_rebuild_executed,
                vte_non_voxel_rebuild_reason
            );
        }

        let mut hud_vertex_count = 0usize;
        let mut hud_batches: Vec<HudDrawBatch> = Vec::new();
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
                render_options.hud_readout_mode,
                render_options.hud_rotation_label.as_deref(),
                render_options.hud_target_hit_voxel,
                render_options.hud_target_hit_face,
                &render_options.hud_player_tags,
                render_options.waila_text.as_deref(),
            );
            line_render_count += hud_line_count;
            hud_vertex_count = hud_quad_count;
            if hud_quad_count > 0 {
                hud_batches.push(HudDrawBatch {
                    first_vertex: 0,
                    vertex_count: hud_quad_count as u32,
                    scissor: self.full_hud_scissor(),
                    texture_slot: HudTextureSlot::Hud,
                });
            }
        }

        if let Some(egui_paint) = render_options.egui_paint.as_ref() {
            if !egui_paint.texture_updates.is_empty() {
                self.apply_egui_texture_updates(queue.clone(), &egui_paint.texture_updates);
            }
            let (egui_vertex_count, mut egui_batches) =
                self.write_egui_overlay(frame_idx, hud_vertex_count, &egui_paint.meshes);
            hud_vertex_count += egui_vertex_count;
            hud_batches.append(&mut egui_batches);
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
                    {
                        let q = self.profiler.next_query_index("present_begin");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render from compute shader buffer
                    {
                        builder
                            .bind_pipeline_graphics(present_pipeline.buffer_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(6, 1, 0, 0) }.unwrap();
                    }
                    {
                        let q = self.profiler.next_query_index("present_buffer");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render the edge lines
                    if line_render_count > 0 {
                        builder
                            .bind_pipeline_graphics(present_pipeline.line_pipeline.clone())
                            .unwrap();
                        unsafe { builder.draw(line_render_count as u32 * 2, 1, 0, 0) }.unwrap();
                    }
                    {
                        let q = self.profiler.next_query_index("present_lines");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // Render HUD quads (text + panels, alpha-blended on top)
                    if hud_vertex_count > 0 {
                        builder
                            .bind_pipeline_graphics(present_pipeline.hud_pipeline.clone())
                            .unwrap();

                        let mut bound_texture_slot: Option<HudTextureSlot> = None;
                        for batch in &hud_batches {
                            let frame = &self.frames_in_flight[frame_idx];
                            let descriptor_set = match batch.texture_slot {
                                HudTextureSlot::Hud => frame.hud_descriptor_set.as_ref(),
                                HudTextureSlot::EguiAtlas => frame.egui_descriptor_set.as_ref(),
                                HudTextureSlot::MaterialIcons => frame
                                    .material_icons_descriptor_set
                                    .as_ref()
                                    .or(frame.egui_descriptor_set.as_ref()),
                            };
                            let Some(descriptor_set) = descriptor_set else {
                                continue;
                            };

                            if bound_texture_slot != Some(batch.texture_slot) {
                                builder
                                    .bind_descriptor_sets(
                                        PipelineBindPoint::Graphics,
                                        present_pipeline.hud_pipeline_layout.clone(),
                                        0,
                                        vec![descriptor_set.clone()],
                                    )
                                    .unwrap();
                                bound_texture_slot = Some(batch.texture_slot);
                            }

                            builder
                                .set_scissor(0, [batch.scissor].into_iter().collect())
                                .unwrap();
                            builder
                                .push_constants(
                                    present_pipeline.hud_pipeline_layout.clone(),
                                    0,
                                    [batch.first_vertex],
                                )
                                .unwrap();
                            unsafe { builder.draw(batch.vertex_count, 1, 0, 0) }.unwrap();
                        }
                    }
                    {
                        let q = self.profiler.next_query_index("present_hud");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }

                    // End render pass
                    builder
                        // We leave the render pass. Note that if we had multiple subpasses we could
                        // have called `next_subpass` to jump to the next subpass.
                        .end_render_pass(Default::default())
                        .unwrap();
                    {
                        let q = self.profiler.next_query_index("present_end");
                        unsafe {
                            builder.write_timestamp(
                                self.frames_in_flight[frame_idx].query_pool.clone(),
                                q,
                                PipelineStage::AllCommands,
                            )
                        }
                        .unwrap();
                    }
                    if render_options.take_framebuffer_screenshot {
                        builder
                            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                                framebuffers[image_index as usize].attachments()[0]
                                    .image()
                                    .clone(),
                                self.cpu_screen_capture_buffer.clone(),
                            ))
                            .unwrap();
                        {
                            let q = self.profiler.next_query_index("present_screenshot_copy");
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
            }
        }

        if render_options.prepare_render_screenshot {
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.sized_buffers.output_pixel_buffer.clone(),
                    self.sized_buffers.output_cpu_pixel_buffer.clone(),
                ))
                .unwrap();
            {
                let q = self.profiler.next_query_index("render_screenshot_copy");
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
                if self.drop_next_profile_sample {
                    self.drop_next_profile_sample = false;
                } else {
                    self.profiler.read_results_and_accumulate(
                        &self.frames_in_flight[prev_idx].query_pool,
                        clipped_count,
                    );
                }

                if self.vte_entity_diag_enabled
                    && self.vte_entity_diag_bvh_readback
                    && self.frames_in_flight[prev_idx].vte_entity_diag_copy_scheduled
                {
                    let prev_non_voxel_tets =
                        self.frames_in_flight[prev_idx].vte_entity_diag_non_voxel_tet_count;
                    if prev_non_voxel_tets > 0 {
                        if self.vte_entity_diag_bvh_topology {
                            if let Ok(bvh_data) = self.sized_buffers.cpu_bvh_nodes_buffer.read() {
                                if let Some(root) = bvh_data.get(0) {
                                    let finite = root.min_bounds.x.is_finite()
                                        && root.min_bounds.y.is_finite()
                                        && root.min_bounds.z.is_finite()
                                        && root.min_bounds.w.is_finite()
                                        && root.max_bounds.x.is_finite()
                                        && root.max_bounds.y.is_finite()
                                        && root.max_bounds.z.is_finite()
                                        && root.max_bounds.w.is_finite();
                                    let ordered = root.min_bounds.x <= root.max_bounds.x
                                        && root.min_bounds.y <= root.max_bounds.y
                                        && root.min_bounds.z <= root.max_bounds.z
                                        && root.min_bounds.w <= root.max_bounds.w;
                                    if let Some(summary) =
                                        summarize_bvh_topology(&bvh_data, prev_non_voxel_tets)
                                    {
                                        let root_child_valid = root.is_leaf != 0
                                            || (root.left_child < summary.total_nodes as u32
                                                && root.right_child < summary.total_nodes as u32);
                                        let root_ready =
                                            root.is_leaf != 0 || root.atomic_visit_count >= 2;
                                        let topology_anomaly = !finite
                                            || !ordered
                                            || !root_child_valid
                                            || !root_ready
                                            || summary.invalid_child_edges > 0
                                            || summary.self_child_edges > 0
                                            || summary.nodes_with_multiple_parents > 0
                                            || summary.nodes_without_parent_excluding_root > 0
                                            || summary.unreachable_internal_nodes > 0
                                            || summary.unreachable_leaf_nodes > 0
                                            || summary.leaf_invalid_tetra_indices > 0
                                            || summary.leaf_duplicate_tetra_indices > 0
                                            || summary.leaf_missing_tetra_indices > 0
                                            || summary.internal_ready < summary.internal_nodes;
                                        if self.vte_entity_diag_verbose || topology_anomaly {
                                            eprintln!(
                                                "[vte-entity-diag][bvh-topology] frame={} prev_frame={} tets={} total_nodes={} internal_ready={}/{} root_ready={} root_finite={} root_ordered={} root_child_valid={} invalid_child_edges={} self_child_edges={} no_parent_ex_root={} multi_parent={} unreachable_internal={} unreachable_leaf={} leaf_invalid_tet={} leaf_duplicate_tet={} leaf_missing_tet={} root(is_leaf={}, visit={}, left={}, right={}) min=({:.3},{:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3},{:.3})",
                                                self.frames_rendered,
                                                self.frames_rendered.saturating_sub(1),
                                                prev_non_voxel_tets,
                                                summary.total_nodes,
                                                summary.internal_ready,
                                                summary.internal_nodes,
                                                root_ready,
                                                finite,
                                                ordered,
                                                root_child_valid,
                                                summary.invalid_child_edges,
                                                summary.self_child_edges,
                                                summary.nodes_without_parent_excluding_root,
                                                summary.nodes_with_multiple_parents,
                                                summary.unreachable_internal_nodes,
                                                summary.unreachable_leaf_nodes,
                                                summary.leaf_invalid_tetra_indices,
                                                summary.leaf_duplicate_tetra_indices,
                                                summary.leaf_missing_tetra_indices,
                                                root.is_leaf,
                                                root.atomic_visit_count,
                                                root.left_child,
                                                root.right_child,
                                                root.min_bounds.x,
                                                root.min_bounds.y,
                                                root.min_bounds.z,
                                                root.min_bounds.w,
                                                root.max_bounds.x,
                                                root.max_bounds.y,
                                                root.max_bounds.z,
                                                root.max_bounds.w,
                                            );
                                        }
                                    }
                                }
                            }
                        } else if let Ok(root_data) = self.sized_buffers.cpu_bvh_root_buffer.read()
                        {
                            if let Some(root) = root_data.get(0) {
                                let finite = root.min_bounds.x.is_finite()
                                    && root.min_bounds.y.is_finite()
                                    && root.min_bounds.z.is_finite()
                                    && root.min_bounds.w.is_finite()
                                    && root.max_bounds.x.is_finite()
                                    && root.max_bounds.y.is_finite()
                                    && root.max_bounds.z.is_finite()
                                    && root.max_bounds.w.is_finite();
                                let ordered = root.min_bounds.x <= root.max_bounds.x
                                    && root.min_bounds.y <= root.max_bounds.y
                                    && root.min_bounds.z <= root.max_bounds.z
                                    && root.min_bounds.w <= root.max_bounds.w;
                                let total_nodes =
                                    prev_non_voxel_tets.saturating_mul(2).saturating_sub(1) as u32;
                                let child_valid = root.is_leaf != 0
                                    || (root.left_child < total_nodes
                                        && root.right_child < total_nodes);
                                if self.vte_entity_diag_verbose
                                    || !finite
                                    || !ordered
                                    || !child_valid
                                {
                                    eprintln!(
                                        "[vte-entity-diag] frame={} prev_frame={} bvh_root finite={} ordered={} child_valid={} is_leaf={} left={} right={} tets={} min=({:.3},{:.3},{:.3},{:.3}) max=({:.3},{:.3},{:.3},{:.3})",
                                        self.frames_rendered,
                                        self.frames_rendered.saturating_sub(1),
                                        finite,
                                        ordered,
                                        child_valid,
                                        root.is_leaf,
                                        root.left_child,
                                        root.right_child,
                                        prev_non_voxel_tets,
                                        root.min_bounds.x,
                                        root.min_bounds.y,
                                        root.min_bounds.z,
                                        root.min_bounds.w,
                                        root.max_bounds.x,
                                        root.max_bounds.y,
                                        root.max_bounds.z,
                                        root.max_bounds.w,
                                    );
                                }
                            }
                        }
                    }
                }
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

                let _ = std::fs::create_dir_all("frames");
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
                                        "  VTE mode={} slice_layer={:?} thick_half_width={} max_steps={} max_distance={:.1} lod_near={:.1} lod_mid={:.1} reference_compare={} mismatch_only={} compare_slice_only={} yslice_fastpath=true chunk_solid_clip=true yslice_lookup_cache={} lod_tint={}",
                                        render_options.vte_display_mode.label(),
                                        render_options.vte_slice_layer,
                                        render_options.vte_thick_half_width,
                                        render_options.vte_max_trace_steps,
                                        render_options.vte_max_trace_distance,
                                        render_options.vte_lod_near_max_distance,
                                        render_options.vte_lod_mid_max_distance,
                                        render_options.vte_reference_compare,
                                        render_options.vte_reference_mismatch_only,
                                        render_options.vte_compare_slice_only,
                                        render_options.vte_y_slice_lookup_cache,
                                        vte_lod_tint_enabled(),
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
}

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
