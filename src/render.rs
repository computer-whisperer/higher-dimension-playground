use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use winit::window::Window;
use vulkano::swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::{DescriptorSetAllocator, StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::{Device, Queue};
use vulkano::image::{Image, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{PolygonMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::{sync, Validated, VulkanError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::sync::GpuFuture;
use bytemuck::{Zeroable};
use exr::prelude::{ImageAttributes, WritableImage};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use common::{get_normal, ModelTetrahedron};
use crate::hypercube::{generate_simplexes_for_k_face_3, Hypercube};
use glam::{Vec4, Vec2, UVec3, Vec4Swizzles};
use image::{ImageBuffer, Rgb, Rgba};
use vulkano::format::Format::{R8G8B8A8_UNORM};
use winit::dpi::PhysicalSize;

/// Collection of all shader modules loaded from Slang-compiled SPIR-V
struct ShaderModules {
    // Present shaders
    line_vs: Arc<ShaderModule>,
    line_fs: Arc<ShaderModule>,
    buffer_vs: Arc<ShaderModule>,
    buffer_fs: Arc<ShaderModule>,
    // Compute shaders
    tetrahedron_cs: Arc<ShaderModule>,
    edge_cs: Arc<ShaderModule>,
    tetrahedron_pixel_cs: Arc<ShaderModule>,
    raytrace_preprocess: Arc<ShaderModule>,
    raytrace_pixel: Arc<ShaderModule>,
    raytrace_clear: Arc<ShaderModule>,
    // BVH compute shaders
    bvh_scene_bounds: Arc<ShaderModule>,
    bvh_morton_codes: Arc<ShaderModule>,
    bvh_bitonic_sort: Arc<ShaderModule>,
    bvh_init_leaves: Arc<ShaderModule>,
    bvh_build_tree: Arc<ShaderModule>,
    bvh_compute_leaf_aabbs: Arc<ShaderModule>,
    bvh_propagate_aabbs: Arc<ShaderModule>,
}

pub struct RenderOptions {
    pub do_frame_clear: bool,
    pub do_raster: bool,
    pub do_raytrace: bool,
    pub do_edges: bool,
    pub do_tetrahedron_edges: bool,
    pub take_framebuffer_screenshot: bool,
    pub prepare_render_screenshot: bool
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            do_frame_clear: false,
            do_raster: true,
            do_raytrace: false,
            do_edges: false,
            do_tetrahedron_edges: false,
            take_framebuffer_screenshot: false,
            prepare_render_screenshot: false,
        }
    }
}

pub fn generate_tesseract_tetrahedrons() -> Vec<ModelTetrahedron> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let cube_vertexes = Hypercube::<3, usize>::generate_vertices();
    let tetrahedron_cells = Hypercube::<4, usize>::generate_k_faces_3();

    let mut output_tetrahedrons = Vec::new();
    let texture_position_simplexes = generate_simplexes_for_k_face_3::<3>([0b000, 0b001, 0b010, 0b100]);

    for cell_id in 0..tetrahedron_cells.len() {
        //for cell_id in 0..1 {
        let position_simplexes = generate_simplexes_for_k_face_3::<4>(tetrahedron_cells[cell_id]);


        for simplex_id in 0..position_simplexes.len() {
            let texture_simplex = texture_position_simplexes[simplex_id];
            
            // Convert arrays to vec4
            let mut vertex_positions = position_simplexes[simplex_id].map(|i| {
                let vertex = tesseract_vertexes[i];
                glam::Vec4::new(vertex[0] as f32, vertex[1] as f32, vertex[2] as f32, vertex[3] as f32)
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
                (vertex_positions[3] - vertex_positions[0]).into()
            ]).into();
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
            
            output_tetrahedrons.push(
                common::ModelTetrahedron {
                    vertex_positions,
                    texture_positions,
                    cell_id: cell_id as u32,
                    padding: [0; 3],
                }
            )
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
            glam::Vec4::new(vertex[0] as f32, vertex[1] as f32, vertex[2] as f32, vertex[3] as f32)
        });        
        
        output_edges.push(common::ModelEdge { vertex_positions});
    }

    output_edges
}


pub struct OneTimeBuffers {
    model_tetrahedron_count: usize,
    model_tetrahedron_buffer: Subbuffer<[common::ModelTetrahedron]>,
    model_edge_count: usize,
    model_edge_buffer: Subbuffer<[common::ModelEdge]>,
    descriptor_set: Arc<DescriptorSet>
}

impl OneTimeBuffers {
    pub fn new(memory_allocator: Arc<dyn MemoryAllocator>, descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>, descriptor_set_layout: Arc<DescriptorSetLayout>) -> Self {
        let model_tetrahedrons = generate_tesseract_tetrahedrons();
        let model_tetrahedron_count = model_tetrahedrons.len();
        let model_tetrahedron_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model_tetrahedrons
        ).unwrap();

        let model_edges = generate_tesseract_edges();
        let model_edge_count = model_edges.len();
        let model_edge_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model_edges
        ).unwrap();
        
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, model_tetrahedron_buffer.clone()),
                WriteDescriptorSet::buffer(1, model_edge_buffer.clone()),
            ],
            [],
        ).unwrap();
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
            });
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            }
        ).unwrap()
    }
}

pub struct SizedBuffers {
    render_dimensions: [u32; 3],
    max_tetrahedrons: usize,
    line_vertexes_buffer: Subbuffer<[Vec2]>,
    output_tetrahedron_buffer: Subbuffer<[common::Tetrahedron]>,
    output_pixel_buffer: Subbuffer<[Vec4]>,
    morton_codes_buffer: Subbuffer<[common::MortonCode]>,
    bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    scene_bounds_buffer: Subbuffer<[common::SceneBounds]>,
    descriptor_set: Arc<DescriptorSet>,
    output_cpu_pixel_buffer: Subbuffer<[Vec4]>,
    // Debug readback buffers
    cpu_bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    cpu_morton_codes_buffer: Subbuffer<[common::MortonCode]>,
}

impl SizedBuffers {
    pub fn new(memory_allocator: Arc<dyn MemoryAllocator>, descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>, descriptor_set_layout: Arc<DescriptorSetLayout>, render_dimensions: [u32; 3]) -> Self {
        let max_tetrahedrons: usize = 10000;

        let line_vertexes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
               ..Default::default()
            },
            vec![glam::Vec2::new(0.0, 0.0); 100000]
        ).unwrap();

        let output_tetrahedron_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::Tetrahedron::zeroed(); max_tetrahedrons]
        ).unwrap();

        let output_pixel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![Vec4::ZERO; (render_dimensions[0]*render_dimensions[1]*render_dimensions[2]) as usize]
        ).unwrap();

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
            vec![Vec4::ZERO; (render_dimensions[0]*render_dimensions[1]*render_dimensions[2]) as usize]
        ).unwrap();

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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::MortonCode { code: u64::MAX, tetrahedron_index: 0xFFFFFFFF, padding: 0 }; morton_codes_padded_count]
        ).unwrap();

        // BVH needs 2N-1 nodes (N leaves + N-1 internal nodes)
        let bvh_node_count = 2 * max_tetrahedrons;
        let bvh_nodes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::BVHNode::zeroed(); bvh_node_count]
        ).unwrap();

        let scene_bounds_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![common::SceneBounds::zeroed(); 1]
        ).unwrap();

        // CPU readback buffers for BVH debugging
        let cpu_bvh_nodes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![common::BVHNode::zeroed(); bvh_node_count]
        ).unwrap();

        let cpu_morton_codes_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            vec![common::MortonCode::zeroed(); morton_codes_padded_count]
        ).unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, line_vertexes_buffer.clone()),
                WriteDescriptorSet::buffer(1, output_tetrahedron_buffer.clone()),
                WriteDescriptorSet::buffer(2, output_pixel_buffer.clone()),
                WriteDescriptorSet::buffer(3, morton_codes_buffer.clone()),
                WriteDescriptorSet::buffer(4, bvh_nodes_buffer.clone()),
                WriteDescriptorSet::buffer(5, scene_bounds_buffer.clone()),
            ],
            []
        ).unwrap();
        Self {
            render_dimensions,
            max_tetrahedrons,
            line_vertexes_buffer,
            output_tetrahedron_buffer,
            output_pixel_buffer,
            morton_codes_buffer,
            bvh_nodes_buffer,
            scene_bounds_buffer,
            output_cpu_pixel_buffer,
            cpu_bvh_nodes_buffer,
            cpu_morton_codes_buffer,
            descriptor_set
        }
    }

    pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        bindings.insert(
            2,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        // BVH buffers
        bindings.insert(
            3,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        bindings.insert(
            4,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        bindings.insert(
            5,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            }
        ).unwrap()
    }

}

pub struct LiveBuffers {
    model_instance_buffer: Subbuffer<[common::ModelInstance]>,
    working_data_buffer: Subbuffer<common::WorkingData>,
    descriptor_set: Arc<DescriptorSet>
}

impl LiveBuffers {
    pub fn new(memory_allocator: Arc<dyn MemoryAllocator>, descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>, descriptor_set_layout: Arc<DescriptorSetLayout>) -> Self {
        let model_instance_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
               ..Default::default()
            },
            vec![common::ModelInstance::zeroed(); 1000]
        ).unwrap();

        let working_data_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
               .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
               ..Default::default()
            },
            common::WorkingData::zeroed()
        ).unwrap();
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, model_instance_buffer.clone()),
                WriteDescriptorSet::buffer(1, working_data_buffer.clone())
            ],
            [],
        ).unwrap();
        Self {
            model_instance_buffer,
            working_data_buffer,
            descriptor_set
        }
    }

    pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
            });
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            }
        ).unwrap()
    }
}

struct PresentPipelineContext {
    line_pipeline: Arc<GraphicsPipeline>,
    buffer_pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>
}

impl PresentPipelineContext {
    pub fn new(device: Arc<Device>, render_pass: Arc<RenderPass>, shaders: &ShaderModules, pipeline_layout: Arc<PipelineLayout>) -> Self {

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

        Self {
            line_pipeline,
            buffer_pipeline,
            pipeline_layout
        }
    }
}

struct ComputePipelineContext {
    tetrahedron_pipeline: Arc<ComputePipeline>,
    edge_pipeline: Arc<ComputePipeline>,
    tetrahedron_pixel_pipeline: Arc<ComputePipeline>,
    raytrace_pre_pipeline: Arc<ComputePipeline>,
    raytrace_pixel_pipeline: Arc<ComputePipeline>,
    raytrace_clear_pipeline: Arc<ComputePipeline>,
    // BVH pipelines
    bvh_scene_bounds_pipeline: Arc<ComputePipeline>,
    bvh_morton_codes_pipeline: Arc<ComputePipeline>,
    bvh_bitonic_sort_pipeline: Arc<ComputePipeline>,
    bvh_init_leaves_pipeline: Arc<ComputePipeline>,
    bvh_build_tree_pipeline: Arc<ComputePipeline>,
    bvh_compute_leaf_aabbs_pipeline: Arc<ComputePipeline>,
    bvh_propagate_aabbs_pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>
}

impl ComputePipelineContext {
    pub fn new(device: Arc<Device>, shaders: &ShaderModules, pipeline_layout: Arc<PipelineLayout>) -> Self {
        Self {
            tetrahedron_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.tetrahedron_cs.entry_point("mainTetrahedronCS").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            edge_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.edge_cs.entry_point("mainEdgeCS").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            tetrahedron_pixel_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.tetrahedron_pixel_cs.entry_point("mainTetrahedronPixelCS").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_pre_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.raytrace_preprocess.entry_point("mainRaytracerTetrahedronPreprocessor").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_pixel_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.raytrace_pixel.entry_point("mainRaytracerPixel").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_clear_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.raytrace_clear.entry_point("mainRaytracerClear").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            // BVH pipelines
            bvh_scene_bounds_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_scene_bounds.entry_point("mainBVHSceneBounds").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_morton_codes_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_morton_codes.entry_point("mainBVHMortonCodes").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_bitonic_sort_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_bitonic_sort.entry_point("mainBVHBitonicSort").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_init_leaves_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_init_leaves.entry_point("mainBVHInitLeaves").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_build_tree_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_build_tree.entry_point("mainBVHBuildTree").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_compute_leaf_aabbs_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_compute_leaf_aabbs.entry_point("mainBVHComputeLeafAABBs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            bvh_propagate_aabbs_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shaders.bvh_propagate_aabbs.entry_point("mainBVHPropagateAABBs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            pipeline_layout
        }
    }
}


pub struct RenderContext {
    pub(crate) window: Option<Arc<Window>>,
    swapchain: Option<Arc<Swapchain>>,
    render_pass: Option<Arc<RenderPass>>,
    framebuffers: Option<Vec<Arc<Framebuffer>>>,
    present_pipeline: Option<PresentPipelineContext>,
    compute_pipeline: ComputePipelineContext,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    one_time_buffers: OneTimeBuffers,
    sized_buffers: SizedBuffers,
    live_buffers: LiveBuffers,
    cpu_screen_capture_buffer: Subbuffer<[u8]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    frames_rendered: usize
}


impl RenderContext {
    pub fn new(device: Arc<Device>, instance: Arc<Instance>, window: Option<Arc<Window>>, render_dimensions: [u32; 3]) -> RenderContext {

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));



        let (surface, window_size) = match window.clone() {
            Some(window) => (Some(Surface::from_window(instance.clone(), window.clone()).unwrap()), window.inner_size()),
            None => (None, PhysicalSize{ width: render_dimensions[0], height: render_dimensions[1] }),
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

                let image_format = image_formats[0].0;
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
            None => {
                (None, None)
            }
        };


        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        
        // Load shaders at runtime from embedded SPIR-V bytes
        // (vulkano_shaders macro can't parse Slang's SPIR-V 1.4 output)
        fn load_shader(device: Arc<Device>, spirv: &[u8]) -> Arc<ShaderModule> {
            unsafe {
                ShaderModule::from_bytes(device, spirv)
                    .expect("Failed to load shader module")
            }
        }

        let spirv_dir = std::path::Path::new(env!("SPIRV_OUT_DIR"));

        let raytrace_pixel = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerPixel.spv")).expect("Failed to read shader"));
        let raytrace_preprocess = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerTetrahedronPreprocessor.spv")).expect("Failed to read shader"));
        let raytrace_clear = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainRaytracerClear.spv")).expect("Failed to read shader"));
        let raster_tet = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainTetrahedronCS.spv")).expect("Failed to read shader"));
        let raster_edge = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainEdgeCS.spv")).expect("Failed to read shader"));
        let raster_pixel = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainTetrahedronPixelCS.spv")).expect("Failed to read shader"));
        let present_line_vs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainLineVS.spv")).expect("Failed to read shader"));
        let present_line_fs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainLineFS.spv")).expect("Failed to read shader"));
        let present_buffer_vs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBufferVS.spv")).expect("Failed to read shader"));
        let present_buffer_fs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBufferFS.spv")).expect("Failed to read shader"));
        // BVH shaders
        let bvh_scene_bounds = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHSceneBounds.spv")).expect("Failed to read shader"));
        let bvh_morton_codes = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHMortonCodes.spv")).expect("Failed to read shader"));
        let bvh_bitonic_sort = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBitonicSort.spv")).expect("Failed to read shader"));
        let bvh_init_leaves = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHInitLeaves.spv")).expect("Failed to read shader"));
        let bvh_build_tree = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHBuildTree.spv")).expect("Failed to read shader"));
        let bvh_compute_leaf_aabbs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHComputeLeafAABBs.spv")).expect("Failed to read shader"));
        let bvh_propagate_aabbs = load_shader(device.clone(),
            &std::fs::read(spirv_dir.join("mainBVHPropagateAABBs.spv")).expect("Failed to read shader"));

        let render_pass = match swapchain.clone() {
            Some(swapchain) => {
                    // The next step is to create a *render pass*, which is an object that describes where the
                    // output of the graphics pipeline will go. It describes the layout of the images where the
                    // colors, depth and/or stencil information will be written.
                Some(vulkano::single_pass_renderpass!(
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
            ).unwrap())
            }
            None => {
                None
            }
        };

        let framebuffers = match render_pass.clone() {
            Some(render_pass) => match images {
                Some(images) => {
                    Some(window_size_dependent_setup(&images, &render_pass))
                },
                None => None
            },
            None => None
        };

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default()
        ));

        let one_time_descriptor_set_layout = OneTimeBuffers::create_descriptor_set_layout(device.clone());
        let one_time_buffers = OneTimeBuffers::new(memory_allocator.clone(), descriptor_set_allocator.clone(), one_time_descriptor_set_layout.clone());
        
        let sized_descriptor_set_layout = SizedBuffers::create_descriptor_set_layout(device.clone());
        let sized_buffers = SizedBuffers::new(memory_allocator.clone(), descriptor_set_allocator.clone(), sized_descriptor_set_layout.clone(), render_dimensions);
        
        let live_descriptor_set_layout = LiveBuffers::create_descriptor_set_layout(device.clone());
        let live_buffers = LiveBuffers::new(memory_allocator.clone(), descriptor_set_allocator.clone(), live_descriptor_set_layout.clone());

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
            PipelineLayoutCreateInfo{
                set_layouts: Vec::from([
                    one_time_descriptor_set_layout.clone(),
                    sized_descriptor_set_layout.clone(),
                    live_descriptor_set_layout.clone()
                ]),
                push_constant_ranges: vec![
                    PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        offset: 0,
                        size: 16, // 4 u32s: stage, step, count, padding
                    }
                ],
                ..Default::default()
            }
        ).unwrap();
        

        // Create ShaderModules struct from individually loaded shaders
        let shaders = ShaderModules {
            line_vs: present_line_vs,
            line_fs: present_line_fs,
            buffer_vs: present_buffer_vs,
            buffer_fs: present_buffer_fs,
            tetrahedron_cs: raster_tet,
            edge_cs: raster_edge,
            tetrahedron_pixel_cs: raster_pixel,
            raytrace_preprocess: raytrace_preprocess,
            raytrace_pixel: raytrace_pixel,
            raytrace_clear: raytrace_clear,
            bvh_scene_bounds,
            bvh_morton_codes,
            bvh_bitonic_sort,
            bvh_init_leaves,
            bvh_build_tree,
            bvh_compute_leaf_aabbs,
            bvh_propagate_aabbs,
        };

        let present_pipeline = match render_pass.clone() {
            Some(render_pass) => {
                Some(PresentPipelineContext::new(device.clone(), render_pass.clone(), &shaders, pipeline_layout.clone()))
            },
            None => None
        };
        let compute_pipeline = ComputePipelineContext::new(device.clone(), &shaders, pipeline_layout.clone());
        
        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let cpu_screen_capture_buffer = create_cpu_screencapture_buffer(memory_allocator.clone(), window_size.width, window_size.height);
        
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

        // In the `window_event` handler below we are going to submit commands to the GPU.
        // Submitting a command produces an object that implements the `GpuFuture` trait, which
        // holds the resources for as long as they are in use by the GPU.
        //
        // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to
        // avoid that, we store the submission of the previous frame here.
        let previous_frame_end = None;

        
        RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            present_pipeline,
            compute_pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
            command_buffer_allocator,
            memory_allocator,
            one_time_buffers,
            sized_buffers,
            live_buffers,
            cpu_screen_capture_buffer,
            frames_rendered: 0,
        }
    }
    
    pub fn recreate_swapchain(&mut self) {
        self.recreate_swapchain = true;
    }
    
    pub fn render(&mut self, device: Arc<Device>, queue: Arc<Queue>, view_matrix: ndarray::Array2<f32>, focal_length_xy: f32, focal_length_zw: f32, model_instances: &[common::ModelInstance], render_options: RenderOptions) {

        let view_matrix_view = view_matrix.into_owned();

        let slice = view_matrix_view.view().to_slice().unwrap();
        let view_matrix_nalgebra: nalgebra::OMatrix<f32, nalgebra::U5, nalgebra::U5> = nalgebra::Matrix5::from_column_slice(slice).transpose();
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


        
        if self.previous_frame_end.is_some() {
            // Wait for previous frame to complete before updating buffers
            // This is needed because the working_data_buffer is read by shaders
            let previous_frame = self.previous_frame_end.take().unwrap();
            if previous_frame.queue().is_some() {
                let fence = previous_frame.then_signal_fence_and_flush().unwrap();
                fence.wait(None).unwrap();
            }
            self.previous_frame_end = None;
        }

        
        let mut force_clear = false;


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

                        let (new_swapchain, new_images) =
                            swapchain
                                .recreate(SwapchainCreateInfo {
                                    image_extent: window_size.into(),
                                    ..swapchain.create_info()
                                })
                                .expect("failed to recreate swapchain");

                        self.swapchain = Some(new_swapchain);

                        // Because framebuffers contains a reference to the old swapchain, we need to
                        // recreate framebuffers as well.
                        self.framebuffers = Some(window_size_dependent_setup(&new_images, &render_pass));

                        self.viewport.extent = window_size.into();

                        self.recreate_swapchain = false;

                        self.cpu_screen_capture_buffer = create_cpu_screencapture_buffer(self.memory_allocator.clone(), window_size.width, window_size.height);
                    }
                }
            }
        }

        let total_tetrahedron_count = self.one_time_buffers.model_tetrahedron_count*model_instances.len();

        // Debug: print scene info on first frame only
        if self.frames_rendered == 0 {
            println!("Scene: {} tetrahedrons ({} per instance × {} instances)",
                     total_tetrahedron_count,
                     self.one_time_buffers.model_tetrahedron_count,
                     model_instances.len());
            println!("BVH: {} internal nodes, {} total nodes",
                     total_tetrahedron_count.saturating_sub(1),
                     2 * total_tetrahedron_count.saturating_sub(1) + 1);
        }
        {
            let mut writer = self.live_buffers.working_data_buffer.write().unwrap();
            writer.view_matrix = view_matrix_nalgebra.into();
            writer.view_matrix_inverse = view_matrix_nalgebra_inv.into();
            writer.render_dimensions = glam::UVec4::new(self.sized_buffers.render_dimensions[0], self.sized_buffers.render_dimensions[1], self.sized_buffers.render_dimensions[2], 0);
            writer.present_dimensions =  match self.window.clone() {
                None => {writer.render_dimensions.xy()}
                Some(window) => {
                    let window_size = window.inner_size();
                    glam::UVec2::new(window_size.width, window_size.height)
                }
            };
            writer.total_num_tetrahedrons = total_tetrahedron_count as u32;
            writer.raytrace_seed = 6364136223846793005u64.wrapping_mul(self.frames_rendered as u64).wrapping_add(1442695040888963407);
            writer.focal_length_xy = focal_length_xy;
            writer.focal_length_zw = focal_length_zw;
            // writer.total_num_tetrahedrons = 1;
            writer.shader_fault = 0;
        }

        {
            let mut writer = self.live_buffers.model_instance_buffer.write().unwrap();
            for i in 0..model_instances.len() {
                writer[i] = model_instances[i];
            }
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
            self.command_buffer_allocator.clone(), queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();


        let (image_index, acquire_future) = match self.swapchain.clone() {
            Some(swapchain) => {
                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    swapchain.clone(),
                    None,
                ).map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    self.recreate_swapchain = true;
                }

                (Some(image_index), Some(acquire_future))
            }
            None => (None, None)
        };

        let cpu_mode = false;
        
        if cpu_mode {
            if self.previous_frame_end.is_some() {
                let previous_frame = self.previous_frame_end.take().unwrap();
                if previous_frame.queue().is_some()
                {
                    let fence =  previous_frame.then_signal_fence_and_flush().unwrap();
                    fence.wait(None).unwrap();
                }
                self.previous_frame_end = None;
            }
        }
        
        if cpu_mode {
            if render_options.do_raster || render_options.do_tetrahedron_edges { // Tetrahedron pre-raster
                unimplemented!();
            }
            
            if render_options.do_raster {
                unimplemented!();
            }
            
            if render_options.do_edges {
                unimplemented!();
            }
            
            if render_options.do_raytrace {
                // CPU shader fallback not available with Slang shaders
                // (Slang compiles to SPIR-V only, not CPU-executable code)
                unimplemented!("CPU raytracing not available - use GPU mode");
            }
            
        }
        else {
            
            builder.bind_descriptor_sets(PipelineBindPoint::Compute, self.compute_pipeline.pipeline_layout.clone(), 0,
                                         vec![
                                             self.one_time_buffers.descriptor_set.clone(),
                                             self.sized_buffers.descriptor_set.clone(),
                                             self.live_buffers.descriptor_set.clone()
                                         ]).unwrap();

            // Set default push constants (required by pipeline layout even for shaders that don't use them)
            let dummy_push_data: [u32; 4] = [0, 0, 0, 0];
            builder.push_constants(self.compute_pipeline.pipeline_layout.clone(), 0, dummy_push_data).unwrap();

            if render_options.do_frame_clear || force_clear {
                builder.bind_pipeline_compute(self.compute_pipeline.raytrace_clear_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([self.sized_buffers.render_dimensions[0]/8, self.sized_buffers.render_dimensions[1]/8, 1])}.unwrap() ; // Do compute stage
            }

            if render_options.do_raster || render_options.do_tetrahedron_edges { // Tetrahedron pre-raster
                builder.bind_pipeline_compute(self.compute_pipeline.tetrahedron_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ; // Do compute stage

                if render_options.do_tetrahedron_edges {
                    line_render_count = total_tetrahedron_count*6;
                }
            }

            if render_options.do_raster { // Tetrahedron pixel raster
                builder.bind_pipeline_compute(self.compute_pipeline.tetrahedron_pixel_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([(self.sized_buffers.render_dimensions[0]+7)/8, (self.sized_buffers.render_dimensions[1]+7)/8, 1])}.unwrap() ; // Do compute stage
            }

            if render_options.do_edges {   // Tetrahedron edge pre-raster
                builder.bind_pipeline_compute(self.compute_pipeline.edge_pipeline.clone()).unwrap();
                let total_edge_count = self.one_time_buffers.model_edge_count*model_instances.len();
                unsafe {builder.dispatch([(total_edge_count as u32 + 63)/64u32, 1, 1])}.unwrap() ; // Do compute stage

                line_render_count = total_edge_count;
            }

            if render_options.do_raytrace {
                // 1. Tetrahedron preprocessing (transform to view space)
                builder.bind_pipeline_compute(self.compute_pipeline.raytrace_pre_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ;

                // 2. BVH Construction
                if total_tetrahedron_count > 0 {
                    // 2a. Compute scene bounds
                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_scene_bounds_pipeline.clone()).unwrap();
                    unsafe {builder.dispatch([1, 1, 1])}.unwrap() ;

                    // 2b. Compute Morton codes (dispatch n_pow2 threads to fill sentinels for padding)
                    let n = total_tetrahedron_count as u32;
                    let n_pow2 = n.next_power_of_two();
                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_morton_codes_pipeline.clone()).unwrap();
                    unsafe {builder.dispatch([(n_pow2 + 63)/64u32, 1, 1])}.unwrap() ;

                    // 2c. Bitonic sort (log^2(n) dispatches)
                    // Sort all n_pow2 elements (including sentinel-padded entries)
                    let num_stages = n_pow2.trailing_zeros(); // log2(n_pow2)

                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_bitonic_sort_pipeline.clone()).unwrap();
                    for stage in 0..num_stages {
                        for step in (0..=stage).rev() {
                            let push_data: [u32; 4] = [stage, step, n_pow2, 0];
                            builder.push_constants(self.compute_pipeline.pipeline_layout.clone(), 0, push_data).unwrap();
                            unsafe {builder.dispatch([(n_pow2 + 63)/64, 1, 1])}.unwrap() ;
                        }
                    }

                    // 2d. Initialize leaf nodes
                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_init_leaves_pipeline.clone()).unwrap();
                    unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ;

                    // 2e. Build internal nodes (Karras algorithm)
                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_build_tree_pipeline.clone()).unwrap();
                    unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ;

                    // 2f. Compute leaf AABBs
                    builder.bind_pipeline_compute(self.compute_pipeline.bvh_compute_leaf_aabbs_pipeline.clone()).unwrap();
                    unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ;

                    // 2g. Propagate AABBs from leaves to root (multi-pass)
                    // Each pass computes AABBs for internal nodes whose children are ready.
                    // Tree depth is at most ceil(log2(N)), so that many passes suffices.
                    let num_internal_nodes = total_tetrahedron_count.saturating_sub(1) as u32;
                    if num_internal_nodes > 0 {
                        let num_passes = 2 * (32 - num_internal_nodes.leading_zeros()).max(1);
                        builder.bind_pipeline_compute(self.compute_pipeline.bvh_propagate_aabbs_pipeline.clone()).unwrap();
                        for _ in 0..num_passes {
                            unsafe {builder.dispatch([(num_internal_nodes + 63)/64, 1, 1])}.unwrap();
                        }
                    }
                }

                // Debug: copy BVH data to CPU on first frame
                if self.frames_rendered == 0 {
                    builder.copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.bvh_nodes_buffer.clone(),
                        self.sized_buffers.cpu_bvh_nodes_buffer.clone(),
                    )).unwrap();
                    builder.copy_buffer(CopyBufferInfo::buffers(
                        self.sized_buffers.morton_codes_buffer.clone(),
                        self.sized_buffers.cpu_morton_codes_buffer.clone(),
                    )).unwrap();
                }

                // 3. Raytrace pixels (using BVH)
                builder.bind_pipeline_compute(self.compute_pipeline.raytrace_pixel_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([(self.sized_buffers.render_dimensions[0]+7)/8, (self.sized_buffers.render_dimensions[1]+7)/8, 1])}.unwrap() ;
            }
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
                        ).unwrap();
                    builder.set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap();
                    builder.bind_descriptor_sets(PipelineBindPoint::Graphics, present_pipeline.pipeline_layout.clone(), 0,
                                                 vec![
                                                     self.one_time_buffers.descriptor_set.clone(),
                                                     self.sized_buffers.descriptor_set.clone(),
                                                     self.live_buffers.descriptor_set.clone()
                                                 ]).unwrap();


                    // Render from compute shader buffer
                    {
                        builder.bind_pipeline_graphics(present_pipeline.buffer_pipeline.clone()).unwrap();
                        unsafe { builder.draw(6, 1, 0, 0) }.unwrap();
                    }

                    // Render the edge lines
                    if line_render_count > 0
                    {
                        builder.bind_pipeline_graphics(present_pipeline.line_pipeline.clone()).unwrap();
                        unsafe { builder.draw(line_render_count as u32*2, 1, 0, 0) }.unwrap();
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
                                framebuffers[image_index as usize].attachments()[0].image().clone(),
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



        // Finish recording the command buffer by calling `end`.
        let command_buffer = builder.build().unwrap();

        let frame_end = match self.previous_frame_end.take() {
            Some(f) => f,
            None => {
                sync::now(device.clone()).boxed()
            }
        };
        self.previous_frame_end = None;

        match acquire_future {
            Some(acquire_future) => {
                let future = frame_end
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
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
                        self.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        self.previous_frame_end = None;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            },
            None => {
                let future = frame_end
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        self.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        self.previous_frame_end = None;
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }

                if self.previous_frame_end.is_some() {
                    let fence =  self.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap();
                    fence.wait(None).unwrap();
                    self.previous_frame_end = None;
                }
            }
        };

        if let Some(window) = self.window.clone() {
            let window_size = window.inner_size();
            // Save frame
            if self.frames_rendered > 3 && render_options.take_framebuffer_screenshot {
                if self.previous_frame_end.is_some() {
                    let fence =  self.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap();
                    fence.wait(None).unwrap();
                    self.previous_frame_end = None;
                }

                let result = self.cpu_screen_capture_buffer.read();
                match result {
                    Ok(buffer_content) => {
                        let screenshot_path = format!("frames/framebuffer_{}.webp", self.frames_rendered-3);

                        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(window_size.width, window_size.height, &buffer_content[..]).unwrap();
                        image.save(screenshot_path.clone()).unwrap();
                        println!("Saved screenshot to {}", screenshot_path);
                    }
                    Err(error) => {
                        eprintln!("Error saving screenshot: {:?}", error);
                    }
                };
            }
        }

        // Debug: print BVH diagnostics on first frame
        if self.frames_rendered == 0 && render_options.do_raytrace {
            // Ensure GPU work is done
            if self.previous_frame_end.is_some() {
                let fence = self.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap();
                fence.wait(None).unwrap();
                self.previous_frame_end = None;
            }

            let bvh_nodes = self.sized_buffers.cpu_bvh_nodes_buffer.read().unwrap();
            let morton_codes = self.sized_buffers.cpu_morton_codes_buffer.read().unwrap();
            let num_leaves = total_tetrahedron_count;
            let num_internal = num_leaves.saturating_sub(1);
            let total_nodes = 2 * num_leaves - 1;

            // Check Morton code sorting
            let mut sorted = true;
            for i in 1..num_leaves {
                if morton_codes[i].code < morton_codes[i-1].code {
                    println!("  SORT ERROR at {}: code[{}]={} > code[{}]={}",
                             i, i-1, morton_codes[i-1].code, i, morton_codes[i].code);
                    sorted = false;
                }
            }
            println!("Morton codes sorted: {}", sorted);

            // Check root node
            let root = &bvh_nodes[0];
            println!("Root node: left={}, right={}, isLeaf={}, visitCount={}",
                     root.left_child, root.right_child, root.is_leaf, root.atomic_visit_count);
            println!("Root AABB: min=({:.2},{:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2},{:.2})",
                     root.min_bounds.x, root.min_bounds.y, root.min_bounds.z, root.min_bounds.w,
                     root.max_bounds.x, root.max_bounds.y, root.max_bounds.z, root.max_bounds.w);

            // Count valid internal nodes
            let mut valid_internal = 0;
            let mut invalid_children = 0;
            let mut zero_aabb_internal = 0;
            for i in 0..num_internal {
                if bvh_nodes[i].atomic_visit_count >= 2 {
                    valid_internal += 1;
                }
                if bvh_nodes[i].left_child >= total_nodes as u32 || bvh_nodes[i].right_child >= total_nodes as u32 {
                    if bvh_nodes[i].left_child != 0xFFFFFFFF && bvh_nodes[i].right_child != 0xFFFFFFFF {
                        invalid_children += 1;
                    }
                }
                let aabb_size = (bvh_nodes[i].max_bounds - bvh_nodes[i].min_bounds).length();
                if aabb_size < 0.001 && bvh_nodes[i].atomic_visit_count >= 2 {
                    zero_aabb_internal += 1;
                }
            }
            println!("Internal nodes: {}/{} valid (visitCount>=2), {} invalid children, {} zero-AABB",
                     valid_internal, num_internal, invalid_children, zero_aabb_internal);

            // Count valid leaves
            let mut zero_aabb_leaves = 0;
            for i in 0..num_leaves {
                let leaf_idx = num_internal + i;
                let aabb_size = (bvh_nodes[leaf_idx].max_bounds - bvh_nodes[leaf_idx].min_bounds).length();
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
        if self.previous_frame_end.is_some() {
            let fence =  self.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap();
            fence.wait(None).unwrap();
            self.previous_frame_end = None;
        }

        let result = self.sized_buffers.output_cpu_pixel_buffer.read();
        match result {
            Ok(buffer_content) => {
                let mut buffer_stored = Vec::new();
                buffer_stored.extend_from_slice(&buffer_content[..]);
                let mut buffer_arc = Arc::new(buffer_stored);
                let buffer_dimensions = self.sized_buffers.render_dimensions;
                let mut layers = Vec::new();
                struct HereGetPixel {
                    buffer_arc: Arc<Vec<Vec4>>,
                    dimensions: [u32; 3],
                    z: Option<u32>
                }
                impl exr::prelude::GetPixel for HereGetPixel {
                    type Pixel = (f32, f32, f32, f32);

                    fn get_pixel(&self, position: exr::math::Vec2<usize>) -> Self::Pixel {
                        // Get cumulative pixel value
                        let mut full_pixel = Vec4::ZERO;
                        for z in 0..self.dimensions[2] {
                            full_pixel += self.buffer_arc[position.x() + position.y()*self.dimensions[0] as usize + (z*self.dimensions[0]*self.dimensions[1]) as usize];
                        }
                        if let Some(z) = self.z {
                            let local_pixel = self.buffer_arc[position.x() + position.y()*self.dimensions[0] as usize + (z*self.dimensions[0]*self.dimensions[1]) as usize];
                            (
                                local_pixel.x/local_pixel.w,
                                local_pixel.y/local_pixel.w,
                                local_pixel.z/local_pixel.w,
                                1.0
                            )
                        }
                        else {
                            (
                                full_pixel.x/full_pixel.w,
                                full_pixel.y/full_pixel.w,
                                full_pixel.z/full_pixel.w,
                                1.0
                            )
                        }
                    }
                }
                // Render the full thing in layer 0 (because apparently support for openexr is rather poor)
                let pixel_getter = HereGetPixel {
                    buffer_arc: buffer_arc.clone(),
                    dimensions: buffer_dimensions,
                    z: None
                };
                layers.push(
                    exr::prelude::Layer::new(
                        (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                        exr::prelude::LayerAttributes{
                            layer_name: Some(exr::prelude::Text::new_or_panic(format!("Full Render"))),
                            .. Default::default()
                        },
                        exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                        exr::prelude::SpecificChannels::rgba(pixel_getter))
                );
                for z in 0..self.sized_buffers.render_dimensions[2] {
                    let pixel_getter = HereGetPixel {
                        buffer_arc: buffer_arc.clone(),
                        dimensions: buffer_dimensions,
                        z: Some(z)
                    };
                    let layer = exr::prelude::Layer::new(
                        (buffer_dimensions[0] as usize, buffer_dimensions[1] as usize),
                        exr::prelude::LayerAttributes{
                            layer_name: Some(exr::prelude::Text::new_or_panic(format!("ZW Slice {}/{}", z, buffer_dimensions[2]))),
                            .. Default::default()
                        },
                        exr::prelude::Encoding::SMALL_FAST_LOSSLESS,
                        exr::prelude::SpecificChannels::rgba(pixel_getter));
                    layers.push(layer);
                }

                let image = exr::image::Image::from_layers(
                    exr::prelude::ImageAttributes {
                        display_window: exr::prelude::IntegerBounds::new(
                            (0, 0),
                            (self.sized_buffers.render_dimensions[0] as usize, self.sized_buffers.render_dimensions[1] as usize)),
                        pixel_aspect: 1.0,
                        chromaticities: None,
                        time_code: None,
                        other: Default::default()
                    },
                    layers);
                image.write().to_file(path).unwrap();
                println!("Saved screenshot to {}", path);
            }
            Err(error) => {
                eprintln!("Error saving screenshot: {:?}", error);
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

fn create_cpu_screencapture_buffer(memory_allocator: Arc<dyn MemoryAllocator>, width: u32, height: u32) -> Subbuffer<[u8]>
{
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
        vec![0; (width*height*4) as usize]
    ).unwrap()
}