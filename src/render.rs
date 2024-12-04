use std::collections::BTreeMap;
use std::f32::consts::PI;
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
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::shader::{ShaderModule, ShaderStages};
use vulkano::{sync, Validated, VulkanError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::sync::GpuFuture;
use bytemuck::{Zeroable};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use common::{get_normal, ModelTetrahedron, VecN};
use crate::hypercube::{generate_simplexes_for_k_face, Hypercube};
use glam::{Vec4, Vec2, UVec3};
use image::{ImageBuffer, Rgba};
use vulkano::format::ClearColorValue;
use vulkano::format::Format::{R16G16B16A16_UNORM, R8G8B8A8_UNORM};

pub struct RenderOptions {
    pub do_frame_clear: bool,
    pub do_raster: bool,
    pub do_raytrace: bool,
    pub do_edges: bool,
    pub do_tetrahedron_edges: bool,
    pub take_screenshot: bool
}

pub fn generate_tesseract_tetrahedrons() -> Vec<ModelTetrahedron> {
    let tesseract_vertexes = Hypercube::<4, usize>::generate_vertices();
    let cube_vertexes = Hypercube::<3, usize>::generate_vertices();
    let tetrahedron_cells = Hypercube::<4, usize>::generate_k_faces::<3>();

    let mut output_tetrahedrons = Vec::new();
    let texture_position_simplexes = generate_simplexes_for_k_face::<3, 3>([0b000, 0b001, 0b010, 0b100]);

    for cell_id in 0..tetrahedron_cells.len() {
        //for cell_id in 0..1 {
        let position_simplexes = generate_simplexes_for_k_face::<4, 3>(tetrahedron_cells[cell_id]);


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
    let tesseract_edges = Hypercube::<4, usize>::generate_k_faces::<1>();

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
    render_dimensions: [u32; 2],
    line_vertexes_buffer: Subbuffer<[Vec2]>,
    output_tetrahedron_buffer: Subbuffer<[common::Tetrahedron]>,
    output_pixel_buffer: Subbuffer<[Vec4]>,
    descriptor_set: Arc<DescriptorSet>,
}

impl SizedBuffers {
    pub fn new(memory_allocator: Arc<dyn MemoryAllocator>, descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>, descriptor_set_layout: Arc<DescriptorSetLayout>, render_dimensions: [u32; 2]) -> Self {
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
            vec![glam::Vec2::new(0.0, 0.0); 10000]
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
            vec![common::Tetrahedron::zeroed(); 1000]
        ).unwrap();

        let output_pixel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                .. Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![Vec4::ZERO; (render_dimensions[0]*render_dimensions[1]) as usize]
        ).unwrap();
        
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, line_vertexes_buffer.clone()),
                WriteDescriptorSet::buffer(1, output_tetrahedron_buffer.clone()),
                WriteDescriptorSet::buffer(2, output_pixel_buffer.clone()),
            ],
            []
        ).unwrap();
        Self {
            render_dimensions,
            line_vertexes_buffer,
            output_tetrahedron_buffer,
            output_pixel_buffer,
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

struct PresentPipelineContext {
    line_pipeline: Arc<GraphicsPipeline>,
    buffer_pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>
}

impl PresentPipelineContext {
    pub fn new(device: Arc<Device>, render_pass: Arc<RenderPass>, loaded_shader: Arc<ShaderModule>, pipeline_layout: Arc<PipelineLayout>) -> Self {

        let line_pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.

            let vs = loaded_shader.entry_point("present::main_line_vs").unwrap();
            let fs = loaded_shader.entry_point("present::main_line_fs").unwrap();

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
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.

            let vs = loaded_shader.entry_point("present::main_buffer_vs").unwrap();
            let fs = loaded_shader.entry_point("present::main_buffer_fs").unwrap();

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
    pipeline_layout: Arc<PipelineLayout>
}

impl ComputePipelineContext {
    pub fn new(device: Arc<Device>, loaded_shader: Arc<ShaderModule>, pipeline_layout: Arc<PipelineLayout>) -> Self {
        Self {
            tetrahedron_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("rasterizer::main_tetrahedron_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            edge_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("rasterizer::main_edge_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            tetrahedron_pixel_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("rasterizer::main_tetrahedron_pixel_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_pre_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("raytracer::main_raytracer_tetrahedron_preprocessor_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_pixel_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("raytracer::main_raytracer_pixel_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            raytrace_clear_pipeline: ComputePipeline::new(device.clone(), None, ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(loaded_shader.entry_point("raytracer::main_raytracer_clear_cs").unwrap()),
                pipeline_layout.clone(),
            )).unwrap(),
            pipeline_layout
        }
    }
}

pub struct RenderContext {
    pub(crate) window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    present_pipeline: PresentPipelineContext,
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
    pub fn new(device: Arc<Device>, instance: Arc<Instance>, window: Arc<Window>, render_width: u32, render_height: u32) -> RenderContext {

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // The objective of this example is to draw a triangle on a window. To do so, we first need
        // to create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
        //
        // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
        // object from it, which represents the drawable surface of a window. For that we must wrap
        // the `winit::window::Window` in an `Arc`.
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let (swapchain, images) = {
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
            let image_format = R8G8B8A8_UNORM;
            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
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
                .unwrap()
        };


        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        
        mod shader_loader {
            vulkano_shaders::shader! {
                bytes: "shaders.spv",
                root_path_env: "SPIRV_OUT_DIR"
            }
        }

        // The next step is to create a *render pass*, which is an object that describes where the
        // output of the graphics pipeline will go. It describes the layout of the images where the
        // colors, depth and/or stencil information will be written.
        let render_pass = vulkano::single_pass_renderpass!(
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
        ).unwrap();

        // The render pass we created above only describes the layout of our framebuffers. Before
        // we can draw we also need to create the actual framebuffers.
        //
        // Since we need to draw to multiple images, we are going to create a different framebuffer
        // for each image.
        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default()
        ));

        let one_time_descriptor_set_layout = OneTimeBuffers::create_descriptor_set_layout(device.clone());
        let one_time_buffers = OneTimeBuffers::new(memory_allocator.clone(), descriptor_set_allocator.clone(), one_time_descriptor_set_layout.clone());
        
        let sized_descriptor_set_layout = SizedBuffers::create_descriptor_set_layout(device.clone());
        let sized_buffers = SizedBuffers::new(memory_allocator.clone(), descriptor_set_allocator.clone(), sized_descriptor_set_layout.clone(), [render_width, render_height]);
        
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
                ..Default::default()
            }
        ).unwrap();
        

        let loaded_shader = shader_loader::load(device.clone()).unwrap();
        let present_pipeline = PresentPipelineContext::new(device.clone(), render_pass.clone(), loaded_shader.clone(), pipeline_layout.clone());
        let compute_pipeline = ComputePipelineContext::new(device.clone(), loaded_shader.clone(), pipeline_layout.clone());
        
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
    
    pub fn render(&mut self, device: Arc<Device>, queue: Arc<Queue>, view_matrix: ndarray::Array2<f32>, model_instances: &[common::ModelInstance], render_options: RenderOptions) {
        let window_size = self.window.inner_size();

        // Do not draw the frame when the screen size is zero. On Windows, this can occur
        // when minimizing the application.
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        
        if self.previous_frame_end.is_some() {
            // It is important to call this function from time to time, otherwise resources
            // will keep accumulating and you will eventually reach an out of memory error.
            // Calling this function polls various fences in order to determine what the GPU
            // has already processed, and frees the resources that are no longer needed.
            self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        }

        
        let mut force_clear = false;


        // Whenever the window resizes we need to recreate everything dependent on the
        // window size. In this example that includes the swapchain, the framebuffers and
        // the dynamic state viewport.
        if self.recreate_swapchain {
            // Use the new dimensions of the window.

            force_clear = true;

            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;

            // Because framebuffers contains a reference to the old swapchain, we need to
            // recreate framebuffers as well.
            self.framebuffers = window_size_dependent_setup(&new_images, &self.render_pass);

            self.viewport.extent = window_size.into();

            self.recreate_swapchain = false;

            self.cpu_screen_capture_buffer = create_cpu_screencapture_buffer(self.memory_allocator.clone(), window_size.width, window_size.height);
        }

        // Before we can draw on the output, we have to *acquire* an image from the
        // swapchain. If no image is available (which happens if you submit draw commands
        // too quickly), then the function will block. This operation returns the index of
        // the image that we are allowed to draw upon.
        //
        // This function can block if no image is available. The parameter is an optional
        // timeout after which the function call will return an error.
        let (image_index, suboptimal, acquire_future) = match acquire_next_image(
            self.swapchain.clone(),
            None,
        )
            .map_err(Validated::unwrap)
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
        )
            .unwrap();
        
        
        
        {
            let mut writer = self.live_buffers.model_instance_buffer.write().unwrap();
            for i in 0..model_instances.len() {
                writer[i] = model_instances[i];
            }
        }


        let total_tetrahedron_count = self.one_time_buffers.model_tetrahedron_count*model_instances.len();
        {
            let mut writer = self.live_buffers.working_data_buffer.write().unwrap();
            writer.view_matrix = view_matrix.into();
            writer.present_dimensions = glam::UVec2::new(window_size.width, window_size.height);
            writer.render_dimensions = glam::UVec2::new(self.sized_buffers.render_dimensions[0], self.sized_buffers.render_dimensions[1]);
            writer.total_num_tetrahedrons = total_tetrahedron_count as u32;
            writer.raytrace_seed = (self.frames_rendered*211) as u32;
           // writer.total_num_tetrahedrons = 1;
            writer.shader_fault = 0;
        }

        let mut line_render_count = 0;
        
        // Do compute stage
        
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
                for i in 0..total_tetrahedron_count {
                    shaders::main_raytracer_tetrahedron_preprocessor_cs(
                        &self.one_time_buffers.model_tetrahedron_buffer.read().unwrap(),
                        &self.live_buffers.model_instance_buffer.read().unwrap(),
                        self.sized_buffers.output_tetrahedron_buffer.write().unwrap().as_mut(),
                        UVec3::new(i as u32, 0, 0),
                        &self.live_buffers.working_data_buffer.read().unwrap(),
                    )
                }

                for x in 0..self.sized_buffers.render_dimensions[0] {
                    for y in 0..self.sized_buffers.render_dimensions[1] {
                        shaders::main_raytracer_pixel_cs(
                            self.sized_buffers.output_tetrahedron_buffer.write().unwrap().as_mut(),
                            UVec3::new(x, y, 0),
                            self.live_buffers.working_data_buffer.write().as_mut().unwrap(),
                            self.sized_buffers.output_pixel_buffer.write().unwrap().as_mut()
                        )
                    }
                }
            }
            
        }
        else {
            
            builder.bind_descriptor_sets(PipelineBindPoint::Compute, self.compute_pipeline.pipeline_layout.clone(), 0,
                                         vec![
                                             self.one_time_buffers.descriptor_set.clone(),
                                             self.sized_buffers.descriptor_set.clone(),
                                             self.live_buffers.descriptor_set.clone()
                                         ]).unwrap();
            
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
                unsafe {builder.dispatch([self.sized_buffers.render_dimensions[0]/8, self.sized_buffers.render_dimensions[1]/8, 1])}.unwrap() ; // Do compute stage
            }

            if render_options.do_edges || render_options.do_tetrahedron_edges {   // Tetrahedron edge pre-raster
                builder.bind_pipeline_compute(self.compute_pipeline.edge_pipeline.clone()).unwrap();
                let total_edge_count = self.one_time_buffers.model_edge_count*model_instances.len();
                unsafe {builder.dispatch([(total_edge_count as u32 + 63)/64u32, 1, 1])}.unwrap() ; // Do compute stage

                line_render_count = total_edge_count;
            }

            if render_options.do_raytrace {
                builder.bind_pipeline_compute(self.compute_pipeline.raytrace_pre_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([(total_tetrahedron_count as u32 + 63)/64u32, 1, 1])}.unwrap() ;

                builder.bind_pipeline_compute(self.compute_pipeline.raytrace_pixel_pipeline.clone()).unwrap();
                unsafe {builder.dispatch([self.sized_buffers.render_dimensions[0]/8, self.sized_buffers.render_dimensions[1]/8, 1])}.unwrap() ;
            }
        }

        
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
                        self.framebuffers[image_index as usize].clone(),
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
        builder.bind_descriptor_sets(PipelineBindPoint::Graphics, self.present_pipeline.pipeline_layout.clone(), 0,
                                     vec![
                                         self.one_time_buffers.descriptor_set.clone(),
                                         self.sized_buffers.descriptor_set.clone(),
                                         self.live_buffers.descriptor_set.clone()
                                     ]).unwrap();
        
        
        // Render from compute shader buffer
        {
            builder.bind_pipeline_graphics(self.present_pipeline.buffer_pipeline.clone()).unwrap();
            unsafe { builder.draw(6, 1, 0, 0) }.unwrap();
        }
        
        // Render the edge lines
        if line_render_count > 0
        {
            builder.bind_pipeline_graphics(self.present_pipeline.line_pipeline.clone()).unwrap();
            unsafe { builder.draw(line_render_count as u32*2, 1, 0, 0) }.unwrap();
        }

        // End render pass
        builder
            // We leave the render pass. Note that if we had multiple subpasses we could
            // have called `next_subpass` to jump to the next subpass.
            .end_render_pass(Default::default())
            .unwrap();
        if render_options.take_screenshot {
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    self.framebuffers[image_index as usize].attachments()[0].image().clone(),
                    self.cpu_screen_capture_buffer.clone(),
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
                    self.swapchain.clone(),
                    image_index,
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

        // Save frame
        if self.frames_rendered > 3 && render_options.take_screenshot {
            if self.previous_frame_end.is_some() {
                let fence =  self.previous_frame_end.take().unwrap().then_signal_fence_and_flush().unwrap();
                fence.wait(None).unwrap();
                self.previous_frame_end = None;
            }

            let result = self.cpu_screen_capture_buffer.read();
            match result {
                Ok(buffer_content) => {
                    let screenshot_path = format!("frames/screenshot_{}.webp", self.frames_rendered-3);

                    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(window_size.width, window_size.height, &buffer_content[..]).unwrap();
                    image.save(screenshot_path.clone()).unwrap();
                    println!("Saved screenshot to {}", screenshot_path);
                }
                Err(error) => {
                    eprintln!("Error saving screenshot: {:?}", error);
                }
            };
        }

        self.frames_rendered += 1;
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