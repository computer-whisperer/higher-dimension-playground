use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::tesseract;


#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderMetadata {
    pub screen_width: u32,
    pub screen_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub depth_factor: u32
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraState {
    pub view_transform: [f32; 32]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct ModelInstance {
    pub(crate) model_transform: [f32; 32],
    pub(crate) cell_texture_ids: [u32; 8]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
struct TetrahedronMetadataInput {
    tetrahedron_count: u32,
    processed_tet_count: u32
}

const RENDER_BUFFER_GROUP_ID: u32 = 0;
const COLOR_BUFFER_IDX: u32 = 0;
const DEPTH_BUFFER_IDX: u32 = 1;
const OVERLAY_BUFFER_IDX: u32 = 2;
const RENDER_META_BUFFER_IDX: u32 = 3;

const ONETIME_BUFFER_GROUP_ID: u32 = 1;
const VERTEX_BUFFER_IDX: u32 = 0;


const LIVE_BUFFER_GROUP_ID: u32 = 2;
const INTERMEDIATE_VERTEX_BUFFER_IDX: u32 = 0;
const INSTANCE_BUFFER_IDX: u32 = 1;
const CAMERA_BUFFER_IDX: u32 = 3;
const TET_BUFFER_0_IDX: u32 = 4;
const TET_BUFFER_1_IDX: u32 = 5;
const TET_METADATA_BUFFER_IDX: u32 = 7;


const MAX_INSTANCES : usize = 1000;
const MAX_INTERMEDIATE_VERTICES : u64 = (MAX_INSTANCES as u64) * (tesseract::TET_VERTICES.len() as u64);

pub struct RasterPipelines {
    clear_pipeline: wgpu::ComputePipeline,
    vertex_pipeline: wgpu::ComputePipeline,
    lines_pipeline: wgpu::ComputePipeline,
    tet_pre_pipeline: wgpu::ComputePipeline,
    sort_0_pipeline: wgpu::ComputePipeline,
    present_pipeline: wgpu::RenderPipeline,
    render_buffers_bind_group_raster_layout: wgpu::BindGroupLayout,
    render_buffers_bind_group_present_layout: wgpu::BindGroupLayout,
    one_time_buffers_bind_group_layout: wgpu::BindGroupLayout,
    live_buffers_bind_group_layout_a: wgpu::BindGroupLayout,
    live_buffers_bind_group_layout_b: wgpu::BindGroupLayout
}


impl RasterPipelines {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let render_buffers_bind_group_raster_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Buffers Bind Group Layout"),
                entries: &[
                wgpu::BindGroupLayoutEntry { // Render Buffer
                    binding: COLOR_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Depth Buffer
                    binding: DEPTH_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Overlay Buffer
                    binding: OVERLAY_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Render Metadata
                    binding: RENDER_META_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },],
            });
        let render_buffers_bind_group_present_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Buffers Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { // Render Buffer
                        binding: COLOR_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { // Depth Buffer
                        binding: DEPTH_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { // Overlay Buffer
                        binding: OVERLAY_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { // Render Metadata
                        binding: RENDER_META_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },],
            });
        let one_time_buffers_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: One Time Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { // Tesseract Vertex Buffer
                        binding: VERTEX_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },

                ],
            });
        let live_buffers_bind_group_layout_a =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Live Bind Group Layout A"),
                entries: &[
                wgpu::BindGroupLayoutEntry { // Intermediate Vertex Buffer
                    binding: INTERMEDIATE_VERTEX_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Instance Buffer
                    binding: INSTANCE_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Camera
                    binding: CAMERA_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { // Tetrahedron Buffer 0
                    binding: TET_BUFFER_0_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                ],
            });
        let live_buffers_bind_group_layout_b =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Live Bind Group Layout B"),
                entries: &[
                    wgpu::BindGroupLayoutEntry { // Tetrahedron Buffer 0
                        binding: TET_BUFFER_0_IDX,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { // Tetrahedron Buffer 1
                        binding: TET_BUFFER_1_IDX,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry { // Tetrahedron Metadata Buffer
                        binding: TET_METADATA_BUFFER_IDX,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
            });

        let raster_layout_a = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raster Pipeline Layout A"),
            bind_group_layouts: &[
                &render_buffers_bind_group_raster_layout,
                &one_time_buffers_bind_group_layout,
                &live_buffers_bind_group_layout_a
            ],
            push_constant_ranges: &[],
        });
        let raster_layout_b = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raster Pipeline Layout B"),
            bind_group_layouts: &[
                &render_buffers_bind_group_raster_layout,
                &one_time_buffers_bind_group_layout,
                &live_buffers_bind_group_layout_b
            ],
            push_constant_ranges: &[],
        });
        let present_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &render_buffers_bind_group_present_layout,
            ],
            push_constant_ranges: &[],
        });
        let vertex_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Vertex Pipeline"),
            layout: Some(&raster_layout_a),
            module: &device.create_shader_module(wgpu::include_wgsl!("raster_vertex.wgsl")),
            entry_point: Some("raster_vertex_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let lines_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Lines Pipeline"),
            layout: Some(&raster_layout_a),
            module: &device.create_shader_module(wgpu::include_wgsl!("raster_lines.wgsl")),
            entry_point: Some("raster_lines_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Pipeline"),
            layout: Some(&raster_layout_a),
            module: &device.create_shader_module(wgpu::include_wgsl!("raster_clear.wgsl")),
            entry_point: Some("raster_clear_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let tet_pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Pre Tetrahedron Pipeline"),
            layout: Some(&raster_layout_a),
            module: &device.create_shader_module(wgpu::include_wgsl!("raster_tet_pre.wgsl")),
            entry_point: Some("raster_tet_pre_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sort_0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Sort Pipeline"),
            layout: Some(&raster_layout_b),
            module: &device.create_shader_module(wgpu::include_wgsl!("raster_sort.wgsl")),
            entry_point: Some("raster_sort_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let present_shader = device.create_shader_module(wgpu::include_wgsl!("present.wgsl"));
        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Present Pipeline"),
            layout: Some(&present_layout),
            vertex: wgpu::VertexState {
                module: &present_shader,
                entry_point: Some("vs_main_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &present_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: Default::default(),
                strip_index_format: None,
                front_face: Default::default(),
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: Default::default(),
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            clear_pipeline,
            vertex_pipeline,
            lines_pipeline,
            tet_pre_pipeline,
            present_pipeline,
            sort_0_pipeline,
            live_buffers_bind_group_layout_a,
            live_buffers_bind_group_layout_b,
            one_time_buffers_bind_group_layout,
            render_buffers_bind_group_raster_layout,
            render_buffers_bind_group_present_layout,
        }
    }
}

struct RenderBuffers {
    color_buffer: wgpu::Buffer,
    depth_buffer: wgpu::Buffer,
    overlay_buffer: wgpu::Buffer,
    render_meta_uniform_buffer: wgpu::Buffer
}

struct LiveBuffers {
    vertex_intermediate_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    camera_uniform_buffer: wgpu::Buffer,
    tet_buffer_0: wgpu::Buffer,
    tet_buffer_1: wgpu::Buffer,
    tet_metadata_buffer: wgpu::Buffer,
}

pub struct RasterBindings {
    render_buffers: RenderBuffers,

    model_vertex_buffer: wgpu::Buffer,
    live_buffers: LiveBuffers,

    raster_render_bind_group: wgpu::BindGroup,
    present_render_bind_group: wgpu::BindGroup,
    one_time_bind_group: wgpu::BindGroup,
    live_bind_group_a: wgpu::BindGroup,
    live_bind_group_b: wgpu::BindGroup,

    latest_render_metadata: RenderMetadata,
}

impl RasterBindings {
    fn create_render_buffers(device: &wgpu::Device, render_metadata: &RenderMetadata) -> RenderBuffers {
        let pixels = render_metadata.render_width as u64 * render_metadata.render_height as u64;

        let depth_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Buffer"),
            size: pixels*4*(render_metadata.depth_factor as u64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let color_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Color Buffer"),
            size: pixels*4*(render_metadata.depth_factor as u64),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let overlay_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Overlay Buffer"),
            size: pixels*4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let render_meta_uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Screen Uniform Buffer"),
                contents: bytemuck::cast_slice(&[(*render_metadata).clone()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        RenderBuffers {color_buffer, depth_buffer, overlay_buffer, render_meta_uniform_buffer}
    }

    fn create_render_bind_group(device: &wgpu::Device,
                                layout: &wgpu::BindGroupLayout,
                                render_buffers: &RenderBuffers) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: COLOR_BUFFER_IDX,
                    resource: render_buffers.color_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: DEPTH_BUFFER_IDX,
                    resource: render_buffers.depth_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: OVERLAY_BUFFER_IDX,
                    resource: render_buffers.overlay_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: RENDER_META_BUFFER_IDX,
                    resource: render_buffers.render_meta_uniform_buffer.as_entire_binding(),
                }
            ],
        })
    }

    fn create_live_buffers(device: &wgpu::Device) -> LiveBuffers {
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Model Instance Buffer"),
                contents: bytemuck::cast_slice(&[ModelInstance{
                    model_transform: [0.0; 32],
                    cell_texture_ids: [0; 8],
                }; MAX_INSTANCES]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }
        );

        let vertex_intermediate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Output Buffer"),
            size: MAX_INTERMEDIATE_VERTICES*32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let camera_uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Uniform Buffer"),
                contents: bytemuck::cast_slice(&[CameraState {
                    view_transform: [0.0; 32],
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });


        let tet_buffer_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tetrahedron Buffer 0"),
            size: MAX_INTERMEDIATE_VERTICES*32,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tet_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tetrahedron Buffer 1"),
            size: 512*(512*32 + 16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tet_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tetrahedron Metadata Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        LiveBuffers {instance_buffer, vertex_intermediate_buffer, camera_uniform_buffer, tet_buffer_0, tet_buffer_1, tet_metadata_buffer}
    }

    fn create_live_bind_group_a(device: &wgpu::Device,
                                layout: &wgpu::BindGroupLayout,
                              live_buffers: &LiveBuffers) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Live Bind Group A"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: CAMERA_BUFFER_IDX,
                    resource: live_buffers.camera_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: INSTANCE_BUFFER_IDX,
                    resource: live_buffers.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: INTERMEDIATE_VERTEX_BUFFER_IDX,
                    resource: live_buffers.vertex_intermediate_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: TET_BUFFER_0_IDX,
                    resource: live_buffers.tet_buffer_0.as_entire_binding(),
                }
            ],
        })
    }

    fn create_live_bind_group_b(device: &wgpu::Device,
                                layout: &wgpu::BindGroupLayout,
                                live_buffers: &LiveBuffers) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Live Bind Group B"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: TET_BUFFER_0_IDX,
                    resource: live_buffers.tet_buffer_0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: TET_BUFFER_1_IDX,
                    resource: live_buffers.tet_buffer_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: TET_METADATA_BUFFER_IDX,
                    resource: live_buffers.tet_metadata_buffer.as_entire_binding(),
                }
            ],
        })
    }

    pub fn new(
        device: &wgpu::Device,
        raster_pipelines: &RasterPipelines,
        render_metadata: &RenderMetadata
    ) -> Self {
        let latest_render_metadata = render_metadata.clone();
        let render_buffers = Self::create_render_buffers(device, &latest_render_metadata);
        let raster_render_bind_group = Self::create_render_bind_group(device, &raster_pipelines.render_buffers_bind_group_raster_layout, &render_buffers);
        let present_render_bind_group = Self::create_render_bind_group(device, &raster_pipelines.render_buffers_bind_group_present_layout, &render_buffers);

        let model_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Model Vertex Buffer"),
                contents: bytemuck::cast_slice(&tesseract::TET_VERTICES),
                usage: wgpu::BufferUsages::STORAGE,
            }
        );
        let one_time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: One Time Buffer Bind Group"),
            layout: &raster_pipelines.one_time_buffers_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: VERTEX_BUFFER_IDX,
                resource: model_vertex_buffer.as_entire_binding(),
            }],
        });

        let live_buffers = Self::create_live_buffers(&device);
        let live_bind_group_a = Self::create_live_bind_group_a(&device, &raster_pipelines.live_buffers_bind_group_layout_a, &live_buffers);
        let live_bind_group_b = Self::create_live_bind_group_b(&device, &raster_pipelines.live_buffers_bind_group_layout_b, &live_buffers);

        Self {
            render_buffers,
            live_buffers,
            model_vertex_buffer,
            one_time_bind_group,
            live_bind_group_a,
            live_bind_group_b,
            raster_render_bind_group,
            present_render_bind_group,
            latest_render_metadata
        }
    }

    pub fn update_render_buffers(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines,
        render_metadata: &RenderMetadata
    ) {
        let render_buffers = Self::create_render_buffers(device, &render_metadata);
        let raster_render_bind_group = Self::create_render_bind_group(device, &pipelines.render_buffers_bind_group_raster_layout, &render_buffers);
        let present_render_bind_group = Self::create_render_bind_group(device, &pipelines.render_buffers_bind_group_present_layout, &render_buffers);
        self.render_buffers = render_buffers;
        self.raster_render_bind_group = raster_render_bind_group;
        self.present_render_bind_group = present_render_bind_group;
        self.latest_render_metadata = render_metadata.clone();
    }

    pub fn update_live_buffers(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines
    ) {
        self.live_buffers = Self::create_live_buffers(&device);
        self.live_bind_group_a = Self::create_live_bind_group_a(&device, &pipelines.live_buffers_bind_group_layout_a, &self.live_buffers);
        self.live_bind_group_b = Self::create_live_bind_group_b(&device, &pipelines.live_buffers_bind_group_layout_b, &self.live_buffers);
    }
}

impl<'a> RasterPipelines {
    pub fn record_cpass<'pass>(
        &'a self,
        cpass: &mut wgpu::ComputePass<'pass>,
        bindings: &'a RasterBindings,
        num_vertices: usize,
    ) where
        'a: 'pass,
    {
        let len = bindings.latest_render_metadata.render_width*bindings.latest_render_metadata.render_height;
        let subgroup_size = 256;
        let padded_size = (subgroup_size - len % subgroup_size) % subgroup_size;

        cpass.set_pipeline(&self.clear_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.raster_render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group_a, &[]);
        cpass.dispatch_workgroups((len + padded_size) / subgroup_size, 1, 1);

        let subgroup_size = 256;
        let padded_size = (subgroup_size - (num_vertices % subgroup_size)) % subgroup_size;
        let dispatch_size = (num_vertices + padded_size) / subgroup_size;
        cpass.set_pipeline(&self.vertex_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.raster_render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group_a, &[]);
        cpass.dispatch_workgroups(dispatch_size as u32, 1, 1);


        let num_tets = num_vertices/4;
        cpass.set_pipeline(&self.lines_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.raster_render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group_a, &[]);
        cpass.dispatch_workgroups(num_tets as u32, 1, 1);

        cpass.set_pipeline(&self.tet_pre_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.raster_render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group_a, &[]);
        cpass.dispatch_workgroups(((num_tets as u32)/256) + 1, 1, 1);

        cpass.set_pipeline(&self.sort_0_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.raster_render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group_b, &[]);
        cpass.dispatch_workgroups(160, 1, 1);

    }

    pub fn record_rpass<'pass>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'pass>,
        bindings: &'a RasterBindings) {

        rpass.set_pipeline(&self.present_pipeline);
        rpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.present_render_bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }

    pub fn do_render(&'a self,
                     raster_bindings: &'a RasterBindings,
                     queue: &wgpu::Queue,
                     encoder: &mut wgpu::CommandEncoder,
                     view: &wgpu::TextureView,
                     model_instances: &[ModelInstance],
                     camera_state: &CameraState) {


        let vertex_count = tesseract::TET_VERTICES.len()*model_instances.len();
        let tetrahedron_count = vertex_count/4;

        queue.write_buffer(&raster_bindings.live_buffers.instance_buffer, 0, bytemuck::cast_slice(model_instances));
        queue.write_buffer(&raster_bindings.live_buffers.camera_uniform_buffer, 0, bytemuck::cast_slice(&[*camera_state]));
        queue.write_buffer(&raster_bindings.live_buffers.tet_metadata_buffer, 0, bytemuck::cast_slice(&[TetrahedronMetadataInput{
            tetrahedron_count: tetrahedron_count as u32,
            processed_tet_count: 0
        }]));

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            self.record_cpass(&mut cpass, &raster_bindings, vertex_count);
        }

        {
            let mut rpass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            self.record_rpass(&mut rpass, &raster_bindings);
        }
    }
}