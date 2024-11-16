pub struct RasterPipelines {
    clear_pipeline: wgpu::ComputePipeline,
    vertex_pipeline: wgpu::ComputePipeline,
    depth_pipeline: wgpu::ComputePipeline,
    pixel_pipeline: wgpu::ComputePipeline
}

const RENDER_BUFFER_GROUP_ID: u32 = 0;
const COLOR_BUFFER_IDX: u32 = 0;
const DEPTH_BUFFER_IDX: u32 = 1;

const ONETIME_BUFFER_GROUP_ID: u32 = 1;
const VERTEX_BUFFER_IDX: u32 = 0;

const LIVE_BUFFER_GROUP_ID: u32 = 2;
const INTERMEDIATE_VERTEX_BUFFER_IDX: u32 = 0;
const INSTANCE_BUFFER_IDX: u32 = 1;
const RENDER_META_BUFFER_IDX: u32 = 2;
const CAMERA_BUFFER_IDX: u32 = 3;

impl RasterPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let render_buffers_bind_group_layout =
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
                }],
            });
        let one_time_bind_group_layout =
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
                }],
            });
        let live_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Intermediate Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry { // Screen
                    binding: RENDER_META_BUFFER_IDX,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raster Pipeline Layout"),
            bind_group_layouts: &[
                &render_buffers_bind_group_layout,
                &one_time_bind_group_layout,
                &live_bind_group_layout
            ],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("raster.wgsl"));
        let vertex_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Vertex Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("raster_vertex_shader"),
            compilation_options: Default::default(),
            cache: None,
        });
        let depth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Depth Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("raster_depth_shader"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pixel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raster Pixel Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("raster_pixel_shader"),
            compilation_options: Default::default(),
            cache: None,
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("clear"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            clear_pipeline,
            vertex_pipeline,
            depth_pipeline,
            pixel_pipeline
        }
    }
}

pub struct RasterBindings {
    render_bind_group: wgpu::BindGroup,
    one_time_bind_group: wgpu::BindGroup,
    live_bind_group: wgpu::BindGroup,
}

impl RasterBindings {
    pub fn new(
        device: &wgpu::Device,
        raster_pipelines: &RasterPipelines,
        color_buffer: &wgpu::Buffer,
        depth_buffer: &wgpu::Buffer,
        vertex_input_buffer: &wgpu::Buffer,
        vertex_intermediate_buffer: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        render_meta_uniform: &wgpu::Buffer,
        camera_uniform: &wgpu::Buffer,
    ) -> Self {
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Output Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(RENDER_BUFFER_GROUP_ID),
            entries: &[
            wgpu::BindGroupEntry {
                binding: COLOR_BUFFER_IDX,
                resource: color_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: DEPTH_BUFFER_IDX,
                resource: depth_buffer.as_entire_binding(),
            }],
        });
        let one_time_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: One Time Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(ONETIME_BUFFER_GROUP_ID),
            entries: &[wgpu::BindGroupEntry {
                binding: VERTEX_BUFFER_IDX,
                resource: vertex_input_buffer.as_entire_binding(),
            }],
        });
        let live_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Live Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(LIVE_BUFFER_GROUP_ID),
            entries: &[
            wgpu::BindGroupEntry {
                binding: RENDER_META_BUFFER_IDX,
                resource: render_meta_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: CAMERA_BUFFER_IDX,
                resource: camera_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: INSTANCE_BUFFER_IDX,
                resource: instance_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: INTERMEDIATE_VERTEX_BUFFER_IDX,
                resource: vertex_intermediate_buffer.as_entire_binding(),
            }],
        });

        Self {
            render_bind_group,
            one_time_bind_group,
            live_bind_group,
        }
    }

    pub fn update_render_buffers(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines,
        color_buffer: &wgpu::Buffer,
        depth_buffer: &wgpu::Buffer,
    ) {
        self.render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Output Buffer Bind Group"),
            layout: &pipelines.vertex_pipeline.get_bind_group_layout(RENDER_BUFFER_GROUP_ID),
            entries: &[
            wgpu::BindGroupEntry {
                binding: COLOR_BUFFER_IDX,
                resource: color_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: DEPTH_BUFFER_IDX,
                resource: depth_buffer.as_entire_binding(),
            }],
        });
    }

    pub fn update_live_buffers(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines,
        render_meta_uniform: &wgpu::Buffer,
        camera_uniform: &wgpu::Buffer,
        instance_buffer: &wgpu::Buffer,
        vertex_intermediate_buffer: &wgpu::Buffer
    ) {
        self.live_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Live Buffer Bind Group"),
            layout: &pipelines.vertex_pipeline.get_bind_group_layout(LIVE_BUFFER_GROUP_ID),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: RENDER_META_BUFFER_IDX,
                    resource: render_meta_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: CAMERA_BUFFER_IDX,
                    resource: camera_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: INSTANCE_BUFFER_IDX,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: INTERMEDIATE_VERTEX_BUFFER_IDX,
                    resource: vertex_intermediate_buffer.as_entire_binding(),
                }],
        });
    }
}

impl<'a> RasterPipelines {
    pub fn record<'pass>(
        &'a self,
        cpass: &mut wgpu::ComputePass<'pass>,
        bindings: &'a RasterBindings,
        width: u32,
        height: u32,
        num_vertices: usize,
    ) where
        'a: 'pass,
    {
        let len = width*height;
        let subgroup_size = 256;
        let padded_size = (subgroup_size - len % subgroup_size) % subgroup_size;

        cpass.set_pipeline(&self.clear_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group, &[]);
        cpass.dispatch_workgroups((len + padded_size) / subgroup_size, 1, 1);

        let subgroup_size = 256;
        let padded_size = (subgroup_size - (num_vertices % subgroup_size)) % subgroup_size;
        let dispatch_size = (num_vertices + padded_size) / subgroup_size;
        cpass.set_pipeline(&self.vertex_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_size as u32, 1, 1);

        let num_tets = num_vertices/4;

        cpass.set_pipeline(&self.depth_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group, &[]);
        cpass.dispatch_workgroups((num_tets) as u32, 1, 1);

        cpass.set_pipeline(&self.pixel_pipeline);
        cpass.set_bind_group(RENDER_BUFFER_GROUP_ID, &bindings.render_bind_group, &[]);
        cpass.set_bind_group(ONETIME_BUFFER_GROUP_ID, &bindings.one_time_bind_group, &[]);
        cpass.set_bind_group(LIVE_BUFFER_GROUP_ID, &bindings.live_bind_group, &[]);
        cpass.dispatch_workgroups((num_tets) as u32, 1, 1);
    }
}