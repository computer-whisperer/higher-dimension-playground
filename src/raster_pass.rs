pub struct RasterPipelines {
    clear_pipeline: wgpu::ComputePipeline,
    vertex_pipeline: wgpu::ComputePipeline,
    depth_pipeline: wgpu::ComputePipeline,
    pixel_pipeline: wgpu::ComputePipeline
}

impl RasterPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let output_color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let vertex_input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Vertex Input Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let vertex_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Vertex Output Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let depth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Depth Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let screen_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raster: Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
                &output_color_bind_group_layout,
                &depth_bind_group_layout,
                &vertex_input_bind_group_layout,
                &screen_uniform_bind_group_layout,
                &camera_bind_group_layout,
                &vertex_output_bind_group_layout,
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
        cpass.set_bind_group(0, &bindings.color_buffer, &[]);
        cpass.set_bind_group(1, &bindings.depth_buffer, &[]);
        cpass.set_bind_group(2, &bindings.vertex_input_buffer, &[]);
        cpass.set_bind_group(3, &bindings.screen_uniform, &[]);
        cpass.set_bind_group(4, &bindings.camera_uniform, &[]);
        cpass.set_bind_group(5, &bindings.vertex_output_buffer, &[]);
        cpass.dispatch_workgroups((len + padded_size) / subgroup_size, 1, 1);

        let subgroup_size = 256;
        let padded_size = (subgroup_size - (num_vertices % subgroup_size)) % subgroup_size;
        let dispatch_size = (num_vertices + padded_size) / subgroup_size;
        cpass.set_pipeline(&self.vertex_pipeline);
        cpass.set_bind_group(0, &bindings.color_buffer, &[]);
        cpass.set_bind_group(1, &bindings.depth_buffer, &[]);
        cpass.set_bind_group(2, &bindings.vertex_input_buffer, &[]);
        cpass.set_bind_group(3, &bindings.screen_uniform, &[]);
        cpass.set_bind_group(4, &bindings.camera_uniform, &[]);
        cpass.set_bind_group(5, &bindings.vertex_output_buffer, &[]);
        cpass.dispatch_workgroups(dispatch_size as u32, 1, 1);

        let num_tets = num_vertices/4;

        cpass.set_pipeline(&self.depth_pipeline);
        cpass.set_bind_group(0, &bindings.color_buffer, &[]);
        cpass.set_bind_group(1, &bindings.depth_buffer, &[]);
        cpass.set_bind_group(2, &bindings.vertex_input_buffer, &[]);
        cpass.set_bind_group(3, &bindings.screen_uniform, &[]);
        cpass.set_bind_group(4, &bindings.camera_uniform, &[]);
        cpass.set_bind_group(5, &bindings.vertex_output_buffer, &[]);
        cpass.dispatch_workgroups((num_tets) as u32, 1, 1);

        cpass.set_pipeline(&self.pixel_pipeline);
        cpass.set_bind_group(0, &bindings.color_buffer, &[]);
        cpass.set_bind_group(1, &bindings.depth_buffer, &[]);
        cpass.set_bind_group(2, &bindings.vertex_input_buffer, &[]);
        cpass.set_bind_group(3, &bindings.screen_uniform, &[]);
        cpass.set_bind_group(4, &bindings.camera_uniform, &[]);
        cpass.set_bind_group(5, &bindings.vertex_output_buffer, &[]);
        cpass.dispatch_workgroups((num_tets) as u32, 1, 1);
    }
}

pub struct RasterBindings {
    pub color_buffer: wgpu::BindGroup,
    vertex_input_buffer: wgpu::BindGroup,
    vertex_output_buffer: wgpu::BindGroup,
    depth_buffer: wgpu::BindGroup,
    screen_uniform: wgpu::BindGroup,
    camera_uniform: wgpu::BindGroup,
}

impl RasterBindings {
    pub fn new(
        device: &wgpu::Device,
        raster_pipelines: &RasterPipelines,
        color_buffer: &wgpu::Buffer,
        vertex_input_buffer: &wgpu::Buffer,
        vertex_output_buffer: &wgpu::Buffer,
        screen_uniform: &wgpu::Buffer,
        camera_uniform: &wgpu::Buffer,
        depth_buffer: &wgpu::Buffer
    ) -> Self {
        let color_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Output Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
        let depth_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Depth Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: depth_buffer.as_entire_binding(),
            }],
        });
        let vertex_input_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Vertex Input Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(2),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_input_buffer.as_entire_binding(),
            }],
        });
        let screen_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Uniform Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(3),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform.as_entire_binding(),
            }],
        });
        let camera_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Camera Uniform Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(4),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform.as_entire_binding(),
            }],
        });
        let vertex_output_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Vertex Output Buffer Bind Group"),
            layout: &raster_pipelines.vertex_pipeline.get_bind_group_layout(5),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_output_buffer.as_entire_binding(),
            }],
        });
        Self {
            color_buffer,
            vertex_input_buffer,
            vertex_output_buffer,
            screen_uniform,
            camera_uniform,
            depth_buffer
        }
    }

    pub fn update_color_buffer(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines,
        color_buffer: &wgpu::Buffer,
    ) {
        self.color_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Output Buffer Bind Group"),
            layout: &pipelines.vertex_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
    }

    pub fn update_depth_buffer(
        &mut self,
        device: &wgpu::Device,
        pipelines: &RasterPipelines,
        depth_buffer: &wgpu::Buffer,
    ) {
        self.depth_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raster: Depth Buffer Bind Group"),
            layout: &pipelines.vertex_pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: depth_buffer.as_entire_binding(),
            }],
        });
    }
}