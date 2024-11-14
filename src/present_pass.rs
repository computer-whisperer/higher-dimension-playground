pub struct PresentPass {
    pipeline: wgpu::RenderPipeline,
}

impl PresentPass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let output_color_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Present: Output Buffer Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Present: Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Present Pipeline Layout"),
            bind_group_layouts: &[&output_color_bind_group_layout, &uniform_bind_group],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::include_wgsl!("present.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Present Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        Self { pipeline }
    }
}

pub struct PresentBindings {
    screen_uniform: wgpu::BindGroup,
    color_buffer: wgpu::BindGroup,
}

impl PresentBindings {
    pub fn new(
        device: &wgpu::Device,
        PresentPass { pipeline }: &PresentPass,
        color_buffer: &wgpu::Buffer,
        screen_uniform: &wgpu::Buffer,
    ) -> Self {
        let color_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Present: Output Buffer Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
        let screen_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Present: Screen Uniform Bind Group"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_uniform.as_entire_binding(),
            }],
        });
        Self {
            color_buffer,
            screen_uniform,
        }
    }

    pub fn update_color_buffer(
        &mut self,
        device: &wgpu::Device,
        PresentPass { pipeline }: &PresentPass,
        color_buffer: &wgpu::Buffer,
    ) {
        self.color_buffer = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Present: Output Buffer Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
    }
}

impl<'a> PresentPass {
    pub fn record<'pass>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'pass>,
        bindings: &'a PresentBindings,
    ) where
        'a: 'pass,
    {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bindings.color_buffer, &[]);
        rpass.set_bind_group(1, &bindings.screen_uniform, &[]);
        rpass.draw(0..6, 0..1);
    }
}
