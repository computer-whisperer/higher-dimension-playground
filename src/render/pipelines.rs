use super::*;

pub(super) struct PresentPipelineContext {
    pub(super) line_pipeline: Arc<GraphicsPipeline>,
    pub(super) buffer_pipeline: Arc<GraphicsPipeline>,
    pub(super) pipeline_layout: Arc<PipelineLayout>,
    pub(super) hud_pipeline: Arc<GraphicsPipeline>,
    pub(super) hud_pipeline_layout: Arc<PipelineLayout>,
}

impl PresentPipelineContext {
    pub(super) fn new(
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
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::VERTEX,
                    offset: 0,
                    size: 4, // u32 vertex base into HUD SSBO
                }],
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
                    dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                        .into_iter()
                        .collect(),
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

pub(super) struct ComputePipelineContext {
    pub(super) tetrahedron_pipeline: Arc<ComputePipeline>,
    pub(super) edge_pipeline: Arc<ComputePipeline>,
    pub(super) tetrahedron_pixel_pipeline: Arc<ComputePipeline>,
    pub(super) bin_tets_pipeline: Arc<ComputePipeline>,
    pub(super) raytrace_pre_pipeline: Arc<ComputePipeline>,
    pub(super) raytrace_pixel_pipeline: Arc<ComputePipeline>,
    pub(super) raytrace_clear_pipeline: Arc<ComputePipeline>,
    // Voxel traversal engine (VTE) pipelines
    pub(super) voxel_trace_stage_a_pipeline: Arc<ComputePipeline>,
    pub(super) voxel_display_stage_b_pipeline: Arc<ComputePipeline>,
    // BVH pipelines
    pub(super) bvh_scene_bounds_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_morton_codes_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_bitonic_sort_local_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_bitonic_sort_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_bitonic_sort_local_merge_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_init_leaves_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_build_tree_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_compute_leaf_aabbs_pipeline: Arc<ComputePipeline>,
    pub(super) bvh_propagate_aabbs_pipeline: Arc<ComputePipeline>,
    pub(super) pipeline_layout: Arc<PipelineLayout>,
}

impl ComputePipelineContext {
    pub(super) fn new(
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

