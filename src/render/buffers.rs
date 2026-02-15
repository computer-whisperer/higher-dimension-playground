use super::*;

pub(super) struct OneTimeBuffers {
    pub(super) model_tetrahedron_count: usize,
    pub(super) model_edge_count: usize,
    pub(super) descriptor_set: Arc<DescriptorSet>,
}

impl OneTimeBuffers {
    pub(super) fn new(
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
            model_tetrahedron_count,
            model_edge_count,
            descriptor_set,
        }
    }

    pub(super) fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
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

pub(super) struct SizedBuffers {
    pub(super) render_dimensions: [u32; 3],
    pub(super) pixel_storage_layers: u32,
    pub(super) max_tetrahedrons: usize,
    pub(super) output_tetrahedron_buffer: Subbuffer<[common::Tetrahedron]>,
    pub(super) output_pixel_buffer: Subbuffer<[Vec4]>,
    pub(super) morton_codes_buffer: Subbuffer<[common::MortonCode]>,
    pub(super) bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    pub(super) scene_bounds_buffer: Subbuffer<[common::SceneBounds]>,
    pub(super) atomic_counter_buffer: Subbuffer<[u32]>,
    pub(super) output_cpu_pixel_buffer: Subbuffer<[Vec4]>,
    // Tile binning buffers
    pub(super) tile_tet_counts_buffer: Subbuffer<[u32]>,
    pub(super) tile_tet_indices_buffer: Subbuffer<[u32]>,
    // Debug readback buffers
    pub(super) cpu_bvh_nodes_buffer: Subbuffer<[common::BVHNode]>,
    pub(super) cpu_morton_codes_buffer: Subbuffer<[common::MortonCode]>,
}

impl SizedBuffers {
    pub(super) fn new(
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
            output_cpu_pixel_buffer,
            tile_tet_counts_buffer,
            tile_tet_indices_buffer,
            cpu_bvh_nodes_buffer,
            cpu_morton_codes_buffer,
        }
    }

    pub(super) fn create_sized_descriptor_set(
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

    pub(super) fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
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

pub(super) struct LiveBuffers {
    pub(super) model_instance_buffer: Subbuffer<[common::ModelInstance]>,
    pub(super) working_data_buffer: Subbuffer<common::WorkingData>,
    pub(super) voxel_frame_meta_buffer: Subbuffer<vte::GpuVoxelFrameMeta>,
    pub(super) voxel_chunk_headers_buffer: Subbuffer<[GpuVoxelChunkHeader]>,
    pub(super) voxel_occupancy_words_buffer: Subbuffer<[u32]>,
    pub(super) voxel_material_words_buffer: Subbuffer<[u32]>,
    pub(super) voxel_macro_words_buffer: Subbuffer<[u32]>,
    pub(super) voxel_visible_chunk_indices_buffer: Subbuffer<[u32]>,
    pub(super) voxel_chunk_lookup_buffer: Subbuffer<[vte::GpuVoxelChunkLookupEntry]>,
    pub(super) voxel_y_slice_bounds_buffer: Subbuffer<[GpuVoxelYSliceBounds]>,
    pub(super) voxel_y_slice_lookup_entries_buffer: Subbuffer<[u32]>,
    pub(super) vte_compare_stats_buffer: Subbuffer<[u32]>,
    pub(super) vte_first_mismatch_buffer: Subbuffer<[u32]>,
    pub(super) descriptor_set: Arc<DescriptorSet>,
}

impl LiveBuffers {
    pub(super) fn new(
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
            vte::GpuVoxelFrameMeta::zeroed(),
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
            vec![GpuVoxelChunkHeader::zeroed(); vte::VTE_MAX_CHUNKS],
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
            vec![0u32; vte::VTE_MAX_CHUNKS * vte::VTE_OCCUPANCY_WORDS_PER_CHUNK],
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
            vec![0u32; vte::VTE_MAX_CHUNKS * vte::VTE_MATERIAL_WORDS_PER_CHUNK],
        )
        .unwrap();

        let voxel_macro_words_buffer = Buffer::from_iter(
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
            vec![0u32; vte::VTE_MAX_CHUNKS * vte::VTE_MACRO_WORDS_PER_CHUNK],
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
            vec![0u32; vte::VTE_MAX_CHUNKS],
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
            vec![vte::GpuVoxelChunkLookupEntry::empty(); vte::VTE_CHUNK_LOOKUP_CAPACITY],
        )
        .unwrap();

        let voxel_y_slice_bounds_buffer = Buffer::from_iter(
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
            vec![GpuVoxelYSliceBounds::zeroed(); vte::VTE_MAX_Y_SLICES],
        )
        .unwrap();

        let voxel_y_slice_lookup_entries_buffer = Buffer::from_iter(
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
            vec![0u32; vte::VTE_MAX_Y_SLICE_LOOKUP_ENTRIES],
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
            vec![0u32; vte::VTE_COMPARE_STATS_WORD_COUNT],
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
            vec![0u32; vte::VTE_FIRST_MISMATCH_WORD_COUNT],
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
                WriteDescriptorSet::buffer(8, voxel_y_slice_bounds_buffer.clone()),
                WriteDescriptorSet::buffer(9, vte_compare_stats_buffer.clone()),
                WriteDescriptorSet::buffer(10, vte_first_mismatch_buffer.clone()),
                WriteDescriptorSet::buffer(11, voxel_macro_words_buffer.clone()),
                WriteDescriptorSet::buffer(12, voxel_y_slice_lookup_entries_buffer.clone()),
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
            voxel_macro_words_buffer,
            voxel_visible_chunk_indices_buffer,
            voxel_chunk_lookup_buffer,
            voxel_y_slice_bounds_buffer,
            voxel_y_slice_lookup_entries_buffer,
            vte_compare_stats_buffer,
            vte_first_mismatch_buffer,
            descriptor_set,
        }
    }

    pub(super) fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
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
        for binding in 2..=12u32 {
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
