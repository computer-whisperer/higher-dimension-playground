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
    pub(super) cpu_bvh_root_buffer: Subbuffer<[common::BVHNode]>,
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
        let tiles_x = render_dimensions[0].div_ceil(8);
        let tiles_y = render_dimensions[1].div_ceil(8);
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
        let cpu_bvh_root_buffer = Buffer::from_iter(
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
            vec![common::BVHNode::zeroed(); 1],
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
            cpu_bvh_root_buffer,
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

pub struct VoxelGpuBuffers {
    pub(super) chunk_headers_buffer: Subbuffer<[GpuVoxelChunkHeader]>,
    pub(super) occupancy_words_buffer: Subbuffer<[u32]>,
    pub(super) material_words_buffer: Subbuffer<[u32]>,
    pub(super) orientation_words_buffer: Subbuffer<[u32]>,
    pub(super) leaf_headers_buffer: Subbuffer<[vte::GpuVoxelLeafHeader]>,
    pub(super) region_bvh_nodes_buffer: Subbuffer<[vte::GpuVoxelChunkBvhNode]>,
    pub(super) leaf_chunk_entries_buffer: Subbuffer<[u32]>,
    pub(super) macro_words_buffer: Subbuffer<[u32]>,
    pub(super) caps: VoxelBufferCapacities,
}

pub(super) struct LiveBuffers {
    pub(super) model_instance_buffer: Subbuffer<[common::ModelInstance]>,
    pub(super) working_data_buffer: Subbuffer<common::WorkingData>,
    pub(super) voxel_frame_meta_buffer: Subbuffer<vte::GpuVoxelFrameMeta>,
    pub(super) voxel: VoxelGpuBuffers,
    pub(super) vte_compare_stats_buffer: Subbuffer<[u32]>,
    pub(super) vte_first_mismatch_buffer: Subbuffer<[u32]>,
    pub(super) vte_world_bvh_ray_diag_buffer: Subbuffer<[u32]>,
    pub(super) descriptor_set: Arc<DescriptorSet>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VoxelBufferCapacities {
    pub(super) dense_chunks: usize,
    pub(super) leaf_headers: usize,
    pub(super) region_bvh_nodes: usize,
    pub(super) leaf_chunk_entries: usize,
}

impl Default for VoxelBufferCapacities {
    fn default() -> Self {
        Self {
            dense_chunks: vte::VTE_MAX_DENSE_CHUNKS,
            leaf_headers: vte::VTE_REGION_LEAF_CAPACITY,
            region_bvh_nodes: vte::VTE_REGION_BVH_NODE_CAPACITY,
            leaf_chunk_entries: vte::VTE_REGION_LEAF_CHUNK_ENTRY_CAPACITY,
        }
    }
}

impl VoxelBufferCapacities {
    pub(super) fn with_minimums(self) -> Self {
        Self {
            dense_chunks: self.dense_chunks.max(1),
            leaf_headers: self.leaf_headers.max(1),
            region_bvh_nodes: self.region_bvh_nodes.max(1),
            leaf_chunk_entries: self.leaf_chunk_entries.max(1),
        }
    }

    pub(super) fn occupancy_words(self) -> usize {
        self.dense_chunks
            .saturating_mul(vte::VTE_OCCUPANCY_WORDS_PER_CHUNK)
    }

    pub(super) fn material_words(self) -> usize {
        self.dense_chunks
            .saturating_mul(vte::VTE_MATERIAL_WORDS_PER_CHUNK)
    }

    pub(super) fn orientation_words(self) -> usize {
        self.dense_chunks
            .saturating_mul(vte::VTE_ORIENTATION_WORDS_PER_CHUNK)
    }

    pub(super) fn macro_words(self) -> usize {
        self.dense_chunks
            .saturating_mul(vte::VTE_MACRO_WORDS_PER_CHUNK)
    }
}

impl VoxelGpuBuffers {
    pub fn new(memory_allocator: Arc<dyn MemoryAllocator>, caps: VoxelBufferCapacities) -> Self {
        let caps = caps.with_minimums();

        let chunk_headers_buffer = Buffer::from_iter(
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
            vec![GpuVoxelChunkHeader::zeroed(); caps.dense_chunks],
        )
        .unwrap();

        let occupancy_words_buffer = Buffer::from_iter(
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
            vec![0u32; caps.occupancy_words()],
        )
        .unwrap();

        let material_words_buffer = Buffer::from_iter(
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
            vec![0u32; caps.material_words()],
        )
        .unwrap();

        let orientation_words_buffer = Buffer::from_iter(
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
            vec![0u32; caps.orientation_words()],
        )
        .unwrap();

        let leaf_headers_buffer = Buffer::from_iter(
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
            vec![vte::GpuVoxelLeafHeader::zeroed(); caps.leaf_headers],
        )
        .unwrap();

        let region_bvh_nodes_buffer = Buffer::from_iter(
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
            vec![vte::GpuVoxelChunkBvhNode::empty(); caps.region_bvh_nodes],
        )
        .unwrap();

        let leaf_chunk_entries_buffer = Buffer::from_iter(
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
            vec![0u32; caps.leaf_chunk_entries],
        )
        .unwrap();

        let macro_words_buffer = Buffer::from_iter(
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
            vec![0u32; caps.macro_words()],
        )
        .unwrap();

        Self {
            chunk_headers_buffer,
            occupancy_words_buffer,
            material_words_buffer,
            orientation_words_buffer,
            leaf_headers_buffer,
            region_bvh_nodes_buffer,
            leaf_chunk_entries_buffer,
            macro_words_buffer,
            caps,
        }
    }

    /// Create GPU buffers pre-populated with the given CPU data.
    /// Used by the background rebuild thread to build GPU buffers off the main thread.
    pub fn from_data(
        memory_allocator: Arc<dyn MemoryAllocator>,
        chunk_headers: &[GpuVoxelChunkHeader],
        occupancy_words: &[u32],
        material_words: &[u32],
        orientation_words: &[u32],
        macro_words: &[u32],
        leaf_headers: &[vte::GpuVoxelLeafHeader],
        region_bvh_nodes: &[vte::GpuVoxelChunkBvhNode],
        leaf_chunk_entries: &[u32],
    ) -> Self {
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        };
        let buf_info = BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        };

        let caps = VoxelBufferCapacities {
            dense_chunks: chunk_headers.len().max(1),
            leaf_headers: leaf_headers.len().max(1),
            region_bvh_nodes: region_bvh_nodes.len().max(1),
            leaf_chunk_entries: leaf_chunk_entries.len().max(1),
        };

        // Buffer::from_iter requires at least 1 element. Use from_iter with
        // copied iterators so we don't need to clone entire Vecs.
        macro_rules! make_buf {
            ($alloc:expr, $info:expr, $ainfo:expr, $data:expr, $zero:expr) => {
                if $data.is_empty() {
                    Buffer::from_iter($alloc.clone(), $info.clone(), $ainfo.clone(), vec![$zero])
                        .unwrap()
                } else {
                    Buffer::from_iter(
                        $alloc.clone(),
                        $info.clone(),
                        $ainfo.clone(),
                        $data.iter().copied(),
                    )
                    .unwrap()
                }
            };
        }

        let chunk_headers_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            chunk_headers,
            GpuVoxelChunkHeader::zeroed()
        );
        let occupancy_words_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            occupancy_words,
            0u32
        );
        let material_words_buffer =
            make_buf!(memory_allocator, buf_info, alloc_info, material_words, 0u32);
        let orientation_words_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            orientation_words,
            0u32
        );
        let macro_words_buffer =
            make_buf!(memory_allocator, buf_info, alloc_info, macro_words, 0u32);
        let leaf_headers_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            leaf_headers,
            vte::GpuVoxelLeafHeader::zeroed()
        );
        let region_bvh_nodes_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            region_bvh_nodes,
            vte::GpuVoxelChunkBvhNode::empty()
        );
        let leaf_chunk_entries_buffer = make_buf!(
            memory_allocator,
            buf_info,
            alloc_info,
            leaf_chunk_entries,
            0u32
        );

        Self {
            chunk_headers_buffer,
            occupancy_words_buffer,
            material_words_buffer,
            orientation_words_buffer,
            leaf_headers_buffer,
            region_bvh_nodes_buffer,
            leaf_chunk_entries_buffer,
            macro_words_buffer,
            caps,
        }
    }

    pub fn capacities(&self) -> VoxelBufferCapacities {
        self.caps
    }
}

impl LiveBuffers {
    pub(super) fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        Self::new_with_voxel_caps(
            memory_allocator,
            descriptor_set_allocator,
            descriptor_set_layout,
            VoxelBufferCapacities::default(),
        )
    }

    pub(super) fn new_with_voxel_caps(
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
        voxel_caps: VoxelBufferCapacities,
    ) -> Self {
        let voxel = VoxelGpuBuffers::new(memory_allocator.clone(), voxel_caps);

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

        let vte_world_bvh_ray_diag_buffer = Buffer::from_iter(
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
            vec![0u32; vte::VTE_WORLD_BVH_RAY_DIAG_WORD_COUNT],
        )
        .unwrap();

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, model_instance_buffer.clone()),
                WriteDescriptorSet::buffer(1, working_data_buffer.clone()),
                WriteDescriptorSet::buffer(2, voxel_frame_meta_buffer.clone()),
                WriteDescriptorSet::buffer(3, voxel.chunk_headers_buffer.clone()),
                WriteDescriptorSet::buffer(4, voxel.occupancy_words_buffer.clone()),
                WriteDescriptorSet::buffer(5, voxel.material_words_buffer.clone()),
                WriteDescriptorSet::buffer(6, voxel.leaf_headers_buffer.clone()),
                WriteDescriptorSet::buffer(7, voxel.region_bvh_nodes_buffer.clone()),
                WriteDescriptorSet::buffer(8, voxel.leaf_chunk_entries_buffer.clone()),
                WriteDescriptorSet::buffer(9, vte_compare_stats_buffer.clone()),
                WriteDescriptorSet::buffer(10, vte_first_mismatch_buffer.clone()),
                WriteDescriptorSet::buffer(11, voxel.macro_words_buffer.clone()),
                WriteDescriptorSet::buffer(12, vte_world_bvh_ray_diag_buffer.clone()),
                WriteDescriptorSet::buffer(13, voxel.orientation_words_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        Self {
            model_instance_buffer,
            working_data_buffer,
            voxel_frame_meta_buffer,
            voxel,
            vte_compare_stats_buffer,
            vte_first_mismatch_buffer,
            vte_world_bvh_ray_diag_buffer,
            descriptor_set,
        }
    }

    pub(super) fn voxel_capacities(&self) -> VoxelBufferCapacities {
        self.voxel.caps
    }

    pub(super) fn install_voxel_buffers(
        &mut self,
        voxel: VoxelGpuBuffers,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) {
        self.voxel = voxel;
        self.descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout,
            [
                WriteDescriptorSet::buffer(0, self.model_instance_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.working_data_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.voxel_frame_meta_buffer.clone()),
                WriteDescriptorSet::buffer(3, self.voxel.chunk_headers_buffer.clone()),
                WriteDescriptorSet::buffer(4, self.voxel.occupancy_words_buffer.clone()),
                WriteDescriptorSet::buffer(5, self.voxel.material_words_buffer.clone()),
                WriteDescriptorSet::buffer(6, self.voxel.leaf_headers_buffer.clone()),
                WriteDescriptorSet::buffer(7, self.voxel.region_bvh_nodes_buffer.clone()),
                WriteDescriptorSet::buffer(8, self.voxel.leaf_chunk_entries_buffer.clone()),
                WriteDescriptorSet::buffer(9, self.vte_compare_stats_buffer.clone()),
                WriteDescriptorSet::buffer(10, self.vte_first_mismatch_buffer.clone()),
                WriteDescriptorSet::buffer(11, self.voxel.macro_words_buffer.clone()),
                WriteDescriptorSet::buffer(12, self.vte_world_bvh_ray_diag_buffer.clone()),
                WriteDescriptorSet::buffer(13, self.voxel.orientation_words_buffer.clone()),
            ],
            [],
        )
        .unwrap();
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
        for binding in 2..=13u32 {
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
