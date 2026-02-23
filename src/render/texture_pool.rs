use super::*;

/// Maximum number of 3D textures that can be in the pool simultaneously.
pub const MAX_TEXTURE_POOL_SLOTS: usize = 256;

struct TextureSlot {
    image_view: Arc<ImageView>,
}

pub(super) struct TexturePool {
    slots: Vec<Option<TextureSlot>>,
    dummy_view: Arc<ImageView>,
    sampler: Arc<Sampler>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_set: Arc<DescriptorSet>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
}

impl TexturePool {
    pub(super) fn new(
        device: Arc<Device>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        queue: Arc<Queue>,
    ) -> Self {
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_layout = Self::create_descriptor_set_layout(device.clone());

        // Create a 1x1x1 dummy 3D texture for empty slots.
        let dummy_view = create_dummy_3d_texture_view(
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
        );

        let slots: Vec<Option<TextureSlot>> = (0..MAX_TEXTURE_POOL_SLOTS).map(|_| None).collect();

        // Build initial descriptor set: all slots point to the dummy texture.
        let image_views: Vec<Arc<ImageView>> = (0..MAX_TEXTURE_POOL_SLOTS)
            .map(|_| dummy_view.clone())
            .collect();
        let descriptor_set = Self::build_descriptor_set(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            &sampler,
            &image_views,
        );

        Self {
            slots,
            dummy_view,
            sampler,
            descriptor_set_layout,
            descriptor_set,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            queue,
        }
    }

    pub(super) fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
        let mut bindings = BTreeMap::new();

        // Binding 0: Sampler (separate)
        bindings.insert(
            0,
            DescriptorSetLayoutBinding {
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
            },
        );

        // Binding 1: SampledImage array[256], partially bound
        bindings.insert(
            1,
            DescriptorSetLayoutBinding {
                binding_flags: DescriptorBindingFlags::PARTIALLY_BOUND,
                descriptor_count: MAX_TEXTURE_POOL_SLOTS as u32,
                stages: ShaderStages::COMPUTE,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
            },
        );

        DescriptorSetLayout::new(
            device,
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn build_descriptor_set(
        allocator: Arc<StandardDescriptorSetAllocator>,
        layout: Arc<DescriptorSetLayout>,
        sampler: &Arc<Sampler>,
        image_views: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            allocator,
            layout,
            [
                WriteDescriptorSet::sampler(0, sampler.clone()),
                WriteDescriptorSet::image_view_array(
                    1,
                    0,
                    image_views.iter().cloned(),
                ),
            ],
            [],
        )
        .unwrap()
    }

    /// Upload a 3D texture into the pool. Returns the pool index (0..255) to use in material tokens.
    pub(super) fn upload_texture_3d(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        depth: u32,
        format: Format,
    ) -> Option<u16> {
        // Find a free slot.
        let slot_index = self.slots.iter().position(|s| s.is_none())?;

        let view = create_3d_texture_view(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            self.queue.clone(),
            width,
            height,
            depth,
            format,
            data,
        );

        self.slots[slot_index] = Some(TextureSlot {
            image_view: view.clone(),
        });

        // Rebuild the descriptor set with the new image view.
        self.rebuild_descriptor_set();

        Some(slot_index as u16)
    }

    /// Remove a texture from the pool, replacing it with the dummy texture.
    pub(super) fn remove_texture(&mut self, index: u16) {
        let idx = index as usize;
        if idx < self.slots.len() {
            self.slots[idx] = None;
            self.rebuild_descriptor_set();
        }
    }

    fn rebuild_descriptor_set(&mut self) {
        let image_views: Vec<Arc<ImageView>> = self
            .slots
            .iter()
            .map(|slot| match slot {
                Some(s) => s.image_view.clone(),
                None => self.dummy_view.clone(),
            })
            .collect();
        self.descriptor_set = Self::build_descriptor_set(
            self.descriptor_set_allocator.clone(),
            self.descriptor_set_layout.clone(),
            &self.sampler,
            &image_views,
        );
    }

    pub(super) fn descriptor_set_layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.descriptor_set_layout
    }

    pub(super) fn descriptor_set(&self) -> &Arc<DescriptorSet> {
        &self.descriptor_set
    }
}

fn create_dummy_3d_texture_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
) -> Arc<ImageView> {
    // 1x1x1 RGBA8 magenta (visible debug color)
    let pixels: [u8; 4] = [255, 0, 255, 255];
    create_3d_texture_view(
        memory_allocator,
        command_buffer_allocator,
        queue,
        1,
        1,
        1,
        Format::R8G8B8A8_SRGB,
        &pixels,
    )
}

fn create_3d_texture_view(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    width: u32,
    height: u32,
    depth: u32,
    format: Format,
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

    let image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format,
            extent: [width.max(1), height.max(1), depth.max(1)],
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
            image.clone(),
        ))
        .unwrap();
    let upload_cmd = upload_builder.build().unwrap();
    let upload_future = sync::now(queue.device().clone())
        .then_execute(queue.clone(), upload_cmd)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    upload_future.wait(None).unwrap();

    ImageView::new_default(image).unwrap()
}
