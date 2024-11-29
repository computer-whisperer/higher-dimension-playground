#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(const_for)]
#![feature(effects)]
#![feature(const_mut_refs)]

mod render;
mod hypercube;
mod matrix_operations;

use std::{error::Error, sync::Arc};
use std::default::Default;
use std::time::Instant;
use vulkano::{buffer::{Buffer, BufferContents}, command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
}, device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
    QueueCreateInfo, QueueFlags,
}, image::{view::ImageView, Image, ImageUsage}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
    graphics::{
        color_blend::{ColorBlendAttachmentState, ColorBlendState},
        input_assembly::InputAssemblyState,
        multisample::MultisampleState,
        rasterization::RasterizationState,
        vertex_input::{Vertex, VertexDefinition},
        viewport::{Viewport, ViewportState},
        GraphicsPipelineCreateInfo,
    }
    ,
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
}, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, swapchain::{
    acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
}, sync::{self, GpuFuture}, Validated, VulkanError, VulkanLibrary};
use vulkano::device::DeviceFeatures;
use vulkano::instance::debug::{DebugUtilsMessenger, DebugUtilsMessengerCallback};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::shader::ShaderStages;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use render::RenderContext;
use crate::matrix_operations::{rotation_matrix_4d_rotate_1, rotation_matrix_one_angle, scale_matrix_4d, translate_matrix_4d};

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    start_time: Instant,
    _callback: Option<DebugUtilsMessenger>
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        
        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw
        // to a window.
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        ).unwrap();

        use vulkano::instance::debug::{
            DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        };

        let _callback = unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo::user_callback(
                    DebugUtilsMessengerCallback::new(|message_severity, message_type, callback_data| {
                        println!("Debug callback: {:?}", callback_data.message);
                    }),
                ),
            ).ok()
        };

        // Choose device extensions that we're going to use. In order to present images to a
        // surface, we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        
        let device_features = DeviceFeatures {
            fill_mode_non_solid: true,
            vulkan_memory_model: true,
            variable_pointers: true,
            variable_pointers_storage_buffer: true,
            .. Default::default()
        };

        // We then choose which physical device to use. First, we enumerate all the available
        // physical devices, then apply filters to narrow them down to those that can support our
        // needs.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // Some devices may not support the extensions or features that your application,
                // or report properties and limits that are not sufficient for your application.
                // These should be filtered out here.
                p.supported_extensions().contains(&device_extensions) &&
                p.supported_features().contains(&device_features)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will
                // execute our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example a
                // draw queue and a compute queue), similar to CPU threads. This is
                // something you have to have to manage manually in Vulkan. Queues
                // of the same type belong to the same queue family.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue
                // to handle data transfers in parallel with graphics operations.
                // You may also need a separate queue for compute operations, if
                // your application uses those.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing
                        // to a window surface, as we do in this example, we also need to check
                        // that queues in this queue family are capable of presenting images to the
                        // surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    // The code here searches for the first queue family that is suitable. If none
                    // is found, `None` is returned to `filter_map`, which
                    // disqualifies this physical device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the
            // application. However, not every device is equal, some are preferred over others.
            // Now, we assign each physical device a score, and pick the device with the lowest
            // ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a
            // "default" or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // An iterator of created queues is returned by the function alongside the device.
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device,
            DeviceCreateInfo {
                // A list of optional features and extensions that our program needs to work
                // correctly. Some parts of the Vulkan specs are optional and must be enabled
                // manually at device creation. In this example the only thing we are going to need
                // is the `khr_swapchain` extension that allows us to draw to a window.
                enabled_extensions: device_extensions,
                
                enabled_features: device_features,

                // The list of queues that we are going to use. Here we only use one queue, from
                // the previously chosen queue family.
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
            .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
        // only use one queue in this example, so we just retrieve the first and only element of
        // the iterator.
        let queue = queues.next().unwrap();


        let rcx = None;

        App {
            _callback,
            instance,
            device,
            queue,
            rcx,
            start_time: Instant::now()
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.rcx = Some(RenderContext::new(self.device.clone(), self.instance.clone(), window));
        let start_time = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain();
            }
            WindowEvent::RedrawRequested => {
                let time_elapsed = self.start_time.elapsed().as_secs_f32();

                let view_matrix = translate_matrix_4d(0.0, 0.0, 4.0, 4.0);
                //let view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 1, time_elapsed / 3.0));
                let view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 2, time_elapsed / 4.0));
                //let view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 3, time_elapsed / 5.0));

                let mut instances = Vec::<common::ModelInstance>::new();

                let block_textures = [
                    [10, 10, 10, 10, 10, 10, 11, 10], // 0
                    [1, 1, 1, 1, 1, 1, 1, 1], // 1
                    [2, 2, 2, 2, 2, 2, 2, 2], // 2
                    [1, 2, 3, 4, 5, 6, 7, 8], // 3
                    [9, 9, 4, 4, 9, 9, 9, 9], // 4
                    [12, 12, 12, 12, 12, 12, 12, 12] // 5
                ];

                struct Block {
                    position: [i32; 4],
                    texture: usize
                }

                let mut blocks = Vec::<Block>::new();

                
                for x in 0..2 {
                    for y in 0..2 {
                        for z in 0..2 {
                            for w in 0..2 {
                                blocks.push(
                                    Block{
                                        position: [x*2 - 1, y*2 - 1, z*2 - 1, w*2 - 1],
                                        texture: 5
                                    }
                                );
                            }
                        }
                    }
                }
                
                blocks.push(
                    Block{
                        position: [0, 0, 0, 0],
                        texture: 3
                    }
                );

                for block in blocks {
                    let model_scale = 1.0;
                    let model_transform = 
                        translate_matrix_4d(
                            (block.position[0] as f32 - 0.5)*model_scale,
                            (block.position[1] as f32 - 0.5)*model_scale,
                            (block.position[2] as f32 - 0.5)*model_scale,
                            (block.position[3] as f32 - 0.5)*model_scale)
                        .dot(&scale_matrix_4d(model_scale));
                    instances.push(
                        common::ModelInstance{
                            model_transform: model_transform.into(),
                            cell_texture_ids: block_textures[block.texture],
                        }
                    );
                }
                
                
                rcx.render(self.device.clone(), self.queue.clone(), view_matrix, &instances);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

