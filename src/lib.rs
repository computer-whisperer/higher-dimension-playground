mod raster_pass;
mod matrix_operations;
mod tesseract;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use web_time::{Instant};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
};
use winit::window::WindowBuilder;
use crate::matrix_operations::*;
use crate::raster_pass::{CameraState, ModelInstance, RasterBindings, RasterPipelines, RenderMetadata};

const MAX_RENDER_WIDTH : u32 = 1920;
const MAX_RENDER_HEIGHT : u32 = 1080;
const MAX_SCREEN_WIDTH : u32 = 3840;
const MAX_SCREEN_HEIGHT : u32 = 2160;
const DEPTH_FACTOR: u32 = 128;

async fn arun() {

    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Debug).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut size = window.inner_size();

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let mut required_features = wgpu::Features::empty();
    let mut required_limits = wgpu::Limits::downlevel_defaults();
    required_limits.max_texture_dimension_1d = 4096;
    required_limits.max_texture_dimension_2d = 4096;
    //required_limits.max_bind_groups = 8;
    required_limits.max_buffer_size = 1073741824;
    required_limits.max_storage_buffer_binding_size = 1073741824;

    //required_limits.max_storage_buffers_per_shader_stage = 2;
    //required_limits.max_storage_buffer_binding_size = 0x40000000;
    required_limits.max_compute_workgroup_size_x = 1024;
    required_limits.max_compute_workgroup_size_y = 1;
    required_limits.max_compute_workgroup_size_z = 1;
    required_limits.max_compute_invocations_per_workgroup = 1024;
    //required_limits.max_compute_workgroups_per_dimension = 2048;

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let mut render_metadata = RenderMetadata {
        screen_width: size.width.min(MAX_SCREEN_WIDTH),
        screen_height: size.height.min(MAX_SCREEN_HEIGHT),
        render_width: MAX_RENDER_WIDTH,
        render_height: MAX_RENDER_HEIGHT,
        depth_factor: DEPTH_FACTOR
    };

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let raster_pipelines = RasterPipelines::new(&device, swapchain_format);
    let mut raster_bindings = RasterBindings::new(&device, &raster_pipelines, &render_metadata);

    let mut config = surface
        .get_default_config(&adapter, render_metadata.screen_width, render_metadata.screen_height)
        .unwrap();
    surface.configure(&device, &config);

    let start_time = Instant::now();

    let window = &window;

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &raster_pipelines, &raster_bindings);

            if let Event::AboutToWait = event {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: None,
                });

                let time_elapsed = start_time.elapsed().as_secs_f32();

                let mut instances = Vec::<ModelInstance>::new();

                let block_textures = [
                    [10, 10, 11, 10, 10, 10, 10, 10], // 0
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
/*
                for x in -4..5 {
                    for z in -4..5 {
                        for w in -4..5 {
                            blocks.push(Block{
                                position: [x as i32, 0, z as i32, w as i32],
                                texture: 0
                            });
                        }
                    }
                }*/
                blocks.push(
                    Block{
                        position: [2, 2, 2, 2],
                        texture: 3
                    }
                );
                /*
                blocks.push(
                    Block{
                        position: [-2, 3, 2, 4],
                        texture: 4
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 1, 0, 0],
                        texture: 1
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 2, 0, 0],
                        texture: 1
                    }
                );

                blocks.push(
                    Block{
                        position: [0, 5, 0, 0],
                        texture: 5
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 5, 0, 1],
                        texture: 5
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 5, 0, 2],
                        texture: 5
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 5, 0, -1],
                        texture: 5
                    }
                );
                blocks.push(
                    Block{
                        position: [0, 5, 0, -2],
                        texture: 5
                    }
                );*/

                for block in blocks {
                    let model_scale = 0.5;
                    let model_transform = matrix_multiply(
                        translate_matrix_4d(block.position[0] as f32, block.position[1] as f32, block.position[2] as f32, block.position[3] as f32),
                        scale_matrix_4d(model_scale));
                    instances.push(
                        ModelInstance{
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            cell_texture_ids: block_textures[block.texture],
                        }
                    );
                }
/*
                {
                    let model_scale = 2.0;
                    let model_angle = time_elapsed/2.0;
                    let model_transform = scale_matrix_4d(model_scale);
                    let model_transform = matrix_multiply(rotation_matrix_4d_rotate_0(model_angle), model_transform);
                    let model_transform = matrix_multiply(rotation_matrix_4d_rotate_3(model_angle), model_transform);
                    let model_transform = matrix_multiply(rotation_matrix_4d_rotate_4(model_angle), model_transform);
                    instances.push(
                        InstanceBufferInput{
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            cell_texture_ids: block_textures[3],
                        }
                    );
                }*/


                // Ground
                /*
                {
                    let model_transform = matrix_multiply(
                        translate_matrix_4d(0.0, -10.0, 0.0, 0.0),
                        scale_matrix_4d_elementwise(40.0, 1.0, 40.0, 40.0)
                    );
                    instances.push(
                        InstanceBufferInput{
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            cell_texture_ids: block_textures[0],
                        }
                    );
                }*/

                let view_transform =  translate_matrix_4d(0.0, -5.0, 14.0, 2.0);
                let view_transform = matrix_multiply(view_transform, rotation_matrix_4d_rotate_1(time_elapsed/4.0));
                //let view_transform = matrix_multiply(view_transform, rotation_matrix_4d_rotate_4(time_elapsed/8.0));
                let camera_state = CameraState {
                    view_transform: flatten_5x5_matrix_for_wgpu(view_transform),
                };
                raster_pipelines.do_render(&raster_bindings, &queue, &mut encoder, &view, &instances, &camera_state);

                queue.submit(Some(encoder.finish()));
                frame.present();

                window.request_redraw();
            };

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        render_metadata.screen_width = config.width.min(MAX_SCREEN_WIDTH);
                        render_metadata.screen_height = config.height.min(MAX_SCREEN_HEIGHT);
                        render_metadata.render_width = config.width.min(MAX_RENDER_WIDTH);
                        render_metadata.render_height = config.height.min(MAX_RENDER_HEIGHT);
                        surface.configure(&device, &config);
                        raster_bindings.update_render_buffers(
                            &device,
                            &raster_pipelines,
                            &render_metadata
                        );
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {},
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub fn run() {

    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(arun());
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(arun());
    }
}
