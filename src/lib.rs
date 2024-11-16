mod present_pass;
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
    window::Window,
};
use winit::window::WindowBuilder;
use crate::present_pass::{PresentBindings, PresentPass};
use crate::matrix_operations::*;
use crate::raster_pass::{RasterBindings, RasterPipelines};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
struct CameraUniformBufferInput {
    view_transform: [f32; 32]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderMetaUniformBufferInput {
    screen_width: u32,
    screen_height: u32,
    render_width: u32,
    render_height: u32,
    depth_factor: u32
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct InstanceBufferInput {
    model_transform: [f32; 32],
    cell_texture_ids: [u32; 8]
}



pub fn create_color_buffer(device: &wgpu::Device, width: u32, height: u32, depth_factor: u32) -> wgpu::Buffer {



    let pixel_size = 4*depth_factor as u64;
    let (width, height) = (width as u64, height as u64);
    let size = pixel_size * width * height;

    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

pub fn create_depth_buffer(device: &wgpu::Device, width: u32, height: u32, depth_factor: u32) -> wgpu::Buffer {

    let pixel_size :u64 = 4*(depth_factor as u64);
    //let pixel_size :u64 = 1;
    let size = pixel_size * width as u64 * height as u64;

    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Depth Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

const MAX_INSTANCES : usize = 1000;
const MAX_INTERMEDIATE_VERTICES : u64 = (MAX_INSTANCES as u64) * (tesseract::TET_VERTICES.len() as u64);
const MAX_RENDER_WIDTH : u32 = 1920/3;
const MAX_RENDER_HEIGHT : u32 = 1080/3;
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
    required_limits.max_buffer_size = 2147483647;
    required_limits.max_storage_buffer_binding_size = 2147483647;

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

    let model_vertex_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Model Vertex Buffer"),
            contents: bytemuck::cast_slice(&tesseract::TET_VERTICES),
            usage: wgpu::BufferUsages::STORAGE,
        }
    );

    let instance_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Model Instance Buffer"),
            contents: bytemuck::cast_slice(&[InstanceBufferInput{
                model_transform: [0.0; 32],
                cell_texture_ids: [0; 8],
            }; MAX_INSTANCES]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        }
    );

    let mut render_metadata = RenderMetaUniformBufferInput {
        screen_width: size.width,
        screen_height: size.height,
        render_width: MAX_RENDER_WIDTH,
        render_height: MAX_RENDER_HEIGHT,
        depth_factor: DEPTH_FACTOR
    };

    let vertex_intermediate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Output Buffer"),
        size: (MAX_INTERMEDIATE_VERTICES*32) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });



    let render_meta_uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Screen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[render_metadata]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let camera_uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniformBufferInput {
                view_transform: [0.0; 32],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });



    let mut depth_buffer = create_depth_buffer(&device, render_metadata.render_width, render_metadata.render_height, render_metadata.depth_factor);
    let mut color_buffer = create_color_buffer(&device, render_metadata.render_width, render_metadata.render_height, render_metadata.depth_factor);

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let present_pass = PresentPass::new(&device, swapchain_format);
    let mut present_bindings = PresentBindings::new(&device, &present_pass, &color_buffer, &render_meta_uniform_buffer);

    let raster_pipelines = RasterPipelines::new(&device);
    let mut raster_bindings = RasterBindings::new(
        &device, &raster_pipelines, &color_buffer,
        &depth_buffer, &model_vertex_buffer, &vertex_intermediate_buffer,
        &instance_buffer,
        &render_meta_uniform_buffer,
        &camera_uniform_buffer);

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let start_time = Instant::now();

    let window = &window;

    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &present_pass, &present_bindings);

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

                let mut instances = Vec::<InstanceBufferInput>::new();

                let block_textures = [
                    [9, 9, 10, 9, 9, 9, 9, 9],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                ];

                struct Block {
                    position: [i32; 4],
                    texture: usize
                }

                let mut blocks = Vec::<Block>::new();

                for x in -4..5 {
                    for z in -4..5 {
                        for w in -4..5 {
                            blocks.push(Block{
                                position: [x as i32, 0, z as i32, w as i32],
                                texture: 0
                            });
                        }
                    }
                }
                blocks.push(
                    Block{
                        position: [2, 2, 2, 2],
                        texture: 2
                    }
                );
                blocks.push(
                    Block{
                        position: [-2, 3, 2, -2],
                        texture: 2
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

                for block in blocks {
                    let model_scale = 0.5;
                    let model_transform = matrix_multiply(
                        translate_matrix_4d(block.position[0] as f32, block.position[1] as f32, block.position[2] as f32, block.position[3] as f32),
                        scale_matrix_4d(model_scale));
                    instances.push(
                        InstanceBufferInput{
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            cell_texture_ids: block_textures[block.texture],
                        }
                    );
                }
                // Ground
                /*
                {
                    let model_transform = matrix_multiply(
                        translate_matrix_4d(0.0, 2.0, 0.0, 0.0),
                        scale_matrix_4d_elementwise(40.0, 2.0, 40.0, 40.0)
                    );
                    instances.push(
                        InstanceBufferInput{
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            cell_texture_ids: block_textures[0],
                        }
                    );
                }*/

                let view_transform =  translate_matrix_4d(0.0, -4.0, 6.0, 6.0);
                let view_transform = matrix_multiply(view_transform, rotation_matrix_4d_rotate_1(time_elapsed/2.0));

                queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
                queue.write_buffer(&render_meta_uniform_buffer, 0, bytemuck::cast_slice(&[render_metadata]));
                queue.write_buffer(&camera_uniform_buffer, 0, bytemuck::cast_slice(&[CameraUniformBufferInput {
                    view_transform: flatten_5x5_matrix_for_wgpu(view_transform),
                }]));

                {
                    let mut cpass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    raster_pipelines.record(&mut cpass, &raster_bindings, render_metadata.render_width, render_metadata.render_height, tesseract::TET_VERTICES.len()*instances.len());
                }

                {
                    let mut rpass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    present_pass.record(&mut rpass, &present_bindings);
                }
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
                        render_metadata.screen_width = config.width;
                        render_metadata.screen_height = config.height;
                        render_metadata.render_width = config.width.min(MAX_RENDER_WIDTH);
                        render_metadata.render_height = config.height.min(MAX_RENDER_HEIGHT);
                        surface.configure(&device, &config);
                        color_buffer = create_color_buffer(
                            &device,
                            render_metadata.render_width,
                            render_metadata.render_height,
                            render_metadata.depth_factor
                        );
                        depth_buffer = create_depth_buffer(
                            &device,
                            render_metadata.render_width,
                            render_metadata.render_height,
                            render_metadata.depth_factor
                        );
                        present_bindings.update_color_buffer(
                            &device,
                            &present_pass,
                            &color_buffer
                        );
                        raster_bindings.update_render_buffers(
                            &device,
                            &raster_pipelines,
                            &color_buffer,
                            &depth_buffer
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
