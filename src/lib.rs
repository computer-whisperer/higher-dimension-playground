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
    view_transform: [f32; 32],
    model_transform: [f32; 32]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
pub struct ScreenUniformBufferInput {
    screen_width: f32,
    screen_height: f32,
    render_width: f32,
    render_height: f32,
    depth_factor: u32
}


const MODEL_COLOR: [f32; 3] = [0.0, 0.0, 1.0];


pub fn create_color_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    use std::mem::size_of;
    #[repr(C)]
    struct Pixel {
        r: f32,
        g: f32,
        b: f32,
    }
    assert!(size_of::<Pixel>() == size_of::<[f32; 3]>());

    let pixel_size = size_of::<Pixel>() as u64;
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

const MAX_RENDER_WIDTH : u32 = 1920/4;
const MAX_RENDER_HEIGHT : u32 = 1080/4;
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
    required_limits.max_bind_groups = 8;
    required_limits.max_buffer_size = 1024*1024*1024;
    required_limits.max_storage_buffer_binding_size = 1024*1024*1024;

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

    let vertex_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Vertex Output Buffer"),
        size: (tesseract::TET_VERTICES.len()*32) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let screen_uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Screen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ScreenUniformBufferInput {
                screen_width: 1024.0,
                screen_height: 1024.0,
                render_width: 1024.0,
                render_height: 1024.0,
                depth_factor: DEPTH_FACTOR
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let camera_uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniformBufferInput {
                view_transform: [0.0; 32],
                model_transform: [0.0; 32],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let mut render_width: u32 = 1920;
    let mut render_height: u32 = 1080;

    let mut depth_buffer = create_depth_buffer(&device, render_width, render_height, DEPTH_FACTOR);
    let mut color_buffer = create_color_buffer(&device, render_width, render_height);

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let present_pass = PresentPass::new(&device, swapchain_format);
    let mut present_bindings = PresentBindings::new(&device, &present_pass, &color_buffer, &screen_uniform_buffer);

    let raster_pipelines = RasterPipelines::new(&device);
    let mut raster_bindings = RasterBindings::new(
        &device, &raster_pipelines, &color_buffer,
        &model_vertex_buffer, &vertex_output_buffer, &screen_uniform_buffer,
        &camera_uniform_buffer, &depth_buffer);

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
                {
                    {
                        let mut cpass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                        raster_pipelines.record(&mut cpass, &raster_bindings, render_width, render_height, tesseract::TET_VERTICES.len());
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
                }

                let time_elapsed = start_time.elapsed().as_secs_f32();
                let model_angle = time_elapsed/2.0;

                let model_scale = 2.0;
                let model_transform = scale_matrix_4d(model_scale);
                let model_transform = matrix_multiply(rotation_matrix_4d_rotate_0(model_angle), model_transform);
                let model_transform = matrix_multiply(rotation_matrix_4d_rotate_3(model_angle), model_transform);
                let model_transform = matrix_multiply(rotation_matrix_4d_rotate_4(model_angle), model_transform);

                let view_transform =  translate_matrix_4d(5.0, 6.0, 15.0, 15.0);

                queue.write_buffer(&camera_uniform_buffer, 0, bytemuck::cast_slice(&[CameraUniformBufferInput {
                    view_transform: flatten_5x5_matrix_for_wgpu(view_transform),
                    model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                }]));
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
                        render_width = config.width.min(MAX_RENDER_WIDTH);
                        render_height = config.height.min(MAX_RENDER_HEIGHT);

                        surface.configure(&device, &config);
                        queue.write_buffer(&screen_uniform_buffer, 0, bytemuck::cast_slice(&[ScreenUniformBufferInput {
                            screen_width: config.width as f32,
                            screen_height: config.height as f32,
                            render_width: render_width as f32,
                            render_height: render_height as f32,
                            depth_factor: DEPTH_FACTOR
                        }]));
                        color_buffer = create_color_buffer(
                            &device,
                            render_width,
                            render_height
                        );
                        depth_buffer = create_depth_buffer(
                            &device,
                            render_width,
                            render_height,
                            DEPTH_FACTOR
                        );
                        present_bindings.update_color_buffer(
                            &device,
                            &present_pass,
                            &color_buffer
                        );
                        raster_bindings.update_color_buffer(
                            &device,
                            &raster_pipelines,
                            &color_buffer
                        );
                        raster_bindings.update_depth_buffer(
                            &device,
                            &raster_pipelines,
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
