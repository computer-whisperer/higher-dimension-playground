mod present_pass;
mod raster_pass;
mod matrix_operations;
mod tesseract;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;


use std::borrow::Cow;
use web_time::{Instant, SystemTime};
use bytemuck::{Pod, Zeroable};
use wgpu::Face;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};
use winit::window::WindowBuilder;
use crate::present_pass::{PresentBindings, PresentPass};
use crate::matrix_operations::*;
use crate::raster_pass::{ClearPass, RasterBindings, RasterPass};

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
}

/*
// Cube
const MODEL_COLOR: [f32; 3] = [0.0, 0.0, 1.0];
const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, -1.0], color: MODEL_COLOR },
    Vertex { position: [1.0, -1.0, -1.0], color: MODEL_COLOR },
    Vertex { position: [1.0, 1.0, -1.0], color: MODEL_COLOR },
    Vertex { position: [-1.0, 1.0, -1.0], color: MODEL_COLOR },
    Vertex { position: [-1.0, -1.0, 1.0], color: MODEL_COLOR },
    Vertex { position: [1.0, -1.0, 1.0], color: MODEL_COLOR },
    Vertex { position: [1.0, 1.0, 1.0], color: MODEL_COLOR },
    Vertex { position: [-1.0, 1.0, 1.0], color: MODEL_COLOR },
];

const INDICES: &[u16] = &[
    0, 1, 3, 3, 1, 2,
    1, 5, 2, 2, 5, 6,
    5, 4, 6, 6, 4, 7,
    4, 0, 7, 7, 0, 3,
    3, 2, 7, 7, 2, 6,
    4, 5, 0, 0, 5, 1
];
*/


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


pub(crate) const WORKGROUP_SIZE: u32 = 256;
pub(crate) const fn dispatch_size(len: u32) -> u32 {
    let subgroup_size = WORKGROUP_SIZE;
    let padded_size = (subgroup_size - len % subgroup_size) % subgroup_size;
    (len + padded_size) / subgroup_size
}


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
            power_preference: wgpu::PowerPreference::default(),
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
    /*
    required_limits.max_storage_buffers_per_shader_stage = 1;
    required_limits.max_storage_buffer_binding_size = 0x40000000;
    required_limits.max_compute_workgroup_size_x = 256;
    required_limits.max_compute_workgroup_size_y = 1;
    required_limits.max_compute_workgroup_size_z = 1;
    required_limits.max_compute_invocations_per_workgroup = 256;
    required_limits.max_compute_workgroups_per_dimension = 2048;*/

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
            contents: bytemuck::cast_slice(&tesseract::VERTICES),
            usage: wgpu::BufferUsages::STORAGE,
        }
    );

    let screen_uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Screen Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ScreenUniformBufferInput {
                screen_width: 1024.0,
                screen_height: 1024.0,
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

    let mut color_buffer = create_color_buffer(&device, 1024, 1024);

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let present_pass = PresentPass::new(&device, swapchain_format);
    let mut present_bindings = PresentBindings::new(&device, &present_pass, &color_buffer, &screen_uniform_buffer);

    let raster_pass = RasterPass::new(&device);
    let mut raster_bindings = RasterBindings::new(&device, &raster_pass, &color_buffer, &model_vertex_buffer, &screen_uniform_buffer, &camera_uniform_buffer);

    let clear_pass = ClearPass::new(&device);

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
                        surface.configure(&device, &config);
                        queue.write_buffer(&screen_uniform_buffer, 0, bytemuck::cast_slice(&[ScreenUniformBufferInput {
                            screen_width: config.width as f32,
                            screen_height: config.height as f32,
                        }]));
                        color_buffer = create_color_buffer(
                            &device,
                            config.width,
                            config.height
                        );
                        present_bindings.update_color_buffer(
                            &device,
                            &present_pass,
                            &color_buffer
                        );
                        raster_bindings.update_color_buffer(
                            &device,
                            &raster_pass,
                            &color_buffer
                        );
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
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
                                clear_pass.record(&mut cpass, &raster_bindings, dispatch_size(config.width * config.height));
                                raster_pass.record(&mut cpass, &raster_bindings, dispatch_size((tesseract::VERTICES.len() as u32)/3));
                            }
                            {
                                let mut rpass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                            view: &view,
                                            resolve_target: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                store: wgpu::StoreOp::Store,
                                            },
                                        })],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                present_pass.record(&mut rpass, &present_bindings);
                            }


                            /*
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_vertex_buffer(0, model_vertex_buffer.slice(..));
                            rpass.set_index_buffer(model_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                            rpass.set_bind_group(0, &uniform_buffer_bind_group, &[]);
                            rpass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);*/
                        }

                        let time_elapsed = start_time.elapsed().as_secs_f32();
                        let model_angle = time_elapsed/2.0;

                        let model_scale = 2.0;
                        let model_transform = matrix_multiply(rotation_matrix_4d_rotate_3(model_angle), scale_matrix_4d(model_scale));
                        //let model_transform = matrix_multiply(rotation_matrix_4d_rotate_1(0.3), scale_matrix_4d(model_scale));
                        //let model_transform = scale_matrix_4d(model_scale);

                        let view_transform =  translate_matrix_4d(0.0, -3.0, 7.0, 7.0);

                        queue.write_buffer(&camera_uniform_buffer, 0, bytemuck::cast_slice(&[CameraUniformBufferInput {
                            view_transform: flatten_5x5_matrix_for_wgpu(view_transform),
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                        }]));
                        queue.submit(Some(encoder.finish()));
                        frame.present();

                        window.request_redraw();
                    }
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
