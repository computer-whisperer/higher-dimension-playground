mod present_pass;

use std::borrow::Cow;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use wgpu::Face;
use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};
use crate::present_pass::{PresentBindings, PresentPass};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 4]
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, Pod, Zeroable)]
struct UniformBufferInput {
    model_transform: [f32; 32],
    view_transform: [f32; 32]
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
const VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, -1.0, -1.0]},
    Vertex { position: [1.0, -1.0, -1.0, -1.0]},
    Vertex { position: [1.0, 1.0, -1.0, -1.0]},
    Vertex { position: [-1.0, 1.0, -1.0, -1.0]},
    Vertex { position: [-1.0, -1.0, 1.0, -1.0]},
    Vertex { position: [1.0, -1.0, 1.0, -1.0]},
    Vertex { position: [1.0, 1.0, 1.0, -1.0]},
    Vertex { position: [-1.0, 1.0, 1.0, -1.0]},

    Vertex { position: [-1.0, -1.0, -1.0, 1.0]},
    Vertex { position: [1.0, -1.0, -1.0, 1.0]},
    Vertex { position: [1.0, 1.0, -1.0, 1.0]},
    Vertex { position: [-1.0, 1.0, -1.0, 1.0]},
    Vertex { position: [-1.0, -1.0, 1.0, 1.0]},
    Vertex { position: [1.0, -1.0, 1.0, 1.0]},
    Vertex { position: [1.0, 1.0, 1.0, 1.0]},
    Vertex { position: [-1.0, 1.0, 1.0, 1.0]},
];

const V_NNNN: u16 = 0;
const V_PNNN: u16 = 1;
const V_PPNN: u16 = 2;
const V_NPNN: u16 = 3;
const V_NNPN: u16 = 4;
const V_PNPN: u16 = 5;
const V_PPPN: u16 = 6;
const V_NPPN: u16 = 7;

const V_NNNP: u16 = 8;
const V_PNNP: u16 = 9;
const V_PPNP: u16 = 10;
const V_NPNP: u16 = 11;
const V_NNPP: u16 = 12;
const V_PNPP: u16 = 13;
const V_PPPP: u16 = 14;
const V_NPPP: u16 = 15;


const INDICES: &[u16] = &[

    // Cell 0 PXXX
    V_PPPP, V_PPNP, V_PPPN,// PPXX
    V_PPPN, V_PPNP, V_PPNN,

    V_PNNN, V_PNNP, V_PNPN,// PNXX
    V_PNPN, V_PNNP, V_PNPP,

    V_PPPP, V_PNPP, V_PPPN, // PXPX
    V_PPPN, V_PNPP, V_PNPN,

    V_PPNP, V_PNNP, V_PPNN, // PXNX
    V_PPNN, V_PNNP, V_PNNN,

    // PXXP

    // PXXN

    // Cell 1 NXXX
    V_NPPP, V_NPNP, V_NPPN,// NPXX
    V_NPPN, V_NPNP, V_NPNN,

    V_NNPP, V_NNNP, V_NNPN,// NNXX
    V_NNPN, V_NNNP, V_NNNN,

    V_NPPP, V_NNPP, V_NPPN,// NXPX
    V_NPPN, V_NNPP, V_NNPN,

    V_NPNP, V_NNNP, V_NPNN, // NXNX
    V_NPNN, V_NNNP, V_NNNN,

    // NXXP
    // NXXN

    // Cell 2 XPXX
    // PPXX
    // NPXX
    V_PPPP, V_NPPP, V_PPPN,// XPPX
    V_PPPN, V_NPPP, V_NPPN,

    V_PPNP, V_NPNP, V_PPNN,// XPNX
    V_PPNN, V_NPNP, V_NPNN,

    // XPXP
    // XPXN

    // Cell 3 XNXX
    // PNXX
    // NNXX
    V_PNPP, V_NNPP, V_PNPN, // XNPX
    V_PNPN, V_NNPP, V_NNPN,

    V_PNNP, V_NNNP, V_PNNN,// XNNX
    V_PNNN, V_NNNP, V_NNNN,
    // XNXP
    // XNXN

    // Cell 4 XXPX
    // PXPX
    // NXPX
    // XPPX
    // XNPX
    // XXPP
    // XXPN

    // Cell 5 XXNX
    // PXNX
    // NXNX
    // XPNX
    // XNNX
    // XXNP
    // XXNN

    // Cell 6 XXXP
    V_NNNP, V_PNNP, V_NPNP, //XXNP
    V_NPNP, V_PNNP, V_PPNP,

    V_PNPP, V_NNPP, V_PPPP, //XXPP
    V_PPPP, V_NNPP, V_NPPP,

    V_PNNP, V_PNPP, V_PPNP, //PXXP
    V_PPNP, V_PNPP, V_PPPP,

    V_NNPP, V_NNNP, V_NPPP, //NXXP
    V_NPPP, V_NNNP, V_NPNP,

    V_NPNP, V_PPNP, V_NPPP, //XPXP
    V_NPPP, V_PPNP, V_PPPP,

    V_NNPN, V_PNPN, V_NNNN, //XNXP
    V_NNNN, V_PNPN, V_PNNN,

    // Cell 7 XXXN
    V_NNNN, V_PNNN, V_NPNN, //XXNN
    V_NPNN, V_PNNN, V_PPNN,

    V_PNPN, V_NNPN, V_PPPN, //XXPN
    V_PPPN, V_NNPN, V_NPPN,

    V_PNNN, V_PNPN, V_PPNN, //PXXN
    V_PPNN, V_PNPN, V_PPPN,

    V_NNPN, V_NNNN, V_NPPN, //NXXN
    V_NPPN, V_NNNN, V_NPNN,

    V_NPNN, V_PPNN, V_NPPN, //XPXN
    V_NPPN, V_PPNN, V_PPPN,

    V_NNPN, V_PNPN, V_NNNN, //XNXN
    V_NNNN, V_PNPN, V_PNNN,




];


fn identity_matrix<const N: usize>() -> [[f32; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        result[i][i] = 1.0;
    }
    result
}

fn flatten_5x5_matrix_for_wgpu(matrix: [[f32; 5]; 5]) -> [f32; 32] {
    let mut result = [0.0; 32];
    for i in 0..5 {
        for j in 0..5 {
            result[i * 5 + j] = matrix[j][i];
        }
    }
    result
}


fn scale_matrix_3d(scale: f32) -> [[f32; 4]; 4] {
    [[scale, 0.0, 0.0, 0.0],
     [0.0, scale, 0.0, 0.0],
     [0.0, 0.0, scale, 0.0],
     [0.0, 0.0,   0.0,  1.0]]
}

fn scale_matrix_4d(scale: f32) -> [[f32; 5]; 5] {
    [   [scale, 0.0,   0.0,   0.0,   0.0],
        [0.0,   scale, 0.0,   0.0,   0.0],
        [0.0,   0.0,   scale, 0.0,   0.0],
        [0.0,   0.0,   0.0,   scale, 0.0],
        [0.0,   0.0,   0.0,   0.0,   1.0]]
}

fn translate_matrix_4d(x: f32, y: f32, z: f32, w: f32) -> [[f32; 5]; 5] {
    [[1.0, 0.0, 0.0, 0.0, x],
     [0.0, 1.0, 0.0, 0.0, y],
     [0.0, 0.0, 1.0, 0.0, z],
     [0.0, 0.0, 0.0, 1.0, w],
     [0.0, 0.0, 0.0, 0.0, 1.0]
    ]
}

fn rotation_matrix_3d_yaw(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[cos_theta, 0.0, sin_theta, 0.0],
     [0.0,       1.0, 0.0, 0.0],
     [-sin_theta, 0.0, cos_theta, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

fn rotation_matrix_3d_pitch(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, cos_theta, -sin_theta, 0.0],
     [0.0, sin_theta, cos_theta, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

fn rotation_matrix_3d_roll(angle: f32) -> [[f32; 4]; 4] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [[cos_theta, -sin_theta, 0.0, 0.0],
     [sin_theta, cos_theta, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
}

fn rotation_matrix_4d_rotate_0(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, -sin_theta, 0.0, 0.0, 0.0],
        [sin_theta, cos_theta,  0.0, 0.0, 0.0],
        [0.0,       0.0,        1.0, 0.0, 0.0],
        [0.0,       0.0,        0.0, 1.0, 0.0],
        [0.0,       0.0,        0.0, 0.0, 1.0]
    ]
}

fn rotation_matrix_4d_rotate_1(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, 0.0, -sin_theta, 0.0, 0.0],
        [0.0,       1.0, 0.0,        0.0, 0.0],
        [sin_theta, 0.0, cos_theta,  0.0, 0.0],
        [0.0,       0.0, 0.0,        1.0, 0.0],
        [0.0,       0.0, 0.0,        0.0, 1.0]
    ]
}

fn rotation_matrix_4d_rotate_3(angle: f32) -> [[f32; 5]; 5] {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();
    [
        [cos_theta, 0.0, 0.0,    -sin_theta, 0.0],
        [0.0,       1.0, 0.0,    0.0, 0.0],
        [0.0,       0.0, 1.0,    0.0, 0.0],
        [sin_theta, 0.0, 0.0,    cos_theta, 0.0],
        [0.0,       0.0, 0.0,    0.0, 1.0]
    ]
}


fn matrix_multiply<const N: usize> (a: [[f32; N]; N], b: [[f32; N]; N]) -> [[f32; N]; N] {
    let mut result = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

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
    required_features.insert(wgpu::Features::POLYGON_MODE_LINE);

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features,
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let model_vertex_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Model Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        }
    );

    let model_index_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Model Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        }
    );

    let uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[UniformBufferInput {
                model_transform: flatten_5x5_matrix_for_wgpu(identity_matrix()),
                view_transform: flatten_5x5_matrix_for_wgpu(identity_matrix())
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let uniform_buffer_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        ],
        label: Some("uniform_buffer_bind_group_layout"),
    });

    let uniform_buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &uniform_buffer_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }
        ],
        label: Some("uniform_buffer_bind_group"),
    });

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&uniform_buffer_bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let present_pass = PresentPass::new(&device, swapchain_format);

    let present_bindings = PresentBindings::new(&device, &present_pass);

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
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

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
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_vertex_buffer(0, model_vertex_buffer.slice(..));
                            rpass.set_index_buffer(model_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                            rpass.set_bind_group(0, &uniform_buffer_bind_group, &[]);
                            rpass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
                        }

                        let time_elapsed = start_time.elapsed().as_secs_f32();
                        let model_angle = time_elapsed/2.0;

                        let model_scale = 2.0;
                        //let model_transform = matrix_multiply(rotation_matrix_4d_rotate_3(model_angle), scale_matrix_4d(model_scale));
                        let model_transform = matrix_multiply(rotation_matrix_4d_rotate_1(0.3), scale_matrix_4d(model_scale));
                        //let model_transform = scale_matrix_4d(model_scale);

                        let view_transform =  translate_matrix_4d(0.0, -3.0, 7.0, 7.0);

                        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[UniformBufferInput {
                            model_transform: flatten_5x5_matrix_for_wgpu(model_transform),
                            view_transform: flatten_5x5_matrix_for_wgpu(view_transform)
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

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}