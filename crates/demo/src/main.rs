use clap::{Parser, ValueEnum};
use higher_dimension_playground::matrix_operations::{
    double_rotation_matrix_4d, rotation_matrix_one_angle, scale_matrix_4d,
    scale_matrix_4d_elementwise, translate_matrix_4d,
};
use higher_dimension_playground::render::{
    FrameParams, RenderBackend, RenderContext, RenderOptions, TetraFrameInput, VoxelFrameInput,
    VTE_REGION_BVH_INVALID_NODE,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::f32::consts::PI;
use std::{error::Error, sync::Arc};
use vulkano::device::{Device, Queue};
use vulkano::instance::debug::DebugUtilsMessenger;
use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long)]
    headless: bool,

    /// Disable rasterization
    #[arg(long)]
    no_raster: bool,

    /// Enable raytracing
    #[arg(long)]
    raytrace: bool,

    /// Enable ZW edge rendering
    #[arg(long)]
    edges: bool,

    /// Enable tetrahedron edge rendering
    #[arg(long)]
    tetrahedron_edges: bool,

    /// Enable camera spin animation
    #[arg(long)]
    spin: bool,

    /// Disable animation (freeze at frame 0)
    #[arg(long)]
    no_animation: bool,

    /// Disable EXR frame export
    #[arg(long)]
    no_export: bool,

    /// Hide the outer 4x4 block grid
    #[arg(long)]
    no_outer_blocks: bool,

    /// Show floor plane
    #[arg(long)]
    floor: bool,

    /// Show wall enclosure
    #[arg(long)]
    walls: bool,

    /// Canvas width in pixels
    #[arg(long, short = 'W', default_value_t = 960)]
    width: u32,

    /// Canvas height in pixels
    #[arg(long, short = 'H', default_value_t = 540)]
    height: u32,

    /// Number of depth layers (supersampling)
    #[arg(long, default_value_t = 4)]
    layers: u32,

    /// Rendering backend to use
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Auto,
    TetraRaster,
    TetraRaytrace,
    VoxelTraversal,
}

impl BackendArg {
    fn to_render_backend(self) -> RenderBackend {
        match self {
            BackendArg::Auto => RenderBackend::Auto,
            BackendArg::TetraRaster => RenderBackend::TetraRaster,
            BackendArg::TetraRaytrace => RenderBackend::TetraRaytrace,
            BackendArg::VoxelTraversal => RenderBackend::VoxelTraversal,
        }
    }
}

fn main() -> Result<(), impl Error> {
    let args = Args::parse();

    if args.headless {
        run_headless(&args);
        Ok(())
    } else {
        let event_loop = EventLoop::new().unwrap();
        let mut app = App::new(&event_loop, args);

        event_loop.run_app(&mut app)
    }
}

fn run_headless(args: &Args) {
    let (instance, device, queue) = vulkan_setup(None);
    let pixel_storage_layers = if args.backend.to_render_backend() == RenderBackend::VoxelTraversal
    {
        Some(1)
    } else {
        None
    };

    let mut rcx = RenderContext::new_with_pixel_storage_layers(
        device.clone(),
        queue.clone(),
        instance.clone(),
        None,
        [args.width, args.height, 1],
        pixel_storage_layers,
    );

    let mut demo_scene = DemoScene::new(args.clone());

    // Render a single frame with PNG export
    demo_scene.update_headless_png(&mut rcx, device.clone(), queue.clone());
}

struct DemoScene {
    args: Args,
    frame_num: u32,
    sub_frame_num: u32,
}

impl DemoScene {
    fn new(args: Args) -> DemoScene {
        DemoScene {
            args,
            frame_num: 0,
            sub_frame_num: 0,
        }
    }

    fn update_headless_png(
        &mut self,
        rcx: &mut RenderContext,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) {
        let mut view_matrix = translate_matrix_4d(0.0, 2.0, 6.0, 4.0);
        let focal_length_xy = 1.0;
        let focal_length_zw = 1.0;

        // Pitch down to see the floor
        view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 2, 1, -PI * 0.15));
        view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 2, PI * 0.125));
        view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 3, PI * 0.125));

        let mut instances = Vec::<common::ModelInstance>::new();

        // Single centered tesseract with distinct materials per cell
        let model_transform =
            translate_matrix_4d(-0.5, -0.5, -0.5, -0.5).dot(&scale_matrix_4d(1.0));
        instances.push(common::ModelInstance {
            model_transform: model_transform.into(),
            cell_material_ids: [1, 2, 3, 4, 5, 6, 7, 8],
        });

        if self.args.floor {
            let width = 500.0;
            let model_transform =
                translate_matrix_4d(-width / 2.0, -4.0, -width / 2.0, -width / 2.0)
                    .dot(&scale_matrix_4d_elementwise(width, 1.0, width, width));
            instances.push(common::ModelInstance {
                model_transform: model_transform.into(),
                cell_material_ids: [11; 8],
            });
        }

        let render_options = RenderOptions {
            do_raster: !self.args.no_raster,
            do_raytrace: self.args.raytrace,
            render_backend: self.args.backend.to_render_backend(),
            do_edges: self.args.edges,
            do_tetrahedron_edges: self.args.tetrahedron_edges,
            prepare_render_screenshot: true,
            ..Default::default()
        };

        let backend = render_options.render_backend;
        let frame_params = FrameParams {
            view_matrix,
            time_ticks_ms: 0,
            focal_length_xy,
            focal_length_zw,
            render_options,
        };
        if backend == RenderBackend::VoxelTraversal {
            rcx.render_voxel_frame(
                device,
                queue,
                frame_params,
                VoxelFrameInput {
                    metadata_generation: 0,
                    mutation_base_generation: None,
                    region_bvh_root_index: VTE_REGION_BVH_INVALID_NODE,
                    chunk_headers: &[],
                    occupancy_words: &[],
                    material_words: &[],
                    orientation_words: &[],
                    macro_words: &[],
                    region_bvh_nodes: &[],
                    leaf_headers: &[],
                    leaf_chunk_entries: &[],
                    mutation_batch: None,
                    dirty_ranges: None,
                },
                &instances,
                &[],
            );
        } else {
            rcx.render_tetra_frame(
                device,
                queue,
                frame_params,
                TetraFrameInput {
                    model_instances: &instances,
                },
            );
        }

        std::fs::create_dir_all("frames").ok();
        rcx.save_rendered_frame_png("frames/headless_raster.png");
        println!("Headless render complete.");
    }

    fn update(&mut self, rcx: &mut RenderContext, device: Arc<Device>, queue: Arc<Queue>) {
        let mut view_matrix = translate_matrix_4d(0.0, 0.0, 8.0, 4.0);
        let focal_length_xy = 1.0;
        let focal_length_zw = 1.0;

        let do_raytrace = self.args.raytrace;
        let do_edges = self.args.edges;
        let do_raster = !self.args.no_raster;

        let do_animation = !self.args.no_animation;

        let do_frame_export = !self.args.no_export;

        let frame_time_hz = 20.0;
        let do_spin = self.args.spin;

        let do_outer_blocks = !self.args.no_outer_blocks;
        let do_floor = self.args.floor;
        let do_walls = self.args.walls;

        let sub_frames_per_frame = if do_raytrace {
            100 // Reduced for faster testing
        } else {
            1
        };
        let sub_frames_per_export = 1000; // Reduced for faster testing

        let time_elapsed = self.frame_num as f32 / frame_time_hz;

        if do_spin {
            view_matrix = view_matrix.dot(&rotation_matrix_one_angle(
                5,
                0,
                2,
                PI * 0.25 + PI * 2.0 * time_elapsed / 30.0,
            ));
            view_matrix = view_matrix.dot(&rotation_matrix_one_angle(
                5,
                0,
                1,
                PI * 0.25 + PI * 2.0 * time_elapsed / 30.0,
            ));
            view_matrix = view_matrix.dot(&rotation_matrix_one_angle(
                5,
                2,
                3,
                PI * 2.0 * time_elapsed / 30.0,
            ));
        }
        //view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 2, 3, PI*0.25));
        view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 2, PI * 0.125));
        view_matrix = view_matrix.dot(&rotation_matrix_one_angle(5, 0, 3, PI * 0.125));

        let mut instances = Vec::<common::ModelInstance>::new();

        struct Block {
            position: [i32; 4],
            materials: [u32; 8],
            rotation: Option<([usize; 2], [usize; 2], f32, f32)>,
            // (plane1, plane2, angular_velocity_1, angular_velocity_2) in rad/s
        }

        let mut blocks = Vec::<Block>::new();

        if do_outer_blocks {
            let mut texture_rot = 0u32;
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        for w in 0..2 {
                            blocks.push(Block {
                                position: [x * 4 - 2, y * 4 - 2, z * 4 - 2, w * 4 - 2],
                                materials: [texture_rot + 1; 8],
                                rotation: None,
                            });
                            texture_rot = (texture_rot + 1) % 5;
                        }
                    }
                }
            }
        }

        /* blocks.push(
            Block{
                position: [0, -2, -1, 0],
                materials: [1, 2, 3, 4, 5, 6, 7, 8]
            }
        );*/
        blocks.push(Block {
            position: [0, 0, 0, 0],
            //materials: [1, 2, 3, 4, 5, 6, 7, 8]
            materials: [13; 8],
            rotation: Some(([0, 1], [2, 3], 0.5, 0.3)), // XY + ZW double rotation
        });

        if do_walls {
            let width = 30.0;

            let model_transform =
                translate_matrix_4d(-width / 2.0, -5.0, -width / 2.0, -width / 2.0)
                    .dot(&scale_matrix_4d_elementwise(width, width, width, width));
            instances.push(common::ModelInstance {
                model_transform: model_transform.into(),
                //cell_material_ids: [1, 2, 3, 4, 5, 6, 13, 8],
                cell_material_ids: [14; 8],
            });
        }

        if do_floor {
            let width = 1000.0;
            let model_transform =
                translate_matrix_4d(-width / 2.0, -4.0, -width / 2.0, -width / 2.0)
                    .dot(&scale_matrix_4d_elementwise(width, 1.0, width, width));
            instances.push(common::ModelInstance {
                model_transform: model_transform.into(),
                cell_material_ids: [11; 8],
            });
        }

        for block in blocks {
            let model_scale = 1.0;
            let cx = block.position[0] as f32 * model_scale;
            let cy = block.position[1] as f32 * model_scale;
            let cz = block.position[2] as f32 * model_scale;
            let cw = block.position[3] as f32 * model_scale;

            let model_transform = if let Some((plane1, plane2, omega1, omega2)) = block.rotation {
                let rot = double_rotation_matrix_4d(
                    plane1,
                    omega1 * time_elapsed,
                    plane2,
                    omega2 * time_elapsed,
                );
                // translate to center → rotate → translate back, then offset by -0.5 and scale
                translate_matrix_4d(cx, cy, cz, cw)
                    .dot(&rot)
                    .dot(&translate_matrix_4d(
                        -0.5 * model_scale,
                        -0.5 * model_scale,
                        -0.5 * model_scale,
                        -0.5 * model_scale,
                    ))
                    .dot(&scale_matrix_4d(model_scale))
            } else {
                translate_matrix_4d(
                    (block.position[0] as f32 - 0.5) * model_scale,
                    (block.position[1] as f32 - 0.5) * model_scale,
                    (block.position[2] as f32 - 0.5) * model_scale,
                    (block.position[3] as f32 - 0.5) * model_scale,
                )
                .dot(&scale_matrix_4d(model_scale))
            };

            instances.push(common::ModelInstance {
                model_transform: model_transform.into(),
                cell_material_ids: block.materials,
            });
        }

        let render_options = RenderOptions {
            do_frame_clear: do_raytrace && self.sub_frame_num == 0,
            do_raster,
            render_backend: self.args.backend.to_render_backend(),
            do_tetrahedron_edges: self.args.tetrahedron_edges,
            do_raytrace,
            do_edges,
            prepare_render_screenshot: do_frame_export
                && ((self.sub_frame_num + 1) % sub_frames_per_export == 0),
            ..Default::default()
        };

        let backend = render_options.render_backend;
        let frame_params = FrameParams {
            view_matrix,
            time_ticks_ms: (self.sub_frame_num as u64).wrapping_mul(16) as u32,
            focal_length_xy,
            focal_length_zw,
            render_options,
        };
        if backend == RenderBackend::VoxelTraversal {
            rcx.render_voxel_frame(
                device,
                queue,
                frame_params,
                VoxelFrameInput {
                    metadata_generation: 0,
                    mutation_base_generation: None,
                    region_bvh_root_index: VTE_REGION_BVH_INVALID_NODE,
                    chunk_headers: &[],
                    occupancy_words: &[],
                    material_words: &[],
                    orientation_words: &[],
                    macro_words: &[],
                    region_bvh_nodes: &[],
                    leaf_headers: &[],
                    leaf_chunk_entries: &[],
                    mutation_batch: None,
                    dirty_ranges: None,
                },
                &instances,
                &[],
            );
        } else {
            rcx.render_tetra_frame(
                device,
                queue,
                frame_params,
                TetraFrameInput {
                    model_instances: &instances,
                },
            );
        }

        self.sub_frame_num += 1;

        if do_frame_export && (self.sub_frame_num % sub_frames_per_export == 0) {
            rcx.save_rendered_frame(&format!(
                "frames/render_{}_{}.exr",
                self.frame_num, self.sub_frame_num
            ))
        }

        if do_animation && self.sub_frame_num >= sub_frames_per_frame {
            self.sub_frame_num = 0;
            self.frame_num += 1;
        }
    }
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    demo_scene: DemoScene,
    _callback: Option<DebugUtilsMessenger>,
}

impl App {
    fn new(event_loop: &EventLoop<()>, args: Args) -> Self {
        let (instance, device, queue) = vulkan_setup(Some(event_loop));

        use vulkano::instance::debug::{
            DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        };

        let _callback = unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
                    |_message_severity, _message_type, callback_data| {
                        println!("Debug callback: {:?}", callback_data.message);
                    },
                )),
            )
            .ok()
        };

        let demo_scene = DemoScene::new(args);

        let rcx = None;

        App {
            _callback,
            instance,
            device,
            queue,
            rcx,
            demo_scene,
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
        let pixel_storage_layers =
            if self.demo_scene.args.backend.to_render_backend() == RenderBackend::VoxelTraversal {
                Some(1)
            } else {
                None
            };
        self.rcx = Some(RenderContext::new_with_pixel_storage_layers(
            self.device.clone(),
            self.queue.clone(),
            self.instance.clone(),
            Some(window),
            [
                self.demo_scene.args.width,
                self.demo_scene.args.height,
                self.demo_scene.args.layers,
            ],
            pixel_storage_layers,
        ));
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
                self.demo_scene
                    .update(rcx, self.device.clone(), self.queue.clone());
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.clone().unwrap().request_redraw();
    }
}
