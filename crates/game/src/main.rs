mod camera;
mod cpu_render;
mod input;
mod scene;
mod voxel;

use clap::{Parser, ValueEnum};
use higher_dimension_playground::render::{
    FrameParams, RenderBackend, RenderContext, RenderOptions, TetraFrameInput, VteDisplayMode,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::Camera4D;
use input::{ControlScheme, InputState, RotationPair};
use scene::{Scene, ScenePreset};

const MOUSE_SENSITIVITY: f32 = 0.002;

#[derive(Parser, Debug, Clone)]
#[command(version, about = "4D game explorer")]
struct Args {
    /// Render buffer width in pixels
    #[arg(long, short = 'W', default_value_t = 480)]
    width: u32,

    /// Render buffer height in pixels
    #[arg(long, short = 'H', default_value_t = 270)]
    height: u32,

    /// Number of depth layers (supersampling)
    #[arg(long, default_value_t = 4)]
    layers: u32,

    /// Voxel scene preset (used by VTE backend)
    #[arg(long, value_enum, default_value_t = SceneArg::Flat)]
    scene: SceneArg,

    /// CPU-only render: produce frames/cpu_render.png and exit (no GPU window)
    #[arg(long)]
    cpu_render: bool,

    /// GPU screenshot: render one frame at debug camera position and exit
    #[arg(long)]
    gpu_screenshot: bool,

    /// Source for --gpu-screenshot capture output
    #[arg(long, value_enum, default_value_t = GpuScreenshotSourceArg::RenderBuffer)]
    gpu_screenshot_source: GpuScreenshotSourceArg,

    /// Override screenshot camera position as 4 values: X Y Z W
    #[arg(
        long,
        num_args = 4,
        value_names = ["X", "Y", "Z", "W"],
        allow_hyphen_values = true
    )]
    screenshot_pos: Option<Vec<f32>>,

    /// Override screenshot camera angles as 4 values (radians): YAW PITCH XW ZW
    #[arg(
        long,
        num_args = 4,
        value_names = ["YAW", "PITCH", "XW", "ZW"],
        allow_hyphen_values = true
    )]
    screenshot_angles: Option<Vec<f32>>,

    /// Override screenshot camera angles as 4 values (degrees): YAW PITCH XW ZW
    #[arg(
        long,
        num_args = 4,
        value_names = ["YAW_DEG", "PITCH_DEG", "XW_DEG", "ZW_DEG"],
        allow_hyphen_values = true
    )]
    screenshot_angles_deg: Option<Vec<f32>>,

    /// Optional screenshot YW deviation angle (radians)
    #[arg(long)]
    screenshot_yw: Option<f32>,

    /// Rendering backend to use
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    /// VTE traversal step budget per ray (quality/perf tradeoff)
    #[arg(long, default_value_t = 320)]
    vte_max_trace_steps: u32,

    /// VTE max ray distance in world units before miss (quality/perf tradeoff)
    #[arg(long, default_value_t = 160.0)]
    vte_max_trace_distance: f32,

    /// VTE Stage-B display operator (integral, slice, thick-slice, debug-compare, debug-integral)
    #[arg(long, value_enum, default_value_t = VteDisplayModeArg::Integral)]
    vte_display_mode: VteDisplayModeArg,

    /// VTE slice center layer index (0..layers-1). Defaults to center layer.
    #[arg(long)]
    vte_slice_layer: Option<u32>,

    /// VTE thick-slice half-width in layer indices.
    #[arg(long, default_value_t = 2)]
    vte_thick_half_width: u32,

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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum VteDisplayModeArg {
    Integral,
    Slice,
    ThickSlice,
    DebugCompare,
    DebugIntegral,
}

impl VteDisplayModeArg {
    fn to_render_mode(self) -> VteDisplayMode {
        match self {
            Self::Integral => VteDisplayMode::Integral,
            Self::Slice => VteDisplayMode::Slice,
            Self::ThickSlice => VteDisplayMode::ThickSlice,
            Self::DebugCompare => VteDisplayMode::DebugCompare,
            Self::DebugIntegral => VteDisplayMode::DebugIntegral,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum GpuScreenshotSourceArg {
    RenderBuffer,
    Framebuffer,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SceneArg {
    Flat,
    DemoCubes,
}

impl SceneArg {
    fn to_scene_preset(self) -> ScenePreset {
        match self {
            SceneArg::Flat => ScenePreset::Flat,
            SceneArg::DemoCubes => ScenePreset::DemoCubes,
        }
    }
}

fn main() {
    let args = Args::parse();

    if args.cpu_render {
        run_cpu_render(args.scene.to_scene_preset(), &args);
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    let (instance, device, queue) = vulkan_setup(Some(&event_loop));

    let gpu_screenshot = args.gpu_screenshot;

    let mut camera = Camera4D::new();
    if gpu_screenshot {
        camera.position = [5.44, 0.47, -1.23, -4.00];
        camera.yaw = -0.49;
        camera.pitch = 0.0;
        camera.xw_angle = 0.58;
        camera.zw_angle = 0.65;
    }

    if let Some(pos) = args.screenshot_pos.as_ref() {
        if pos.len() == 4 {
            camera.position = [pos[0], pos[1], pos[2], pos[3]];
        }
    }
    if let Some(angles) = args.screenshot_angles.as_ref() {
        if angles.len() == 4 {
            camera.yaw = angles[0];
            camera.pitch = angles[1];
            camera.xw_angle = angles[2];
            camera.zw_angle = angles[3];
        }
    } else if let Some(angles_deg) = args.screenshot_angles_deg.as_ref() {
        if angles_deg.len() == 4 {
            camera.yaw = angles_deg[0].to_radians();
            camera.pitch = angles_deg[1].to_radians();
            camera.xw_angle = angles_deg[2].to_radians();
            camera.zw_angle = angles_deg[3].to_radians();
        }
    }
    if let Some(yw) = args.screenshot_yw {
        camera.yw_deviation = yw;
    }

    if gpu_screenshot {
        eprintln!(
            "GPU screenshot camera: pos=({:+.4}, {:+.4}, {:+.4}, {:+.4}) angles(rad)=({:+.4}, {:+.4}, {:+.4}, {:+.4}) yw={:+.4}",
            camera.position[0],
            camera.position[1],
            camera.position[2],
            camera.position[3],
            camera.yaw,
            camera.pitch,
            camera.xw_angle,
            camera.zw_angle,
            camera.yw_deviation
        );
    }

    let vte_reference_mismatch_only_enabled =
        env_flag_enabled("R4D_VTE_REFERENCE_MISMATCH_ONLY");
    let vte_compare_slice_only_enabled = env_flag_enabled("R4D_VTE_COMPARE_SLICE_ONLY");
    let vte_reference_compare_enabled =
        env_flag_enabled("R4D_VTE_REFERENCE_COMPARE")
            || vte_reference_mismatch_only_enabled
            || vte_compare_slice_only_enabled;

    let mut app = App {
        instance,
        device,
        queue,
        rcx: None,
        scene: Scene::new(args.scene.to_scene_preset()),
        camera,
        input: InputState::new(),
        start_time: Instant::now(),
        last_frame: Instant::now(),
        mouse_grabbed: false,
        should_exit_after_render: false,
        gpu_screenshot_countdown: if gpu_screenshot {
            match args.gpu_screenshot_source {
                GpuScreenshotSourceArg::RenderBuffer => 3,
                GpuScreenshotSourceArg::Framebuffer => 6,
            }
        } else {
            0
        },
        args,
        control_scheme: ControlScheme::SideButtonLayers,
        scroll_cycle_pair: RotationPair::Standard,
        move_speed: 5.0,
        vte_reference_compare_enabled,
        vte_reference_mismatch_only_enabled,
        vte_compare_slice_only_enabled,
    };

    if app.vte_reference_compare_enabled {
        eprintln!("VTE reference compare enabled via R4D_VTE_REFERENCE_COMPARE");
    }
    if app.vte_reference_mismatch_only_enabled {
        eprintln!(
            "VTE mismatch-only visualization enabled via R4D_VTE_REFERENCE_MISMATCH_ONLY"
        );
    }
    if app.vte_compare_slice_only_enabled {
        eprintln!("VTE compare slice-only mode enabled via R4D_VTE_COMPARE_SLICE_ONLY");
    }

    event_loop.run_app(&mut app).unwrap();
}

fn run_cpu_render(scene_preset: ScenePreset, args: &Args) {
    use common::MatN;

    let mut scene = Scene::new(scene_preset);
    let mut camera = Camera4D::new();

    // Debug camera: specific position/orientation where GPU renders incorrectly
    camera.position = [5.44, 0.47, -1.23, -4.00];
    camera.yaw = -0.49;
    camera.pitch = 0.0;
    camera.xw_angle = 0.58;
    camera.zw_angle = 0.65;

    if let Some(pos) = args.screenshot_pos.as_ref() {
        if pos.len() == 4 {
            camera.position = [pos[0], pos[1], pos[2], pos[3]];
        }
    }
    if let Some(angles) = args.screenshot_angles.as_ref() {
        if angles.len() == 4 {
            camera.yaw = angles[0];
            camera.pitch = angles[1];
            camera.xw_angle = angles[2];
            camera.zw_angle = angles[3];
        }
    } else if let Some(angles_deg) = args.screenshot_angles_deg.as_ref() {
        if angles_deg.len() == 4 {
            camera.yaw = angles_deg[0].to_radians();
            camera.pitch = angles_deg[1].to_radians();
            camera.xw_angle = angles_deg[2].to_radians();
            camera.zw_angle = angles_deg[3].to_radians();
        }
    }
    if let Some(yw) = args.screenshot_yw {
        camera.yw_deviation = yw;
    }

    scene.update_surfaces_if_dirty();
    let instances = scene.build_instances(camera.position);

    let view_matrix_ndarray = camera.view_matrix();
    let view_matrix: MatN<5> = MatN::from(&view_matrix_ndarray);

    let model_tets = higher_dimension_playground::render::generate_tesseract_tetrahedrons();

    let params = cpu_render::CpuRenderParams {
        view_matrix,
        focal_length_xy: 1.0,
        focal_length_zw: 1.0,
        width: args.width.max(16),
        height: args.height.max(16),
        ..Default::default()
    };

    eprintln!("CPU render: {}x{}", params.width, params.height);
    let start = Instant::now();
    let img = cpu_render::cpu_render(instances, &model_tets, &params);
    let elapsed = start.elapsed();
    eprintln!("CPU render done in {:.2}s", elapsed.as_secs_f32());

    let _ = std::fs::create_dir_all("frames");
    img.save("frames/cpu_render.png").unwrap();
    eprintln!("Saved frames/cpu_render.png");
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    scene: Scene,
    camera: Camera4D,
    input: InputState,
    start_time: Instant,
    last_frame: Instant,
    mouse_grabbed: bool,
    should_exit_after_render: bool,
    gpu_screenshot_countdown: u32,
    args: Args,
    control_scheme: ControlScheme,
    scroll_cycle_pair: RotationPair,
    move_speed: f32,
    vte_reference_compare_enabled: bool,
    vte_reference_mismatch_only_enabled: bool,
    vte_compare_slice_only_enabled: bool,
}

fn latest_framebuffer_screenshot_path() -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir("frames").ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        if !(name.starts_with("framebuffer_") && name.ends_with(".webp")) {
            continue;
        }
        let modified = entry.metadata().ok()?.modified().ok()?;
        let replace = best
            .as_ref()
            .map(|(best_time, _)| modified > *best_time)
            .unwrap_or(true);
        if replace {
            best = Some((modified, path));
        }
    }
    best.map(|(_, path)| path)
}

fn env_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "off" || s == "no")
        }
        Err(_) => false,
    }
}

impl App {
    fn grab_mouse(&mut self, window: &Window) {
        let result = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if result.is_ok() {
            window.set_cursor_visible(false);
            self.mouse_grabbed = true;
            self.input.clear_mouse_delta();
        }
    }

    fn release_mouse(&mut self, window: &Window) {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.mouse_grabbed = false;
        self.input.clear_mouse_delta();
    }

    fn active_rotation_pair(&self) -> RotationPair {
        if self.input.mouse_back_held() && self.input.mouse_forward_held() {
            RotationPair::DoubleRotation
        } else if self.input.mouse_back_held() {
            RotationPair::FourD
        } else {
            match self.control_scheme {
                ControlScheme::SideButtonLayers => RotationPair::Standard,
                ControlScheme::ScrollCycle => self.scroll_cycle_pair,
            }
        }
    }

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Scheme cycle (Tab)
        if self.input.take_scheme_cycle() {
            self.control_scheme = self.control_scheme.next();
            self.scroll_cycle_pair = RotationPair::Standard;
        }

        // Reset orientation (R)
        if self.input.take_reset_orientation() {
            self.camera.reset_orientation();
        }

        // Scroll wheel
        let scroll = self.input.take_scroll();
        if scroll != 0.0 {
            match self.control_scheme {
                ControlScheme::SideButtonLayers => {
                    let factor = 1.1_f32.powf(scroll);
                    self.move_speed = (self.move_speed * factor).clamp(0.5, 100.0);
                }
                ControlScheme::ScrollCycle => {
                    if scroll > 0.0 {
                        self.scroll_cycle_pair = self.scroll_cycle_pair.next();
                    } else {
                        // Reverse cycle
                        self.scroll_cycle_pair = match self.scroll_cycle_pair {
                            RotationPair::Standard => RotationPair::FourD,
                            RotationPair::FourD => RotationPair::Standard,
                            RotationPair::DoubleRotation => RotationPair::Standard,
                        };
                    }
                }
            }
        }

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        // Mouse look
        if self.mouse_grabbed {
            let (dx, dy) = self.input.take_mouse_delta();
            self.camera.apply_mouse_look_on(
                dx,
                dy,
                MOUSE_SENSITIVITY,
                pair.h_target(),
                pair.v_target(),
            );
        } else {
            self.input.take_mouse_delta();
        }

        // Auto-level when not in double rotation
        if pair != RotationPair::DoubleRotation {
            self.camera.auto_level(dt);
        }

        // Toggle flight mode on double-tap space
        if self.input.take_fly_toggle() {
            self.camera.toggle_flying();
        }

        // Jump when in gravity mode, consume jump either way
        if self.camera.is_flying {
            self.input.take_jump();
        } else if self.input.take_jump() {
            self.camera.jump();
        }

        // Movement (vertical zeroed in gravity mode internally)
        let (forward, strafe, vertical, w_axis) = self.input.movement_axes();
        self.camera
            .apply_movement(forward, strafe, vertical, w_axis, dt, self.move_speed);

        // Apply gravity physics
        self.camera.update_physics(dt);

        // Build view matrix and scene
        let view_matrix = self.camera.view_matrix();
        let backend = self.args.backend.to_render_backend();

        let auto_screenshot = if self.gpu_screenshot_countdown > 1 {
            self.gpu_screenshot_countdown -= 1;
            false
        } else if self.gpu_screenshot_countdown == 1 {
            self.gpu_screenshot_countdown = 0;
            true
        } else {
            false
        };
        let manual_screenshot = self.input.take_screenshot();
        let mut take_screenshot = manual_screenshot;
        if auto_screenshot && self.args.gpu_screenshot_source == GpuScreenshotSourceArg::Framebuffer
        {
            take_screenshot = true;
        }
        if take_screenshot {
            let _ = std::fs::create_dir_all("frames");
            if auto_screenshot {
                self.should_exit_after_render = true;
            }
        }

        let render_options = RenderOptions {
            do_raster: true,
            render_backend: backend,
            vte_max_trace_steps: self.args.vte_max_trace_steps.max(1),
            vte_max_trace_distance: self.args.vte_max_trace_distance.max(1.0),
            vte_display_mode: self.args.vte_display_mode.to_render_mode(),
            vte_slice_layer: self.args.vte_slice_layer,
            vte_thick_half_width: self.args.vte_thick_half_width,
            vte_reference_compare: self.vte_reference_compare_enabled,
            vte_reference_mismatch_only: self.vte_reference_mismatch_only_enabled,
            vte_compare_slice_only: self.vte_compare_slice_only_enabled,
            do_navigation_hud: true,
            take_framebuffer_screenshot: take_screenshot,
            prepare_render_screenshot: auto_screenshot,
            hud_rotation_label: Some({
                let inv = if self.camera.is_y_inverted() {
                    " Y-INV"
                } else {
                    ""
                };
                format!(
                    "{} [{}]  spd:{:.1}{}\n\
                     yaw:{:+.0} pit:{:+.0} xw:{:+.0} zw:{:+.0} yw:{:+.1}",
                    pair.label(),
                    self.control_scheme.label(),
                    self.move_speed,
                    inv,
                    self.camera.yaw.to_degrees(),
                    self.camera.pitch.to_degrees(),
                    self.camera.xw_angle.to_degrees(),
                    self.camera.zw_angle.to_degrees(),
                    self.camera.yw_deviation.to_degrees(),
                )
            }),
            ..Default::default()
        };

        let frame_params = FrameParams {
            view_matrix,
            focal_length_xy: 1.0,
            focal_length_zw: 1.0,
            render_options,
        };

        if backend == RenderBackend::VoxelTraversal {
            let voxel_frame = self.scene.build_voxel_frame_data(self.camera.position);
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_frame.as_input(),
            );
        } else {
            self.scene.update_surfaces_if_dirty();
            let instances = self.scene.build_instances(self.camera.position);
            self.rcx.as_mut().unwrap().render_tetra_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                TetraFrameInput {
                    model_instances: instances,
                },
            );
        }

        if auto_screenshot {
            let _ = std::fs::create_dir_all("frames");
            match self.args.gpu_screenshot_source {
                GpuScreenshotSourceArg::RenderBuffer => {
                    self.rcx
                        .as_mut()
                        .unwrap()
                        .save_rendered_frame_png("frames/gpu_render.png");
                }
                GpuScreenshotSourceArg::Framebuffer => {
                    if let Some(src_path) = latest_framebuffer_screenshot_path() {
                        if let Err(err) = std::fs::copy(&src_path, "frames/gpu_render.webp") {
                            eprintln!(
                                "Failed to copy framebuffer screenshot {} -> frames/gpu_render.webp: {}",
                                src_path.display(),
                                err
                            );
                        }
                        match image::open(&src_path) {
                            Ok(img) => {
                                if let Err(err) = img.save("frames/gpu_render.png") {
                                    eprintln!("Failed to save frames/gpu_render.png: {err}");
                                } else {
                                    println!("Saved PNG to frames/gpu_render.png");
                                }
                            }
                            Err(err) => {
                                eprintln!(
                                    "Failed to decode framebuffer screenshot {}: {}",
                                    src_path.display(),
                                    err
                                );
                            }
                        }
                    } else {
                        eprintln!(
                            "No framebuffer screenshot found under frames/ after auto-capture."
                        );
                    }
                }
            }
            self.should_exit_after_render = true;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        self.grab_mouse(&window);
        self.rcx = Some(RenderContext::new(
            self.device.clone(),
            self.queue.clone(),
            self.instance.clone(),
            Some(window),
            [self.args.width, self.args.height, self.args.layers],
        ));
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.handle_key_event(&event);

                if self.input.take_escape() {
                    if self.mouse_grabbed {
                        let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                        self.release_mouse(&window);
                    } else {
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Left => {
                    if state.is_pressed() && !self.mouse_grabbed {
                        let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                        self.grab_mouse(&window);
                    }
                }
                MouseButton::Back | MouseButton::Forward => {
                    self.input.handle_mouse_button(button, state);
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                self.input.handle_scroll(y);
            }
            WindowEvent::Focused(false) => {
                if self.mouse_grabbed {
                    let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                    self.release_mouse(&window);
                }
            }
            WindowEvent::RedrawRequested => {
                self.update_and_render();
                if self.should_exit_after_render {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_grabbed {
                self.input.handle_mouse_motion(delta);
                if let Some(rcx) = self.rcx.as_ref() {
                    if let Some(window) = rcx.window.as_ref() {
                        window.request_redraw();
                    }
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(rcx) = self.rcx.as_ref() {
            if let Some(window) = rcx.window.as_ref() {
                window.request_redraw();
            }
        }
    }
}
