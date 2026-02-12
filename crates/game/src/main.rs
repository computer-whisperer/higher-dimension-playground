mod camera;
mod cpu_render;
mod input;
mod scene;
mod voxel;

use clap::{Parser, ValueEnum};
use higher_dimension_playground::render::{
    CustomOverlayLine, FrameParams, RenderBackend, RenderContext, RenderOptions, TetraFrameInput,
    VteDisplayMode,
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
const BLOCK_EDIT_REACH_DEFAULT: f32 = 8.0;
const BLOCK_EDIT_REACH_MIN: f32 = 1.0;
const BLOCK_EDIT_REACH_MAX: f32 = 48.0;
const BLOCK_EDIT_PLACE_MATERIAL_DEFAULT: u8 = 3;
const BLOCK_EDIT_PLACE_MATERIAL_MIN: u8 = 1;
const BLOCK_EDIT_PLACE_MATERIAL_MAX: u8 = 20;
const TARGET_OUTLINE_COLOR: [f32; 4] = [0.14, 0.70, 0.70, 1.00];
const PLACE_OUTLINE_COLOR: [f32; 4] = [0.70, 0.42, 0.14, 1.00];
const WORLD_FILE_DEFAULT: &str = "saves/world.v4dw";
const VTE_TEST_ENTITY_CENTER: [f32; 4] = [0.0, 3.0, 0.0, 0.0];

#[derive(Copy, Clone, Debug, ValueEnum)]
enum EditHighlightModeArg {
    Faces,
    Edges,
    Both,
    Off,
}

impl EditHighlightModeArg {
    fn uses_faces(self) -> bool {
        matches!(self, Self::Faces | Self::Both)
    }

    fn uses_edges(self) -> bool {
        matches!(self, Self::Edges | Self::Both)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Faces => "faces",
            Self::Edges => "edges",
            Self::Both => "both",
            Self::Off => "off",
        }
    }
}

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

    /// Edit highlight mode for pointed/placement voxel guides.
    /// `faces` uses in-VTE occlusion-correct face highlighting.
    /// `edges` keeps the legacy overlay-line outline debug view.
    #[arg(long, value_enum, default_value_t = EditHighlightModeArg::Faces)]
    edit_highlight_mode: EditHighlightModeArg,

    /// Block interaction reach limit in world units.
    #[arg(long, default_value_t = BLOCK_EDIT_REACH_DEFAULT)]
    edit_reach: f32,

    /// Path used for manual world save/load.
    #[arg(long, default_value = WORLD_FILE_DEFAULT)]
    world_file: PathBuf,

    /// Load `--world-file` at startup.
    #[arg(long)]
    load_world: bool,
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

    let vte_reference_mismatch_only_enabled = env_flag_enabled("R4D_VTE_REFERENCE_MISMATCH_ONLY");
    let vte_compare_slice_only_enabled = env_flag_enabled("R4D_VTE_COMPARE_SLICE_ONLY");
    let vte_reference_compare_enabled = env_flag_enabled("R4D_VTE_REFERENCE_COMPARE")
        || vte_reference_mismatch_only_enabled
        || vte_compare_slice_only_enabled;

    let world_file = args.world_file.clone();
    let mut scene = Scene::new(args.scene.to_scene_preset());
    if args.load_world {
        match scene.load_world_from_path(&world_file) {
            Ok(chunks) => eprintln!(
                "Loaded world from {} ({} non-empty chunks)",
                world_file.display(),
                chunks
            ),
            Err(err) => eprintln!("Failed to load world from {}: {err}", world_file.display()),
        }
    }

    let mut app = App {
        instance,
        device,
        queue,
        rcx: None,
        scene,
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
        control_scheme: ControlScheme::IntuitiveUpright,
        scroll_cycle_pair: RotationPair::Standard,
        move_speed: 5.0,
        place_material: BLOCK_EDIT_PLACE_MATERIAL_DEFAULT,
        world_file,
        vte_reference_compare_enabled,
        vte_reference_mismatch_only_enabled,
        vte_compare_slice_only_enabled,
    };

    if app.vte_reference_compare_enabled {
        eprintln!("VTE reference compare enabled via R4D_VTE_REFERENCE_COMPARE");
    }
    if app.vte_reference_mismatch_only_enabled {
        eprintln!("VTE mismatch-only visualization enabled via R4D_VTE_REFERENCE_MISMATCH_ONLY");
    }
    if app.vte_compare_slice_only_enabled {
        eprintln!("VTE compare slice-only mode enabled via R4D_VTE_COMPARE_SLICE_ONLY");
    }

    event_loop.run_app(&mut app).unwrap();
}

fn run_cpu_render(scene_preset: ScenePreset, args: &Args) {
    use common::MatN;

    let mut scene = Scene::new(scene_preset);
    if args.load_world {
        if let Err(err) = scene.load_world_from_path(&args.world_file) {
            eprintln!(
                "Failed to load world from {}: {err}",
                args.world_file.display()
            );
        }
    }
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
    place_material: u8,
    world_file: PathBuf,
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

fn project_world_point_to_ndc(
    view_matrix: &ndarray::Array2<f32>,
    world_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<[f32; 2]> {
    let input = [
        world_point[0],
        world_point[1],
        world_point[2],
        world_point[3],
        1.0,
    ];
    let mut view_h = [0.0f32; 5];
    for row in 0..5 {
        for col in 0..5 {
            view_h[row] += view_matrix[[row, col]] * input[col];
        }
    }

    let inv_w = if view_h[4].abs() > 1e-6 {
        view_h[4].recip()
    } else {
        1.0
    };
    let view = [
        view_h[0] * inv_w,
        view_h[1] * inv_w,
        view_h[2] * inv_w,
        view_h[3] * inv_w,
    ];

    let depth = (view[2] * view[2] + view[3] * view[3]).sqrt();
    if !depth.is_finite() || depth < 1e-4 {
        return None;
    }
    let projection_divisor = depth / focal_length_xy.max(1e-4);
    let x = view[0] / projection_divisor;
    let y = aspect * (-view[1]) / projection_divisor;
    if x.is_finite() && y.is_finite() {
        Some([x, y])
    } else {
        None
    }
}

fn append_voxel_outline_lines(
    overlay_lines: &mut Vec<CustomOverlayLine>,
    view_matrix: &ndarray::Array2<f32>,
    voxel: [i32; 4],
    focal_length_xy: f32,
    aspect: f32,
    color: [f32; 4],
) {
    let mut projected_vertices: [Option<[f32; 2]>; 16] = [None; 16];
    for vertex_mask in 0..16usize {
        let world_point = [
            voxel[0] as f32 + (vertex_mask & 1) as f32,
            voxel[1] as f32 + ((vertex_mask >> 1) & 1) as f32,
            voxel[2] as f32 + ((vertex_mask >> 2) & 1) as f32,
            voxel[3] as f32 + ((vertex_mask >> 3) & 1) as f32,
        ];
        projected_vertices[vertex_mask] =
            project_world_point_to_ndc(view_matrix, world_point, focal_length_xy, aspect);
    }

    for vertex_mask in 0..16usize {
        for axis in 0..4usize {
            if ((vertex_mask >> axis) & 1) != 0 {
                continue;
            }
            let next_mask = vertex_mask | (1usize << axis);
            let Some(start_ndc) = projected_vertices[vertex_mask] else {
                continue;
            };
            let Some(end_ndc) = projected_vertices[next_mask] else {
                continue;
            };
            overlay_lines.push(CustomOverlayLine {
                start_ndc,
                end_ndc,
                color,
            });
        }
    }
}

fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if len_sq <= 1e-8 {
        return v;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

fn rotate_basis_plane(basis: &mut [[f32; 4]; 4], axis_a: usize, axis_b: usize, angle: f32) {
    let c = angle.cos();
    let s = angle.sin();
    let old_a = basis[axis_a];
    let old_b = basis[axis_b];
    for i in 0..4 {
        basis[axis_a][i] = c * old_a[i] + s * old_b[i];
        basis[axis_b][i] = -s * old_a[i] + c * old_b[i];
    }
}

fn build_place_preview_instance(
    camera: &Camera4D,
    selected_material: u8,
    time_s: f32,
    control_scheme: ControlScheme,
) -> common::ModelInstance {
    let material = selected_material
        .clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX)
        as u32;
    let (right, up, view_z, view_w) = match control_scheme {
        ControlScheme::IntuitiveUpright => camera.view_basis_upright(),
        ControlScheme::LookTransport | ControlScheme::RotorFree => camera.view_basis_look_frame(),
        ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
            camera.view_basis()
        }
    };

    let center_forward = normalize4([
        view_z[0] + view_w[0],
        view_z[1] + view_w[1],
        view_z[2] + view_w[2],
        view_z[3] + view_w[3],
    ]);
    let side_w = normalize4([
        view_w[0] - view_z[0],
        view_w[1] - view_z[1],
        view_w[2] - view_z[2],
        view_w[3] - view_z[3],
    ]);

    let anchor = [
        camera.position[0] + 0.92 * right[0] - 0.78 * up[0]
            + 1.35 * center_forward[0]
            + 0.52 * side_w[0],
        camera.position[1] + 0.92 * right[1] - 0.78 * up[1]
            + 1.35 * center_forward[1]
            + 0.52 * side_w[1],
        camera.position[2] + 0.92 * right[2] - 0.78 * up[2]
            + 1.35 * center_forward[2]
            + 0.52 * side_w[2],
        camera.position[3] + 0.92 * right[3] - 0.78 * up[3]
            + 1.35 * center_forward[3]
            + 0.52 * side_w[3],
    ];

    let mut basis = [right, up, center_forward, side_w];
    rotate_basis_plane(&mut basis, 0, 2, time_s * 0.85 + 0.2);
    rotate_basis_plane(&mut basis, 0, 3, time_s * 0.55 + 0.9);

    let scale = 0.23;
    let mut model_transform = common::MatN::<5>::identity();
    for row in 0..4 {
        model_transform[[row, 0]] = basis[0][row] * scale;
        model_transform[[row, 1]] = basis[1][row] * scale;
        model_transform[[row, 2]] = basis[2][row] * scale;
        model_transform[[row, 3]] = basis[3][row] * scale;
        model_transform[[row, 4]] = anchor[row];
    }

    common::ModelInstance {
        model_transform,
        cell_material_ids: [material; 8],
    }
}

fn build_vte_test_entity_instance(time_s: f32) -> common::ModelInstance {
    let mut basis = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    rotate_basis_plane(&mut basis, 0, 2, time_s * 0.55);
    rotate_basis_plane(&mut basis, 1, 3, time_s * 0.85 + 0.3);
    rotate_basis_plane(&mut basis, 0, 3, time_s * 0.35 + 1.1);

    let scale = 0.65;
    let mut model_transform = common::MatN::<5>::identity();
    for row in 0..4 {
        model_transform[[row, 0]] = basis[0][row] * scale;
        model_transform[[row, 1]] = basis[1][row] * scale;
        model_transform[[row, 2]] = basis[2][row] * scale;
        model_transform[[row, 3]] = basis[3][row] * scale;

        let center_offset = 0.5
            * (model_transform[[row, 0]]
                + model_transform[[row, 1]]
                + model_transform[[row, 2]]
                + model_transform[[row, 3]]);
        model_transform[[row, 4]] = VTE_TEST_ENTITY_CENTER[row] - center_offset;
    }

    common::ModelInstance {
        model_transform,
        cell_material_ids: [12; 8],
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

    fn cycle_place_material_prev(&mut self) {
        self.place_material = if self.place_material <= BLOCK_EDIT_PLACE_MATERIAL_MIN {
            BLOCK_EDIT_PLACE_MATERIAL_MAX
        } else {
            self.place_material.saturating_sub(1)
        };
    }

    fn cycle_place_material_next(&mut self) {
        self.place_material = if self.place_material >= BLOCK_EDIT_PLACE_MATERIAL_MAX {
            BLOCK_EDIT_PLACE_MATERIAL_MIN
        } else {
            self.place_material.saturating_add(1)
        };
    }

    fn active_rotation_pair(&self) -> RotationPair {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => {
                if self.input.mouse_back_held() {
                    RotationPair::FourD
                } else {
                    RotationPair::Standard
                }
            }
            ControlScheme::LookTransport | ControlScheme::RotorFree => RotationPair::Standard,
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                if self.input.mouse_back_held() && self.input.mouse_forward_held() {
                    RotationPair::DoubleRotation
                } else if self.input.mouse_back_held() {
                    RotationPair::FourD
                } else if self.control_scheme.uses_scroll_pair_cycle() {
                    self.scroll_cycle_pair
                } else {
                    RotationPair::Standard
                }
            }
        }
    }

    fn current_look_direction(&self) -> [f32; 4] {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.look_direction_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.look_direction_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.look_direction()
            }
        }
    }

    fn current_view_matrix(&self) -> ndarray::Array2<f32> {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.view_matrix_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.view_matrix_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.view_matrix()
            }
        }
    }

    fn current_y_inverted(&self) -> bool {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.is_y_inverted_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.is_y_inverted_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.is_y_inverted()
            }
        }
    }

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Scheme cycle (Tab)
        if self.input.take_scheme_cycle() {
            let previous_scheme = self.control_scheme;
            self.control_scheme = self.control_scheme.next();
            self.scroll_cycle_pair = RotationPair::Standard;
            if self.control_scheme.is_upright_primary() {
                self.camera.enforce_upright_constraints();
            } else if self.control_scheme.uses_look_frame()
                && (!previous_scheme.uses_look_frame() || !self.camera.look_frame_initialized())
            {
                if previous_scheme.is_upright_primary() {
                    self.camera.sync_look_frame_from_upright_rotation();
                } else {
                    self.camera.sync_look_frame_from_standard_rotation();
                }
            }
        }

        // Reset orientation (R)
        if self.input.take_reset_orientation() {
            self.camera.reset_orientation();
        }

        if self.input.take_save_world() {
            match self.scene.save_world_to_path(&self.world_file) {
                Ok(chunks) => eprintln!(
                    "Saved world to {} ({} non-empty chunks)",
                    self.world_file.display(),
                    chunks
                ),
                Err(err) => eprintln!(
                    "Failed to save world to {}: {err}",
                    self.world_file.display()
                ),
            }
        }

        if self.input.take_load_world() {
            match self.scene.load_world_from_path(&self.world_file) {
                Ok(chunks) => eprintln!(
                    "Loaded world from {} ({} non-empty chunks)",
                    self.world_file.display(),
                    chunks
                ),
                Err(err) => eprintln!(
                    "Failed to load world from {}: {err}",
                    self.world_file.display()
                ),
            }
        }

        // Scroll wheel
        let scroll_steps = self.input.take_scroll_steps();
        if scroll_steps != 0 {
            if self.control_scheme.uses_scroll_pair_cycle() {
                for _ in 0..scroll_steps.abs() {
                    if scroll_steps > 0 {
                        self.scroll_cycle_pair = self.scroll_cycle_pair.next();
                    } else {
                        // Reverse cycle for the legacy scroll mode.
                        self.scroll_cycle_pair = match self.scroll_cycle_pair {
                            RotationPair::Standard => RotationPair::FourD,
                            RotationPair::FourD => RotationPair::Standard,
                            RotationPair::DoubleRotation => RotationPair::Standard,
                        };
                    }
                }
            } else {
                for _ in 0..scroll_steps.abs() {
                    if scroll_steps > 0 {
                        self.cycle_place_material_next();
                    } else {
                        self.cycle_place_material_prev();
                    }
                }
                eprintln!("Selected place material: {}", self.place_material);
            }
        }

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        // Mouse look
        if self.mouse_grabbed {
            let (dx, dy) = self.input.take_mouse_delta();
            match self.control_scheme {
                ControlScheme::LookTransport => {
                    self.camera.apply_mouse_look_transport(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                    );
                }
                ControlScheme::RotorFree => {
                    self.camera.apply_mouse_look_rotor(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        self.input.mouse_back_held(),
                        self.input.mouse_forward_held(),
                    );
                }
                ControlScheme::IntuitiveUpright
                | ControlScheme::LegacySideButtonLayers
                | ControlScheme::LegacyScrollCycle => {
                    self.camera.apply_mouse_look_on(
                        dx,
                        dy,
                        MOUSE_SENSITIVITY,
                        pair.h_target(),
                        pair.v_target(),
                    );
                }
            }
        } else {
            self.input.take_mouse_delta();
        }

        match self.control_scheme {
            ControlScheme::IntuitiveUpright => {
                self.camera.enforce_upright_constraints();
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                // Auto-level when not in double rotation
                if pair != RotationPair::DoubleRotation {
                    self.camera.auto_level(dt);
                }
            }
            ControlScheme::LookTransport | ControlScheme::RotorFree => {}
        }

        // Toggle flight mode on double-tap space
        if self.input.take_fly_toggle() {
            self.camera.toggle_flying();
        }

        // Block place material selection.
        if self.input.take_place_material_prev() {
            self.cycle_place_material_prev();
            eprintln!("Selected place material: {}", self.place_material);
        }
        if self.input.take_place_material_next() {
            self.cycle_place_material_next();
            eprintln!("Selected place material: {}", self.place_material);
        }
        if let Some(material_digit) = self.input.take_place_material_digit() {
            self.place_material =
                material_digit.clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX);
            eprintln!("Selected place material: {}", self.place_material);
        }

        // Jump when in gravity mode, consume jump either way
        if self.camera.is_flying {
            self.input.take_jump();
        } else if self.input.take_jump() {
            self.camera.jump();
        }

        // Movement (vertical zeroed in gravity mode internally)
        let prev_position = self.camera.position;
        let look_dir = self.current_look_direction();
        let edit_reach = self
            .args
            .edit_reach
            .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);
        let (forward, strafe, vertical, w_axis) = self.input.movement_axes();
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => {
                self.camera.apply_movement_upright(
                    forward,
                    strafe,
                    vertical,
                    w_axis,
                    dt,
                    self.move_speed,
                );
            }
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.apply_movement_look_frame(
                    forward,
                    strafe,
                    vertical,
                    w_axis,
                    dt,
                    self.move_speed,
                );
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera
                    .apply_movement(forward, strafe, vertical, w_axis, dt, self.move_speed);
            }
        }
        let preview_time_s = (now - self.start_time).as_secs_f32();
        let preview_instance = build_place_preview_instance(
            &self.camera,
            self.place_material,
            preview_time_s,
            self.control_scheme,
        );

        // Apply gravity physics
        self.camera.update_physics(dt);
        if self.camera.is_flying {
            self.camera.is_grounded = false;
        } else {
            let (resolved_pos, grounded) = self.scene.resolve_player_collision(
                prev_position,
                self.camera.position,
                &mut self.camera.velocity_y,
            );
            self.camera.position = resolved_pos;
            self.camera.is_grounded = grounded;
        }

        if self.mouse_grabbed {
            let remove_requested = self.input.take_remove_block();
            let place_requested = self.input.take_place_block();
            if remove_requested || place_requested {
                if remove_requested {
                    if let Some([x, y, z, w]) = self.scene.remove_block_along_ray(
                        self.camera.position,
                        look_dir,
                        edit_reach,
                    ) {
                        eprintln!("Removed voxel at ({x}, {y}, {z}, {w})");
                    }
                } else if place_requested {
                    if let Some([x, y, z, w]) = self.scene.place_block_along_ray(
                        self.camera.position,
                        look_dir,
                        edit_reach,
                        voxel::VoxelType(self.place_material),
                    ) {
                        eprintln!(
                            "Placed voxel material {} at ({x}, {y}, {z}, {w})",
                            self.place_material
                        );
                    }
                }
            }
        } else {
            self.input.take_remove_block();
            self.input.take_place_block();
        }

        // Build view matrix and scene
        let view_matrix = self.current_view_matrix();
        let backend = self.args.backend.to_render_backend();
        let highlight_mode = self.args.edit_highlight_mode;
        let targets =
            if self.mouse_grabbed && (highlight_mode.uses_faces() || highlight_mode.uses_edges()) {
                Some(
                    self.scene
                        .block_edit_targets(self.camera.position, look_dir, edit_reach),
                )
            } else {
                None
            };
        let mut hud_target_hit_voxel = None;
        let mut hud_target_hit_face = None;
        if let Some(targets) = targets {
            hud_target_hit_voxel = targets.hit_voxel;
            if let (Some(hit), Some(place)) = (targets.hit_voxel, targets.place_voxel) {
                let face = [
                    place[0] - hit[0],
                    place[1] - hit[1],
                    place[2] - hit[2],
                    place[3] - hit[3],
                ];
                let manhattan = face[0].abs() + face[1].abs() + face[2].abs() + face[3].abs();
                if manhattan == 1 {
                    hud_target_hit_face = Some(face);
                }
            }
        }

        let mut custom_overlay_lines = Vec::with_capacity(64);
        if highlight_mode.uses_edges() {
            if let Some(targets) = targets {
                let aspect = self.args.width.max(1) as f32 / self.args.height.max(1) as f32;
                if let Some(hit_voxel) = targets.hit_voxel {
                    append_voxel_outline_lines(
                        &mut custom_overlay_lines,
                        &view_matrix,
                        hit_voxel,
                        1.0,
                        aspect,
                        TARGET_OUTLINE_COLOR,
                    );
                }
                if let Some(place_voxel) = targets.place_voxel {
                    append_voxel_outline_lines(
                        &mut custom_overlay_lines,
                        &view_matrix,
                        place_voxel,
                        1.0,
                        aspect,
                        PLACE_OUTLINE_COLOR,
                    );
                }
            }
        }

        let mut vte_highlight_hit_voxel = None;
        let mut vte_highlight_place_voxel = None;
        if backend == RenderBackend::VoxelTraversal && highlight_mode.uses_faces() {
            if let Some(targets) = targets {
                vte_highlight_hit_voxel = targets.hit_voxel;
                vte_highlight_place_voxel = targets.place_voxel;
            }
        }

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
            vte_highlight_hit_voxel,
            vte_highlight_place_voxel,
            do_navigation_hud: true,
            custom_overlay_lines,
            take_framebuffer_screenshot: take_screenshot,
            prepare_render_screenshot: auto_screenshot,
            hud_rotation_label: Some({
                let inv = self.current_y_inverted();
                let inv = if inv { " Y-INV" } else { "" };
                if self.control_scheme.uses_look_frame() {
                    format!(
                        "LOOK-FRAME [{}]  spd:{:.1}{}\n\
                         look:{:+.2} {:+.2} {:+.2} {:+.2}\n\
                         edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                         mat:[ ]/wheel cycle, 1-0 direct\n\
                         world:F5 save, F9 load",
                        self.control_scheme.label(),
                        self.move_speed,
                        inv,
                        look_dir[0],
                        look_dir[1],
                        look_dir[2],
                        look_dir[3],
                        self.place_material,
                        edit_reach,
                        highlight_mode.label(),
                    )
                } else {
                    format!(
                        "{} [{}]  spd:{:.1}{}\n\
                         yaw:{:+.0} pit:{:+.0} xw:{:+.0} zw:{:+.0} yw:{:+.1}\n\
                         edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                         mat:[ ]/wheel cycle, 1-0 direct\n\
                         world:F5 save, F9 load",
                        pair.label(),
                        self.control_scheme.label(),
                        self.move_speed,
                        inv,
                        self.camera.yaw.to_degrees(),
                        self.camera.pitch.to_degrees(),
                        self.camera.xw_angle.to_degrees(),
                        self.camera.zw_angle.to_degrees(),
                        self.camera.yw_deviation.to_degrees(),
                        self.place_material,
                        edit_reach,
                        highlight_mode.label(),
                    )
                }
            }),
            hud_target_hit_voxel,
            hud_target_hit_face,
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
            let vte_entity_instances = [build_vte_test_entity_instance(preview_time_s)];
            let preview_overlay_instances = [preview_instance];
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_frame.as_input(),
                &vte_entity_instances,
                &preview_overlay_instances,
            );
        } else {
            self.scene.update_surfaces_if_dirty();
            let instances = self.scene.build_instances(self.camera.position);
            let mut render_instances = Vec::with_capacity(instances.len() + 1);
            render_instances.extend_from_slice(instances);
            render_instances.push(preview_instance);
            self.rcx.as_mut().unwrap().render_tetra_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                TetraFrameInput {
                    model_instances: &render_instances,
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
        let backend = self.args.backend.to_render_backend();
        let vte_mode = self.args.vte_display_mode.to_render_mode();
        let pixel_storage_layers =
            if backend == RenderBackend::VoxelTraversal && vte_mode == VteDisplayMode::Integral {
                Some(1)
            } else {
                None
            };
        self.rcx = Some(RenderContext::new_with_pixel_storage_layers(
            self.device.clone(),
            self.queue.clone(),
            self.instance.clone(),
            Some(window),
            [self.args.width, self.args.height, self.args.layers],
            pixel_storage_layers,
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
                    if state.is_pressed() {
                        if !self.mouse_grabbed {
                            let window = self.rcx.as_ref().unwrap().window.clone().unwrap();
                            self.grab_mouse(&window);
                        } else {
                            self.input.handle_mouse_button(button, state);
                        }
                    }
                }
                MouseButton::Right | MouseButton::Back | MouseButton::Forward => {
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
