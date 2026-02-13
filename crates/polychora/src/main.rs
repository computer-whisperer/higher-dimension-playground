mod camera;
mod cpu_render;
mod input;
mod multiplayer;
mod scene;
mod voxel;

use base64::Engine;
use clap::{ArgAction, Parser, ValueEnum};
use higher_dimension_playground::render::{
    CustomOverlayLine, FrameParams, HudPlayerTag, HudReadoutMode, RenderBackend, RenderContext,
    RenderOptions, TetraFrameInput, VteDisplayMode,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowId},
};

use camera::{Camera4D, PLAYER_HEIGHT};
use input::{ControlScheme, InputState, RotationPair};
use multiplayer::{ClientMessage as MultiplayerClientMessage, MultiplayerClient, MultiplayerEvent};
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
const VTE_SWEEP_SAMPLE_FRAMES: usize = 120;
const VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV: &str = "R4D_VTE_SWEEP_INCLUDE_NO_ENTITIES";
const FOCAL_LENGTH_MIN: f32 = 0.20;
const FOCAL_LENGTH_MAX: f32 = 4.00;
const FOCAL_LENGTH_STEP: f32 = 0.05;
const VTE_TRACE_DISTANCE_MIN: f32 = 10.0;
const VTE_TRACE_DISTANCE_MAX: f32 = 4096.0;
const VTE_TRACE_DISTANCE_STEP: f32 = 10.0;
const VTE_INTEGRAL_SKY_SCALE_MIN: f32 = 0.0;
const VTE_INTEGRAL_SKY_SCALE_MAX: f32 = 2.0;
const VTE_INTEGRAL_SKY_SCALE_STEP: f32 = 0.05;
const VTE_INTEGRAL_HIT_EMISSIVE_MIN: f32 = 0.0;
const VTE_INTEGRAL_HIT_EMISSIVE_MAX: f32 = 0.20;
const VTE_INTEGRAL_HIT_EMISSIVE_STEP: f32 = 0.005;
const VTE_INTEGRAL_LOG_MERGE_K_MIN: f32 = 0.0;
const VTE_INTEGRAL_LOG_MERGE_K_MAX: f32 = 64.0;
const VTE_INTEGRAL_LOG_MERGE_K_STEP: f32 = 0.5;
const VTE_TRACE_STEPS_MIN: u32 = 16;
const VTE_TRACE_STEPS_MAX: u32 = 4096;
const MULTIPLAYER_DEFAULT_PORT: u16 = 4000;
const MULTIPLAYER_PLAYER_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const MULTIPLAYER_PENDING_EDIT_TIMEOUT: Duration = Duration::from_secs(5);
const MULTIPLAYER_PENDING_EDIT_MAX: usize = 512;
const AVATAR_MATERIAL_ID: u32 = 21;
const AVATAR_FORWARD_FRAGMENT_COUNT: usize = 4;
const REMOTE_AVATAR_PART_COUNT_ESTIMATE: usize = 7 + AVATAR_FORWARD_FRAGMENT_COUNT;
const AVATAR_THICKNESS_SCALE: f32 = 1.30;
const AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE: f32 = 0.50;
const REMOTE_PLAYER_POSITION_SMOOTH_HZ: f32 = 12.0;
const REMOTE_PLAYER_LOOK_SMOOTH_HZ: f32 = 16.0;
const REMOTE_PLAYER_PREDICTION_LEAD_S: f32 = 0.05;
const REMOTE_PLAYER_MAX_PREDICTION_S: f32 = 0.22;
const REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE: f32 = 8.0;
const REMOTE_PLAYER_MAX_PREDICTED_SPEED: f32 = 24.0;
const REMOTE_PLAYER_TAG_FOV_DOT_MIN: f32 = 0.16;
const REMOTE_PLAYER_TAG_MAX_COUNT: usize = 32;

#[derive(Copy, Clone)]
struct VteRuntimeProfile {
    label: &'static str,
    entities: bool,
    y_slice_lookup_cache: bool,
}

const VTE_SWEEP_PROFILES_ENTITIES: [VteRuntimeProfile; 2] = [
    VteRuntimeProfile {
        label: "A baseline",
        entities: true,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "B no-lookup-cache",
        entities: true,
        y_slice_lookup_cache: false,
    },
];

const VTE_SWEEP_PROFILES_EXTENDED: [VteRuntimeProfile; 4] = [
    VteRuntimeProfile {
        label: "A baseline",
        entities: true,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "B no-entities",
        entities: false,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "C no-lookup-cache",
        entities: true,
        y_slice_lookup_cache: false,
    },
    VteRuntimeProfile {
        label: "D no-entities-no-lookup-cache",
        entities: false,
        y_slice_lookup_cache: false,
    },
];

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
#[command(version, about = "4D polychora explorer")]
struct Args {
    /// Render buffer width in pixels
    #[arg(long, short = 'W', default_value_t = 1920)]
    width: u32,

    /// Render buffer height in pixels
    #[arg(long, short = 'H', default_value_t = 1080)]
    height: u32,

    /// Number of depth layers (supersampling)
    #[arg(long, default_value_t = 128)]
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
    #[arg(long, value_enum, default_value_t = BackendArg::VoxelTraversal)]
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

    /// Enable tetra-entity integration in VTE Stage A (test tesseract + entity BVH).
    /// Set to false to profile pure voxel traversal without the secondary tetra pipeline.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    vte_entities: bool,

    /// Deprecated no-op: y-slice fastpath is now always enabled on normal VTE path.
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    vte_y_slice_fastpath: bool,

    /// Deprecated no-op: chunk solid clipping is now always enabled.
    #[arg(long, action = ArgAction::Set, default_value_t = true, hide = true)]
    vte_chunk_solid_clip: bool,

    /// Enable y-slice direct chunk-lookup table in Stage A.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    vte_y_slice_lookup_cache: bool,

    /// Enable fused-integral tweak: dim sky contribution and add small hit emissive floor.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    vte_integral_sky_emissive_tweak: bool,

    /// Sky scale applied in fused-integral tweak mode.
    #[arg(long, default_value_t = 0.40)]
    vte_integral_sky_scale: f32,

    /// Extra emissive term added to hit samples in fused-integral tweak mode.
    #[arg(long, default_value_t = 0.025)]
    vte_integral_hit_emissive_boost: f32,

    /// Enable fused-integral logarithmic hit-vs-sky merge curve.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    vte_integral_log_merge_tweak: bool,

    /// Curve strength k for log merge: blend = log(1+k*p) / log(1+k).
    #[arg(long, default_value_t = 8.0)]
    vte_integral_log_merge_k: f32,

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

    /// Multiplayer server address (`IP` or `IP:PORT`).
    /// If the port is omitted, port 4000 is used.
    #[arg(long)]
    server: Option<String>,

    /// Display name sent to multiplayer server on connect.
    #[arg(long)]
    player_name: Option<String>,
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

#[derive(Copy, Clone, PartialEq, Eq)]
enum InfoPanelMode {
    Full,
    VectorTable,
    Off,
}

impl InfoPanelMode {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::VectorTable => "vectors",
            Self::Off => "off",
        }
    }

    fn step(self, delta: i32) -> Self {
        let index = match self {
            Self::Full => 0,
            Self::VectorTable => 1,
            Self::Off => 2,
        };
        match (index + delta).rem_euclid(3) {
            0 => Self::Full,
            1 => Self::VectorTable,
            _ => Self::Off,
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
    let vte_sweep_include_no_entities = env_flag_enabled(VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV);
    let initial_vte_entities_enabled = args.vte_entities;
    let initial_vte_y_slice_lookup_cache_enabled = args.vte_y_slice_lookup_cache;
    let initial_vte_integral_sky_emissive_enabled = args.vte_integral_sky_emissive_tweak;
    let initial_vte_integral_log_merge_enabled = args.vte_integral_log_merge_tweak;
    let initial_vte_integral_sky_scale = args.vte_integral_sky_scale.max(0.0);
    let initial_vte_integral_hit_emissive_boost = args.vte_integral_hit_emissive_boost.max(0.0);
    let initial_vte_integral_log_merge_k = args.vte_integral_log_merge_k.max(0.0);
    let initial_vte_max_trace_steps = args.vte_max_trace_steps.max(1);
    let initial_vte_max_trace_distance = args.vte_max_trace_distance.max(1.0);

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

    let multiplayer = if let Some(server) = args.server.as_ref() {
        let server_addr = normalize_server_addr(server);
        let player_name = args
            .player_name
            .clone()
            .unwrap_or_else(default_multiplayer_player_name);
        match MultiplayerClient::connect(server_addr.clone(), player_name.clone()) {
            Ok(client) => {
                eprintln!(
                    "Connecting to multiplayer server {} as '{}'",
                    client.server_addr(),
                    player_name
                );
                Some(client)
            }
            Err(error) => {
                eprintln!(
                    "Failed to connect multiplayer server {}: {}",
                    server_addr, error
                );
                None
            }
        }
    } else {
        None
    };

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
        control_scheme: ControlScheme::LookTransport,
        scroll_cycle_pair: RotationPair::Standard,
        move_speed: 5.0,
        info_panel_mode: InfoPanelMode::VectorTable,
        focal_length_xy: 1.0,
        focal_length_zw: 1.0,
        place_material: BLOCK_EDIT_PLACE_MATERIAL_DEFAULT,
        world_file,
        vte_reference_compare_enabled,
        vte_reference_mismatch_only_enabled,
        vte_compare_slice_only_enabled,
        vte_entities_enabled: initial_vte_entities_enabled,
        vte_y_slice_lookup_cache_enabled: initial_vte_y_slice_lookup_cache_enabled,
        vte_integral_sky_emissive_enabled: initial_vte_integral_sky_emissive_enabled,
        vte_integral_sky_scale: initial_vte_integral_sky_scale,
        vte_integral_hit_emissive_boost: initial_vte_integral_hit_emissive_boost,
        vte_integral_log_merge_enabled: initial_vte_integral_log_merge_enabled,
        vte_integral_log_merge_k: initial_vte_integral_log_merge_k,
        vte_max_trace_steps: initial_vte_max_trace_steps,
        vte_max_trace_distance: initial_vte_max_trace_distance,
        vte_sweep_include_no_entities,
        vte_sweep_state: None,
        vte_sweep_run_id: 0,
        menu_open: false,
        menu_selection: 0,
        multiplayer,
        multiplayer_self_id: None,
        next_multiplayer_edit_id: 1,
        pending_voxel_edits: Vec::new(),
        remote_players: HashMap::new(),
        last_multiplayer_player_update: Instant::now(),
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
    if !app.vte_entities_enabled {
        eprintln!("VTE tetra-entity pipeline disabled via --vte-entities=false");
    }
    if !app.args.vte_y_slice_fastpath {
        eprintln!(
            "Ignoring deprecated --vte-y-slice-fastpath=false; y-slice fastpath is always enabled."
        );
    }
    if !app.args.vte_chunk_solid_clip {
        eprintln!(
            "Ignoring deprecated --vte-chunk-solid-clip=false; chunk solid clipping is always enabled."
        );
    }
    if !app.vte_y_slice_lookup_cache_enabled {
        eprintln!("VTE y-slice lookup cache disabled via --vte-y-slice-lookup-cache=false");
    }
    if app.vte_integral_sky_emissive_enabled {
        eprintln!(
            "VTE integral sky/emissive tweak enabled via --vte-integral-sky-emissive-tweak=true (sky_scale={:.3}, hit_emissive_boost={:.3})",
            app.vte_integral_sky_scale,
            app.vte_integral_hit_emissive_boost
        );
    }
    if app.vte_integral_log_merge_enabled {
        eprintln!(
            "VTE integral log-merge tweak enabled via --vte-integral-log-merge-tweak=true (k={:.3})",
            app.vte_integral_log_merge_k
        );
    }
    if app.vte_sweep_include_no_entities {
        eprintln!(
            "VTE sweep: extended mode enabled via {} (includes no-entities profiles).",
            VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV
        );
    } else {
        eprintln!("VTE sweep: entities mode (default, entities stay enabled).");
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
    info_panel_mode: InfoPanelMode,
    focal_length_xy: f32,
    focal_length_zw: f32,
    place_material: u8,
    world_file: PathBuf,
    vte_reference_compare_enabled: bool,
    vte_reference_mismatch_only_enabled: bool,
    vte_compare_slice_only_enabled: bool,
    vte_entities_enabled: bool,
    vte_y_slice_lookup_cache_enabled: bool,
    vte_integral_sky_emissive_enabled: bool,
    vte_integral_sky_scale: f32,
    vte_integral_hit_emissive_boost: f32,
    vte_integral_log_merge_enabled: bool,
    vte_integral_log_merge_k: f32,
    vte_max_trace_steps: u32,
    vte_max_trace_distance: f32,
    vte_sweep_include_no_entities: bool,
    vte_sweep_state: Option<VteSweepState>,
    vte_sweep_run_id: u32,
    menu_open: bool,
    menu_selection: usize,
    multiplayer: Option<MultiplayerClient>,
    multiplayer_self_id: Option<u64>,
    next_multiplayer_edit_id: u64,
    pending_voxel_edits: Vec<PendingVoxelEdit>,
    remote_players: HashMap<u64, RemotePlayerState>,
    last_multiplayer_player_update: Instant,
}

#[derive(Copy, Clone)]
struct VteSweepState {
    run_id: u32,
    profile_index: usize,
    frames_remaining: usize,
    previous_entities: bool,
    previous_y_slice_lookup_cache: bool,
}

#[derive(Clone)]
struct RemotePlayerState {
    name: String,
    // Latest network-authoritative state from the server.
    position: [f32; 4],
    look: [f32; 4],
    last_update_ms: u64,
    // Render-smoothed state.
    render_position: [f32; 4],
    render_look: [f32; 4],
    velocity: [f32; 4],
    last_received_at: Instant,
}

#[derive(Clone)]
struct PendingVoxelEdit {
    client_edit_id: u64,
    position: [i32; 4],
    material: u8,
    created_at: Instant,
}

#[derive(Copy, Clone)]
enum PauseMenuItem {
    Resume,
    InfoPanel,
    ControlScheme,
    FocalLengthXy,
    FocalLengthZw,
    VteMaxTraceSteps,
    VteMaxTraceDistance,
    VteEntities,
    VteYSliceLookupCache,
    IntegralSkyEmissive,
    IntegralSkyScale,
    IntegralHitEmissiveBoost,
    IntegralLogMerge,
    IntegralLogMergeK,
    SaveWorld,
    LoadWorld,
    Quit,
}

const PAUSE_MENU_ITEMS: [PauseMenuItem; 17] = [
    PauseMenuItem::Resume,
    PauseMenuItem::InfoPanel,
    PauseMenuItem::ControlScheme,
    PauseMenuItem::FocalLengthXy,
    PauseMenuItem::FocalLengthZw,
    PauseMenuItem::VteMaxTraceSteps,
    PauseMenuItem::VteMaxTraceDistance,
    PauseMenuItem::VteEntities,
    PauseMenuItem::VteYSliceLookupCache,
    PauseMenuItem::IntegralSkyEmissive,
    PauseMenuItem::IntegralSkyScale,
    PauseMenuItem::IntegralHitEmissiveBoost,
    PauseMenuItem::IntegralLogMerge,
    PauseMenuItem::IntegralLogMergeK,
    PauseMenuItem::SaveWorld,
    PauseMenuItem::LoadWorld,
    PauseMenuItem::Quit,
];

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

fn normalize_server_addr(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return format!("127.0.0.1:{MULTIPLAYER_DEFAULT_PORT}");
    }
    if trimmed.contains(':') || trimmed.starts_with('[') {
        trimmed.to_string()
    } else {
        format!("{trimmed}:{MULTIPLAYER_DEFAULT_PORT}")
    }
}

fn default_multiplayer_player_name() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .ok()
        .map(|v| {
            let trimmed = v.trim();
            if trimmed.is_empty() {
                "player".to_string()
            } else {
                trimmed.to_string()
            }
        })
        .unwrap_or_else(|| "player".to_string())
}

fn project_world_point_to_ndc_with_depth(
    view_matrix: &ndarray::Array2<f32>,
    world_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<([f32; 2], f32)> {
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
        Some(([x, y], depth))
    } else {
        None
    }
}

fn project_world_point_to_ndc(
    view_matrix: &ndarray::Array2<f32>,
    world_point: [f32; 4],
    focal_length_xy: f32,
    aspect: f32,
) -> Option<[f32; 2]> {
    project_world_point_to_ndc_with_depth(view_matrix, world_point, focal_length_xy, aspect)
        .map(|(ndc, _)| ndc)
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

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn distance4(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    let d3 = a[3] - b[3];
    (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt()
}

fn normalize4_with_fallback(v: [f32; 4], fallback: [f32; 4]) -> [f32; 4] {
    let len_sq = dot4(v, v);
    if len_sq <= 1e-8 {
        return fallback;
    }
    let inv_len = len_sq.sqrt().recip();
    [
        v[0] * inv_len,
        v[1] * inv_len,
        v[2] * inv_len,
        v[3] * inv_len,
    ]
}

fn orthonormal_basis_from_forward(forward: [f32; 4]) -> [[f32; 4]; 4] {
    let forward = normalize4_with_fallback(forward, [0.0, 0.0, 1.0, 0.0]);
    let mut ortho = [[0.0; 4]; 4];
    ortho[0] = forward;
    let mut count = 1usize;

    let candidates = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ];

    for candidate in candidates {
        if count >= 4 {
            break;
        }
        let mut v = candidate;
        for basis_idx in 0..count {
            let projection = dot4(v, ortho[basis_idx]);
            for axis in 0..4 {
                v[axis] -= projection * ortho[basis_idx][axis];
            }
        }

        let len_sq = dot4(v, v);
        if len_sq <= 1e-6 {
            continue;
        }

        let inv_len = len_sq.sqrt().recip();
        for axis in 0..4 {
            v[axis] *= inv_len;
        }
        ortho[count] = v;
        count += 1;
    }

    if count < 4 {
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    // Return basis as [right, up, forward, side].
    // `ortho[1]` is world-up for planar forward inputs, so keep it in the up slot.
    [ortho[2], ortho[1], ortho[0], ortho[3]]
}

fn offset_point_along_basis(
    origin: [f32; 4],
    basis: &[[f32; 4]; 4],
    local_offset: [f32; 4],
) -> [f32; 4] {
    let mut p = origin;
    for row in 0..4 {
        p[row] += basis[0][row] * local_offset[0]
            + basis[1][row] * local_offset[1]
            + basis[2][row] * local_offset[2]
            + basis[3][row] * local_offset[3];
    }
    p
}

fn build_centered_model_instance(
    center: [f32; 4],
    basis: &[[f32; 4]; 4],
    axis_scale: [f32; 4],
    cell_material_ids: [u32; 8],
) -> common::ModelInstance {
    let mut model_transform = common::MatN::<5>::identity();
    for row in 0..4 {
        model_transform[[row, 0]] = basis[0][row] * axis_scale[0];
        model_transform[[row, 1]] = basis[1][row] * axis_scale[1];
        model_transform[[row, 2]] = basis[2][row] * axis_scale[2];
        model_transform[[row, 3]] = basis[3][row] * axis_scale[3];

        let center_offset = 0.5
            * (model_transform[[row, 0]]
                + model_transform[[row, 1]]
                + model_transform[[row, 2]]
                + model_transform[[row, 3]]);
        model_transform[[row, 4]] = center[row] - center_offset;
    }

    common::ModelInstance {
        model_transform,
        cell_material_ids,
    }
}

fn avatar_cell_mask(cell_indices: &[usize]) -> [u32; 8] {
    let mut ids = [0u32; 8];
    for &cell in cell_indices {
        if cell < ids.len() {
            ids[cell] = AVATAR_MATERIAL_ID;
        }
    }
    ids
}

fn stable_name_hash(name: &str) -> u32 {
    let mut hash = 0x811C_9DC5u32;
    for b in name.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
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

fn build_remote_player_avatar_instances(
    client_id: u64,
    name_hash: u32,
    position: [f32; 4],
    look: [f32; 4],
    time_s: f32,
) -> Vec<common::ModelInstance> {
    let mut instances = Vec::with_capacity(REMOTE_AVATAR_PART_COUNT_ESTIMATE);

    let look_phase = (name_hash as f32) * 0.0078125;
    let id_phase = ((client_id as f32) * 0.173205 + look_phase).rem_euclid(std::f32::consts::TAU);
    let idle_bob = (time_s * 1.9 + id_phase).sin() * 0.04;

    let planar_forward =
        normalize4_with_fallback([look[0], 0.0, look[2], look[3]], [0.0, 0.0, 1.0, 0.0]);
    let full_forward = normalize4_with_fallback(look, planar_forward);
    let mut avatar_basis = orthonormal_basis_from_forward(planar_forward);
    rotate_basis_plane(&mut avatar_basis, 0, 3, 0.18 * id_phase.sin());

    let head_center = offset_point_along_basis(
        position,
        &avatar_basis,
        [0.0, 0.05 + idle_bob * 0.4, 0.0, 0.0],
    );
    let mut head_basis = avatar_basis;
    rotate_basis_plane(&mut head_basis, 0, 2, time_s * 2.6 + id_phase * 0.6);
    rotate_basis_plane(&mut head_basis, 1, 3, time_s * 1.7 + id_phase);
    instances.push(build_centered_model_instance(
        head_center,
        &head_basis,
        [
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
            0.30 * AVATAR_THICKNESS_SCALE,
        ],
        [AVATAR_MATERIAL_ID; 8],
    ));

    let body_parts: [([f32; 4], [f32; 4], &[usize]); 6] = [
        (
            [0.0, -0.34 * PLAYER_HEIGHT + idle_bob, 0.03, 0.0],
            [
                0.22 * AVATAR_THICKNESS_SCALE,
                0.20 * AVATAR_THICKNESS_SCALE,
                0.18 * AVATAR_THICKNESS_SCALE,
                0.16 * AVATAR_THICKNESS_SCALE,
            ],
            &[0, 6],
        ),
        (
            [-0.20, -0.40 * PLAYER_HEIGHT + idle_bob, 0.0, 0.10],
            [
                0.13 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
            ],
            &[2],
        ),
        (
            [0.20, -0.40 * PLAYER_HEIGHT + idle_bob, 0.0, -0.10],
            [
                0.13 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
                0.12 * AVATAR_THICKNESS_SCALE,
            ],
            &[5],
        ),
        (
            [-0.12, -0.72 * PLAYER_HEIGHT + idle_bob, 0.02, 0.08],
            [
                0.14 * AVATAR_THICKNESS_SCALE,
                0.19 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
            ],
            &[7],
        ),
        (
            [0.12, -0.72 * PLAYER_HEIGHT + idle_bob, -0.02, -0.08],
            [
                0.14 * AVATAR_THICKNESS_SCALE,
                0.19 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
                0.11 * AVATAR_THICKNESS_SCALE,
            ],
            &[1],
        ),
        (
            [0.0, -0.58 * PLAYER_HEIGHT + idle_bob, -0.02, 0.0],
            [
                0.17 * AVATAR_THICKNESS_SCALE,
                0.16 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
                0.14 * AVATAR_THICKNESS_SCALE,
            ],
            &[3, 4],
        ),
    ];

    for (part_index, (offset, scales, cells)) in body_parts.iter().enumerate() {
        let mut part_basis = avatar_basis;
        let part_phase = id_phase + part_index as f32 * 0.7;
        rotate_basis_plane(&mut part_basis, 0, 2, 0.22 * part_phase.sin());
        rotate_basis_plane(&mut part_basis, 2, 3, 0.17 * part_phase.cos());
        instances.push(build_centered_model_instance(
            offset_point_along_basis(position, &avatar_basis, *offset),
            &part_basis,
            *scales,
            avatar_cell_mask(cells),
        ));
    }

    for fragment_idx in 0..AVATAR_FORWARD_FRAGMENT_COUNT {
        let fragment_phase = id_phase + fragment_idx as f32 * 0.83;
        let fragment_distance = (0.54 + fragment_idx as f32 * 0.30) * AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE;
        let swirl = time_s * 3.1 + fragment_phase;
        let lateral_offset = [
            0.10 * swirl.cos(),
            -0.04 + 0.05 * (swirl * 0.9).sin(),
            0.10 * (swirl + 1.1).sin(),
        ];
        let forward_distance =
            fragment_distance + 0.06 * AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE * (swirl * 1.2).sin();
        let mut fragment_center = head_center;
        for axis in 0..4 {
            fragment_center[axis] += avatar_basis[0][axis] * lateral_offset[0]
                + avatar_basis[1][axis] * lateral_offset[1]
                + avatar_basis[3][axis] * lateral_offset[2]
                + full_forward[axis] * forward_distance;
        }

        let mut fragment_basis = orthonormal_basis_from_forward(full_forward);
        rotate_basis_plane(&mut fragment_basis, 0, 3, time_s * 1.7 + fragment_phase);
        rotate_basis_plane(&mut fragment_basis, 1, 2, time_s * 2.3 + fragment_phase * 0.7);

        let fragment_cells: &[usize] = match fragment_idx % 4 {
            0 => &[0],
            1 => &[2],
            2 => &[5],
            _ => &[7],
        };

        let depth_scale = 1.0 - fragment_idx as f32 * 0.12;
        instances.push(build_centered_model_instance(
            fragment_center,
            &fragment_basis,
            [
                0.10 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.09 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.13 * AVATAR_THICKNESS_SCALE * depth_scale,
                0.09 * AVATAR_THICKNESS_SCALE * depth_scale,
            ],
            avatar_cell_mask(fragment_cells),
        ));
    }

    instances
}

impl App {
    fn vte_sweep_profiles(&self) -> &'static [VteRuntimeProfile] {
        if self.vte_sweep_include_no_entities {
            &VTE_SWEEP_PROFILES_EXTENDED
        } else {
            &VTE_SWEEP_PROFILES_ENTITIES
        }
    }

    fn vte_sweep_mode_label(&self) -> &'static str {
        if self.vte_sweep_include_no_entities {
            "extended"
        } else {
            "entities"
        }
    }

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

    fn cycle_control_scheme(&mut self) {
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

    fn cycle_control_scheme_by(&mut self, delta: i32) {
        if delta >= 0 {
            self.cycle_control_scheme();
            return;
        }
        // ControlScheme currently has 5 variants, so 4 forward steps == one backward step.
        for _ in 0..4 {
            self.cycle_control_scheme();
        }
    }

    fn step_f32(value: f32, delta: i32, step: f32, min: f32, max: f32) -> f32 {
        let signed_step = if delta > 0 {
            step
        } else if delta < 0 {
            -step
        } else {
            0.0
        };
        (value + signed_step).clamp(min, max)
    }

    fn adjust_info_panel_mode(&mut self, delta: i32) {
        self.info_panel_mode = self.info_panel_mode.step(delta);
    }

    fn adjust_focal_length_xy(&mut self, delta: i32) {
        self.focal_length_xy = Self::step_f32(
            self.focal_length_xy,
            delta,
            FOCAL_LENGTH_STEP,
            FOCAL_LENGTH_MIN,
            FOCAL_LENGTH_MAX,
        );
    }

    fn adjust_focal_length_zw(&mut self, delta: i32) {
        self.focal_length_zw = Self::step_f32(
            self.focal_length_zw,
            delta,
            FOCAL_LENGTH_STEP,
            FOCAL_LENGTH_MIN,
            FOCAL_LENGTH_MAX,
        );
    }

    fn adjust_vte_max_trace_steps(&mut self, delta: i32) {
        if delta > 0 {
            self.vte_max_trace_steps = (self.vte_max_trace_steps.saturating_mul(2))
                .clamp(VTE_TRACE_STEPS_MIN, VTE_TRACE_STEPS_MAX);
        } else if delta < 0 {
            self.vte_max_trace_steps =
                (self.vte_max_trace_steps / 2).clamp(VTE_TRACE_STEPS_MIN, VTE_TRACE_STEPS_MAX);
        }
    }

    fn adjust_vte_max_trace_distance(&mut self, delta: i32) {
        self.vte_max_trace_distance = Self::step_f32(
            self.vte_max_trace_distance,
            delta,
            VTE_TRACE_DISTANCE_STEP,
            VTE_TRACE_DISTANCE_MIN,
            VTE_TRACE_DISTANCE_MAX,
        );
    }

    fn adjust_vte_integral_sky_scale(&mut self, delta: i32) {
        self.vte_integral_sky_scale = Self::step_f32(
            self.vte_integral_sky_scale,
            delta,
            VTE_INTEGRAL_SKY_SCALE_STEP,
            VTE_INTEGRAL_SKY_SCALE_MIN,
            VTE_INTEGRAL_SKY_SCALE_MAX,
        );
    }

    fn adjust_vte_integral_hit_emissive_boost(&mut self, delta: i32) {
        self.vte_integral_hit_emissive_boost = Self::step_f32(
            self.vte_integral_hit_emissive_boost,
            delta,
            VTE_INTEGRAL_HIT_EMISSIVE_STEP,
            VTE_INTEGRAL_HIT_EMISSIVE_MIN,
            VTE_INTEGRAL_HIT_EMISSIVE_MAX,
        );
    }

    fn adjust_vte_integral_log_merge_k(&mut self, delta: i32) {
        self.vte_integral_log_merge_k = Self::step_f32(
            self.vte_integral_log_merge_k,
            delta,
            VTE_INTEGRAL_LOG_MERGE_K_STEP,
            VTE_INTEGRAL_LOG_MERGE_K_MIN,
            VTE_INTEGRAL_LOG_MERGE_K_MAX,
        );
    }

    fn save_world(&mut self) {
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

    fn load_world(&mut self) {
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

    fn poll_multiplayer_events(&mut self) {
        loop {
            let event = match self
                .multiplayer
                .as_ref()
                .and_then(|client| client.try_recv())
            {
                Some(event) => event,
                None => break,
            };

            match event {
                MultiplayerEvent::Message(message) => self.handle_multiplayer_message(message),
                MultiplayerEvent::Disconnected(reason) => {
                    eprintln!("Multiplayer disconnected: {reason}");
                    self.multiplayer = None;
                    self.multiplayer_self_id = None;
                    self.next_multiplayer_edit_id = 1;
                    self.pending_voxel_edits.clear();
                    self.remote_players.clear();
                    break;
                }
            }
        }
    }

    fn upsert_remote_player_snapshot(
        &mut self,
        player: multiplayer::PlayerSnapshot,
        received_at: Instant,
    ) {
        let normalized_look = normalize4_with_fallback(player.look, [0.0, 0.0, 1.0, 0.0]);
        if let Some(existing) = self.remote_players.get_mut(&player.client_id) {
            let previous_position = existing.position;
            let previous_update_ms = existing.last_update_ms;

            existing.name = player.name;
            existing.position = player.position;
            existing.look = normalized_look;
            existing.last_update_ms = player.last_update_ms;
            existing.last_received_at = received_at;

            let update_delta_ms = player.last_update_ms.saturating_sub(previous_update_ms);
            if update_delta_ms > 0 {
                let dt_s = (update_delta_ms as f32) * 0.001;
                if dt_s > 1e-4 {
                    let mut velocity = [0.0f32; 4];
                    for axis in 0..4 {
                        velocity[axis] = (player.position[axis] - previous_position[axis]) / dt_s;
                    }
                    let speed = dot4(velocity, velocity).sqrt();
                    if speed.is_finite() && speed > REMOTE_PLAYER_MAX_PREDICTED_SPEED {
                        let clamp = REMOTE_PLAYER_MAX_PREDICTED_SPEED / speed;
                        for v in &mut velocity {
                            *v *= clamp;
                        }
                    }
                    existing.velocity = velocity;
                }
            } else {
                for v in &mut existing.velocity {
                    *v *= 0.85;
                }
            }

            if distance4(previous_position, player.position) > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE {
                existing.render_position = existing.position;
                existing.render_look = existing.look;
                existing.velocity = [0.0; 4];
            }
            return;
        }

        self.remote_players.insert(
            player.client_id,
            RemotePlayerState {
                name: player.name,
                position: player.position,
                look: normalized_look,
                last_update_ms: player.last_update_ms,
                render_position: player.position,
                render_look: normalized_look,
                velocity: [0.0; 4],
                last_received_at: received_at,
            },
        );
    }

    fn smooth_remote_players(&mut self, dt: f32, now: Instant) {
        let dt = dt.clamp(0.0, 0.25);
        if dt <= 0.0 {
            return;
        }
        let pos_alpha = 1.0 - (-REMOTE_PLAYER_POSITION_SMOOTH_HZ * dt).exp();
        let look_alpha = 1.0 - (-REMOTE_PLAYER_LOOK_SMOOTH_HZ * dt).exp();
        for player in self.remote_players.values_mut() {
            let network_age = now.duration_since(player.last_received_at).as_secs_f32();
            let predict_time = (network_age + REMOTE_PLAYER_PREDICTION_LEAD_S)
                .clamp(0.0, REMOTE_PLAYER_MAX_PREDICTION_S);

            let mut predicted_position = player.position;
            for axis in 0..4 {
                predicted_position[axis] += player.velocity[axis] * predict_time;
            }

            if distance4(player.render_position, predicted_position)
                > REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE
            {
                player.render_position = predicted_position;
            } else {
                player.render_position = lerp4(player.render_position, predicted_position, pos_alpha);
            }

            player.render_look = normalize4_with_fallback(
                lerp4(player.render_look, player.look, look_alpha),
                player.look,
            );
        }
    }

    fn acknowledge_pending_voxel_edit(
        &mut self,
        client_edit_id: Option<u64>,
        position: [i32; 4],
        material: u8,
    ) {
        let index = if let Some(edit_id) = client_edit_id {
            self.pending_voxel_edits
                .iter()
                .position(|entry| entry.client_edit_id == edit_id)
        } else {
            self.pending_voxel_edits
                .iter()
                .position(|entry| entry.position == position && entry.material == material)
                .or_else(|| {
                    self.pending_voxel_edits
                        .iter()
                        .position(|entry| entry.position == position)
                })
        };
        if let Some(index) = index {
            let _ = self.pending_voxel_edits.remove(index);
        }
    }

    fn reapply_pending_voxel_edits(&mut self, now: Instant) {
        self.pending_voxel_edits.retain(|entry| {
            now.saturating_duration_since(entry.created_at) <= MULTIPLAYER_PENDING_EDIT_TIMEOUT
        });
        for entry in &self.pending_voxel_edits {
            self.scene.world.set_voxel(
                entry.position[0],
                entry.position[1],
                entry.position[2],
                entry.position[3],
                voxel::VoxelType(entry.material),
            );
        }
    }

    fn handle_multiplayer_message(&mut self, message: multiplayer::ServerMessage) {
        let received_at = Instant::now();
        match message {
            multiplayer::ServerMessage::Welcome {
                client_id,
                tick_hz,
                world,
                ..
            } => {
                self.multiplayer_self_id = Some(client_id);
                self.next_multiplayer_edit_id = 1;
                self.pending_voxel_edits.clear();
                eprintln!(
                    "Multiplayer connected: client_id={} world_rev={} chunks={} server_tick_hz={:.2}",
                    client_id, world.revision, world.non_empty_chunks, tick_hz
                );
                if let Some(client) = self.multiplayer.as_ref() {
                    client.send(MultiplayerClientMessage::RequestWorldSnapshot);
                }
            }
            multiplayer::ServerMessage::Error { message } => {
                eprintln!("Multiplayer server error: {message}");
            }
            multiplayer::ServerMessage::PlayerJoined { player } => {
                if Some(player.client_id) == self.multiplayer_self_id {
                    return;
                }
                self.upsert_remote_player_snapshot(player, received_at);
            }
            multiplayer::ServerMessage::PlayerLeft { client_id } => {
                self.remote_players.remove(&client_id);
            }
            multiplayer::ServerMessage::PlayerPositions { players, .. } => {
                let mut seen = Vec::with_capacity(players.len());
                for player in players {
                    if Some(player.client_id) == self.multiplayer_self_id {
                        continue;
                    }
                    seen.push(player.client_id);
                    self.upsert_remote_player_snapshot(player, received_at);
                }
                self.remote_players
                    .retain(|client_id, _| seen.contains(client_id));
            }
            multiplayer::ServerMessage::WorldVoxelSet {
                position,
                material,
                source_client_id,
                client_edit_id,
                ..
            } => {
                self.scene.world.set_voxel(
                    position[0],
                    position[1],
                    position[2],
                    position[3],
                    voxel::VoxelType(material),
                );
                if source_client_id == self.multiplayer_self_id {
                    self.acknowledge_pending_voxel_edit(client_edit_id, position, material);
                }
            }
            multiplayer::ServerMessage::WorldSnapshot { world } => {
                let decoded =
                    base64::engine::general_purpose::STANDARD.decode(world.bytes_base64.as_bytes());
                let Ok(bytes) = decoded else {
                    eprintln!("Multiplayer world snapshot decode failed");
                    return;
                };
                let mut cursor = &bytes[..];
                match voxel::io::load_world(&mut cursor) {
                    Ok(new_world) => {
                        self.scene.replace_world(new_world);
                        eprintln!(
                            "Applied multiplayer world snapshot rev={} chunks={}",
                            world.revision, world.non_empty_chunks
                        );
                    }
                    Err(error) => {
                        eprintln!("Failed to load multiplayer world snapshot: {error}");
                    }
                }
            }
            multiplayer::ServerMessage::Pong { .. } => {}
        }
    }

    fn send_multiplayer_player_update(&mut self, now: Instant, look_dir: [f32; 4]) {
        if now.duration_since(self.last_multiplayer_player_update)
            < MULTIPLAYER_PLAYER_UPDATE_INTERVAL
        {
            return;
        }
        self.last_multiplayer_player_update = now;
        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::UpdatePlayer {
                position: self.camera.position,
                look: look_dir,
            });
        }
    }

    fn send_multiplayer_voxel_update(&mut self, now: Instant, position: [i32; 4], material: u8) {
        if self.multiplayer.is_none() {
            return;
        }

        let client_edit_id = self.next_multiplayer_edit_id;
        self.next_multiplayer_edit_id = self.next_multiplayer_edit_id.wrapping_add(1).max(1);
        self.pending_voxel_edits.push(PendingVoxelEdit {
            client_edit_id,
            position,
            material,
            created_at: now,
        });

        if self.pending_voxel_edits.len() > MULTIPLAYER_PENDING_EDIT_MAX {
            let overflow = self.pending_voxel_edits.len() - MULTIPLAYER_PENDING_EDIT_MAX;
            self.pending_voxel_edits.drain(0..overflow);
        }

        if let Some(client) = self.multiplayer.as_ref() {
            client.send(MultiplayerClientMessage::SetVoxel {
                position,
                material,
                client_edit_id: Some(client_edit_id),
            });
        }
    }

    fn remote_player_instances(&self, time_s: f32) -> Vec<common::ModelInstance> {
        let mut ids: Vec<u64> = self.remote_players.keys().copied().collect();
        ids.sort_unstable();
        let mut instances = Vec::with_capacity(ids.len() * REMOTE_AVATAR_PART_COUNT_ESTIMATE);
        for client_id in ids {
            if let Some(player) = self.remote_players.get(&client_id) {
                instances.extend(build_remote_player_avatar_instances(
                    client_id,
                    stable_name_hash(&player.name),
                    player.render_position,
                    player.render_look,
                    time_s,
                ));
            }
        }
        instances
    }

    fn remote_player_tags(
        &self,
        view_matrix: &ndarray::Array2<f32>,
        look_dir: [f32; 4],
        focal_length_xy: f32,
        aspect: f32,
    ) -> Vec<HudPlayerTag> {
        let forward = normalize4_with_fallback(look_dir, [0.0, 0.0, 1.0, 0.0]);
        let mut ids: Vec<u64> = self.remote_players.keys().copied().collect();
        ids.sort_unstable();

        let mut tags = Vec::with_capacity(ids.len().min(REMOTE_PLAYER_TAG_MAX_COUNT));
        for client_id in ids {
            if tags.len() >= REMOTE_PLAYER_TAG_MAX_COUNT {
                break;
            }
            let Some(player) = self.remote_players.get(&client_id) else {
                continue;
            };

            let anchor = [
                player.render_position[0],
                player.render_position[1] + 0.48,
                player.render_position[2],
                player.render_position[3],
            ];
            let to_anchor = [
                anchor[0] - self.camera.position[0],
                anchor[1] - self.camera.position[1],
                anchor[2] - self.camera.position[2],
                anchor[3] - self.camera.position[3],
            ];
            let distance_sq = dot4(to_anchor, to_anchor);
            if !distance_sq.is_finite() || distance_sq <= 1e-6 {
                continue;
            }
            let distance = distance_sq.sqrt();
            let inv_distance = distance.recip();
            let dir = [
                to_anchor[0] * inv_distance,
                to_anchor[1] * inv_distance,
                to_anchor[2] * inv_distance,
                to_anchor[3] * inv_distance,
            ];

            if dot4(dir, forward) < REMOTE_PLAYER_TAG_FOV_DOT_MIN {
                continue;
            }

            let Some((ndc, _depth)) =
                project_world_point_to_ndc_with_depth(view_matrix, anchor, focal_length_xy, aspect)
            else {
                continue;
            };
            if ndc[0].abs() > 1.10 || ndc[1].abs() > 1.10 {
                continue;
            }

            let scale = (3.2 / (distance + 1.0)).clamp(0.55, 1.45);
            let bg_alpha = (0.82 - distance * 0.05).clamp(0.35, 0.82);
            let text = if player.name.trim().is_empty() {
                format!("player-{client_id}")
            } else {
                player.name.clone()
            };

            tags.push(HudPlayerTag {
                text,
                anchor_ndc: ndc,
                scale,
                bg_alpha,
            });
        }

        tags
    }

    fn toggle_vte_entities(&mut self) {
        if let Some(state) = self.vte_sweep_state {
            eprintln!(
                "[VTE sweep #{}] ignoring manual entities toggle while sweep is active.",
                state.run_id
            );
            return;
        }
        self.vte_entities_enabled = !self.vte_entities_enabled;
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "VTE runtime entities: {}",
            if self.vte_entities_enabled {
                "on"
            } else {
                "off"
            }
        );
    }

    fn toggle_vte_y_slice_lookup_cache(&mut self) {
        if let Some(state) = self.vte_sweep_state {
            eprintln!(
                "[VTE sweep #{}] ignoring manual y-slice lookup cache toggle while sweep is active.",
                state.run_id
            );
            return;
        }
        self.vte_y_slice_lookup_cache_enabled = !self.vte_y_slice_lookup_cache_enabled;
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "VTE runtime y-slice lookup cache: {}",
            if self.vte_y_slice_lookup_cache_enabled {
                "on"
            } else {
                "off"
            }
        );
    }

    fn toggle_vte_integral_sky_emissive(&mut self) {
        self.vte_integral_sky_emissive_enabled = !self.vte_integral_sky_emissive_enabled;
        eprintln!(
            "VTE fused integral sky/emissive tweak: {} (sky_scale={:.3}, hit_emissive_boost={:.3})",
            if self.vte_integral_sky_emissive_enabled {
                "on"
            } else {
                "off"
            },
            self.vte_integral_sky_scale,
            self.vte_integral_hit_emissive_boost,
        );
    }

    fn toggle_vte_integral_log_merge(&mut self) {
        self.vte_integral_log_merge_enabled = !self.vte_integral_log_merge_enabled;
        eprintln!(
            "VTE fused integral log merge tweak: {} (k={:.3})",
            if self.vte_integral_log_merge_enabled {
                "on"
            } else {
                "off"
            },
            self.vte_integral_log_merge_k,
        );
    }

    fn drain_gameplay_inputs_while_menu_open(&mut self) {
        self.input.take_scheme_cycle();
        self.input.take_reset_orientation();
        self.input.take_vte_sweep();
        self.input.take_vte_entities_toggle();
        self.input.take_vte_y_slice_lookup_cache_toggle();
        self.input.take_vte_integral_sky_emissive_toggle();
        self.input.take_vte_integral_log_merge_toggle();
        self.input.take_save_world();
        self.input.take_load_world();
        self.input.take_scroll_steps();
        self.input.take_fly_toggle();
        self.input.take_jump();
        self.input.take_place_material_prev();
        self.input.take_place_material_next();
        self.input.take_place_material_digit();
        self.input.take_remove_block();
        self.input.take_place_block();
        self.input.take_mouse_delta();
    }

    fn move_menu_selection(&mut self, delta: i32) {
        let len = PAUSE_MENU_ITEMS.len() as i32;
        let index = self.menu_selection as i32;
        self.menu_selection = (index + delta).rem_euclid(len) as usize;
    }

    fn selected_menu_item(&self) -> PauseMenuItem {
        PAUSE_MENU_ITEMS[self.menu_selection]
    }

    fn adjust_selected_menu_item(&mut self, delta: i32) {
        match self.selected_menu_item() {
            PauseMenuItem::InfoPanel => self.adjust_info_panel_mode(delta),
            PauseMenuItem::ControlScheme => self.cycle_control_scheme_by(delta),
            PauseMenuItem::FocalLengthXy => self.adjust_focal_length_xy(delta),
            PauseMenuItem::FocalLengthZw => self.adjust_focal_length_zw(delta),
            PauseMenuItem::VteMaxTraceSteps => self.adjust_vte_max_trace_steps(delta),
            PauseMenuItem::VteMaxTraceDistance => self.adjust_vte_max_trace_distance(delta),
            PauseMenuItem::VteEntities => self.toggle_vte_entities(),
            PauseMenuItem::VteYSliceLookupCache => self.toggle_vte_y_slice_lookup_cache(),
            PauseMenuItem::IntegralSkyEmissive => self.toggle_vte_integral_sky_emissive(),
            PauseMenuItem::IntegralSkyScale => self.adjust_vte_integral_sky_scale(delta),
            PauseMenuItem::IntegralHitEmissiveBoost => {
                self.adjust_vte_integral_hit_emissive_boost(delta)
            }
            PauseMenuItem::IntegralLogMerge => self.toggle_vte_integral_log_merge(),
            PauseMenuItem::IntegralLogMergeK => self.adjust_vte_integral_log_merge_k(delta),
            PauseMenuItem::Resume
            | PauseMenuItem::SaveWorld
            | PauseMenuItem::LoadWorld
            | PauseMenuItem::Quit => {}
        }
    }

    fn activate_selected_menu_item(&mut self) {
        match self.selected_menu_item() {
            PauseMenuItem::Resume => {
                self.menu_open = false;
                let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
                if let Some(window) = window {
                    self.grab_mouse(&window);
                }
            }
            PauseMenuItem::SaveWorld => self.save_world(),
            PauseMenuItem::LoadWorld => self.load_world(),
            PauseMenuItem::Quit => self.should_exit_after_render = true,
            _ => self.adjust_selected_menu_item(1),
        }
    }

    fn pause_menu_item_text(&self, item: PauseMenuItem) -> String {
        match item {
            PauseMenuItem::Resume => "Resume".to_string(),
            PauseMenuItem::InfoPanel => {
                format!("Info panel: {}", self.info_panel_mode.label())
            }
            PauseMenuItem::ControlScheme => {
                format!("Control scheme: {}", self.control_scheme.label())
            }
            PauseMenuItem::FocalLengthXy => {
                format!("Focal XY: {:.2}", self.focal_length_xy)
            }
            PauseMenuItem::FocalLengthZw => {
                format!("Focal ZW: {:.2}", self.focal_length_zw)
            }
            PauseMenuItem::VteMaxTraceSteps => {
                format!("VTE max steps: {}", self.vte_max_trace_steps)
            }
            PauseMenuItem::VteMaxTraceDistance => {
                format!("VTE max dist: {:.0}", self.vte_max_trace_distance)
            }
            PauseMenuItem::VteEntities => format!(
                "VTE entities: {}",
                if self.vte_entities_enabled {
                    "on"
                } else {
                    "off"
                }
            ),
            PauseMenuItem::VteYSliceLookupCache => format!(
                "VTE y-cache: {}",
                if self.vte_y_slice_lookup_cache_enabled {
                    "on"
                } else {
                    "off"
                }
            ),
            PauseMenuItem::IntegralSkyEmissive => format!(
                "Integral sky+emi: {}",
                if self.vte_integral_sky_emissive_enabled {
                    "on"
                } else {
                    "off"
                }
            ),
            PauseMenuItem::IntegralSkyScale => {
                format!("Integral sky scale: {:.3}", self.vte_integral_sky_scale)
            }
            PauseMenuItem::IntegralHitEmissiveBoost => format!(
                "Integral hit emissive: {:.3}",
                self.vte_integral_hit_emissive_boost
            ),
            PauseMenuItem::IntegralLogMerge => format!(
                "Integral log merge: {}",
                if self.vte_integral_log_merge_enabled {
                    "on"
                } else {
                    "off"
                }
            ),
            PauseMenuItem::IntegralLogMergeK => {
                format!("Integral log K: {:.3}", self.vte_integral_log_merge_k)
            }
            PauseMenuItem::SaveWorld => "Save world".to_string(),
            PauseMenuItem::LoadWorld => "Load world".to_string(),
            PauseMenuItem::Quit => "Quit".to_string(),
        }
    }

    fn current_info_hud_text(
        &self,
        pair: RotationPair,
        look_dir: [f32; 4],
        edit_reach: f32,
        highlight_mode: EditHighlightModeArg,
        vte_sweep_status: &str,
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
    ) -> Option<String> {
        match self.info_panel_mode {
            InfoPanelMode::Off => None,
            InfoPanelMode::VectorTable => {
                Some(self.vector_table_hud_text(look_dir, target_hit_voxel, target_hit_face))
            }
            InfoPanelMode::Full => Some(self.full_info_hud_text(
                pair,
                look_dir,
                edit_reach,
                highlight_mode,
                vte_sweep_status,
            )),
        }
    }

    fn vector_table_hud_text(
        &self,
        look_dir: [f32; 4],
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
    ) -> String {
        let mut text = String::new();
        text.push_str(
            "vec       |        X |        Y |        Z |        W\n\
             ----------+----------+----------+----------+----------",
        );
        text.push_str(&format!(
            "\n{:<9} | {:+8.2} | {:+8.2} | {:+8.2} | {:+8.2}",
            "pos",
            self.camera.position[0],
            self.camera.position[1],
            self.camera.position[2],
            self.camera.position[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "look", look_dir[0], look_dir[1], look_dir[2], look_dir[3],
        ));
        if let Some(hit) = target_hit_voxel {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "block", hit[0], hit[1], hit[2], hit[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "block", "--", "--", "--", "--"
            ));
        }
        if let Some(face) = target_hit_face {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "face", face[0], face[1], face[2], face[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "face", "--", "--", "--", "--"
            ));
        }
        text
    }

    fn full_info_hud_text(
        &self,
        pair: RotationPair,
        look_dir: [f32; 4],
        edit_reach: f32,
        highlight_mode: EditHighlightModeArg,
        vte_sweep_status: &str,
    ) -> String {
        let inv = self.current_y_inverted();
        let inv = if inv { " Y-INV" } else { "" };
        if self.control_scheme.uses_look_frame() {
            format!(
                "LOOK-FRAME [{}]  spd:{:.1}{}\n\
                 look:{:+.2} {:+.2} {:+.2} {:+.2}\n\
                 lens:xy:{:.2} zw:{:.2} trace:{} dist:{:.0}\n\
                 edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                 mat:[ ]/wheel cycle, 1-0 direct\n\
                 world:F5 save, F9 load\n\
                 vte:F6 ent:{} F7 ycache:{} F8 sweep:{}\n\
                 tweak:F10 sky+emi:{} F11 log:{}",
                self.control_scheme.label(),
                self.move_speed,
                inv,
                look_dir[0],
                look_dir[1],
                look_dir[2],
                look_dir[3],
                self.focal_length_xy,
                self.focal_length_zw,
                self.vte_max_trace_steps,
                self.vte_max_trace_distance,
                self.place_material,
                edit_reach,
                highlight_mode.label(),
                if self.vte_entities_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_y_slice_lookup_cache_enabled {
                    "on"
                } else {
                    "off"
                },
                vte_sweep_status,
                if self.vte_integral_sky_emissive_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_integral_log_merge_enabled {
                    "on"
                } else {
                    "off"
                },
            )
        } else {
            format!(
                "{} [{}]  spd:{:.1}{}\n\
                 yaw:{:+.0} pit:{:+.0} xw:{:+.0} zw:{:+.0} yw:{:+.1}\n\
                 lens:xy:{:.2} zw:{:.2} trace:{} dist:{:.0}\n\
                 edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                 mat:[ ]/wheel cycle, 1-0 direct\n\
                 world:F5 save, F9 load\n\
                 vte:F6 ent:{} F7 ycache:{} F8 sweep:{}\n\
                 tweak:F10 sky+emi:{} F11 log:{}",
                pair.label(),
                self.control_scheme.label(),
                self.move_speed,
                inv,
                self.camera.yaw.to_degrees(),
                self.camera.pitch.to_degrees(),
                self.camera.xw_angle.to_degrees(),
                self.camera.zw_angle.to_degrees(),
                self.camera.yw_deviation.to_degrees(),
                self.focal_length_xy,
                self.focal_length_zw,
                self.vte_max_trace_steps,
                self.vte_max_trace_distance,
                self.place_material,
                edit_reach,
                highlight_mode.label(),
                if self.vte_entities_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_y_slice_lookup_cache_enabled {
                    "on"
                } else {
                    "off"
                },
                vte_sweep_status,
                if self.vte_integral_sky_emissive_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_integral_log_merge_enabled {
                    "on"
                } else {
                    "off"
                },
            )
        }
    }

    fn pause_menu_hud_text(&self) -> String {
        let mut text = String::from("MENU  (Up/Down select, Left/Right tune, Enter action)\n");
        for (i, item) in PAUSE_MENU_ITEMS.iter().enumerate() {
            let prefix = if i == self.menu_selection { ">" } else { " " };
            text.push_str(prefix);
            text.push(' ');
            text.push_str(&self.pause_menu_item_text(*item));
            text.push('\n');
        }
        text.push_str("Click window or Resume to return to capture.");
        text
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

    fn set_vte_runtime_flags(&mut self, entities: bool, y_slice_lookup_cache: bool) {
        self.vte_entities_enabled = entities;
        self.vte_y_slice_lookup_cache_enabled = y_slice_lookup_cache;
    }

    fn toggle_vte_runtime_sweep(&mut self) {
        if self.args.backend.to_render_backend() != RenderBackend::VoxelTraversal {
            eprintln!(
                "VTE runtime sweep requires --backend voxel-traversal (current: {:?}).",
                self.args.backend.to_render_backend()
            );
            return;
        }

        if let Some(state) = self.vte_sweep_state.take() {
            self.set_vte_runtime_flags(
                state.previous_entities,
                state.previous_y_slice_lookup_cache,
            );
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.reset_gpu_profile_window();
            }
            eprintln!(
                "[VTE sweep #{}] cancelled; restored runtime flags (entities={}, y_slice_lookup_cache={}).",
                state.run_id,
                self.vte_entities_enabled,
                self.vte_y_slice_lookup_cache_enabled
            );
            return;
        }

        self.vte_sweep_run_id = self.vte_sweep_run_id.wrapping_add(1);
        if self.vte_sweep_run_id == 0 {
            self.vte_sweep_run_id = 1;
        }
        let run_id = self.vte_sweep_run_id;
        let previous_entities = self.vte_entities_enabled;
        let previous_y_slice_lookup_cache = self.vte_y_slice_lookup_cache_enabled;
        let profiles = self.vte_sweep_profiles();
        let profile_count = profiles.len();
        let first_profile = profiles[0];
        self.set_vte_runtime_flags(first_profile.entities, first_profile.y_slice_lookup_cache);
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        self.vte_sweep_state = Some(VteSweepState {
            run_id,
            profile_index: 0,
            frames_remaining: VTE_SWEEP_SAMPLE_FRAMES,
            previous_entities,
            previous_y_slice_lookup_cache,
        });
        eprintln!(
            "[VTE sweep #{}] started on live scene (mode={}): {} profiles × {} frames/profile.",
            run_id,
            self.vte_sweep_mode_label(),
            profile_count,
            VTE_SWEEP_SAMPLE_FRAMES
        );
        eprintln!(
            "[VTE sweep #{}] profile 1/{} '{}' (entities={}, y_slice_lookup_cache={}).",
            run_id,
            profile_count,
            first_profile.label,
            first_profile.entities,
            first_profile.y_slice_lookup_cache
        );
    }

    fn advance_vte_runtime_sweep_after_frame(&mut self) {
        let Some(mut state) = self.vte_sweep_state else {
            return;
        };
        if state.frames_remaining > 0 {
            state.frames_remaining -= 1;
        }
        if state.frames_remaining > 0 {
            self.vte_sweep_state = Some(state);
            return;
        }

        let profiles = self.vte_sweep_profiles();
        let profile_count = profiles.len();
        let finished_profile = profiles[state.profile_index];
        eprintln!(
            "[VTE sweep #{}] finished profile {}/{} '{}'.",
            state.run_id,
            state.profile_index + 1,
            profile_count,
            finished_profile.label
        );

        if state.profile_index + 1 < profile_count {
            state.profile_index += 1;
            state.frames_remaining = VTE_SWEEP_SAMPLE_FRAMES;
            let next_profile = profiles[state.profile_index];
            self.set_vte_runtime_flags(next_profile.entities, next_profile.y_slice_lookup_cache);
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.reset_gpu_profile_window();
            }
            eprintln!(
                "[VTE sweep #{}] profile {}/{} '{}' (entities={}, y_slice_lookup_cache={}).",
                state.run_id,
                state.profile_index + 1,
                profile_count,
                next_profile.label,
                next_profile.entities,
                next_profile.y_slice_lookup_cache
            );
            self.vte_sweep_state = Some(state);
            return;
        }

        self.set_vte_runtime_flags(state.previous_entities, state.previous_y_slice_lookup_cache);
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "[VTE sweep #{}] completed; restored runtime flags (entities={}, y_slice_lookup_cache={}).",
            state.run_id,
            self.vte_entities_enabled,
            self.vte_y_slice_lookup_cache_enabled
        );
        self.vte_sweep_state = None;
    }

    fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.poll_multiplayer_events();
        self.reapply_pending_voxel_edits(now);
        self.smooth_remote_players(dt, now);

        if self.menu_open {
            if self.input.take_menu_up() {
                self.move_menu_selection(-1);
            }
            if self.input.take_menu_left() {
                self.adjust_selected_menu_item(-1);
            }
            if self.input.take_menu_right() {
                self.adjust_selected_menu_item(1);
            }
            if self.input.take_menu_down() {
                self.move_menu_selection(1);
            }
            if self.input.take_menu_activate() {
                self.activate_selected_menu_item();
            }
            self.drain_gameplay_inputs_while_menu_open();
        } else {
            self.input.take_menu_left();
            self.input.take_menu_right();
            self.input.take_menu_up();
            self.input.take_menu_down();
            self.input.take_menu_activate();

            // Scheme cycle (Tab)
            if self.input.take_scheme_cycle() {
                self.cycle_control_scheme();
            }

            // Reset orientation (R)
            if self.input.take_reset_orientation() {
                self.camera.reset_orientation();
            }

            if self.input.take_vte_sweep() {
                self.toggle_vte_runtime_sweep();
            }
            if self.input.take_vte_entities_toggle() {
                self.toggle_vte_entities();
            }
            if self.input.take_vte_y_slice_lookup_cache_toggle() {
                self.toggle_vte_y_slice_lookup_cache();
            }
            if self.input.take_vte_integral_sky_emissive_toggle() {
                self.toggle_vte_integral_sky_emissive();
            }
            if self.input.take_vte_integral_log_merge_toggle() {
                self.toggle_vte_integral_log_merge();
            }

            if self.input.take_save_world() {
                self.save_world();
            }
            if self.input.take_load_world() {
                self.load_world();
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

            // Mouse look
            if self.mouse_grabbed {
                let pair = self.active_rotation_pair();
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

            let pair = self.active_rotation_pair();
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
                self.place_material = material_digit
                    .clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX);
                eprintln!("Selected place material: {}", self.place_material);
            }
        }

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        let edit_reach = self
            .args
            .edit_reach
            .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);

        if !self.menu_open {
            // Jump when in gravity mode, consume jump either way.
            if self.camera.is_flying {
                self.input.take_jump();
            } else if self.input.take_jump() {
                self.camera.jump();
            }

            // Movement (vertical zeroed in gravity mode internally).
            let prev_position = self.camera.position;
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
                    self.camera.apply_movement(
                        forward,
                        strafe,
                        vertical,
                        w_axis,
                        dt,
                        self.move_speed,
                    );
                }
            }

            // Apply gravity physics.
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

            // Block edit actions.
            let look_dir_for_edit = self.current_look_direction();
            if self.mouse_grabbed {
                let remove_requested = self.input.take_remove_block();
                let place_requested = self.input.take_place_block();
                if remove_requested || place_requested {
                    if remove_requested {
                        if let Some([x, y, z, w]) = self.scene.remove_block_along_ray(
                            self.camera.position,
                            look_dir_for_edit,
                            edit_reach,
                        ) {
                            eprintln!("Removed voxel at ({x}, {y}, {z}, {w})");
                            self.send_multiplayer_voxel_update(
                                now,
                                [x, y, z, w],
                                voxel::VoxelType::AIR.0,
                            );
                        }
                    } else if place_requested {
                        if let Some([x, y, z, w]) = self.scene.place_block_along_ray(
                            self.camera.position,
                            look_dir_for_edit,
                            edit_reach,
                            voxel::VoxelType(self.place_material),
                        ) {
                            eprintln!(
                                "Placed voxel material {} at ({x}, {y}, {z}, {w})",
                                self.place_material
                            );
                            self.send_multiplayer_voxel_update(
                                now,
                                [x, y, z, w],
                                self.place_material,
                            );
                        }
                    }
                }
            } else {
                self.input.take_remove_block();
                self.input.take_place_block();
            }
        } else {
            self.input.take_jump();
            self.input.take_remove_block();
            self.input.take_place_block();
        }

        let look_dir = self.current_look_direction();
        self.send_multiplayer_player_update(now, look_dir);
        let preview_time_s = (now - self.start_time).as_secs_f32();
        let preview_instance = build_place_preview_instance(
            &self.camera,
            self.place_material,
            preview_time_s,
            self.control_scheme,
        );

        // Build view matrix and scene
        let view_matrix = self.current_view_matrix();
        let aspect = self.args.width.max(1) as f32 / self.args.height.max(1) as f32;
        let backend = self.args.backend.to_render_backend();
        let highlight_mode = self.args.edit_highlight_mode;
        let hud_player_tags =
            self.remote_player_tags(&view_matrix, look_dir, self.focal_length_xy, aspect);
        let targets = if !self.menu_open
            && self.mouse_grabbed
            && (highlight_mode.uses_faces() || highlight_mode.uses_edges())
        {
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
        let vte_sweep_status = if let Some(state) = self.vte_sweep_state {
            let profiles = self.vte_sweep_profiles();
            let profile = profiles[state.profile_index];
            format!(
                "#{} {}/{}:{} {}f",
                state.run_id,
                state.profile_index + 1,
                profiles.len(),
                profile.label,
                state.frames_remaining
            )
        } else {
            "off".to_string()
        };
        let do_navigation_hud = self.menu_open || self.info_panel_mode != InfoPanelMode::Off;
        let hud_readout_mode =
            if !self.menu_open && self.info_panel_mode == InfoPanelMode::VectorTable {
                HudReadoutMode::CompactVectors
            } else {
                HudReadoutMode::Full
            };
        let hud_rotation_label = if self.menu_open {
            Some(self.pause_menu_hud_text())
        } else {
            self.current_info_hud_text(
                pair,
                look_dir,
                edit_reach,
                highlight_mode,
                &vte_sweep_status,
                hud_target_hit_voxel,
                hud_target_hit_face,
            )
        };

        let render_options = RenderOptions {
            do_raster: true,
            render_backend: backend,
            vte_max_trace_steps: self.vte_max_trace_steps,
            vte_max_trace_distance: self.vte_max_trace_distance,
            vte_display_mode: self.args.vte_display_mode.to_render_mode(),
            vte_slice_layer: self.args.vte_slice_layer,
            vte_thick_half_width: self.args.vte_thick_half_width,
            vte_reference_compare: self.vte_reference_compare_enabled,
            vte_reference_mismatch_only: self.vte_reference_mismatch_only_enabled,
            vte_compare_slice_only: self.vte_compare_slice_only_enabled,
            vte_y_slice_lookup_cache: self.vte_y_slice_lookup_cache_enabled,
            vte_integral_sky_emissive_tweak: self.vte_integral_sky_emissive_enabled,
            vte_integral_sky_scale: self.vte_integral_sky_scale,
            vte_integral_hit_emissive_boost: self.vte_integral_hit_emissive_boost,
            vte_integral_log_merge_tweak: self.vte_integral_log_merge_enabled,
            vte_integral_log_merge_k: self.vte_integral_log_merge_k,
            vte_highlight_hit_voxel,
            vte_highlight_place_voxel,
            do_navigation_hud,
            custom_overlay_lines,
            take_framebuffer_screenshot: take_screenshot,
            prepare_render_screenshot: auto_screenshot,
            hud_readout_mode,
            hud_rotation_label,
            hud_target_hit_voxel,
            hud_target_hit_face,
            hud_player_tags,
            ..Default::default()
        };

        let frame_params = FrameParams {
            view_matrix,
            focal_length_xy: self.focal_length_xy,
            focal_length_zw: self.focal_length_zw,
            render_options,
        };

        if backend == RenderBackend::VoxelTraversal {
            let voxel_frame = self.scene.build_voxel_frame_data(self.camera.position);
            let mut vte_entity_instances = Vec::new();
            if self.vte_entities_enabled {
                vte_entity_instances.push(build_vte_test_entity_instance(preview_time_s));
            }
            vte_entity_instances.extend(self.remote_player_instances(preview_time_s));
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
            let remote_instances = self.remote_player_instances(preview_time_s);
            self.scene.update_surfaces_if_dirty();
            let instances = self.scene.build_instances(self.camera.position);
            let mut render_instances =
                Vec::with_capacity(instances.len() + remote_instances.len() + 1);
            render_instances.extend_from_slice(instances);
            render_instances.extend(remote_instances);
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

        self.advance_vte_runtime_sweep_after_frame();
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
                    let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
                    if self.menu_open {
                        self.menu_open = false;
                        if let Some(window) = window {
                            self.grab_mouse(&window);
                        }
                    } else if self.mouse_grabbed {
                        if let Some(window) = window {
                            self.release_mouse(&window);
                        }
                        self.menu_open = true;
                        self.menu_selection = 0;
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
                            self.menu_open = false;
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
                    self.menu_open = true;
                    self.menu_selection = 0;
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
