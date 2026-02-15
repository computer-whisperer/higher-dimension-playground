mod audio;
mod app_hud;
mod app_helpers;
mod app_multiplayer;
mod app_perf;
mod app_main_menu;
mod app_runtime;
mod app_ui;
mod camera;
mod cpu_render;
mod input;
mod material_icons;
mod materials;
mod multiplayer;
mod scene;
mod voxel;

use clap::{ArgAction, Parser, ValueEnum};
use egui::RichText;
use higher_dimension_playground::render::{
    CustomOverlayLine, EguiPaintData, EguiPaintMesh, EguiPaintVertex, EguiTextureSlot,
    EguiTextureUpdate, FrameParams, HudPlayerTag, HudReadoutMode, RenderBackend, RenderContext,
    RenderOptions, TetraFrameInput, VteDisplayMode,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use vulkano::device::{Device, Queue};
use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use audio::{AudioEngine, SoundEffect};
use app_helpers::*;
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
const BLOCK_EDIT_PLACE_MATERIAL_MAX: u8 = materials::MAX_MATERIAL_ID;
const SPRINT_SPEED_MULTIPLIER: f32 = 1.8;
const FOOTSTEP_DISTANCE_WALK: f32 = 3.50;
const FOOTSTEP_DISTANCE_SPRINT: f32 = 2.40;
const FOOTSTEP_MIN_XZW_SPEED: f32 = 0.55;
const REMOTE_FOOTSTEP_MAX_DISTANCE: f32 = 36.0;
const REMOTE_FOOTSTEP_MIN_XZW_SPEED: f32 = 0.40;
const REMOTE_FOOTSTEP_MAX_VERTICAL_SPEED: f32 = 2.4;
const REMOTE_FOOTSTEP_MAX_NETWORK_AGE_S: f32 = 0.35;
const REMOTE_FOOTSTEP_MAX_PER_FRAME: usize = 6;
const TARGET_OUTLINE_COLOR: [f32; 4] = [0.14, 0.70, 0.70, 1.00];
const PLACE_OUTLINE_COLOR: [f32; 4] = [0.70, 0.42, 0.14, 1.00];
const WORLD_FILE_DEFAULT: &str = "saves/world.v4dw";
const VTE_SWEEP_SAMPLE_FRAMES: usize = 120;
const VTE_SWEEP_INCLUDE_NO_NON_VOXEL_ENV: &str = "R4D_VTE_SWEEP_INCLUDE_NO_NON_VOXEL_INSTANCES";
const VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV_LEGACY: &str = "R4D_VTE_SWEEP_INCLUDE_NO_ENTITIES";
const VTE_OVERLAY_RASTER_ENV: &str = "R4D_VTE_OVERLAY_RASTER";
// Tags held-block preview instances so shaders can apply preview-only shading boosts.
const PREVIEW_MATERIAL_FLAG: u32 = 0x8000_0000;
const FOCAL_LENGTH_MIN: f32 = 0.20;
const FOCAL_LENGTH_MAX: f32 = 4.00;
const VTE_TRACE_DISTANCE_MIN: f32 = 10.0;
const VTE_TRACE_DISTANCE_MAX: f32 = 4096.0;
const VTE_INTEGRAL_SKY_SCALE_MIN: f32 = 0.0;
const VTE_INTEGRAL_SKY_SCALE_MAX: f32 = 2.0;
const VTE_INTEGRAL_HIT_EMISSIVE_MIN: f32 = 0.0;
const VTE_INTEGRAL_HIT_EMISSIVE_MAX: f32 = 0.20;
const VTE_INTEGRAL_LOG_MERGE_K_MIN: f32 = 0.0;
const VTE_INTEGRAL_LOG_MERGE_K_MAX: f32 = 64.0;
const VTE_TRACE_STEPS_MIN: u32 = 16;
const VTE_TRACE_STEPS_MAX: u32 = 4096;
const MULTIPLAYER_DEFAULT_PORT: u16 = 4000;
const MULTIPLAYER_PLAYER_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const MULTIPLAYER_PENDING_EDIT_TIMEOUT: Duration = Duration::from_secs(5);
const MULTIPLAYER_PENDING_EDIT_MAX: usize = 512;
const CLIENT_PROFILE_REPORT_INTERVAL: Duration = Duration::from_secs(2);
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
const PERF_SUITE_WARMUP_FRAMES_DEFAULT: u32 = 180;
const PERF_SUITE_SAMPLE_FRAMES_DEFAULT: u32 = 600;
const PERF_SUITE_REPORT_DIR_DEFAULT: &str = "profiles";
const MENU_ORBIT_CENTER: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
const MENU_ORBIT_RADIUS_XZ: f32 = 16.0;
const MENU_ORBIT_RADIUS_W: f32 = 7.0;
const MENU_ORBIT_HEIGHT_BASE: f32 = PLAYER_HEIGHT + 1.2;
const MENU_ORBIT_HEIGHT_BOB: f32 = 0.8;
const MENU_ORBIT_RATE_XZ: f32 = 0.23;
const MENU_ORBIT_RATE_W: f32 = 0.17;
const MENU_ORBIT_RATE_Y: f32 = 0.11;
const MENU_ORBIT_TARGET_Y_OFFSET: f32 = 0.6;

#[derive(Copy, Clone)]
struct VteRuntimeProfile {
    label: &'static str,
    non_voxel_instances: bool,
    y_slice_lookup_cache: bool,
}

#[derive(Copy, Clone)]
struct PerfSuiteScenario {
    label: &'static str,
    position: [f32; 4],
    yaw: f32,
    pitch: f32,
    xw_angle: f32,
    zw_angle: f32,
    yw_deviation: f32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum PerfSuitePhase {
    Warmup,
    Sample,
}

struct PerfSuiteState {
    run_started_at: Instant,
    scenario_index: usize,
    phase: PerfSuitePhase,
    frames_remaining: u32,
    warmup_frames: u32,
    sample_frames: u32,
    sample_frame_samples: u32,
    sample_client_cpu_ms_sum: f64,
    sample_client_cpu_ms_max: f64,
    sample_gpu_ms_sum: f64,
    sample_gpu_ms_max: f64,
    sample_gpu_samples: u32,
    results: Vec<PerfSuiteScenarioResult>,
}

struct PerfSuiteScenarioResult {
    scenario_index: usize,
    scenario_label: &'static str,
    client_cpu_avg_ms: f64,
    client_cpu_max_ms: f64,
    client_cpu_frames: u32,
    render_gpu_avg_ms: Option<f64>,
    render_gpu_max_ms: Option<f64>,
    render_gpu_samples: u32,
}

impl PerfSuiteState {
    fn new(warmup_frames: u32, sample_frames: u32, started_at: Instant) -> Self {
        Self {
            run_started_at: started_at,
            scenario_index: 0,
            phase: PerfSuitePhase::Warmup,
            frames_remaining: warmup_frames,
            warmup_frames,
            sample_frames,
            sample_frame_samples: 0,
            sample_client_cpu_ms_sum: 0.0,
            sample_client_cpu_ms_max: 0.0,
            sample_gpu_ms_sum: 0.0,
            sample_gpu_ms_max: 0.0,
            sample_gpu_samples: 0,
            results: Vec::new(),
        }
    }

    fn reset_sample_accumulators(&mut self) {
        self.sample_frame_samples = 0;
        self.sample_client_cpu_ms_sum = 0.0;
        self.sample_client_cpu_ms_max = 0.0;
        self.sample_gpu_ms_sum = 0.0;
        self.sample_gpu_ms_max = 0.0;
        self.sample_gpu_samples = 0;
    }
}

const PERF_SUITE_SCENARIOS: [PerfSuiteScenario; 5] = [
    PerfSuiteScenario {
        label: "ground-3d",
        position: [0.0, 8.0, -24.0, -4.0],
        yaw: 1.25,
        pitch: -0.15,
        xw_angle: 0.0,
        zw_angle: 0.0,
        yw_deviation: 0.0,
    },
    PerfSuiteScenario {
        label: "ground-4d",
        position: [0.0, 8.0, -24.0, -4.0],
        yaw: 1.25,
        pitch: -0.15,
        xw_angle: 0.70,
        zw_angle: 0.95,
        yw_deviation: 0.0,
    },
    PerfSuiteScenario {
        label: "sky-4d",
        position: [0.0, 64.0, 0.0, -4.0],
        yaw: 0.35,
        pitch: -0.70,
        xw_angle: 0.90,
        zw_angle: 0.78,
        yw_deviation: 0.0,
    },
    PerfSuiteScenario {
        label: "oblique-far",
        position: [96.0, 20.0, 96.0, -4.0],
        yaw: -2.35,
        pitch: -0.22,
        xw_angle: -0.45,
        zw_angle: 0.42,
        yw_deviation: 0.0,
    },
    PerfSuiteScenario {
        label: "ridge-4d",
        position: [-64.0, 14.0, 32.0, -4.0],
        yaw: 0.95,
        pitch: -0.08,
        xw_angle: 0.46,
        zw_angle: 0.52,
        yw_deviation: 0.0,
    },
];

const VTE_SWEEP_PROFILES_ENTITIES: [VteRuntimeProfile; 2] = [
    VteRuntimeProfile {
        label: "A baseline",
        non_voxel_instances: true,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "B no-lookup-cache",
        non_voxel_instances: true,
        y_slice_lookup_cache: false,
    },
];

const VTE_SWEEP_PROFILES_EXTENDED: [VteRuntimeProfile; 4] = [
    VteRuntimeProfile {
        label: "A baseline",
        non_voxel_instances: true,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "B no-nonvoxel",
        non_voxel_instances: false,
        y_slice_lookup_cache: true,
    },
    VteRuntimeProfile {
        label: "C no-lookup-cache",
        non_voxel_instances: true,
        y_slice_lookup_cache: false,
    },
    VteRuntimeProfile {
        label: "D no-nonvoxel-no-lookup-cache",
        non_voxel_instances: false,
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

#[derive(Debug, Clone)]
enum AutoCommand {
    Press(KeyCode),
    Wait(u32),
    Screenshot,
}

fn parse_keycode(key: &str) -> Option<KeyCode> {
    match key.to_lowercase().as_str() {
        "escape" => Some(KeyCode::Escape),
        "e" => Some(KeyCode::KeyE),
        "i" => Some(KeyCode::KeyI),
        "w" => Some(KeyCode::KeyW),
        "a" => Some(KeyCode::KeyA),
        "s" => Some(KeyCode::KeyS),
        "d" => Some(KeyCode::KeyD),
        "q" => Some(KeyCode::KeyQ),
        "r" => Some(KeyCode::KeyR),
        "f" => Some(KeyCode::KeyF),
        "g" => Some(KeyCode::KeyG),
        "tab" => Some(KeyCode::Tab),
        "space" => Some(KeyCode::Space),
        "shift" => Some(KeyCode::ShiftLeft),
        "1" => Some(KeyCode::Digit1),
        "2" => Some(KeyCode::Digit2),
        "3" => Some(KeyCode::Digit3),
        "4" => Some(KeyCode::Digit4),
        "5" => Some(KeyCode::Digit5),
        "6" => Some(KeyCode::Digit6),
        "7" => Some(KeyCode::Digit7),
        "8" => Some(KeyCode::Digit8),
        "9" => Some(KeyCode::Digit9),
        "0" => Some(KeyCode::Digit0),
        "f5" => Some(KeyCode::F5),
        "f9" => Some(KeyCode::F9),
        "f12" => Some(KeyCode::F12),
        "enter" => Some(KeyCode::Enter),
        "up" => Some(KeyCode::ArrowUp),
        "down" => Some(KeyCode::ArrowDown),
        "left" => Some(KeyCode::ArrowLeft),
        "right" => Some(KeyCode::ArrowRight),
        "lbracket" => Some(KeyCode::BracketLeft),
        "rbracket" => Some(KeyCode::BracketRight),
        _ => None,
    }
}

fn parse_commands(input: &str) -> Vec<AutoCommand> {
    let mut commands = Vec::new();
    for cmd_str in input.split(';') {
        let cmd_str = cmd_str.trim();
        if cmd_str.is_empty() {
            continue;
        }
        if let Some((cmd_type, arg)) = cmd_str.split_once(':') {
            match cmd_type.trim() {
                "press" => {
                    if let Some(keycode) = parse_keycode(arg.trim()) {
                        commands.push(AutoCommand::Press(keycode));
                    } else {
                        eprintln!("Warning: unknown key '{}'", arg.trim());
                    }
                }
                "wait" => {
                    if let Ok(frames) = arg.trim().parse::<u32>() {
                        commands.push(AutoCommand::Wait(frames));
                    } else {
                        eprintln!("Warning: invalid wait frames '{}'", arg.trim());
                    }
                }
                _ => {
                    eprintln!("Warning: unknown command type '{}'", cmd_type.trim());
                }
            }
        } else if cmd_str == "screenshot" {
            commands.push(AutoCommand::Screenshot);
        } else {
            eprintln!("Warning: invalid command '{}'", cmd_str);
        }
    }
    commands
}

#[derive(Parser, Debug, Clone)]
#[command(version, about = "4D polychora explorer")]
struct Args {
    /// Render buffer width in pixels
    #[arg(long, short = 'W', default_value_t = 960)]
    width: u32,

    /// Render buffer height in pixels
    #[arg(long, short = 'H', default_value_t = 540)]
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

    /// World-space distance where L0 tracing hands off to coarser LODs.
    #[arg(long, default_value_t = 48.0)]
    vte_lod_near_max_distance: f32,

    /// World-space distance where L1 tracing hands off to L2 tracing.
    #[arg(long, default_value_t = 112.0)]
    vte_lod_mid_max_distance: f32,

    /// VTE Stage-B display operator (integral, slice, thick-slice, debug-compare, debug-integral)
    #[arg(long, value_enum, default_value_t = VteDisplayModeArg::Integral)]
    vte_display_mode: VteDisplayModeArg,

    /// VTE slice center layer index (0..layers-1). Defaults to center layer.
    #[arg(long)]
    vte_slice_layer: Option<u32>,

    /// VTE thick-slice half-width in layer indices.
    #[arg(long, default_value_t = 2)]
    vte_thick_half_width: u32,

    /// Enable non-voxel instance integration in VTE Stage A (test tesseract + non-voxel BVH).
    /// Set to false to profile pure voxel traversal without the secondary tetra pipeline.
    #[arg(
        long = "vte-non-voxel-instances",
        alias = "vte-entities",
        action = ArgAction::Set,
        default_value_t = true
    )]
    vte_non_voxel_instances: bool,

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
    #[arg(long, default_value_t = 0.25)]
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

    /// Integrated singleplayer server tick rate in Hz.
    #[arg(long, default_value_t = 10.0)]
    singleplayer_tick_hz: f32,

    /// Integrated singleplayer entity simulation rate in Hz.
    #[arg(long, default_value_t = 30.0)]
    singleplayer_entity_sim_hz: f32,

    /// Autosave interval for integrated singleplayer server (seconds).
    #[arg(long, default_value_t = 5)]
    singleplayer_save_interval_secs: u64,

    /// Send full world snapshot on join for integrated singleplayer server.
    #[arg(long, default_value_t = true)]
    singleplayer_snapshot_on_join: bool,

    /// Enable server-managed random structure generation in singleplayer.
    #[arg(long, default_value_t = true)]
    singleplayer_procgen_structures: bool,

    /// Chunk radius around player for streamed near (L0) chunk updates in singleplayer.
    #[arg(long, default_value_t = 6)]
    singleplayer_procgen_chunk_radius: i32,

    /// Chunk radius around player for streamed mid (L1) chunk updates in singleplayer.
    #[arg(long, default_value_t = 10)]
    singleplayer_procgen_mid_chunk_radius: i32,

    /// Chunk radius around player for streamed far (L2) chunk updates in singleplayer.
    #[arg(long, default_value_t = 6)]
    singleplayer_procgen_far_chunk_radius: i32,

    /// Derive a procgen keepout mask from persisted world chunks in singleplayer.
    #[arg(long, default_value_t = true)]
    singleplayer_procgen_keepout_from_existing_world: bool,

    /// Keepout chunk padding around persisted chunks used to block new procgen placements.
    #[arg(long, default_value_t = 1)]
    singleplayer_procgen_keepout_padding_chunks: i32,

    /// World seed used by integrated singleplayer server procgen.
    #[arg(long, default_value_t = 1337)]
    singleplayer_world_seed: u64,

    /// Window inner width (overrides OS default)
    #[arg(long)]
    window_width: Option<u32>,

    /// Window inner height (overrides OS default)
    #[arg(long)]
    window_height: Option<u32>,

    /// Output path for --gpu-screenshot (default: frames/gpu_render.png)
    #[arg(long, default_value = "frames/gpu_render.png")]
    screenshot_output: PathBuf,

    /// Suppress all HUD/overlay elements in screenshot
    #[arg(long)]
    no_hud: bool,

    /// Automated command sequence (semicolon-separated).
    /// Commands: press:<key>, wait:<frames>, screenshot
    #[arg(long)]
    commands: Option<String>,

    /// Run deterministic built-in performance scenario suite.
    /// Input is ignored while active.
    #[arg(long, action = ArgAction::Set, default_value_t = false)]
    perf_suite: bool,

    /// Warmup frames before collecting samples for each perf-suite scenario.
    #[arg(long, default_value_t = PERF_SUITE_WARMUP_FRAMES_DEFAULT)]
    perf_suite_warmup_frames: u32,

    /// Sample frames collected per perf-suite scenario.
    #[arg(long, default_value_t = PERF_SUITE_SAMPLE_FRAMES_DEFAULT)]
    perf_suite_sample_frames: u32,

    /// Exit automatically when the perf-suite finishes all scenarios.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    perf_suite_exit_on_complete: bool,

    /// Output path for perf-suite machine-readable report (JSON).
    /// Defaults to profiles/perf-suite-<unix-seconds>.json when omitted.
    #[arg(long)]
    perf_suite_report: Option<PathBuf>,

    /// Enable client-side audio output (sound effects).
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    audio: bool,

    /// Master volume for client-side sound effects.
    #[arg(long, default_value_t = 0.7)]
    audio_volume: f32,
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
    VectorTable2,
    Off,
}

impl InfoPanelMode {
    fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::VectorTable => "vectors",
            Self::VectorTable2 => "vectors2",
            Self::Off => "off",
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
    let vte_sweep_include_no_non_voxel = env_flag_enabled(VTE_SWEEP_INCLUDE_NO_NON_VOXEL_ENV)
        || env_flag_enabled(VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV_LEGACY);
    let initial_vte_non_voxel_instances_enabled = args.vte_non_voxel_instances;
    let initial_vte_y_slice_lookup_cache_enabled = args.vte_y_slice_lookup_cache;
    let initial_vte_integral_sky_emissive_enabled = args.vte_integral_sky_emissive_tweak;
    let initial_vte_integral_log_merge_enabled = args.vte_integral_log_merge_tweak;
    let initial_vte_integral_sky_scale = args.vte_integral_sky_scale.max(0.0);
    let initial_vte_integral_hit_emissive_boost = args.vte_integral_hit_emissive_boost.max(0.0);
    let initial_vte_integral_log_merge_k = args.vte_integral_log_merge_k.max(0.0);
    let initial_vte_max_trace_steps = args.vte_max_trace_steps.max(1);
    let initial_vte_max_trace_distance = args.vte_max_trace_distance.max(1.0);
    let initial_vte_lod_near_max_distance = args
        .vte_lod_near_max_distance
        .max(1.0)
        .min(initial_vte_max_trace_distance);
    let initial_vte_lod_mid_max_distance = args
        .vte_lod_mid_max_distance
        .max(initial_vte_lod_near_max_distance)
        .min(initial_vte_max_trace_distance);

    let world_file = args.world_file.clone();
    // Determine whether to skip the main menu and go straight to playing.
    // Skip if any CLI arg implies the user wants to play immediately.
    let skip_main_menu = args.load_world
        || args.server.is_some()
        || args.gpu_screenshot
        || args.cpu_render
        || args.commands.is_some()
        || args.perf_suite
        || !matches!(args.scene, SceneArg::Flat);
    let start_with_integrated_singleplayer = args.server.is_none() && skip_main_menu;
    let initial_app_state = if skip_main_menu {
        AppState::Playing
    } else {
        AppState::MainMenu
    };

    let start_with_network_world = args.server.is_some() || start_with_integrated_singleplayer;
    let scene_preset = if initial_app_state == AppState::MainMenu {
        ScenePreset::DemoCubes
    } else if start_with_network_world {
        // Networked play should render server-authoritative world data only.
        ScenePreset::Empty
    } else {
        args.scene.to_scene_preset()
    };
    let mut scene = Scene::new(scene_preset);
    if args.load_world && !start_with_integrated_singleplayer {
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
    } else if start_with_integrated_singleplayer {
        let player_name = args
            .player_name
            .clone()
            .unwrap_or_else(default_multiplayer_player_name);
        let runtime_config = build_singleplayer_runtime_config(&args, world_file.clone());
        match MultiplayerClient::connect_local(runtime_config, player_name.clone()) {
            Ok(client) => {
                eprintln!(
                    "Starting integrated singleplayer server for {} as '{}'",
                    world_file.display(),
                    player_name
                );
                Some(client)
            }
            Err(error) => {
                eprintln!(
                    "Failed to start integrated singleplayer server for {}: {}",
                    world_file.display(),
                    error
                );
                None
            }
        }
    } else {
        None
    };

    let mut command_queue: VecDeque<AutoCommand> = if let Some(cmd_str) = &args.commands {
        parse_commands(cmd_str).into()
    } else {
        VecDeque::new()
    };
    if args.perf_suite && !command_queue.is_empty() {
        eprintln!("Perf suite active: ignoring --commands automation.");
        command_queue.clear();
    }

    let perf_suite_state = if args.perf_suite {
        Some(PerfSuiteState::new(
            args.perf_suite_warmup_frames.max(1),
            args.perf_suite_sample_frames.max(1),
            Instant::now(),
        ))
    } else {
        None
    };

    let initial_render_width = args.width;
    let initial_render_height = args.height;
    let initial_render_layers = args.layers;
    let initial_player_name = args
        .player_name
        .clone()
        .unwrap_or_else(default_multiplayer_player_name);
    let audio = AudioEngine::new(args.audio, args.audio_volume);
    if args.audio {
        if audio.is_active() {
            eprintln!(
                "Audio enabled (volume {:.2})",
                args.audio_volume.clamp(0.0, 2.0)
            );
        } else {
            eprintln!("Audio disabled (no output device)");
        }
    } else {
        eprintln!("Audio disabled via --audio=false");
    }

    let mut app = App {
        instance,
        device,
        queue,
        rcx: None,
        scene,
        camera,
        input: InputState::new(),
        audio,
        footstep_distance_accum: 0.0,
        was_grounded_last_frame: false,
        start_time: Instant::now(),
        last_frame: Instant::now(),
        mouse_grabbed: false,
        should_exit_after_render: false,
        gpu_screenshot_countdown: if gpu_screenshot && args.commands.is_none() {
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
        sprint_enabled: false,
        info_panel_mode: InfoPanelMode::VectorTable,
        focal_length_xy: 1.0,
        focal_length_zw: 1.0,
        place_material: BLOCK_EDIT_PLACE_MATERIAL_DEFAULT,
        world_file,
        vte_reference_compare_enabled,
        vte_reference_mismatch_only_enabled,
        vte_compare_slice_only_enabled,
        vte_non_voxel_instances_enabled: initial_vte_non_voxel_instances_enabled,
        vte_y_slice_lookup_cache_enabled: initial_vte_y_slice_lookup_cache_enabled,
        vte_integral_sky_emissive_enabled: initial_vte_integral_sky_emissive_enabled,
        vte_integral_sky_scale: initial_vte_integral_sky_scale,
        vte_integral_hit_emissive_boost: initial_vte_integral_hit_emissive_boost,
        vte_integral_log_merge_enabled: initial_vte_integral_log_merge_enabled,
        vte_integral_log_merge_k: initial_vte_integral_log_merge_k,
        vte_max_trace_steps: initial_vte_max_trace_steps,
        vte_max_trace_distance: initial_vte_max_trace_distance,
        vte_sweep_include_no_non_voxel,
        vte_lod_near_max_distance: initial_vte_lod_near_max_distance,
        vte_lod_mid_max_distance: initial_vte_lod_mid_max_distance,
        vte_sweep_state: None,
        vte_sweep_run_id: 0,
        hotbar_slots: [3, 27, 28, 29, 31, 12, 13, 1, 4],
        hotbar_selected_index: 0,
        inventory_open: false,
        teleport_dialog_open: false,
        teleport_coords: [
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
            "0".to_string(),
        ],
        controls_dialog_open: false,
        menu_open: false,
        menu_selection: 0,
        egui_ctx: egui::Context::default(),
        egui_winit_state: None,
        material_icon_sheet: None,
        material_icons_texture_id: None,
        multiplayer,
        multiplayer_self_id: None,
        next_multiplayer_edit_id: 1,
        pending_voxel_edits: Vec::new(),
        remote_players: HashMap::new(),
        remote_entities: HashMap::new(),
        last_multiplayer_player_update: Instant::now(),
        command_queue,
        command_wait_frames: 0,
        app_state: initial_app_state,
        main_menu_page: MainMenuPage::Root,
        main_menu_server_address: "c-gateway.computer-whisperer.network:4000".to_string(),
        main_menu_player_name: initial_player_name,
        main_menu_world_files: Vec::new(),
        main_menu_selected_world: None,
        main_menu_connect_error: None,
        look_at_target: None,
        menu_camera: make_menu_camera(),
        menu_time: 0.0,
        pending_render_width: initial_render_width,
        pending_render_height: initial_render_height,
        pending_render_layers: initial_render_layers,
        profile_window_start: Instant::now(),
        profile_frame_samples: 0,
        profile_client_cpu_ms_sum: 0.0,
        profile_client_cpu_ms_max: 0.0,
        profile_gpu_ms_sum: 0.0,
        profile_gpu_ms_max: 0.0,
        profile_gpu_samples: 0,
        perf_suite_state,
        world_ready: initial_app_state == AppState::MainMenu,
        vte_overlay_raster_enabled: env_flag_enabled_or(VTE_OVERLAY_RASTER_ENV, false),
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
    if !app.vte_non_voxel_instances_enabled {
        eprintln!(
            "VTE non-voxel pipeline disabled via --vte-non-voxel-instances=false (alias: --vte-entities=false)"
        );
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
    if app.vte_sweep_include_no_non_voxel {
        eprintln!(
            "VTE sweep: extended mode enabled via {} or {} (includes no-nonvoxel profiles).",
            VTE_SWEEP_INCLUDE_NO_NON_VOXEL_ENV, VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV_LEGACY
        );
    } else {
        eprintln!("VTE sweep: non-voxel mode (default, non-voxel instances stay enabled).");
    }
    if app.vte_overlay_raster_enabled {
        eprintln!(
            "VTE overlay raster enabled via {} (legacy held-block raster pass; higher GPU cost).",
            VTE_OVERLAY_RASTER_ENV
        );
    } else {
        eprintln!(
            "VTE held-block preview uses VTE entity path (default). Set {}=1 to use legacy overlay raster path.",
            VTE_OVERLAY_RASTER_ENV
        );
    }
    if app.perf_suite_active() {
        let report_path = app.perf_suite_report_path();
        eprintln!(
            "[perf-suite] enabled: {} scenarios, warmup={}f, sample={}f, exit_on_complete={}",
            PERF_SUITE_SCENARIOS.len(),
            app.args.perf_suite_warmup_frames.max(1),
            app.args.perf_suite_sample_frames.max(1),
            app.args.perf_suite_exit_on_complete
        );
        eprintln!("[perf-suite] report path: {}", report_path.display());
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

#[derive(Copy, Clone, PartialEq, Eq)]
enum AppState {
    MainMenu,
    Playing,
}

#[derive(Clone, PartialEq)]
enum MainMenuPage {
    Root,
    Singleplayer,
    Multiplayer,
}

enum MainMenuTransition {
    NewWorld,
    LoadWorld(PathBuf),
    ConnectMultiplayer(String),
}

struct WorldFileEntry {
    path: PathBuf,
    display_name: String,
    size_bytes: u64,
}

fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

#[derive(Copy, Clone)]
enum LookAtTarget {
    Angles {
        yaw: f32,
        pitch: f32,
        xw_angle: f32,
        zw_angle: f32,
    },
    Direction([f32; 4]),
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    rcx: Option<RenderContext>,
    scene: Scene,
    camera: Camera4D,
    input: InputState,
    audio: AudioEngine,
    footstep_distance_accum: f32,
    was_grounded_last_frame: bool,
    start_time: Instant,
    last_frame: Instant,
    mouse_grabbed: bool,
    should_exit_after_render: bool,
    gpu_screenshot_countdown: u32,
    args: Args,
    control_scheme: ControlScheme,
    scroll_cycle_pair: RotationPair,
    move_speed: f32,
    sprint_enabled: bool,
    info_panel_mode: InfoPanelMode,
    focal_length_xy: f32,
    focal_length_zw: f32,
    place_material: u8,
    world_file: PathBuf,
    vte_reference_compare_enabled: bool,
    vte_reference_mismatch_only_enabled: bool,
    vte_compare_slice_only_enabled: bool,
    vte_non_voxel_instances_enabled: bool,
    vte_y_slice_lookup_cache_enabled: bool,
    vte_integral_sky_emissive_enabled: bool,
    vte_integral_sky_scale: f32,
    vte_integral_hit_emissive_boost: f32,
    vte_integral_log_merge_enabled: bool,
    vte_integral_log_merge_k: f32,
    vte_max_trace_steps: u32,
    vte_max_trace_distance: f32,
    vte_sweep_include_no_non_voxel: bool,
    vte_lod_near_max_distance: f32,
    vte_lod_mid_max_distance: f32,
    vte_sweep_state: Option<VteSweepState>,
    vte_sweep_run_id: u32,
    hotbar_slots: [u8; 9],
    hotbar_selected_index: usize,
    inventory_open: bool,
    teleport_dialog_open: bool,
    teleport_coords: [String; 4],
    controls_dialog_open: bool,
    menu_open: bool,
    menu_selection: usize,
    egui_ctx: egui::Context,
    egui_winit_state: Option<egui_winit::State>,
    material_icon_sheet: Option<material_icons::MaterialIconSheet>,
    material_icons_texture_id: Option<egui::TextureId>,
    multiplayer: Option<MultiplayerClient>,
    multiplayer_self_id: Option<u64>,
    next_multiplayer_edit_id: u64,
    pending_voxel_edits: Vec<PendingVoxelEdit>,
    remote_players: HashMap<u64, RemotePlayerState>,
    remote_entities: HashMap<u64, RemoteEntityState>,
    last_multiplayer_player_update: Instant,
    command_queue: VecDeque<AutoCommand>,
    command_wait_frames: u32,
    app_state: AppState,
    main_menu_page: MainMenuPage,
    main_menu_server_address: String,
    main_menu_player_name: String,
    main_menu_world_files: Vec<WorldFileEntry>,
    main_menu_selected_world: Option<usize>,
    main_menu_connect_error: Option<String>,
    look_at_target: Option<LookAtTarget>,
    menu_camera: Camera4D,
    menu_time: f32,
    // Runtime resolution UI state (edited values before Apply)
    pending_render_width: u32,
    pending_render_height: u32,
    pending_render_layers: u32,
    profile_window_start: Instant,
    profile_frame_samples: u32,
    profile_client_cpu_ms_sum: f64,
    profile_client_cpu_ms_max: f64,
    profile_gpu_ms_sum: f64,
    profile_gpu_ms_max: f64,
    profile_gpu_samples: u32,
    perf_suite_state: Option<PerfSuiteState>,
    world_ready: bool,
    vte_overlay_raster_enabled: bool,
}

#[derive(Copy, Clone)]
struct VteSweepState {
    run_id: u32,
    profile_index: usize,
    frames_remaining: usize,
    previous_non_voxel_instances: bool,
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
    footstep_distance_accum: f32,
}

#[derive(Clone)]
struct RemoteEntityState {
    kind: multiplayer::EntityKind,
    position: [f32; 4],
    orientation: [f32; 4],
    scale: f32,
    material: u8,
    render_position: [f32; 4],
    render_orientation: [f32; 4],
    last_received_at: Instant,
}

const REMOTE_ENTITY_POSITION_SMOOTH_HZ: f32 = 12.0;
const REMOTE_ENTITY_ORIENTATION_SMOOTH_HZ: f32 = 16.0;
const REMOTE_ENTITY_TELEPORT_SNAP_DISTANCE: f32 = 20.0;

#[derive(Clone)]
struct PendingVoxelEdit {
    client_edit_id: u64,
    position: [i32; 4],
    material: u8,
    created_at: Instant,
}

impl App {
    fn inject_key_press(&mut self, keycode: KeyCode) {
        match keycode {
            KeyCode::Escape => {
                if self.teleport_dialog_open {
                    self.teleport_dialog_open = false;
                } else if self.inventory_open {
                    self.inventory_open = false;
                } else {
                    self.menu_open = !self.menu_open;
                }
            }
            KeyCode::KeyI => {
                self.inventory_open = !self.inventory_open;
            }
            KeyCode::Digit1 => self.hotbar_selected_index = 0,
            KeyCode::Digit2 => self.hotbar_selected_index = 1,
            KeyCode::Digit3 => self.hotbar_selected_index = 2,
            KeyCode::Digit4 => self.hotbar_selected_index = 3,
            KeyCode::Digit5 => self.hotbar_selected_index = 4,
            KeyCode::Digit6 => self.hotbar_selected_index = 5,
            KeyCode::Digit7 => self.hotbar_selected_index = 6,
            KeyCode::Digit8 => self.hotbar_selected_index = 7,
            KeyCode::Digit9 => self.hotbar_selected_index = 8,
            KeyCode::F5 => {
                // Save world
                if let Err(e) = self.scene.save_world_to_path(&self.world_file) {
                    eprintln!("Failed to save world: {}", e);
                } else {
                    eprintln!("Saved world to {}", self.world_file.display());
                }
            }
            KeyCode::F9 => {
                // Load world
                if let Err(e) = self.scene.load_world_from_path(&self.world_file) {
                    eprintln!("Failed to load world: {}", e);
                } else {
                    eprintln!("Loaded world from {}", self.world_file.display());
                }
            }
            KeyCode::F12 => {
                // Screenshot will be handled by setting the flag
            }
            KeyCode::ArrowUp => {
                if self.menu_open && self.menu_selection > 0 {
                    self.menu_selection -= 1;
                }
            }
            KeyCode::ArrowDown => {
                if self.menu_open && self.menu_selection < 2 {
                    self.menu_selection += 1;
                }
            }
            KeyCode::Enter => {
                if self.menu_open {
                    // Handle menu activation
                }
            }
            KeyCode::KeyG => {
                self.input.request_look_at();
            }
            _ => {}
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

    fn cycle_hotbar_material_prev(&mut self) {
        let slot = &mut self.hotbar_slots[self.hotbar_selected_index];
        *slot = if *slot <= BLOCK_EDIT_PLACE_MATERIAL_MIN {
            BLOCK_EDIT_PLACE_MATERIAL_MAX
        } else {
            slot.saturating_sub(1)
        };
        self.place_material = *slot;
        eprintln!(
            "Hotbar slot {} material: {} ({})",
            self.hotbar_selected_index + 1,
            self.place_material,
            materials::material_name(self.place_material),
        );
    }

    fn cycle_hotbar_material_next(&mut self) {
        let slot = &mut self.hotbar_slots[self.hotbar_selected_index];
        *slot = if *slot >= BLOCK_EDIT_PLACE_MATERIAL_MAX {
            BLOCK_EDIT_PLACE_MATERIAL_MIN
        } else {
            slot.saturating_add(1)
        };
        self.place_material = *slot;
        eprintln!(
            "Hotbar slot {} material: {} ({})",
            self.hotbar_selected_index + 1,
            self.place_material,
            materials::material_name(self.place_material),
        );
    }

    fn toggle_inventory(&mut self) {
        self.inventory_open = !self.inventory_open;
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        if let Some(window) = window {
            if self.inventory_open {
                self.release_mouse(&window);
            } else {
                self.grab_mouse(&window);
            }
        }
    }

    fn toggle_teleport_dialog(&mut self) {
        self.teleport_dialog_open = !self.teleport_dialog_open;
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        if let Some(window) = window {
            if self.teleport_dialog_open {
                // Pre-fill with current position
                let pos = self.camera.position;
                self.teleport_coords = [
                    format!("{:.1}", pos[0]),
                    format!("{:.1}", pos[1]),
                    format!("{:.1}", pos[2]),
                    format!("{:.1}", pos[3]),
                ];
                self.release_mouse(&window);
            } else {
                self.grab_mouse(&window);
            }
        }
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

    fn set_control_scheme(&mut self, target: ControlScheme) {
        while self.control_scheme != target {
            self.cycle_control_scheme();
        }
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

    fn toggle_vte_non_voxel_instances(&mut self) {
        if let Some(state) = self.vte_sweep_state {
            eprintln!(
                "[VTE sweep #{}] ignoring manual non-voxel toggle while sweep is active.",
                state.run_id
            );
            return;
        }
        self.vte_non_voxel_instances_enabled = !self.vte_non_voxel_instances_enabled;
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "VTE runtime non-voxel instances: {}",
            if self.vte_non_voxel_instances_enabled {
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
        self.input.take_menu_left();
        self.input.take_menu_right();
        self.input.take_menu_up();
        self.input.take_menu_down();
        self.input.take_menu_activate();
        // Do NOT drain take_look_at() - G key should work during gameplay
        self.input.take_scheme_cycle();
        self.input.take_vte_sweep();
        self.input.take_vte_non_voxel_instances_toggle();
        self.input.take_vte_y_slice_lookup_cache_toggle();
        self.input.take_vte_integral_sky_emissive_toggle();
        self.input.take_vte_integral_log_merge_toggle();
        self.input.take_save_world();
        self.input.take_load_world();
        self.input.take_scroll_steps();
        self.input.take_fly_toggle();
        self.input.take_sprint_toggle();
        self.input.take_jump();
        self.input.take_place_material_prev();
        self.input.take_place_material_next();
        self.input.take_place_material_digit();
        self.input.take_remove_block();
        self.input.take_place_block();
        self.input.take_pick_material();
        self.input.take_inventory_toggle();
        self.input.take_teleport_dialog();
        self.input.take_mouse_delta();
    }

    fn update_and_render(&mut self) {
        let frame_start = Instant::now();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        if self.app_state == AppState::MainMenu {
            self.update_and_render_main_menu(dt);
            if self.perf_suite_active() {
                self.advance_perf_suite_after_frame(frame_start);
            } else {
                self.record_runtime_profile_sample(frame_start);
            }
            return;
        }

        self.poll_multiplayer_events();
        self.reapply_pending_voxel_edits(now);
        self.smooth_remote_players(dt, now);
        self.smooth_remote_entities(dt);
        if let Some(scenario_index) = self
            .perf_suite_state
            .as_ref()
            .map(|state| state.scenario_index)
        {
            self.set_perf_suite_camera_pose(scenario_index);
        }

        // Process command queue
        let mut command_screenshot_requested = false;
        if !self.perf_suite_active() {
            if self.command_wait_frames > 0 {
                self.command_wait_frames -= 1;
            } else if let Some(cmd) = self.command_queue.pop_front() {
                match cmd {
                    AutoCommand::Press(keycode) => {
                        self.inject_key_press(keycode);
                    }
                    AutoCommand::Wait(n) => {
                        self.command_wait_frames = n;
                    }
                    AutoCommand::Screenshot => {
                        command_screenshot_requested = true;
                    }
                }
            } else if !self.command_queue.is_empty() || self.command_wait_frames > 0 {
                // Still processing commands
            } else if self.args.commands.is_some() && self.args.gpu_screenshot {
                // Commands finished and we're in screenshot mode, exit
                self.should_exit_after_render = true;
            }
        }

        if self.menu_open || self.inventory_open || self.teleport_dialog_open {
            self.drain_gameplay_inputs_while_menu_open();
        } else {
            self.input.take_menu_left();
            self.input.take_menu_right();
            self.input.take_menu_up();
            self.input.take_menu_down();
            self.input.take_menu_activate();

            // (Tab scheme cycling removed - Tab now opens inventory)

            if self.input.take_vte_sweep() {
                self.toggle_vte_runtime_sweep();
            }
            if self.input.take_vte_non_voxel_instances_toggle() {
                self.toggle_vte_non_voxel_instances();
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
                    // Scroll wheel cycles hotbar selection.
                    for _ in 0..scroll_steps.abs() {
                        if scroll_steps > 0 {
                            self.hotbar_selected_index = (self.hotbar_selected_index + 1) % 9;
                        } else {
                            self.hotbar_selected_index = (self.hotbar_selected_index + 8) % 9;
                        }
                    }
                    self.place_material = self.hotbar_slots[self.hotbar_selected_index];
                    eprintln!(
                        "Hotbar slot {} selected: material {} ({})",
                        self.hotbar_selected_index + 1,
                        self.place_material,
                        materials::material_name(self.place_material),
                    );
                }
            }

            // Mouse look
            if self.mouse_grabbed {
                let pair = self.active_rotation_pair();
                let (dx, dy) = self.input.take_mouse_delta();
                // Cancel look-at pull on any mouse movement
                if dx.abs() > 0.5 || dy.abs() > 0.5 {
                    self.look_at_target = None;
                }
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

            if self.input.reset_orientation_held() || self.input.pull_to_3d_held() {
                let pull_home = self.input.reset_orientation_held();
                self.look_at_target = None; // Cancel look-at when holding R or F
                match self.control_scheme {
                    ControlScheme::LookTransport | ControlScheme::RotorFree => {
                        if pull_home {
                            self.camera.pull_toward_home_look_frame(dt);
                        } else {
                            self.camera.pull_toward_nearest_3d_look_frame(dt);
                        }
                    }
                    ControlScheme::IntuitiveUpright
                    | ControlScheme::LegacySideButtonLayers
                    | ControlScheme::LegacyScrollCycle => {
                        if pull_home {
                            self.camera.pull_toward_home_angles(dt);
                        } else {
                            self.camera.pull_toward_nearest_3d_angles(dt);
                        }
                    }
                }
            }

            // Look-at: on G press, fan-cast across the ZW viewing wedge to
            // find the nearest solid block and smoothly rotate toward it.
            if self.input.take_look_at() && self.mouse_grabbed {
                let edit_reach = self
                    .args
                    .edit_reach
                    .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);
                let (_right, _up, view_z, view_w) = self.current_view_basis();
                let hit = self.scene.fan_cast_nearest_block(
                    self.camera.position,
                    view_z,
                    view_w,
                    self.focal_length_zw,
                    edit_reach,
                    32,
                );
                if let Some([x, y, z, w]) = hit {
                    let target_pos = [
                        x as f32 + 0.5,
                        y as f32 + 0.5,
                        z as f32 + 0.5,
                        w as f32 + 0.5,
                    ];
                    let dir = [
                        target_pos[0] - self.camera.position[0],
                        target_pos[1] - self.camera.position[1],
                        target_pos[2] - self.camera.position[2],
                        target_pos[3] - self.camera.position[3],
                    ];
                    match self.control_scheme {
                        ControlScheme::LookTransport | ControlScheme::RotorFree => {
                            self.look_at_target = Some(LookAtTarget::Direction(dir));
                        }
                        _ => {
                            let (ty, tp, txw, tzw) = Camera4D::angles_for_direction_upright(dir);
                            self.look_at_target = Some(LookAtTarget::Angles {
                                yaw: ty,
                                pitch: tp,
                                xw_angle: txw,
                                zw_angle: tzw,
                            });
                        }
                    }
                }
            }

            // Apply smooth pull toward look-at target
            if let Some(target) = self.look_at_target {
                let converged = match target {
                    LookAtTarget::Angles {
                        yaw,
                        pitch,
                        xw_angle,
                        zw_angle,
                    } => self
                        .camera
                        .pull_toward_target_angles(yaw, pitch, xw_angle, zw_angle, dt),
                    LookAtTarget::Direction(dir) => {
                        self.camera.pull_toward_target_direction_look_frame(dir, dt)
                    }
                };
                if converged {
                    self.look_at_target = None;
                }
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
            if self.input.take_sprint_toggle() && !self.sprint_enabled {
                self.sprint_enabled = true;
                eprintln!("Sprint: on");
            }

            // Block place material selection via bracket keys.
            if self.input.take_place_material_prev() {
                self.cycle_hotbar_material_prev();
            }
            if self.input.take_place_material_next() {
                self.cycle_hotbar_material_next();
            }
            // Number keys 1-9 select hotbar slot.
            if let Some(digit) = self.input.take_place_material_digit() {
                if digit >= 1 && digit <= 9 {
                    self.hotbar_selected_index = (digit - 1) as usize;
                    self.place_material = self.hotbar_slots[self.hotbar_selected_index];
                    eprintln!(
                        "Hotbar slot {} selected: material {} ({})",
                        digit,
                        self.place_material,
                        materials::material_name(self.place_material),
                    );
                }
            }
            // E key toggles inventory.
            if self.input.take_inventory_toggle() {
                self.toggle_inventory();
            }
            // T key toggles teleport dialog.
            if self.input.take_teleport_dialog() {
                self.toggle_teleport_dialog();
            }
        }

        // Determine active rotation pair
        let pair = self.active_rotation_pair();

        let edit_reach = self
            .args
            .edit_reach
            .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);

        if !self.menu_open && !self.inventory_open && !self.teleport_dialog_open {
            // Jump when in gravity mode, consume jump either way.
            if self.camera.is_flying {
                self.input.take_jump();
            } else if self.input.take_jump() {
                let was_grounded_for_jump = self.camera.is_grounded;
                self.camera.jump();
                if was_grounded_for_jump && !self.camera.is_grounded {
                    self.audio.play(SoundEffect::Jump);
                }
            }

            // Movement (vertical zeroed in gravity mode internally).
            let prev_position = self.camera.position;
            let (forward, strafe, vertical, w_axis) = self.input.movement_axes();
            let has_movement_input = forward.abs() > 1e-6
                || strafe.abs() > 1e-6
                || vertical.abs() > 1e-6
                || w_axis.abs() > 1e-6;
            if self.sprint_enabled && !has_movement_input {
                self.sprint_enabled = false;
                eprintln!("Sprint: off");
            }
            let move_speed = if self.sprint_enabled && forward > 0.0 {
                self.move_speed * SPRINT_SPEED_MULTIPLIER
            } else {
                self.move_speed
            };
            match self.control_scheme {
                ControlScheme::IntuitiveUpright => {
                    self.camera
                        .apply_movement_upright(forward, strafe, vertical, w_axis, dt, move_speed);
                }
                ControlScheme::LookTransport | ControlScheme::RotorFree => {
                    self.camera.apply_movement_look_frame(
                        forward, strafe, vertical, w_axis, dt, move_speed,
                    );
                }
                ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                    self.camera
                        .apply_movement(forward, strafe, vertical, w_axis, dt, move_speed);
                }
            }

            // Apply gravity physics (no-op while flying), then always resolve voxel collisions.
            self.camera.update_physics(dt);
            let (resolved_pos, grounded) = self.scene.resolve_player_collision(
                prev_position,
                self.camera.position,
                &mut self.camera.velocity_y,
            );
            self.camera.position = resolved_pos;
            self.camera.is_grounded = grounded;

            if self.camera.is_grounded && !self.was_grounded_last_frame && !self.camera.is_flying {
                self.audio.play(SoundEffect::Land);
            }
            let moved_dx = self.camera.position[0] - prev_position[0];
            let moved_dz = self.camera.position[2] - prev_position[2];
            let moved_dw = self.camera.position[3] - prev_position[3];
            let moved_xzw =
                (moved_dx * moved_dx + moved_dz * moved_dz + moved_dw * moved_dw).sqrt();
            let moved_speed_xzw = if dt > 1e-5 { moved_xzw / dt } else { 0.0 };
            if self.camera.is_grounded
                && !self.camera.is_flying
                && has_movement_input
                && moved_speed_xzw > FOOTSTEP_MIN_XZW_SPEED
            {
                self.footstep_distance_accum += moved_xzw;
                let stride = if self.sprint_enabled {
                    FOOTSTEP_DISTANCE_SPRINT
                } else {
                    FOOTSTEP_DISTANCE_WALK
                };
                while self.footstep_distance_accum >= stride {
                    let intensity = (moved_speed_xzw / (self.move_speed * SPRINT_SPEED_MULTIPLIER))
                        .clamp(0.65, 1.25);
                    self.audio.play_scaled(SoundEffect::Footstep, intensity);
                    self.footstep_distance_accum -= stride;
                }
            } else {
                self.footstep_distance_accum = 0.0;
            }
            self.was_grounded_last_frame = self.camera.is_grounded;

            // Block edit actions.
            let look_dir_for_edit = self.current_look_direction();
            if self.mouse_grabbed {
                let pick_requested = self.input.take_pick_material();
                let remove_requested = self.input.take_remove_block();
                let place_requested = self.input.take_place_block();
                if pick_requested {
                    if let Some([x, y, z, w]) = self
                        .scene
                        .block_edit_targets(self.camera.position, look_dir_for_edit, edit_reach)
                        .hit_voxel
                    {
                        let material = self.scene.world.get_voxel(x, y, z, w).0;
                        let clamped = material
                            .clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX);
                        self.place_material = clamped;
                        self.hotbar_slots[self.hotbar_selected_index] = clamped;
                        eprintln!(
                            "Picked voxel material {} ({}) from ({x}, {y}, {z}, {w})",
                            clamped,
                            materials::material_name(clamped),
                        );
                    }
                }
                if remove_requested || place_requested {
                    if remove_requested {
                        if let Some([x, y, z, w]) = self.scene.remove_block_along_ray(
                            self.camera.position,
                            look_dir_for_edit,
                            edit_reach,
                        ) {
                            eprintln!("Removed voxel at ({x}, {y}, {z}, {w})");
                            self.audio.play(SoundEffect::Break);
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
                            self.audio.play(SoundEffect::Place);
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
                self.input.take_pick_material();
            }
        } else {
            self.input.take_jump();
            self.input.take_remove_block();
            self.input.take_place_block();
            self.input.take_pick_material();
            self.footstep_distance_accum = 0.0;
            self.was_grounded_last_frame = self.camera.is_grounded;
        }
        if let Some(scenario_index) = self
            .perf_suite_state
            .as_ref()
            .map(|state| state.scenario_index)
        {
            self.set_perf_suite_camera_pose(scenario_index);
        }

        let look_dir = self.current_look_direction();
        self.send_multiplayer_player_update(now, look_dir);
        let preview_elapsed = now - self.start_time;
        let preview_time_s = preview_elapsed.as_secs_f32();
        let preview_time_ticks_ms = preview_elapsed.as_millis() as u32;
        let aspect = self
            .rcx
            .as_ref()
            .and_then(|rcx| rcx.window.as_ref())
            .map(|window| {
                let size = window.inner_size();
                size.width.max(1) as f32 / size.height.max(1) as f32
            })
            .unwrap_or_else(|| self.args.width.max(1) as f32 / self.args.height.max(1) as f32);
        let preview_instance = build_place_preview_instance(
            &self.camera,
            self.place_material,
            preview_time_s,
            self.control_scheme,
            aspect,
        );

        // Build view matrix and scene
        let view_matrix = self.current_view_matrix();
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

        // WAILA: show targeted block name below crosshair
        let waila_text = if !self.menu_open && self.mouse_grabbed && !self.args.no_hud {
            let waila_targets =
                self.scene
                    .block_edit_targets(self.camera.position, look_dir, edit_reach);
            if let Some([x, y, z, w]) = waila_targets.hit_voxel {
                let voxel = self.scene.world.get_voxel(x, y, z, w);
                if voxel.0 != 0 {
                    Some(materials::material_name(voxel.0).to_string())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

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
        if self.args.no_hud {
            custom_overlay_lines.clear();
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
        let mut take_screenshot = manual_screenshot || command_screenshot_requested;
        if auto_screenshot && self.args.gpu_screenshot_source == GpuScreenshotSourceArg::Framebuffer
        {
            take_screenshot = true;
        }
        let auto_screenshot = auto_screenshot || command_screenshot_requested;
        if take_screenshot {
            if let Some(parent) = self.args.screenshot_output.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
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
        let egui_paint = if self.args.no_hud {
            None
        } else {
            self.run_egui_frame()
        };
        let mut do_navigation_hud = !self.menu_open && self.info_panel_mode != InfoPanelMode::Off;
        if self.args.no_hud {
            do_navigation_hud = false;
        }
        let hud_readout_mode = if !self.menu_open
            && matches!(
                self.info_panel_mode,
                InfoPanelMode::VectorTable | InfoPanelMode::VectorTable2
            ) {
            HudReadoutMode::CompactVectors
        } else {
            HudReadoutMode::Full
        };
        let hud_rotation_label = self.current_info_hud_text(
            pair,
            look_dir,
            edit_reach,
            highlight_mode,
            &vte_sweep_status,
            hud_target_hit_voxel,
            hud_target_hit_face,
        );
        let hud_rotation_label = if self.args.no_hud {
            None
        } else {
            hud_rotation_label
        };

        let render_options = RenderOptions {
            do_raster: true,
            render_backend: backend,
            vte_max_trace_steps: self.vte_max_trace_steps,
            vte_max_trace_distance: self.vte_max_trace_distance,
            vte_lod_near_max_distance: self.vte_lod_near_max_distance,
            vte_lod_mid_max_distance: self.vte_lod_mid_max_distance,
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
            vte_highlight_hit_voxel: if self.args.no_hud {
                None
            } else {
                vte_highlight_hit_voxel
            },
            vte_highlight_place_voxel: if self.args.no_hud {
                None
            } else {
                vte_highlight_place_voxel
            },
            do_navigation_hud,
            custom_overlay_lines,
            take_framebuffer_screenshot: take_screenshot,
            prepare_render_screenshot: auto_screenshot,
            hud_readout_mode,
            hud_rotation_label,
            hud_target_hit_voxel,
            hud_target_hit_face,
            hud_player_tags: if self.args.no_hud {
                Vec::new()
            } else {
                hud_player_tags
            },
            waila_text,
            egui_paint,
            ..Default::default()
        };

        let frame_params = FrameParams {
            view_matrix,
            time_ticks_ms: preview_time_ticks_ms,
            focal_length_xy: self.focal_length_xy,
            focal_length_zw: self.focal_length_zw,
            render_options,
        };

        if backend == RenderBackend::VoxelTraversal {
            let mut vte_non_voxel_instances = Vec::new();
            vte_non_voxel_instances.extend(self.remote_player_instances(preview_time_s));
            vte_non_voxel_instances.extend(self.remote_entity_instances());
            let mut preview_overlay_instances: &[common::ModelInstance] = &[];
            if self.vte_overlay_raster_enabled {
                preview_overlay_instances = std::slice::from_ref(&preview_instance);
            } else {
                // Default: render held-block preview through Stage A entity path for full quality.
                vte_non_voxel_instances.push(preview_instance);
            }
            let voxel_frame = self.scene.build_voxel_frame_data(
                self.camera.position,
                look_dir,
                self.vte_lod_near_max_distance,
                self.vte_lod_mid_max_distance,
                self.vte_max_trace_distance,
            );

            // If we are playing without a live server connection, do not keep the loading gate up.
            if !self.world_ready
                && self.app_state == AppState::Playing
                && self.multiplayer.is_none()
            {
                self.world_ready = true;
                eprintln!("World ready: no multiplayer connection");
            }
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_frame.as_input(),
                &vte_non_voxel_instances,
                preview_overlay_instances,
            );
        } else {
            let remote_instances = self.remote_player_instances(preview_time_s);
            let entity_instances = self.remote_entity_instances();
            self.scene.update_surfaces_if_dirty();
            let instances = self.scene.build_instances(self.camera.position);
            let mut render_instances = Vec::with_capacity(
                instances.len() + remote_instances.len() + entity_instances.len() + 1,
            );
            render_instances.extend_from_slice(instances);
            render_instances.extend(remote_instances);
            render_instances.extend(entity_instances);
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
            if let Some(parent) = self.args.screenshot_output.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = std::fs::create_dir_all(parent);
                }
            }
            match self.args.gpu_screenshot_source {
                GpuScreenshotSourceArg::RenderBuffer => {
                    self.rcx.as_mut().unwrap().save_rendered_frame_png(
                        self.args
                            .screenshot_output
                            .to_str()
                            .unwrap_or("frames/gpu_render.png"),
                    );
                }
                GpuScreenshotSourceArg::Framebuffer => {
                    if let Some(src_path) = latest_framebuffer_screenshot_path() {
                        let webp_path = self.args.screenshot_output.with_extension("webp");
                        if let Err(err) = std::fs::copy(&src_path, &webp_path) {
                            eprintln!(
                                "Failed to copy framebuffer screenshot {} -> {}: {}",
                                src_path.display(),
                                webp_path.display(),
                                err
                            );
                        }
                        match image::open(&src_path) {
                            Ok(img) => {
                                if let Err(err) = img.save(&self.args.screenshot_output) {
                                    eprintln!(
                                        "Failed to save {}: {err}",
                                        self.args.screenshot_output.display()
                                    );
                                } else {
                                    println!(
                                        "Saved PNG to {}",
                                        self.args.screenshot_output.display()
                                    );
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
            // Write JSON sidecar metadata
            {
                let json_path = self.args.screenshot_output.with_extension("json");
                let window_size = self
                    .rcx
                    .as_ref()
                    .and_then(|rcx| rcx.window.as_ref())
                    .map(|w| {
                        let size = w.inner_size();
                        [size.width, size.height]
                    })
                    .unwrap_or([0, 0]);

                let metadata = serde_json::json!({
                    "render_width": self.args.width,
                    "render_height": self.args.height,
                    "render_layers": self.args.layers,
                    "window_width": window_size[0],
                    "window_height": window_size[1],
                    "camera_position": [
                        self.camera.position[0],
                        self.camera.position[1],
                        self.camera.position[2],
                        self.camera.position[3],
                    ],
                    "camera_angles_rad": [
                        self.camera.yaw,
                        self.camera.pitch,
                        self.camera.xw_angle,
                        self.camera.zw_angle,
                    ],
                    "backend": format!("{:?}", self.args.backend),
                    "vte_display_mode": format!("{:?}", self.args.vte_display_mode),
                    "gpu_screenshot_source": format!("{:?}", self.args.gpu_screenshot_source),
                    "world_file": if self.args.load_world {
                        Some(self.args.world_file.to_string_lossy().into_owned())
                    } else {
                        None
                    },
                    "scene": format!("{:?}", self.args.scene),
                    "no_hud": self.args.no_hud,
                });

                match std::fs::write(&json_path, serde_json::to_string_pretty(&metadata).unwrap()) {
                    Ok(()) => println!("Saved metadata to {}", json_path.display()),
                    Err(err) => {
                        eprintln!("Failed to save metadata {}: {}", json_path.display(), err)
                    }
                }
            }
            self.should_exit_after_render = true;
        }

        self.advance_vte_runtime_sweep_after_frame();
        if self.perf_suite_active() {
            self.advance_perf_suite_after_frame(frame_start);
        } else {
            self.record_runtime_profile_sample(frame_start);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window_attrs = {
            let attrs = Window::default_attributes();
            // When explicit window size given, use it.
            // When --gpu-screenshot active but no explicit window size, match render buffer.
            let (w, h) = if self.args.window_width.is_some() || self.args.window_height.is_some() {
                (
                    self.args.window_width.unwrap_or(self.args.width),
                    self.args.window_height.unwrap_or(self.args.height),
                )
            } else if self.args.gpu_screenshot {
                (self.args.width, self.args.height)
            } else {
                (0, 0) // sentinel: use default
            };
            if w > 0 && h > 0 {
                attrs.with_inner_size(LogicalSize::new(w, h))
            } else {
                attrs
            }
        };
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        self.egui_winit_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        ));

        if self.app_state == AppState::Playing && !self.perf_suite_active() {
            self.grab_mouse(&window);
        }
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
            Some(window.clone()),
            [self.args.width, self.args.height, self.args.layers],
            pixel_storage_layers,
        ));

        // Generate material icon sprite sheet and upload to GPU
        if self.material_icon_sheet.is_none() {
            let start = Instant::now();
            let model_tets = higher_dimension_playground::render::generate_tesseract_tetrahedrons();
            let sheet = material_icons::generate_material_icon_sheet(&model_tets);
            eprintln!(
                "Generated material icon sprite sheet ({}x{}) in {:.2}s",
                sheet.width,
                sheet.height,
                start.elapsed().as_secs_f32()
            );
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.upload_material_icons_texture(
                    self.queue.clone(),
                    sheet.width,
                    sheet.height,
                    &sheet.pixels,
                );
            }
            // Use User(1) as the egui texture ID for material icons
            self.material_icons_texture_id = Some(egui::TextureId::User(1));
            self.material_icon_sheet = Some(sheet);
        }

        self.last_frame = Instant::now();
        if self.perf_suite_active() {
            self.begin_perf_suite_phase(true);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        let show_egui_overlay = self.app_state == AppState::MainMenu
            || self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open;
        let egui_consumed = if let (Some(egui_state), Some(window)) =
            (self.egui_winit_state.as_mut(), window.as_ref())
        {
            show_egui_overlay && egui_state.on_window_event(window, &event).consumed
        } else {
            false
        };
        let perf_suite_input_locked = self.perf_suite_active();

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
                if perf_suite_input_locked {
                    return;
                }
                if !egui_consumed {
                    self.input.handle_key_event(&event);
                }

                if is_escape_pressed(&event) {
                    if self.app_state == AppState::MainMenu {
                        // In main menu, Escape goes back or quits
                        if self.main_menu_page != MainMenuPage::Root {
                            self.main_menu_page = MainMenuPage::Root;
                            self.main_menu_connect_error = None;
                        } else {
                            event_loop.exit();
                        }
                    } else if self.teleport_dialog_open {
                        self.teleport_dialog_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.inventory_open {
                        self.inventory_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.menu_open {
                        self.menu_open = false;
                        if let Some(window) = window.as_ref() {
                            self.grab_mouse(window);
                        }
                    } else if self.mouse_grabbed {
                        if let Some(window) = window.as_ref() {
                            self.release_mouse(window);
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
                    if perf_suite_input_locked {
                        return;
                    }
                    if state.is_pressed() {
                        if self.app_state == AppState::MainMenu {
                            // Don't grab mouse in main menu -- egui handles clicks
                        } else if self.mouse_grabbed {
                            if !egui_consumed {
                                self.input.handle_mouse_button(button, state);
                            }
                        } else if self.teleport_dialog_open && !egui_consumed {
                            // Click outside teleport dialog — close it and grab mouse
                            self.toggle_teleport_dialog();
                        } else if !self.menu_open
                            && !self.inventory_open
                            && !self.teleport_dialog_open
                        {
                            if let Some(window) = window.as_ref() {
                                self.grab_mouse(window);
                                self.menu_open = false;
                            }
                        }
                    }
                }
                MouseButton::Middle
                | MouseButton::Right
                | MouseButton::Back
                | MouseButton::Forward => {
                    if perf_suite_input_locked {
                        return;
                    }
                    if !egui_consumed {
                        self.input.handle_mouse_button(button, state);
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                if perf_suite_input_locked {
                    return;
                }
                if !egui_consumed {
                    let y = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                    };
                    self.input.handle_scroll(y);
                }
            }
            WindowEvent::Focused(false) => {
                if perf_suite_input_locked {
                    return;
                }
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
        if self.perf_suite_active() {
            return;
        }
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
