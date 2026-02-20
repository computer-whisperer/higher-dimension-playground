mod app_bootstrap;
mod app_console;
mod app_controls;
mod app_events;
mod app_gameplay_loop;
mod app_helpers;
mod app_hud;
mod app_main_menu;
mod app_multiplayer;
mod app_perf;
mod app_runtime;
mod app_settings;
mod app_ui;
mod audio;
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
    EguiPaintData, EguiPaintMesh, EguiPaintVertex, EguiTextureSlot, EguiTextureUpdate, FrameParams,
    HudPlayerTag, HudReadoutMode, RenderBackend, RenderContext, RenderOptions, TetraFrameInput,
    VteDisplayMode,
};
use higher_dimension_playground::vulkan_setup::vulkan_setup;
use std::collections::{HashMap, VecDeque};
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

use app_bootstrap::{parse_commands, run_cpu_render};
use app_helpers::*;
use audio::{AudioEngine, SoundEffect};
use camera::{Camera4D, PLAYER_HEIGHT};
use input::{ControlScheme, InputState, RotationPair};
use multiplayer::{ClientMessage as MultiplayerClientMessage, MultiplayerClient, MultiplayerEvent};
use polychora::shared::spatial::Aabb4i;
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
const WORLD_FILE_DEFAULT: &str = "saves/world";
const VTE_SWEEP_SAMPLE_FRAMES: usize = 120;
const VTE_SWEEP_INCLUDE_NO_NON_VOXEL_ENV: &str = "R4D_VTE_SWEEP_INCLUDE_NO_NON_VOXEL_INSTANCES";
const VTE_SWEEP_INCLUDE_NO_ENTITIES_ENV_LEGACY: &str = "R4D_VTE_SWEEP_INCLUDE_NO_ENTITIES";
const VTE_OVERLAY_RASTER_ENV: &str = "R4D_VTE_OVERLAY_RASTER";
const CLIENT_REGION_TREE_BOUNDS_DIAG_ENV: &str = "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG";
const CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_NODES_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_NODES";
const CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY";
const CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS_ENV: &str = "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS";
const CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS";
const CLIENT_REGION_TREE_COMPARE_DIAG_ENV: &str = "R4D_CLIENT_REGION_TREE_COMPARE_DIAG";
const CLIENT_REGION_TREE_COMPARE_DIAG_MAX_CHUNKS_ENV: &str =
    "R4D_CLIENT_REGION_TREE_COMPARE_DIAG_MAX_CHUNKS";
const CLIENT_REGION_TREE_COMPARE_DIAG_LOG_INTERVAL_ENV: &str =
    "R4D_CLIENT_REGION_TREE_COMPARE_DIAG_LOG_INTERVAL";
// Tags held-block preview instances so shaders can apply preview-only shading boosts.
const PREVIEW_MATERIAL_FLAG: u32 = 0x8000_0000;
const FOCAL_LENGTH_MIN: f32 = 0.20;
const FOCAL_LENGTH_MAX: f32 = 4.00;
const ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN: f32 = 0.0;
const ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX: f32 = 1.0;
const ZW_ANGLE_COLOR_SHIFT_STRENGTH_DEFAULT: f32 = 0.35;
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
const MULTIPLAYER_PENDING_PLAYER_MODIFIER_MAX: usize = 128;
const MULTIPLAYER_PLAYER_MODIFIER_MAX_TRANSLATION: f32 = 12.0;
const MULTIPLAYER_PLAYER_MODIFIER_MAX_VELOCITY_Y_DELTA: f32 = 16.0;
const MULTIPLAYER_PLAYER_MODIFIER_IMPULSE_GAIN_XZW: f32 = 18.0;
const MULTIPLAYER_PLAYER_MODIFIER_DECAY_HZ: f32 = 2.5;
const MULTIPLAYER_PLAYER_MODIFIER_MAX_EXTERNAL_SPEED_XZW: f32 = 60.0;
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SingleplayerWorldTypeArg {
    FlatFloor,
    MassivePlatforms,
}

impl SingleplayerWorldTypeArg {
    fn to_runtime(self) -> polychora::server::WorldGeneratorKind {
        match self {
            SingleplayerWorldTypeArg::FlatFloor => polychora::server::WorldGeneratorKind::FlatFloor,
            SingleplayerWorldTypeArg::MassivePlatforms => {
                polychora::server::WorldGeneratorKind::MassivePlatforms
            }
        }
    }
}

#[derive(Debug, Clone)]
enum AutoCommand {
    Press(KeyCode),
    Wait(u32),
    Screenshot,
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

    /// Apply an optional red/blue tint across Z/W layers to make hidden-angle sampling visible.
    #[arg(long, action = ArgAction::Set, default_value_t = true)]
    zw_angle_color_shift: bool,

    /// Strength of the optional Z/W angle red/blue color shift.
    #[arg(long, default_value_t = ZW_ANGLE_COLOR_SHIFT_STRENGTH_DEFAULT)]
    zw_angle_color_shift_strength: f32,

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

    /// World generator used when creating/loading integrated singleplayer worlds without metadata.
    #[arg(long, value_enum, default_value_t = SingleplayerWorldTypeArg::FlatFloor)]
    singleplayer_world_type: SingleplayerWorldTypeArg,

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
    let mut args = Args::parse();
    let settings_file_path = app_settings::settings_file_path();
    let cli_overrides = app_settings::CliOverrides::from_process_args();
    let loaded_settings = app_settings::load_settings(&settings_file_path);
    if let Some(settings) = loaded_settings.as_ref() {
        app_settings::apply_settings_to_args(&mut args, settings, cli_overrides);
    }
    let initial_singleplayer_world_generator = args.singleplayer_world_type.to_runtime();

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
    let scene = Scene::new(scene_preset);
    if args.load_world && !start_with_integrated_singleplayer {
        eprintln!(
            "--load-world is only supported through the integrated server path; skipping client-side world file load."
        );
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
        let runtime_config = build_singleplayer_runtime_config(
            &args,
            world_file.clone(),
            initial_singleplayer_world_generator,
        );
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
    let initial_zw_angle_color_shift_enabled = args.zw_angle_color_shift;
    let initial_zw_angle_color_shift_strength = args.zw_angle_color_shift_strength.clamp(
        ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
        ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
    );
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
        zw_angle_color_shift_enabled: initial_zw_angle_color_shift_enabled,
        zw_angle_color_shift_strength: initial_zw_angle_color_shift_strength,
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
        dev_console_open: false,
        dev_console_input: String::new(),
        dev_console_log: VecDeque::new(),
        dev_console_focus_input: false,
        controls_dialog_open: false,
        menu_open: false,
        menu_selection: 0,
        egui_ctx: egui::Context::default(),
        egui_winit_state: None,
        material_icon_sheet: None,
        material_icons_texture_id: None,
        multiplayer,
        multiplayer_self_id: None,
        multiplayer_last_world_request_center_chunk: None,
        multiplayer_last_world_request_bounds: None,
        multiplayer_stream_tree_diag: polychora::shared::region_tree::RegionChunkTree::new(),
        multiplayer_stream_tree_diag_enabled: env_flag_enabled(CLIENT_REGION_TREE_BOUNDS_DIAG_ENV),
        multiplayer_stream_tree_diag_max_nodes: env_usize_or(
            CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_NODES_ENV,
            192,
        ),
        multiplayer_stream_tree_diag_non_empty_only: env_flag_enabled_or(
            CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY_ENV,
            true,
        ),
        multiplayer_stream_tree_diag_labels_enabled: env_flag_enabled_or(
            CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS_ENV,
            true,
        ),
        multiplayer_stream_tree_diag_max_labels: env_usize_or(
            CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS_ENV,
            28,
        )
        .clamp(1, 256),
        multiplayer_stream_tree_compare_diag_enabled: env_flag_enabled(
            CLIENT_REGION_TREE_COMPARE_DIAG_ENV,
        ),
        multiplayer_stream_tree_compare_diag_max_chunks: env_usize_or(
            CLIENT_REGION_TREE_COMPARE_DIAG_MAX_CHUNKS_ENV,
            64,
        )
        .clamp(1, 2048),
        multiplayer_stream_tree_compare_diag_log_interval: env_usize_or(
            CLIENT_REGION_TREE_COMPARE_DIAG_LOG_INTERVAL_ENV,
            120,
        )
        .max(1),
        multiplayer_stream_tree_compare_diag_last_hash: None,
        multiplayer_stream_tree_compare_diag_frame_counter: 0,
        pending_player_movement_modifiers: VecDeque::new(),
        player_modifier_external_velocity: [0.0; 4],
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
        main_menu_new_world_generator: initial_singleplayer_world_generator,
        main_menu_connect_error: None,
        main_menu_migration_status: None,
        main_menu_migrate_trim_input: "saves/world.v4dw".to_string(),
        main_menu_migrate_trim_output: "saves/world.migrated.v4dw".to_string(),
        main_menu_migrate_trim_keep_min: "0 -2 -2 -2".to_string(),
        main_menu_migrate_trim_keep_max: "0 0 2 2".to_string(),
        main_menu_migrate_v3_input: "saves/world-v3".to_string(),
        main_menu_migrate_v3_output: "saves/world-migrated-v4".to_string(),
        main_menu_migrate_v3_overwrite: false,
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
        settings_file_path: settings_file_path.clone(),
        settings_last_saved: app_settings::PersistedSettings::default(),
        settings_last_save_attempt: Instant::now(),
    };

    if let Some(settings) = loaded_settings.as_ref() {
        app.apply_runtime_settings(settings);
        eprintln!(
            "Loaded persisted settings from {}",
            settings_file_path.display()
        );
    }
    app.settings_last_saved = app.capture_persisted_settings();
    app.settings_last_save_attempt = Instant::now();

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
    if app.zw_angle_color_shift_enabled {
        eprintln!(
            "ZW angle color shift enabled via --zw-angle-color-shift=true (strength={:.2})",
            app.zw_angle_color_shift_strength
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
    if app.multiplayer_stream_tree_diag_enabled {
        eprintln!(
            "Client region-tree bounds diagnostics enabled via {} (max nodes {}, non_empty_only={} via {}, labels={} via {}, max_labels={} via {}).",
            CLIENT_REGION_TREE_BOUNDS_DIAG_ENV,
            app.multiplayer_stream_tree_diag_max_nodes,
            app.multiplayer_stream_tree_diag_non_empty_only,
            CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY_ENV,
            app.multiplayer_stream_tree_diag_labels_enabled,
            CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS_ENV,
            app.multiplayer_stream_tree_diag_max_labels,
            CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS_ENV
        );
    }
    if app.multiplayer_stream_tree_compare_diag_enabled {
        eprintln!(
            "Client region-tree compare diagnostics enabled via {} (max mismatch chunks {}, log interval {}f).",
            CLIENT_REGION_TREE_COMPARE_DIAG_ENV,
            app.multiplayer_stream_tree_compare_diag_max_chunks,
            app.multiplayer_stream_tree_compare_diag_log_interval
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

#[derive(Copy, Clone, PartialEq, Eq)]
enum AppState {
    MainMenu,
    Playing,
}

#[derive(Clone, PartialEq)]
enum MainMenuPage {
    Root,
    Singleplayer,
    SingleplayerMigrations,
    SingleplayerMigrationLegacyTrim,
    SingleplayerMigrationV3ToV4,
    Multiplayer,
}

enum MainMenuTransition {
    NewWorld(polychora::server::WorldGeneratorKind),
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
    zw_angle_color_shift_enabled: bool,
    zw_angle_color_shift_strength: f32,
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
    dev_console_open: bool,
    dev_console_input: String,
    dev_console_log: VecDeque<String>,
    dev_console_focus_input: bool,
    controls_dialog_open: bool,
    menu_open: bool,
    menu_selection: usize,
    egui_ctx: egui::Context,
    egui_winit_state: Option<egui_winit::State>,
    material_icon_sheet: Option<material_icons::MaterialIconSheet>,
    material_icons_texture_id: Option<egui::TextureId>,
    multiplayer: Option<MultiplayerClient>,
    multiplayer_self_id: Option<u64>,
    multiplayer_last_world_request_center_chunk: Option<[i32; 4]>,
    multiplayer_last_world_request_bounds: Option<Aabb4i>,
    multiplayer_stream_tree_diag: polychora::shared::region_tree::RegionChunkTree,
    multiplayer_stream_tree_diag_enabled: bool,
    multiplayer_stream_tree_diag_max_nodes: usize,
    multiplayer_stream_tree_diag_non_empty_only: bool,
    multiplayer_stream_tree_diag_labels_enabled: bool,
    multiplayer_stream_tree_diag_max_labels: usize,
    multiplayer_stream_tree_compare_diag_enabled: bool,
    multiplayer_stream_tree_compare_diag_max_chunks: usize,
    multiplayer_stream_tree_compare_diag_log_interval: usize,
    multiplayer_stream_tree_compare_diag_last_hash: Option<u64>,
    multiplayer_stream_tree_compare_diag_frame_counter: u64,
    pending_player_movement_modifiers: VecDeque<PendingPlayerMovementModifier>,
    player_modifier_external_velocity: [f32; 4],
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
    main_menu_new_world_generator: polychora::server::WorldGeneratorKind,
    main_menu_connect_error: Option<String>,
    main_menu_migration_status: Option<String>,
    main_menu_migrate_trim_input: String,
    main_menu_migrate_trim_output: String,
    main_menu_migrate_trim_keep_min: String,
    main_menu_migrate_trim_keep_max: String,
    main_menu_migrate_v3_input: String,
    main_menu_migrate_v3_output: String,
    main_menu_migrate_v3_overwrite: bool,
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
    settings_file_path: PathBuf,
    settings_last_saved: app_settings::PersistedSettings,
    settings_last_save_attempt: Instant,
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
    owner_client_id: Option<u64>,
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
    velocity: [f32; 4],
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
struct PendingPlayerMovementModifier {
    delta_position: [f32; 4],
    delta_velocity_y: f32,
    source_entity_id: Option<u64>,
}
