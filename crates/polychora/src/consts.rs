use std::time::Duration;

use super::camera::PLAYER_HEIGHT;

// ---------------------------------------------------------------------------
// Gameplay
// ---------------------------------------------------------------------------

pub(crate) const MOUSE_SENSITIVITY: f32 = 0.002;
pub(crate) const BLOCK_EDIT_REACH_DEFAULT: f32 = 8.0;
pub(crate) const BLOCK_EDIT_REACH_MIN: f32 = 1.0;
pub(crate) const BLOCK_EDIT_REACH_MAX: f32 = 48.0;
pub(crate) const SPRINT_SPEED_MULTIPLIER: f32 = 1.8;
pub(crate) const FOOTSTEP_DISTANCE_WALK: f32 = 3.50;
pub(crate) const FOOTSTEP_DISTANCE_SPRINT: f32 = 2.40;
pub(crate) const FOOTSTEP_MIN_XZW_SPEED: f32 = 0.55;
pub(crate) const REMOTE_FOOTSTEP_MAX_DISTANCE: f32 = 36.0;
pub(crate) const REMOTE_FOOTSTEP_MIN_XZW_SPEED: f32 = 0.40;
pub(crate) const REMOTE_FOOTSTEP_MAX_VERTICAL_SPEED: f32 = 2.4;
pub(crate) const REMOTE_FOOTSTEP_MAX_NETWORK_AGE_S: f32 = 0.35;
pub(crate) const REMOTE_FOOTSTEP_MAX_PER_FRAME: usize = 6;

// ---------------------------------------------------------------------------
// Rendering / VTE
// ---------------------------------------------------------------------------

/// Tags held-block preview instances so shaders can apply preview-only shading boosts.
pub(crate) const PREVIEW_MATERIAL_FLAG: u32 = 0x8000_0000;
pub(crate) const FOCAL_LENGTH_MIN: f32 = 0.20;
pub(crate) const FOCAL_LENGTH_MAX: f32 = 4.00;
pub(crate) const ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN: f32 = 0.0;
pub(crate) const ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX: f32 = 1.0;
pub(crate) const ZW_ANGLE_COLOR_SHIFT_STRENGTH_DEFAULT: f32 = 0.35;
pub(crate) const VTE_TRACE_DISTANCE_MIN: f32 = 10.0;
pub(crate) const VTE_TRACE_DISTANCE_MAX: f32 = 4096.0;
pub(crate) const VTE_INTEGRAL_SKY_SCALE_MIN: f32 = 0.0;
pub(crate) const VTE_INTEGRAL_SKY_SCALE_MAX: f32 = 2.0;
pub(crate) const VTE_INTEGRAL_HIT_EMISSIVE_MIN: f32 = 0.0;
pub(crate) const VTE_INTEGRAL_HIT_EMISSIVE_MAX: f32 = 0.20;
pub(crate) const VTE_INTEGRAL_LOG_MERGE_K_MIN: f32 = 0.0;
pub(crate) const VTE_INTEGRAL_LOG_MERGE_K_MAX: f32 = 64.0;
pub(crate) const VTE_TRACE_STEPS_MIN: u32 = 16;
pub(crate) const VTE_TRACE_STEPS_MAX: u32 = 4096;

// ---------------------------------------------------------------------------
// Multiplayer & networking
// ---------------------------------------------------------------------------

pub(crate) const MULTIPLAYER_DEFAULT_PORT: u16 = 4000;
pub(crate) const MULTIPLAYER_PLAYER_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const MULTIPLAYER_PENDING_PLAYER_MODIFIER_MAX: usize = 128;
pub(crate) const MULTIPLAYER_PLAYER_MODIFIER_MAX_TRANSLATION: f32 = 12.0;
pub(crate) const MULTIPLAYER_PLAYER_MODIFIER_MAX_VELOCITY_Y_DELTA: f32 = 16.0;
pub(crate) const MULTIPLAYER_PLAYER_MODIFIER_IMPULSE_GAIN_XZW: f32 = 18.0;
pub(crate) const MULTIPLAYER_PLAYER_MODIFIER_DECAY_HZ: f32 = 2.5;
pub(crate) const MULTIPLAYER_PLAYER_MODIFIER_MAX_EXTERNAL_SPEED_XZW: f32 = 60.0;
pub(crate) const CLIENT_PROFILE_REPORT_INTERVAL: Duration = Duration::from_secs(2);

// ---------------------------------------------------------------------------
// Avatars & remote players/entities
// ---------------------------------------------------------------------------

pub(crate) const AVATAR_MATERIAL_ID: u32 = 21;
pub(crate) const AVATAR_FORWARD_FRAGMENT_COUNT: usize = 4;
pub(crate) const REMOTE_AVATAR_PART_COUNT_ESTIMATE: usize = 7 + AVATAR_FORWARD_FRAGMENT_COUNT;
pub(crate) const AVATAR_THICKNESS_SCALE: f32 = 1.30;
pub(crate) const AVATAR_FORWARD_FRAGMENT_LENGTH_SCALE: f32 = 0.50;
pub(crate) const REMOTE_PLAYER_POSITION_SMOOTH_HZ: f32 = 12.0;
pub(crate) const REMOTE_PLAYER_LOOK_SMOOTH_HZ: f32 = 16.0;
pub(crate) const REMOTE_PLAYER_PREDICTION_LEAD_S: f32 = 0.05;
pub(crate) const REMOTE_PLAYER_MAX_PREDICTION_S: f32 = 0.22;
pub(crate) const REMOTE_PLAYER_TELEPORT_SNAP_DISTANCE: f32 = 8.0;
pub(crate) const REMOTE_PLAYER_MAX_PREDICTED_SPEED: f32 = 24.0;
pub(crate) const REMOTE_PLAYER_TAG_FOV_DOT_MIN: f32 = 0.16;
pub(crate) const REMOTE_PLAYER_TAG_MAX_COUNT: usize = 32;
pub(crate) const REMOTE_ENTITY_POSITION_SMOOTH_HZ: f32 = 12.0;
pub(crate) const REMOTE_ENTITY_ORIENTATION_SMOOTH_HZ: f32 = 16.0;
pub(crate) const REMOTE_ENTITY_TELEPORT_SNAP_DISTANCE: f32 = 20.0;

// ---------------------------------------------------------------------------
// Menu orbit camera
// ---------------------------------------------------------------------------

pub(crate) const MENU_ORBIT_CENTER: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
pub(crate) const MENU_ORBIT_RADIUS_XZ: f32 = 16.0;
pub(crate) const MENU_ORBIT_RADIUS_W: f32 = 7.0;
pub(crate) const MENU_ORBIT_HEIGHT_BASE: f32 = PLAYER_HEIGHT + 1.2;
pub(crate) const MENU_ORBIT_HEIGHT_BOB: f32 = 0.8;
pub(crate) const MENU_ORBIT_RATE_XZ: f32 = 0.23;
pub(crate) const MENU_ORBIT_RATE_W: f32 = 0.17;
pub(crate) const MENU_ORBIT_RATE_Y: f32 = 0.11;
pub(crate) const MENU_ORBIT_TARGET_Y_OFFSET: f32 = 0.6;

// ---------------------------------------------------------------------------
// Environment variable names
// ---------------------------------------------------------------------------

pub(crate) const WORLD_FILE_DEFAULT: &str = "saves/world";
pub(crate) const VTE_OVERLAY_RASTER_ENV: &str = "R4D_VTE_OVERLAY_RASTER";
pub(crate) const CLIENT_REGION_TREE_BOUNDS_DIAG_ENV: &str = "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG";
pub(crate) const CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_NODES_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_NODES";
pub(crate) const CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_NON_EMPTY_ONLY";
pub(crate) const CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_LABELS";
pub(crate) const CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS_ENV: &str =
    "R4D_CLIENT_REGION_TREE_BOUNDS_DIAG_MAX_LABELS";
pub(crate) const CLIENT_REGION_TREE_COMPARE_DIAG_ENV: &str = "R4D_CLIENT_REGION_TREE_COMPARE_DIAG";
pub(crate) const CLIENT_REGION_TREE_COMPARE_DIAG_MAX_CHUNKS_ENV: &str =
    "R4D_CLIENT_REGION_TREE_COMPARE_DIAG_MAX_CHUNKS";
pub(crate) const CLIENT_REGION_TREE_COMPARE_DIAG_LOG_INTERVAL_ENV: &str =
    "R4D_CLIENT_REGION_TREE_COMPARE_DIAG_LOG_INTERVAL";
pub(crate) const CLIENT_WORLD_CHUNK_SAMPLE_DIAG_ENV: &str = "R4D_CLIENT_WORLD_CHUNK_SAMPLE_DIAG";
pub(crate) const CLIENT_WORLD_CHUNK_SAMPLE_DIAG_HISTORY_ENV: &str =
    "R4D_CLIENT_WORLD_CHUNK_SAMPLE_DIAG_HISTORY";
pub(crate) const CLIENT_WORLD_PATCH_FULL_STATS_ENV: &str = "R4D_CLIENT_WORLD_PATCH_FULL_STATS";

// ---------------------------------------------------------------------------
// Perf suite
// ---------------------------------------------------------------------------

#[derive(Copy, Clone)]
pub(crate) struct VteRuntimeProfile {
    pub(crate) label: &'static str,
}

#[derive(Copy, Clone)]
pub(crate) struct PerfSuiteScenario {
    pub(crate) label: &'static str,
    pub(crate) position: [f32; 4],
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) xw_angle: f32,
    pub(crate) zw_angle: f32,
    pub(crate) yw_deviation: f32,
    pub(crate) vte_max_trace_steps: Option<u32>,
    pub(crate) vte_max_trace_distance: Option<f32>,
}

pub(crate) const PERF_SUITE_WARMUP_FRAMES_DEFAULT: u32 = 180;
pub(crate) const PERF_SUITE_SAMPLE_FRAMES_DEFAULT: u32 = 600;
pub(crate) const PERF_SUITE_REPORT_DIR_DEFAULT: &str = "profiles";
pub(crate) const VTE_SWEEP_SAMPLE_FRAMES: usize = 120;

/// (max_trace_steps, max_trace_distance) for perf suite render distance tiers.
pub(crate) const PERF_TIER_LOW: (u32, f32) = (160, 80.0);
pub(crate) const PERF_TIER_DEFAULT: (u32, f32) = (320, 160.0);
pub(crate) const PERF_TIER_HIGH: (u32, f32) = (640, 320.0);

pub(crate) const PERF_SUITE_SETTLE_STABLE_FRAMES: u32 = 60;
pub(crate) const PERF_SUITE_SETTLE_MAX_FRAMES: u32 = 600;

// Base poses for perf suite scenarios (MassivePlatforms seed 1337).
// Each pose is stamped across 3 render distance tiers.
const PERF_POSE_PLATFORM_SURFACE: PerfSuiteScenario = PerfSuiteScenario {
    label: "",
    position: [0.0, 8.0, -24.0, -4.0],
    yaw: 1.25,
    pitch: -0.15,
    xw_angle: 0.0,
    zw_angle: 0.0,
    yw_deviation: 0.0,
    vte_max_trace_steps: None,
    vte_max_trace_distance: None,
};

const PERF_POSE_PLATFORM_4D: PerfSuiteScenario = PerfSuiteScenario {
    label: "",
    position: [0.0, 8.0, -24.0, -4.0],
    yaw: 1.25,
    pitch: -0.15,
    xw_angle: 0.70,
    zw_angle: 0.95,
    yw_deviation: 0.0,
    vte_max_trace_steps: None,
    vte_max_trace_distance: None,
};

const PERF_POSE_OPEN_SKY: PerfSuiteScenario = PerfSuiteScenario {
    label: "",
    position: [0.0, 64.0, 0.0, -4.0],
    yaw: 0.35,
    pitch: -0.70,
    xw_angle: 0.90,
    zw_angle: 0.78,
    yw_deviation: 0.0,
    vte_max_trace_steps: None,
    vte_max_trace_distance: None,
};

const PERF_POSE_CORRIDOR: PerfSuiteScenario = PerfSuiteScenario {
    label: "",
    position: [96.0, 20.0, 96.0, -4.0],
    yaw: -2.35,
    pitch: -0.22,
    xw_angle: -0.45,
    zw_angle: 0.42,
    yw_deviation: 0.0,
    vte_max_trace_steps: None,
    vte_max_trace_distance: None,
};

const PERF_POSE_FAR_OBLIQUE: PerfSuiteScenario = PerfSuiteScenario {
    label: "",
    position: [-64.0, 14.0, 32.0, -4.0],
    yaw: 0.95,
    pitch: -0.08,
    xw_angle: 0.46,
    zw_angle: 0.52,
    yw_deviation: 0.0,
    vte_max_trace_steps: None,
    vte_max_trace_distance: None,
};

pub(crate) const PERF_SUITE_SCENARIOS: [PerfSuiteScenario; 15] = [
    // platform-surface: standing on platform, 3D-like view
    PerfSuiteScenario {
        label: "platform-surface/low",
        vte_max_trace_steps: Some(PERF_TIER_LOW.0),
        vte_max_trace_distance: Some(PERF_TIER_LOW.1),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    PerfSuiteScenario {
        label: "platform-surface/default",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    PerfSuiteScenario {
        label: "platform-surface/high",
        vte_max_trace_steps: Some(PERF_TIER_HIGH.0),
        vte_max_trace_distance: Some(PERF_TIER_HIGH.1),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    // platform-4d: same spot, strong 4D rotation
    PerfSuiteScenario {
        label: "platform-4d/low",
        vte_max_trace_steps: Some(PERF_TIER_LOW.0),
        vte_max_trace_distance: Some(PERF_TIER_LOW.1),
        ..PERF_POSE_PLATFORM_4D
    },
    PerfSuiteScenario {
        label: "platform-4d/default",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_PLATFORM_4D
    },
    PerfSuiteScenario {
        label: "platform-4d/high",
        vte_max_trace_steps: Some(PERF_TIER_HIGH.0),
        vte_max_trace_distance: Some(PERF_TIER_HIGH.1),
        ..PERF_POSE_PLATFORM_4D
    },
    // open-sky: high altitude looking down
    PerfSuiteScenario {
        label: "open-sky/low",
        vte_max_trace_steps: Some(PERF_TIER_LOW.0),
        vte_max_trace_distance: Some(PERF_TIER_LOW.1),
        ..PERF_POSE_OPEN_SKY
    },
    PerfSuiteScenario {
        label: "open-sky/default",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_OPEN_SKY
    },
    PerfSuiteScenario {
        label: "open-sky/high",
        vte_max_trace_steps: Some(PERF_TIER_HIGH.0),
        vte_max_trace_distance: Some(PERF_TIER_HIGH.1),
        ..PERF_POSE_OPEN_SKY
    },
    // corridor: oblique far view between platforms
    PerfSuiteScenario {
        label: "corridor/low",
        vte_max_trace_steps: Some(PERF_TIER_LOW.0),
        vte_max_trace_distance: Some(PERF_TIER_LOW.1),
        ..PERF_POSE_CORRIDOR
    },
    PerfSuiteScenario {
        label: "corridor/default",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_CORRIDOR
    },
    PerfSuiteScenario {
        label: "corridor/high",
        vte_max_trace_steps: Some(PERF_TIER_HIGH.0),
        vte_max_trace_distance: Some(PERF_TIER_HIGH.1),
        ..PERF_POSE_CORRIDOR
    },
    // far-oblique: distant ridge terrain with 4D
    PerfSuiteScenario {
        label: "far-oblique/low",
        vte_max_trace_steps: Some(PERF_TIER_LOW.0),
        vte_max_trace_distance: Some(PERF_TIER_LOW.1),
        ..PERF_POSE_FAR_OBLIQUE
    },
    PerfSuiteScenario {
        label: "far-oblique/default",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_FAR_OBLIQUE
    },
    PerfSuiteScenario {
        label: "far-oblique/high",
        vte_max_trace_steps: Some(PERF_TIER_HIGH.0),
        vte_max_trace_distance: Some(PERF_TIER_HIGH.1),
        ..PERF_POSE_FAR_OBLIQUE
    },
];

pub(crate) const VTE_SWEEP_PROFILES: [VteRuntimeProfile; 2] = [
    VteRuntimeProfile { label: "A bvh" },
    VteRuntimeProfile { label: "B bvh" },
];

pub(crate) const VTE_SWEEP_MODE_LABEL: &str = "bvh";
