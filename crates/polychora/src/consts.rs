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
pub(crate) struct PerfScenarioSetup {
    pub(crate) entity_spawns: &'static [(&'static str, [f32; 4])],
    pub(crate) explosions: &'static [([f32; 4], i32)],
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
    pub(crate) setup: Option<&'static PerfScenarioSetup>,
}

pub(crate) const PERF_SUITE_WARMUP_FRAMES_DEFAULT: u32 = 180;
pub(crate) const PERF_SUITE_SAMPLE_FRAMES_DEFAULT: u32 = 600;
pub(crate) const PERF_SUITE_REPORT_DIR_DEFAULT: &str = "profiles";
pub(crate) const VTE_SWEEP_SAMPLE_FRAMES: usize = 120;

/// (max_trace_steps, max_trace_distance) for perf suite default render distance tier.
pub(crate) const PERF_TIER_DEFAULT: (u32, f32) = (320, 160.0);

pub(crate) const PERF_SUITE_SETTLE_STABLE_FRAMES: u32 = 60;
pub(crate) const PERF_SUITE_SETTLE_MAX_FRAMES: u32 = 600;

// ---------------------------------------------------------------------------
// Per-scenario setup configurations
// ---------------------------------------------------------------------------

/// 8 cubes in a ring around camera (offsets at ±4/±8 in XZW, Y+2).
static SETUP_ENTITIES_NEAR: PerfScenarioSetup = PerfScenarioSetup {
    entity_spawns: &[
        ("cube", [4.0, 2.0, 0.0, 0.0]),
        ("cube", [-4.0, 2.0, 0.0, 0.0]),
        ("cube", [0.0, 2.0, 4.0, 0.0]),
        ("cube", [0.0, 2.0, -4.0, 0.0]),
        ("cube", [8.0, 2.0, 0.0, 4.0]),
        ("cube", [-8.0, 2.0, 0.0, -4.0]),
        ("cube", [0.0, 2.0, 0.0, 8.0]),
        ("cube", [0.0, 2.0, 0.0, -8.0]),
    ],
    explosions: &[],
};

/// One small explosion offset from camera.
static SETUP_EXPLOSION_SMALL: PerfScenarioSetup = PerfScenarioSetup {
    entity_spawns: &[],
    explosions: &[([8.0, -2.0, 8.0, 0.0], 4)],
};

/// One large explosion directly below camera.
static SETUP_EXPLOSION_LARGE: PerfScenarioSetup = PerfScenarioSetup {
    entity_spawns: &[],
    explosions: &[([0.0, -1.0, 0.0, 0.0], 6)],
};

/// Combined: 8 entities + 1 explosion.
static SETUP_ENTITIES_AND_EXPLOSION: PerfScenarioSetup = PerfScenarioSetup {
    entity_spawns: &[
        ("cube", [4.0, 2.0, 0.0, 0.0]),
        ("cube", [-4.0, 2.0, 0.0, 0.0]),
        ("cube", [0.0, 2.0, 4.0, 0.0]),
        ("cube", [0.0, 2.0, -4.0, 0.0]),
        ("cube", [8.0, 2.0, 0.0, 4.0]),
        ("cube", [-8.0, 2.0, 0.0, -4.0]),
        ("cube", [0.0, 2.0, 0.0, 8.0]),
        ("cube", [0.0, 2.0, 0.0, -8.0]),
    ],
    explosions: &[([0.0, -1.0, 0.0, 0.0], 5)],
};

// ---------------------------------------------------------------------------
// Base poses for perf suite scenarios (MassivePlatforms seed 1337)
// ---------------------------------------------------------------------------

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
    setup: None,
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
    setup: None,
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
    setup: None,
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
    setup: None,
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
    setup: None,
};

pub(crate) const PERF_SUITE_SCENARIOS: &[PerfSuiteScenario] = &[
    // -----------------------------------------------------------------------
    // Baselines (5): existing poses, no setup, default render tier
    // -----------------------------------------------------------------------
    PerfSuiteScenario {
        label: "platform-surface",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    PerfSuiteScenario {
        label: "platform-4d",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_PLATFORM_4D
    },
    PerfSuiteScenario {
        label: "open-sky",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_OPEN_SKY
    },
    PerfSuiteScenario {
        label: "corridor",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_CORRIDOR
    },
    PerfSuiteScenario {
        label: "far-oblique",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        ..PERF_POSE_FAR_OBLIQUE
    },
    // -----------------------------------------------------------------------
    // Entities (3): nearby entities for BVH overhead testing
    // -----------------------------------------------------------------------
    PerfSuiteScenario {
        label: "platform-surface+entities",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_ENTITIES_NEAR),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    PerfSuiteScenario {
        label: "corridor+entities",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_ENTITIES_NEAR),
        ..PERF_POSE_CORRIDOR
    },
    PerfSuiteScenario {
        label: "far-oblique+entities",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_ENTITIES_NEAR),
        ..PERF_POSE_FAR_OBLIQUE
    },
    // -----------------------------------------------------------------------
    // Explosions (2): terrain craters
    // -----------------------------------------------------------------------
    PerfSuiteScenario {
        label: "platform-surface+explosion",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_EXPLOSION_SMALL),
        ..PERF_POSE_PLATFORM_SURFACE
    },
    PerfSuiteScenario {
        label: "corridor+explosion",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_EXPLOSION_LARGE),
        ..PERF_POSE_CORRIDOR
    },
    // -----------------------------------------------------------------------
    // Combined (2): entities + explosion
    // -----------------------------------------------------------------------
    PerfSuiteScenario {
        label: "platform-4d+combined",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_ENTITIES_AND_EXPLOSION),
        ..PERF_POSE_PLATFORM_4D
    },
    PerfSuiteScenario {
        label: "far-oblique+combined",
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: Some(&SETUP_ENTITIES_AND_EXPLOSION),
        ..PERF_POSE_FAR_OBLIQUE
    },
    // -----------------------------------------------------------------------
    // Structures (3): seeds with structures/mazes in view
    // NOTE: These scenarios require their specific seed to produce the expected
    // structures. The perf suite runner script iterates seeds; these will only
    // show structure content when run with the matching seed.
    // -----------------------------------------------------------------------
    // seed=23: 2 structures nearby (braided_transit + cross_shrine)
    PerfSuiteScenario {
        label: "structure-2x-s23",
        position: [0.0, 8.0, -24.0, -4.0],
        yaw: 1.25,
        pitch: -0.15,
        xw_angle: 0.0,
        zw_angle: 0.0,
        yw_deviation: 0.0,
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: None,
    },
    // seed=40: structure + maze combo (hypercube_frame + 11x3x11x15 maze)
    PerfSuiteScenario {
        label: "struct+maze-s40",
        position: [96.0, 20.0, 96.0, -4.0],
        yaw: -2.35,
        pitch: -0.22,
        xw_angle: -0.45,
        zw_angle: 0.42,
        yw_deviation: 0.0,
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: None,
    },
    // seed=46: large maze (11x7x15x15 grid cells)
    PerfSuiteScenario {
        label: "large-maze-s46",
        position: [0.0, 8.0, 0.0, 0.0],
        yaw: 0.0,
        pitch: -0.10,
        xw_angle: 0.0,
        zw_angle: 0.0,
        yw_deviation: 0.0,
        vte_max_trace_steps: Some(PERF_TIER_DEFAULT.0),
        vte_max_trace_distance: Some(PERF_TIER_DEFAULT.1),
        setup: None,
    },
];

pub(crate) const VTE_SWEEP_PROFILES: [VteRuntimeProfile; 2] = [
    VteRuntimeProfile { label: "A bvh" },
    VteRuntimeProfile { label: "B bvh" },
];

pub(crate) const VTE_SWEEP_MODE_LABEL: &str = "bvh";
