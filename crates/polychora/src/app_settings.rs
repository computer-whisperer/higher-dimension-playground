use super::*;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) const SETTINGS_AUTOSAVE_MIN_INTERVAL: Duration = Duration::from_millis(350);
const SETTINGS_SCHEMA_VERSION: u32 = 1;
const SETTINGS_FILE_NAME: &str = "settings.json";
const SETTINGS_APP_DIR: &str = "polychora";
const DEFAULT_RENDER_WIDTH: u32 = 960;
const DEFAULT_RENDER_HEIGHT: u32 = 540;
const DEFAULT_RENDER_LAYERS: u32 = 128;
const DEFAULT_AUDIO_VOLUME: f32 = 0.7;
const DEFAULT_AUDIO_SPATIAL_FALLOFF_POWER: f32 = AUDIO_SPATIAL_FALLOFF_POWER_DEFAULT;
const DEFAULT_MAIN_MENU_SERVER_ADDRESS: &str = "c-gateway.computer-whisperer.network:4000";
const MAX_HOTBAR_SLOT_INDEX: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CliOverrides {
    pub width: bool,
    pub height: bool,
    pub layers: bool,
    pub player_name: bool,
    pub audio_volume: bool,
    pub audio_spatial_falloff_power: bool,
    pub vte_y_slice_lookup_cache: bool,
    pub vte_integral_sky_emissive_tweak: bool,
    pub vte_integral_sky_scale: bool,
    pub vte_integral_hit_emissive_boost: bool,
    pub vte_integral_log_merge_tweak: bool,
    pub vte_integral_log_merge_k: bool,
    pub zw_angle_color_shift: bool,
    pub zw_angle_color_shift_strength: bool,
    pub vte_max_trace_steps: bool,
    pub vte_max_trace_distance: bool,
}

impl CliOverrides {
    pub(super) fn from_process_args() -> Self {
        let raw_args: Vec<OsString> = std::env::args_os().collect();
        Self {
            width: arg_present(&raw_args, "--width", Some("-W")),
            height: arg_present(&raw_args, "--height", Some("-H")),
            layers: arg_present(&raw_args, "--layers", None),
            player_name: arg_present(&raw_args, "--player-name", None),
            audio_volume: arg_present(&raw_args, "--audio-volume", None),
            audio_spatial_falloff_power: arg_present(
                &raw_args,
                "--audio-spatial-falloff-power",
                None,
            ),
            vte_y_slice_lookup_cache: arg_present(&raw_args, "--vte-y-slice-lookup-cache", None),
            vte_integral_sky_emissive_tweak: arg_present(
                &raw_args,
                "--vte-integral-sky-emissive-tweak",
                None,
            ),
            vte_integral_sky_scale: arg_present(&raw_args, "--vte-integral-sky-scale", None),
            vte_integral_hit_emissive_boost: arg_present(
                &raw_args,
                "--vte-integral-hit-emissive-boost",
                None,
            ),
            vte_integral_log_merge_tweak: arg_present(
                &raw_args,
                "--vte-integral-log-merge-tweak",
                None,
            ),
            vte_integral_log_merge_k: arg_present(&raw_args, "--vte-integral-log-merge-k", None),
            zw_angle_color_shift: arg_present(&raw_args, "--zw-angle-color-shift", None),
            zw_angle_color_shift_strength: arg_present(
                &raw_args,
                "--zw-angle-color-shift-strength",
                None,
            ),
            vte_max_trace_steps: arg_present(&raw_args, "--vte-max-trace-steps", None),
            vte_max_trace_distance: arg_present(&raw_args, "--vte-max-trace-distance", None),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub(super) struct PersistedSettings {
    pub schema_version: u32,
    pub control_scheme: PersistedControlScheme,
    pub info_panel_mode: PersistedInfoPanelMode,
    pub focal_length_xy: f32,
    pub focal_length_zw: f32,
    pub zw_angle_color_shift_enabled: bool,
    pub zw_angle_color_shift_strength: f32,
    pub place_material: u8, // legacy name kept for JSON compat
    pub hotbar_slots: [u8; 9], // legacy format kept for JSON compat
    pub hotbar_selected_index: usize,
    pub render_width: u32,
    pub render_height: u32,
    pub render_layers: u32,
    pub audio_volume: f32,
    pub audio_spatial_falloff_power: f32,
    pub vte_y_slice_lookup_cache_enabled: bool,
    pub vte_integral_sky_emissive_enabled: bool,
    pub vte_integral_sky_scale: f32,
    pub vte_integral_hit_emissive_boost: f32,
    pub vte_integral_log_merge_enabled: bool,
    pub vte_integral_log_merge_k: f32,
    pub vte_max_trace_steps: u32,
    pub vte_max_trace_distance: f32,
    pub main_menu_server_address: String,
    pub main_menu_player_name: String,
}

impl Default for PersistedSettings {
    fn default() -> Self {
        Self {
            schema_version: SETTINGS_SCHEMA_VERSION,
            control_scheme: PersistedControlScheme::LookTransport,
            info_panel_mode: PersistedInfoPanelMode::VectorTable,
            focal_length_xy: 1.0,
            focal_length_zw: 1.0,
            zw_angle_color_shift_enabled: true,
            zw_angle_color_shift_strength: ZW_ANGLE_COLOR_SHIFT_STRENGTH_DEFAULT,
            place_material: 3,
            hotbar_slots: [3, 27, 28, 29, 31, 12, 13, 1, 4],
            hotbar_selected_index: 0,
            render_width: DEFAULT_RENDER_WIDTH,
            render_height: DEFAULT_RENDER_HEIGHT,
            render_layers: DEFAULT_RENDER_LAYERS,
            audio_volume: DEFAULT_AUDIO_VOLUME,
            audio_spatial_falloff_power: DEFAULT_AUDIO_SPATIAL_FALLOFF_POWER,
            vte_y_slice_lookup_cache_enabled: true,
            vte_integral_sky_emissive_enabled: true,
            vte_integral_sky_scale: 0.25,
            vte_integral_hit_emissive_boost: 0.025,
            vte_integral_log_merge_enabled: true,
            vte_integral_log_merge_k: 8.0,
            vte_max_trace_steps: 320,
            vte_max_trace_distance: 160.0,
            main_menu_server_address: DEFAULT_MAIN_MENU_SERVER_ADDRESS.to_string(),
            main_menu_player_name: default_multiplayer_player_name(),
        }
    }
}

impl PersistedSettings {
    pub(super) fn sanitized(mut self) -> Self {
        self.schema_version = SETTINGS_SCHEMA_VERSION;
        self.focal_length_xy = self
            .focal_length_xy
            .clamp(FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX);
        self.focal_length_zw = self
            .focal_length_zw
            .clamp(FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX);
        self.zw_angle_color_shift_strength = self.zw_angle_color_shift_strength.clamp(
            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
        );
        self.place_material = self
            .place_material
            .clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX);
        for slot in &mut self.hotbar_slots {
            *slot = (*slot).clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX);
        }
        self.hotbar_selected_index = self.hotbar_selected_index.min(MAX_HOTBAR_SLOT_INDEX);
        self.render_width = self.render_width.clamp(128, 3840);
        self.render_height = self.render_height.clamp(128, 2160);
        self.render_layers = self.render_layers.clamp(1, 512);
        self.audio_volume = self.audio_volume.clamp(0.0, 2.0);
        self.audio_spatial_falloff_power = self.audio_spatial_falloff_power.clamp(
            AUDIO_SPATIAL_FALLOFF_POWER_MIN,
            AUDIO_SPATIAL_FALLOFF_POWER_MAX,
        );
        self.vte_integral_sky_scale = self
            .vte_integral_sky_scale
            .clamp(VTE_INTEGRAL_SKY_SCALE_MIN, VTE_INTEGRAL_SKY_SCALE_MAX);
        self.vte_integral_hit_emissive_boost = self
            .vte_integral_hit_emissive_boost
            .clamp(VTE_INTEGRAL_HIT_EMISSIVE_MIN, VTE_INTEGRAL_HIT_EMISSIVE_MAX);
        self.vte_integral_log_merge_k = self
            .vte_integral_log_merge_k
            .clamp(VTE_INTEGRAL_LOG_MERGE_K_MIN, VTE_INTEGRAL_LOG_MERGE_K_MAX);
        self.vte_max_trace_steps = self
            .vte_max_trace_steps
            .clamp(VTE_TRACE_STEPS_MIN, VTE_TRACE_STEPS_MAX);
        self.vte_max_trace_distance = self
            .vte_max_trace_distance
            .clamp(VTE_TRACE_DISTANCE_MIN, VTE_TRACE_DISTANCE_MAX);
        let server = self.main_menu_server_address.trim();
        self.main_menu_server_address = if server.is_empty() {
            DEFAULT_MAIN_MENU_SERVER_ADDRESS.to_string()
        } else {
            server.to_string()
        };
        let player = self.main_menu_player_name.trim();
        self.main_menu_player_name = if player.is_empty() {
            default_multiplayer_player_name()
        } else {
            player.to_string()
        };
        self
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum PersistedControlScheme {
    IntuitiveUpright,
    LookTransport,
    RotorFree,
    LegacySideButtonLayers,
    LegacyScrollCycle,
}

impl From<ControlScheme> for PersistedControlScheme {
    fn from(value: ControlScheme) -> Self {
        match value {
            ControlScheme::IntuitiveUpright => Self::IntuitiveUpright,
            ControlScheme::LookTransport => Self::LookTransport,
            ControlScheme::RotorFree => Self::RotorFree,
            ControlScheme::LegacySideButtonLayers => Self::LegacySideButtonLayers,
            ControlScheme::LegacyScrollCycle => Self::LegacyScrollCycle,
        }
    }
}

impl From<PersistedControlScheme> for ControlScheme {
    fn from(value: PersistedControlScheme) -> Self {
        match value {
            PersistedControlScheme::IntuitiveUpright => ControlScheme::IntuitiveUpright,
            PersistedControlScheme::LookTransport => ControlScheme::LookTransport,
            PersistedControlScheme::RotorFree => ControlScheme::RotorFree,
            PersistedControlScheme::LegacySideButtonLayers => ControlScheme::LegacySideButtonLayers,
            PersistedControlScheme::LegacyScrollCycle => ControlScheme::LegacyScrollCycle,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum PersistedInfoPanelMode {
    Full,
    VectorTable,
    VectorTable2,
    Off,
}

impl From<InfoPanelMode> for PersistedInfoPanelMode {
    fn from(value: InfoPanelMode) -> Self {
        match value {
            InfoPanelMode::Full => Self::Full,
            InfoPanelMode::VectorTable => Self::VectorTable,
            InfoPanelMode::VectorTable2 => Self::VectorTable2,
            InfoPanelMode::Off => Self::Off,
        }
    }
}

impl From<PersistedInfoPanelMode> for InfoPanelMode {
    fn from(value: PersistedInfoPanelMode) -> Self {
        match value {
            PersistedInfoPanelMode::Full => InfoPanelMode::Full,
            PersistedInfoPanelMode::VectorTable => InfoPanelMode::VectorTable,
            PersistedInfoPanelMode::VectorTable2 => InfoPanelMode::VectorTable2,
            PersistedInfoPanelMode::Off => InfoPanelMode::Off,
        }
    }
}

pub(super) fn settings_file_path() -> PathBuf {
    if let Some(base) = platform_config_dir() {
        return base.join(SETTINGS_APP_DIR).join(SETTINGS_FILE_NAME);
    }
    PathBuf::from("saves").join(SETTINGS_FILE_NAME)
}

pub(super) fn load_settings(path: &Path) -> Option<PersistedSettings> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(error) => {
            if error.kind() != io::ErrorKind::NotFound {
                eprintln!("Failed to read settings {}: {}", path.display(), error);
            }
            return None;
        }
    };

    match serde_json::from_str::<PersistedSettings>(&raw) {
        Ok(settings) => Some(settings.sanitized()),
        Err(error) => {
            eprintln!(
                "Failed to parse settings {}: {} (ignoring file)",
                path.display(),
                error
            );
            None
        }
    }
}

pub(super) fn save_settings(path: &Path, settings: &PersistedSettings) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(settings)
        .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
    std::fs::write(path, bytes)
}

pub(super) fn apply_settings_to_args(
    args: &mut Args,
    settings: &PersistedSettings,
    cli_overrides: CliOverrides,
) {
    let settings = settings.clone().sanitized();

    if !cli_overrides.width {
        args.width = settings.render_width;
    }
    if !cli_overrides.height {
        args.height = settings.render_height;
    }
    if !cli_overrides.layers {
        args.layers = settings.render_layers;
    }
    if !cli_overrides.audio_volume {
        args.audio_volume = settings.audio_volume;
    }
    if !cli_overrides.audio_spatial_falloff_power {
        args.audio_spatial_falloff_power = settings.audio_spatial_falloff_power;
    }
    if !cli_overrides.player_name && args.player_name.is_none() {
        args.player_name = Some(settings.main_menu_player_name.clone());
    }
    if !cli_overrides.vte_y_slice_lookup_cache {
        args.vte_y_slice_lookup_cache = settings.vte_y_slice_lookup_cache_enabled;
    }
    if !cli_overrides.vte_integral_sky_emissive_tweak {
        args.vte_integral_sky_emissive_tweak = settings.vte_integral_sky_emissive_enabled;
    }
    if !cli_overrides.vte_integral_sky_scale {
        args.vte_integral_sky_scale = settings.vte_integral_sky_scale;
    }
    if !cli_overrides.vte_integral_hit_emissive_boost {
        args.vte_integral_hit_emissive_boost = settings.vte_integral_hit_emissive_boost;
    }
    if !cli_overrides.vte_integral_log_merge_tweak {
        args.vte_integral_log_merge_tweak = settings.vte_integral_log_merge_enabled;
    }
    if !cli_overrides.vte_integral_log_merge_k {
        args.vte_integral_log_merge_k = settings.vte_integral_log_merge_k;
    }
    if !cli_overrides.zw_angle_color_shift {
        args.zw_angle_color_shift = settings.zw_angle_color_shift_enabled;
    }
    if !cli_overrides.zw_angle_color_shift_strength {
        args.zw_angle_color_shift_strength = settings.zw_angle_color_shift_strength;
    }
    if !cli_overrides.vte_max_trace_steps {
        args.vte_max_trace_steps = settings.vte_max_trace_steps;
    }
    if !cli_overrides.vte_max_trace_distance {
        args.vte_max_trace_distance = settings.vte_max_trace_distance;
    }
}

impl App {
    pub(super) fn apply_runtime_settings(&mut self, settings: &PersistedSettings) {
        let settings = settings.clone().sanitized();
        self.set_control_scheme(settings.control_scheme.into());
        self.info_panel_mode = settings.info_panel_mode.into();
        self.focal_length_xy = settings.focal_length_xy;
        self.focal_length_zw = settings.focal_length_zw;
        for (i, &mat) in settings.hotbar_slots.iter().enumerate() {
            self.hotbar_slots[i] =
                Some(polychora::shared::protocol::ItemStack::block(0, mat as u32, 1));
        }
        self.hotbar_selected_index = settings.hotbar_selected_index.min(MAX_HOTBAR_SLOT_INDEX);
        self.selected_block =
            block_data_from_slot(&self.hotbar_slots[self.hotbar_selected_index]);
        self.main_menu_server_address = settings.main_menu_server_address;
        if self.args.player_name.is_none() {
            self.main_menu_player_name = settings.main_menu_player_name;
        }
        self.pending_render_width = self.args.width;
        self.pending_render_height = self.args.height;
        self.pending_render_layers = self.args.layers;
        self.audio.master_volume = self.args.audio_volume.clamp(0.0, 2.0);
        self.audio.spatial_falloff_power = self.args.audio_spatial_falloff_power.clamp(
            AUDIO_SPATIAL_FALLOFF_POWER_MIN,
            AUDIO_SPATIAL_FALLOFF_POWER_MAX,
        );
    }

    pub(super) fn capture_persisted_settings(&self) -> PersistedSettings {
        PersistedSettings {
            schema_version: SETTINGS_SCHEMA_VERSION,
            control_scheme: self.control_scheme.into(),
            info_panel_mode: self.info_panel_mode.into(),
            focal_length_xy: self.focal_length_xy,
            focal_length_zw: self.focal_length_zw,
            zw_angle_color_shift_enabled: self.zw_angle_color_shift_enabled,
            zw_angle_color_shift_strength: self.zw_angle_color_shift_strength,
            place_material: self.selected_block.block_type as u8,
            hotbar_slots: {
                let mut slots = [3u8; 9];
                for (i, slot) in self.hotbar_slots.iter().enumerate() {
                    slots[i] = block_material_from_slot(slot);
                }
                slots
            },
            hotbar_selected_index: self.hotbar_selected_index.min(MAX_HOTBAR_SLOT_INDEX),
            render_width: self.args.width,
            render_height: self.args.height,
            render_layers: self.args.layers,
            audio_volume: self.audio.master_volume,
            audio_spatial_falloff_power: self.audio.spatial_falloff_power,
            vte_y_slice_lookup_cache_enabled: self.vte_y_slice_lookup_cache_enabled,
            vte_integral_sky_emissive_enabled: self.vte_integral_sky_emissive_enabled,
            vte_integral_sky_scale: self.vte_integral_sky_scale,
            vte_integral_hit_emissive_boost: self.vte_integral_hit_emissive_boost,
            vte_integral_log_merge_enabled: self.vte_integral_log_merge_enabled,
            vte_integral_log_merge_k: self.vte_integral_log_merge_k,
            vte_max_trace_steps: self.vte_max_trace_steps,
            vte_max_trace_distance: self.vte_max_trace_distance,
            main_menu_server_address: self.main_menu_server_address.clone(),
            main_menu_player_name: self.main_menu_player_name.clone(),
        }
        .sanitized()
    }

    pub(super) fn persist_settings_if_needed(&mut self, force: bool) {
        let snapshot = self.capture_persisted_settings();
        if snapshot == self.settings_last_saved {
            return;
        }
        let now = Instant::now();
        if !force
            && now.duration_since(self.settings_last_save_attempt) < SETTINGS_AUTOSAVE_MIN_INTERVAL
        {
            return;
        }
        self.settings_last_save_attempt = now;
        match save_settings(&self.settings_file_path, &snapshot) {
            Ok(()) => {
                self.settings_last_saved = snapshot;
            }
            Err(error) => {
                eprintln!(
                    "Failed to save settings {}: {}",
                    self.settings_file_path.display(),
                    error
                );
            }
        }
    }
}

fn arg_present(args: &[OsString], long: &str, short: Option<&str>) -> bool {
    let long_equals = format!("{long}=");
    let short_equals = short.map(|name| format!("{name}="));

    args.iter().any(|arg| {
        let Some(s) = arg.to_str() else {
            return false;
        };
        if s == long || s.starts_with(&long_equals) {
            return true;
        }
        if let Some(short_name) = short {
            if s == short_name {
                return true;
            }
            if let Some(short_eq) = short_equals.as_ref() {
                if s.starts_with(short_eq) {
                    return true;
                }
            }
            if short_name.len() == 2 && s.starts_with(short_name) && s.len() > short_name.len() {
                return true;
            }
        }
        false
    })
}

#[cfg(target_os = "windows")]
fn platform_config_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from).or_else(|| {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .map(|home| home.join("AppData").join("Roaming"))
    })
}

#[cfg(target_os = "macos")]
fn platform_config_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library").join("Application Support"))
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn platform_config_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })
}
