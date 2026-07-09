use damascene_core::prelude::{
    badge, button, card, card_content, card_header, card_title, column, image as damascene_image, mono,
    render_bundle, row, scroll, spacer, spinner, stack, table, table_body, table_cell, table_head,
    table_header, table_row, tabs, tabs_list, text, text_input_with, toggle, tokens, write_bundle,
    Align, Axis, Color, Cursor, El, Image, ImageFit, Justify, Kind, Rect, Sides, Size,
    StyleProfile, SurfaceRole, TextInputOpts, UiEventKind,
};
use damascene_core::widgets::slider as damascene_slider;
use damascene_core::widgets::text_input as damascene_text_input;
use polychora::content_registry::ContentRegistry;
use polychora::shared::inventory::{InventoryTab, HOTBAR_SIZE, INVENTORY_COLS};
use polychora::shared::protocol::ItemStack;
use std::{panic::Location, path::Path};

use super::{
    block_data_from_slot, format_cbor_for_display, format_file_size, App, InfoPanelMode,
    MainMenuPage, MainMenuTransition, PlacementPreviewMode, SettingsPage, WailaTarget,
};
use crate::audio::{AUDIO_SPATIAL_FALLOFF_POWER_MAX, AUDIO_SPATIAL_FALLOFF_POWER_MIN};
use crate::consts::{
    FOCAL_LENGTH_MAX, FOCAL_LENGTH_MIN, VTE_INTEGRAL_HIT_EMISSIVE_MAX,
    VTE_INTEGRAL_HIT_EMISSIVE_MIN, VTE_INTEGRAL_LOG_MERGE_K_MAX, VTE_INTEGRAL_LOG_MERGE_K_MIN,
    VTE_INTEGRAL_SKY_SCALE_MAX, VTE_INTEGRAL_SKY_SCALE_MIN, VTE_TRACE_DISTANCE_MAX,
    VTE_TRACE_DISTANCE_MIN, VTE_TRACE_STEPS_MAX, VTE_TRACE_STEPS_MIN,
    ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX, ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
};
use crate::input::ControlScheme;
use crate::material_icons::MaterialIconSheet;

const HOTBAR_SLOT_KEY_PREFIX: &str = "damascene_hotbar_slot_";
const ORIENTATION_KEY_PREFIX: &str = "damascene_orientation_";
const INVENTORY_TABS_KEY: &str = "damascene_inventory_tabs";
const INVENTORY_BLOCK_KEY_PREFIX: &str = "damascene_inventory_block_";
const INVENTORY_ENTITY_KEY_PREFIX: &str = "damascene_inventory_entity_";
const INVENTORY_SLOT_KEY_PREFIX: &str = "damascene_inventory_slot_";
const INVENTORY_CLOSE_KEY: &str = "damascene_inventory_close";
const PAUSE_RESUME_KEY: &str = "damascene_pause_resume";
const PAUSE_MAIN_MENU_KEY: &str = "damascene_pause_main_menu";
const PAUSE_QUIT_KEY: &str = "damascene_pause_quit";
const PAUSE_MENU_MODE_TABS_KEY: &str = "damascene_pause_mode_tabs";
const PAUSE_SETTINGS_TABS_KEY: &str = "damascene_pause_settings_tabs";
const PAUSE_CONTROL_SCHEME_KEY_PREFIX: &str = "damascene_pause_control_scheme_";
const PAUSE_INFO_PANEL_KEY_PREFIX: &str = "damascene_pause_info_panel_";
const PAUSE_PLACEMENT_PREVIEW_KEY_PREFIX: &str = "damascene_pause_placement_preview_";
const PAUSE_TOGGLE_PREVIEW_HIDE_CAMERA_KEY: &str = "damascene_pause_preview_hide_camera";
const PAUSE_TOGGLE_PREVIEW_HIDE_SAME_SCALE_KEY: &str = "damascene_pause_preview_hide_same_scale";
const PAUSE_TOGGLE_ZW_SHIFT_KEY: &str = "damascene_pause_zw_shift";
const PAUSE_TOGGLE_INTEGRAL_SKY_KEY: &str = "damascene_pause_integral_sky";
const PAUSE_TOGGLE_LOG_MERGE_KEY: &str = "damascene_pause_log_merge";
const PAUSE_TOGGLE_STREAM_TREE_BOUNDS_KEY: &str = "damascene_pause_stream_tree_bounds";
const PAUSE_TOGGLE_STREAM_COMPARE_BOUNDS_KEY: &str = "damascene_pause_stream_compare_bounds";
const PAUSE_TOGGLE_STREAM_LABELS_KEY: &str = "damascene_pause_stream_labels";
const PAUSE_TOGGLE_STREAM_NON_EMPTY_KEY: &str = "damascene_pause_stream_non_empty";
const PAUSE_TOGGLE_STREAM_BRANCH_KEY: &str = "damascene_pause_stream_branch";
const PAUSE_TOGGLE_STREAM_UNIFORM_KEY: &str = "damascene_pause_stream_uniform";
const PAUSE_TOGGLE_STREAM_CHUNK_ARRAY_KEY: &str = "damascene_pause_stream_chunk_array";
const PAUSE_TOGGLE_STREAM_PROCEDURAL_KEY: &str = "damascene_pause_stream_procedural";
const PAUSE_TOGGLE_STREAM_EMPTY_KEY: &str = "damascene_pause_stream_empty";
const PAUSE_TOGGLE_SAMPLE_RAY_BOUNDS_KEY: &str = "damascene_pause_sample_ray_bounds";
const PAUSE_DUMP_TREES_KEY: &str = "damascene_pause_dump_trees";
const PAUSE_SLIDER_MASTER_VOLUME_KEY: &str = "damascene_pause_slider_master_volume";
const PAUSE_SLIDER_SPATIAL_FALLOFF_KEY: &str = "damascene_pause_slider_spatial_falloff";
const PAUSE_SLIDER_FOCAL_XY_KEY: &str = "damascene_pause_slider_focal_xy";
const PAUSE_SLIDER_FOCAL_ZW_KEY: &str = "damascene_pause_slider_focal_zw";
const PAUSE_SLIDER_ZW_SHIFT_KEY: &str = "damascene_pause_slider_zw_shift";
const PAUSE_SLIDER_TRACE_STEPS_KEY: &str = "damascene_pause_slider_trace_steps";
const PAUSE_SLIDER_TRACE_DISTANCE_KEY: &str = "damascene_pause_slider_trace_distance";
const PAUSE_SLIDER_SKY_SCALE_KEY: &str = "damascene_pause_slider_sky_scale";
const PAUSE_SLIDER_HIT_EMISSIVE_KEY: &str = "damascene_pause_slider_hit_emissive";
const PAUSE_SLIDER_LOG_MERGE_KEY: &str = "damascene_pause_slider_log_merge";
const PAUSE_SLIDER_STREAM_MAX_NODES_KEY: &str = "damascene_pause_slider_stream_max_nodes";
const PAUSE_SLIDER_SAMPLE_RAY_MAX_NODES_KEY: &str = "damascene_pause_slider_sample_ray_max_nodes";
const PAUSE_SLIDER_LABEL_MAX_COUNT_KEY: &str = "damascene_pause_slider_label_max_count";
const PAUSE_SLIDER_COMPARE_MAX_CHUNKS_KEY: &str = "damascene_pause_slider_compare_max_chunks";
const PAUSE_SLIDER_COMPARE_LOG_INTERVAL_KEY: &str = "damascene_pause_slider_compare_log_interval";
const TELEPORT_FIELD_KEY_PREFIX: &str = "damascene_teleport_coord_";
const TELEPORT_APPLY_KEY: &str = "damascene_teleport_apply";
const TELEPORT_ORIGIN_KEY: &str = "damascene_teleport_origin";
const TELEPORT_CLOSE_KEY: &str = "damascene_teleport_close";
const TELEPORT_PLAYER_KEY_PREFIX: &str = "damascene_teleport_player_";
const DEV_CONSOLE_INPUT_KEY: &str = "damascene_dev_console_input";
const DEV_CONSOLE_RUN_KEY: &str = "damascene_dev_console_run";
const DEV_CONSOLE_CLOSE_KEY: &str = "damascene_dev_console_close";
const BLOCK_GUI_CLOSE_KEY: &str = "damascene_block_gui_close";
const BLOCK_GUI_SLOT_KEY_PREFIX: &str = "damascene_block_gui_slot_";
const MAIN_MENU_SINGLEPLAYER_KEY: &str = "damascene_main_singleplayer";
const MAIN_MENU_MULTIPLAYER_KEY: &str = "damascene_main_multiplayer";
const MAIN_MENU_QUIT_KEY: &str = "damascene_main_quit";
const MAIN_MENU_BACK_KEY: &str = "damascene_main_back";
const MAIN_MENU_WORLD_KEY_PREFIX: &str = "damascene_main_world_";
const MAIN_MENU_LOAD_SELECTED_KEY: &str = "damascene_main_load_selected";
const MAIN_MENU_CREATE_WORLD_KEY: &str = "damascene_main_create_world";
const MAIN_MENU_MIGRATIONS_KEY: &str = "damascene_main_migrations";
const MAIN_MENU_WORLD_TYPE_TABS_KEY: &str = "damascene_main_world_type_tabs";
const MAIN_MENU_PLAYER_NAME_KEY: &str = "damascene_main_player_name";
const MAIN_MENU_SERVER_ADDRESS_KEY: &str = "damascene_main_server_address";
const MAIN_MENU_CONNECT_KEY: &str = "damascene_main_connect";
const MAIN_MENU_MIGRATE_LEGACY_KEY: &str = "damascene_main_migrate_legacy";
const MAIN_MENU_MIGRATE_V3_KEY: &str = "damascene_main_migrate_v3";
const MAIN_MENU_TRIM_INPUT_KEY: &str = "damascene_main_trim_input";
const MAIN_MENU_TRIM_OUTPUT_KEY: &str = "damascene_main_trim_output";
const MAIN_MENU_TRIM_MIN_KEY: &str = "damascene_main_trim_min";
const MAIN_MENU_TRIM_MAX_KEY: &str = "damascene_main_trim_max";
const MAIN_MENU_TRIM_RUN_KEY: &str = "damascene_main_trim_run";
const MAIN_MENU_V3_INPUT_KEY: &str = "damascene_main_v3_input";
const MAIN_MENU_V4_OUTPUT_KEY: &str = "damascene_main_v4_output";
const MAIN_MENU_V3_OVERWRITE_KEY: &str = "damascene_main_v3_overwrite";
const MAIN_MENU_V3_RUN_KEY: &str = "damascene_main_v3_run";
const NAV_HUD_BOTTOM_LEFT_RESERVED_WIDTH: f32 = 150.0;

struct HotbarSlotView {
    name: String,
    count: u32,
    scale_label: Option<String>,
    color: Color,
    icon: Option<Image>,
    selected: bool,
}

fn write_damascene_bundle(
    overlay: &mut El,
    width: u32,
    height: u32,
    out_dir: &Path,
    name: &str,
) -> std::io::Result<()> {
    let viewport = Rect::new(0.0, 0.0, width as f32, height as f32);
    let bundle = render_bundle(overlay, viewport);
    let written = write_bundle(&bundle, out_dir, name)?;
    eprintln!(
        "Wrote Damascene overlay bundle '{}' artifacts to {}",
        name,
        out_dir.display()
    );
    for path in written {
        eprintln!("  {}", path.display());
    }
    if !bundle.lint.findings.is_empty() {
        eprintln!("\nDamascene lint findings ({}):", bundle.lint.findings.len());
        eprint!("{}", bundle.lint.text());
    }
    Ok(())
}

impl App {
    fn write_damascene_named_bundle(&self, name: &str, mut overlay: El) -> std::io::Result<()> {
        write_damascene_bundle(
            &mut overlay,
            self.args.width,
            self.args.height,
            &self.args.damascene_bundle_dir,
            name,
        )
    }

    pub(super) fn dump_damascene_overlay_bundle(&mut self) -> std::io::Result<()> {
        let saved_menu_open = self.menu_open;
        let saved_controls_dialog_open = self.controls_dialog_open;
        let saved_inventory_open = self.inventory_open;
        let saved_teleport_dialog_open = self.teleport_dialog_open;
        let saved_dev_console_open = self.dev_console_open;
        let saved_settings_page = self.settings_page;
        let saved_main_menu_page = self.main_menu_page.clone();

        self.menu_open = false;
        self.controls_dialog_open = false;
        self.inventory_open = false;
        self.teleport_dialog_open = false;
        self.dev_console_open = false;

        if let Some(overlay) = self.build_damascene_overlay(None) {
            self.write_damascene_named_bundle("damascene_hud", overlay)?;
        }
        self.write_damascene_named_bundle("damascene_loading", self.build_damascene_loading_overlay())?;

        self.inventory_open = true;
        if let Some(overlay) = self.build_damascene_overlay(None) {
            self.write_damascene_named_bundle("damascene_inventory", overlay)?;
        }
        self.inventory_open = false;

        self.teleport_dialog_open = true;
        if let Some(overlay) = self.build_damascene_overlay(None) {
            self.write_damascene_named_bundle("damascene_teleport", overlay)?;
        }
        self.teleport_dialog_open = false;

        self.dev_console_open = true;
        if let Some(overlay) = self.build_damascene_overlay(None) {
            self.write_damascene_named_bundle("damascene_dev_console", overlay)?;
        }
        self.dev_console_open = false;

        self.menu_open = true;
        self.controls_dialog_open = false;
        for page in SettingsPage::ALL {
            self.settings_page = page;
            if let Some(overlay) = self.build_damascene_overlay(None) {
                let name = format!("damascene_pause_settings_{}", settings_page_token(page));
                self.write_damascene_named_bundle(&name, overlay)?;
            }
        }

        self.controls_dialog_open = true;
        if let Some(overlay) = self.build_damascene_overlay(None) {
            self.write_damascene_named_bundle("damascene_pause_controls", overlay)?;
        }
        self.menu_open = false;
        self.controls_dialog_open = false;

        for (page, name) in [
            (MainMenuPage::Root, "damascene_main_menu_root"),
            (MainMenuPage::Singleplayer, "damascene_main_menu_singleplayer"),
            (
                MainMenuPage::SingleplayerMigrations,
                "damascene_main_menu_migrations",
            ),
            (
                MainMenuPage::SingleplayerMigrationLegacyTrim,
                "damascene_main_menu_migrate_legacy_trim",
            ),
            (
                MainMenuPage::SingleplayerMigrationV3ToV4,
                "damascene_main_menu_migrate_v3_to_v4",
            ),
            (MainMenuPage::Multiplayer, "damascene_main_menu_multiplayer"),
        ] {
            self.main_menu_page = page;
            self.write_damascene_named_bundle(name, self.build_damascene_main_menu())?;
        }

        self.menu_open = saved_menu_open;
        self.controls_dialog_open = saved_controls_dialog_open;
        self.inventory_open = saved_inventory_open;
        self.teleport_dialog_open = saved_teleport_dialog_open;
        self.dev_console_open = saved_dev_console_open;
        self.settings_page = saved_settings_page;
        self.main_menu_page = saved_main_menu_page;

        Ok(())
    }

    pub(super) fn build_damascene_main_menu(&self) -> El {
        let panel = match self.main_menu_page {
            MainMenuPage::Root => self.build_damascene_main_menu_root(),
            MainMenuPage::Singleplayer => self.build_damascene_main_menu_singleplayer(),
            MainMenuPage::SingleplayerMigrations => {
                self.build_damascene_main_menu_singleplayer_migrations()
            }
            MainMenuPage::SingleplayerMigrationLegacyTrim => {
                self.build_damascene_main_menu_migrate_legacy_trim()
            }
            MainMenuPage::SingleplayerMigrationV3ToV4 => {
                self.build_damascene_main_menu_migrate_v3_to_v4()
            }
            MainMenuPage::Multiplayer => self.build_damascene_main_menu_multiplayer(),
        };
        build_damascene_center_modal_shell(panel, false)
    }

    fn build_damascene_main_menu_root(&self) -> El {
        main_menu_panel(
            [
                text("Polychora").display().bold(),
                text("4D Voxel Explorer").muted(),
            ],
            [
                button("Singleplayer")
                    .key(MAIN_MENU_SINGLEPLAYER_KEY)
                    .primary()
                    .width(Size::Fixed(240.0))
                    .height(Size::Fixed(38.0)),
                button("Multiplayer")
                    .key(MAIN_MENU_MULTIPLAYER_KEY)
                    .secondary()
                    .width(Size::Fixed(240.0))
                    .height(Size::Fixed(38.0)),
                button("Quit")
                    .key(MAIN_MENU_QUIT_KEY)
                    .ghost()
                    .width(Size::Fixed(240.0))
                    .height(Size::Fixed(34.0)),
            ],
            360.0,
            Size::Hug,
        )
    }

    fn build_damascene_main_menu_singleplayer(&self) -> El {
        let world_rows: Vec<El> = if self.main_menu_world_files.is_empty() {
            vec![
                text("No v4 world save directories found in saves/ or the current directory.")
                    .caption()
                    .muted()
                    .width(Size::Fill(1.0)),
            ]
        } else {
            self.main_menu_world_files
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    let label = format!(
                        "{}   {}",
                        entry.display_name,
                        format_file_size(entry.size_bytes)
                    );
                    let item = button(label)
                        .key(format!("{MAIN_MENU_WORLD_KEY_PREFIX}{index}"))
                        .width(Size::Fill(1.0))
                        .height(Size::Fixed(32.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0));
                    if self.main_menu_selected_world == Some(index) {
                        item.primary()
                    } else {
                        item.secondary()
                    }
                })
                .collect()
        };

        let mut load_button = button("Load Selected")
            .key(MAIN_MENU_LOAD_SELECTED_KEY)
            .primary()
            .height(Size::Fixed(32.0))
            .padding(Sides::xy(tokens::SPACE_2, 0.0));
        if self.main_menu_selected_world.is_none() {
            load_button = load_button.disabled();
        }

        let body = column([
            main_menu_section(
                "Saved Worlds",
                scroll(world_rows)
                    .key("damascene_main_worlds_scroll")
                    .height(Size::Fixed(160.0))
                    .width(Size::Fill(1.0)),
            ),
            main_menu_status(self.main_menu_connect_error.as_deref()),
            main_menu_section(
                "New World Type",
                tabs_list(
                    MAIN_MENU_WORLD_TYPE_TABS_KEY,
                    &world_generator_token(self.main_menu_new_world_generator),
                    [
                        ("flat_floor", "Flat Floor"),
                        ("massive_platforms", "Massive Platforms"),
                    ],
                )
                .width(Size::Fixed(390.0)),
            ),
            row([
                load_button,
                button("Create New World")
                    .key(MAIN_MENU_CREATE_WORLD_KEY)
                    .secondary()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                button("Migrations")
                    .key(MAIN_MENU_MIGRATIONS_KEY)
                    .secondary()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                button("Back")
                    .key(MAIN_MENU_BACK_KEY)
                    .ghost()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            ])
            .gap(tokens::SPACE_2)
            .align(Align::Center)
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        main_menu_page_panel("Singleplayer", "Local worlds", body, 690.0, Size::Hug)
    }

    fn build_damascene_main_menu_singleplayer_migrations(&self) -> El {
        let body = column([
            main_menu_action_card(
                "Legacy .v4dw Keep-Bounds Trim",
                "Drop override chunks outside selected chunk bounds.",
                MAIN_MENU_MIGRATE_LEGACY_KEY,
            ),
            main_menu_action_card(
                "v3 Save Root -> v4 Save Root",
                "Upgrade a v3 save root directory into a v4 save root.",
                MAIN_MENU_MIGRATE_V3_KEY,
            ),
            main_menu_status(self.main_menu_migration_status.as_deref()),
            row([button("Back")
                .key(MAIN_MENU_BACK_KEY)
                .ghost()
                .height(Size::Fixed(32.0))
                .padding(Sides::xy(tokens::SPACE_2, 0.0))])
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        main_menu_page_panel(
            "Singleplayer Migrations",
            "World maintenance",
            body,
            560.0,
            Size::Hug,
        )
    }

    fn build_damascene_main_menu_migrate_legacy_trim(&self) -> El {
        let body = column([
            main_menu_input(
                "Input .v4dw",
                &self.main_menu_migrate_trim_input,
                &self.damascene_selection,
                MAIN_MENU_TRIM_INPUT_KEY,
                "saves/world.v4dw",
            ),
            main_menu_input(
                "Output .v4dw",
                &self.main_menu_migrate_trim_output,
                &self.damascene_selection,
                MAIN_MENU_TRIM_OUTPUT_KEY,
                "saves/world.migrated.v4dw",
            ),
            row([
                main_menu_input(
                    "Keep Min Chunk",
                    &self.main_menu_migrate_trim_keep_min,
                    &self.damascene_selection,
                    MAIN_MENU_TRIM_MIN_KEY,
                    "0 -2 -2 -2",
                )
                .width(Size::Fill(1.0)),
                main_menu_input(
                    "Keep Max Chunk",
                    &self.main_menu_migrate_trim_keep_max,
                    &self.damascene_selection,
                    MAIN_MENU_TRIM_MAX_KEY,
                    "0 0 2 2",
                )
                .width(Size::Fill(1.0)),
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0)),
            main_menu_status(self.main_menu_migration_status.as_deref()),
            row([
                button("Run Migration")
                    .key(MAIN_MENU_TRIM_RUN_KEY)
                    .primary()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                button("Back")
                    .key(MAIN_MENU_BACK_KEY)
                    .ghost()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        main_menu_page_panel(
            "Legacy Keep-Bounds Trim",
            "Migration tool",
            body,
            680.0,
            Size::Hug,
        )
    }

    fn build_damascene_main_menu_migrate_v3_to_v4(&self) -> El {
        let body = column([
            main_menu_input(
                "Input v3 save root directory",
                &self.main_menu_migrate_v3_input,
                &self.damascene_selection,
                MAIN_MENU_V3_INPUT_KEY,
                "saves/world-v3",
            ),
            main_menu_input(
                "Output v4 save root directory",
                &self.main_menu_migrate_v3_output,
                &self.damascene_selection,
                MAIN_MENU_V4_OUTPUT_KEY,
                "saves/world-migrated-v4",
            ),
            toggle(
                MAIN_MENU_V3_OVERWRITE_KEY,
                self.main_menu_migrate_v3_overwrite,
                "Overwrite output directory",
            )
            .height(Size::Fixed(30.0))
            .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            main_menu_status(self.main_menu_migration_status.as_deref()),
            row([
                button("Run Migration")
                    .key(MAIN_MENU_V3_RUN_KEY)
                    .primary()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                button("Back")
                    .key(MAIN_MENU_BACK_KEY)
                    .ghost()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        main_menu_page_panel(
            "v3 Save Root -> v4",
            "Migration tool",
            body,
            680.0,
            Size::Hug,
        )
    }

    fn build_damascene_main_menu_multiplayer(&self) -> El {
        let body = column([
            main_menu_input(
                "Player name",
                &self.main_menu_player_name,
                &self.damascene_selection,
                MAIN_MENU_PLAYER_NAME_KEY,
                "Player",
            ),
            main_menu_input(
                "Server address",
                &self.main_menu_server_address,
                &self.damascene_selection,
                MAIN_MENU_SERVER_ADDRESS_KEY,
                "host:4000",
            ),
            main_menu_status(self.main_menu_connect_error.as_deref()),
            row([
                button("Connect")
                    .key(MAIN_MENU_CONNECT_KEY)
                    .primary()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                button("Back")
                    .key(MAIN_MENU_BACK_KEY)
                    .ghost()
                    .height(Size::Fixed(32.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            ])
            .gap(tokens::SPACE_2)
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0));

        main_menu_page_panel("Multiplayer", "Remote server", body, 460.0, Size::Hug)
    }

    pub(super) fn build_damascene_loading_overlay(&self) -> El {
        let panel = El::new(Kind::Custom("polychora_loading_panel"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Row)
            .children([
                spinner()
                    .width(Size::Fixed(24.0))
                    .height(Size::Fixed(24.0))
                    .fill(tokens::PRIMARY),
                column([
                    text("Loading world").bold(),
                    text("Streaming initial chunks").caption().muted(),
                ])
                .gap(1.0),
            ])
            .align(Align::Center)
            .gap(tokens::SPACE_3)
            .width(Size::Fixed(320.0))
            .height(Size::Hug)
            .padding(tokens::SPACE_4)
            .fill(tokens::POPOVER.with_alpha_u8(244))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG);
        build_damascene_center_modal_shell(panel, false)
    }

    pub(super) fn build_damascene_overlay(&self, info_readout: Option<&str>) -> Option<El> {
        if self.menu_open {
            return Some(build_damascene_center_modal_shell(
                self.build_damascene_pause_menu_panel(),
                true,
            ));
        }
        if self.teleport_dialog_open {
            return Some(build_damascene_center_modal_shell(
                self.build_damascene_teleport_panel(),
                false,
            ));
        }
        if self.block_gui_session.is_some() {
            return Some(build_damascene_center_modal_shell(
                self.build_damascene_block_gui_panel(),
                false,
            ));
        }

        let hotbar = self.build_damascene_hotbar();
        let orientation = self.build_damascene_orientation_controls();
        let modal = self
            .inventory_open
            .then(|| self.build_damascene_inventory_panel());
        let console = self
            .dev_console_open
            .then(|| self.build_damascene_dev_console_panel());
        let crosshair = (!self.inventory_open && !self.dev_console_open)
            .then(build_damascene_crosshair);
        let status = self
            .hud_status
            .as_ref()
            .filter(|(_, until)| std::time::Instant::now() < *until)
            .map(|(message, _)| build_damascene_hud_status(message));
        Some(build_damascene_overlay_shell(
            hotbar,
            orientation,
            self.build_damascene_waila_panel(),
            info_readout.map(build_damascene_info_readout_panel),
            modal,
            console,
            crosshair,
            status,
        ))
    }

    fn build_damascene_hotbar(&self) -> El {
        let slots = (0..9).map(|i| self.build_damascene_hotbar_slot(i));
        build_damascene_hotbar_from_slots(slots)
    }

    fn build_damascene_hotbar_slot(&self, index: usize) -> El {
        build_damascene_hotbar_slot_for_stack(
            &self.content_registry,
            self.material_icon_sheet.as_ref(),
            index,
            self.inventory.hotbar_slot(index),
            index == self.hotbar_selected_index,
        )
    }

    fn build_damascene_orientation_controls(&self) -> El {
        use polychora::shared::voxel::TesseractOrientation;

        let is_rotated = self.placement_orientation != TesseractOrientation::IDENTITY;
        let label = if is_rotated {
            format!("Ori: {}", self.placement_orientation.0)
        } else {
            "Ori: 0".to_string()
        };

        build_damascene_orientation_controls_view(label, is_rotated)
    }

    fn build_damascene_waila_panel(&self) -> Option<El> {
        let target = self.waila_target.as_ref()?;
        let panel = match target {
            WailaTarget::Block { coords, block } => {
                let entry = self
                    .content_registry
                    .block_entry(block.namespace, block.block_type);
                let name = entry.map(|e| e.name.as_str()).unwrap_or("Unknown");
                let category = entry.map(|e| e.category.label()).unwrap_or("Unknown");
                let ns_label = self.content_registry.namespace_label(block.namespace);
                let scale_label = if block.scale_exp != 0 {
                    format!("  scale: {}", block.scale_exp)
                } else {
                    String::new()
                };

                waila_panel(
                    name,
                    category,
                    [
                        format!(
                            "ns: {:#010x} ({})  type: {:#010x}",
                            block.namespace, ns_label, block.block_type
                        ),
                        format!(
                            "[{}, {}, {}, {}]{}",
                            coords[0], coords[1], coords[2], coords[3], scale_label
                        ),
                    ],
                )
            }
            WailaTarget::Entity {
                entity_id,
                entity_type_ns,
                entity_type,
                position,
                orientation,
                scale,
                data,
                distance,
            } => {
                let entry = self
                    .content_registry
                    .entity_lookup(*entity_type_ns, *entity_type);
                let canonical_name = entry
                    .map(|e| e.canonical_name.as_str())
                    .unwrap_or("unknown");
                let category = entry
                    .map(|e| format!("{:?}", e.category))
                    .unwrap_or_else(|| "Unknown".to_string());
                let player_name = self.remote_players.get(entity_id).map(|p| p.name.clone());
                let display_name = if let Some(name) = player_name {
                    format!("{} ({})", canonical_name, name)
                } else {
                    canonical_name.to_string()
                };
                let ns_label = self.content_registry.namespace_label(*entity_type_ns);
                let mut lines = vec![
                    format!(
                        "id: {}  ns: {:#010x} ({})  type: {:#010x}",
                        entity_id, entity_type_ns, ns_label, entity_type
                    ),
                    format!(
                        "pos: [{:.1}, {:.1}, {:.1}, {:.1}]",
                        position[0], position[1], position[2], position[3]
                    ),
                    format!(
                        "ori: [{:.2}, {:.2}, {:.2}, {:.2}]  scale: {:.2}",
                        orientation[0], orientation[1], orientation[2], orientation[3], scale
                    ),
                ];
                if let Some(entry) = entry {
                    if let Some(config) = &entry.sim_config {
                        lines.push(format!(
                            "{:?}: {:?} spd={:.1}",
                            config.mode, config.locomotion, config.move_speed
                        ));
                    }
                }
                if let Some(decoded) = format_cbor_for_display(data) {
                    lines.push(format!("data: {}", decoded));
                }

                waila_panel(
                    display_name,
                    format!("{} {:.1}m", category, distance),
                    lines,
                )
            }
        };

        Some(panel)
    }

    fn build_damascene_inventory_panel(&self) -> El {
        let content = match self.inventory_tab {
            InventoryTab::Creative => self.build_damascene_creative_inventory(),
            InventoryTab::Survival => self.build_damascene_survival_inventory(),
        };

        El::new(Kind::Custom("polychora_inventory_panel"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Column)
            .children([
                row([
                    text("Inventory").bold(),
                    spacer(),
                    tabs_list(
                        INVENTORY_TABS_KEY,
                        &inventory_tab_token(self.inventory_tab),
                        inventory_tab_options()
                            .map(|(tab, token)| (token, inventory_tab_label(tab))),
                    )
                    .width(Size::Fixed(220.0)),
                    button("Close")
                        .key(INVENTORY_CLOSE_KEY)
                        .secondary()
                        .height(Size::Fixed(28.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
                content,
                text("Tab or Esc closes inventory. Right-click a survival slot to drop one item.")
                    .caption()
                    .muted()
                    .width(Size::Fill(1.0)),
            ])
            .width(Size::Fixed(742.0))
            .height(Size::Fixed(520.0))
            .padding(tokens::SPACE_3)
            .gap(tokens::SPACE_3)
            .fill(tokens::POPOVER.with_alpha_u8(238))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG)
            .block_pointer()
    }

    fn build_damascene_creative_inventory(&self) -> El {
        let mut rows = Vec::new();
        let mut block_tiles = Vec::new();
        for entry in self.content_registry.all_blocks_ordered() {
            let icon = self.material_icon_sheet.as_ref().and_then(|sheet| {
                sheet.damascene_image(entry.texture.namespace, entry.texture.texture_id)
            });
            block_tiles.push(inventory_item_tile(
                format!(
                    "{INVENTORY_BLOCK_KEY_PREFIX}{}_{}",
                    entry.namespace, entry.block_type
                ),
                entry.name.clone(),
                entry.category.label(),
                Color::srgb_u8(entry.color[0], entry.color[1], entry.color[2]),
                icon,
                None,
                false,
            ));
        }
        rows.push(text("Blocks").caption().muted());
        rows.extend(tile_rows(block_tiles, 8));

        let mut entity_tiles = Vec::new();
        let mut entities: Vec<_> = self.content_registry.spawnable_entities().collect();
        entities.sort_by(|a, b| a.canonical_name.cmp(&b.canonical_name));
        for entity in entities {
            let icon = (entity.spawn_egg_texture_id != 0)
                .then_some(())
                .and_then(|_| {
                    self.material_icon_sheet
                        .as_ref()
                        .and_then(|sheet| sheet.damascene_image(0, entity.spawn_egg_texture_id))
                });
            entity_tiles.push(inventory_item_tile(
                format!(
                    "{INVENTORY_ENTITY_KEY_PREFIX}{}_{}",
                    entity.namespace, entity.entity_type
                ),
                entity.canonical_name.clone(),
                entity.category.label(),
                Color::srgb_u8(
                    entity.base_color[0],
                    entity.base_color[1],
                    entity.base_color[2],
                ),
                icon,
                None,
                false,
            ));
        }
        rows.push(text("Spawn Eggs").caption().muted());
        rows.extend(tile_rows(entity_tiles, 8));

        scroll(rows)
            .key("damascene_inventory_creative_scroll")
            .height(Size::Fixed(380.0))
            .width(Size::Fill(1.0))
    }

    fn build_damascene_survival_inventory(&self) -> El {
        let mut rows = Vec::new();

        for row_index in (1..4).rev() {
            let mut cells = Vec::new();
            for col in 0..INVENTORY_COLS {
                let slot_idx = row_index * INVENTORY_COLS + col;
                cells.push(self.build_damascene_inventory_slot(slot_idx, false));
            }
            rows.push(row(cells).gap(tokens::SPACE_1).width(Size::Hug));
        }

        rows.push(text("Hotbar").caption().muted());
        let mut hotbar = Vec::new();
        for col in 0..HOTBAR_SIZE {
            hotbar.push(self.build_damascene_inventory_slot(col, col == self.hotbar_selected_index));
        }
        rows.push(row(hotbar).gap(tokens::SPACE_1).width(Size::Hug));

        column(rows)
            .gap(tokens::SPACE_1)
            .align(Align::Center)
            .width(Size::Fill(1.0))
            .height(Size::Fixed(410.0))
    }

    fn build_damascene_inventory_slot(&self, slot_idx: usize, selected: bool) -> El {
        let (name, color, icon, count) = self
            .inventory
            .slot(slot_idx)
            .map(|stack| self.inventory_stack_view(stack))
            .unwrap_or_else(|| ("Empty".to_string(), Color::srgb_u8(32, 34, 40), None, 0));

        inventory_item_tile(
            format!("{INVENTORY_SLOT_KEY_PREFIX}{slot_idx}"),
            name,
            if slot_idx < HOTBAR_SIZE {
                format!("{}", slot_idx + 1)
            } else {
                String::new()
            },
            color,
            icon,
            (count > 1).then_some(count),
            selected,
        )
        .width(Size::Fixed(72.0))
    }

    fn inventory_stack_view(&self, stack: &ItemStack) -> (String, Color, Option<Image>, u32) {
        let tex = self
            .content_registry
            .resolve_item_thumbnail_texture(&stack.item);
        let icon = tex.and_then(|tex| {
            self.material_icon_sheet
                .as_ref()?
                .damascene_image(tex.namespace, tex.texture_id)
        });

        let (name, color) = if let Some(block) = stack.to_block_data() {
            let entry = self
                .content_registry
                .block_entry(block.namespace, block.block_type);
            let name = entry
                .map(|entry| entry.name.clone())
                .unwrap_or_else(|| "Unknown".to_string());
            let [r, g, b] = entry.map(|entry| entry.color).unwrap_or([128, 128, 128]);
            (name, Color::srgb_u8(r, g, b))
        } else if let Some((entity_ns, entity_type)) = stack.spawn_egg_entity_key() {
            let entry = self.content_registry.entity_lookup(entity_ns, entity_type);
            let name = entry
                .map(|entry| entry.canonical_name.clone())
                .unwrap_or_else(|| "Unknown".to_string());
            let [r, g, b] = entry
                .map(|entry| entry.base_color)
                .unwrap_or([128, 128, 128]);
            (name, Color::srgb_u8(r, g, b))
        } else {
            let [r, g, b] = self
                .content_registry
                .item_color(stack.item.namespace, stack.item.item_type);
            (
                self.content_registry
                    .item_name(stack.item.namespace, stack.item.item_type)
                    .to_string(),
                Color::srgb_u8(r, g, b),
            )
        };

        (name, color, icon, stack.count)
    }

    fn build_damascene_teleport_panel(&self) -> El {
        let coord_rows = ["X", "Y", "Z", "W"]
            .into_iter()
            .enumerate()
            .map(|(i, label)| {
                let key = format!("{TELEPORT_FIELD_KEY_PREFIX}{i}");
                row([
                    text(label).width(Size::Fixed(24.0)),
                    text_input_with(
                        &key,
                        &self.teleport_coords[i],
                        &self.damascene_selection,
                        TextInputOpts::default().placeholder("0.0"),
                    )
                    .width(Size::Fixed(150.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Hug)
            });

        let player_rows: Vec<El> = if self.multiplayer.is_some() && !self.remote_players.is_empty()
        {
            let mut sorted_ids: Vec<u64> = self.remote_players.keys().copied().collect();
            sorted_ids.sort();
            sorted_ids
                .into_iter()
                .filter_map(|entity_id| {
                    let player = self.remote_players.get(&entity_id)?;
                    let name = if player.name.is_empty() {
                        player
                            .owner_client_id
                            .map(|id| format!("Player {}", id))
                            .unwrap_or_else(|| format!("Entity {}", entity_id))
                    } else {
                        player.name.clone()
                    };
                    let pos = player.position;
                    Some(
                        button(format!(
                            "{} ({:.1}, {:.1}, {:.1}, {:.1})",
                            name, pos[0], pos[1], pos[2], pos[3],
                        ))
                        .key(format!("{TELEPORT_PLAYER_KEY_PREFIX}{entity_id}"))
                        .secondary()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0))
                        .width(Size::Fill(1.0)),
                    )
                })
                .collect()
        } else {
            vec![text("No remote players").caption().muted()]
        };

        El::new(Kind::Custom("polychora_teleport_dialog"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Column)
            .children([
                row([
                    column([
                        text("Teleport").bold(),
                        text("Coordinates").caption().muted(),
                    ])
                    .gap(1.0),
                    spacer(),
                    button("Close")
                        .key(TELEPORT_CLOSE_KEY)
                        .ghost()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
                column(coord_rows)
                    .gap(tokens::SPACE_2)
                    .width(Size::Fill(1.0)),
                row([
                    button("Teleport")
                        .key(TELEPORT_APPLY_KEY)
                        .primary()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                    button("Go to Origin")
                        .key(TELEPORT_ORIGIN_KEY)
                        .secondary()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Hug),
                settings_section(
                    "Players",
                    scroll(player_rows)
                        .key("damascene_teleport_players_scroll")
                        .height(Size::Fixed(120.0))
                        .width(Size::Fill(1.0)),
                ),
            ])
            .width(Size::Fixed(390.0))
            .height(Size::Hug)
            .padding(tokens::SPACE_4)
            .gap(tokens::SPACE_3)
            .fill(tokens::POPOVER.with_alpha_u8(244))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG)
            .block_pointer()
    }

    fn build_damascene_dev_console_panel(&self) -> El {
        let log_lines: Vec<El> = self
            .dev_console_log
            .iter()
            .rev()
            .take(28)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|line| mono(line.as_str()).caption().width(Size::Fill(1.0)))
            .collect();

        El::new(Kind::Custom("polychora_dev_console"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Column)
            .children([
                row([
                    column([
                        text("Developer Console").bold(),
                        text("Commands: /help, /tp, /spawn, /explode -- Up/Down: history")
                            .caption()
                            .muted(),
                    ])
                    .gap(1.0),
                    spacer(),
                    button("Close")
                        .key(DEV_CONSOLE_CLOSE_KEY)
                        .ghost()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
                scroll(log_lines)
                    .key("damascene_dev_console_log_scroll")
                    .height(Size::Fixed(178.0))
                    .width(Size::Fill(1.0)),
                row([
                    text_input_with(
                        DEV_CONSOLE_INPUT_KEY,
                        &self.dev_console_input,
                        &self.damascene_selection,
                        TextInputOpts::default().placeholder("e.g. /tp 0 8 0 0"),
                    )
                    .width(Size::Fill(1.0)),
                    button("Run")
                        .key(DEV_CONSOLE_RUN_KEY)
                        .primary()
                        .height(Size::Fixed(34.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ])
            .width(Size::Fixed(760.0))
            .height(Size::Hug)
            .padding(tokens::SPACE_3)
            .gap(tokens::SPACE_2)
            .fill(tokens::POPOVER.with_alpha_u8(244))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG)
            .block_pointer()
    }

    fn build_damascene_block_gui_panel(&self) -> El {
        let Some(session) = &self.block_gui_session else {
            return column(std::iter::empty::<El>());
        };

        let mut body = Vec::new();
        for group in &session.slot_groups {
            if let Some(label) = &group.label {
                body.push(text(label.as_str()).caption().muted());
            }

            let start = group.slot_start as usize;
            let end = (group.slot_end as usize).min(session.slots.len());
            let columns = (group.columns as usize).max(1);
            if start >= end {
                continue;
            }

            for (row_index, row_slots) in session.slots[start..end].chunks(columns).enumerate() {
                let cells = row_slots.iter().enumerate().map(|(slot_offset, slot)| {
                    let global_idx = start + row_index * columns + slot_offset;
                    self.build_damascene_block_gui_slot(slot, global_idx)
                });
                body.push(row(cells).gap(tokens::SPACE_1).width(Size::Hug));
            }
        }

        if session.show_player_inventory {
            body.push(text("Inventory").caption().muted());
            let player_start = session.block_slot_count as usize;
            if player_start < session.slots.len() {
                for (row_index, row_slots) in session.slots[player_start..].chunks(9).enumerate() {
                    let cells = row_slots.iter().enumerate().map(|(slot_offset, slot)| {
                        let global_idx = player_start + row_index * 9 + slot_offset;
                        self.build_damascene_block_gui_slot(slot, global_idx)
                    });
                    body.push(row(cells).gap(tokens::SPACE_1).width(Size::Hug));
                }
            }
        }

        let hint = if session.held_slot.is_some() {
            "Click a slot to place the item."
        } else {
            "Click a slot to pick up items."
        };

        El::new(Kind::Custom("polychora_block_gui"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Column)
            .children([
                row([
                    text(session.title.as_str()).bold(),
                    spacer(),
                    button("Close")
                        .key(BLOCK_GUI_CLOSE_KEY)
                        .ghost()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
                scroll(body)
                    .key("damascene_block_gui_slots_scroll")
                    .height(Size::Fixed(420.0))
                    .width(Size::Fill(1.0)),
                text(hint).caption().muted(),
            ])
            .width(Size::Fixed(640.0))
            .height(Size::Hug)
            .padding(tokens::SPACE_3)
            .gap(tokens::SPACE_2)
            .fill(tokens::POPOVER.with_alpha_u8(244))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG)
            .block_pointer()
    }

    fn build_damascene_block_gui_slot(
        &self,
        slot: &polychora_plugin_api::gui_abi::ItemSlot,
        global_idx: usize,
    ) -> El {
        let selected = self
            .block_gui_session
            .as_ref()
            .is_some_and(|session| session.held_slot == Some(global_idx as u32));
        let (name, color, icon) = if slot.is_empty() {
            ("Empty".to_string(), Color::srgb_u8(32, 34, 40), None)
        } else {
            let item = polychora::shared::protocol::Item {
                namespace: slot.item_ns,
                item_type: slot.item_type,
                data: slot.data.clone(),
            };
            let tex = self.content_registry.resolve_item_thumbnail_texture(&item);
            let icon = tex.and_then(|tex| {
                self.material_icon_sheet
                    .as_ref()?
                    .damascene_image(tex.namespace, tex.texture_id)
            });
            let [r, g, b] = self
                .content_registry
                .item_color(slot.item_ns, slot.item_type);
            (
                self.content_registry
                    .item_name(slot.item_ns, slot.item_type)
                    .to_string(),
                Color::srgb_u8(r, g, b),
                icon,
            )
        };

        inventory_item_tile(
            format!("{BLOCK_GUI_SLOT_KEY_PREFIX}{global_idx}"),
            name,
            format!("{}", global_idx + 1),
            color,
            icon,
            (slot.count > 1).then_some(slot.count),
            selected,
        )
        .width(Size::Fixed(72.0))
    }

    fn build_damascene_pause_menu_panel(&self) -> El {
        let body = if self.controls_dialog_open {
            self.build_damascene_controls_panel()
        } else {
            column([
                tabs_list(
                    PAUSE_SETTINGS_TABS_KEY,
                    &settings_page_token(self.settings_page),
                    SettingsPage::ALL.map(|page| (settings_page_token(page), page.label())),
                ),
                scroll([self.build_damascene_settings_page()])
                    .key("damascene_pause_settings_scroll")
                    .height(Size::Fill(1.0))
                    .width(Size::Fill(1.0)),
            ])
            .gap(tokens::SPACE_3)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0))
        };

        El::new(Kind::Custom("polychora_pause_menu"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Popover)
            .axis(Axis::Column)
            .children([
                row([
                    column([text("Polychora").bold(), text("Paused").caption().muted()]).gap(1.0),
                    spacer(),
                    tabs_list(
                        PAUSE_MENU_MODE_TABS_KEY,
                        &pause_menu_mode_token(self.controls_dialog_open),
                        [("settings", "Settings"), ("controls", "Controls")],
                    )
                    .width(Size::Fixed(230.0)),
                    button("Resume")
                        .key(PAUSE_RESUME_KEY)
                        .primary()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                    button("Main Menu")
                        .key(PAUSE_MAIN_MENU_KEY)
                        .secondary()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                    button("Quit")
                        .key(PAUSE_QUIT_KEY)
                        .ghost()
                        .height(Size::Fixed(30.0))
                        .padding(Sides::xy(tokens::SPACE_2, 0.0)),
                ])
                .align(Align::Center)
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
                body,
            ])
            .width(Size::Fixed(650.0))
            .height(Size::Fixed(520.0))
            .padding(tokens::SPACE_4)
            .gap(tokens::SPACE_3)
            .fill(tokens::POPOVER.with_alpha_u8(244))
            .stroke(tokens::BORDER)
            .radius(8.0)
            .shadow(tokens::SHADOW_LG)
            .block_pointer()
    }

    fn build_damascene_settings_page(&self) -> El {
        match self.settings_page {
            SettingsPage::Gameplay => self.build_damascene_gameplay_settings(),
            SettingsPage::Rendering => self.build_damascene_rendering_settings(),
            SettingsPage::Advanced => self.build_damascene_advanced_settings(),
            SettingsPage::Debug => self.build_damascene_debug_settings(),
        }
    }

    fn build_damascene_gameplay_settings(&self) -> El {
        let control_scheme_rows = tile_rows(
            control_scheme_options()
                .map(|(scheme, token)| {
                    setting_choice_tile(
                        format!("{PAUSE_CONTROL_SCHEME_KEY_PREFIX}{token}"),
                        scheme.label(),
                        self.control_scheme == scheme,
                    )
                })
                .collect(),
            4,
        );

        column([
            settings_section(
                "Control Scheme",
                column(control_scheme_rows)
                    .gap(tokens::SPACE_1)
                    .width(Size::Fill(1.0)),
            ),
            settings_section(
                "Info Panel",
                row(info_panel_options().map(|(mode, token)| {
                    setting_choice_tile(
                        format!("{PAUSE_INFO_PANEL_KEY_PREFIX}{token}"),
                        mode.label(),
                        self.info_panel_mode == mode,
                    )
                }))
                .gap(tokens::SPACE_1)
                .width(Size::Hug),
            ),
            settings_section(
                "Placement Preview",
                column([
                    row(placement_preview_options().map(|(mode, token)| {
                        setting_choice_tile(
                            format!("{PAUSE_PLACEMENT_PREVIEW_KEY_PREFIX}{token}"),
                            mode.label(),
                            self.placement_preview_mode == mode,
                        )
                    }))
                    .gap(tokens::SPACE_1)
                    .width(Size::Hug),
                    row([
                        setting_toggle(
                            PAUSE_TOGGLE_PREVIEW_HIDE_CAMERA_KEY,
                            self.placement_preview_hide_camera_intersect,
                            "Hide camera intersect",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_PREVIEW_HIDE_SAME_SCALE_KEY,
                            self.placement_preview_hide_same_scale,
                            "Hide same scale",
                        ),
                    ])
                    .gap(tokens::SPACE_1)
                    .width(Size::Hug),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
            settings_section(
                "Audio",
                column([
                    setting_slider(
                        "Master Volume",
                        format!("{:.0}%", self.audio.master_volume * 100.0),
                        PAUSE_SLIDER_MASTER_VOLUME_KEY,
                        self.audio.master_volume / 2.0,
                    ),
                    setting_slider(
                        "Spatial Falloff",
                        format!("{:.2}", self.audio.spatial_falloff_power),
                        PAUSE_SLIDER_SPATIAL_FALLOFF_KEY,
                        normalize_range(
                            self.audio.spatial_falloff_power,
                            AUDIO_SPATIAL_FALLOFF_POWER_MIN,
                            AUDIO_SPATIAL_FALLOFF_POWER_MAX,
                        ),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    }

    fn build_damascene_rendering_settings(&self) -> El {
        column([
            settings_section(
                "Projection",
                column([
                    setting_slider(
                        "Focal Length XY",
                        format!("{:.2}", self.focal_length_xy),
                        PAUSE_SLIDER_FOCAL_XY_KEY,
                        normalize_range(self.focal_length_xy, FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX),
                    ),
                    setting_slider(
                        "Focal Length ZW",
                        format!("{:.2}", self.focal_length_zw),
                        PAUSE_SLIDER_FOCAL_ZW_KEY,
                        normalize_range(self.focal_length_zw, FOCAL_LENGTH_MIN, FOCAL_LENGTH_MAX),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
            settings_section(
                "ZW Angle Color",
                column([
                    setting_toggle(
                        PAUSE_TOGGLE_ZW_SHIFT_KEY,
                        self.zw_angle_color_shift_enabled,
                        "Red/Blue Shift",
                    ),
                    setting_slider(
                        "Shift Strength",
                        format!("{:.2}", self.zw_angle_color_shift_strength),
                        PAUSE_SLIDER_ZW_SHIFT_KEY,
                        normalize_range(
                            self.zw_angle_color_shift_strength,
                            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
                            ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
                        ),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
            settings_section(
                "Voxel Traversal",
                column([
                    setting_slider(
                        "Max Trace Steps",
                        format!("{}", self.vte_max_trace_steps),
                        PAUSE_SLIDER_TRACE_STEPS_KEY,
                        normalize_range(
                            self.vte_max_trace_steps as f32,
                            VTE_TRACE_STEPS_MIN as f32,
                            VTE_TRACE_STEPS_MAX as f32,
                        ),
                    ),
                    setting_slider(
                        "Max Trace Distance",
                        format!("{:.0}", self.vte_max_trace_distance),
                        PAUSE_SLIDER_TRACE_DISTANCE_KEY,
                        normalize_range(
                            self.vte_max_trace_distance,
                            VTE_TRACE_DISTANCE_MIN,
                            VTE_TRACE_DISTANCE_MAX,
                        ),
                    ),
                    mono(format!(
                        "Resolution {}x{}x{}",
                        self.args.width, self.args.height, self.args.layers
                    ))
                    .caption()
                    .muted(),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    }

    fn build_damascene_advanced_settings(&self) -> El {
        column([
            settings_section(
                "Integral Sky + Emissive",
                column([
                    setting_toggle(
                        PAUSE_TOGGLE_INTEGRAL_SKY_KEY,
                        self.vte_integral_sky_emissive_enabled,
                        "Enabled",
                    ),
                    setting_slider(
                        "Sky Scale",
                        format!("{:.3}", self.vte_integral_sky_scale),
                        PAUSE_SLIDER_SKY_SCALE_KEY,
                        normalize_range(
                            self.vte_integral_sky_scale,
                            VTE_INTEGRAL_SKY_SCALE_MIN,
                            VTE_INTEGRAL_SKY_SCALE_MAX,
                        ),
                    ),
                    setting_slider(
                        "Hit Emissive",
                        format!("{:.3}", self.vte_integral_hit_emissive_boost),
                        PAUSE_SLIDER_HIT_EMISSIVE_KEY,
                        normalize_range(
                            self.vte_integral_hit_emissive_boost,
                            VTE_INTEGRAL_HIT_EMISSIVE_MIN,
                            VTE_INTEGRAL_HIT_EMISSIVE_MAX,
                        ),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
            settings_section(
                "Integral Log Merge",
                column([
                    setting_toggle(
                        PAUSE_TOGGLE_LOG_MERGE_KEY,
                        self.vte_integral_log_merge_enabled,
                        "Enabled",
                    ),
                    setting_slider(
                        "Log-Merge K",
                        format!("{:.2}", self.vte_integral_log_merge_k),
                        PAUSE_SLIDER_LOG_MERGE_KEY,
                        normalize_range(
                            self.vte_integral_log_merge_k,
                            VTE_INTEGRAL_LOG_MERGE_K_MIN,
                            VTE_INTEGRAL_LOG_MERGE_K_MAX,
                        ),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    }

    fn build_damascene_debug_settings(&self) -> El {
        column([
            settings_section(
                "Region-Tree Bounds",
                column([
                    row([
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_TREE_BOUNDS_KEY,
                            self.multiplayer_stream_tree_diag_enabled,
                            "Stream bounds",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_COMPARE_BOUNDS_KEY,
                            self.multiplayer_stream_tree_compare_diag_enabled,
                            "Mismatch bounds",
                        ),
                    ])
                    .gap(tokens::SPACE_1)
                    .width(Size::Hug),
                    row([
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_LABELS_KEY,
                            self.multiplayer_stream_tree_diag_labels_enabled,
                            "Labels",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_NON_EMPTY_KEY,
                            self.multiplayer_stream_tree_diag_non_empty_only,
                            "Non-empty only",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_SAMPLE_RAY_BOUNDS_KEY,
                            self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled,
                            "Sample ray",
                        ),
                    ])
                    .gap(tokens::SPACE_1)
                    .width(Size::Hug),
                    row([
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_BRANCH_KEY,
                            self.multiplayer_stream_tree_diag_show_branch_bounds,
                            "Branch",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_UNIFORM_KEY,
                            self.multiplayer_stream_tree_diag_show_uniform_bounds,
                            "Uniform",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_CHUNK_ARRAY_KEY,
                            self.multiplayer_stream_tree_diag_show_chunk_array_bounds,
                            "ChunkArray",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_PROCEDURAL_KEY,
                            self.multiplayer_stream_tree_diag_show_procedural_bounds,
                            "Procedural",
                        ),
                        setting_toggle(
                            PAUSE_TOGGLE_STREAM_EMPTY_KEY,
                            self.multiplayer_stream_tree_diag_show_empty_bounds,
                            "Empty",
                        ),
                    ])
                    .gap(tokens::SPACE_1)
                    .width(Size::Hug),
                    setting_slider(
                        "Bounds max nodes",
                        format!("{}", self.multiplayer_stream_tree_diag_max_nodes),
                        PAUSE_SLIDER_STREAM_MAX_NODES_KEY,
                        normalize_range(
                            self.multiplayer_stream_tree_diag_max_nodes as f32,
                            1.0,
                            4096.0,
                        ),
                    ),
                    setting_slider(
                        "Sample-ray max nodes",
                        format!("{}", self.multiplayer_stream_tree_diag_sample_ray_max_nodes),
                        PAUSE_SLIDER_SAMPLE_RAY_MAX_NODES_KEY,
                        normalize_range(
                            self.multiplayer_stream_tree_diag_sample_ray_max_nodes as f32,
                            1.0,
                            512.0,
                        ),
                    ),
                    setting_slider(
                        "Label max count",
                        format!("{}", self.multiplayer_stream_tree_diag_max_labels),
                        PAUSE_SLIDER_LABEL_MAX_COUNT_KEY,
                        normalize_range(
                            self.multiplayer_stream_tree_diag_max_labels as f32,
                            1.0,
                            512.0,
                        ),
                    ),
                    setting_slider(
                        "Mismatch sample cap",
                        format!("{}", self.multiplayer_stream_tree_compare_diag_max_chunks),
                        PAUSE_SLIDER_COMPARE_MAX_CHUNKS_KEY,
                        normalize_range(
                            self.multiplayer_stream_tree_compare_diag_max_chunks as f32,
                            1.0,
                            4096.0,
                        ),
                    ),
                    setting_slider(
                        "Mismatch log interval",
                        format!("{}", self.multiplayer_stream_tree_compare_diag_log_interval),
                        PAUSE_SLIDER_COMPARE_LOG_INTERVAL_KEY,
                        normalize_range(
                            self.multiplayer_stream_tree_compare_diag_log_interval as f32,
                            1.0,
                            1200.0,
                        ),
                    ),
                ])
                .gap(tokens::SPACE_2)
                .width(Size::Fill(1.0)),
            ),
            settings_section(
                "Tree Dumps",
                button("Dump world + render trees to stderr")
                    .key(PAUSE_DUMP_TREES_KEY)
                    .secondary()
                    .height(Size::Fixed(30.0))
                    .padding(Sides::xy(tokens::SPACE_2, 0.0)),
            ),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    }

    fn build_damascene_controls_panel(&self) -> El {
        scroll([
            controls_section(
                "Movement",
                [
                    ("W / A / S / D", "Move forward / left / backward / right"),
                    ("W W (double-tap)", "Toggle sprint"),
                    ("Space", "Jump (double-tap to toggle fly mode)"),
                    ("Shift", "Descend / Crouch"),
                    ("Q / E", "Move in 4D (W-axis negative / positive)"),
                ],
            ),
            controls_section(
                "Camera",
                [
                    ("Mouse", "Look around"),
                    ("R (hold)", "Reset orientation"),
                    ("F (hold)", "Pull to 3D"),
                    ("G", "Look at nearest block"),
                ],
            ),
            controls_section(
                "Building",
                [
                    ("Left Click", "Break block"),
                    ("Right Click", "Place block"),
                    ("Middle Click", "Pick material"),
                    ("[ / ]", "Scale down / up"),
                    ("Z / X / C", "Rotate block: XZ / YZ / XW"),
                    ("Scroll Wheel", "Cycle hotbar slot"),
                    ("1-9, 0", "Select hotbar slot"),
                    ("B", "Drop held item"),
                ],
            ),
            controls_section(
                "UI",
                [
                    ("Escape", "Open / close menu"),
                    ("Tab / I", "Toggle inventory"),
                    ("T", "Toggle teleport dialog"),
                    ("`", "Toggle developer console"),
                    ("F12", "Save screenshot"),
                    ("F8 / F10 / F11", "VTE dev toggles (sweep / sky / log-merge)"),
                ],
            ),
        ])
        .key("damascene_pause_controls_scroll")
        .height(Size::Fill(1.0))
        .width(Size::Fill(1.0))
    }
}

fn settings_section(title: &str, content: El) -> El {
    El::new(Kind::Custom("polychora_settings_section"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([text(title).bold(), content])
        .width(Size::Fill(1.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_3)
        .gap(tokens::SPACE_2)
        .fill(tokens::CARD.with_alpha_u8(170))
        .stroke(tokens::BORDER.with_alpha_u8(160))
        .radius(6.0)
}

fn main_menu_panel(
    header: impl IntoIterator<Item = El>,
    body: impl IntoIterator<Item = El>,
    width: f32,
    height: Size,
) -> El {
    El::new(Kind::Custom("polychora_main_menu"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Popover)
        .axis(Axis::Column)
        .children([
            column(header)
                .gap(tokens::SPACE_1)
                .align(Align::Center)
                .width(Size::Fill(1.0)),
            column(body)
                .gap(tokens::SPACE_2)
                .align(Align::Center)
                .width(Size::Fill(1.0)),
        ])
        .width(Size::Fixed(width))
        .height(height)
        .padding(tokens::SPACE_5)
        .gap(tokens::SPACE_4)
        .align(Align::Center)
        .fill(tokens::POPOVER.with_alpha_u8(244))
        .stroke(tokens::BORDER)
        .radius(8.0)
        .shadow(tokens::SHADOW_LG)
        .block_pointer()
}

fn main_menu_page_panel(title: &str, subtitle: &str, body: El, width: f32, height: Size) -> El {
    El::new(Kind::Custom("polychora_main_menu_page"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Popover)
        .axis(Axis::Column)
        .children([
            row([
                column([text(title).title().bold(), text(subtitle).caption().muted()]).gap(1.0),
                spacer(),
            ])
            .align(Align::Center)
            .width(Size::Fill(1.0)),
            body,
        ])
        .width(Size::Fixed(width))
        .height(height)
        .padding(tokens::SPACE_4)
        .gap(tokens::SPACE_3)
        .fill(tokens::POPOVER.with_alpha_u8(244))
        .stroke(tokens::BORDER)
        .radius(8.0)
        .shadow(tokens::SHADOW_LG)
        .block_pointer()
}

fn main_menu_section(title: &str, body: El) -> El {
    El::new(Kind::Custom("polychora_main_menu_section"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([text(title).caption().muted(), body])
        .width(Size::Fill(1.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_3)
        .gap(tokens::SPACE_2)
        .fill(tokens::CARD.with_alpha_u8(205))
        .stroke(tokens::BORDER.with_alpha_u8(170))
        .radius(6.0)
}

fn main_menu_action_card(title: &str, description: &str, key: &str) -> El {
    El::new(Kind::Custom("polychora_main_menu_action"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Row)
        .children([
            column([text(title).semibold(), text(description).caption().muted()])
                .gap(2.0)
                .width(Size::Fill(1.0)),
            button("Open")
                .key(key)
                .secondary()
                .height(Size::Fixed(30.0))
                .padding(Sides::xy(tokens::SPACE_2, 0.0)),
        ])
        .align(Align::Center)
        .width(Size::Fill(1.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_3)
        .gap(tokens::SPACE_3)
        .fill(tokens::CARD.with_alpha_u8(205))
        .stroke(tokens::BORDER.with_alpha_u8(170))
        .radius(6.0)
}

fn main_menu_input(
    label: &str,
    value: &str,
    selection: &damascene_core::Selection,
    key: &str,
    placeholder: &str,
) -> El {
    column([
        text(label).caption().muted(),
        text_input_with(
            key,
            value,
            selection,
            TextInputOpts::default().placeholder(placeholder),
        )
        .width(Size::Fill(1.0)),
    ])
    .gap(tokens::SPACE_1)
    .width(Size::Fill(1.0))
}

fn main_menu_status(status: Option<&str>) -> El {
    if let Some(status) = status {
        let is_error = status.starts_with("Error:") || status.starts_with("Failed");
        text(status)
            .caption()
            .width(Size::Fill(1.0))
            .color(if is_error {
                Color::srgb_u8(255, 120, 120)
            } else {
                Color::srgb_u8(130, 220, 150)
            })
    } else {
        column(std::iter::empty::<El>())
            .height(Size::Fixed(0.0))
            .width(Size::Fill(1.0))
    }
}

fn setting_choice_tile(key: String, label: &str, selected: bool) -> El {
    let button = button(label)
        .key(key)
        .height(Size::Fixed(28.0))
        .width(Size::Fixed(112.0))
        .padding(tokens::SPACE_1);
    if selected {
        button.primary()
    } else {
        button.secondary()
    }
}

fn setting_toggle(key: &str, pressed: bool, label: &str) -> El {
    toggle(key, pressed, label)
        .height(Size::Fixed(28.0))
        .padding(Sides::xy(tokens::SPACE_2, 0.0))
}

fn setting_slider(label: &str, value_label: String, key: &str, value: f32) -> El {
    row([
        column([
            text(label).caption().muted(),
            mono(value_label).caption().width(Size::Fixed(64.0)),
        ])
        .gap(1.0)
        .width(Size::Fixed(150.0)),
        damascene_slider::slider(key, value)
            .height(Size::Fixed(24.0))
            .width(Size::Fill(1.0)),
    ])
    .align(Align::Center)
    .gap(tokens::SPACE_3)
    .width(Size::Fill(1.0))
}

fn controls_section<'a>(title: &'a str, rows: impl IntoIterator<Item = (&'a str, &'a str)>) -> El {
    let rows = rows.into_iter().map(|(key, description)| {
        row([
            mono(key).bold().width(Size::Fixed(126.0)),
            text(description).caption().muted().width(Size::Fill(1.0)),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
    });
    settings_section(
        title,
        column(rows).gap(tokens::SPACE_1).width(Size::Fill(1.0)),
    )
}

fn normalize_range(value: f32, min: f32, max: f32) -> f32 {
    ((value - min) / (max - min).max(f32::EPSILON)).clamp(0.0, 1.0)
}

fn denormalize_range(value: f32, min: f32, max: f32) -> f32 {
    min + value.clamp(0.0, 1.0) * (max - min)
}

fn pause_menu_mode_token(controls_open: bool) -> &'static str {
    if controls_open {
        "controls"
    } else {
        "settings"
    }
}

fn pause_menu_mode_from_token(token: &str) -> Option<bool> {
    match token {
        "settings" => Some(false),
        "controls" => Some(true),
        _ => None,
    }
}

fn world_generator_token(generator: polychora::server::WorldGeneratorKind) -> &'static str {
    match generator {
        polychora::server::WorldGeneratorKind::FlatFloor => "flat_floor",
        polychora::server::WorldGeneratorKind::MassivePlatforms => "massive_platforms",
    }
}

fn world_generator_from_token(token: &str) -> Option<polychora::server::WorldGeneratorKind> {
    match token {
        "flat_floor" => Some(polychora::server::WorldGeneratorKind::FlatFloor),
        "massive_platforms" => Some(polychora::server::WorldGeneratorKind::MassivePlatforms),
        _ => None,
    }
}

fn settings_page_token(page: SettingsPage) -> &'static str {
    match page {
        SettingsPage::Gameplay => "gameplay",
        SettingsPage::Rendering => "rendering",
        SettingsPage::Advanced => "advanced",
        SettingsPage::Debug => "debug",
    }
}

fn settings_page_from_token(token: &str) -> Option<SettingsPage> {
    match token {
        "gameplay" => Some(SettingsPage::Gameplay),
        "rendering" => Some(SettingsPage::Rendering),
        "advanced" => Some(SettingsPage::Advanced),
        "debug" => Some(SettingsPage::Debug),
        _ => None,
    }
}

fn inventory_tab_options() -> impl Iterator<Item = (InventoryTab, &'static str)> {
    [
        (InventoryTab::Creative, "creative"),
        (InventoryTab::Survival, "survival"),
    ]
    .into_iter()
}

fn inventory_tab_token(tab: InventoryTab) -> &'static str {
    match tab {
        InventoryTab::Creative => "creative",
        InventoryTab::Survival => "survival",
    }
}

fn inventory_tab_label(tab: InventoryTab) -> &'static str {
    match tab {
        InventoryTab::Creative => "Creative",
        InventoryTab::Survival => "Survival",
    }
}

fn inventory_tab_from_token(token: &str) -> Option<InventoryTab> {
    inventory_tab_options()
        .find(|(_, option_token)| *option_token == token)
        .map(|(tab, _)| tab)
}

fn control_scheme_options() -> impl Iterator<Item = (ControlScheme, &'static str)> {
    [
        (ControlScheme::IntuitiveUpright, "upright"),
        (ControlScheme::LookTransport, "look"),
        (ControlScheme::TransportUniform, "uniform"),
        (ControlScheme::TransportDecoupled, "decoupled"),
        (ControlScheme::TransportScaled, "scaled"),
        (ControlScheme::RotorFree, "rotor"),
        (ControlScheme::LegacySideButtonLayers, "legacy_side"),
        (ControlScheme::LegacyScrollCycle, "legacy_scroll"),
    ]
    .into_iter()
}

fn control_scheme_from_token(token: &str) -> Option<ControlScheme> {
    control_scheme_options()
        .find(|(_, option_token)| *option_token == token)
        .map(|(scheme, _)| scheme)
}

fn info_panel_options() -> impl Iterator<Item = (InfoPanelMode, &'static str)> {
    [
        (InfoPanelMode::Full, "full"),
        (InfoPanelMode::VectorTable, "vectors"),
        (InfoPanelMode::VectorTable2, "vectors2"),
        (InfoPanelMode::Off, "off"),
    ]
    .into_iter()
}

fn info_panel_from_token(token: &str) -> Option<InfoPanelMode> {
    info_panel_options()
        .find(|(_, option_token)| *option_token == token)
        .map(|(mode, _)| mode)
}

fn placement_preview_options() -> impl Iterator<Item = (PlacementPreviewMode, &'static str)> {
    [
        (PlacementPreviewMode::Ghost, "ghost"),
        (PlacementPreviewMode::Wireframe, "wireframe"),
        (PlacementPreviewMode::Off, "off"),
    ]
    .into_iter()
}

fn placement_preview_from_token(token: &str) -> Option<PlacementPreviewMode> {
    placement_preview_options()
        .find(|(_, option_token)| *option_token == token)
        .map(|(mode, _)| mode)
}

fn build_damascene_center_modal_shell(modal: El, roomy: bool) -> El {
    stack([modal]).fill_size().layout(move |cx| {
        let (measured_w, measured_h) = (cx.measure)(&cx.children[0]);
        let margin_x = 24.0_f32.min(cx.container.w * 0.08);
        let margin_y = 24.0_f32.min(cx.container.h * 0.08);
        let max_w = (cx.container.w - margin_x * 2.0).max(280.0);
        let max_h = (cx.container.h - margin_y * 2.0).max(280.0);
        let target_w = if roomy {
            (cx.container.w * 0.82).clamp(measured_w, 1120.0)
        } else {
            measured_w
        };
        let target_h = if roomy {
            (cx.container.h * 0.78).clamp(measured_h, 860.0)
        } else {
            measured_h
        };
        let modal_w = target_w.min(max_w);
        let modal_h = target_h.min(max_h);
        vec![Rect::new(
            cx.container.x + ((cx.container.w - modal_w) * 0.5).max(margin_x),
            cx.container.y + ((cx.container.h - modal_h) * 0.5).max(margin_y),
            modal_w,
            modal_h,
        )]
    })
}

fn build_damascene_overlay_shell(
    hotbar: El,
    orientation: El,
    waila: Option<El>,
    info: Option<El>,
    inventory: Option<El>,
    console: Option<El>,
    crosshair: Option<El>,
    status: Option<El>,
) -> El {
    let has_waila = waila.is_some();
    let has_info = info.is_some();
    let has_inventory = inventory.is_some();
    let has_console = console.is_some();
    let has_crosshair = crosshair.is_some();
    let has_status = status.is_some();
    let mut children = vec![hotbar, orientation];
    if let Some(waila) = waila {
        children.push(waila);
    }
    if let Some(info) = info {
        children.push(info);
    }
    if let Some(inventory) = inventory {
        children.push(inventory);
    }
    if let Some(console) = console {
        children.push(console);
    }
    if let Some(crosshair) = crosshair {
        children.push(crosshair);
    }
    if let Some(status) = status {
        children.push(status);
    }
    let info_index = has_info.then_some(2 + usize::from(has_waila));
    let inventory_index =
        has_inventory.then_some(2 + usize::from(has_waila) + usize::from(has_info));
    let console_index = has_console
        .then_some(2 + usize::from(has_waila) + usize::from(has_info) + usize::from(has_inventory));
    let crosshair_index = has_crosshair.then_some(
        2 + usize::from(has_waila)
            + usize::from(has_info)
            + usize::from(has_inventory)
            + usize::from(has_console),
    );
    let status_index = has_status.then_some(
        2 + usize::from(has_waila)
            + usize::from(has_info)
            + usize::from(has_inventory)
            + usize::from(has_console)
            + usize::from(has_crosshair),
    );

    stack(children).fill_size().layout(move |cx| {
        let (hotbar_w, hotbar_h) = (cx.measure)(&cx.children[0]);
        let centered_hotbar_x = cx.container.x + (cx.container.w - hotbar_w) * 0.5;
        let max_hotbar_x = (cx.container.right() - hotbar_w - 12.0).max(cx.container.x + 12.0);
        let min_hotbar_x = (cx.container.x + NAV_HUD_BOTTOM_LEFT_RESERVED_WIDTH).min(max_hotbar_x);
        let hotbar_x = centered_hotbar_x.clamp(min_hotbar_x, max_hotbar_x);
        let hotbar_y = cx.container.bottom() - hotbar_h - 12.0;
        let hotbar_rect = Rect::new(hotbar_x, hotbar_y, hotbar_w, hotbar_h);

        let mut rects = Vec::with_capacity(cx.children.len());
        for (index, child) in cx.children.iter().enumerate() {
            let (measured_w, measured_h) = (cx.measure)(child);
            let rect = match index {
                0 => hotbar_rect,
                1 => {
                    let left_x = hotbar_rect.x - measured_w - 10.0;
                    if left_x >= cx.container.x + 12.0 {
                        Rect::new(
                            left_x,
                            hotbar_rect.y + (hotbar_rect.h - measured_h) * 0.5,
                            measured_w,
                            measured_h,
                        )
                    } else {
                        Rect::new(
                            cx.container.x + ((cx.container.w - measured_w) * 0.5).max(12.0),
                            (hotbar_rect.y - measured_h - 10.0).max(cx.container.y + 12.0),
                            measured_w,
                            measured_h,
                        )
                    }
                }
                index if Some(index) == info_index => {
                    let width = measured_w.min((cx.container.w - 24.0).max(260.0));
                    Rect::new(
                        cx.container.x + 12.0,
                        cx.container.y + 18.0,
                        width,
                        measured_h,
                    )
                }
                index if Some(index) == inventory_index => {
                    let width = measured_w.min((cx.container.w - 24.0).max(280.0));
                    let height = measured_h.min((cx.container.h - 24.0).max(280.0));
                    Rect::new(
                        cx.container.x + ((cx.container.w - width) * 0.5).max(12.0),
                        cx.container.y + ((cx.container.h - height) * 0.5).max(12.0),
                        width,
                        height,
                    )
                }
                index if Some(index) == console_index => {
                    let width = measured_w.min((cx.container.w - 24.0).max(280.0));
                    Rect::new(
                        cx.container.x + ((cx.container.w - width) * 0.5).max(12.0),
                        cx.container.y + 14.0,
                        width,
                        measured_h,
                    )
                }
                index if Some(index) == crosshair_index => Rect::new(
                    cx.container.x + (cx.container.w - measured_w) * 0.5,
                    cx.container.y + (cx.container.h - measured_h) * 0.5,
                    measured_w,
                    measured_h,
                ),
                index if Some(index) == status_index => {
                    // Sit above the hotbar and the orientation panel (which
                    // falls back to centered-above-hotbar in narrow windows).
                    let anchor_y = rects
                        .get(1)
                        .map(|rect: &Rect| rect.y)
                        .unwrap_or(hotbar_rect.y)
                        .min(hotbar_rect.y);
                    Rect::new(
                        cx.container.x + (cx.container.w - measured_w) * 0.5,
                        (anchor_y - measured_h - 10.0).max(cx.container.y + 12.0),
                        measured_w,
                        measured_h,
                    )
                }
                _ => {
                    let width = measured_w.min((cx.container.w - 24.0).max(260.0));
                    Rect::new(
                        cx.container.x + ((cx.container.w - width) * 0.5).max(12.0),
                        cx.container.y + 30.0,
                        width,
                        measured_h,
                    )
                }
            };
            rects.push(rect);
        }
        rects
    })
}

/// Small center-screen aim reticle: a plus sign of two translucent bars.
fn build_damascene_crosshair() -> El {
    let bar = |w: f32, h: f32| {
        El::new(Kind::Group)
            .width(Size::Fixed(w))
            .height(Size::Fixed(h))
            .fill(Color::srgb_u8(235, 240, 245).with_alpha_u8(190))
    };
    stack([bar(16.0, 2.0), bar(2.0, 16.0)])
        .width(Size::Fixed(16.0))
        .height(Size::Fixed(16.0))
        .layout(|cx| {
            cx.children
                .iter()
                .map(|child| {
                    let (w, h) = (cx.measure)(child);
                    Rect::new(
                        cx.container.x + (cx.container.w - w) * 0.5,
                        cx.container.y + (cx.container.h - h) * 0.5,
                        w,
                        h,
                    )
                })
                .collect()
        })
}

/// Transient status toast shown above the hotbar (sprint/scale/scheme changes).
fn build_damascene_hud_status(message: &str) -> El {
    El::new(Kind::Custom("polychora_hud_status"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Row)
        .children([text(message).caption()])
        .width(Size::Hug)
        .height(Size::Hug)
        .padding(Sides::xy(tokens::SPACE_2, tokens::SPACE_1))
        .fill(tokens::CARD.with_alpha_u8(205))
        .stroke(tokens::BORDER.with_alpha_u8(150))
        .radius(6.0)
        .shadow(tokens::SHADOW_SM)
}

fn build_damascene_info_readout_panel(readout: &str) -> El {
    if let Some(table) = build_damascene_vector_readout_table(readout) {
        return table;
    }

    let lines = readout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| mono(line).caption().muted().width(Size::Fill(1.0)));

    El::new(Kind::Custom("polychora_info_readout"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children(lines)
        .width(Size::Fixed(390.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_2)
        .gap(tokens::SPACE_1)
        .fill(tokens::CARD.with_alpha_u8(190))
        .stroke(tokens::BORDER.with_alpha_u8(150))
        .radius(6.0)
        .shadow(tokens::SHADOW_SM)
}

fn build_damascene_vector_readout_table(readout: &str) -> Option<El> {
    let mut lines = readout.lines().filter(|line| !line.trim().is_empty());
    let header_line = lines.next()?;
    let headers = split_vector_table_line(header_line)?;
    if headers.first().map(String::as_str) != Some("vec") || headers.len() != 5 {
        return None;
    }

    let mut rows = Vec::new();
    for line in lines {
        if line.contains("---") {
            continue;
        }
        let cells = split_vector_table_line(line)?;
        if cells.len() == headers.len() {
            rows.push(cells);
        }
    }
    if rows.is_empty() {
        return None;
    }

    let header = table_header([table_row(headers.iter().enumerate().map(
        |(index, label)| {
            if index == 0 {
                table_head(label.as_str()).width(Size::Fixed(72.0))
            } else {
                table_head(label.as_str()).center_text()
            }
        },
    ))]);

    let body_rows = rows.into_iter().map(|cells| {
        table_row(cells.into_iter().enumerate().map(|(index, value)| {
            let cell = table_cell(mono(value).caption().center_text().width(Size::Fill(1.0)));
            if index == 0 {
                cell.width(Size::Fixed(72.0))
            } else {
                cell
            }
        }))
        .height(Size::Fixed(30.0))
        .padding(tokens::SPACE_1)
        .gap(tokens::SPACE_1)
    });

    Some(
        El::new(Kind::Custom("polychora_vector_readout"))
            .style_profile(StyleProfile::Surface)
            .surface_role(SurfaceRole::Panel)
            .axis(Axis::Column)
            .children([
                text("Vectors").caption().muted(),
                table([header, table_body(body_rows)]),
            ])
            .width(Size::Fixed(392.0))
            .height(Size::Hug)
            .padding(tokens::SPACE_2)
            .gap(tokens::SPACE_1)
            .fill(tokens::CARD.with_alpha_u8(190))
            .stroke(tokens::BORDER.with_alpha_u8(150))
            .radius(6.0)
            .shadow(tokens::SHADOW_SM),
    )
}

fn split_vector_table_line(line: &str) -> Option<Vec<String>> {
    let cells: Vec<String> = line
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect();
    cells.iter().all(|cell| !cell.is_empty()).then_some(cells)
}

fn tile_rows(mut tiles: Vec<El>, per_row: usize) -> Vec<El> {
    let mut rows = Vec::new();
    while !tiles.is_empty() {
        let take = tiles.len().min(per_row);
        let row_tiles: Vec<_> = tiles.drain(..take).collect();
        rows.push(row(row_tiles).gap(tokens::SPACE_1).width(Size::Hug));
    }
    rows
}

fn inventory_item_tile(
    key: String,
    name: String,
    meta: impl Into<String>,
    color: Color,
    icon: Option<Image>,
    count: Option<u32>,
    selected: bool,
) -> El {
    let icon = if let Some(icon) = icon {
        damascene_image(icon)
            .image_fit(ImageFit::Contain)
            .width(Size::Fixed(42.0))
            .height(Size::Fixed(28.0))
            .radius(4.0)
    } else {
        let color = if color == Color::srgb_u8(220, 220, 220) {
            tokens::MUTED
        } else {
            color
        };
        column(std::iter::empty::<El>())
            .width(Size::Fixed(42.0))
            .height(Size::Fixed(28.0))
            .fill(color)
            .radius(4.0)
    };

    El::new(Kind::Custom("polychora_inventory_item"))
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([
            row([
                text(meta.into())
                    .caption()
                    .muted()
                    .ellipsis()
                    .max_lines(1)
                    .width(Size::Fill(1.0)),
                if let Some(count) = count {
                    badge(format!("{count}")).muted()
                } else {
                    column(std::iter::empty::<El>()).width(Size::Fixed(0.0))
                },
            ])
            .width(Size::Fill(1.0)),
            icon,
            text(short_label(name))
                .caption()
                .center_text()
                .ellipsis()
                .max_lines(1)
                .width(Size::Fixed(64.0)),
        ])
        .key(key)
        .focusable()
        .cursor(Cursor::Pointer)
        .width(Size::Fixed(84.0))
        .height(Size::Fixed(76.0))
        .padding(5.0)
        .gap(2.0)
        .align(Align::Center)
        .fill(tokens::CARD.with_alpha_u8(205))
        .stroke(if selected {
            tokens::WARNING
        } else {
            tokens::BORDER.with_alpha_u8(170)
        })
        .stroke_width(if selected { 2.0 } else { 1.0 })
        .radius(6.0)
}

fn build_damascene_hotbar_from_slots(slots: impl IntoIterator<Item = El>) -> El {
    row(slots)
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .height(Size::Hug)
        .width(Size::Hug)
}

fn build_damascene_hotbar_slot_for_stack(
    content_registry: &ContentRegistry,
    material_icon_sheet: Option<&MaterialIconSheet>,
    index: usize,
    stack: &Option<ItemStack>,
    selected: bool,
) -> El {
    let (name, count, scale_label, color, icon) = stack
        .as_ref()
        .map(|stack| {
            let tex = content_registry.resolve_item_thumbnail_texture(&stack.item);
            let icon =
                tex.and_then(|tex| material_icon_sheet?.damascene_image(tex.namespace, tex.texture_id));

            let (name, scale_label, color) = if let Some(block) = stack.to_block_data() {
                let entry = content_registry.block_entry(block.namespace, block.block_type);
                let name = entry
                    .map(|entry| entry.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let scale_label = (block.scale_exp != 0).then(|| format!("s{}", block.scale_exp));
                let [r, g, b] = entry.map(|entry| entry.color).unwrap_or([128, 128, 128]);
                (name, scale_label, Color::srgb_u8(r, g, b))
            } else if let Some((entity_ns, entity_type)) = stack.spawn_egg_entity_key() {
                let entry = content_registry.entity_lookup(entity_ns, entity_type);
                let name = entry
                    .map(|entry| entry.canonical_name.clone())
                    .unwrap_or_else(|| "Unknown".to_string());
                let [r, g, b] = entry
                    .map(|entry| entry.base_color)
                    .unwrap_or([128, 128, 128]);
                (name, None, Color::srgb_u8(r, g, b))
            } else {
                let [r, g, b] =
                    content_registry.item_color(stack.item.namespace, stack.item.item_type);
                (
                    content_registry
                        .item_name(stack.item.namespace, stack.item.item_type)
                        .to_string(),
                    None,
                    Color::srgb_u8(r, g, b),
                )
            };

            (name, stack.count, scale_label, color, icon)
        })
        .unwrap_or_else(|| ("Empty".to_string(), 0, None, Color::srgb_u8(44, 48, 58), None));

    build_damascene_hotbar_slot_view(
        index,
        HotbarSlotView {
            name,
            count,
            scale_label,
            color,
            icon,
            selected,
        },
    )
}

#[track_caller]
fn build_damascene_hotbar_slot_view(index: usize, slot: HotbarSlotView) -> El {
    let label = short_label(slot.name);
    let scale_label = slot.scale_label.unwrap_or_default();
    let icon = if let Some(icon) = slot.icon {
        damascene_image(icon)
            .image_fit(ImageFit::Contain)
            .width(Size::Fixed(44.0))
            .height(Size::Fixed(30.0))
            .radius(4.0)
    } else {
        column(std::iter::empty::<El>())
            .width(Size::Fixed(44.0))
            .height(Size::Fixed(30.0))
            .fill(slot.color)
            .radius(4.0)
    };

    let slot_body = El::new(Kind::Custom("polychora_hotbar_slot_body"))
        .at_loc(Location::caller())
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([
            row([
                text(format!("{}", index + 1)).caption().muted(),
                spacer(),
                if slot.count > 1 {
                    badge(format!("{}", slot.count)).muted()
                } else {
                    text("")
                },
            ])
            .width(Size::Fill(1.0))
            .align(Align::Center),
            icon,
            row([
                text(scale_label)
                    .caption()
                    .color(Color::srgb_u8(140, 200, 255))
                    .max_lines(1)
                    .width(Size::Fixed(22.0)),
                text(label)
                    .caption()
                    .center_text()
                    .ellipsis()
                    .max_lines(1)
                    .width(Size::Fill(1.0)),
                column(std::iter::empty::<El>()).width(Size::Fixed(22.0)),
            ])
            .width(Size::Fill(1.0))
            .align(Align::Center),
        ])
        .width(Size::Fixed(80.0))
        .height(Size::Fixed(82.0))
        .padding(6.0)
        .gap(2.0)
        .align(Align::Center)
        .fill(tokens::CARD.with_alpha_u8(205))
        .stroke(if slot.selected {
            tokens::WARNING.with_alpha_u8(155)
        } else {
            tokens::BORDER.with_alpha_u8(180)
        })
        .stroke_width(1.0)
        .radius(6.0);

    El::new(Kind::Custom("polychora_hotbar_slot"))
        .at_loc(Location::caller())
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Column)
        .children([slot_body])
        .key(format!("damascene_hotbar_slot_{index}"))
        .focusable()
        .cursor(Cursor::Pointer)
        .width(Size::Fixed(80.0))
        .height(Size::Fixed(82.0))
        .padding(0.0)
        .align(Align::Center)
        .fill(Color::srgb_u8a(0, 0, 0, 0))
        .stroke(if slot.selected {
            tokens::WARNING
        } else {
            Color::srgb_u8a(0, 0, 0, 0)
        })
        .stroke_width(if slot.selected { 2.5 } else { 0.0 })
        .paint_overflow(Sides::all(4.0))
        .radius(8.0)
        .shadow(if slot.selected {
            tokens::SHADOW_MD
        } else {
            0.0
        })
}

#[track_caller]
fn build_damascene_orientation_controls_view(label: String, is_rotated: bool) -> El {
    El::new(Kind::Custom("polychora_orientation_controls"))
        .at_loc(Location::caller())
        .style_profile(StyleProfile::Surface)
        .surface_role(SurfaceRole::Panel)
        .axis(Axis::Row)
        .children([
            column([
                row([
                    orientation_button("XZ", "xz", "Z key: rotate in XZ plane"),
                    orientation_button("YZ", "yz", "X key: rotate in YZ plane"),
                    orientation_button("XW", "xw", "C key: rotate in XW plane"),
                ])
                .gap(tokens::SPACE_1),
                row([
                    orientation_button("XY", "xy", "Rotate in XY plane"),
                    orientation_button("YW", "yw", "Rotate in YW plane"),
                    orientation_button("ZW", "zw", "Rotate in ZW plane"),
                ])
                .gap(tokens::SPACE_1),
            ])
            .gap(tokens::SPACE_1),
            column([
                button("Reset")
                    .key(format!("{ORIENTATION_KEY_PREFIX}reset"))
                    .tooltip("Reset orientation")
                    .secondary()
                    .width(Size::Fixed(60.0))
                    .height(Size::Fixed(24.0))
                    .padding(0.0),
                text(label)
                    .caption()
                    .center_text()
                    .width(Size::Fixed(60.0))
                    .color(if is_rotated {
                        Color::srgb_u8(210, 196, 255)
                    } else {
                        tokens::MUTED_FOREGROUND
                    }),
            ])
            .gap(tokens::SPACE_1),
        ])
        .width(Size::Fixed(218.0))
        .height(Size::Hug)
        .padding(tokens::SPACE_2)
        .gap(tokens::SPACE_2)
        .align(Align::Center)
        .fill(if is_rotated {
            Color::srgb_u8a(42, 34, 76, 210)
        } else {
            tokens::CARD.with_alpha_u8(205)
        })
        .stroke(tokens::BORDER.with_alpha_u8(170))
        .radius(6.0)
        .shadow(tokens::SHADOW_MD)
}

fn short_label(name: String) -> String {
    if name.chars().count() > 14 {
        let prefix = name.chars().take(11).collect::<String>();
        format!("{prefix}...")
    } else {
        name
    }
}

fn orientation_button(label: &str, action: &str, tooltip: &str) -> El {
    button(label)
        .key(format!("{ORIENTATION_KEY_PREFIX}{action}"))
        .tooltip(tooltip)
        .ghost()
        .width(Size::Fixed(38.0))
        .height(Size::Fixed(24.0))
        .padding(0.0)
}

fn waila_panel(
    title: impl Into<String>,
    badge_label: impl Into<String>,
    detail_lines: impl IntoIterator<Item = String>,
) -> El {
    let details: Vec<El> = detail_lines
        .into_iter()
        .map(|line| mono(line).caption().muted().width(Size::Fill(1.0)))
        .collect();

    card([
        card_header([row([
            card_title(title).line_height(tokens::TEXT_BASE.size),
            spacer(),
            badge(badge_label).info(),
        ])
        .align(Align::Center)
        .gap(tokens::SPACE_3)]),
        card_content([column(details).gap(tokens::SPACE_1).width(Size::Fill(1.0))]).pt(0.0),
    ])
    .width(Size::Fixed(500.0))
    .height(Size::Hug)
    .axis(Axis::Column)
    .justify(Justify::Start)
    .shadow(tokens::SHADOW_LG)
}

impl App {
    pub(super) fn handle_damascene_ui_events(&mut self, events: Vec<damascene_core::UiEvent>) -> bool {
        let mut consumed = false;
        for event in events {
            if let Some(selection) = event.selection.clone() {
                self.damascene_selection = selection;
                consumed = true;
            }
            let Some(route) = event.route() else {
                continue;
            };

            if route == MAIN_MENU_SINGLEPLAYER_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_page = MainMenuPage::Singleplayer;
                    self.main_menu_connect_error = None;
                    self.scan_world_files();
                }
            } else if route == MAIN_MENU_MULTIPLAYER_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_page = MainMenuPage::Multiplayer;
                    self.main_menu_connect_error = None;
                }
            } else if route == MAIN_MENU_QUIT_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.should_exit_after_render = true;
                }
            } else if route == MAIN_MENU_BACK_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_connect_error = None;
                    match self.main_menu_page {
                        MainMenuPage::SingleplayerMigrationLegacyTrim
                        | MainMenuPage::SingleplayerMigrationV3ToV4 => {
                            self.main_menu_page = MainMenuPage::SingleplayerMigrations;
                        }
                        MainMenuPage::SingleplayerMigrations => {
                            self.main_menu_page = MainMenuPage::Singleplayer;
                        }
                        MainMenuPage::Singleplayer | MainMenuPage::Multiplayer => {
                            self.main_menu_page = MainMenuPage::Root;
                            self.main_menu_migration_status = None;
                        }
                        MainMenuPage::Root => {}
                    }
                }
            } else if let Some(world_index) = damascene_main_menu_world_index(route) {
                consumed = true;
                if event.is_click_or_activate(route)
                    && world_index < self.main_menu_world_files.len()
                {
                    self.main_menu_selected_world = Some(world_index);
                }
            } else if route == MAIN_MENU_LOAD_SELECTED_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(entry) = self
                        .main_menu_selected_world
                        .and_then(|idx| self.main_menu_world_files.get(idx))
                    {
                        self.handle_damascene_main_menu_transition(MainMenuTransition::LoadWorld(
                            entry.path.clone(),
                        ));
                    }
                }
            } else if route == MAIN_MENU_CREATE_WORLD_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.handle_damascene_main_menu_transition(MainMenuTransition::NewWorld(
                        self.main_menu_new_world_generator,
                    ));
                }
            } else if route == MAIN_MENU_MIGRATIONS_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_migration_status = None;
                    self.main_menu_page = MainMenuPage::SingleplayerMigrations;
                }
            } else if tabs::apply_event(
                &mut self.main_menu_new_world_generator,
                &event,
                MAIN_MENU_WORLD_TYPE_TABS_KEY,
                world_generator_from_token,
            ) {
                consumed = true;
            } else if route == MAIN_MENU_PLAYER_NAME_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_player_name,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_SERVER_ADDRESS_KEY {
                consumed = true;
                if is_damascene_enter_key(&event) {
                    self.handle_damascene_main_menu_transition(MainMenuTransition::ConnectMultiplayer(
                        self.main_menu_server_address.clone(),
                    ));
                } else {
                    damascene_text_input::apply_event(
                        &mut self.main_menu_server_address,
                        &mut self.damascene_selection,
                        &event,
                        route,
                    );
                }
            } else if route == MAIN_MENU_CONNECT_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.handle_damascene_main_menu_transition(MainMenuTransition::ConnectMultiplayer(
                        self.main_menu_server_address.clone(),
                    ));
                }
            } else if route == MAIN_MENU_MIGRATE_LEGACY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_page = MainMenuPage::SingleplayerMigrationLegacyTrim;
                }
            } else if route == MAIN_MENU_MIGRATE_V3_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_page = MainMenuPage::SingleplayerMigrationV3ToV4;
                }
            } else if route == MAIN_MENU_TRIM_INPUT_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_trim_input,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_TRIM_OUTPUT_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_trim_output,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_TRIM_MIN_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_trim_keep_min,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_TRIM_MAX_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_trim_keep_max,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_TRIM_RUN_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.run_main_menu_migration_legacy_trim();
                }
            } else if route == MAIN_MENU_V3_INPUT_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_v3_input,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_V4_OUTPUT_KEY {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.main_menu_migrate_v3_output,
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == MAIN_MENU_V3_OVERWRITE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.main_menu_migrate_v3_overwrite = !self.main_menu_migrate_v3_overwrite;
                }
            } else if route == MAIN_MENU_V3_RUN_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.run_main_menu_migration_v3_to_v4();
                }
            } else if route == DEV_CONSOLE_INPUT_KEY {
                consumed = true;
                if is_damascene_enter_key(&event) {
                    self.submit_damascene_dev_console_input();
                } else if is_damascene_key_down(&event, damascene_core::UiKey::ArrowUp) {
                    self.dev_console_history_prev();
                } else if is_damascene_key_down(&event, damascene_core::UiKey::ArrowDown) {
                    self.dev_console_history_next();
                } else {
                    damascene_text_input::apply_event(
                        &mut self.dev_console_input,
                        &mut self.damascene_selection,
                        &event,
                        route,
                    );
                }
            } else if route == DEV_CONSOLE_RUN_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.submit_damascene_dev_console_input();
                }
            } else if route == DEV_CONSOLE_CLOSE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.close_dev_console();
                }
            } else if route == BLOCK_GUI_CLOSE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                        self.close_block_gui(&window);
                    }
                }
            } else if let Some(slot_index) = damascene_block_gui_slot_index(route) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.handle_damascene_block_gui_slot_click(slot_index);
                }
            } else if let Some(index) = damascene_teleport_field_index(route) {
                consumed = true;
                damascene_text_input::apply_event(
                    &mut self.teleport_coords[index],
                    &mut self.damascene_selection,
                    &event,
                    route,
                );
            } else if route == TELEPORT_APPLY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(pos) = self.parse_damascene_teleport_coords() {
                        self.apply_damascene_teleport_target(pos);
                    }
                }
            } else if route == TELEPORT_ORIGIN_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.apply_damascene_teleport_target([0.0, 0.0, 0.0, 0.0]);
                }
            } else if route == TELEPORT_CLOSE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.close_damascene_teleport_dialog();
                }
            } else if let Some(entity_id) = damascene_teleport_player_id(route) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(player) = self.remote_players.get(&entity_id) {
                        self.apply_damascene_teleport_target(player.position);
                    }
                }
            } else if route == PAUSE_RESUME_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.resume_damascene_pause_menu();
                }
            } else if tabs::apply_event(
                &mut self.controls_dialog_open,
                &event,
                PAUSE_MENU_MODE_TABS_KEY,
                pause_menu_mode_from_token,
            ) {
                consumed = true;
            } else if route == PAUSE_MAIN_MENU_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                        self.transition_to_main_menu(&window);
                    }
                }
            } else if route == PAUSE_QUIT_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.should_exit_after_render = true;
                }
            } else if tabs::apply_event(
                &mut self.settings_page,
                &event,
                PAUSE_SETTINGS_TABS_KEY,
                settings_page_from_token,
            ) {
                consumed = true;
            } else if let Some(token) = route.strip_prefix(PAUSE_CONTROL_SCHEME_KEY_PREFIX) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(scheme) = control_scheme_from_token(token) {
                        self.set_control_scheme(scheme);
                    }
                }
            } else if let Some(token) = route.strip_prefix(PAUSE_INFO_PANEL_KEY_PREFIX) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(mode) = info_panel_from_token(token) {
                        self.info_panel_mode = mode;
                    }
                }
            } else if let Some(token) = route.strip_prefix(PAUSE_PLACEMENT_PREVIEW_KEY_PREFIX) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if let Some(mode) = placement_preview_from_token(token) {
                        self.placement_preview_mode = mode;
                    }
                }
            } else if route == PAUSE_TOGGLE_PREVIEW_HIDE_CAMERA_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.placement_preview_hide_camera_intersect =
                        !self.placement_preview_hide_camera_intersect;
                }
            } else if route == PAUSE_TOGGLE_PREVIEW_HIDE_SAME_SCALE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.placement_preview_hide_same_scale =
                        !self.placement_preview_hide_same_scale;
                }
            } else if route == PAUSE_TOGGLE_ZW_SHIFT_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.zw_angle_color_shift_enabled = !self.zw_angle_color_shift_enabled;
                }
            } else if route == PAUSE_TOGGLE_INTEGRAL_SKY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.toggle_vte_integral_sky_emissive();
                }
            } else if route == PAUSE_TOGGLE_LOG_MERGE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.toggle_vte_integral_log_merge();
                }
            } else if route == PAUSE_TOGGLE_STREAM_TREE_BOUNDS_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_enabled =
                        !self.multiplayer_stream_tree_diag_enabled;
                }
            } else if route == PAUSE_TOGGLE_STREAM_COMPARE_BOUNDS_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_compare_diag_enabled =
                        !self.multiplayer_stream_tree_compare_diag_enabled;
                }
            } else if route == PAUSE_TOGGLE_STREAM_LABELS_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_labels_enabled =
                        !self.multiplayer_stream_tree_diag_labels_enabled;
                }
            } else if route == PAUSE_TOGGLE_STREAM_NON_EMPTY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_non_empty_only =
                        !self.multiplayer_stream_tree_diag_non_empty_only;
                }
            } else if route == PAUSE_TOGGLE_STREAM_BRANCH_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_show_branch_bounds =
                        !self.multiplayer_stream_tree_diag_show_branch_bounds;
                }
            } else if route == PAUSE_TOGGLE_STREAM_UNIFORM_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_show_uniform_bounds =
                        !self.multiplayer_stream_tree_diag_show_uniform_bounds;
                }
            } else if route == PAUSE_TOGGLE_STREAM_CHUNK_ARRAY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_show_chunk_array_bounds =
                        !self.multiplayer_stream_tree_diag_show_chunk_array_bounds;
                }
            } else if route == PAUSE_TOGGLE_STREAM_PROCEDURAL_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_show_procedural_bounds =
                        !self.multiplayer_stream_tree_diag_show_procedural_bounds;
                }
            } else if route == PAUSE_TOGGLE_STREAM_EMPTY_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_show_empty_bounds =
                        !self.multiplayer_stream_tree_diag_show_empty_bounds;
                }
            } else if route == PAUSE_TOGGLE_SAMPLE_RAY_BOUNDS_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled =
                        !self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled;
                }
            } else if route == PAUSE_DUMP_TREES_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.scene.dump_world_tree();
                    self.scene.dump_render_trees();
                    eprintln!("--- tree dump complete ---");
                }
            } else if route == PAUSE_SLIDER_MASTER_VOLUME_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.audio.master_volume,
                    &event,
                    route,
                    0.0,
                    2.0,
                    0.025,
                    0.125,
                );
            } else if route == PAUSE_SLIDER_SPATIAL_FALLOFF_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.audio.spatial_falloff_power,
                    &event,
                    route,
                    AUDIO_SPATIAL_FALLOFF_POWER_MIN,
                    AUDIO_SPATIAL_FALLOFF_POWER_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_FOCAL_XY_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.focal_length_xy,
                    &event,
                    route,
                    FOCAL_LENGTH_MIN,
                    FOCAL_LENGTH_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_FOCAL_ZW_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.focal_length_zw,
                    &event,
                    route,
                    FOCAL_LENGTH_MIN,
                    FOCAL_LENGTH_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_ZW_SHIFT_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.zw_angle_color_shift_strength,
                    &event,
                    route,
                    ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
                    ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_TRACE_STEPS_KEY {
                consumed = true;
                apply_slider_to_u32(
                    &mut self.vte_max_trace_steps,
                    &event,
                    route,
                    VTE_TRACE_STEPS_MIN,
                    VTE_TRACE_STEPS_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_TRACE_DISTANCE_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.vte_max_trace_distance,
                    &event,
                    route,
                    VTE_TRACE_DISTANCE_MIN,
                    VTE_TRACE_DISTANCE_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_SKY_SCALE_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.vte_integral_sky_scale,
                    &event,
                    route,
                    VTE_INTEGRAL_SKY_SCALE_MIN,
                    VTE_INTEGRAL_SKY_SCALE_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_HIT_EMISSIVE_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.vte_integral_hit_emissive_boost,
                    &event,
                    route,
                    VTE_INTEGRAL_HIT_EMISSIVE_MIN,
                    VTE_INTEGRAL_HIT_EMISSIVE_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_LOG_MERGE_KEY {
                consumed = true;
                apply_slider_to_f32(
                    &mut self.vte_integral_log_merge_k,
                    &event,
                    route,
                    VTE_INTEGRAL_LOG_MERGE_K_MIN,
                    VTE_INTEGRAL_LOG_MERGE_K_MAX,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_STREAM_MAX_NODES_KEY {
                consumed = true;
                apply_slider_to_usize(
                    &mut self.multiplayer_stream_tree_diag_max_nodes,
                    &event,
                    route,
                    1,
                    4096,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_SAMPLE_RAY_MAX_NODES_KEY {
                consumed = true;
                apply_slider_to_usize(
                    &mut self.multiplayer_stream_tree_diag_sample_ray_max_nodes,
                    &event,
                    route,
                    1,
                    512,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_LABEL_MAX_COUNT_KEY {
                consumed = true;
                apply_slider_to_usize(
                    &mut self.multiplayer_stream_tree_diag_max_labels,
                    &event,
                    route,
                    1,
                    512,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_COMPARE_MAX_CHUNKS_KEY {
                consumed = true;
                apply_slider_to_usize(
                    &mut self.multiplayer_stream_tree_compare_diag_max_chunks,
                    &event,
                    route,
                    1,
                    4096,
                    0.02,
                    0.10,
                );
            } else if route == PAUSE_SLIDER_COMPARE_LOG_INTERVAL_KEY {
                consumed = true;
                apply_slider_to_usize(
                    &mut self.multiplayer_stream_tree_compare_diag_log_interval,
                    &event,
                    route,
                    1,
                    1200,
                    0.02,
                    0.10,
                );
            } else if let Some(slot_index) = damascene_hotbar_slot_index(route) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.hotbar_selected_index = slot_index;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
            } else if route == INVENTORY_CLOSE_KEY {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.close_damascene_inventory();
                }
            } else if tabs::apply_event(
                &mut self.inventory_tab,
                &event,
                INVENTORY_TABS_KEY,
                inventory_tab_from_token,
            ) {
                consumed = true;
            } else if let Some((namespace, block_type)) =
                damascene_inventory_key_pair(route, INVENTORY_BLOCK_KEY_PREFIX)
            {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.inventory.set_slot(
                        self.hotbar_selected_index,
                        Some(ItemStack::block(namespace, block_type, 1, 0)),
                    );
                    self.inventory_dirty = true;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
            } else if let Some((namespace, entity_type)) =
                damascene_inventory_key_pair(route, INVENTORY_ENTITY_KEY_PREFIX)
            {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.inventory.set_slot(
                        self.hotbar_selected_index,
                        Some(ItemStack::spawn_egg(namespace, entity_type)),
                    );
                    self.inventory_dirty = true;
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                }
            } else if let Some(slot_index) = damascene_inventory_slot_index(route) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    if slot_index != self.hotbar_selected_index {
                        self.inventory
                            .swap_slots(slot_index, self.hotbar_selected_index);
                        self.inventory_dirty = true;
                    }
                    self.selected_block = block_data_from_slot(
                        self.inventory.hotbar_slot(self.hotbar_selected_index),
                    );
                } else if event.kind == UiEventKind::SecondaryClick
                    && event.is_route(route)
                    && self.inventory.slot(slot_index).is_some()
                {
                    self.inventory.decrement_slot(slot_index);
                    self.send_drop_item(slot_index as u8);
                    self.send_inventory_sync();
                    self.inventory_dirty = true;
                    if slot_index == self.hotbar_selected_index {
                        self.selected_block = block_data_from_slot(
                            self.inventory.hotbar_slot(self.hotbar_selected_index),
                        );
                    }
                }
            } else if let Some(action) = route.strip_prefix(ORIENTATION_KEY_PREFIX) {
                consumed = true;
                if event.is_click_or_activate(route) {
                    self.apply_damascene_orientation_action(action);
                }
            }
        }
        consumed
    }

    fn handle_damascene_main_menu_transition(&mut self, transition: MainMenuTransition) {
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.handle_main_menu_transition(transition, &window);
        }
    }

    fn close_damascene_inventory(&mut self) {
        self.inventory_open = false;
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.grab_mouse(&window);
        }
    }

    fn resume_damascene_pause_menu(&mut self) {
        self.menu_open = false;
        self.controls_dialog_open = false;
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.grab_mouse(&window);
        }
    }

    fn parse_damascene_teleport_coords(&self) -> Option<[f32; 4]> {
        Some([
            self.teleport_coords[0].parse().ok()?,
            self.teleport_coords[1].parse().ok()?,
            self.teleport_coords[2].parse().ok()?,
            self.teleport_coords[3].parse().ok()?,
        ])
    }

    fn apply_damascene_teleport_target(&mut self, pos: [f32; 4]) {
        self.camera.position = pos;
        self.teleport_dialog_open = false;
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.grab_mouse(&window);
        }
        eprintln!(
            "Teleported to ({:.1}, {:.1}, {:.1}, {:.1})",
            pos[0], pos[1], pos[2], pos[3],
        );
    }

    fn close_damascene_teleport_dialog(&mut self) {
        self.teleport_dialog_open = false;
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.grab_mouse(&window);
        }
    }

    pub(super) fn focus_damascene_dev_console_input(&mut self) {
        self.damascene_selection =
            damascene_core::Selection::caret(DEV_CONSOLE_INPUT_KEY, self.dev_console_input.len());
    }

    pub(super) fn handle_damascene_dev_console_key_fallback(
        &mut self,
        key: Option<damascene_core::UiKey>,
        text: Option<String>,
        modifiers: damascene_core::KeyModifiers,
        _repeat: bool,
    ) -> bool {
        if !self.dev_console_open {
            return false;
        }

        if !self.damascene_selection.is_within(DEV_CONSOLE_INPUT_KEY) {
            self.focus_damascene_dev_console_input();
        }

        if matches!(key, Some(damascene_core::UiKey::Enter)) {
            self.submit_damascene_dev_console_input();
            return true;
        }
        if matches!(key, Some(damascene_core::UiKey::Backspace)) {
            self.dev_console_input.pop();
            self.focus_damascene_dev_console_input();
            return true;
        }
        if matches!(key, Some(damascene_core::UiKey::ArrowUp)) {
            self.dev_console_history_prev();
            return true;
        }
        if matches!(key, Some(damascene_core::UiKey::ArrowDown)) {
            self.dev_console_history_next();
            return true;
        }

        let Some(text) = text else {
            return key.is_some();
        };
        if (modifiers.ctrl && !modifiers.alt) || modifiers.logo {
            return true;
        }
        let filtered: String = text.chars().filter(|ch| !ch.is_control()).collect();
        if !filtered.is_empty() {
            self.dev_console_input.push_str(&filtered);
            self.focus_damascene_dev_console_input();
        }
        true
    }

    fn submit_damascene_dev_console_input(&mut self) {
        let command = self.dev_console_input.trim().to_string();
        self.dev_console_input.clear();
        if !command.is_empty() {
            self.dev_console_history_push(&command);
            self.execute_dev_console_command(&command);
        }
        self.focus_damascene_dev_console_input();
    }

    fn handle_damascene_block_gui_slot_click(&mut self, slot_idx: u32) {
        let Some(held_idx) = self
            .block_gui_session
            .as_ref()
            .and_then(|session| session.held_slot)
        else {
            let non_empty = self
                .block_gui_session
                .as_ref()
                .and_then(|session| session.slots.get(slot_idx as usize))
                .is_some_and(|slot| !slot.is_empty());
            if non_empty {
                if let Some(session) = self.block_gui_session.as_mut() {
                    session.held_slot = Some(slot_idx);
                }
            }
            return;
        };

        if held_idx == slot_idx {
            if let Some(session) = self.block_gui_session.as_mut() {
                session.held_slot = None;
            }
            return;
        }

        let count = self
            .block_gui_session
            .as_ref()
            .and_then(|session| session.slots.get(held_idx as usize))
            .map(|slot| slot.count)
            .unwrap_or(0);
        if count == 0 {
            return;
        }

        let action = polychora_plugin_api::gui_abi::GuiAction::MoveStack {
            from_slot: held_idx,
            to_slot: slot_idx,
            count,
        };
        if let (Some(wasm), Some(session)) = (
            self.wasm_model_manager.as_mut(),
            self.block_gui_session.as_mut(),
        ) {
            let accepted = polychora::block_gui::send_gui_action(wasm, session, action);
            if accepted {
                session.held_slot = None;
            }
        }
    }

    fn apply_damascene_orientation_action(&mut self, action: &str) {
        use polychora::shared::voxel::TesseractOrientation;

        self.placement_orientation = match action {
            "xz" => TesseractOrientation::ROT_XZ.compose(self.placement_orientation),
            "yz" => TesseractOrientation::ROT_YZ.compose(self.placement_orientation),
            "xw" => TesseractOrientation::ROT_XW.compose(self.placement_orientation),
            "xy" => TesseractOrientation::ROT_XY.compose(self.placement_orientation),
            "yw" => TesseractOrientation::ROT_YW.compose(self.placement_orientation),
            "zw" => TesseractOrientation::ROT_ZW.compose(self.placement_orientation),
            "reset" => TesseractOrientation::IDENTITY,
            _ => self.placement_orientation,
        };
    }
}

fn apply_slider_to_f32(
    value: &mut f32,
    event: &damascene_core::UiEvent,
    key: &str,
    min: f32,
    max: f32,
    step: f32,
    page_step: f32,
) -> bool {
    let mut normalized = normalize_range(*value, min, max);
    if damascene_slider::apply_event(&mut normalized, event, key, step, page_step) {
        *value = denormalize_range(normalized, min, max);
        true
    } else {
        false
    }
}

fn apply_slider_to_u32(
    value: &mut u32,
    event: &damascene_core::UiEvent,
    key: &str,
    min: u32,
    max: u32,
    step: f32,
    page_step: f32,
) -> bool {
    let mut normalized = normalize_range(*value as f32, min as f32, max as f32);
    if damascene_slider::apply_event(&mut normalized, event, key, step, page_step) {
        *value = denormalize_range(normalized, min as f32, max as f32).round() as u32;
        true
    } else {
        false
    }
}

fn apply_slider_to_usize(
    value: &mut usize,
    event: &damascene_core::UiEvent,
    key: &str,
    min: usize,
    max: usize,
    step: f32,
    page_step: f32,
) -> bool {
    let mut normalized = normalize_range(*value as f32, min as f32, max as f32);
    if damascene_slider::apply_event(&mut normalized, event, key, step, page_step) {
        *value = denormalize_range(normalized, min as f32, max as f32).round() as usize;
        true
    } else {
        false
    }
}

fn is_damascene_enter_key(event: &damascene_core::UiEvent) -> bool {
    is_damascene_key_down(event, damascene_core::UiKey::Enter)
}

fn is_damascene_key_down(event: &damascene_core::UiEvent, key: damascene_core::UiKey) -> bool {
    event.kind == UiEventKind::KeyDown
        && event
            .key_press
            .as_ref()
            .is_some_and(|key_press| key_press.key == key)
}

fn damascene_hotbar_slot_index(route: &str) -> Option<usize> {
    let suffix = route.strip_prefix(HOTBAR_SLOT_KEY_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index < 9).then_some(index)
}

fn damascene_main_menu_world_index(route: &str) -> Option<usize> {
    route.strip_prefix(MAIN_MENU_WORLD_KEY_PREFIX)?.parse().ok()
}

fn damascene_inventory_slot_index(route: &str) -> Option<usize> {
    let suffix = route.strip_prefix(INVENTORY_SLOT_KEY_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index < polychora::shared::inventory::INVENTORY_SIZE).then_some(index)
}

fn damascene_teleport_field_index(route: &str) -> Option<usize> {
    let suffix = route.strip_prefix(TELEPORT_FIELD_KEY_PREFIX)?;
    let index = suffix.parse::<usize>().ok()?;
    (index < 4).then_some(index)
}

fn damascene_teleport_player_id(route: &str) -> Option<u64> {
    route.strip_prefix(TELEPORT_PLAYER_KEY_PREFIX)?.parse().ok()
}

fn damascene_block_gui_slot_index(route: &str) -> Option<u32> {
    route.strip_prefix(BLOCK_GUI_SLOT_KEY_PREFIX)?.parse().ok()
}

fn damascene_inventory_key_pair(route: &str, prefix: &str) -> Option<(u32, u32)> {
    let suffix = route.strip_prefix(prefix)?;
    let (namespace, value) = suffix.split_once('_')?;
    Some((namespace.parse().ok()?, value.parse().ok()?))
}
