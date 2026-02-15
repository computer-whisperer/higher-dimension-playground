use super::*;

fn estimate_directory_size_bytes(root: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    total
}

fn parse_chunk_coord4(raw: &str, label: &str) -> Result<[i32; 4], String> {
    let cleaned = raw.replace(',', " ");
    let parts: Vec<&str> = cleaned.split_whitespace().collect();
    if parts.len() != 4 {
        return Err(format!(
            "{label} must contain exactly 4 integers (got {})",
            parts.len()
        ));
    }
    let mut out = [0i32; 4];
    for (idx, part) in parts.iter().enumerate() {
        out[idx] = part
            .parse::<i32>()
            .map_err(|error| format!("{label} value {idx} ('{part}') is invalid: {error}"))?;
    }
    Ok(out)
}

fn validate_chunk_bounds(min_chunk: [i32; 4], max_chunk: [i32; 4]) -> Result<(), String> {
    for axis in 0..4 {
        if min_chunk[axis] > max_chunk[axis] {
            return Err(format!(
                "invalid bounds on axis {}: min {} > max {}",
                axis, min_chunk[axis], max_chunk[axis]
            ));
        }
    }
    Ok(())
}

fn drop_overrides_outside_chunk_bounds(
    world: &mut polychora::shared::voxel::VoxelWorld,
    min_chunk: [i32; 4],
    max_chunk: [i32; 4],
) -> usize {
    let positions: Vec<polychora::shared::voxel::ChunkPos> = world.chunks.keys().copied().collect();
    let mut dropped = 0usize;
    for pos in positions {
        let outside = pos.x < min_chunk[0]
            || pos.x > max_chunk[0]
            || pos.y < min_chunk[1]
            || pos.y > max_chunk[1]
            || pos.z < min_chunk[2]
            || pos.z > max_chunk[2]
            || pos.w < min_chunk[3]
            || pos.w > max_chunk[3];
        if outside && world.remove_chunk_override(pos) {
            dropped = dropped.saturating_add(1);
        }
    }
    dropped
}

fn run_legacy_trim_migration(
    input: &Path,
    output: &Path,
    min_chunk: [i32; 4],
    max_chunk: [i32; 4],
) -> Result<(usize, usize), String> {
    let file = std::fs::File::open(input)
        .map_err(|error| format!("failed to open input {}: {error}", input.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut world = polychora::shared::voxel::load_world(&mut reader)
        .map_err(|error| format!("failed to parse {}: {error}", input.display()))?;
    let dropped = drop_overrides_outside_chunk_bounds(&mut world, min_chunk, max_chunk);

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "failed to create output directory {}: {error}",
                    parent.display()
                )
            })?;
        }
    }
    let file = std::fs::File::create(output)
        .map_err(|error| format!("failed to create output {}: {error}", output.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    polychora::shared::voxel::save_world(&world, &mut writer)
        .map_err(|error| format!("failed to write output {}: {error}", output.display()))?;
    use std::io::Write;
    writer
        .flush()
        .map_err(|error| format!("failed to flush output {}: {error}", output.display()))?;

    Ok((dropped, world.non_empty_chunk_count()))
}

fn run_legacy_to_v3_migration(
    input: &Path,
    sidecar: Option<&Path>,
    output: &Path,
    world_seed: u64,
    overwrite: bool,
) -> Result<polychora::save_v3::SaveResult, String> {
    if output.exists() {
        if !overwrite {
            return Err(format!(
                "output {} already exists; enable overwrite to replace it",
                output.display()
            ));
        }
        if output.is_file() {
            return Err(format!(
                "output {} is a file, expected a directory",
                output.display()
            ));
        }
        std::fs::remove_dir_all(output).map_err(|error| {
            format!(
                "failed to clear output directory {}: {error}",
                output.display()
            )
        })?;
    }
    polychora::save_v3::migrate_legacy_world_to_v3(
        input,
        sidecar,
        output,
        world_seed,
        polychora::save_v3::now_unix_ms(),
    )
    .map_err(|error| format!("v3 migration failed: {error}"))
}

impl App {
    pub(super) fn scan_world_files(&mut self) {
        self.main_menu_world_files.clear();
        self.main_menu_selected_world = None;
        let saves_dir = Path::new("saves");
        let dirs_to_scan: Vec<&Path> = if saves_dir.is_dir() {
            vec![saves_dir]
        } else {
            vec![]
        };
        // Also scan current directory for v3 save roots.
        let cwd = Path::new(".");
        let all_dirs: Vec<&Path> = {
            let mut v = vec![cwd];
            v.extend(dirs_to_scan);
            v
        };
        let mut seen = std::collections::HashSet::new();
        for dir in all_dirs {
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() || !polychora::save_v3::is_v3_save_root(&path) {
                    continue;
                }
                let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                if !seen.insert(canonical) {
                    continue;
                }
                let display_name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let size_bytes = estimate_directory_size_bytes(&path);
                self.main_menu_world_files.push(WorldFileEntry {
                    path,
                    display_name,
                    size_bytes,
                });
            }
        }
        self.main_menu_world_files
            .sort_by(|a, b| a.display_name.cmp(&b.display_name));
    }

    pub(super) fn reset_multiplayer_connection_state(&mut self) {
        self.multiplayer = None;
        self.multiplayer_self_id = None;
        self.next_multiplayer_edit_id = 1;
        self.pending_voxel_edits.clear();
        self.remote_players.clear();
        self.remote_entities.clear();
        self.last_multiplayer_player_update = Instant::now();
    }

    pub(super) fn connect_multiplayer_remote(&mut self, server_addr: String) -> Result<(), String> {
        let player_name = if self.main_menu_player_name.trim().is_empty() {
            default_multiplayer_player_name()
        } else {
            self.main_menu_player_name.trim().to_string()
        };
        match MultiplayerClient::connect(server_addr.clone(), player_name.clone()) {
            Ok(client) => {
                eprintln!(
                    "Connecting to multiplayer server {} as '{}'",
                    client.server_addr(),
                    player_name
                );
                self.reset_multiplayer_connection_state();
                self.multiplayer = Some(client);
                Ok(())
            }
            Err(error) => Err(format!("Failed to connect to {}: {}", server_addr, error)),
        }
    }

    pub(super) fn connect_singleplayer_local(&mut self, world_file: PathBuf) -> Result<(), String> {
        let player_name = if self.main_menu_player_name.trim().is_empty() {
            default_multiplayer_player_name()
        } else {
            self.main_menu_player_name.trim().to_string()
        };
        let config = build_singleplayer_runtime_config(&self.args, world_file.clone());
        match MultiplayerClient::connect_local(config, player_name.clone()) {
            Ok(client) => {
                eprintln!(
                    "Starting integrated singleplayer server for {} as '{}'",
                    world_file.display(),
                    player_name
                );
                self.reset_multiplayer_connection_state();
                self.multiplayer = Some(client);
                Ok(())
            }
            Err(error) => Err(format!(
                "Failed to start integrated singleplayer server for {}: {}",
                world_file.display(),
                error
            )),
        }
    }

    fn run_main_menu_migration_legacy_trim(&mut self) {
        let input = PathBuf::from(self.main_menu_migrate_trim_input.trim());
        let output = PathBuf::from(self.main_menu_migrate_trim_output.trim());
        let min_chunk = match parse_chunk_coord4(&self.main_menu_migrate_trim_keep_min, "keep-min")
        {
            Ok(v) => v,
            Err(error) => {
                self.main_menu_migration_status = Some(format!("Error: {error}"));
                return;
            }
        };
        let max_chunk = match parse_chunk_coord4(&self.main_menu_migrate_trim_keep_max, "keep-max")
        {
            Ok(v) => v,
            Err(error) => {
                self.main_menu_migration_status = Some(format!("Error: {error}"));
                return;
            }
        };
        if let Err(error) = validate_chunk_bounds(min_chunk, max_chunk) {
            self.main_menu_migration_status = Some(format!("Error: {error}"));
            return;
        }
        if input.as_os_str().is_empty() || output.as_os_str().is_empty() {
            self.main_menu_migration_status =
                Some("Error: input and output paths are required".to_string());
            return;
        }
        match run_legacy_trim_migration(&input, &output, min_chunk, max_chunk) {
            Ok((dropped, non_empty_chunks)) => {
                self.main_menu_migration_status = Some(format!(
                    "Legacy trim migration complete: dropped {} overrides, wrote {} non-empty chunks to {}",
                    dropped,
                    non_empty_chunks,
                    output.display(),
                ));
            }
            Err(error) => {
                self.main_menu_migration_status = Some(format!("Error: {error}"));
            }
        }
    }

    fn run_main_menu_migration_legacy_to_v3(&mut self) {
        let input = PathBuf::from(self.main_menu_migrate_v3_input.trim());
        let output = PathBuf::from(self.main_menu_migrate_v3_output.trim());
        if input.as_os_str().is_empty() || output.as_os_str().is_empty() {
            self.main_menu_migration_status =
                Some("Error: input and output paths are required".to_string());
            return;
        }
        let sidecar = self.main_menu_migrate_v3_sidecar.trim();
        let sidecar_path = if sidecar.is_empty() {
            None
        } else {
            Some(PathBuf::from(sidecar))
        };
        let world_seed = match self.main_menu_migrate_v3_world_seed.trim().parse::<u64>() {
            Ok(v) => v,
            Err(error) => {
                self.main_menu_migration_status =
                    Some(format!("Error: world seed is invalid: {error}"));
                return;
            }
        };
        match run_legacy_to_v3_migration(
            &input,
            sidecar_path.as_deref(),
            &output,
            world_seed,
            self.main_menu_migrate_v3_overwrite,
        ) {
            Ok(save_result) => {
                self.main_menu_migration_status = Some(format!(
                    "Legacy -> v3 migration complete: generation {} (block regions {}, entity regions {}) at {}",
                    save_result.generation,
                    save_result.saved_block_regions,
                    save_result.saved_entity_regions,
                    output.display(),
                ));
                self.scan_world_files();
            }
            Err(error) => {
                self.main_menu_migration_status = Some(format!("Error: {error}"));
            }
        }
    }

    pub(super) fn enter_play_state(&mut self, window: &Window) {
        // Start from an empty client scene; multiplayer snapshot/chunks are authoritative.
        self.scene = Scene::new(ScenePreset::Empty);
        self.camera = Camera4D::new();
        self.app_state = AppState::Playing;
        self.menu_open = false;
        self.dev_console_open = false;
        self.dev_console_focus_input = false;
        self.dev_console_input.clear();
        self.main_menu_connect_error = None;
        self.world_ready = false;

        // Pre-load chunks around spawn position
        self.scene
            .preload_spawn_chunks(self.camera.position, self.vte_lod_near_max_distance);

        if !self.perf_suite_active() {
            self.grab_mouse(window);
        }
    }

    pub(super) fn handle_main_menu_transition(
        &mut self,
        transition: MainMenuTransition,
        window: &Window,
    ) {
        match transition {
            MainMenuTransition::NewWorld => {
                let path = generate_new_singleplayer_world_path();
                match self.connect_singleplayer_local(path.clone()) {
                    Ok(()) => {
                        self.world_file = path.clone();
                        self.enter_play_state(window);
                        eprintln!(
                            "Started new integrated singleplayer world {}",
                            path.display()
                        );
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        self.main_menu_connect_error = Some(msg);
                    }
                }
            }
            MainMenuTransition::LoadWorld(path) => {
                match self.connect_singleplayer_local(path.clone()) {
                    Ok(()) => {
                        self.world_file = path.clone();
                        self.enter_play_state(window);
                        eprintln!(
                            "Loaded integrated singleplayer world from {}",
                            path.display()
                        );
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        self.main_menu_connect_error = Some(msg);
                    }
                }
            }
            MainMenuTransition::ConnectMultiplayer(addr) => {
                let server_addr = normalize_server_addr(&addr);
                match self.connect_multiplayer_remote(server_addr.clone()) {
                    Ok(()) => {
                        self.enter_play_state(window);
                    }
                    Err(msg) => {
                        eprintln!("{msg}");
                        self.main_menu_connect_error = Some(msg);
                    }
                }
            }
        }
    }

    pub(super) fn transition_to_main_menu(&mut self, window: &Window) {
        // Disconnect multiplayer if connected
        if self.multiplayer.is_some() {
            self.reset_multiplayer_connection_state();
            eprintln!("Disconnected from multiplayer server");
        }
        // Reset scene to show demo background (DemoCubes has visible geometry near origin)
        self.scene = Scene::new(ScenePreset::DemoCubes);
        self.app_state = AppState::MainMenu;
        self.main_menu_page = MainMenuPage::Root;
        self.main_menu_connect_error = None;
        self.main_menu_migration_status = None;
        self.menu_open = false;
        self.inventory_open = false;
        self.teleport_dialog_open = false;
        self.dev_console_open = false;
        self.dev_console_focus_input = false;
        self.dev_console_input.clear();
        self.world_ready = true;
        self.menu_time = 0.0;
        self.menu_camera = make_menu_camera();
        self.release_mouse(window);
    }

    pub(super) fn update_and_render_main_menu(&mut self, dt: f32) {
        // Drain any inputs that accumulated
        self.input.take_mouse_delta();
        self.input.take_jump();
        self.input.take_remove_block();
        self.input.take_place_block();
        self.input.take_pick_material();
        let _ = self.input.take_scroll_steps();

        // Advance menu camera animation
        self.menu_time += dt.min(0.1);
        apply_menu_camera_orbit_pose(&mut self.menu_camera, self.menu_time);

        let egui_paint = if self.args.no_hud {
            None
        } else {
            self.run_egui_frame()
        };

        let view_matrix = self.menu_camera.view_matrix_upright();
        let backend = self.args.backend.to_render_backend();
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
            zw_angle_color_shift_enabled: self.zw_angle_color_shift_enabled,
            zw_angle_color_shift_strength: self.zw_angle_color_shift_strength,
            do_navigation_hud: false,
            egui_paint,
            ..Default::default()
        };

        let preview_elapsed = Instant::now() - self.start_time;
        let preview_time_ticks_ms = preview_elapsed.as_millis() as u32;
        let frame_params = FrameParams {
            view_matrix,
            time_ticks_ms: preview_time_ticks_ms,
            focal_length_xy: self.focal_length_xy,
            focal_length_zw: self.focal_length_zw,
            render_options,
        };

        if backend == RenderBackend::VoxelTraversal {
            let voxel_frame = self.scene.build_voxel_frame_data(
                self.menu_camera.position,
                self.menu_camera.look_direction(),
                self.vte_lod_near_max_distance,
                self.vte_lod_mid_max_distance,
                self.vte_max_trace_distance,
            );
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_frame.as_input(),
                &[],
                &[],
            );
        } else {
            self.scene.update_surfaces_if_dirty();
            let instances = self.scene.build_instances(self.menu_camera.position);
            self.rcx.as_mut().unwrap().render_tetra_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                TetraFrameInput {
                    model_instances: instances,
                },
            );
        }
    }

    pub(super) fn draw_egui_loading_screen(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.4);
                ui.heading("Loading world...");
            });
        });
    }

    pub(super) fn draw_egui_main_menu(
        &mut self,
        ctx: &egui::Context,
        transition: &mut Option<MainMenuTransition>,
    ) {
        match self.main_menu_page {
            MainMenuPage::Root => {
                self.draw_egui_main_menu_root(ctx, transition);
            }
            MainMenuPage::Singleplayer => {
                self.draw_egui_main_menu_singleplayer(ctx, transition);
            }
            MainMenuPage::SingleplayerMigrations => {
                self.draw_egui_main_menu_singleplayer_migrations(ctx, transition);
            }
            MainMenuPage::SingleplayerMigrationLegacyTrim => {
                self.draw_egui_main_menu_singleplayer_migrate_legacy_trim(ctx, transition);
            }
            MainMenuPage::SingleplayerMigrationLegacyToV3 => {
                self.draw_egui_main_menu_singleplayer_migrate_legacy_to_v3(ctx, transition);
            }
            MainMenuPage::Multiplayer => {
                self.draw_egui_main_menu_multiplayer(ctx, transition);
            }
        }
    }

    pub(super) fn draw_egui_main_menu_root(
        &mut self,
        ctx: &egui::Context,
        _transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_root")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([300.0, 260.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(16.0);
                    ui.heading(RichText::new("Polychora").size(32.0).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("4D Voxel Explorer").size(14.0).weak());
                    ui.add_space(24.0);

                    let button_size = egui::vec2(200.0, 36.0);
                    if ui
                        .add_sized(
                            button_size,
                            egui::Button::new(RichText::new("Singleplayer").size(16.0)),
                        )
                        .clicked()
                    {
                        self.main_menu_page = MainMenuPage::Singleplayer;
                        self.main_menu_connect_error = None;
                        self.scan_world_files();
                    }
                    ui.add_space(8.0);
                    if ui
                        .add_sized(
                            button_size,
                            egui::Button::new(RichText::new("Multiplayer").size(16.0)),
                        )
                        .clicked()
                    {
                        self.main_menu_page = MainMenuPage::Multiplayer;
                        self.main_menu_connect_error = None;
                    }
                    ui.add_space(8.0);
                    if ui
                        .add_sized(
                            button_size,
                            egui::Button::new(RichText::new("Quit").size(16.0)),
                        )
                        .clicked()
                    {
                        self.should_exit_after_render = true;
                    }
                    ui.add_space(16.0);
                });
            });
    }

    pub(super) fn draw_egui_main_menu_singleplayer(
        &mut self,
        ctx: &egui::Context,
        transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_singleplayer")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([400.0, 400.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Singleplayer").size(24.0).strong());
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.label(RichText::new("Saved Worlds").strong());
                ui.add_space(4.0);

                if self.main_menu_world_files.is_empty() {
                    ui.label("No v3 world save directories found in saves/ or current directory.");
                } else {
                    let selected = self.main_menu_selected_world;
                    egui::ScrollArea::vertical()
                        .max_height(200.0)
                        .show(ui, |ui| {
                            for (i, entry) in self.main_menu_world_files.iter().enumerate() {
                                let is_selected = selected == Some(i);
                                let size_label = format_file_size(entry.size_bytes);
                                let label = format!("{} ({})", entry.display_name, size_label);
                                if ui.selectable_label(is_selected, &label).clicked() {
                                    self.main_menu_selected_world = Some(i);
                                }
                            }
                        });
                }

                if let Some(error) = &self.main_menu_connect_error {
                    ui.add_space(6.0);
                    ui.colored_label(egui::Color32::from_rgb(255, 100, 100), error.as_str());
                }

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    let has_selection = self.main_menu_selected_world.is_some();
                    if ui
                        .add_enabled(has_selection, egui::Button::new("Load Selected"))
                        .clicked()
                    {
                        if let Some(idx) = self.main_menu_selected_world {
                            if let Some(entry) = self.main_menu_world_files.get(idx) {
                                *transition =
                                    Some(MainMenuTransition::LoadWorld(entry.path.clone()));
                            }
                        }
                    }
                    if ui.button("Create New World").clicked() {
                        *transition = Some(MainMenuTransition::NewWorld);
                    }
                    if ui.button("Migrations").clicked() {
                        self.main_menu_migration_status = None;
                        self.main_menu_page = MainMenuPage::SingleplayerMigrations;
                    }
                    if ui.button("Back").clicked() {
                        self.main_menu_page = MainMenuPage::Root;
                        self.main_menu_connect_error = None;
                        self.main_menu_migration_status = None;
                    }
                });
            });
    }

    fn draw_main_menu_migration_status(&self, ui: &mut egui::Ui) {
        if let Some(status) = &self.main_menu_migration_status {
            let is_error = status.starts_with("Error:");
            let color = if is_error {
                egui::Color32::from_rgb(255, 110, 110)
            } else {
                egui::Color32::from_rgb(120, 220, 140)
            };
            ui.colored_label(color, status.as_str());
        }
    }

    pub(super) fn draw_egui_main_menu_singleplayer_migrations(
        &mut self,
        ctx: &egui::Context,
        _transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_singleplayer_migrations")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([520.0, 280.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Singleplayer Migrations").size(24.0).strong());
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                ui.label("Choose a migration tool:");
                ui.add_space(6.0);

                if ui.button("Legacy .v4dw Keep-Bounds Trim").clicked() {
                    self.main_menu_page = MainMenuPage::SingleplayerMigrationLegacyTrim;
                }
                ui.small(
                    "Drops override chunks outside selected chunk bounds and writes a migrated .v4dw.",
                );
                ui.add_space(10.0);

                if ui.button("Legacy .v4dw -> v3 Save Root").clicked() {
                    self.main_menu_page = MainMenuPage::SingleplayerMigrationLegacyToV3;
                }
                ui.small("Converts a legacy .v4dw (plus optional .entities.json) into a v3 save directory.");

                ui.add_space(12.0);
                self.draw_main_menu_migration_status(ui);

                ui.add_space(10.0);
                if ui.button("Back").clicked() {
                    self.main_menu_page = MainMenuPage::Singleplayer;
                }
            });
    }

    pub(super) fn draw_egui_main_menu_singleplayer_migrate_legacy_trim(
        &mut self,
        ctx: &egui::Context,
        _transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_singleplayer_migrate_legacy_trim")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([620.0, 360.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Legacy Keep-Bounds Trim").size(22.0).strong());
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.label("Input .v4dw:");
                ui.text_edit_singleline(&mut self.main_menu_migrate_trim_input);
                ui.add_space(4.0);

                ui.label("Output .v4dw:");
                ui.text_edit_singleline(&mut self.main_menu_migrate_trim_output);
                ui.add_space(4.0);

                ui.label("Keep Min Chunk (X Y Z W):");
                ui.text_edit_singleline(&mut self.main_menu_migrate_trim_keep_min);
                ui.add_space(2.0);

                ui.label("Keep Max Chunk (X Y Z W):");
                ui.text_edit_singleline(&mut self.main_menu_migrate_trim_keep_max);
                ui.add_space(10.0);

                self.draw_main_menu_migration_status(ui);
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    if ui.button("Run Migration").clicked() {
                        self.run_main_menu_migration_legacy_trim();
                    }
                    if ui.button("Back").clicked() {
                        self.main_menu_page = MainMenuPage::SingleplayerMigrations;
                    }
                });
            });
    }

    pub(super) fn draw_egui_main_menu_singleplayer_migrate_legacy_to_v3(
        &mut self,
        ctx: &egui::Context,
        _transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_singleplayer_migrate_legacy_to_v3")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([640.0, 410.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Legacy .v4dw -> v3").size(22.0).strong());
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);

                ui.label("Input .v4dw:");
                ui.text_edit_singleline(&mut self.main_menu_migrate_v3_input);
                ui.add_space(4.0);

                ui.label("Input sidecar (.entities.json, optional):");
                ui.text_edit_singleline(&mut self.main_menu_migrate_v3_sidecar);
                ui.add_space(4.0);

                ui.label("Output v3 save root directory:");
                ui.text_edit_singleline(&mut self.main_menu_migrate_v3_output);
                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label("World seed:");
                    ui.text_edit_singleline(&mut self.main_menu_migrate_v3_world_seed);
                });
                ui.checkbox(
                    &mut self.main_menu_migrate_v3_overwrite,
                    "Overwrite output directory if it exists",
                );
                ui.add_space(10.0);

                self.draw_main_menu_migration_status(ui);
                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    if ui.button("Run Migration").clicked() {
                        self.run_main_menu_migration_legacy_to_v3();
                    }
                    if ui.button("Back").clicked() {
                        self.main_menu_page = MainMenuPage::SingleplayerMigrations;
                    }
                });
            });
    }

    pub(super) fn draw_egui_main_menu_multiplayer(
        &mut self,
        ctx: &egui::Context,
        transition: &mut Option<MainMenuTransition>,
    ) {
        egui::Window::new("main_menu_multiplayer")
            .title_bar(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([400.0, 200.0])
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Multiplayer").size(24.0).strong());
                });
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    ui.label("Player name:");
                    ui.text_edit_singleline(&mut self.main_menu_player_name);
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Server address:");
                    ui.text_edit_singleline(&mut self.main_menu_server_address);
                });
                ui.add_space(8.0);

                if let Some(error) = &self.main_menu_connect_error {
                    ui.colored_label(egui::Color32::from_rgb(255, 100, 100), error.as_str());
                    ui.add_space(4.0);
                }

                ui.horizontal(|ui| {
                    if ui.button("Connect").clicked() {
                        let addr = self.main_menu_server_address.clone();
                        *transition = Some(MainMenuTransition::ConnectMultiplayer(addr));
                    }
                    if ui.button("Back").clicked() {
                        self.main_menu_page = MainMenuPage::Root;
                        self.main_menu_connect_error = None;
                    }
                });
            });
    }
}
