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
                    if ui.button("Back").clicked() {
                        self.main_menu_page = MainMenuPage::Root;
                        self.main_menu_connect_error = None;
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
