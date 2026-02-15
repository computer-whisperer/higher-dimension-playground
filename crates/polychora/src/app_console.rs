use super::*;

const DEV_CONSOLE_MAX_LOG_LINES: usize = 96;

impl App {
    pub(super) fn toggle_dev_console(&mut self) {
        if self.app_state != AppState::Playing {
            return;
        }

        if self.dev_console_open {
            self.close_dev_console();
            return;
        }

        self.dev_console_open = true;
        self.menu_open = false;
        self.inventory_open = false;
        self.teleport_dialog_open = false;
        self.controls_dialog_open = false;
        self.dev_console_focus_input = true;
        if self.dev_console_log.is_empty() {
            self.append_dev_console_log_line(
                "Developer console ready. Use /help for available commands.",
            );
        }
        if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
            self.release_mouse(&window);
        }
    }

    pub(super) fn close_dev_console(&mut self) {
        if !self.dev_console_open {
            return;
        }

        self.dev_console_open = false;
        self.dev_console_focus_input = false;
        if self.app_state == AppState::Playing
            && !self.perf_suite_active()
            && !self.menu_open
            && !self.inventory_open
            && !self.teleport_dialog_open
        {
            if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                self.grab_mouse(&window);
            }
        }
    }

    pub(super) fn draw_egui_dev_console(
        &mut self,
        ctx: &egui::Context,
        submitted_command: &mut Option<String>,
        close_console: &mut bool,
    ) {
        let mut open = true;
        egui::Window::new("Developer Console")
            .open(&mut open)
            .anchor(egui::Align2::CENTER_TOP, [0.0, 14.0])
            .resizable(false)
            .collapsible(false)
            .default_width(760.0)
            .show(ctx, |ui| {
                ui.label("Commands: /help, /tp, /spawn");
                ui.add_space(2.0);
                egui::ScrollArea::vertical()
                    .max_height(170.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.dev_console_log {
                            ui.monospace(line);
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    let run_clicked = ui.button("Run").clicked();
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.dev_console_input)
                            .desired_width(f32::INFINITY)
                            .hint_text("e.g. /tp 0 8 0 0"),
                    );
                    if self.dev_console_focus_input {
                        response.request_focus();
                        self.dev_console_focus_input = false;
                    }
                    let enter_pressed =
                        response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if run_clicked || enter_pressed {
                        let command = self.dev_console_input.trim().to_string();
                        self.dev_console_input.clear();
                        if !command.is_empty() {
                            *submitted_command = Some(command);
                            self.dev_console_focus_input = true;
                        }
                    }
                });
            });

        if !open {
            *close_console = true;
        }
    }

    pub(super) fn execute_dev_console_command(&mut self, raw_command: &str) {
        let mut command = raw_command.trim();
        if command.is_empty() {
            return;
        }
        self.append_dev_console_log_line(format!("> {command}"));

        if let Some(stripped) = command.strip_prefix('/') {
            command = stripped;
        }
        let mut parts = command.split_whitespace();
        let Some(command_name) = parts.next() else {
            return;
        };
        let args: Vec<&str> = parts.collect();

        if command_name.eq_ignore_ascii_case("help") {
            self.append_dev_console_log_line("Usage:");
            self.append_dev_console_log_line("  /tp <x> <y> <z> <w>");
            self.append_dev_console_log_line("  /tp origin");
            self.append_dev_console_log_line(
                "  /spawn <cube|rotor|drifter> [x y z w] [material-id|material-name]",
            );
            return;
        }

        if command_name.eq_ignore_ascii_case("tp") || command_name.eq_ignore_ascii_case("teleport")
        {
            if args.len() == 1 && args[0].eq_ignore_ascii_case("origin") {
                self.camera.position = [0.0, 0.0, 0.0, 0.0];
                self.look_at_target = None;
                self.append_dev_console_log_line("Teleported to origin.");
                return;
            }
            let Ok(pos) = Self::parse_console_vec4(&args) else {
                self.append_dev_console_log_line("Usage: /tp <x> <y> <z> <w> | /tp origin");
                return;
            };
            self.camera.position = pos;
            self.look_at_target = None;
            self.append_dev_console_log_line(format!(
                "Teleported to ({:.2}, {:.2}, {:.2}, {:.2}).",
                pos[0], pos[1], pos[2], pos[3]
            ));
            return;
        }

        if command_name.eq_ignore_ascii_case("spawn") {
            if args.is_empty() {
                self.append_dev_console_log_line(
                    "Usage: /spawn <cube|rotor|drifter> [x y z w] [material-id|material-name]",
                );
                return;
            }

            let Some(kind) = Self::parse_console_entity_kind(args[0]) else {
                self.append_dev_console_log_line(format!("Unknown entity kind '{}'.", args[0]));
                return;
            };

            let parse_spawn_usage =
                || "Usage: /spawn <cube|rotor|drifter> [x y z w] [material-id|material-name]";

            let (position, material_id) = match args.len() {
                1 => (
                    self.default_spawn_entity_position(),
                    self.default_spawn_entity_material(),
                ),
                2 => {
                    let Some(material_id) = Self::parse_console_material_id(args[1]) else {
                        self.append_dev_console_log_line(format!(
                            "Unknown material '{}'.",
                            args[1]
                        ));
                        return;
                    };
                    (self.default_spawn_entity_position(), material_id)
                }
                5 => {
                    let Ok(pos) = Self::parse_console_vec4(&args[1..5]) else {
                        self.append_dev_console_log_line(parse_spawn_usage());
                        return;
                    };
                    (pos, self.default_spawn_entity_material())
                }
                6 => {
                    let Ok(pos) = Self::parse_console_vec4(&args[1..5]) else {
                        self.append_dev_console_log_line(parse_spawn_usage());
                        return;
                    };
                    let Some(material_id) = Self::parse_console_material_id(args[5]) else {
                        self.append_dev_console_log_line(format!(
                            "Unknown material '{}'.",
                            args[5]
                        ));
                        return;
                    };
                    (pos, material_id)
                }
                _ => {
                    self.append_dev_console_log_line(parse_spawn_usage());
                    return;
                }
            };

            let orientation =
                normalize4_with_fallback(self.current_look_direction(), [0.0, 0.0, 1.0, 0.0]);
            let scale = Self::default_spawn_entity_scale(kind);
            if !self.send_multiplayer_spawn_entity(kind, position, orientation, scale, material_id)
            {
                self.append_dev_console_log_line("Cannot spawn entity without an active server.");
                return;
            }

            self.append_dev_console_log_line(format!(
                "Spawned {} with material {} ({}) at ({:.2}, {:.2}, {:.2}, {:.2}).",
                Self::console_entity_kind_name(kind),
                material_id,
                materials::material_name(material_id),
                position[0],
                position[1],
                position[2],
                position[3]
            ));
            return;
        }

        self.append_dev_console_log_line(format!(
            "Unknown command '{}'. Use /help for command list.",
            command_name
        ));
    }

    fn append_dev_console_log_line(&mut self, line: impl Into<String>) {
        self.dev_console_log.push_back(line.into());
        while self.dev_console_log.len() > DEV_CONSOLE_MAX_LOG_LINES {
            self.dev_console_log.pop_front();
        }
    }

    fn parse_console_vec4(args: &[&str]) -> Result<[f32; 4], ()> {
        if args.len() != 4 {
            return Err(());
        }
        let x = args[0].parse::<f32>().map_err(|_| ())?;
        let y = args[1].parse::<f32>().map_err(|_| ())?;
        let z = args[2].parse::<f32>().map_err(|_| ())?;
        let w = args[3].parse::<f32>().map_err(|_| ())?;
        Ok([x, y, z, w])
    }

    fn default_spawn_entity_position(&self) -> [f32; 4] {
        let edit_reach = self
            .args
            .edit_reach
            .clamp(BLOCK_EDIT_REACH_MIN, BLOCK_EDIT_REACH_MAX);
        let look_dir = self.current_look_direction();
        let targets = self
            .scene
            .block_edit_targets(self.camera.position, look_dir, edit_reach);
        if let Some(place_voxel) = targets.place_voxel {
            return [
                place_voxel[0] as f32 + 0.5,
                place_voxel[1] as f32 + 0.5,
                place_voxel[2] as f32 + 0.5,
                place_voxel[3] as f32 + 0.5,
            ];
        }
        if let Some(hit_voxel) = targets.hit_voxel {
            return [
                hit_voxel[0] as f32 + 0.5,
                hit_voxel[1] as f32 + 0.5,
                hit_voxel[2] as f32 + 0.5,
                hit_voxel[3] as f32 + 0.5,
            ];
        }

        [
            self.camera.position[0] + look_dir[0] * 3.0,
            self.camera.position[1] + look_dir[1] * 3.0,
            self.camera.position[2] + look_dir[2] * 3.0,
            self.camera.position[3] + look_dir[3] * 3.0,
        ]
    }

    fn default_spawn_entity_material(&self) -> u8 {
        self.place_material
            .clamp(BLOCK_EDIT_PLACE_MATERIAL_MIN, BLOCK_EDIT_PLACE_MATERIAL_MAX)
    }

    fn default_spawn_entity_scale(kind: multiplayer::EntityKind) -> f32 {
        match kind {
            multiplayer::EntityKind::TestCube => 0.50,
            multiplayer::EntityKind::TestRotor => 0.54,
            multiplayer::EntityKind::TestDrifter => 0.48,
        }
    }

    fn parse_console_entity_kind(token: &str) -> Option<multiplayer::EntityKind> {
        let normalized = Self::normalize_material_token(token);
        match normalized.as_str() {
            "cube" | "testcube" => Some(multiplayer::EntityKind::TestCube),
            "rotor" | "testrotor" => Some(multiplayer::EntityKind::TestRotor),
            "drifter" | "testdrifter" => Some(multiplayer::EntityKind::TestDrifter),
            _ => None,
        }
    }

    fn console_entity_kind_name(kind: multiplayer::EntityKind) -> &'static str {
        match kind {
            multiplayer::EntityKind::TestCube => "cube",
            multiplayer::EntityKind::TestRotor => "rotor",
            multiplayer::EntityKind::TestDrifter => "drifter",
        }
    }

    fn parse_console_material_id(token: &str) -> Option<u8> {
        if let Ok(id) = token.parse::<u8>() {
            if (BLOCK_EDIT_PLACE_MATERIAL_MIN..=BLOCK_EDIT_PLACE_MATERIAL_MAX).contains(&id) {
                return Some(id);
            }
        }

        let normalized_token = Self::normalize_material_token(token);
        materials::MATERIALS.iter().find_map(|material| {
            let normalized_name = Self::normalize_material_token(material.name);
            (normalized_name == normalized_token).then_some(material.id)
        })
    }

    fn normalize_material_token(token: &str) -> String {
        token
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .map(|ch| ch.to_ascii_lowercase())
            .collect()
    }
}
