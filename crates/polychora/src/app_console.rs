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
        self.focus_damascene_dev_console_input();
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
                "  /spawn <entity-kind> [x y z w] [material-id|material-name]",
            );
            self.append_dev_console_log_line(
                "  /resync  -- force full server resync and report deltas",
            );
            self.append_dev_console_log_line(
                "  /resync-render  -- rebuild render tree from world tree",
            );
            self.append_dev_console_log_line(
                "  /check   -- run world + render tree integrity check",
            );
            self.append_dev_console_log_line(
                "  /explode <x y z w> [radius]  -- server-side explosion",
            );
            self.append_dev_console_log_line(
                "Other /commands are forwarded to the server. Up/Down recalls history.",
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

        if command_name.eq_ignore_ascii_case("resync") {
            self.trigger_world_force_resync();
            return;
        }

        if command_name.eq_ignore_ascii_case("resync-render") {
            self.append_dev_console_log_line("Rebuilding render tree from world tree...");
            self.scene.force_render_rebuild();
            self.append_dev_console_log_line("Render tree rebuilt.");
            return;
        }

        if command_name.eq_ignore_ascii_case("check") {
            self.append_dev_console_log_line("Running world tree integrity check...");
            let report = self.scene.check_world_tree_integrity();

            // Summary to console
            if let Some(rb) = report.root_bounds {
                self.append_dev_console_log_line(format!(
                    "  root=[{},{},{},{}]->[{},{},{},{}]",
                    rb.min[0].to_num::<i32>(),
                    rb.min[1].to_num::<i32>(),
                    rb.min[2].to_num::<i32>(),
                    rb.min[3].to_num::<i32>(),
                    rb.max[0].to_num::<i32>(),
                    rb.max[1].to_num::<i32>(),
                    rb.max[2].to_num::<i32>(),
                    rb.max[3].to_num::<i32>(),
                ));
            } else {
                self.append_dev_console_log_line("  tree is empty");
            }
            self.append_dev_console_log_line(format!(
                "  depth={} branches={} chunks={} uniforms={} empty={} procref={} cells={}",
                report.max_depth,
                report.branch_count,
                report.chunk_array_count,
                report.uniform_count,
                report.empty_count,
                report.procedural_ref_count,
                report.total_chunk_cells,
            ));
            if !report.scale_histogram.is_empty() {
                let scales: Vec<String> = report
                    .scale_histogram
                    .iter()
                    .map(|(s, c)| format!("s{}={}", s, c))
                    .collect();
                self.append_dev_console_log_line(format!("  scales: {}", scales.join(" "),));
            }

            let bounds_overlap_count = report.bounds_overlaps.len();
            if bounds_overlap_count > 0 {
                self.append_dev_console_log_line(format!(
                    "  FAIL: {} sibling bounds overlap(s) detected!",
                    bounds_overlap_count,
                ));
                // Detailed report to stderr
                eprintln!("[/check] BOUNDS OVERLAPS ({} total):", bounds_overlap_count);
                for (i, (a, b)) in report.bounds_overlaps.iter().enumerate() {
                    eprintln!(
                        "  [{i}] [{},{},{},{}]->[{},{},{},{}] vs [{},{},{},{}]->[{},{},{},{}]",
                        a.min[0].to_num::<i32>(),
                        a.min[1].to_num::<i32>(),
                        a.min[2].to_num::<i32>(),
                        a.min[3].to_num::<i32>(),
                        a.max[0].to_num::<i32>(),
                        a.max[1].to_num::<i32>(),
                        a.max[2].to_num::<i32>(),
                        a.max[3].to_num::<i32>(),
                        b.min[0].to_num::<i32>(),
                        b.min[1].to_num::<i32>(),
                        b.min[2].to_num::<i32>(),
                        b.min[3].to_num::<i32>(),
                        b.max[0].to_num::<i32>(),
                        b.max[1].to_num::<i32>(),
                        b.max[2].to_num::<i32>(),
                        b.max[3].to_num::<i32>(),
                    );
                    if i >= 19 {
                        eprintln!("  ... ({} more)", bounds_overlap_count - 20);
                        break;
                    }
                }
            } else {
                self.append_dev_console_log_line("  OK: no sibling bounds overlaps");
            }

            if let Some(ref e) = report.data_overlap_error {
                self.append_dev_console_log_line(format!("  FAIL: data overlap: {e}"));
                eprintln!("[/check] DATA OVERLAP: {e}");
            } else {
                self.append_dev_console_log_line("  OK: no data overlaps");
            }

            // Also dump full tree to stderr for offline analysis
            self.scene.dump_world_tree();
            self.append_dev_console_log_line("  (world tree dumped to stderr)");

            // --- Render tree integrity ---
            self.append_dev_console_log_line("Running render tree integrity check...");
            let rr = self.scene.check_render_tree_integrity();

            // CPU BVH
            self.append_dev_console_log_line(format!(
                "  cpu-bvh: root={:?} nodes={}/{} leaves={}/{} depth={} (uniform={} ca={})",
                rr.cpu_bvh_root,
                rr.cpu_bvh_reachable_nodes,
                rr.cpu_bvh_node_count,
                rr.cpu_bvh_reachable_leaves,
                rr.cpu_bvh_leaf_count,
                rr.cpu_bvh_max_depth,
                rr.cpu_bvh_uniform_leaves,
                rr.cpu_bvh_chunk_array_leaves,
            ));
            for err in &rr.cpu_bvh_errors {
                self.append_dev_console_log_line(format!("  FAIL cpu-bvh: {err}"));
                eprintln!("[/check] CPU-BVH ERROR: {err}");
            }

            // GPU BVH
            self.append_dev_console_log_line(format!(
                "  gpu-bvh: root={} nodes={}/{} leaves={}/{} depth={} overlaps={} (uniform={} ca={})",
                if rr.gpu_bvh_root == u32::MAX { "INVALID".to_string() } else { rr.gpu_bvh_root.to_string() },
                rr.gpu_bvh_reachable_nodes, rr.gpu_bvh_node_count,
                rr.gpu_bvh_reachable_leaves, rr.gpu_bvh_leaf_count,
                rr.gpu_bvh_max_depth,
                rr.gpu_bvh_sibling_overlaps,
                rr.gpu_bvh_uniform_leaves, rr.gpu_bvh_chunk_array_leaves,
            ));
            for err in &rr.gpu_bvh_errors {
                self.append_dev_console_log_line(format!("  FAIL gpu-bvh: {err}"));
                eprintln!("[/check] GPU-BVH ERROR: {err}");
            }

            self.append_dev_console_log_line("  (render trees dumped to stderr)");
            return;
        }

        if command_name.eq_ignore_ascii_case("spawn") {
            if !self.send_multiplayer_console_command(raw_command) {
                self.append_dev_console_log_line("Cannot spawn entity without an active server.");
                return;
            }
            self.append_dev_console_log_line("Spawn command sent to server.");
            return;
        }

        // Not a client command: forward to the server console (which answers
        // unknown commands with its own supported list via ServerMessage::Error).
        if self.send_multiplayer_console_command(raw_command) {
            self.append_dev_console_log_line(format!("Sent to server: {raw_command}"));
        } else {
            self.append_dev_console_log_line(format!(
                "Unknown command '{}'. Use /help for command list.",
                command_name
            ));
        }
    }

    /// Record a submitted command for Up/Down-arrow recall.
    pub(super) fn dev_console_history_push(&mut self, command: &str) {
        if self.dev_console_history.last().map(String::as_str) != Some(command) {
            self.dev_console_history.push(command.to_string());
        }
        self.dev_console_history_pos = None;
        self.dev_console_history_stash.clear();
    }

    /// Up arrow: step back through history, stashing the in-progress line.
    pub(super) fn dev_console_history_prev(&mut self) {
        if self.dev_console_history.is_empty() {
            return;
        }
        let pos = match self.dev_console_history_pos {
            None => {
                self.dev_console_history_stash = self.dev_console_input.clone();
                self.dev_console_history.len() - 1
            }
            Some(pos) => pos.saturating_sub(1),
        };
        self.dev_console_history_pos = Some(pos);
        self.dev_console_input = self.dev_console_history[pos].clone();
        self.focus_damascene_dev_console_input();
    }

    /// Down arrow: step forward through history, back to the stashed line.
    pub(super) fn dev_console_history_next(&mut self) {
        let Some(pos) = self.dev_console_history_pos else {
            return;
        };
        if pos + 1 < self.dev_console_history.len() {
            self.dev_console_history_pos = Some(pos + 1);
            self.dev_console_input = self.dev_console_history[pos + 1].clone();
        } else {
            self.dev_console_history_pos = None;
            self.dev_console_input = std::mem::take(&mut self.dev_console_history_stash);
        }
        self.focus_damascene_dev_console_input();
    }

    pub(super) fn append_dev_console_log_line(&mut self, line: impl Into<String>) {
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
}
