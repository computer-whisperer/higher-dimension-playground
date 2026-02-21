use super::*;
use higher_dimension_playground::render::{
    OVERLAY_EDGE_TAG_PLACE, OVERLAY_EDGE_TAG_REGION_BRANCH, OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
    OVERLAY_EDGE_TAG_REGION_UNIFORM, OVERLAY_EDGE_TAG_TARGET,
};
use polychora::shared::render_tree::{DebugRayBvhNodeHit, DebugRayBvhNodeKind};

fn describe_sample_ray_hit_for_hud(hit: &DebugRayBvhNodeHit) -> String {
    let span = [
        hit.bounds.max[0]
            .saturating_sub(hit.bounds.min[0])
            .saturating_add(1),
        hit.bounds.max[1]
            .saturating_sub(hit.bounds.min[1])
            .saturating_add(1),
        hit.bounds.max[2]
            .saturating_sub(hit.bounds.min[2])
            .saturating_add(1),
        hit.bounds.max[3]
            .saturating_sub(hit.bounds.min[3])
            .saturating_add(1),
    ];
    match hit.kind {
        DebugRayBvhNodeKind::Internal => format!(
            "kind=Internal bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
        DebugRayBvhNodeKind::LeafUniform { material } => format!(
            "kind=LeafUniform material={} bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            material,
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
        DebugRayBvhNodeKind::LeafChunkArray => format!(
            "kind=LeafChunkArray bounds=({:+},{:+},{:+},{:+})->({:+},{:+},{:+},{:+}) span={}x{}x{}x{} t={:.3}",
            hit.bounds.min[0],
            hit.bounds.min[1],
            hit.bounds.min[2],
            hit.bounds.min[3],
            hit.bounds.max[0],
            hit.bounds.max[1],
            hit.bounds.max[2],
            hit.bounds.max[3],
            span[0],
            span[1],
            span[2],
            span[3],
            hit.t_enter,
        ),
    }
}

fn overlay_edge_tag_for_sample_ray_hit(hit: &DebugRayBvhNodeHit) -> u32 {
    match hit.kind {
        DebugRayBvhNodeKind::Internal => OVERLAY_EDGE_TAG_REGION_BRANCH,
        DebugRayBvhNodeKind::LeafUniform { .. } => OVERLAY_EDGE_TAG_REGION_UNIFORM,
        DebugRayBvhNodeKind::LeafChunkArray => OVERLAY_EDGE_TAG_REGION_CHUNK_ARRAY,
    }
}

impl App {
    pub(super) fn update_and_render(&mut self) {
        let frame_start = Instant::now();
        self.begin_runtime_profile_frame();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        if self.app_state == AppState::MainMenu {
            let main_menu_update_start = Instant::now();
            self.update_and_render_main_menu(dt);
            self.set_runtime_profile_update_ms(
                main_menu_update_start.elapsed().as_secs_f64() * 1000.0,
            );
            if self.perf_suite_active() {
                self.advance_perf_suite_after_frame(frame_start);
            } else {
                self.record_runtime_profile_sample(frame_start);
            }
            self.persist_settings_if_needed(false);
            return;
        }

        let gameplay_update_start = Instant::now();
        self.poll_multiplayer_events();
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

        if self.menu_open
            || self.inventory_open
            || self.teleport_dialog_open
            || self.dev_console_open
        {
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
            if self.input.take_vte_y_slice_lookup_cache_toggle() {
                self.toggle_vte_y_slice_lookup_cache();
            }
            if self.input.take_vte_integral_sky_emissive_toggle() {
                self.toggle_vte_integral_sky_emissive();
            }
            if self.input.take_vte_integral_log_merge_toggle() {
                self.toggle_vte_integral_log_merge();
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
                        if self.input.mouse_forward_held() {
                            self.camera.apply_mouse_look_transport_with_modifiers(
                                dx,
                                dy,
                                MOUSE_SENSITIVITY,
                                self.input.mouse_back_held(),
                                true,
                            );
                        } else {
                            self.camera.apply_mouse_look_transport(
                                dx,
                                dy,
                                MOUSE_SENSITIVITY,
                                self.input.mouse_back_held(),
                            );
                        }
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

        if !self.menu_open
            && !self.inventory_open
            && !self.teleport_dialog_open
            && !self.dev_console_open
        {
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
            if dt > 0.0 {
                let external_velocity = self.player_modifier_external_velocity;
                self.camera.position[0] += external_velocity[0] * dt;
                self.camera.position[2] += external_velocity[2] * dt;
                self.camera.position[3] += external_velocity[3] * dt;
                let decay = (-MULTIPLAYER_PLAYER_MODIFIER_DECAY_HZ * dt.clamp(0.0, 0.25)).exp();
                self.player_modifier_external_velocity[0] *= decay;
                self.player_modifier_external_velocity[2] *= decay;
                self.player_modifier_external_velocity[3] *= decay;
                if self.player_modifier_external_velocity[0].abs() < 1e-3 {
                    self.player_modifier_external_velocity[0] = 0.0;
                }
                if self.player_modifier_external_velocity[2].abs() < 1e-3 {
                    self.player_modifier_external_velocity[2] = 0.0;
                }
                if self.player_modifier_external_velocity[3].abs() < 1e-3 {
                    self.player_modifier_external_velocity[3] = 0.0;
                }
            }
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
                        let material = self.scene.get_voxel(x, y, z, w).0;
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
                        let removed = self
                            .scene
                            .block_edit_targets(self.camera.position, look_dir_for_edit, edit_reach)
                            .hit_voxel;
                        if let Some([x, y, z, w]) = removed {
                            eprintln!("Removed voxel at ({x}, {y}, {z}, {w})");
                            self.audio.play(SoundEffect::Break);
                            self.send_multiplayer_voxel_update(
                                now,
                                [x, y, z, w],
                                voxel::VoxelType::AIR.0,
                            );
                        }
                    } else if place_requested {
                        let placed = self
                            .scene
                            .block_edit_targets(self.camera.position, look_dir_for_edit, edit_reach)
                            .place_voxel;
                        if let Some([x, y, z, w]) = placed {
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
        self.apply_pending_player_movement_modifiers();

        let look_dir = self.current_look_direction();
        self.send_multiplayer_player_update(now, look_dir);
        self.send_multiplayer_chunk_sample_diag_request();
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
        let disable_remote_non_voxel = env_flag_enabled("R4D_DISABLE_REMOTE_NON_VOXEL");
        let highlight_mode = self.args.edit_highlight_mode;
        let mut hud_player_tags =
            self.remote_player_tags(&view_matrix, look_dir, self.focal_length_xy, aspect);
        self.append_multiplayer_stream_tree_diag_hud_tags(
            &mut hud_player_tags,
            &view_matrix,
            self.focal_length_xy,
            aspect,
        );
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
        let sample_ray_node_hits = if !self.menu_open && self.mouse_grabbed {
            self.scene.debug_render_bvh_ray_node_hits(
                self.camera.position,
                look_dir,
                edit_reach,
                self.multiplayer_stream_tree_diag_sample_ray_max_nodes
                    .max(1),
            )
        } else {
            Vec::new()
        };
        let hud_stream_first_node_desc = sample_ray_node_hits
            .first()
            .map(describe_sample_ray_hit_for_hud);
        let hud_stream_final_solid_leaf_desc = sample_ray_node_hits
            .iter()
            .rev()
            .find(|hit| {
                matches!(
                    hit.kind,
                    DebugRayBvhNodeKind::LeafUniform { .. } | DebugRayBvhNodeKind::LeafChunkArray
                )
            })
            .map(describe_sample_ray_hit_for_hud);

        // WAILA: show targeted block name below crosshair
        let waila_text = if !self.menu_open && self.mouse_grabbed && !self.args.no_hud {
            let waila_targets =
                self.scene
                    .block_edit_targets(self.camera.position, look_dir, edit_reach);
            if let Some([x, y, z, w]) = waila_targets.hit_voxel {
                let voxel = self.scene.get_voxel(x, y, z, w);
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

        let overlay_edge_capacity = 2usize
            .saturating_add(if self.multiplayer_stream_tree_diag_enabled {
                self.multiplayer_stream_tree_diag_max_nodes.max(1)
            } else {
                0
            })
            .saturating_add(
                if self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled {
                    self.multiplayer_stream_tree_diag_sample_ray_max_nodes
                        .max(1)
                } else {
                    0
                },
            )
            .saturating_add(if self.multiplayer_stream_tree_compare_diag_enabled {
                self.multiplayer_stream_tree_compare_diag_max_chunks
                    .saturating_mul(2)
            } else {
                0
            });
        let mut custom_overlay_edge_instances = Vec::with_capacity(overlay_edge_capacity);
        if highlight_mode.uses_edges() {
            if let Some(targets) = targets {
                if let Some(hit_voxel) = targets.hit_voxel {
                    append_voxel_outline_edge_instance(
                        &mut custom_overlay_edge_instances,
                        hit_voxel,
                        OVERLAY_EDGE_TAG_TARGET,
                    );
                }
                if let Some(place_voxel) = targets.place_voxel {
                    append_voxel_outline_edge_instance(
                        &mut custom_overlay_edge_instances,
                        place_voxel,
                        OVERLAY_EDGE_TAG_PLACE,
                    );
                }
            }
        }
        if self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled {
            for hit in &sample_ray_node_hits {
                append_chunk_bounds_outline_edge_instance(
                    &mut custom_overlay_edge_instances,
                    hit.bounds.min,
                    hit.bounds.max,
                    overlay_edge_tag_for_sample_ray_hit(hit),
                );
            }
        }
        self.append_multiplayer_stream_tree_diag_overlay_instances(
            &mut custom_overlay_edge_instances,
        );
        self.append_multiplayer_stream_tree_compare_overlay_instances(
            &mut custom_overlay_edge_instances,
        );
        if self.args.no_hud {
            custom_overlay_edge_instances.clear();
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
        let egui_paint = if self.args.no_hud && !self.dev_console_open {
            None
        } else {
            self.run_egui_frame()
        };
        let mut do_navigation_hud =
            !self.menu_open && !self.dev_console_open && self.info_panel_mode != InfoPanelMode::Off;
        if self.args.no_hud {
            do_navigation_hud = false;
        }
        let hud_readout_mode = if !self.menu_open
            && !self.dev_console_open
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
            hud_stream_first_node_desc.as_deref(),
            hud_stream_final_solid_leaf_desc.as_deref(),
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
            custom_overlay_lines: Vec::new(),
            custom_overlay_edge_instances,
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
        self.set_runtime_profile_update_ms(gameplay_update_start.elapsed().as_secs_f64() * 1000.0);

        if backend == RenderBackend::VoxelTraversal {
            let voxel_build_start = Instant::now();
            let voxel_frame = self.scene.build_voxel_frame_data(
                self.camera.position,
                look_dir,
                self.vte_max_trace_distance,
            );
            let voxel_build_elapsed_ms = voxel_build_start.elapsed().as_secs_f64() * 1000.0;

            // If we are playing without a live server connection, do not keep the loading gate up.
            if !self.world_ready
                && self.app_state == AppState::Playing
                && self.multiplayer.is_none()
            {
                self.world_ready = true;
                eprintln!("World ready: no multiplayer connection");
            }
            // Failsafe: do not let the loading gate stick forever if the first
            // world subtree patch is delayed or dropped.
            if !self.world_ready
                && self.app_state == AppState::Playing
                && self.multiplayer.is_some()
            {
                if let Some(wait_since) = self.multiplayer_initial_world_wait_since {
                    const MULTIPLAYER_WORLD_READY_FALLBACK_SECS: f32 = 3.0;
                    if wait_since.elapsed().as_secs_f32() >= MULTIPLAYER_WORLD_READY_FALLBACK_SECS {
                        self.world_ready = true;
                        eprintln!(
                            "World ready fallback: no subtree patch after {:.1}s (continuing with async world streaming)",
                            MULTIPLAYER_WORLD_READY_FALLBACK_SECS
                        );
                    }
                }
            }
            let render_submit_start = Instant::now();
            self.rcx.as_mut().unwrap().render_voxel_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                voxel_frame.as_input(),
            );
            self.set_runtime_profile_voxel_build_ms(voxel_build_elapsed_ms);
            self.set_runtime_profile_render_submit_ms(
                render_submit_start.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            let remote_instances = if disable_remote_non_voxel {
                Vec::new()
            } else {
                self.remote_player_instances(preview_time_s)
            };
            let entity_instances = if disable_remote_non_voxel {
                Vec::new()
            } else {
                self.remote_entity_instances()
            };
            let mut render_instances =
                Vec::with_capacity(remote_instances.len() + entity_instances.len() + 1);
            render_instances.extend(remote_instances);
            render_instances.extend(entity_instances);
            render_instances.push(preview_instance);
            let render_submit_start = Instant::now();
            self.rcx.as_mut().unwrap().render_tetra_frame(
                self.device.clone(),
                self.queue.clone(),
                frame_params,
                TetraFrameInput {
                    model_instances: &render_instances,
                },
            );
            self.set_runtime_profile_render_submit_ms(
                render_submit_start.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let post_render_start = Instant::now();
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
        self.set_runtime_profile_post_render_ms(post_render_start.elapsed().as_secs_f64() * 1000.0);
        if self.perf_suite_active() {
            self.advance_perf_suite_after_frame(frame_start);
        } else {
            self.record_runtime_profile_sample(frame_start);
        }
        self.persist_settings_if_needed(false);
    }
}
