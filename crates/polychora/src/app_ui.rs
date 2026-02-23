use super::*;

impl App {
    pub(super) fn draw_egui_pause_menu(
        &mut self,
        ctx: &egui::Context,
        close_menu: &mut bool,
        return_to_main_menu: &mut bool,
    ) {
        egui::Window::new("Polychora")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .fixed_size([460.0, 500.0])
            .show(ctx, |ui| {
                ui.heading(RichText::new("Immediate Menu").strong());
                ui.label("Adjust runtime settings while paused.");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("Resume").clicked() {
                        *close_menu = true;
                    }
                    if ui.button("Controls").clicked() {
                        self.controls_dialog_open = !self.controls_dialog_open;
                    }
                    if ui.button("Main Menu").clicked() {
                        *return_to_main_menu = true;
                    }
                    if ui.button("Quit").clicked() {
                        self.should_exit_after_render = true;
                    }
                });

                ui.separator();

                let mut selected_info_panel = self.info_panel_mode;
                egui::ComboBox::from_label("Info Panel")
                    .selected_text(selected_info_panel.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            InfoPanelMode::Full,
                            InfoPanelMode::VectorTable,
                            InfoPanelMode::VectorTable2,
                            InfoPanelMode::Off,
                        ] {
                            ui.selectable_value(&mut selected_info_panel, mode, mode.label());
                        }
                    });
                self.info_panel_mode = selected_info_panel;

                let mut selected_control_scheme = self.control_scheme;
                egui::ComboBox::from_label("Control Scheme")
                    .selected_text(selected_control_scheme.label())
                    .show_ui(ui, |ui| {
                        for scheme in [
                            ControlScheme::IntuitiveUpright,
                            ControlScheme::LookTransport,
                            ControlScheme::RotorFree,
                            ControlScheme::LegacySideButtonLayers,
                            ControlScheme::LegacyScrollCycle,
                        ] {
                            ui.selectable_value(
                                &mut selected_control_scheme,
                                scheme,
                                scheme.label(),
                            );
                        }
                    });
                if selected_control_scheme != self.control_scheme {
                    self.set_control_scheme(selected_control_scheme);
                }

                ui.add(
                    egui::Slider::new(
                        &mut self.focal_length_xy,
                        FOCAL_LENGTH_MIN..=FOCAL_LENGTH_MAX,
                    )
                    .text("Focal Length XY"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.focal_length_zw,
                        FOCAL_LENGTH_MIN..=FOCAL_LENGTH_MAX,
                    )
                    .text("Focal Length ZW"),
                );
                ui.checkbox(
                    &mut self.zw_angle_color_shift_enabled,
                    "ZW Angle Red/Blue Shift",
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.zw_angle_color_shift_strength,
                        ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN..=ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
                    )
                    .text("ZW Shift Strength"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.vte_max_trace_steps,
                        VTE_TRACE_STEPS_MIN..=VTE_TRACE_STEPS_MAX,
                    )
                    .logarithmic(true)
                    .text("VTE Max Trace Steps"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.vte_max_trace_distance,
                        VTE_TRACE_DISTANCE_MIN..=VTE_TRACE_DISTANCE_MAX,
                    )
                    .text("VTE Max Trace Distance"),
                );
                self.vte_max_trace_distance = self
                    .vte_max_trace_distance
                    .clamp(VTE_TRACE_DISTANCE_MIN, VTE_TRACE_DISTANCE_MAX);
                self.zw_angle_color_shift_strength = self.zw_angle_color_shift_strength.clamp(
                    ZW_ANGLE_COLOR_SHIFT_STRENGTH_MIN,
                    ZW_ANGLE_COLOR_SHIFT_STRENGTH_MAX,
                );
                {
                    let mut mat_id = self.selected_block.block_type as u8;
                    let response = ui.add(
                        egui::Slider::new(
                            &mut mat_id,
                            BLOCK_EDIT_PLACE_MATERIAL_MIN..=BLOCK_EDIT_PLACE_MATERIAL_MAX,
                        )
                        .text("Place Material"),
                    );
                    if response.changed() {
                        self.selected_block = polychora::shared::voxel::BlockData::simple(0, mat_id as u32);
                        self.hotbar_slots[self.hotbar_selected_index] =
                            Some(polychora::shared::protocol::ItemStack::block(0, mat_id as u32, 1));
                    }
                }
                ui.add(
                    egui::Slider::new(&mut self.audio.master_volume, 0.0..=2.0)
                        .text("Master Volume")
                        .step_by(0.05),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.audio.spatial_falloff_power,
                        AUDIO_SPATIAL_FALLOFF_POWER_MIN..=AUDIO_SPATIAL_FALLOFF_POWER_MAX,
                    )
                    .text("Spatial Falloff (1/r^N)")
                    .step_by(0.05),
                );
                self.audio.spatial_falloff_power = self.audio.spatial_falloff_power.clamp(
                    AUDIO_SPATIAL_FALLOFF_POWER_MIN,
                    AUDIO_SPATIAL_FALLOFF_POWER_MAX,
                );

                ui.separator();
                ui.label("Render Resolution");
                ui.horizontal(|ui| {
                    ui.label("Width:");
                    ui.add(
                        egui::DragValue::new(&mut self.pending_render_width)
                            .range(128..=3840)
                            .speed(16),
                    );
                    ui.label("Height:");
                    ui.add(
                        egui::DragValue::new(&mut self.pending_render_height)
                            .range(128..=2160)
                            .speed(16),
                    );
                    ui.label("Layers:");
                    ui.add(
                        egui::DragValue::new(&mut self.pending_render_layers)
                            .range(1..=512)
                            .speed(1),
                    );
                });
                let dims_changed = self.pending_render_width != self.args.width
                    || self.pending_render_height != self.args.height
                    || self.pending_render_layers != self.args.layers;
                ui.horizontal(|ui| {
                    let apply_btn =
                        ui.add_enabled(dims_changed, egui::Button::new("Apply Resolution"));
                    if apply_btn.clicked() {
                        self.args.width = self.pending_render_width;
                        self.args.height = self.pending_render_height;
                        self.args.layers = self.pending_render_layers;
                        if let Some(rcx) = self.rcx.as_mut() {
                            rcx.recreate_sized_buffers(
                                [self.args.width, self.args.height, self.args.layers],
                                None,
                            );
                        }
                    }
                    if dims_changed {
                        ui.label(format!(
                            "(current: {}x{}x{})",
                            self.args.width, self.args.height, self.args.layers
                        ));
                    }
                });

                ui.separator();

                let mut y_cache_enabled = self.vte_y_slice_lookup_cache_enabled;
                if ui
                    .checkbox(&mut y_cache_enabled, "VTE Y-Slice Lookup Cache")
                    .changed()
                {
                    self.toggle_vte_y_slice_lookup_cache();
                }
                let mut sky_emissive_tweak = self.vte_integral_sky_emissive_enabled;
                if ui
                    .checkbox(&mut sky_emissive_tweak, "Integral Sky+Emissive Tweak")
                    .changed()
                {
                    self.toggle_vte_integral_sky_emissive();
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.vte_integral_sky_scale,
                        VTE_INTEGRAL_SKY_SCALE_MIN..=VTE_INTEGRAL_SKY_SCALE_MAX,
                    )
                    .text("Integral Sky Scale"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.vte_integral_hit_emissive_boost,
                        VTE_INTEGRAL_HIT_EMISSIVE_MIN..=VTE_INTEGRAL_HIT_EMISSIVE_MAX,
                    )
                    .text("Integral Hit Emissive"),
                );
                let mut log_merge_tweak = self.vte_integral_log_merge_enabled;
                if ui
                    .checkbox(&mut log_merge_tweak, "Integral Log Merge")
                    .changed()
                {
                    self.toggle_vte_integral_log_merge();
                }
                ui.add(
                    egui::Slider::new(
                        &mut self.vte_integral_log_merge_k,
                        VTE_INTEGRAL_LOG_MERGE_K_MIN..=VTE_INTEGRAL_LOG_MERGE_K_MAX,
                    )
                    .text("Integral Log-Merge K"),
                );

                ui.separator();
                ui.label("Region-Tree Bounds Debug");
                ui.checkbox(
                    &mut self.multiplayer_stream_tree_diag_enabled,
                    "Render stream tree bounds",
                );
                ui.checkbox(
                    &mut self.multiplayer_stream_tree_compare_diag_enabled,
                    "Render stream/world mismatch bounds",
                );
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_labels_enabled,
                        "Labels",
                    );
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_non_empty_only,
                        "Non-empty only",
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_show_branch_bounds,
                        "Branch",
                    );
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_show_uniform_bounds,
                        "Uniform",
                    );
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_show_chunk_array_bounds,
                        "ChunkArray",
                    );
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_show_procedural_bounds,
                        "Procedural",
                    );
                    ui.checkbox(
                        &mut self.multiplayer_stream_tree_diag_show_empty_bounds,
                        "Empty",
                    );
                });
                ui.checkbox(
                    &mut self.multiplayer_stream_tree_diag_sample_ray_bounds_enabled,
                    "Render sample-ray BVH node bounds",
                );
                ui.add(
                    egui::Slider::new(&mut self.multiplayer_stream_tree_diag_max_nodes, 1..=4096)
                        .text("Bounds max nodes"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.multiplayer_stream_tree_diag_sample_ray_max_nodes,
                        1..=512,
                    )
                    .text("Sample-ray max nodes"),
                );
                ui.add(
                    egui::Slider::new(&mut self.multiplayer_stream_tree_diag_max_labels, 1..=512)
                        .text("Label max count"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.multiplayer_stream_tree_compare_diag_max_chunks,
                        1..=4096,
                    )
                    .text("Mismatch sample cap"),
                );
                ui.add(
                    egui::Slider::new(
                        &mut self.multiplayer_stream_tree_compare_diag_log_interval,
                        1..=1200,
                    )
                    .text("Mismatch log interval (frames)"),
                );

                ui.separator();
                ui.label("Press Esc to close this menu.");
            });
    }

    pub(super) fn draw_egui_hotbar(&self, ctx: &egui::Context) {
        let screen_rect = ctx.content_rect();
        let slot_size = 80.0;
        let gap = 6.5;
        let total_width = 9.0 * slot_size + 8.0 * gap;
        let start_x = (screen_rect.width() - total_width) / 2.0;
        let start_y = screen_rect.height() - slot_size - 65.0;

        egui::Area::new(egui::Id::new("hotbar"))
            .fixed_pos(egui::pos2(start_x, start_y))
            .interactable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(gap, 0.0);
                    for i in 0..9 {
                        let material_id = block_material_from_slot(&self.hotbar_slots[i]);
                        let [r, g, b] = self.content_registry.material_color_by_token(material_id as u16);
                        let is_selected = i == self.hotbar_selected_index;

                        let (rect, _response) = ui.allocate_exact_size(
                            egui::vec2(slot_size, slot_size),
                            egui::Sense::hover(),
                        );

                        // Background
                        let bg_color = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160);
                        ui.painter().rect_filled(rect, 3.0, bg_color);

                        // Material icon (tesseract image or color fallback)
                        let icon_rect = rect.shrink(5.0);
                        if let (Some(sheet), Some(tex_id)) =
                            (&self.material_icon_sheet, self.material_icons_texture_id)
                        {
                            if let Some([u0, v0, u1, v1]) = sheet.uv_rect(material_id) {
                                ui.painter().image(
                                    tex_id,
                                    icon_rect,
                                    egui::Rect::from_min_max(
                                        egui::pos2(u0, v0),
                                        egui::pos2(u1, v1),
                                    ),
                                    egui::Color32::WHITE,
                                );
                            } else {
                                let mat_color = egui::Color32::from_rgb(r, g, b);
                                ui.painter().rect_filled(icon_rect, 2.0, mat_color);
                            }
                        } else {
                            let mat_color = egui::Color32::from_rgb(r, g, b);
                            ui.painter().rect_filled(icon_rect, 2.0, mat_color);
                        }

                        // Selection border
                        if is_selected {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(3.0, egui::Color32::from_rgb(255, 255, 100)),
                                egui::epaint::StrokeKind::Outside,
                            );
                        } else {
                            ui.painter().rect_stroke(
                                rect,
                                3.0,
                                egui::Stroke::new(
                                    1.3,
                                    egui::Color32::from_rgba_unmultiplied(200, 200, 200, 80),
                                ),
                                egui::epaint::StrokeKind::Outside,
                            );
                        }

                        // Slot number label (top-left corner)
                        let label_pos = rect.left_top() + egui::vec2(4.0, 1.3);
                        ui.painter().text(
                            label_pos,
                            egui::Align2::LEFT_TOP,
                            format!("{}", i + 1),
                            egui::FontId::proportional(13.0),
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                        );

                        // Material name (bottom center, small text)
                        let name = self.content_registry.material_name_by_token(material_id as u16);
                        let label_pos = rect.center_bottom() + egui::vec2(0.0, -3.0);
                        ui.painter().text(
                            label_pos,
                            egui::Align2::CENTER_BOTTOM,
                            name,
                            egui::FontId::proportional(10.0),
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200),
                        );
                    }
                });
            });
    }

    pub(super) fn draw_egui_teleport_dialog(
        &mut self,
        ctx: &egui::Context,
        teleport_target: &mut Option<[f32; 4]>,
        close_teleport: &mut bool,
    ) {
        let mut open = true;
        egui::Window::new("Teleport")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.label("Coordinates:");
                let labels = ["X:", "Y:", "Z:", "W:"];
                for (i, label) in labels.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(*label);
                        ui.add(
                            egui::TextEdit::singleline(&mut self.teleport_coords[i])
                                .desired_width(120.0),
                        );
                    });
                }

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    if ui.button("Teleport").clicked() {
                        let parsed: Option<[f32; 4]> = (|| {
                            Some([
                                self.teleport_coords[0].parse().ok()?,
                                self.teleport_coords[1].parse().ok()?,
                                self.teleport_coords[2].parse().ok()?,
                                self.teleport_coords[3].parse().ok()?,
                            ])
                        })();
                        if let Some(pos) = parsed {
                            *teleport_target = Some(pos);
                        }
                    }

                    if ui.button("Go to Origin").clicked() {
                        *teleport_target = Some([0.0, 0.0, 0.0, 0.0]);
                    }
                });

                if self.multiplayer.is_some() && !self.remote_players.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label("Players:");
                    let mut sorted_ids: Vec<u64> = self.remote_players.keys().copied().collect();
                    sorted_ids.sort();
                    for entity_id in sorted_ids {
                        if let Some(player) = self.remote_players.get(&entity_id) {
                            let name = if player.name.is_empty() {
                                player
                                    .owner_client_id
                                    .map(|id| format!("Player {}", id))
                                    .unwrap_or_else(|| format!("Entity {}", entity_id))
                            } else {
                                player.name.clone()
                            };
                            let pos = player.position;
                            let label = format!(
                                "{} ({:.1}, {:.1}, {:.1}, {:.1})",
                                name, pos[0], pos[1], pos[2], pos[3],
                            );
                            if ui.button(label).clicked() {
                                *teleport_target = Some(pos);
                            }
                        }
                    }
                }
            });
        if !open {
            *close_teleport = true;
        }
    }

    pub(super) fn draw_egui_inventory(
        &self,
        ctx: &egui::Context,
        close_inventory: &mut bool,
        inventory_pick: &mut Option<u8>,
    ) {
        let mut open = true;
        egui::Window::new("Creative Inventory")
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut open)
            .default_width(676.0)
            .show(ctx, |ui| {
                // Category tabs
                ui.horizontal(|ui| {
                    for cat in materials::MaterialCategory::ALL {
                        if ui
                            .selectable_label(false, cat.label())
                            .clicked()
                        {
                            // Future: could filter by category. For now just a visual cue.
                        }
                    }
                    ui.label("|");
                    ui.label("All");
                });
                ui.separator();

                // Material grid
                let items_per_row = 10;
                let cell_size = 74.0;
                let cell_gap = 5.0;

                egui::ScrollArea::vertical()
                    .max_height(416.0)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = egui::vec2(cell_gap, cell_gap);
                            for (idx, entry) in self.content_registry.all_blocks_ordered().enumerate() {
                                if idx > 0 && idx % items_per_row == 0 {
                                    ui.end_row();
                                }
                                let [r, g, b] = entry.color;
                                let mat_color = egui::Color32::from_rgb(r, g, b);
                                let token = entry.material_token as u8;

                                let (rect, response) = ui.allocate_exact_size(
                                    egui::vec2(cell_size, cell_size),
                                    egui::Sense::click(),
                                );

                                // Background
                                let bg = if response.hovered() {
                                    egui::Color32::from_rgba_unmultiplied(80, 80, 80, 200)
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(40, 40, 40, 200)
                                };
                                ui.painter().rect_filled(rect, 3.0, bg);

                                // Material icon (tesseract image or color fallback)
                                let icon_rect = rect.shrink(4.0);
                                if let (Some(sheet), Some(tex_id)) =
                                    (&self.material_icon_sheet, self.material_icons_texture_id)
                                {
                                    if let Some([u0, v0, u1, v1]) = sheet.uv_rect(token) {
                                        ui.painter().image(
                                            tex_id,
                                            icon_rect,
                                            egui::Rect::from_min_max(
                                                egui::pos2(u0, v0),
                                                egui::pos2(u1, v1),
                                            ),
                                            egui::Color32::WHITE,
                                        );
                                    } else {
                                        ui.painter().rect_filled(icon_rect, 2.0, mat_color);
                                    }
                                } else {
                                    ui.painter().rect_filled(icon_rect, 2.0, mat_color);
                                }

                                // Material name
                                let text_pos = egui::pos2(rect.center().x, rect.bottom() - 3.0);
                                ui.painter().text(
                                    text_pos,
                                    egui::Align2::CENTER_BOTTOM,
                                    entry.name,
                                    egui::FontId::proportional(10.0),
                                    egui::Color32::from_rgba_unmultiplied(220, 220, 220, 255),
                                );
                                let category_pos = egui::pos2(rect.center().x, rect.top() + 4.0);
                                ui.painter().text(
                                    category_pos,
                                    egui::Align2::CENTER_TOP,
                                    entry.category.label(),
                                    egui::FontId::proportional(8.0),
                                    egui::Color32::from_rgba_unmultiplied(180, 180, 180, 220),
                                );

                                if response.hovered() {
                                    ui.painter().rect_stroke(
                                        rect,
                                        3.0,
                                        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 255, 100)),
                                        egui::epaint::StrokeKind::Outside,
                                    );
                                }

                                if response.clicked() {
                                    *inventory_pick = Some(token);
                                }
                            }
                        });
                    });

                ui.separator();
                ui.label("Click a material to place it in the selected hotbar slot. Press I or Esc to close.");
            });

        if !open {
            *close_inventory = true;
        }
    }

    pub(super) fn draw_egui_controls_dialog(&mut self, ctx: &egui::Context) {
        let mut open = true;
        egui::Window::new("Controls")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Movement");
                egui::Grid::new("controls_movement")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("W / A / S / D").strong());
                        ui.label("Move forward / left / backward / right");
                        ui.end_row();

                        ui.label(egui::RichText::new("Space").strong());
                        ui.label("Jump (double-tap to toggle fly mode)");
                        ui.end_row();

                        ui.label(egui::RichText::new("Shift").strong());
                        ui.label("Descend / Crouch");
                        ui.end_row();

                        ui.label(egui::RichText::new("Q / E").strong());
                        ui.label("Move in 4D (W-axis negative / positive)");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("Camera");
                egui::Grid::new("controls_camera")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Mouse").strong());
                        ui.label("Look around");
                        ui.end_row();

                        ui.label(egui::RichText::new("R (hold)").strong());
                        ui.label("Reset orientation");
                        ui.end_row();

                        ui.label(egui::RichText::new("F (hold)").strong());
                        ui.label("Pull to 3D");
                        ui.end_row();

                        ui.label(egui::RichText::new("G").strong());
                        ui.label("Look at nearest block");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("Building");
                egui::Grid::new("controls_building")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Left Click").strong());
                        ui.label("Break block");
                        ui.end_row();

                        ui.label(egui::RichText::new("Right Click").strong());
                        ui.label("Place block");
                        ui.end_row();

                        ui.label(egui::RichText::new("Middle Click").strong());
                        ui.label("Pick material");
                        ui.end_row();

                        ui.label(egui::RichText::new("[ / ]").strong());
                        ui.label("Previous / Next material");
                        ui.end_row();

                        ui.label(egui::RichText::new("Scroll Wheel").strong());
                        ui.label("Cycle hotbar slot");
                        ui.end_row();

                        ui.label(egui::RichText::new("1-9, 0").strong());
                        ui.label("Select hotbar slot");
                        ui.end_row();
                    });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(4.0);

                ui.heading("UI");
                egui::Grid::new("controls_ui")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Escape").strong());
                        ui.label("Open / close menu");
                        ui.end_row();

                        ui.label(egui::RichText::new("Tab / I").strong());
                        ui.label("Toggle inventory");
                        ui.end_row();

                        ui.label(egui::RichText::new("T").strong());
                        ui.label("Toggle teleport dialog");
                        ui.end_row();

                        ui.label(egui::RichText::new("`").strong());
                        ui.label("Toggle developer console");
                        ui.end_row();
                    });
            });

        if !open {
            self.controls_dialog_open = false;
        }
    }

    pub(super) fn run_egui_frame(&mut self) -> Option<EguiPaintData> {
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone())?;
        let raw_input = self.egui_winit_state.as_mut()?.take_egui_input(&window);

        let egui_ctx = self.egui_ctx.clone();
        let mut close_menu = false;
        let mut close_inventory = false;
        let mut inventory_pick: Option<u8> = None;
        let mut teleport_target: Option<[f32; 4]> = None;
        let mut close_teleport = false;
        let mut close_console = false;
        let mut console_command: Option<String> = None;
        let mut transition_to_playing: Option<MainMenuTransition> = None;
        let mut return_to_main_menu = false;
        let full_output = egui_ctx.run(raw_input, |ctx| {
            if self.app_state == AppState::MainMenu {
                self.draw_egui_main_menu(ctx, &mut transition_to_playing);
            } else if !self.world_ready {
                self.draw_egui_loading_screen(ctx);
            } else {
                if self.menu_open {
                    self.draw_egui_pause_menu(ctx, &mut close_menu, &mut return_to_main_menu);
                }
                if self.inventory_open {
                    self.draw_egui_inventory(ctx, &mut close_inventory, &mut inventory_pick);
                }
                if self.teleport_dialog_open {
                    self.draw_egui_teleport_dialog(ctx, &mut teleport_target, &mut close_teleport);
                }
                if self.controls_dialog_open {
                    self.draw_egui_controls_dialog(ctx);
                }
                if self.dev_console_open {
                    self.draw_egui_dev_console(ctx, &mut console_command, &mut close_console);
                }
                self.draw_egui_hotbar(ctx);
                self.draw_egui_waila(ctx);
            }
        });

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = full_output;
        if let Some(egui_state) = self.egui_winit_state.as_mut() {
            egui_state.handle_platform_output(&window, platform_output);
        }

        if let Some(transition) = transition_to_playing {
            self.handle_main_menu_transition(transition, &window);
        }
        if return_to_main_menu {
            self.transition_to_main_menu(&window);
        }
        if close_menu {
            self.menu_open = false;
            self.controls_dialog_open = false;
            self.grab_mouse(&window);
        }
        if close_inventory || inventory_pick.is_some() {
            self.inventory_open = false;
            self.grab_mouse(&window);
        }
        if let Some(material_id) = inventory_pick {
            self.hotbar_slots[self.hotbar_selected_index] =
                Some(polychora::shared::protocol::ItemStack::block(0, material_id as u32, 1));
            self.selected_block = polychora::shared::voxel::BlockData::simple(0, material_id as u32);
            eprintln!(
                "Inventory: set hotbar slot {} to material {} ({})",
                self.hotbar_selected_index + 1,
                material_id,
                self.content_registry.material_name_by_token(material_id as u16),
            );
        }
        if close_teleport {
            self.teleport_dialog_open = false;
            if let Some(window) = self.rcx.as_ref().and_then(|rcx| rcx.window.clone()) {
                self.grab_mouse(&window);
            }
        }
        if let Some(pos) = teleport_target {
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
        if close_console {
            self.close_dev_console();
        }
        if let Some(command) = console_command {
            self.execute_dev_console_command(&command);
        }

        let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

        let mut texture_updates = Vec::new();
        for (texture_id, delta) in textures_delta.set {
            if !matches!(texture_id, egui::TextureId::Managed(0)) {
                continue;
            }
            let (size, pixels) = match delta.image {
                egui::ImageData::Color(image) => {
                    let size = [image.size[0] as u32, image.size[1] as u32];
                    let mut pixels = Vec::with_capacity(image.pixels.len() * 4);
                    for pixel in image.pixels.iter() {
                        let [r, g, b, a] = pixel.to_srgba_unmultiplied();
                        pixels.push(r);
                        pixels.push(g);
                        pixels.push(b);
                        pixels.push(a);
                    }
                    (size, pixels)
                }
            };
            texture_updates.push(EguiTextureUpdate {
                size,
                pos: delta.pos.map(|[x, y]| [x as u32, y as u32]),
                pixels,
            });
        }

        let material_icons_tid = self.material_icons_texture_id;
        let mut meshes = Vec::new();
        for clipped in clipped_primitives {
            let egui::epaint::Primitive::Mesh(mesh) = clipped.primitive else {
                continue;
            };
            let texture_slot = if matches!(mesh.texture_id, egui::TextureId::Managed(0)) {
                EguiTextureSlot::EguiAtlas
            } else if Some(mesh.texture_id) == material_icons_tid {
                EguiTextureSlot::MaterialIcons
            } else {
                continue;
            };

            let mut vertices = Vec::with_capacity(mesh.indices.len());
            for &index in &mesh.indices {
                let Some(vertex) = mesh.vertices.get(index as usize) else {
                    continue;
                };
                let [r, g, b, a] = vertex.color.to_srgba_unmultiplied();
                vertices.push(EguiPaintVertex {
                    position_px: [
                        vertex.pos.x * pixels_per_point,
                        vertex.pos.y * pixels_per_point,
                    ],
                    uv: [vertex.uv.x, vertex.uv.y],
                    color: [
                        r as f32 / 255.0,
                        g as f32 / 255.0,
                        b as f32 / 255.0,
                        a as f32 / 255.0,
                    ],
                });
            }
            if vertices.is_empty() {
                continue;
            }

            meshes.push(EguiPaintMesh {
                clip_rect_px: [
                    clipped.clip_rect.min.x * pixels_per_point,
                    clipped.clip_rect.min.y * pixels_per_point,
                    clipped.clip_rect.max.x * pixels_per_point,
                    clipped.clip_rect.max.y * pixels_per_point,
                ],
                vertices,
                texture_slot,
            });
        }

        Some(EguiPaintData {
            texture_updates,
            meshes,
        })
    }

    fn draw_egui_waila(&self, ctx: &egui::Context) {
        let target = match &self.waila_target {
            Some(t) => t,
            None => return,
        };

        let screen_rect = ctx.content_rect();
        let panel_x = screen_rect.width() / 2.0;
        let panel_y = 30.0;

        egui::Area::new(egui::Id::new("waila_panel"))
            .fixed_pos(egui::pos2(panel_x, panel_y))
            .pivot(egui::Align2::CENTER_TOP)
            .interactable(false)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgba_unmultiplied(15, 15, 22, 190))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        match target {
                            WailaTarget::Block { coords, block } => {
                                self.draw_waila_block(ui, *coords, block);
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
                                self.draw_waila_entity(
                                    ui,
                                    *entity_id,
                                    *entity_type_ns,
                                    *entity_type,
                                    *position,
                                    *orientation,
                                    *scale,
                                    data,
                                    *distance,
                                );
                            }
                        }
                    });
            });
    }

    fn draw_waila_block(&self, ui: &mut egui::Ui, coords: [i32; 4], block: &polychora::shared::voxel::BlockData) {
        let entry = self.content_registry.block_entry(block.namespace, block.block_type);
        let name = entry.map(|e| e.name).unwrap_or("Unknown");
        let category = entry.map(|e| e.category.label()).unwrap_or("Unknown");
        let [r, g, b] = entry.map(|e| e.color).unwrap_or([128, 128, 128]);
        let material_id = entry.map(|e| e.material_token as u8).unwrap_or(1);

        // Header row: icon + name + category + ID
        ui.horizontal(|ui| {
            let icon_size = 28.0;
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_size, icon_size), egui::Sense::hover());

            if let (Some(sheet), Some(tex_id)) =
                (&self.material_icon_sheet, self.material_icons_texture_id)
            {
                if let Some([u0, v0, u1, v1]) = sheet.uv_rect(material_id) {
                    ui.painter().image(
                        tex_id,
                        icon_rect,
                        egui::Rect::from_min_max(
                            egui::pos2(u0, v0),
                            egui::pos2(u1, v1),
                        ),
                        egui::Color32::WHITE,
                    );
                } else {
                    ui.painter()
                        .rect_filled(icon_rect, 2.0, egui::Color32::from_rgb(r, g, b));
                }
            } else {
                ui.painter()
                    .rect_filled(icon_rect, 2.0, egui::Color32::from_rgb(r, g, b));
            }

            ui.label(
                egui::RichText::new(name)
                    .strong()
                    .size(15.0)
                    .color(egui::Color32::from_rgb(240, 240, 240)),
            );
            ui.label(
                egui::RichText::new(format!("{} #{}", category, material_id))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(160, 160, 170)),
            );
        });

        // Coordinates
        ui.label(
            egui::RichText::new(format!(
                "[{}, {}, {}, {}]",
                coords[0], coords[1], coords[2], coords[3]
            ))
            .monospace()
            .size(11.0)
            .color(egui::Color32::from_rgb(140, 145, 160)),
        );
    }

    fn draw_waila_entity(
        &self,
        ui: &mut egui::Ui,
        entity_id: u64,
        entity_type_ns: u32,
        entity_type: u32,
        position: [f32; 4],
        orientation: [f32; 4],
        scale: f32,
        data: &[u8],
        distance: f32,
    ) {
        let entry = self.content_registry.entity_lookup(entity_type_ns, entity_type);
        let canonical_name = entry
            .map(|e| e.canonical_name)
            .unwrap_or("unknown");
        let category = entry
            .map(|e| format!("{:?}", e.category))
            .unwrap_or_else(|| "Unknown".to_string());
        let base_material_token = self.content_registry.entity_base_material_token(entity_type_ns, entity_type);
        let [r, g, b] = self.content_registry.material_color_by_token(base_material_token);

        // Check if this is a player
        let player_name = self
            .remote_players
            .get(&entity_id)
            .map(|p| p.name.clone());

        // Header row: icon + name + category + distance
        ui.horizontal(|ui| {
            let icon_size = 28.0;
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(icon_size, icon_size), egui::Sense::hover());

            if let (Some(sheet), Some(tex_id)) =
                (&self.material_icon_sheet, self.material_icons_texture_id)
            {
                if let Some([u0, v0, u1, v1]) = sheet.uv_rect(base_material_token as u8) {
                    ui.painter().image(
                        tex_id,
                        icon_rect,
                        egui::Rect::from_min_max(
                            egui::pos2(u0, v0),
                            egui::pos2(u1, v1),
                        ),
                        egui::Color32::WHITE,
                    );
                } else {
                    ui.painter()
                        .rect_filled(icon_rect, 2.0, egui::Color32::from_rgb(r, g, b));
                }
            } else {
                ui.painter()
                    .rect_filled(icon_rect, 2.0, egui::Color32::from_rgb(r, g, b));
            }

            let display_name = if let Some(ref pname) = player_name {
                format!("{} ({})", canonical_name, pname)
            } else {
                canonical_name.to_string()
            };

            ui.label(
                egui::RichText::new(display_name)
                    .strong()
                    .size(15.0)
                    .color(egui::Color32::from_rgb(240, 240, 240)),
            );
            ui.label(
                egui::RichText::new(format!("{} {:.1}m", category, distance))
                    .size(12.0)
                    .color(egui::Color32::from_rgb(160, 160, 170)),
            );
        });

        let info_color = egui::Color32::from_rgb(140, 145, 160);
        let info_size = 11.0;

        // Entity ID + type
        ui.label(
            egui::RichText::new(format!("id: {}  type: {}:{}", entity_id, entity_type_ns, entity_type))
                .monospace()
                .size(info_size)
                .color(info_color),
        );

        // Position
        ui.label(
            egui::RichText::new(format!(
                "pos: [{:.1}, {:.1}, {:.1}, {:.1}]",
                position[0], position[1], position[2], position[3]
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Orientation + scale
        ui.label(
            egui::RichText::new(format!(
                "ori: [{:.2}, {:.2}, {:.2}, {:.2}]  scale: {:.2}",
                orientation[0], orientation[1], orientation[2], orientation[3], scale
            ))
            .monospace()
            .size(info_size)
            .color(info_color),
        );

        // Mob archetype
        if let Some(entry) = entry {
            if let Some(archetype) = entry.mob_archetype {
                ui.label(
                    egui::RichText::new(format!("archetype: {:?}", archetype))
                        .monospace()
                        .size(info_size)
                        .color(info_color),
                );
            }
        }

        // CBOR data decode
        if let Some(decoded) = format_cbor_for_display(data) {
            ui.label(
                egui::RichText::new(format!("data: {}", decoded))
                    .monospace()
                    .size(info_size)
                    .color(egui::Color32::from_rgb(160, 170, 140)),
            );
        }
    }
}
