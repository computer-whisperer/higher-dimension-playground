use super::*;

impl RenderContext {
    pub(super) fn full_hud_scissor(&self) -> Scissor {
        let (width, height) = match self.window.as_ref() {
            Some(window) => {
                let size = window.inner_size();
                (size.width.max(1), size.height.max(1))
            }
            None => (
                self.sized_buffers.render_dimensions[0].max(1),
                self.sized_buffers.render_dimensions[1].max(1),
            ),
        };
        Scissor {
            offset: [0, 0],
            extent: [width, height],
        }
    }

    /// Screen-space raster region for VTE overlay tetra pass (held block preview).
    /// Keeping this bounded avoids full-frame tet pixel traversal for a tiny HUD element.
    pub(super) fn vte_overlay_raster_region(&self) -> [u32; 4] {
        let render_w = self.sized_buffers.render_dimensions[0].max(1);
        let render_h = self.sized_buffers.render_dimensions[1].max(1);

        let side = ((render_w.min(render_h) as f32) * 0.28).round() as u32;
        let side = side.clamp(208, 360).min(render_w).min(render_h);
        let width = side;
        let height = side;
        let margin_x = (render_w / 64).clamp(8, 24);
        let margin_y = (render_h / 64).clamp(8, 24);

        let origin_x = render_w.saturating_sub(width + margin_x);
        let origin_y = render_h.saturating_sub(height + margin_y);

        [origin_x, origin_y, width.max(1), height.max(1)]
    }

    pub(super) fn refresh_egui_descriptor_sets(&mut self) {
        let Some(present_ctx) = self.present_pipeline.as_ref() else {
            return;
        };
        let Some(egui_resources) = self.egui_resources.as_ref() else {
            return;
        };
        let descriptor_set_layout = present_ctx
            .hud_pipeline_layout
            .set_layouts()
            .first()
            .unwrap()
            .clone();

        for frame in &mut self.frames_in_flight {
            frame.egui_descriptor_set = frame.hud_vertex_buffer.as_ref().map(|hud_buffer| {
                create_hud_descriptor_set(
                    self.descriptor_set_allocator.clone(),
                    descriptor_set_layout.clone(),
                    hud_buffer.clone(),
                    egui_resources.atlas_view.clone(),
                    egui_resources.atlas_sampler.clone(),
                )
            });
        }
    }

    /// Upload a material icons sprite sheet and create descriptor sets so it can
    /// be used as a separate texture in the HUD/egui rendering pipeline.
    pub fn upload_material_icons_texture(
        &mut self,
        queue: Arc<Queue>,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) {
        let view = create_rgba8_srgb_texture_view(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            queue.clone(),
            width,
            height,
            pixels,
        );
        let sampler = Sampler::new(
            queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        self.material_icons_view = Some(view.clone());
        self.material_icons_sampler = Some(sampler.clone());

        // Create descriptor sets for all frames in flight
        if let Some(present_ctx) = self.present_pipeline.as_ref() {
            let descriptor_set_layout = present_ctx
                .hud_pipeline_layout
                .set_layouts()
                .first()
                .unwrap()
                .clone();

            for frame in &mut self.frames_in_flight {
                frame.material_icons_descriptor_set =
                    frame.hud_vertex_buffer.as_ref().map(|hud_buffer| {
                        create_hud_descriptor_set(
                            self.descriptor_set_allocator.clone(),
                            descriptor_set_layout.clone(),
                            hud_buffer.clone(),
                            view.clone(),
                            sampler.clone(),
                        )
                    });
            }
        }
    }

    pub(super) fn apply_egui_texture_updates(
        &mut self,
        queue: Arc<Queue>,
        updates: &[EguiTextureUpdate],
    ) {
        if updates.is_empty() {
            return;
        }

        let (new_size, new_pixels) = {
            let Some(egui_resources) = self.egui_resources.as_mut() else {
                return;
            };
            let mut did_change = false;
            for update in updates {
                let width = update.size[0].max(1);
                let height = update.size[1].max(1);
                let expected_len = (width as usize)
                    .saturating_mul(height as usize)
                    .saturating_mul(4);
                if update.pixels.len() != expected_len {
                    continue;
                }

                match update.pos {
                    None => {
                        egui_resources.texture_size = [width, height];
                        egui_resources.texture_pixels.clear();
                        egui_resources
                            .texture_pixels
                            .extend_from_slice(&update.pixels);
                        did_change = true;
                    }
                    Some([x, y]) => {
                        let atlas_w = egui_resources.texture_size[0].max(1);
                        let atlas_h = egui_resources.texture_size[1].max(1);
                        if egui_resources.texture_pixels.len()
                            != (atlas_w as usize)
                                .saturating_mul(atlas_h as usize)
                                .saturating_mul(4)
                        {
                            continue;
                        }
                        if x + width > atlas_w || y + height > atlas_h {
                            continue;
                        }

                        for row in 0..height as usize {
                            let src_start = row * width as usize * 4;
                            let src_end = src_start + width as usize * 4;
                            let dst_start =
                                ((y as usize + row) * atlas_w as usize + x as usize) * 4;
                            let dst_end = dst_start + width as usize * 4;
                            egui_resources.texture_pixels[dst_start..dst_end]
                                .copy_from_slice(&update.pixels[src_start..src_end]);
                        }
                        did_change = true;
                    }
                }
            }

            if !did_change {
                return;
            }
            (
                egui_resources.texture_size,
                egui_resources.texture_pixels.clone(),
            )
        };

        let new_view = create_rgba8_srgb_texture_view(
            self.memory_allocator.clone(),
            self.command_buffer_allocator.clone(),
            queue,
            new_size[0],
            new_size[1],
            &new_pixels,
        );
        if let Some(egui_resources) = self.egui_resources.as_mut() {
            let old_view = std::mem::replace(&mut egui_resources.atlas_view, new_view);
            egui_resources
                .retired_atlas_views
                .push((old_view, self.frames_rendered));
            egui_resources
                .retired_atlas_views
                .retain(|&(_, retired_frame)| {
                    self.frames_rendered.saturating_sub(retired_frame) < FRAMES_IN_FLIGHT
                });
        }
        self.refresh_egui_descriptor_sets();
    }

    pub(super) fn write_egui_overlay(
        &mut self,
        frame_idx: usize,
        base_hud_vertex: usize,
        meshes: &[EguiPaintMesh],
    ) -> (usize, Vec<HudDrawBatch>) {
        let Some(hud_buf) = self.frames_in_flight[frame_idx].hud_vertex_buffer.as_ref() else {
            return (0, Vec::new());
        };
        if meshes.is_empty() || base_hud_vertex >= HUD_VERTEX_CAPACITY {
            return (0, Vec::new());
        }

        let present_size = match self.window.as_ref() {
            Some(window) => {
                let size = window.inner_size();
                [size.width.max(1), size.height.max(1)]
            }
            None => [
                self.sized_buffers.render_dimensions[0].max(1),
                self.sized_buffers.render_dimensions[1].max(1),
            ],
        };
        let present_w = present_size[0] as f32;
        let present_h = present_size[1] as f32;
        let mut writer = hud_buf.write().unwrap();
        let mut batches = Vec::with_capacity(meshes.len());
        let mut cursor = base_hud_vertex;

        for mesh in meshes {
            if cursor >= HUD_VERTEX_CAPACITY || mesh.vertices.is_empty() {
                break;
            }

            let clip_min_x = mesh.clip_rect_px[0].clamp(0.0, present_w);
            let clip_min_y = mesh.clip_rect_px[1].clamp(0.0, present_h);
            let clip_max_x = mesh.clip_rect_px[2].clamp(0.0, present_w);
            let clip_max_y = mesh.clip_rect_px[3].clamp(0.0, present_h);
            if clip_max_x <= clip_min_x || clip_max_y <= clip_min_y {
                continue;
            }

            let scissor_x = clip_min_x.floor() as u32;
            let scissor_y = clip_min_y.floor() as u32;
            let scissor_w = (clip_max_x.ceil() as u32).saturating_sub(scissor_x);
            let scissor_h = (clip_max_y.ceil() as u32).saturating_sub(scissor_y);
            if scissor_w == 0 || scissor_h == 0 {
                continue;
            }

            let first_vertex = cursor;
            for v in &mesh.vertices {
                if cursor >= HUD_VERTEX_CAPACITY {
                    break;
                }
                let position = pixels_to_ndc(
                    Vec2::new(v.position_px[0], present_h - v.position_px[1]),
                    present_size,
                );
                writer[cursor] = HudVertex::new(
                    position,
                    Vec2::new(v.uv[0], v.uv[1]),
                    Vec4::new(v.color[0], v.color[1], v.color[2], v.color[3]),
                );
                cursor += 1;
            }

            let vertex_count = cursor.saturating_sub(first_vertex);
            if vertex_count > 0 {
                batches.push(HudDrawBatch {
                    first_vertex: first_vertex as u32,
                    vertex_count: vertex_count as u32,
                    scissor: Scissor {
                        offset: [scissor_x, scissor_y],
                        extent: [scissor_w, scissor_h],
                    },
                    texture_slot: match mesh.texture_slot {
                        EguiTextureSlot::MaterialIcons => HudTextureSlot::MaterialIcons,
                        EguiTextureSlot::EguiAtlas => HudTextureSlot::EguiAtlas,
                    },
                });
            }
        }

        (cursor.saturating_sub(base_hud_vertex), batches)
    }

    /// Returns (line_count, hud_vertex_count)
    pub(super) fn write_navigation_hud_overlay(
        &mut self,
        frame_idx: usize,
        base_line_count: usize,
        view_matrix: &nalgebra::Matrix5<f32>,
        view_matrix_inverse: &nalgebra::Matrix5<f32>,
        focal_length_xy: f32,
        model_instances: &[common::ModelInstance],
        readout_mode: HudReadoutMode,
        rotation_label: Option<&str>,
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
        player_tags: &[HudPlayerTag],
        waila_text: Option<&str>,
    ) -> (usize, usize) {
        let max_lines = LINE_VERTEX_CAPACITY / 2;
        if base_line_count >= max_lines {
            return (0, 0);
        }

        // -- Cohesive HUD color palette --
        let axis_colors = [
            Vec4::new(0.95, 0.35, 0.35, 1.0), // +X  red
            Vec4::new(0.35, 0.90, 0.45, 1.0), // +Y  green
            Vec4::new(0.40, 0.65, 1.00, 1.0), // +Z  blue
            Vec4::new(1.00, 0.75, 0.25, 1.0), // +W  amber
        ];
        let rose_frame_color = Vec4::new(0.70, 0.72, 0.78, 0.85);
        let xy_frame_color = Vec4::new(0.35, 0.80, 0.85, 0.90);
        let zw_frame_color = Vec4::new(0.80, 0.50, 0.85, 0.90);
        let map_axis_color = Vec4::new(0.50, 0.52, 0.58, 0.50);
        let map_grid_color = Vec4::new(0.45, 0.47, 0.52, 0.18);
        let marker_xy_color = Vec4::new(0.55, 0.90, 0.92, 1.0);
        let marker_zw_color = Vec4::new(0.92, 0.72, 0.92, 1.0);
        let breadcrumb_xy_start = Vec4::new(0.18, 0.38, 0.45, 0.5);
        let breadcrumb_xy_end = Vec4::new(0.30, 0.85, 0.88, 1.0);
        let breadcrumb_zw_start = Vec4::new(0.32, 0.20, 0.38, 0.5);
        let breadcrumb_zw_end = Vec4::new(0.90, 0.48, 0.85, 1.0);
        let altimeter_color = Vec4::new(1.00, 0.75, 0.25, 1.0);
        let drift_color = Vec4::new(0.92, 0.90, 0.35, 1.0);
        let crosshair_color = Vec4::new(0.95, 0.95, 0.95, 1.0);
        let crosshair_outline = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let text_color = Vec4::new(0.82, 0.84, 0.88, 1.0);
        let text_dim_color = Vec4::new(0.58, 0.60, 0.65, 1.0);

        let (present_size, dpi_scale) = match self.window.as_ref() {
            Some(window) => {
                let s = window.inner_size();
                (
                    [s.width.max(1), s.height.max(1)],
                    window.scale_factor() as f32,
                )
            }
            None => (
                [
                    self.sized_buffers.render_dimensions[0].max(1),
                    self.sized_buffers.render_dimensions[1].max(1),
                ],
                1.0,
            ),
        };
        let present_w = present_size[0] as f32;
        let present_h = present_size[1] as f32;
        let dpi_scale = dpi_scale.max(1.0);
        let logical_height = present_h / dpi_scale;
        let layout_scale = (logical_height / 1080.0).clamp(0.85, 1.30);
        let hud_scale = dpi_scale * layout_scale;
        let text_scale = hud_scale * 1.5;
        let aspect = present_w / present_h;
        let px_top_left_to_ndc = |p: Vec2| -> Vec2 {
            Vec2::new((p.x / present_w) * 2.0 - 1.0, (p.y / present_h) * 2.0 - 1.0)
        };
        let px_delta_to_ndc =
            |d: Vec2| -> Vec2 { Vec2::new((d.x / present_w) * 2.0, (d.y / present_h) * 2.0) };

        let camera_h = mat5_mul_vec5(view_matrix_inverse, [0.0, 0.0, 0.0, 0.0, 1.0]);
        let inv_w = if camera_h[4].abs() > 1e-6 {
            1.0 / camera_h[4]
        } else {
            1.0
        };
        let camera_position = [
            camera_h[0] * inv_w,
            camera_h[1] * inv_w,
            camera_h[2] * inv_w,
            camera_h[3] * inv_w,
        ];
        let look_view = [
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let look_world_h = mat5_mul_vec5(view_matrix_inverse, look_view);
        let mut look_world = [
            look_world_h[0],
            look_world_h[1],
            look_world_h[2],
            look_world_h[3],
        ];
        let look_len = (look_world[0] * look_world[0]
            + look_world[1] * look_world[1]
            + look_world[2] * look_world[2]
            + look_world[3] * look_world[3])
            .sqrt();
        if look_len > 1e-6 {
            for v in &mut look_world {
                *v /= look_len;
            }
        }

        let now = Instant::now();
        if let (Some(prev_camera), Some(prev_time)) =
            (self.hud_previous_camera, self.hud_previous_sample_time)
        {
            let dt = (now - prev_time).as_secs_f32();
            if dt > 1e-4 {
                self.hud_w_velocity = (camera_position[3] - prev_camera[3]) / dt;
            }
        }
        self.hud_previous_camera = Some(camera_position);
        self.hud_previous_sample_time = Some(now);

        let should_append_breadcrumb = match self.hud_breadcrumbs.back() {
            Some(last) => {
                let dx = camera_position[0] - last[0];
                let dy = camera_position[1] - last[1];
                let dz = camera_position[2] - last[2];
                let dw = camera_position[3] - last[3];
                (dx * dx + dy * dy + dz * dz + dw * dw).sqrt() >= HUD_BREADCRUMB_MIN_STEP
            }
            None => true,
        };
        if should_append_breadcrumb {
            self.hud_breadcrumbs.push_back(camera_position);
            while self.hud_breadcrumbs.len() > HUD_BREADCRUMB_CAPACITY {
                self.hud_breadcrumbs.pop_front();
            }
        }

        let mut instance_centers = Vec::<[f32; 4]>::with_capacity(model_instances.len());
        for instance in model_instances {
            instance_centers.push(transform_model_point(
                &instance.model_transform,
                [0.5, 0.5, 0.5, 0.5],
            ));
        }

        let mut lines = Vec::<OverlayLine>::with_capacity(2048);

        // Axis rose: orientation widget with labeled arrows and circle boundary.
        let rose_margin_px = Vec2::new(24.0, 24.0) * hud_scale;
        let rose_radius_px = 58.0 * hud_scale;
        let rose_origin = px_top_left_to_ndc(Vec2::new(
            rose_margin_px.x + rose_radius_px,
            present_h - rose_margin_px.y - rose_radius_px,
        ));
        let rose_radius = px_delta_to_ndc(Vec2::splat(rose_radius_px));

        // Circle boundary -- double ring for thickness
        let circle_segments = 48;
        let outer_offset = px_delta_to_ndc(Vec2::splat(1.5 * hud_scale));
        for ring in 0..2 {
            let ring_radius = Vec2::new(
                rose_radius.x + outer_offset.x * ring as f32,
                rose_radius.y + outer_offset.y * ring as f32,
            );
            let ring_color = if ring == 0 {
                rose_frame_color
            } else {
                rose_frame_color * Vec4::new(1.0, 1.0, 1.0, 0.4)
            };
            for i in 0..circle_segments {
                let a0 = (i as f32 / circle_segments as f32) * std::f32::consts::TAU;
                let a1 = ((i + 1) as f32 / circle_segments as f32) * std::f32::consts::TAU;
                push_line(
                    &mut lines,
                    rose_origin + Vec2::new(a0.cos() * ring_radius.x, a0.sin() * ring_radius.y),
                    rose_origin + Vec2::new(a1.cos() * ring_radius.x, a1.sin() * ring_radius.y),
                    ring_color,
                );
            }
        }

        let axis_directions = [
            [1.0, 0.0, 0.0, 0.0], // X
            [0.0, 1.0, 0.0, 0.0], // Y
            [0.0, 0.0, 1.0, 0.0], // Z
            [0.0, 0.0, 0.0, 1.0], // W
        ];
        let fallback_dirs = [
            Vec2::new(1.0, 0.0),  // X: right
            Vec2::new(0.0, -1.0), // Y: up (Vulkan +Y=down)
            Vec2::new(-1.0, 0.0), // Z: left
            Vec2::new(0.0, 1.0),  // W: down (Vulkan +Y=down)
        ];
        let axis_labels = ["X", "Y", "Z", "W"];
        let arrow_len_px = rose_radius_px * 0.70;
        let arrowhead_back_px = rose_radius_px * 0.12;
        let arrowhead_side_px = rose_radius_px * 0.07;

        // Compute arrow data for all axes, then draw lines and labels
        struct ArrowData {
            color: Vec4,
            label_pos: Vec2,
        }
        let mut arrows = Vec::with_capacity(4);

        for axis_id in 0..4 {
            let dir = axis_directions[axis_id];
            let view_dir = mat5_mul_vec5(view_matrix, [dir[0], dir[1], dir[2], dir[3], 0.0]);
            let projected = project_view_point_to_ndc(
                [view_dir[0], view_dir[1], view_dir[2], view_dir[3]],
                focal_length_xy,
                aspect,
            );

            let mut ray_dir = projected
                .map(|p| Vec2::new(p.x * present_w * 0.5, p.y * present_h * 0.5))
                .unwrap_or(fallback_dirs[axis_id]);
            if ray_dir.length_squared() > 1e-8 {
                ray_dir = ray_dir.normalize();
            } else {
                ray_dir = fallback_dirs[axis_id];
            }

            // Depth shading: smooth fade based on how much axis points away
            // Positive Z or W in view space means pointing away from camera
            let depth_component = view_dir[2] + view_dir[3];
            let depth_norm = (depth_component / 1.5).clamp(-1.0, 1.0);
            let brightness = 1.0 - depth_norm.max(0.0) * 0.6; // 1.0 at front, 0.4 at back
            let mut axis_color = axis_colors[axis_id];
            axis_color.x *= brightness;
            axis_color.y *= brightness;
            axis_color.z *= brightness;
            axis_color.w = axis_colors[axis_id].w * (0.5 + brightness * 0.5);

            let tip = rose_origin + px_delta_to_ndc(ray_dir * arrow_len_px);
            push_line(&mut lines, rose_origin, tip, axis_color);

            // Arrowhead
            let side = Vec2::new(-ray_dir.y, ray_dir.x);
            push_line(
                &mut lines,
                tip,
                tip - px_delta_to_ndc(ray_dir * arrowhead_back_px)
                    + px_delta_to_ndc(side * arrowhead_side_px),
                axis_color,
            );
            push_line(
                &mut lines,
                tip,
                tip - px_delta_to_ndc(ray_dir * arrowhead_back_px)
                    - px_delta_to_ndc(side * arrowhead_side_px),
                axis_color,
            );

            // Label position: pushed outside the circle boundary for clean separation
            let label_pos = rose_origin + px_delta_to_ndc(ray_dir * (rose_radius_px * 1.08));
            arrows.push(ArrowData {
                color: axis_color,
                label_pos,
            });
        }

        let mut hud_quads = Vec::<HudVertex>::with_capacity(2048);

        let rose_label_text_size = 13.0 * text_scale;
        let readout_text_size = 12.0 * text_scale;

        // Rose background -- slightly larger than circle, with rounded feel via filled rect
        if let Some(hud_res) = self.hud_resources.as_ref() {
            let panel_bg = Vec4::new(0.05, 0.06, 0.08, 0.55);
            let rose_bg_half = px_delta_to_ndc(Vec2::splat(rose_radius_px * 1.22));
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                rose_origin - rose_bg_half,
                rose_origin + rose_bg_half,
                panel_bg,
            );
        }

        // Render axis labels at arrow tips
        for (axis_id, arrow) in arrows.iter().enumerate() {
            // Center the label on the label point: offset left by half-char width,
            // and up by half the cap height
            let char_half_w = rose_label_text_size * 0.28;
            let char_half_h = rose_label_text_size * 0.35;
            let label_offset = px_delta_to_ndc(Vec2::new(-char_half_w, char_half_h));
            let label_ndc = arrow.label_pos + label_offset;
            let label_px = ndc_to_pixels(label_ndc, present_size);
            if let Some(hud_res) = self.hud_resources.as_ref() {
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    axis_labels[axis_id],
                    label_px,
                    rose_label_text_size,
                    arrow.color,
                    present_size,
                );
            } else if let Some(font) = self.hud_font.as_ref() {
                push_text_lines(
                    &mut lines,
                    font,
                    axis_labels[axis_id],
                    label_px,
                    rose_label_text_size,
                    arrow.color,
                    present_size,
                );
            }
        }

        // Dual minimap: XZ on top (ground plane), YW below (height + 4th dim).
        let panel_margin_px = Vec2::new(24.0, 24.0) * hud_scale;
        let panel_gap_px = 18.0 * hud_scale;
        let min_panel_width_px = 220.0 * dpi_scale;
        let max_panel_width_px = (present_w - panel_margin_px.x * 2.0).max(min_panel_width_px);
        let panel_width_px = (300.0 * hud_scale).clamp(min_panel_width_px, max_panel_width_px);
        let min_panel_height_px = 120.0 * dpi_scale;
        let max_panel_height_px =
            ((present_h - panel_margin_px.y * 2.0 - panel_gap_px) * 0.5).max(min_panel_height_px);
        let panel_height_px = (168.0 * hud_scale).clamp(min_panel_height_px, max_panel_height_px);
        let right_px = present_w - panel_margin_px.x;
        let left_px = (right_px - panel_width_px).max(panel_margin_px.x);
        let xz_top_px = panel_margin_px.y;
        let xz_bottom_px = xz_top_px + panel_height_px;
        let yw_top_px = xz_bottom_px + panel_gap_px;
        let yw_bottom_px = yw_top_px + panel_height_px;

        let xz_center = px_top_left_to_ndc(Vec2::new(
            (left_px + right_px) * 0.5,
            (xz_top_px + xz_bottom_px) * 0.5,
        ));
        let yw_center = px_top_left_to_ndc(Vec2::new(
            (left_px + right_px) * 0.5,
            (yw_top_px + yw_bottom_px) * 0.5,
        ));
        let xz_min = px_top_left_to_ndc(Vec2::new(left_px, xz_top_px));
        let xz_max = px_top_left_to_ndc(Vec2::new(right_px, xz_bottom_px));
        let yw_min = px_top_left_to_ndc(Vec2::new(left_px, yw_top_px));
        let yw_max = px_top_left_to_ndc(Vec2::new(right_px, yw_bottom_px));
        let panel_half = Vec2::new(
            (right_px - left_px) / present_w,
            (xz_bottom_px - xz_top_px) / present_h,
        );
        let map_range = 12.0;
        let xz_frame_color = xy_frame_color;
        let yw_frame_color = zw_frame_color;
        let marker_xz_color = marker_xy_color;
        let marker_yw_color = marker_zw_color;
        let breadcrumb_xz_start = breadcrumb_xy_start;
        let breadcrumb_xz_end = breadcrumb_xy_end;
        let breadcrumb_yw_start = breadcrumb_zw_start;
        let breadcrumb_yw_end = breadcrumb_zw_end;

        // Panel frames
        push_rect(&mut lines, xz_min, xz_max, xz_frame_color);
        push_rect(&mut lines, yw_min, yw_max, yw_frame_color);

        // Grid lines -- subtle background grid every 4 world units
        let grid_interval = 4.0;
        let grid_max = (map_range / grid_interval) as i32;
        for &(center, pmin, pmax) in &[(xz_center, xz_min, xz_max), (yw_center, yw_min, yw_max)] {
            for i in 1..=grid_max {
                let frac = (i as f32 * grid_interval) / map_range;
                for sign in [-1.0f32, 1.0] {
                    // Vertical grid lines
                    let gx = center.x + sign * frac * panel_half.x * 0.9;
                    if gx > pmin.x && gx < pmax.x {
                        push_line(
                            &mut lines,
                            Vec2::new(gx, pmin.y),
                            Vec2::new(gx, pmax.y),
                            map_grid_color,
                        );
                    }
                    // Horizontal grid lines
                    let gy = center.y + sign * frac * panel_half.y * 0.9;
                    if gy > pmin.y && gy < pmax.y {
                        push_line(
                            &mut lines,
                            Vec2::new(pmin.x, gy),
                            Vec2::new(pmax.x, gy),
                            map_grid_color,
                        );
                    }
                }
            }
        }

        // Crosshair axes (drawn on top of grid)
        push_line(
            &mut lines,
            Vec2::new(xz_min.x, xz_center.y),
            Vec2::new(xz_max.x, xz_center.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(xz_center.x, xz_min.y),
            Vec2::new(xz_center.x, xz_max.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(yw_min.x, yw_center.y),
            Vec2::new(yw_max.x, yw_center.y),
            map_axis_color,
        );
        push_line(
            &mut lines,
            Vec2::new(yw_center.x, yw_min.y),
            Vec2::new(yw_center.x, yw_max.y),
            map_axis_color,
        );

        // Scale tick marks on crosshair axes
        let tick_size = px_delta_to_ndc(Vec2::splat(4.0 * hud_scale));
        let tick_interval = 4.0;
        let max_ticks = (map_range / tick_interval) as i32;
        for &(center, pmin, pmax) in &[(xz_center, xz_min, xz_max), (yw_center, yw_min, yw_max)] {
            for i in 1..=max_ticks {
                let frac = (i as f32 * tick_interval) / map_range;
                for sign in [-1.0f32, 1.0] {
                    let tx = center.x + sign * frac * panel_half.x * 0.9;
                    if tx > pmin.x && tx < pmax.x {
                        push_line(
                            &mut lines,
                            Vec2::new(tx, center.y - tick_size.y),
                            Vec2::new(tx, center.y + tick_size.y),
                            map_axis_color * 0.8,
                        );
                    }
                    let ty = center.y + sign * frac * panel_half.y * 0.9;
                    if ty > pmin.y && ty < pmax.y {
                        push_line(
                            &mut lines,
                            Vec2::new(center.x - tick_size.x, ty),
                            Vec2::new(center.x + tick_size.x, ty),
                            map_axis_color * 0.8,
                        );
                    }
                }
            }
        }

        // Direction indicator on XZ panel: filled arrow + FOV cone
        {
            let yaw_x = look_world[0];
            let yaw_z = look_world[2];
            let yaw_len = (yaw_x * yaw_x + yaw_z * yaw_z).sqrt();
            if yaw_len > 1e-4 {
                let dx = yaw_x / yaw_len;
                let dz = yaw_z / yaw_len;
                let panel_aspect = panel_half.x / panel_half.y;
                let wedge_dir = Vec2::new(dx, -dz * panel_aspect).normalize();
                let wedge_side = Vec2::new(-wedge_dir.y, wedge_dir.x);
                let wedge_len = px_delta_to_ndc(Vec2::splat(14.0 * hud_scale));
                let wedge_width = px_delta_to_ndc(Vec2::splat(7.0 * hud_scale));
                let wedge_tip =
                    xz_center + Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y);
                let wedge_left = xz_center
                    - Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y) * 0.3
                    + Vec2::new(wedge_side.x * wedge_width.x, wedge_side.y * wedge_width.y);
                let wedge_right = xz_center
                    - Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y) * 0.3
                    - Vec2::new(wedge_side.x * wedge_width.x, wedge_side.y * wedge_width.y);
                let wedge_color = Vec4::new(1.0, 1.0, 1.0, 0.95);
                // Fill the triangle with horizontal scan lines
                let fill_steps = 6i32;
                for s in 0..=fill_steps {
                    let t = s as f32 / fill_steps as f32;
                    let l = wedge_tip.lerp(wedge_left, t);
                    let r = wedge_tip.lerp(wedge_right, t);
                    push_line(
                        &mut lines,
                        l,
                        r,
                        Vec4::new(1.0, 1.0, 1.0, 0.65 * (1.0 - t * 0.5)),
                    );
                }
                // Outline
                push_line(&mut lines, wedge_tip, wedge_left, wedge_color);
                push_line(&mut lines, wedge_tip, wedge_right, wedge_color);
                push_line(&mut lines, wedge_left, wedge_right, wedge_color);

                // FOV cone: two lines extending from center showing approximate field of view
                let fov_half_angle = 0.45_f32; // ~26 degrees half-angle
                let fov_len = px_delta_to_ndc(Vec2::splat(panel_width_px * 0.28));
                let cos_a = fov_half_angle.cos();
                let sin_a = fov_half_angle.sin();
                let fov_left_dir = Vec2::new(
                    wedge_dir.x * cos_a - wedge_dir.y * sin_a,
                    wedge_dir.x * sin_a + wedge_dir.y * cos_a,
                );
                let fov_right_dir = Vec2::new(
                    wedge_dir.x * cos_a + wedge_dir.y * sin_a,
                    -wedge_dir.x * sin_a + wedge_dir.y * cos_a,
                );
                let fov_color = Vec4::new(1.0, 1.0, 1.0, 0.12);
                push_line(
                    &mut lines,
                    xz_center,
                    xz_center + Vec2::new(fov_left_dir.x * fov_len.x, fov_left_dir.y * fov_len.y),
                    fov_color,
                );
                push_line(
                    &mut lines,
                    xz_center,
                    xz_center + Vec2::new(fov_right_dir.x * fov_len.x, fov_right_dir.y * fov_len.y),
                    fov_color,
                );
            }
        }

        // Direction indicator on YW panel: filled arrow
        {
            let look_y = look_world[1];
            let look_w = look_world[3];
            let yw_len = (look_y * look_y + look_w * look_w).sqrt();
            if yw_len > 1e-4 {
                let dy = look_y / yw_len;
                let dw = look_w / yw_len;
                let panel_aspect = panel_half.x / panel_half.y;
                let wedge_dir = Vec2::new(dy, -dw * panel_aspect).normalize();
                let wedge_side = Vec2::new(-wedge_dir.y, wedge_dir.x);
                let wedge_len = px_delta_to_ndc(Vec2::splat(14.0 * hud_scale));
                let wedge_width = px_delta_to_ndc(Vec2::splat(7.0 * hud_scale));
                let wedge_tip =
                    yw_center + Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y);
                let wedge_left = yw_center
                    - Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y) * 0.3
                    + Vec2::new(wedge_side.x * wedge_width.x, wedge_side.y * wedge_width.y);
                let wedge_right = yw_center
                    - Vec2::new(wedge_dir.x * wedge_len.x, wedge_dir.y * wedge_len.y) * 0.3
                    - Vec2::new(wedge_side.x * wedge_width.x, wedge_side.y * wedge_width.y);
                let wedge_color = Vec4::new(1.0, 1.0, 1.0, 0.95);
                // Fill the triangle
                let fill_steps = 6i32;
                for s in 0..=fill_steps {
                    let t = s as f32 / fill_steps as f32;
                    let l = wedge_tip.lerp(wedge_left, t);
                    let r = wedge_tip.lerp(wedge_right, t);
                    push_line(
                        &mut lines,
                        l,
                        r,
                        Vec4::new(1.0, 1.0, 1.0, 0.65 * (1.0 - t * 0.5)),
                    );
                }
                // Outline
                push_line(&mut lines, wedge_tip, wedge_left, wedge_color);
                push_line(&mut lines, wedge_tip, wedge_right, wedge_color);
                push_line(&mut lines, wedge_left, wedge_right, wedge_color);
            }
        }

        // Panel labels: title, cardinal directions, and scale bar
        let minimap_label_size = 11.0 * text_scale;
        let cardinal_label_size = 9.0 * text_scale;
        let label_inset = px_delta_to_ndc(Vec2::new(10.0 * hud_scale, 8.0 * hud_scale));
        let edge_inset = px_delta_to_ndc(Vec2::new(6.0 * hud_scale, 5.0 * hud_scale));
        if let Some(hud_res) = self.hud_resources.as_ref() {
            // Panel titles inside top-left corner
            let xz_title_px = ndc_to_pixels(
                Vec2::new(xz_min.x + label_inset.x, xz_min.y + label_inset.y),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "XZ",
                xz_title_px,
                minimap_label_size,
                xz_frame_color,
                present_size,
            );
            let yw_title_px = ndc_to_pixels(
                Vec2::new(yw_min.x + label_inset.x, yw_min.y + label_inset.y),
                present_size,
            );
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "YW",
                yw_title_px,
                minimap_label_size,
                yw_frame_color,
                present_size,
            );

            // Cardinal labels on XZ panel edges: +X right, -X left, +Z top, -Z bottom
            // In Vulkan NDC: xz_min.y = top of screen (lower NDC), xz_max.y = bottom
            let xz_cardinals: &[(&str, Vec2)] = &[
                ("+X", Vec2::new(xz_max.x - edge_inset.x * 3.0, xz_center.y)),
                ("-X", Vec2::new(xz_min.x + edge_inset.x, xz_center.y)),
                (
                    "+Z",
                    Vec2::new(xz_center.x - edge_inset.x, xz_min.y + edge_inset.y),
                ),
                (
                    "-Z",
                    Vec2::new(xz_center.x - edge_inset.x, xz_max.y - edge_inset.y * 3.0),
                ),
            ];
            for (label, ndc_pos) in xz_cardinals {
                let px = ndc_to_pixels(*ndc_pos, present_size);
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    label,
                    px,
                    cardinal_label_size,
                    text_dim_color,
                    present_size,
                );
            }

            // Cardinal labels on YW panel edges: +Y right, -Y left, +W top, -W bottom
            let yw_cardinals: &[(&str, Vec2)] = &[
                ("+Y", Vec2::new(yw_max.x - edge_inset.x * 3.0, yw_center.y)),
                ("-Y", Vec2::new(yw_min.x + edge_inset.x, yw_center.y)),
                (
                    "+W",
                    Vec2::new(yw_center.x - edge_inset.x, yw_min.y + edge_inset.y),
                ),
                (
                    "-W",
                    Vec2::new(yw_center.x - edge_inset.x, yw_max.y - edge_inset.y * 3.0),
                ),
            ];
            for (label, ndc_pos) in yw_cardinals {
                let px = ndc_to_pixels(*ndc_pos, present_size);
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    label,
                    px,
                    cardinal_label_size,
                    text_dim_color,
                    present_size,
                );
            }

            // Scale bar at bottom-right of XZ panel
            let scale_bar_world = 4.0_f32; // 4 world units
            let scale_bar_frac = scale_bar_world / map_range;
            let scale_bar_ndc_w = scale_bar_frac * panel_half.x * 0.9 * 2.0;
            let bar_right = xz_max.x - edge_inset.x;
            let bar_left = bar_right - scale_bar_ndc_w;
            let bar_y = xz_max.y - edge_inset.y * 2.0;
            let bar_tick_h = px_delta_to_ndc(Vec2::new(0.0, 3.0 * hud_scale)).y;
            let scale_bar_color = Vec4::new(0.70, 0.72, 0.78, 0.7);
            push_line(
                &mut lines,
                Vec2::new(bar_left, bar_y),
                Vec2::new(bar_right, bar_y),
                scale_bar_color,
            );
            push_line(
                &mut lines,
                Vec2::new(bar_left, bar_y - bar_tick_h),
                Vec2::new(bar_left, bar_y + bar_tick_h),
                scale_bar_color,
            );
            push_line(
                &mut lines,
                Vec2::new(bar_right, bar_y - bar_tick_h),
                Vec2::new(bar_right, bar_y + bar_tick_h),
                scale_bar_color,
            );
            // Scale bar label
            let bar_label_ndc = Vec2::new(
                bar_left,
                bar_y + bar_tick_h + px_delta_to_ndc(Vec2::new(0.0, 2.0 * hud_scale)).y,
            );
            let bar_label_px = ndc_to_pixels(bar_label_ndc, present_size);
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                "4u",
                bar_label_px,
                cardinal_label_size,
                scale_bar_color,
                present_size,
            );
        } else if let Some(font) = self.hud_font.as_ref() {
            let xz_title_px = ndc_to_pixels(
                Vec2::new(xz_min.x + label_inset.x, xz_min.y + label_inset.y),
                present_size,
            );
            push_text_lines(
                &mut lines,
                font,
                "XZ",
                xz_title_px,
                minimap_label_size,
                xz_frame_color,
                present_size,
            );
            let yw_title_px = ndc_to_pixels(
                Vec2::new(yw_min.x + label_inset.x, yw_min.y + label_inset.y),
                present_size,
            );
            push_text_lines(
                &mut lines,
                font,
                "YW",
                yw_title_px,
                minimap_label_size,
                yw_frame_color,
                present_size,
            );
        }

        // Instance markers: XZ panel uses indices [0]=X, [2]=Z; YW panel uses [1]=Y, [3]=W
        let marker_radius = px_delta_to_ndc(Vec2::splat(3.0 * hud_scale));
        for center in &instance_centers {
            let xz_point = map_to_panel(
                xz_center,
                panel_half * 0.9,
                map_range,
                center[0] - camera_position[0],
                center[2] - camera_position[2],
            );
            let yw_point = map_to_panel(
                yw_center,
                panel_half * 0.9,
                map_range,
                center[1] - camera_position[1],
                center[3] - camera_position[3],
            );
            push_cross(&mut lines, xz_point, marker_radius, marker_xz_color);
            push_cross(&mut lines, yw_point, marker_radius, marker_yw_color);
        }

        // Breadcrumb trails: XZ uses [0],[2]; YW uses [1],[3]
        if self.hud_breadcrumbs.len() >= 2 {
            let denom = (self.hud_breadcrumbs.len() - 1) as f32;
            for i in 1..self.hud_breadcrumbs.len() {
                let prev = self.hud_breadcrumbs[i - 1];
                let next = self.hud_breadcrumbs[i];
                let t = i as f32 / denom;
                let breadcrumb_xz_color = breadcrumb_xz_start.lerp(breadcrumb_xz_end, t);
                let breadcrumb_yw_color = breadcrumb_yw_start.lerp(breadcrumb_yw_end, t);

                let xz_prev = map_to_panel(
                    xz_center,
                    panel_half * 0.9,
                    map_range,
                    prev[0] - camera_position[0],
                    prev[2] - camera_position[2],
                );
                let xz_next = map_to_panel(
                    xz_center,
                    panel_half * 0.9,
                    map_range,
                    next[0] - camera_position[0],
                    next[2] - camera_position[2],
                );
                push_line(&mut lines, xz_prev, xz_next, breadcrumb_xz_color);

                let yw_prev = map_to_panel(
                    yw_center,
                    panel_half * 0.9,
                    map_range,
                    prev[1] - camera_position[1],
                    prev[3] - camera_position[3],
                );
                let yw_next = map_to_panel(
                    yw_center,
                    panel_half * 0.9,
                    map_range,
                    next[1] - camera_position[1],
                    next[3] - camera_position[3],
                );
                push_line(&mut lines, yw_prev, yw_next, breadcrumb_yw_color);
            }
        }

        // W altimeter and drift gauge to visualize W position and velocity.
        let altimeter_x = px_top_left_to_ndc(Vec2::new(left_px - 34.0 * hud_scale, 0.0)).x;
        let altimeter_min_y = xz_min.y;
        let altimeter_max_y = yw_max.y;
        let altimeter_range = 12.0;
        let zero_ratio = 0.5;
        let zero_y = altimeter_min_y + (altimeter_max_y - altimeter_min_y) * zero_ratio;
        let w_ratio = ((camera_position[3] / altimeter_range) * 0.5 + 0.5).clamp(0.0, 1.0);
        // Higher W should be higher on screen, so map to smaller NDC Y values.
        let w_y = altimeter_min_y + (altimeter_max_y - altimeter_min_y) * (1.0 - w_ratio);
        let alt_tick_small = px_delta_to_ndc(Vec2::new(8.0 * hud_scale, 0.0)).x;
        let alt_tick_mid = px_delta_to_ndc(Vec2::new(12.0 * hud_scale, 0.0)).x;
        let alt_tick_large = px_delta_to_ndc(Vec2::new(14.0 * hud_scale, 0.0)).x;

        push_line(
            &mut lines,
            Vec2::new(altimeter_x, altimeter_min_y),
            Vec2::new(altimeter_x, altimeter_max_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - alt_tick_small, altimeter_min_y),
            Vec2::new(altimeter_x + alt_tick_small, altimeter_min_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - alt_tick_small, altimeter_max_y),
            Vec2::new(altimeter_x + alt_tick_small, altimeter_max_y),
            altimeter_color,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - alt_tick_mid, zero_y),
            Vec2::new(altimeter_x + alt_tick_mid, zero_y),
            altimeter_color * 0.8,
        );
        push_line(
            &mut lines,
            Vec2::new(altimeter_x - alt_tick_large, w_y),
            Vec2::new(altimeter_x + alt_tick_large, w_y),
            altimeter_color,
        );

        let drift_center = Vec2::new(
            altimeter_x,
            yw_max.y + px_delta_to_ndc(Vec2::new(0.0, 26.0 * hud_scale)).y,
        );
        let drift_half_width = px_delta_to_ndc(Vec2::new(52.0 * hud_scale, 0.0)).x;
        let drift_minor_half_height = px_delta_to_ndc(Vec2::new(0.0, 5.0 * hud_scale)).y;
        let drift_major_half_height = px_delta_to_ndc(Vec2::new(0.0, 8.0 * hud_scale)).y;
        let drift_ratio = (self.hud_w_velocity / 6.0).clamp(-1.0, 1.0);
        let drift_x = drift_center.x + drift_ratio * drift_half_width;

        push_line(
            &mut lines,
            Vec2::new(drift_center.x - drift_half_width, drift_center.y),
            Vec2::new(drift_center.x + drift_half_width, drift_center.y),
            drift_color * 0.7,
        );
        push_line(
            &mut lines,
            Vec2::new(drift_center.x, drift_center.y - drift_minor_half_height),
            Vec2::new(drift_center.x, drift_center.y + drift_minor_half_height),
            drift_color * 0.8,
        );
        push_line(
            &mut lines,
            Vec2::new(drift_x, drift_center.y - drift_major_half_height),
            Vec2::new(drift_x, drift_center.y + drift_major_half_height),
            drift_color,
        );

        let fps = if self.frame_time_ms > 0.1 {
            1000.0 / self.frame_time_ms
        } else {
            0.0
        };
        let mut readout_text = String::new();
        if let Some(label) = rotation_label {
            readout_text.push_str(label);
        }
        if readout_mode == HudReadoutMode::Full {
            if !readout_text.is_empty() {
                readout_text.push('\n');
            }
            // Performance line -- compact
            readout_text.push_str(&format!(
                "{:.0} fps  {:.1}/{:.1} ms  [{}]",
                fps,
                self.frame_time_ms,
                self.profiler.last_gpu_total_ms,
                self.last_backend.label()
            ));
            // Position and orientation table -- clean aligned columns
            readout_text.push_str("\n          X        Y        Z        W");
            readout_text.push_str(&format!(
                "\npos  {:+7.1}  {:+7.1}  {:+7.1}  {:+7.1}",
                camera_position[0], camera_position[1], camera_position[2], camera_position[3],
            ));
            readout_text.push_str(&format!(
                "\ndir  {:+7.3}  {:+7.3}  {:+7.3}  {:+7.3}",
                look_world[0], look_world[1], look_world[2], look_world[3],
            ));
            if let Some(hit) = target_hit_voxel {
                readout_text.push_str(&format!(
                    "\nhit  {:+7}  {:+7}  {:+7}  {:+7}",
                    hit[0], hit[1], hit[2], hit[3]
                ));
            }
            if let Some(face) = target_hit_face {
                readout_text.push_str(&format!(
                    "\nface {:+7}  {:+7}  {:+7}  {:+7}",
                    face[0], face[1], face[2], face[3]
                ));
            }
            if self.last_backend == RenderBackend::VoxelTraversal {
                readout_text.push_str(&format!(
                    "\nVTE c:{} f:{} e:{} m:{}",
                    self.vte_debug_counters.candidate_chunks,
                    self.vte_debug_counters.frustum_culled_chunks,
                    self.vte_debug_counters.empty_chunks_skipped,
                    self.vte_debug_counters.macro_cells_skipped,
                ));
                readout_text.push_str(&format!(
                    "\n    cs:{} vs:{} h:{} s:{}",
                    self.vte_debug_counters.chunk_steps,
                    self.vte_debug_counters.voxel_steps,
                    self.vte_debug_counters.primary_hits,
                    self.vte_debug_counters.s_samples
                ));
                if self.vte_debug_counters.visible_set_hash_valid {
                    readout_text.push_str(&format!(
                        "\n    vh:{:08x}",
                        self.vte_debug_counters.visible_set_hash
                    ));
                }
                if self.vte_compare_stats.compared > 0 || self.vte_compare_stats.mismatches > 0 {
                    readout_text.push_str(&format!(
                        "\n    cmp:{}/{} mm:{} hs:{} cm:{}",
                        self.vte_compare_stats.matches,
                        self.vte_compare_stats.compared,
                        self.vte_compare_stats.mismatches,
                        self.vte_compare_stats.hit_state_mismatches,
                        self.vte_compare_stats.chunk_material_mismatches
                    ));
                }
            }
            if !self.profiler.last_frame_phases.is_empty() {
                readout_text.push('\n');
                for (name, ms) in &self.profiler.last_frame_phases {
                    readout_text.push_str(&format!("\n {}:{:.1}", name, ms));
                }
            }
        } else if readout_text.is_empty() {
            readout_text.push_str("vectors");
        }
        // Top-left text panel in Vulkan NDC (+Y is down), anchored in pixel space.
        let text_margin_px = Vec2::new(18.0, 18.0) * hud_scale;
        let readout_bg_min = px_top_left_to_ndc(text_margin_px);
        let panel_padding_px = 10.0 * hud_scale;
        let max_line_chars = readout_text
            .lines()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(1) as f32;
        let estimated_char_width_px = readout_text_size * 0.56;
        let panel_width_px = (max_line_chars * estimated_char_width_px + panel_padding_px * 2.0)
            .clamp(320.0 * hud_scale, (present_w * 0.62).max(320.0 * hud_scale));
        let line_height_px = (readout_text_size * 1.25).max(12.0 * dpi_scale);
        let readout_line_count = readout_text.lines().count().max(1) as f32;
        let panel_height_px = readout_line_count * line_height_px + panel_padding_px * 2.0;
        let width_ndc = (panel_width_px / present_w) * 2.0;
        let height_ndc = (panel_height_px / present_h) * 2.0;
        let readout_bg_max = Vec2::new(
            (readout_bg_min.x + width_ndc).min(0.95),
            (readout_bg_min.y + height_ndc).min(0.95),
        );
        let readout_anchor_ndc = Vec2::new(
            readout_bg_min.x + (panel_padding_px / present_w) * 2.0,
            readout_bg_min.y + (panel_padding_px / present_h) * 2.0,
        );

        if let Some(hud_res) = self.hud_resources.as_ref() {
            let panel_bg = Vec4::new(0.05, 0.06, 0.08, 0.50);
            let text_panel_bg = Vec4::new(0.04, 0.05, 0.07, 0.82);

            // Semi-transparent backgrounds behind minimap panels
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                xz_min,
                xz_max,
                panel_bg,
            );
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                yw_min,
                yw_max,
                panel_bg,
            );

            // Darker background behind text readout
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                readout_bg_min,
                readout_bg_max,
                text_panel_bg,
            );

            // Accent border on panels (top edge highlight)
            let border_h = px_delta_to_ndc(Vec2::new(0.0, 1.0 * hud_scale)).y;
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                Vec2::new(xz_min.x, xz_min.y),
                Vec2::new(xz_max.x, xz_min.y + border_h),
                Vec4::new(xz_frame_color.x, xz_frame_color.y, xz_frame_color.z, 0.35),
            );
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                Vec2::new(yw_min.x, yw_min.y),
                Vec2::new(yw_max.x, yw_min.y + border_h),
                Vec4::new(yw_frame_color.x, yw_frame_color.y, yw_frame_color.z, 0.35),
            );
            push_filled_rect_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                Vec2::new(readout_bg_min.x, readout_bg_min.y),
                Vec2::new(readout_bg_max.x, readout_bg_min.y + border_h),
                Vec4::new(0.45, 0.50, 0.60, 0.40),
            );

            let readout_anchor_px = ndc_to_pixels(readout_anchor_ndc, present_size);
            push_text_quads(
                &mut hud_quads,
                &hud_res.font_atlas,
                &readout_text,
                readout_anchor_px,
                readout_text_size,
                text_color,
                present_size,
            );
        } else if let Some(font) = self.hud_font.as_ref() {
            let readout_anchor_px = ndc_to_pixels(readout_anchor_ndc, present_size);
            push_text_lines(
                &mut lines,
                font,
                &readout_text,
                readout_anchor_px,
                readout_text_size,
                text_color,
                present_size,
            );
        }

        if let Some(hud_res) = self.hud_resources.as_ref() {
            if !player_tags.is_empty() {
                let mut sorted_tags: Vec<&HudPlayerTag> = player_tags.iter().collect();
                sorted_tags.sort_by(|a, b| {
                    a.scale
                        .partial_cmp(&b.scale)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for tag in sorted_tags.into_iter().take(48) {
                    let text = tag.text.trim();
                    if text.is_empty() {
                        continue;
                    }

                    let scale = tag.scale.clamp(0.55, 1.8);
                    let text_size =
                        (11.0 * text_scale * scale).clamp(10.0 * hud_scale, 26.0 * hud_scale);
                    let pad_px = (4.0 * hud_scale * scale).clamp(2.0 * dpi_scale, 10.0 * hud_scale);
                    let estimated_char_width_px = text_size * 0.56;
                    let panel_width_px = (text.chars().count().max(1) as f32
                        * estimated_char_width_px
                        + pad_px * 2.0)
                        .clamp(36.0 * dpi_scale, 540.0 * hud_scale);
                    let panel_height_px =
                        (text_size * 1.25 + pad_px * 2.0).clamp(16.0 * dpi_scale, 88.0 * hud_scale);

                    let width_ndc = (panel_width_px / present_w) * 2.0;
                    let height_ndc = (panel_height_px / present_h) * 2.0;
                    let offset_ndc_y = (16.0 * hud_scale * scale / present_h) * 2.0;
                    let margin_ndc = px_delta_to_ndc(Vec2::splat(8.0 * hud_scale));

                    let center_x = tag.anchor_ndc[0].clamp(
                        -1.0 + width_ndc * 0.5 + margin_ndc.x,
                        1.0 - width_ndc * 0.5 - margin_ndc.x,
                    );
                    let top_y = (tag.anchor_ndc[1] - offset_ndc_y)
                        .clamp(-1.0 + height_ndc + margin_ndc.y, 1.0 - margin_ndc.y);

                    let bg_min = Vec2::new(center_x - width_ndc * 0.5, top_y - height_ndc);
                    let bg_max = Vec2::new(center_x + width_ndc * 0.5, top_y);

                    push_filled_rect_quads(
                        &mut hud_quads,
                        &hud_res.font_atlas,
                        bg_min,
                        bg_max,
                        Vec4::new(0.0, 0.0, 0.0, tag.bg_alpha.clamp(0.25, 0.90)),
                    );

                    let text_anchor_ndc = Vec2::new(
                        bg_min.x + (pad_px / present_w) * 2.0,
                        bg_min.y + (pad_px / present_h) * 2.0,
                    );
                    let text_anchor_px = ndc_to_pixels(text_anchor_ndc, present_size);
                    push_text_quads(
                        &mut hud_quads,
                        &hud_res.font_atlas,
                        text,
                        text_anchor_px,
                        text_size,
                        Vec4::new(0.96, 0.97, 1.00, 1.0),
                        present_size,
                    );
                }
            }
        } else if let Some(font) = self.hud_font.as_ref() {
            for tag in player_tags.iter().take(48) {
                let text = tag.text.trim();
                if text.is_empty() {
                    continue;
                }
                let scale = tag.scale.clamp(0.55, 1.8);
                let text_size =
                    (11.0 * text_scale * scale).clamp(10.0 * hud_scale, 26.0 * hud_scale);
                let anchor = Vec2::new(tag.anchor_ndc[0], tag.anchor_ndc[1]);
                let label_ndc = anchor + px_delta_to_ndc(Vec2::new(0.0, -18.0 * hud_scale * scale));
                let label_px = ndc_to_pixels(label_ndc, present_size);
                push_text_lines(
                    &mut lines,
                    font,
                    text,
                    label_px,
                    text_size,
                    Vec4::new(0.96, 0.97, 1.00, 1.0),
                    present_size,
                );
            }
        }

        push_minecraft_crosshair(
            &mut lines,
            present_size,
            Vec2::ZERO,
            dpi_scale,
            crosshair_color,
            crosshair_outline,
        );

        // WAILA: show targeted block name in a styled box at the top center of the screen
        if let Some(name) = waila_text {
            if let Some(hud_res) = self.hud_resources.as_ref() {
                let waila_size = 16.0 * hud_scale;
                let waila_color = Vec4::new(0.95, 0.95, 0.95, 0.85);
                let tw = text_width_px(&hud_res.font_atlas, name, waila_size);

                // Position at top center of screen
                let center_x = present_w * 0.5;
                let padding_h = 8.0 * hud_scale;
                let padding_v = 4.0 * hud_scale;

                // Y-up coordinate system: Y=0 is bottom, Y=present_h is top
                // Place box 40px from top edge
                let box_top_y = present_h - 40.0 * hud_scale;

                // Calculate text height (approximate based on font size)
                let text_height = waila_size;
                let box_height = text_height + padding_v * 2.0;
                let box_width = tw + padding_h * 2.0;

                // Box positioning in pixels (Y-up)
                let box_min_x = center_x - box_width * 0.5;
                let box_max_x = center_x + box_width * 0.5;
                let box_min_y = box_top_y - box_height;
                let box_max_y = box_top_y;

                // Convert to NDC for background rect
                let box_min_ndc = pixels_to_ndc(Vec2::new(box_min_x, box_min_y), present_size);
                let box_max_ndc = pixels_to_ndc(Vec2::new(box_max_x, box_max_y), present_size);

                // Draw dark semi-transparent background
                let bg_color = Vec4::new(0.1, 0.1, 0.15, 0.7);
                push_filled_rect_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    box_min_ndc,
                    box_max_ndc,
                    bg_color,
                );

                // Draw text centered in the box
                let text_top_left_px = Vec2::new(center_x - tw * 0.5, box_max_y - padding_v);
                push_text_quads(
                    &mut hud_quads,
                    &hud_res.font_atlas,
                    name,
                    text_top_left_px,
                    waila_size,
                    waila_color,
                    present_size,
                );
            }
        }

        // Write line data to GPU buffer
        let lines_to_write = lines.len().min(max_lines - base_line_count);
        if lines_to_write > 0 {
            let mut writer = self.frames_in_flight[frame_idx]
                .line_vertexes_buffer
                .write()
                .unwrap();
            for (line_id, segment) in lines.iter().take(lines_to_write).enumerate() {
                let vertex_id = (base_line_count + line_id) * 2;
                writer[vertex_id] =
                    LineVertex::new_with_style(segment.start, segment.color, segment.style);
                writer[vertex_id + 1] =
                    LineVertex::new_with_style(segment.end, segment.color, segment.style);
            }
        }

        // Write HUD quad data to GPU buffer
        let hud_verts_to_write = hud_quads.len().min(HUD_VERTEX_CAPACITY);
        if hud_verts_to_write > 0 {
            if let Some(hud_buf) = self.frames_in_flight[frame_idx].hud_vertex_buffer.as_ref() {
                let mut writer = hud_buf.write().unwrap();
                for (i, v) in hud_quads.iter().take(hud_verts_to_write).enumerate() {
                    writer[i] = *v;
                }
            }
        }

        (lines_to_write, hud_verts_to_write)
    }

    pub(super) fn write_custom_overlay_lines(
        &mut self,
        frame_idx: usize,
        base_line_count: usize,
        custom_lines: &[CustomOverlayLine],
    ) -> usize {
        let max_lines = LINE_VERTEX_CAPACITY / 2;
        if base_line_count >= max_lines || custom_lines.is_empty() {
            return 0;
        }

        let lines_to_write = custom_lines.len().min(max_lines - base_line_count);
        let mut writer = self.frames_in_flight[frame_idx]
            .line_vertexes_buffer
            .write()
            .unwrap();
        for (line_id, line) in custom_lines.iter().take(lines_to_write).enumerate() {
            let vertex_id = (base_line_count + line_id) * 2;
            let start = Vec2::from_array(line.start_ndc);
            let end = Vec2::from_array(line.end_ndc);
            let color = Vec4::from_array(line.color);
            writer[vertex_id] = LineVertex::new(start, color);
            writer[vertex_id + 1] = LineVertex::new(end, color);
        }

        lines_to_write
    }
}
