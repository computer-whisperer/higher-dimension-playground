use super::*;

impl App {
    pub(super) fn current_info_hud_text(
        &self,
        pair: RotationPair,
        look_dir: [f32; 4],
        edit_reach: f32,
        highlight_mode: EditHighlightModeArg,
        vte_sweep_status: &str,
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
        stream_first_node_desc: Option<&str>,
        stream_final_solid_leaf_desc: Option<&str>,
    ) -> Option<String> {
        match self.info_panel_mode {
            InfoPanelMode::Off => None,
            InfoPanelMode::VectorTable => {
                Some(self.vector_table_hud_text(look_dir, target_hit_voxel, target_hit_face))
            }
            InfoPanelMode::VectorTable2 => {
                Some(self.vector_table2_hud_text(look_dir, target_hit_voxel, target_hit_face))
            }
            InfoPanelMode::Full => Some(self.full_info_hud_text(
                pair,
                look_dir,
                edit_reach,
                highlight_mode,
                vte_sweep_status,
                target_hit_voxel,
                target_hit_face,
                stream_first_node_desc,
                stream_final_solid_leaf_desc,
            )),
        }
    }

    pub(super) fn vector_table_hud_text(
        &self,
        _look_dir: [f32; 4],
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
    ) -> String {
        let (_view_x, _view_y, view_z, view_w) = self.current_view_basis();
        let center_forward = normalize4([
            view_z[0] + view_w[0],
            view_z[1] + view_w[1],
            view_z[2] + view_w[2],
            view_z[3] + view_w[3],
        ]);
        let hidden_side = normalize4([
            view_w[0] - view_z[0],
            view_w[1] - view_z[1],
            view_w[2] - view_z[2],
            view_w[3] - view_z[3],
        ]);

        let mut text = String::new();
        text.push_str(
            "vec       |        X |        Y |        Z |        W\n\
             ----------+----------+----------+----------+----------",
        );
        text.push_str(&format!(
            "\n{:<9} | {:+8.2} | {:+8.2} | {:+8.2} | {:+8.2}",
            "pos",
            self.camera.position[0],
            self.camera.position[1],
            self.camera.position[2],
            self.camera.position[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "fwd_ctr", center_forward[0], center_forward[1], center_forward[2], center_forward[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "side_w", hidden_side[0], hidden_side[1], hidden_side[2], hidden_side[3],
        ));
        if let Some(hit) = target_hit_voxel {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "block", hit[0], hit[1], hit[2], hit[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "block", "--", "--", "--", "--"
            ));
        }
        if let Some(face) = target_hit_face {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "face", face[0], face[1], face[2], face[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "face", "--", "--", "--", "--"
            ));
        }
        text
    }

    pub(super) fn vector_table2_hud_text(
        &self,
        _look_dir: [f32; 4],
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
    ) -> String {
        let (view_x, view_y, view_z, view_w) = self.current_view_basis();
        let center_forward = normalize4([
            view_z[0] + view_w[0],
            view_z[1] + view_w[1],
            view_z[2] + view_w[2],
            view_z[3] + view_w[3],
        ]);
        let hidden_side = normalize4([
            view_w[0] - view_z[0],
            view_w[1] - view_z[1],
            view_w[2] - view_z[2],
            view_w[3] - view_z[3],
        ]);

        let mut text = String::new();
        text.push_str(
            "vec       |        X |        Y |        Z |        W\n\
             ----------+----------+----------+----------+----------",
        );
        text.push_str(&format!(
            "\n{:<9} | {:+8.2} | {:+8.2} | {:+8.2} | {:+8.2}",
            "pos",
            self.camera.position[0],
            self.camera.position[1],
            self.camera.position[2],
            self.camera.position[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "fwd_ctr", center_forward[0], center_forward[1], center_forward[2], center_forward[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "side_w", hidden_side[0], hidden_side[1], hidden_side[2], hidden_side[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "view_x", view_x[0], view_x[1], view_x[2], view_x[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "view_y", view_y[0], view_y[1], view_y[2], view_y[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "view_z", view_z[0], view_z[1], view_z[2], view_z[3],
        ));
        text.push_str(&format!(
            "\n{:<9} | {:+8.3} | {:+8.3} | {:+8.3} | {:+8.3}",
            "view_w", view_w[0], view_w[1], view_w[2], view_w[3],
        ));
        if let Some(hit) = target_hit_voxel {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "block", hit[0], hit[1], hit[2], hit[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "block", "--", "--", "--", "--"
            ));
        }
        if let Some(face) = target_hit_face {
            text.push_str(&format!(
                "\n{:<9} | {:+8} | {:+8} | {:+8} | {:+8}",
                "face", face[0], face[1], face[2], face[3]
            ));
        } else {
            text.push_str(&format!(
                "\n{:<9} | {:>8} | {:>8} | {:>8} | {:>8}",
                "face", "--", "--", "--", "--"
            ));
        }
        text
    }

    pub(super) fn full_info_hud_text(
        &self,
        pair: RotationPair,
        look_dir: [f32; 4],
        edit_reach: f32,
        highlight_mode: EditHighlightModeArg,
        vte_sweep_status: &str,
        target_hit_voxel: Option<[i32; 4]>,
        target_hit_face: Option<[i32; 4]>,
        stream_first_node_desc: Option<&str>,
        stream_final_solid_leaf_desc: Option<&str>,
    ) -> String {
        let inv = self.current_y_inverted();
        let inv = if inv { " Y-INV" } else { "" };
        let target_text = target_hit_voxel
            .map(|hit| format!("({:+},{:+},{:+},{:+})", hit[0], hit[1], hit[2], hit[3]))
            .unwrap_or_else(|| "--".to_string());
        let face_text = target_hit_face
            .map(|face| format!("({:+},{:+},{:+},{:+})", face[0], face[1], face[2], face[3]))
            .unwrap_or_else(|| "--".to_string());
        let stream_first_node_text = stream_first_node_desc.unwrap_or("--");
        let stream_final_solid_leaf_text = stream_final_solid_leaf_desc.unwrap_or("--");
        let format_bounds = |bounds: Option<Aabb4i>| -> String {
            bounds
                .map(|b| {
                    format!(
                        "[{:+},{:+},{:+},{:+}]->[{:+},{:+},{:+},{:+}]",
                        b.min[0],
                        b.min[1],
                        b.min[2],
                        b.min[3],
                        b.max[0],
                        b.max[1],
                        b.max[2],
                        b.max[3]
                    )
                })
                .unwrap_or_else(|| "--".to_string())
        };
        let cache_tree_bounds_text = format_bounds(self.scene.debug_world_tree_root_bounds());
        let cache_interest_bounds_text = format_bounds(self.multiplayer_last_world_request_bounds);
        let (cpu_dense_chunks, cpu_leaf_headers, cpu_bvh_nodes, cpu_leaf_entries) =
            self.scene.debug_voxel_frame_buffer_lengths();
        let (gpu_dense_cap, gpu_leaf_cap, gpu_node_cap, gpu_leaf_entry_cap) = self
            .rcx
            .as_ref()
            .map(|rcx| rcx.voxel_buffer_capacities())
            .unwrap_or((0, 0, 0, 0));
        if self.control_scheme.uses_look_frame() {
            format!(
                "LOOK-FRAME [{}]  spd:{:.1}{}\n\
                 look:{:+.2} {:+.2} {:+.2} {:+.2}\n\
                 lens:xy:{:.2} zw:{:.2} trace:{} dist:{:.0}\n\
                 edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                 mat:[ ]/wheel cycle, 1-0 direct\n\
                 target:{} face:{}\n\
                 stream-first-node:{}\n\
                 stream-final-solid:{}\n\
                 cache-tree:{}\n\
                 known-interest:{}\n\
                 vte-buf:d{}/{} l{}/{} n{}/{} e{}/{}\n\
                 vte:F7 ycache:{} F8 sweep:{}\n\
                 tweak:F10 sky+emi:{} F11 log:{}",
                self.control_scheme.label(),
                self.move_speed,
                inv,
                look_dir[0],
                look_dir[1],
                look_dir[2],
                look_dir[3],
                self.focal_length_xy,
                self.focal_length_zw,
                self.vte_max_trace_steps,
                self.vte_max_trace_distance,
                self.selected_block.block_type,
                edit_reach,
                highlight_mode.label(),
                target_text,
                face_text,
                stream_first_node_text,
                stream_final_solid_leaf_text,
                cache_tree_bounds_text,
                cache_interest_bounds_text,
                cpu_dense_chunks,
                gpu_dense_cap,
                cpu_leaf_headers,
                gpu_leaf_cap,
                cpu_bvh_nodes,
                gpu_node_cap,
                cpu_leaf_entries,
                gpu_leaf_entry_cap,
                if self.vte_y_slice_lookup_cache_enabled {
                    "on"
                } else {
                    "off"
                },
                vte_sweep_status,
                if self.vte_integral_sky_emissive_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_integral_log_merge_enabled {
                    "on"
                } else {
                    "off"
                },
            )
        } else {
            format!(
                "{} [{}]  spd:{:.1}{}\n\
                 yaw:{:+.0} pit:{:+.0} xw:{:+.0} zw:{:+.0} yw:{:+.1}\n\
                 lens:xy:{:.2} zw:{:.2} trace:{} dist:{:.0}\n\
                 edit:LMB- RMB+ mat:{} reach:{:.1} hl:{}\n\
                 mat:[ ]/wheel cycle, 1-0 direct\n\
                 target:{} face:{}\n\
                 stream-first-node:{}\n\
                 stream-final-solid:{}\n\
                 cache-tree:{}\n\
                 known-interest:{}\n\
                 vte-buf:d{}/{} l{}/{} n{}/{} e{}/{}\n\
                 vte:F7 ycache:{} F8 sweep:{}\n\
                 tweak:F10 sky+emi:{} F11 log:{}",
                pair.label(),
                self.control_scheme.label(),
                self.move_speed,
                inv,
                self.camera.yaw.to_degrees(),
                self.camera.pitch.to_degrees(),
                self.camera.xw_angle.to_degrees(),
                self.camera.zw_angle.to_degrees(),
                self.camera.yw_deviation.to_degrees(),
                self.focal_length_xy,
                self.focal_length_zw,
                self.vte_max_trace_steps,
                self.vte_max_trace_distance,
                self.selected_block.block_type,
                edit_reach,
                highlight_mode.label(),
                target_text,
                face_text,
                stream_first_node_text,
                stream_final_solid_leaf_text,
                cache_tree_bounds_text,
                cache_interest_bounds_text,
                cpu_dense_chunks,
                gpu_dense_cap,
                cpu_leaf_headers,
                gpu_leaf_cap,
                cpu_bvh_nodes,
                gpu_node_cap,
                cpu_leaf_entries,
                gpu_leaf_entry_cap,
                if self.vte_y_slice_lookup_cache_enabled {
                    "on"
                } else {
                    "off"
                },
                vte_sweep_status,
                if self.vte_integral_sky_emissive_enabled {
                    "on"
                } else {
                    "off"
                },
                if self.vte_integral_log_merge_enabled {
                    "on"
                } else {
                    "off"
                },
            )
        }
    }
}
