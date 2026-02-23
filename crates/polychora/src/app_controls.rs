use super::*;

impl App {
    pub(super) fn grab_mouse(&mut self, window: &Window) {
        let result = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if result.is_ok() {
            window.set_cursor_visible(false);
            self.mouse_grabbed = true;
            self.input.clear_mouse_delta();
        }
    }

    pub(super) fn release_mouse(&mut self, window: &Window) {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.mouse_grabbed = false;
        self.input.clear_mouse_delta();
    }

    pub(super) fn cycle_hotbar_material_prev(&mut self) {
        use polychora::shared::protocol::ItemStack;
        let current = block_material_from_slot(&self.hotbar_slots[self.hotbar_selected_index]);
        let next = if current <= BLOCK_EDIT_PLACE_MATERIAL_MIN {
            BLOCK_EDIT_PLACE_MATERIAL_MAX
        } else {
            current.saturating_sub(1)
        };
        self.hotbar_slots[self.hotbar_selected_index] =
            Some(ItemStack::block(0, next as u32, 1));
        self.selected_block = block_data_from_slot(&self.hotbar_slots[self.hotbar_selected_index]);
        eprintln!(
            "Hotbar slot {} material: {} ({})",
            self.hotbar_selected_index + 1,
            self.selected_block.block_type,
            self.content_registry.block_name(self.selected_block.namespace, self.selected_block.block_type),
        );
    }

    pub(super) fn cycle_hotbar_material_next(&mut self) {
        use polychora::shared::protocol::ItemStack;
        let current = block_material_from_slot(&self.hotbar_slots[self.hotbar_selected_index]);
        let next = if current >= BLOCK_EDIT_PLACE_MATERIAL_MAX {
            BLOCK_EDIT_PLACE_MATERIAL_MIN
        } else {
            current.saturating_add(1)
        };
        self.hotbar_slots[self.hotbar_selected_index] =
            Some(ItemStack::block(0, next as u32, 1));
        self.selected_block = block_data_from_slot(&self.hotbar_slots[self.hotbar_selected_index]);
        eprintln!(
            "Hotbar slot {} material: {} ({})",
            self.hotbar_selected_index + 1,
            self.selected_block.block_type,
            self.content_registry.block_name(self.selected_block.namespace, self.selected_block.block_type),
        );
    }

    pub(super) fn toggle_inventory(&mut self) {
        self.inventory_open = !self.inventory_open;
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        if let Some(window) = window {
            if self.inventory_open {
                self.release_mouse(&window);
            } else {
                self.grab_mouse(&window);
            }
        }
    }

    pub(super) fn toggle_teleport_dialog(&mut self) {
        self.teleport_dialog_open = !self.teleport_dialog_open;
        let window = self.rcx.as_ref().and_then(|rcx| rcx.window.clone());
        if let Some(window) = window {
            if self.teleport_dialog_open {
                // Pre-fill with current position
                let pos = self.camera.position;
                self.teleport_coords = [
                    format!("{:.1}", pos[0]),
                    format!("{:.1}", pos[1]),
                    format!("{:.1}", pos[2]),
                    format!("{:.1}", pos[3]),
                ];
                self.release_mouse(&window);
            } else {
                self.grab_mouse(&window);
            }
        }
    }

    pub(super) fn cycle_control_scheme(&mut self) {
        let previous_scheme = self.control_scheme;
        self.control_scheme = self.control_scheme.next();
        self.scroll_cycle_pair = RotationPair::Standard;
        if self.control_scheme.is_upright_primary() {
            self.camera.enforce_upright_constraints();
        } else if self.control_scheme.uses_look_frame()
            && (!previous_scheme.uses_look_frame() || !self.camera.look_frame_initialized())
        {
            if previous_scheme.is_upright_primary() {
                self.camera.sync_look_frame_from_upright_rotation();
            } else {
                self.camera.sync_look_frame_from_standard_rotation();
            }
        }
    }

    pub(super) fn set_control_scheme(&mut self, target: ControlScheme) {
        while self.control_scheme != target {
            self.cycle_control_scheme();
        }
    }

    pub(super) fn toggle_vte_y_slice_lookup_cache(&mut self) {
        if let Some(state) = self.vte_sweep_state {
            eprintln!(
                "[VTE sweep #{}] ignoring manual y-slice lookup cache toggle while sweep is active.",
                state.run_id
            );
            return;
        }
        self.vte_y_slice_lookup_cache_enabled = !self.vte_y_slice_lookup_cache_enabled;
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "VTE runtime y-slice lookup cache: {}",
            if self.vte_y_slice_lookup_cache_enabled {
                "on"
            } else {
                "off"
            }
        );
    }

    pub(super) fn toggle_vte_integral_sky_emissive(&mut self) {
        self.vte_integral_sky_emissive_enabled = !self.vte_integral_sky_emissive_enabled;
        eprintln!(
            "VTE fused integral sky/emissive tweak: {} (sky_scale={:.3}, hit_emissive_boost={:.3})",
            if self.vte_integral_sky_emissive_enabled {
                "on"
            } else {
                "off"
            },
            self.vte_integral_sky_scale,
            self.vte_integral_hit_emissive_boost,
        );
    }

    pub(super) fn toggle_vte_integral_log_merge(&mut self) {
        self.vte_integral_log_merge_enabled = !self.vte_integral_log_merge_enabled;
        eprintln!(
            "VTE fused integral log merge tweak: {} (k={:.3})",
            if self.vte_integral_log_merge_enabled {
                "on"
            } else {
                "off"
            },
            self.vte_integral_log_merge_k,
        );
    }

    pub(super) fn drain_gameplay_inputs_while_menu_open(&mut self) {
        self.input.take_menu_left();
        self.input.take_menu_right();
        self.input.take_menu_up();
        self.input.take_menu_down();
        self.input.take_menu_activate();
        // Do NOT drain take_look_at() - G key should work during gameplay
        self.input.take_scheme_cycle();
        self.input.take_vte_sweep();
        self.input.take_vte_y_slice_lookup_cache_toggle();
        self.input.take_vte_integral_sky_emissive_toggle();
        self.input.take_vte_integral_log_merge_toggle();
        self.input.take_scroll_steps();
        self.input.take_fly_toggle();
        self.input.take_sprint_toggle();
        self.input.take_jump();
        self.input.take_place_material_prev();
        self.input.take_place_material_next();
        self.input.take_place_material_digit();
        self.input.take_remove_block();
        self.input.take_place_block();
        self.input.take_pick_material();
        self.input.take_inventory_toggle();
        self.input.take_teleport_dialog();
        self.input.take_mouse_delta();
    }

    pub(super) fn inject_key_press(&mut self, keycode: KeyCode) {
        match keycode {
            KeyCode::Escape => {
                if self.teleport_dialog_open {
                    self.teleport_dialog_open = false;
                } else if self.inventory_open {
                    self.inventory_open = false;
                } else {
                    self.menu_open = !self.menu_open;
                }
            }
            KeyCode::KeyI => {
                self.inventory_open = !self.inventory_open;
            }
            KeyCode::Digit1 => self.hotbar_selected_index = 0,
            KeyCode::Digit2 => self.hotbar_selected_index = 1,
            KeyCode::Digit3 => self.hotbar_selected_index = 2,
            KeyCode::Digit4 => self.hotbar_selected_index = 3,
            KeyCode::Digit5 => self.hotbar_selected_index = 4,
            KeyCode::Digit6 => self.hotbar_selected_index = 5,
            KeyCode::Digit7 => self.hotbar_selected_index = 6,
            KeyCode::Digit8 => self.hotbar_selected_index = 7,
            KeyCode::Digit9 => self.hotbar_selected_index = 8,
            KeyCode::F12 => {
                // Screenshot will be handled by setting the flag
            }
            KeyCode::ArrowUp => {
                if self.menu_open && self.menu_selection > 0 {
                    self.menu_selection -= 1;
                }
            }
            KeyCode::ArrowDown => {
                if self.menu_open && self.menu_selection < 2 {
                    self.menu_selection += 1;
                }
            }
            KeyCode::Enter => {
                if self.menu_open {
                    // Handle menu activation
                }
            }
            KeyCode::KeyG => {
                self.input.request_look_at();
            }
            _ => {}
        }
    }
}
