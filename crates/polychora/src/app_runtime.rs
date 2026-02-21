use super::*;

impl App {
    pub(super) fn active_rotation_pair(&self) -> RotationPair {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => {
                if self.input.mouse_back_held() {
                    RotationPair::FourD
                } else {
                    RotationPair::Standard
                }
            }
            ControlScheme::LookTransport | ControlScheme::RotorFree => RotationPair::Standard,
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                if self.input.mouse_back_held() && self.input.mouse_forward_held() {
                    RotationPair::DoubleRotation
                } else if self.input.mouse_back_held() {
                    RotationPair::FourD
                } else if self.control_scheme.uses_scroll_pair_cycle() {
                    self.scroll_cycle_pair
                } else {
                    RotationPair::Standard
                }
            }
        }
    }

    pub(super) fn current_look_direction(&self) -> [f32; 4] {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.look_direction_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.look_direction_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.look_direction()
            }
        }
    }

    pub(super) fn current_view_matrix(&self) -> ndarray::Array2<f32> {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.view_matrix_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.view_matrix_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.view_matrix()
            }
        }
    }

    pub(super) fn current_view_basis(&self) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4]) {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.view_basis_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.view_basis_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.view_basis()
            }
        }
    }

    pub(super) fn play_spatial_sound(
        &self,
        effect: SoundEffect,
        emitter_position: [f32; 4],
        scale: f32,
    ) {
        self.audio.play_spatial_scaled(
            effect,
            scale,
            self.camera.position,
            self.current_view_basis(),
            emitter_position,
        );
    }

    pub(super) fn play_spatial_sound_voxel(
        &self,
        effect: SoundEffect,
        voxel_position: [i32; 4],
        scale: f32,
    ) {
        self.audio.play_spatial_voxel_scaled(
            effect,
            scale,
            self.camera.position,
            self.current_view_basis(),
            voxel_position,
        );
    }

    pub(super) fn current_y_inverted(&self) -> bool {
        match self.control_scheme {
            ControlScheme::IntuitiveUpright => self.camera.is_y_inverted_upright(),
            ControlScheme::LookTransport | ControlScheme::RotorFree => {
                self.camera.is_y_inverted_look_frame()
            }
            ControlScheme::LegacySideButtonLayers | ControlScheme::LegacyScrollCycle => {
                self.camera.is_y_inverted()
            }
        }
    }

    pub(super) fn set_vte_runtime_flags(&mut self, y_slice_lookup_cache: bool) {
        self.vte_y_slice_lookup_cache_enabled = y_slice_lookup_cache;
    }

    pub(super) fn toggle_vte_runtime_sweep(&mut self) {
        if self.args.backend.to_render_backend() != RenderBackend::VoxelTraversal {
            eprintln!(
                "VTE runtime sweep requires --backend voxel-traversal (current: {:?}).",
                self.args.backend.to_render_backend()
            );
            return;
        }

        if let Some(state) = self.vte_sweep_state.take() {
            self.set_vte_runtime_flags(state.previous_y_slice_lookup_cache);
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.reset_gpu_profile_window();
            }
            eprintln!(
                "[VTE sweep #{}] cancelled; restored runtime flags (y_slice_lookup_cache={}).",
                state.run_id, self.vte_y_slice_lookup_cache_enabled
            );
            return;
        }

        self.vte_sweep_run_id = self.vte_sweep_run_id.wrapping_add(1);
        if self.vte_sweep_run_id == 0 {
            self.vte_sweep_run_id = 1;
        }
        let run_id = self.vte_sweep_run_id;
        let previous_y_slice_lookup_cache = self.vte_y_slice_lookup_cache_enabled;
        let profiles = self.vte_sweep_profiles();
        let profile_count = profiles.len();
        let first_profile = profiles[0];
        self.set_vte_runtime_flags(first_profile.y_slice_lookup_cache);
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        self.vte_sweep_state = Some(VteSweepState {
            run_id,
            profile_index: 0,
            frames_remaining: VTE_SWEEP_SAMPLE_FRAMES,
            previous_y_slice_lookup_cache,
        });
        eprintln!(
            "[VTE sweep #{}] started on live scene (mode={}): {} profiles × {} frames/profile.",
            run_id,
            self.vte_sweep_mode_label(),
            profile_count,
            VTE_SWEEP_SAMPLE_FRAMES
        );
        eprintln!(
            "[VTE sweep #{}] profile 1/{} '{}' (y_slice_lookup_cache={}).",
            run_id, profile_count, first_profile.label, first_profile.y_slice_lookup_cache
        );
    }

    pub(super) fn advance_vte_runtime_sweep_after_frame(&mut self) {
        let Some(mut state) = self.vte_sweep_state else {
            return;
        };
        if state.frames_remaining > 0 {
            state.frames_remaining -= 1;
        }
        if state.frames_remaining > 0 {
            self.vte_sweep_state = Some(state);
            return;
        }

        let profiles = self.vte_sweep_profiles();
        let profile_count = profiles.len();
        let finished_profile = profiles[state.profile_index];
        eprintln!(
            "[VTE sweep #{}] finished profile {}/{} '{}'.",
            state.run_id,
            state.profile_index + 1,
            profile_count,
            finished_profile.label
        );

        if state.profile_index + 1 < profile_count {
            state.profile_index += 1;
            state.frames_remaining = VTE_SWEEP_SAMPLE_FRAMES;
            let next_profile = profiles[state.profile_index];
            self.set_vte_runtime_flags(next_profile.y_slice_lookup_cache);
            if let Some(rcx) = self.rcx.as_mut() {
                rcx.reset_gpu_profile_window();
            }
            eprintln!(
                "[VTE sweep #{}] profile {}/{} '{}' (y_slice_lookup_cache={}).",
                state.run_id,
                state.profile_index + 1,
                profile_count,
                next_profile.label,
                next_profile.y_slice_lookup_cache
            );
            self.vte_sweep_state = Some(state);
            return;
        }

        self.set_vte_runtime_flags(state.previous_y_slice_lookup_cache);
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }
        eprintln!(
            "[VTE sweep #{}] completed; restored runtime flags (y_slice_lookup_cache={}).",
            state.run_id, self.vte_y_slice_lookup_cache_enabled
        );
        self.vte_sweep_state = None;
    }
}
