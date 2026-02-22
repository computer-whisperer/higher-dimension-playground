use super::*;

impl App {
    pub(super) fn vte_sweep_profiles(&self) -> &'static [VteRuntimeProfile] {
        &VTE_SWEEP_PROFILES
    }

    pub(super) fn vte_sweep_mode_label(&self) -> &'static str {
        VTE_SWEEP_MODE_LABEL
    }

    pub(super) fn perf_suite_active(&self) -> bool {
        self.perf_suite_state.is_some()
    }

    pub(super) fn perf_suite_scenario(&self, scenario_index: usize) -> PerfSuiteScenario {
        PERF_SUITE_SCENARIOS[scenario_index % PERF_SUITE_SCENARIOS.len()]
    }

    pub(super) fn set_perf_suite_camera_pose(&mut self, scenario_index: usize) {
        let scenario = self.perf_suite_scenario(scenario_index);
        self.camera.position = scenario.position;
        self.camera.yaw = scenario.yaw;
        self.camera.pitch = scenario.pitch;
        self.camera.xw_angle = scenario.xw_angle;
        self.camera.zw_angle = scenario.zw_angle;
        self.camera.yw_deviation = scenario.yw_deviation;
        self.camera.is_flying = true;
        self.camera.is_grounded = false;
        self.camera.velocity_y = 0.0;
        self.look_at_target = None;
        self.menu_open = false;
        self.inventory_open = false;
        self.teleport_dialog_open = false;
        self.controls_dialog_open = false;
        self.sprint_enabled = false;

        // Apply per-scenario render config overrides (or restore defaults).
        self.vte_max_trace_steps = scenario
            .vte_max_trace_steps
            .unwrap_or(self.perf_suite_default_trace_steps);
        self.vte_max_trace_distance = scenario
            .vte_max_trace_distance
            .unwrap_or(self.perf_suite_default_trace_distance);
    }

    pub(super) fn reset_runtime_profile_window(&mut self) {
        self.profile_window_start = Instant::now();
        self.profile_frame_samples = 0;
        self.profile_client_cpu_ms_sum = 0.0;
        self.profile_client_cpu_ms_max = 0.0;
        self.profile_gpu_ms_sum = 0.0;
        self.profile_gpu_ms_max = 0.0;
        self.profile_gpu_samples = 0;
        self.profile_cpu_phase_window = RuntimeCpuPhaseWindow::default();
    }

    pub(super) fn begin_runtime_profile_frame(&mut self) {
        self.profile_cpu_phase_current = RuntimeCpuPhaseMetrics::default();
    }

    pub(super) fn set_runtime_profile_update_ms(&mut self, elapsed_ms: f64) {
        self.profile_cpu_phase_current.update_ms = elapsed_ms.max(0.0);
    }

    pub(super) fn set_runtime_profile_voxel_build_ms(&mut self, elapsed_ms: f64) {
        self.profile_cpu_phase_current.voxel_build_ms = elapsed_ms.max(0.0);
    }

    pub(super) fn set_runtime_profile_render_submit_ms(&mut self, elapsed_ms: f64) {
        self.profile_cpu_phase_current.render_submit_ms = elapsed_ms.max(0.0);
    }

    pub(super) fn set_runtime_profile_post_render_ms(&mut self, elapsed_ms: f64) {
        self.profile_cpu_phase_current.post_render_ms = elapsed_ms.max(0.0);
    }

    pub(super) fn note_runtime_profile_multiplayer_patch(&mut self, elapsed_ms: f64) {
        self.profile_cpu_phase_current.multiplayer_patch_ms += elapsed_ms.max(0.0);
        self.profile_cpu_phase_current.multiplayer_patch_count = self
            .profile_cpu_phase_current
            .multiplayer_patch_count
            .saturating_add(1);
    }

    pub(super) fn begin_perf_suite_phase(&mut self, announce: bool) {
        let (scenario_index, phase, frames_remaining, warmup_frames, sample_frames) = {
            let Some(state) = self.perf_suite_state.as_ref() else {
                return;
            };
            (
                state.scenario_index,
                state.phase,
                state.frames_remaining,
                state.warmup_frames,
                state.sample_frames,
            )
        };
        self.set_perf_suite_camera_pose(scenario_index);
        self.drain_gameplay_inputs_while_menu_open();
        self.input.take_escape();
        self.input.take_screenshot();
        self.reset_runtime_profile_window();
        if let Some(rcx) = self.rcx.as_mut() {
            rcx.reset_gpu_profile_window();
        }

        // Auto-spawn entities on first scenario's warmup phase (once).
        if scenario_index == 0 && phase == PerfSuitePhase::Warmup && !self.perf_suite_entities_spawned {
            self.perf_suite_auto_spawn_entities();
        }

        if announce {
            let scenario = self.perf_suite_scenario(scenario_index);
            let scenario_num = scenario_index + 1;
            let scenario_total = PERF_SUITE_SCENARIOS.len();
            let phase_label = match phase {
                PerfSuitePhase::WorldSettle => "world-settle",
                PerfSuitePhase::Warmup => "warmup",
                PerfSuitePhase::Sample => "sample",
            };
            eprintln!(
                "[perf-suite] scenario {}/{} '{}': {} {} frames (warmup={} sample={}) trace_steps={} trace_dist={:.1}",
                scenario_num,
                scenario_total,
                scenario.label,
                phase_label,
                frames_remaining,
                warmup_frames,
                sample_frames,
                self.vte_max_trace_steps,
                self.vte_max_trace_distance,
            );
        }
    }

    fn perf_suite_auto_spawn_entities(&mut self) {
        let count = self.args.perf_suite_spawn_entities;
        if count == 0 {
            return;
        }
        self.perf_suite_entities_spawned = true;

        let [cx, cy, cz, cw] = self.camera.position;
        // Place entities in a grid near camera position.
        let grid_size = (count as f32).sqrt().ceil() as u32;
        let spacing = 4.0_f32;
        let mut spawned = 0u32;
        for ix in 0..grid_size {
            for iz in 0..grid_size {
                if spawned >= count {
                    break;
                }
                let x = cx + (ix as f32 - grid_size as f32 / 2.0) * spacing;
                let z = cz + (iz as f32 - grid_size as f32 / 2.0) * spacing;
                let cmd = format!("spawn cube {:.1} {:.1} {:.1} {:.1}", x, cy + 2.0, z, cw);
                if !self.send_multiplayer_console_command(&cmd) {
                    eprintln!(
                        "[perf-suite] entity spawn failed (no active server); spawned {}/{}",
                        spawned, count
                    );
                    return;
                }
                spawned += 1;
            }
        }
        eprintln!("[perf-suite] spawned {} entities for BVH testing", spawned);
    }

    pub(super) fn perf_suite_report_path(&self) -> PathBuf {
        if let Some(path) = self.args.perf_suite_report.clone() {
            return path;
        }
        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        PathBuf::from(format!(
            "{}/perf-suite-{}.json",
            PERF_SUITE_REPORT_DIR_DEFAULT, unix_secs
        ))
    }

    pub(super) fn write_perf_suite_report(&self, state: &PerfSuiteState, elapsed_s: f32) {
        let mut scenarios = Vec::with_capacity(state.results.len());
        for result in &state.results {
            let scenario = self.perf_suite_scenario(result.scenario_index);
            let render_gpu = if let (Some(avg_ms), Some(max_ms)) =
                (result.render_gpu_avg_ms, result.render_gpu_max_ms)
            {
                serde_json::json!({
                    "avg_ms": avg_ms,
                    "max_ms": max_ms,
                    "samples": result.render_gpu_samples,
                })
            } else {
                serde_json::json!(null)
            };
            let gpu_phases: Vec<serde_json::Value> = result
                .render_gpu_phases
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "name": p.name,
                        "avg_ms": p.avg_ms,
                        "max_ms": p.max_ms,
                        "samples": p.samples,
                    })
                })
                .collect();
            scenarios.push(serde_json::json!({
                "index": result.scenario_index,
                "label": result.scenario_label,
                "pose": {
                    "position": scenario.position,
                    "yaw": scenario.yaw,
                    "pitch": scenario.pitch,
                    "xw_angle": scenario.xw_angle,
                    "zw_angle": scenario.zw_angle,
                    "yw_deviation": scenario.yw_deviation,
                },
                "render_config": {
                    "vte_max_trace_steps": result.vte_max_trace_steps,
                    "vte_max_trace_distance": result.vte_max_trace_distance,
                },
                "client_cpu": {
                    "avg_ms": result.client_cpu_avg_ms,
                    "max_ms": result.client_cpu_max_ms,
                    "frames": result.client_cpu_frames,
                },
                "render_gpu": render_gpu,
                "render_gpu.phases": gpu_phases,
            }));
        }

        let singleplayer_world_type = format!("{:?}", self.args.singleplayer_world_type);
        let report = serde_json::json!({
            "schema": "polychora.perf_suite.v2",
            "generated_unix_seconds": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "elapsed_seconds": elapsed_s,
            "world_file": self.world_file.to_string_lossy(),
            "render_backend": format!("{:?}", self.args.backend),
            "vte_display_mode": format!("{:?}", self.args.vte_display_mode),
            "vte_max_trace_steps": self.perf_suite_default_trace_steps,
            "vte_max_trace_distance": self.perf_suite_default_trace_distance,
            "singleplayer_world_type": singleplayer_world_type,
            "singleplayer_world_seed": self.args.singleplayer_world_seed,
            "warmup_frames": state.warmup_frames,
            "sample_frames": state.sample_frames,
            "scenario_count": scenarios.len(),
            "scenarios": scenarios,
        });

        let report_path = self.perf_suite_report_path();
        if let Some(parent) = report_path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(error) = std::fs::create_dir_all(parent) {
                    eprintln!(
                        "[perf-suite] failed to create report directory {}: {}",
                        parent.display(),
                        error
                    );
                    return;
                }
            }
        }
        let bytes = match serde_json::to_vec_pretty(&report) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("[perf-suite] failed to serialize JSON report: {error}");
                return;
            }
        };
        if let Err(error) = std::fs::write(&report_path, bytes) {
            eprintln!(
                "[perf-suite] failed to write report {}: {}",
                report_path.display(),
                error
            );
            return;
        }
        eprintln!("[perf-suite] wrote report {}", report_path.display());
        println!("perf-suite-report path={}", report_path.display());
    }

    pub(super) fn advance_perf_suite_after_frame(&mut self, frame_start: Instant) {
        let Some(mut state) = self.perf_suite_state.take() else {
            return;
        };

        if !self.world_ready {
            self.perf_suite_state = Some(state);
            return;
        }

        // WorldSettle phase: wait for chunk streaming to reach steady state.
        if state.phase == PerfSuitePhase::WorldSettle {
            state.settle_total_frames += 1;
            let patches_this_frame = self.profile_cpu_phase_current.multiplayer_patch_count;
            if patches_this_frame == 0 {
                state.settle_stable_count += 1;
            } else {
                state.settle_stable_count = 0;
            }
            // Log progress every 60 frames.
            if state.settle_total_frames % 60 == 0 {
                eprintln!(
                    "[perf-suite] world-settle: frame {} stable={}/{} (patches_this_frame={})",
                    state.settle_total_frames,
                    state.settle_stable_count,
                    PERF_SUITE_SETTLE_STABLE_FRAMES,
                    patches_this_frame,
                );
            }
            let settled = state.settle_stable_count >= PERF_SUITE_SETTLE_STABLE_FRAMES
                || state.settle_total_frames >= PERF_SUITE_SETTLE_MAX_FRAMES;
            if !settled {
                self.perf_suite_state = Some(state);
                return;
            }
            let reason = if state.settle_stable_count >= PERF_SUITE_SETTLE_STABLE_FRAMES {
                "stable"
            } else {
                "timeout"
            };
            eprintln!(
                "[perf-suite] world settled ({}) after {} frames ({} stable)",
                reason, state.settle_total_frames, state.settle_stable_count
            );
            // Transition to first scenario warmup.
            state.phase = PerfSuitePhase::Warmup;
            state.frames_remaining = state.warmup_frames;
            self.perf_suite_state = Some(state);
            self.begin_perf_suite_phase(true);
            return;
        }

        let client_cpu_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        if state.phase == PerfSuitePhase::Sample {
            state.sample_frame_samples = state.sample_frame_samples.saturating_add(1);
            state.sample_client_cpu_ms_sum += client_cpu_ms;
            state.sample_client_cpu_ms_max = state.sample_client_cpu_ms_max.max(client_cpu_ms);

            if let Some(rcx) = self.rcx.as_ref() {
                let gpu_ms = rcx.last_gpu_frame_ms().max(0.0) as f64;
                state.sample_gpu_samples = state.sample_gpu_samples.saturating_add(1);
                state.sample_gpu_ms_sum += gpu_ms;
                state.sample_gpu_ms_max = state.sample_gpu_ms_max.max(gpu_ms);

                // Accumulate per-phase GPU breakdown.
                for &(name, phase_ms) in rcx.last_gpu_phase_breakdown() {
                    let phase_ms = phase_ms.max(0.0) as f64;
                    if let Some(accum) = state.sample_gpu_phases.iter_mut().find(|a| a.name == name)
                    {
                        accum.sum_ms += phase_ms;
                        accum.max_ms = accum.max_ms.max(phase_ms);
                        accum.samples += 1;
                    } else {
                        state.sample_gpu_phases.push(GpuPhaseAccum {
                            name,
                            sum_ms: phase_ms,
                            max_ms: phase_ms,
                            samples: 1,
                        });
                    }
                }
            }
        }

        if state.frames_remaining > 0 {
            state.frames_remaining -= 1;
        }
        if state.frames_remaining > 0 {
            self.perf_suite_state = Some(state);
            return;
        }

        let scenario = self.perf_suite_scenario(state.scenario_index);
        let scenario_num = state.scenario_index + 1;
        let scenario_total = PERF_SUITE_SCENARIOS.len();

        match state.phase {
            PerfSuitePhase::WorldSettle => unreachable!("handled above"),
            PerfSuitePhase::Warmup => {
                state.phase = PerfSuitePhase::Sample;
                state.frames_remaining = state.sample_frames;
                state.reset_sample_accumulators();
                self.perf_suite_state = Some(state);
                self.begin_perf_suite_phase(true);
            }
            PerfSuitePhase::Sample => {
                if state.sample_frame_samples > 0 {
                    let frame_samples = state.sample_frame_samples as f64;
                    let client_avg_ms = state.sample_client_cpu_ms_sum / frame_samples;
                    let client_max_ms = state.sample_client_cpu_ms_max;
                    let (render_gpu_avg_ms, render_gpu_max_ms) = if state.sample_gpu_samples > 0 {
                        (
                            Some(state.sample_gpu_ms_sum / state.sample_gpu_samples as f64),
                            Some(state.sample_gpu_ms_max),
                        )
                    } else {
                        (None, None)
                    };
                    if state.sample_gpu_samples > 0 {
                        let gpu_avg_ms = render_gpu_avg_ms.unwrap_or(0.0);
                        let gpu_max_ms = render_gpu_max_ms.unwrap_or(0.0);
                        eprintln!(
                            "[perf-suite] result {}/{} '{}': client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu avg={:.3}ms max={:.3}ms samples={}",
                            scenario_num,
                            scenario_total,
                            scenario.label,
                            client_avg_ms,
                            client_max_ms,
                            state.sample_frame_samples,
                            gpu_avg_ms,
                            gpu_max_ms,
                            state.sample_gpu_samples
                        );
                    } else {
                        eprintln!(
                            "[perf-suite] result {}/{} '{}': client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu unavailable",
                            scenario_num,
                            scenario_total,
                            scenario.label,
                            client_avg_ms,
                            client_max_ms,
                            state.sample_frame_samples
                        );
                    }
                    let render_gpu_phases: Vec<GpuPhaseResult> = state
                        .sample_gpu_phases
                        .iter()
                        .map(|a| GpuPhaseResult {
                            name: a.name,
                            avg_ms: if a.samples > 0 {
                                a.sum_ms / a.samples as f64
                            } else {
                                0.0
                            },
                            max_ms: a.max_ms,
                            samples: a.samples,
                        })
                        .collect();
                    state.results.push(PerfSuiteScenarioResult {
                        scenario_index: state.scenario_index,
                        scenario_label: scenario.label,
                        client_cpu_avg_ms: client_avg_ms,
                        client_cpu_max_ms: client_max_ms,
                        client_cpu_frames: state.sample_frame_samples,
                        render_gpu_avg_ms,
                        render_gpu_max_ms,
                        render_gpu_samples: state.sample_gpu_samples,
                        render_gpu_phases,
                        vte_max_trace_steps: self.vte_max_trace_steps,
                        vte_max_trace_distance: self.vte_max_trace_distance,
                    });
                }

                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.flush_gpu_profile_report_now();
                }

                if state.scenario_index + 1 < scenario_total {
                    state.scenario_index += 1;
                    state.phase = PerfSuitePhase::Warmup;
                    state.frames_remaining = state.warmup_frames;
                    state.reset_sample_accumulators();
                    self.perf_suite_state = Some(state);
                    self.begin_perf_suite_phase(true);
                } else {
                    let elapsed_s = state.run_started_at.elapsed().as_secs_f32();
                    self.write_perf_suite_report(&state, elapsed_s);
                    eprintln!(
                        "[perf-suite] completed {} scenarios in {:.2}s.",
                        scenario_total, elapsed_s
                    );
                    self.perf_suite_state = None;
                    if self.args.perf_suite_exit_on_complete {
                        self.should_exit_after_render = true;
                    }
                }
            }
        }
    }

    pub(super) fn record_runtime_profile_sample(&mut self, frame_start: Instant) {
        let client_cpu_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.profile_frame_samples = self.profile_frame_samples.saturating_add(1);
        self.profile_client_cpu_ms_sum += client_cpu_ms;
        self.profile_client_cpu_ms_max = self.profile_client_cpu_ms_max.max(client_cpu_ms);
        self.profile_cpu_phase_window.update_sum_ms += self.profile_cpu_phase_current.update_ms;
        self.profile_cpu_phase_window.update_max_ms = self
            .profile_cpu_phase_window
            .update_max_ms
            .max(self.profile_cpu_phase_current.update_ms);
        self.profile_cpu_phase_window.voxel_build_sum_ms +=
            self.profile_cpu_phase_current.voxel_build_ms;
        self.profile_cpu_phase_window.voxel_build_max_ms = self
            .profile_cpu_phase_window
            .voxel_build_max_ms
            .max(self.profile_cpu_phase_current.voxel_build_ms);
        self.profile_cpu_phase_window.render_submit_sum_ms +=
            self.profile_cpu_phase_current.render_submit_ms;
        self.profile_cpu_phase_window.render_submit_max_ms = self
            .profile_cpu_phase_window
            .render_submit_max_ms
            .max(self.profile_cpu_phase_current.render_submit_ms);
        self.profile_cpu_phase_window.post_render_sum_ms +=
            self.profile_cpu_phase_current.post_render_ms;
        self.profile_cpu_phase_window.post_render_max_ms = self
            .profile_cpu_phase_window
            .post_render_max_ms
            .max(self.profile_cpu_phase_current.post_render_ms);
        self.profile_cpu_phase_window.multiplayer_patch_sum_ms +=
            self.profile_cpu_phase_current.multiplayer_patch_ms;
        self.profile_cpu_phase_window.multiplayer_patch_max_ms = self
            .profile_cpu_phase_window
            .multiplayer_patch_max_ms
            .max(self.profile_cpu_phase_current.multiplayer_patch_ms);
        self.profile_cpu_phase_window.multiplayer_patch_count_sum = self
            .profile_cpu_phase_window
            .multiplayer_patch_count_sum
            .saturating_add(self.profile_cpu_phase_current.multiplayer_patch_count as u64);

        if let Some(rcx) = self.rcx.as_ref() {
            let gpu_ms = rcx.last_gpu_frame_ms().max(0.0) as f64;
            self.profile_gpu_samples = self.profile_gpu_samples.saturating_add(1);
            self.profile_gpu_ms_sum += gpu_ms;
            self.profile_gpu_ms_max = self.profile_gpu_ms_max.max(gpu_ms);
        }

        if self.profile_window_start.elapsed() < CLIENT_PROFILE_REPORT_INTERVAL {
            return;
        }

        if self.profile_frame_samples > 0 {
            let frame_samples = self.profile_frame_samples as f64;
            let client_avg_ms = self.profile_client_cpu_ms_sum / frame_samples;
            let client_max_ms = self.profile_client_cpu_ms_max;
            let update_avg_ms = self.profile_cpu_phase_window.update_sum_ms / frame_samples;
            let voxel_build_avg_ms =
                self.profile_cpu_phase_window.voxel_build_sum_ms / frame_samples;
            let render_submit_avg_ms =
                self.profile_cpu_phase_window.render_submit_sum_ms / frame_samples;
            let post_render_avg_ms =
                self.profile_cpu_phase_window.post_render_sum_ms / frame_samples;
            let multiplayer_patch_avg_ms =
                self.profile_cpu_phase_window.multiplayer_patch_sum_ms / frame_samples;
            let multiplayer_patch_avg_count =
                self.profile_cpu_phase_window.multiplayer_patch_count_sum as f64 / frame_samples;
            let instrumented_avg_ms =
                update_avg_ms + voxel_build_avg_ms + render_submit_avg_ms + post_render_avg_ms;
            let unattributed_avg_ms = (client_avg_ms - instrumented_avg_ms).max(0.0);
            let (gpu_avg_ms, gpu_max_ms, gpu_samples) = if self.profile_gpu_samples > 0 {
                (
                    self.profile_gpu_ms_sum / self.profile_gpu_samples as f64,
                    self.profile_gpu_ms_max,
                    self.profile_gpu_samples,
                )
            } else {
                (0.0, 0.0, 0)
            };
            if gpu_samples > 0 {
                eprintln!(
                    "profile client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu avg={:.3}ms max={:.3}ms samples={} | cpu-phases update avg={:.3}ms max={:.3}ms voxel-build avg={:.3}ms max={:.3}ms render-submit avg={:.3}ms max={:.3}ms post-render avg={:.3}ms max={:.3}ms mp-patch avg={:.3}ms max={:.3}ms patches/frame={:.3} unattributed avg={:.3}ms",
                    client_avg_ms,
                    client_max_ms,
                    self.profile_frame_samples,
                    gpu_avg_ms,
                    gpu_max_ms,
                    gpu_samples,
                    update_avg_ms,
                    self.profile_cpu_phase_window.update_max_ms,
                    voxel_build_avg_ms,
                    self.profile_cpu_phase_window.voxel_build_max_ms,
                    render_submit_avg_ms,
                    self.profile_cpu_phase_window.render_submit_max_ms,
                    post_render_avg_ms,
                    self.profile_cpu_phase_window.post_render_max_ms,
                    multiplayer_patch_avg_ms,
                    self.profile_cpu_phase_window.multiplayer_patch_max_ms,
                    multiplayer_patch_avg_count,
                    unattributed_avg_ms,
                );
            } else {
                eprintln!(
                    "profile client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu unavailable | cpu-phases update avg={:.3}ms max={:.3}ms voxel-build avg={:.3}ms max={:.3}ms render-submit avg={:.3}ms max={:.3}ms post-render avg={:.3}ms max={:.3}ms mp-patch avg={:.3}ms max={:.3}ms patches/frame={:.3} unattributed avg={:.3}ms",
                    client_avg_ms,
                    client_max_ms,
                    self.profile_frame_samples,
                    update_avg_ms,
                    self.profile_cpu_phase_window.update_max_ms,
                    voxel_build_avg_ms,
                    self.profile_cpu_phase_window.voxel_build_max_ms,
                    render_submit_avg_ms,
                    self.profile_cpu_phase_window.render_submit_max_ms,
                    post_render_avg_ms,
                    self.profile_cpu_phase_window.post_render_max_ms,
                    multiplayer_patch_avg_ms,
                    self.profile_cpu_phase_window.multiplayer_patch_max_ms,
                    multiplayer_patch_avg_count,
                    unattributed_avg_ms,
                );
            }
        }

        self.reset_runtime_profile_window();
    }
}
