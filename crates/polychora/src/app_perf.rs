use super::*;

impl App {
    pub(super) fn vte_sweep_profiles(&self) -> &'static [VteRuntimeProfile] {
        if self.vte_sweep_include_no_non_voxel {
            &VTE_SWEEP_PROFILES_EXTENDED
        } else {
            &VTE_SWEEP_PROFILES_ENTITIES
        }
    }

    pub(super) fn vte_sweep_mode_label(&self) -> &'static str {
        if self.vte_sweep_include_no_non_voxel {
            "extended"
        } else {
            "non-voxel"
        }
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
    }

    pub(super) fn reset_runtime_profile_window(&mut self) {
        self.profile_window_start = Instant::now();
        self.profile_frame_samples = 0;
        self.profile_client_cpu_ms_sum = 0.0;
        self.profile_client_cpu_ms_max = 0.0;
        self.profile_gpu_ms_sum = 0.0;
        self.profile_gpu_ms_max = 0.0;
        self.profile_gpu_samples = 0;
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

        if announce {
            let scenario = self.perf_suite_scenario(scenario_index);
            let scenario_num = scenario_index + 1;
            let scenario_total = PERF_SUITE_SCENARIOS.len();
            let phase_label = match phase {
                PerfSuitePhase::Warmup => "warmup",
                PerfSuitePhase::Sample => "sample",
            };
            eprintln!(
                "[perf-suite] scenario {}/{} '{}': {} {} frames (warmup={} sample={})",
                scenario_num,
                scenario_total,
                scenario.label,
                phase_label,
                frames_remaining,
                warmup_frames,
                sample_frames
            );
        }
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
                "client_cpu": {
                    "avg_ms": result.client_cpu_avg_ms,
                    "max_ms": result.client_cpu_max_ms,
                    "frames": result.client_cpu_frames,
                },
                "render_gpu": render_gpu,
            }));
        }

        let report = serde_json::json!({
            "schema": "polychora.perf_suite.v1",
            "generated_unix_seconds": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            "elapsed_seconds": elapsed_s,
            "world_file": self.world_file.to_string_lossy(),
            "render_backend": format!("{:?}", self.args.backend),
            "vte_display_mode": format!("{:?}", self.args.vte_display_mode),
            "vte_max_trace_steps": self.vte_max_trace_steps,
            "vte_max_trace_distance": self.vte_max_trace_distance,
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
                    state.results.push(PerfSuiteScenarioResult {
                        scenario_index: state.scenario_index,
                        scenario_label: scenario.label,
                        client_cpu_avg_ms: client_avg_ms,
                        client_cpu_max_ms: client_max_ms,
                        client_cpu_frames: state.sample_frame_samples,
                        render_gpu_avg_ms,
                        render_gpu_max_ms,
                        render_gpu_samples: state.sample_gpu_samples,
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
                    "profile client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu avg={:.3}ms max={:.3}ms samples={}",
                    client_avg_ms,
                    client_max_ms,
                    self.profile_frame_samples,
                    gpu_avg_ms,
                    gpu_max_ms,
                    gpu_samples,
                );
            } else {
                eprintln!(
                    "profile client-cpu avg={:.3}ms max={:.3}ms frames={} | render-gpu unavailable",
                    client_avg_ms, client_max_ms, self.profile_frame_samples,
                );
            }
        }

        self.reset_runtime_profile_window();
    }
}
