use super::*;

pub(super) const PROFILER_MAX_TIMESTAMPS: u32 = 64;
const PROFILER_REPORT_INTERVAL: usize = 100;
const PROFILER_SLOW_FRAME_THRESHOLD_MS: f64 = 100.0;
const PROFILER_SLOW_FRAME_REPORT_INTERVAL: usize = 60;

pub(super) struct GpuProfiler {
    pub(super) timestamp_period_ns: f32,
    pub(super) next_query: u32,
    /// Phase name for each timestamp (interval is from previous to this timestamp)
    pub(super) phase_names: Vec<&'static str>,
    /// Accumulated (total_ms, count) per phase name
    pub(super) accum: Vec<(&'static str, f64, usize)>,
    pub(super) total_frames: usize,
    pub(super) last_slow_report_frame: Option<usize>,
    /// Last frame's GPU total in ms (for HUD display)
    pub(super) last_gpu_total_ms: f32,
    /// Last frame's per-phase breakdown (for HUD display)
    pub(super) last_frame_phases: Vec<(&'static str, f32)>,
    pub(super) vte_frame_samples: usize,
    pub(super) vte_chunk_headers_sum: u64,
    pub(super) vte_visible_chunks_sum: u64,
    pub(super) vte_visible_lod0_sum: u64,
    pub(super) vte_visible_lod1_sum: u64,
    pub(super) vte_visible_lod2_sum: u64,
    pub(super) vte_y_slices_sum: u64,
    pub(super) vte_y_slice_lookup_entries_sum: u64,
    pub(super) vte_dense_cap_last: usize,
    pub(super) vte_leaf_cap_last: usize,
    pub(super) vte_node_cap_last: usize,
    pub(super) vte_leaf_entry_cap_last: usize,
    pub(super) raster_frame_samples: usize,
    pub(super) raster_tetrahedrons_sum: u64,
    pub(super) entity_tetrahedrons_sum: u64,
}

impl GpuProfiler {
    pub(super) fn new(device: Arc<Device>) -> Self {
        let timestamp_period_ns = device.physical_device().properties().timestamp_period;

        GpuProfiler {
            timestamp_period_ns,
            next_query: 0,
            phase_names: Vec::new(),
            accum: Vec::new(),
            total_frames: 0,
            last_slow_report_frame: None,
            last_gpu_total_ms: 0.0,
            last_frame_phases: Vec::new(),
            vte_frame_samples: 0,
            vte_chunk_headers_sum: 0,
            vte_visible_chunks_sum: 0,
            vte_visible_lod0_sum: 0,
            vte_visible_lod1_sum: 0,
            vte_visible_lod2_sum: 0,
            vte_y_slices_sum: 0,
            vte_y_slice_lookup_entries_sum: 0,
            vte_dense_cap_last: 0,
            vte_leaf_cap_last: 0,
            vte_node_cap_last: 0,
            vte_leaf_entry_cap_last: 0,
            raster_frame_samples: 0,
            raster_tetrahedrons_sum: 0,
            entity_tetrahedrons_sum: 0,
        }
    }

    pub(super) fn create_query_pool(device: &Arc<Device>) -> Arc<QueryPool> {
        QueryPool::new(
            device.clone(),
            QueryPoolCreateInfo {
                query_count: PROFILER_MAX_TIMESTAMPS,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap()
    }

    pub(super) fn begin_frame(&mut self) {
        self.next_query = 0;
        self.phase_names.clear();
    }

    pub(super) fn next_query_index(&mut self, name: &'static str) -> u32 {
        let idx = self.next_query;
        self.phase_names.push(name);
        self.next_query += 1;
        idx
    }

    pub(super) fn read_results_and_accumulate(
        &mut self,
        query_pool: &Arc<QueryPool>,
        clipped_tet_count: u32,
    ) {
        let n = self.next_query;
        if n < 2 {
            return;
        }

        let mut results = vec![0u64; n as usize];
        match query_pool.get_results::<u64>(0..n, &mut results, QueryResultFlags::empty()) {
            Ok(true) => {}
            _ => return,
        }

        let period_ms = self.timestamp_period_ns as f64 / 1_000_000.0;

        self.last_frame_phases.clear();
        let mut total_ms = 0.0f64;

        // Compute intervals between consecutive timestamps
        for i in 1..n as usize {
            let interval_ms = (results[i] - results[i - 1]) as f64 * period_ms;
            let name = self.phase_names[i];
            total_ms += interval_ms;
            self.last_frame_phases.push((name, interval_ms as f32));

            // Accumulate into named buckets
            if let Some(entry) = self.accum.iter_mut().find(|(n, _, _)| *n == name) {
                entry.1 += interval_ms;
                entry.2 += 1;
            } else {
                self.accum.push((name, interval_ms, 1));
            }
        }
        self.last_gpu_total_ms = total_ms as f32;

        self.total_frames += 1;

        if total_ms > PROFILER_SLOW_FRAME_THRESHOLD_MS {
            let should_report = match self.last_slow_report_frame {
                None => true,
                Some(last_frame) => {
                    self.total_frames.saturating_sub(last_frame)
                        >= PROFILER_SLOW_FRAME_REPORT_INTERVAL
                }
            };
            if should_report {
                println!(
                    "!!! SLOW FRAME ({:.1} ms) — clipped_tets={} — per-phase:",
                    total_ms, clipped_tet_count
                );
                for (name, ms) in &self.last_frame_phases {
                    println!("  {}: {:.3} ms", name, ms);
                }
                self.last_slow_report_frame = Some(self.total_frames);
            }
        }

        if self.total_frames > 0 && self.total_frames % PROFILER_REPORT_INTERVAL == 0 {
            self.print_report();
        }
    }

    pub(super) fn record_scene_stats(
        &mut self,
        is_vte: bool,
        vte_chunk_headers: usize,
        vte_visible_chunks: usize,
        vte_visible_lod_counts: [u32; 3],
        vte_y_slices: usize,
        vte_y_slice_lookup_entries: usize,
        vte_dense_cap: usize,
        vte_leaf_cap: usize,
        vte_node_cap: usize,
        vte_leaf_entry_cap: usize,
        raster_tetrahedrons: usize,
        entity_tetrahedrons: usize,
    ) {
        if is_vte {
            self.vte_frame_samples = self.vte_frame_samples.saturating_add(1);
            self.vte_chunk_headers_sum = self
                .vte_chunk_headers_sum
                .saturating_add(vte_chunk_headers as u64);
            self.vte_visible_chunks_sum = self
                .vte_visible_chunks_sum
                .saturating_add(vte_visible_chunks as u64);
            self.vte_visible_lod0_sum = self
                .vte_visible_lod0_sum
                .saturating_add(vte_visible_lod_counts[0] as u64);
            self.vte_visible_lod1_sum = self
                .vte_visible_lod1_sum
                .saturating_add(vte_visible_lod_counts[1] as u64);
            self.vte_visible_lod2_sum = self
                .vte_visible_lod2_sum
                .saturating_add(vte_visible_lod_counts[2] as u64);
            self.vte_y_slices_sum = self.vte_y_slices_sum.saturating_add(vte_y_slices as u64);
            self.vte_y_slice_lookup_entries_sum = self
                .vte_y_slice_lookup_entries_sum
                .saturating_add(vte_y_slice_lookup_entries as u64);
            self.vte_dense_cap_last = vte_dense_cap;
            self.vte_leaf_cap_last = vte_leaf_cap;
            self.vte_node_cap_last = vte_node_cap;
            self.vte_leaf_entry_cap_last = vte_leaf_entry_cap;
        }

        self.raster_frame_samples = self.raster_frame_samples.saturating_add(1);
        self.raster_tetrahedrons_sum = self
            .raster_tetrahedrons_sum
            .saturating_add(raster_tetrahedrons as u64);
        self.entity_tetrahedrons_sum = self
            .entity_tetrahedrons_sum
            .saturating_add(entity_tetrahedrons as u64);
    }

    pub(super) fn print_report(&mut self) {
        println!("=== GPU Profile ({} frames) ===", self.total_frames);

        let mut rows: Vec<(&'static str, f64, usize)> = self
            .accum
            .iter()
            .filter_map(|(name, total_ms, count)| {
                if *count == 0 {
                    None
                } else {
                    Some((*name, total_ms / *count as f64, *count))
                }
            })
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));

        let total_avg: f64 = rows.iter().map(|(_, avg, _)| *avg).sum();
        for (name, avg, count) in rows {
            let pct = if total_avg > 0.0 {
                (avg / total_avg) * 100.0
            } else {
                0.0
            };
            println!(
                "  {}: {:.3} ms ({:.1}%) (avg over {} samples)",
                name, avg, pct, count
            );
        }
        println!(
            "  Total avg: {:.3} ms ({:.0} FPS)",
            total_avg,
            1000.0 / total_avg.max(0.001)
        );
        if self.vte_frame_samples > 0 {
            let s = self.vte_frame_samples as f64;
            println!(
                "  VTE avg chunks: headers {:.1} visible {:.1} (L0 {:.1} / L1 {:.1} / L2 {:.1}) y_slices {:.1} y_lookup {:.1}",
                self.vte_chunk_headers_sum as f64 / s,
                self.vte_visible_chunks_sum as f64 / s,
                self.vte_visible_lod0_sum as f64 / s,
                self.vte_visible_lod1_sum as f64 / s,
                self.vte_visible_lod2_sum as f64 / s,
                self.vte_y_slices_sum as f64 / s,
                self.vte_y_slice_lookup_entries_sum as f64 / s,
            );
            println!(
                "  VTE caps: dense {} leaf {} nodes {} leaf_entries {}",
                self.vte_dense_cap_last,
                self.vte_leaf_cap_last,
                self.vte_node_cap_last,
                self.vte_leaf_entry_cap_last
            );
        }
        if self.raster_frame_samples > 0 {
            let s = self.raster_frame_samples as f64;
            println!(
                "  Tetra avg: raster_tets {:.1} entity_tets {:.1}",
                self.raster_tetrahedrons_sum as f64 / s,
                self.entity_tetrahedrons_sum as f64 / s,
            );
        }

        println!("================================");

        // Reset accumulators
        self.accum.clear();
        self.vte_frame_samples = 0;
        self.vte_chunk_headers_sum = 0;
        self.vte_visible_chunks_sum = 0;
        self.vte_visible_lod0_sum = 0;
        self.vte_visible_lod1_sum = 0;
        self.vte_visible_lod2_sum = 0;
        self.vte_y_slices_sum = 0;
        self.vte_y_slice_lookup_entries_sum = 0;
        self.vte_dense_cap_last = 0;
        self.vte_leaf_cap_last = 0;
        self.vte_node_cap_last = 0;
        self.vte_leaf_entry_cap_last = 0;
        self.raster_frame_samples = 0;
        self.raster_tetrahedrons_sum = 0;
        self.entity_tetrahedrons_sum = 0;
    }
}
