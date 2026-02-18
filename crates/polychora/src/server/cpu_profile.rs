use super::SERVER_CPU_PROFILE_INTERVAL;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub(super) struct ServerCpuProfile {
    window_start: Instant,
    message_samples: u64,
    message_cpu_ms_sum: f64,
    message_cpu_ms_max: f64,
    tick_samples: u64,
    tick_cpu_ms_sum: f64,
    tick_cpu_ms_max: f64,
    tick_player_snapshots_sum: u64,
    tick_player_snapshots_max: u64,
    tick_entity_snapshots_sum: u64,
    tick_entity_snapshots_max: u64,
}

#[derive(Copy, Clone, Debug)]
pub(super) struct ServerCpuProfileReport {
    pub(super) message_samples: u64,
    pub(super) message_cpu_ms_sum: f64,
    pub(super) message_cpu_ms_max: f64,
    pub(super) tick_samples: u64,
    pub(super) tick_cpu_ms_sum: f64,
    pub(super) tick_cpu_ms_max: f64,
    pub(super) tick_player_snapshots_sum: u64,
    pub(super) tick_player_snapshots_max: u64,
    pub(super) tick_entity_snapshots_sum: u64,
    pub(super) tick_entity_snapshots_max: u64,
}

impl ServerCpuProfile {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            window_start: now,
            message_samples: 0,
            message_cpu_ms_sum: 0.0,
            message_cpu_ms_max: 0.0,
            tick_samples: 0,
            tick_cpu_ms_sum: 0.0,
            tick_cpu_ms_max: 0.0,
            tick_player_snapshots_sum: 0,
            tick_player_snapshots_max: 0,
            tick_entity_snapshots_sum: 0,
            tick_entity_snapshots_max: 0,
        }
    }

    pub(super) fn record_message_sample(&mut self, elapsed: Duration) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.message_samples = self.message_samples.saturating_add(1);
        self.message_cpu_ms_sum += elapsed_ms;
        self.message_cpu_ms_max = self.message_cpu_ms_max.max(elapsed_ms);
    }

    pub(super) fn record_tick_sample(
        &mut self,
        elapsed: Duration,
        player_snapshots: usize,
        entity_snapshots: usize,
    ) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.tick_samples = self.tick_samples.saturating_add(1);
        self.tick_cpu_ms_sum += elapsed_ms;
        self.tick_cpu_ms_max = self.tick_cpu_ms_max.max(elapsed_ms);
        let player_snapshots = player_snapshots as u64;
        let entity_snapshots = entity_snapshots as u64;
        self.tick_player_snapshots_sum = self
            .tick_player_snapshots_sum
            .saturating_add(player_snapshots);
        self.tick_player_snapshots_max = self.tick_player_snapshots_max.max(player_snapshots);
        self.tick_entity_snapshots_sum = self
            .tick_entity_snapshots_sum
            .saturating_add(entity_snapshots);
        self.tick_entity_snapshots_max = self.tick_entity_snapshots_max.max(entity_snapshots);
    }

    pub(super) fn take_report_if_due(&mut self, now: Instant) -> Option<ServerCpuProfileReport> {
        if now.duration_since(self.window_start) < SERVER_CPU_PROFILE_INTERVAL {
            return None;
        }

        let report = ServerCpuProfileReport {
            message_samples: self.message_samples,
            message_cpu_ms_sum: self.message_cpu_ms_sum,
            message_cpu_ms_max: self.message_cpu_ms_max,
            tick_samples: self.tick_samples,
            tick_cpu_ms_sum: self.tick_cpu_ms_sum,
            tick_cpu_ms_max: self.tick_cpu_ms_max,
            tick_player_snapshots_sum: self.tick_player_snapshots_sum,
            tick_player_snapshots_max: self.tick_player_snapshots_max,
            tick_entity_snapshots_sum: self.tick_entity_snapshots_sum,
            tick_entity_snapshots_max: self.tick_entity_snapshots_max,
        };

        self.window_start = now;
        self.message_samples = 0;
        self.message_cpu_ms_sum = 0.0;
        self.message_cpu_ms_max = 0.0;
        self.tick_samples = 0;
        self.tick_cpu_ms_sum = 0.0;
        self.tick_cpu_ms_max = 0.0;
        self.tick_player_snapshots_sum = 0;
        self.tick_player_snapshots_max = 0;
        self.tick_entity_snapshots_sum = 0;
        self.tick_entity_snapshots_max = 0;

        if report.message_samples == 0 && report.tick_samples == 0 {
            None
        } else {
            Some(report)
        }
    }
}
