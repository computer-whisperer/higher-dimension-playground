use super::*;

pub(super) struct ServerState {
    pub(super) next_object_id: u64,
    pub(super) entity_store: EntityStore,
    pub(super) entity_records: HashMap<u64, EntityRecord>,
    pub(super) world: ServerWorldField,
    pub(super) players: HashMap<u64, PlayerState>,
    pub(super) mobs: HashMap<u64, MobState>,
    pub(super) mob_nav_debug: bool,
    pub(super) mob_nav_simple_steer: bool,
    pub(super) clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
    pub(super) client_visible_entities: HashMap<u64, HashSet<u64>>,
    pub(super) cpu_profile: ServerCpuProfile,
}

pub(super) type SharedState = Arc<Mutex<ServerState>>;

pub(super) fn monotonic_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

pub(super) fn allocate_server_object_id(state: &mut ServerState) -> u64 {
    let id = state.next_object_id;
    state.next_object_id = state.next_object_id.wrapping_add(1).max(1);
    id
}

pub(super) fn allocate_or_reserve_server_object_id(
    state: &mut ServerState,
    persisted_id: Option<EntityId>,
) -> EntityId {
    match persisted_id {
        Some(entity_id) if entity_id > 0 => {
            state.next_object_id = state.next_object_id.max(entity_id.saturating_add(1).max(1));
            entity_id
        }
        _ => allocate_server_object_id(state),
    }
}

pub(super) fn upsert_entity_record(
    state: &mut ServerState,
    entity_id: u64,
    class: EntityClass,
    owner_client_id: Option<u64>,
    display_name: Option<String>,
    persistent: bool,
    now_ms: u64,
) {
    let display_name_for_insert = display_name.clone();
    let record = state
        .entity_records
        .entry(entity_id)
        .or_insert(EntityRecord {
            entity_id,
            class,
            owner_client_id,
            display_name: display_name_for_insert,
            persistent,
            spawned_at_ms: now_ms,
            lifecycle: EntityLifecycle::Live,
            despawned_at_ms: None,
        });
    record.class = class;
    record.owner_client_id = owner_client_id;
    record.persistent = persistent;
    if display_name.is_some() || class != EntityClass::Player {
        record.display_name = display_name;
    }
    if record.lifecycle == EntityLifecycle::Despawned {
        record.spawned_at_ms = now_ms;
        record.lifecycle = EntityLifecycle::Live;
        record.despawned_at_ms = None;
    }
}

pub(super) fn mark_entity_record_despawned(state: &mut ServerState, entity_id: u64, now_ms: Option<u64>) {
    let Some(record) = state.entity_records.get_mut(&entity_id) else {
        return;
    };
    record.lifecycle = EntityLifecycle::Despawned;
    if record.despawned_at_ms.is_none() {
        record.despawned_at_ms = now_ms;
    }
}

pub(super) fn summarize_entity_records(state: &ServerState) -> EntityRecordSummary {
    let mut summary = EntityRecordSummary::default();
    for (&record_id, record) in &state.entity_records {
        debug_assert_eq!(record.entity_id, record_id);
        match record.lifecycle {
            EntityLifecycle::Live => {
                summary.live_total = summary.live_total.saturating_add(1);
                if record.persistent {
                    summary.live_persistent = summary.live_persistent.saturating_add(1);
                }
                if record.owner_client_id.is_some() {
                    summary.live_owned = summary.live_owned.saturating_add(1);
                }
                match record.class {
                    EntityClass::Player => {
                        summary.live_players = summary.live_players.saturating_add(1)
                    }
                    EntityClass::Accent => {
                        summary.live_accents = summary.live_accents.saturating_add(1)
                    }
                    EntityClass::Mob => summary.live_mobs = summary.live_mobs.saturating_add(1),
                }
            }
            EntityLifecycle::Despawned => {
                summary.tombstones = summary.tombstones.saturating_add(1);
            }
        }
    }
    summary
}

pub(super) fn record_server_cpu_sample(
    state: &SharedState,
    message_elapsed: Option<Duration>,
    tick_sample: Option<(Duration, usize, usize)>,
) {
    let maybe_report = {
        let mut guard = state.lock().expect("server state lock poisoned");
        if let Some(elapsed) = message_elapsed {
            guard.cpu_profile.record_message_sample(elapsed);
        }
        if let Some((elapsed, player_snapshots, entity_snapshots)) = tick_sample {
            guard
                .cpu_profile
                .record_tick_sample(elapsed, player_snapshots, entity_snapshots);
        }
        guard
            .cpu_profile
            .take_report_if_due(Instant::now())
            .map(|report| {
                (
                    report,
                    guard.players.len(),
                    guard.entity_store.len(),
                    summarize_entity_records(&guard),
                )
            })
    };

    let Some((report, player_count, entity_count, entity_records)) = maybe_report else {
        return;
    };

    let msg_avg_ms = if report.message_samples > 0 {
        report.message_cpu_ms_sum / report.message_samples as f64
    } else {
        0.0
    };
    let tick_avg_ms = if report.tick_samples > 0 {
        report.tick_cpu_ms_sum / report.tick_samples as f64
    } else {
        0.0
    };
    let tick_players_avg = if report.tick_samples > 0 {
        report.tick_player_snapshots_sum as f64 / report.tick_samples as f64
    } else {
        0.0
    };
    let tick_entities_avg = if report.tick_samples > 0 {
        report.tick_entity_snapshots_sum as f64 / report.tick_samples as f64
    } else {
        0.0
    };

    eprintln!(
        "profile server-cpu msg_avg={:.3}ms msg_max={:.3}ms msg_samples={} tick_avg={:.3}ms tick_max={:.3}ms tick_samples={} tick_players_avg={:.1} tick_players_max={} tick_entities_avg={:.1} tick_entities_max={} players={} entities={} rec_live={} rec_players={} rec_accents={} rec_mobs={} rec_persistent={} rec_owned={} rec_tombstones={}",
        msg_avg_ms,
        report.message_cpu_ms_max,
        report.message_samples,
        tick_avg_ms,
        report.tick_cpu_ms_max,
        report.tick_samples,
        tick_players_avg,
        report.tick_player_snapshots_max,
        tick_entities_avg,
        report.tick_entity_snapshots_max,
        player_count,
        entity_count,
        entity_records.live_total,
        entity_records.live_players,
        entity_records.live_accents,
        entity_records.live_mobs,
        entity_records.live_persistent,
        entity_records.live_owned,
        entity_records.tombstones,
    );
}
