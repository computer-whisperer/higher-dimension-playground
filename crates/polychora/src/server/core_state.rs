use super::*;
use crate::shared::chunk_payload::ResolvedChunkPayload;
use crate::materials::block_to_material_appearance;
use crate::shared::region_tree::{chunk_key_from_chunk_pos, RegionChunkTree, RegionTreeCore};
use crate::shared::voxel::{BlockData, CHUNK_VOLUME};

pub(super) struct ServerState {
    pub(super) next_object_id: u64,
    pub(super) entity_store: EntityStore,
    pub(super) entity_records: HashMap<u64, EntityRecord>,
    world: ServerWorldOverlay,
    pub(super) mob_nav_chunk_cache: RegionChunkTree,
    pub(super) players: HashMap<u64, PlayerState>,
    pub(super) mobs: HashMap<u64, MobState>,
    pub(super) mob_nav_debug: bool,
    pub(super) mob_nav_simple_steer: bool,
    pub(super) clients: HashMap<u64, mpsc::Sender<ServerMessage>>,
    pub(super) client_world_interest_bounds: HashMap<u64, Aabb4i>,
    pub(super) client_visible_entities: HashMap<u64, HashSet<u64>>,
    pub(super) cpu_profile: ServerCpuProfile,
}

impl ServerState {
    pub(super) fn new(
        world: ServerWorldOverlay,
        next_object_id: u64,
        mob_nav_debug: bool,
        mob_nav_simple_steer: bool,
        start: Instant,
    ) -> Self {
        Self {
            next_object_id,
            entity_store: EntityStore::new(),
            entity_records: HashMap::new(),
            world,
            mob_nav_chunk_cache: RegionChunkTree::new(),
            players: HashMap::new(),
            mobs: HashMap::new(),
            mob_nav_debug,
            mob_nav_simple_steer,
            clients: HashMap::new(),
            client_world_interest_bounds: HashMap::new(),
            client_visible_entities: HashMap::new(),
            cpu_profile: ServerCpuProfile::new(start),
        }
    }

    pub(super) fn query_world_subtree(&mut self, bounds: Aabb4i) -> Arc<RegionTreeCore> {
        let total_start = Instant::now();
        let prepare_start = Instant::now();
        let loaded_regions = match self.world.prepare_query_bounds(bounds) {
            Ok(count) => count,
            Err(error) => {
                eprintln!(
                    "failed to hydrate persisted world bounds {:?}->{:?}: {}",
                    bounds.min, bounds.max, error
                );
                0
            }
        };
        let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;

        let query_start = Instant::now();
        let subtree = self
            .world
            .query_region_core(QueryVolume { bounds }, QueryDetail::Exact);
        self.absorb_world_subtree_into_mob_nav_cache(bounds, subtree.as_ref());
        let query_ms = query_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        if total_ms >= 50.0 {
            eprintln!(
                "[server-world-subtree-slow] bounds={:?}->{:?} loaded_regions={} prepare_ms={:.3} query_ms={:.3} total_ms={:.3}",
                bounds.min,
                bounds.max,
                loaded_regions,
                prepare_ms,
                query_ms,
                total_ms
            );
        }
        subtree
    }

    pub(super) fn mob_nav_cached_chunk_at(
        &self,
        chunk_pos: ChunkPos,
    ) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.mob_nav_chunk_cache
            .chunk_payload(chunk_key_from_chunk_pos(chunk_pos))
            .and_then(dense_chunk_from_payload)
    }

    pub(super) fn evict_far_mob_nav_cache_chunks(
        &mut self,
        keep_radius_chunks: i32,
        max_subtree_drops: usize,
    ) -> Option<Aabb4i> {
        let keep_bounds = self.mob_nav_cache_keep_bounds(keep_radius_chunks);
        self.mob_nav_chunk_cache
            .lazy_drop_outside_bounds(keep_bounds, max_subtree_drops)
    }

    fn absorb_world_subtree_into_mob_nav_cache(&mut self, bounds: Aabb4i, subtree: &RegionTreeCore) {
        if !bounds.is_valid() {
            return;
        }
        let _ = self
            .mob_nav_chunk_cache
            .splice_core_in_bounds(bounds, subtree);
    }

    fn mob_nav_cache_keep_bounds(&self, keep_radius_chunks: i32) -> Aabb4i {
        let radius = keep_radius_chunks.max(0);
        let mut keep_bounds = None::<Aabb4i>;
        for player in self.players.values() {
            let Some(snapshot) = self.entity_store.snapshot(player.entity_id) else {
                continue;
            };
            let center = world_chunk_from_position(snapshot.position);
            let player_keep = Aabb4i::new(
                [
                    center[0] - radius,
                    center[1] - radius,
                    center[2] - radius,
                    center[3] - radius,
                ],
                [
                    center[0] + radius,
                    center[1] + radius,
                    center[2] + radius,
                    center[3] + radius,
                ],
            );
            keep_bounds = Some(match keep_bounds {
                Some(acc) => union_bounds(acc, player_keep),
                None => player_keep,
            });
        }

        keep_bounds.unwrap_or_else(|| Aabb4i::new([1, 1, 1, 1], [0, 0, 0, 0]))
    }

    pub(super) fn world_bounds(&self) -> crate::shared::protocol::WorldBounds {
        self.world.world_bounds()
    }

    pub(super) fn world_seed(&self) -> u64 {
        self.world.world_seed()
    }

    pub(super) fn world_non_empty_chunk_count(&self) -> usize {
        self.world.non_empty_chunk_count()
    }

    pub(super) fn world_chunk_at(&self, chunk_pos: ChunkPos) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.world.chunk_at(chunk_pos)
    }

    pub(super) fn world_effective_chunk(
        &self,
        chunk_pos: ChunkPos,
        preserve_explicit_empty_chunk: bool,
    ) -> Option<[VoxelType; CHUNK_VOLUME]> {
        self.world
            .effective_chunk(chunk_pos, preserve_explicit_empty_chunk)
    }

    pub(super) fn apply_world_voxel_edit(
        &mut self,
        position: [i32; 4],
        block: BlockData,
    ) -> Option<ChunkPos> {
        self.world.apply_voxel_edit(position, block)
    }

    pub(super) fn world_take_dirty_bounds(&mut self) -> Vec<Aabb4i> {
        self.world.take_dirty_bounds()
    }

    pub(super) fn persist_world_if_dirty(
        &mut self,
        now_ms: u64,
    ) -> io::Result<Option<crate::save_v4::SaveResult>> {
        self.world
            .persist_dirty_overrides(self.next_object_id, now_ms)
    }

    pub(super) fn set_client_world_interest_bounds(
        &mut self,
        client_id: u64,
        bounds: Aabb4i,
    ) -> Option<Aabb4i> {
        self.client_world_interest_bounds.insert(client_id, bounds)
    }
}

fn dense_chunk_from_payload(resolved: ResolvedChunkPayload) -> Option<[VoxelType; CHUNK_VOLUME]> {
    let materials = resolved.payload.dense_materials().ok()?;
    if materials.len() != CHUNK_VOLUME {
        return None;
    }
    let mut chunk = [VoxelType::AIR; CHUNK_VOLUME];
    for (idx, palette_idx) in materials.into_iter().enumerate() {
        let block = resolved
            .block_palette
            .get(palette_idx as usize)
            .cloned()
            .unwrap_or(BlockData::AIR);
        let mat = block_to_material_appearance(block.namespace, block.block_type);
        chunk[idx] = VoxelType(mat);
    }
    Some(chunk)
}

fn union_bounds(a: Aabb4i, b: Aabb4i) -> Aabb4i {
    Aabb4i::new(
        [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
            a.min[3].min(b.min[3]),
        ],
        [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
            a.max[3].max(b.max[3]),
        ],
    )
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

pub(super) fn mark_entity_record_despawned(
    state: &mut ServerState,
    entity_id: u64,
    now_ms: Option<u64>,
) {
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
